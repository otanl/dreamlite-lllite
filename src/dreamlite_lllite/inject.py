"""Inject LLLite adapters into a DreamLite UNet.

Strategy mirrors kohya-ss LLLite v1:

  * Walk every Transformer2DModel container in the UNet.
  * For each Linear inside, attach an LLLiteModule **only** to:
      - attn1.to_q, attn1.to_k, attn1.to_v   (self-attention)
      - attn2.to_q                           (cross-attention query)
    Cross-attention K/V come from the text encoder so the sequence-length
    differs from the spatial features and an LLLite-style channel concat
    cannot align — these are skipped.
  * Block resolution is auto-detected via a hooked forward pass on a
    randomly initialised input (no model weights touched, no gradients).

The per-block feature spatial size determines how aggressively the
conditioning encoder downsamples the cond image so the embedding aligns with
the block's token grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .lllite import LLLiteController, LLLiteModule


# ----------------------------------------------------------------------
# Target enumeration
# ----------------------------------------------------------------------
@dataclass
class LLLiteTarget:
    """Description of one Linear we will attach an adapter to."""

    name: str  # e.g. "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q"
    container: str  # name of the Transformer2DModel that owns it
    block_root: str  # "down_blocks.0", "mid_block", "up_blocks.1", ...
    attn_kind: str  # "attn1" or "attn2"
    proj_kind: str  # "to_q" / "to_k" / "to_v"
    in_features: int
    out_features: int
    feature_hw: Optional[int] = None  # set after probing


def _is_target_linear(name: str) -> bool:
    """attn1.to_{q,k,v} OR attn2.to_q only, per kohya v1 LLLite."""
    if name.endswith(".attn1.to_q") or name.endswith(".attn1.to_k") or name.endswith(".attn1.to_v"):
        return True
    if name.endswith(".attn2.to_q"):
        return True
    return False


def _block_root_of(name: str) -> str:
    parts = name.split(".")
    if parts[0] in ("down_blocks", "up_blocks"):
        return f"{parts[0]}.{parts[1]}"
    if parts[0] == "mid_block":
        return "mid_block"
    return parts[0]


def list_lllite_targets(unet: nn.Module) -> List[LLLiteTarget]:
    """Find every (attn1.qkv | attn2.q) Linear inside Transformer2DModel hosts."""
    targets: List[LLLiteTarget] = []
    transformer_names: List[str] = [
        n for n, m in unet.named_modules() if type(m).__name__ == "Transformer2DModel"
    ]
    transformer_set = set(transformer_names)

    for name, module in unet.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not _is_target_linear(name):
            continue
        # Only those inside a Transformer2DModel
        container = next((t for t in transformer_set if name.startswith(t + ".")), None)
        if container is None:
            continue
        attn_kind = "attn1" if ".attn1." in name else "attn2"
        proj_kind = name.rsplit(".", 1)[-1]
        targets.append(
            LLLiteTarget(
                name=name,
                container=container,
                block_root=_block_root_of(name),
                attn_kind=attn_kind,
                proj_kind=proj_kind,
                in_features=module.in_features,
                out_features=module.out_features,
            )
        )
    return targets


# ----------------------------------------------------------------------
# Spatial-size probing
# ----------------------------------------------------------------------
def _static_feature_sizes(unet: nn.Module, sample_size: int) -> Dict[str, int]:
    """Compute the spatial size of each Transformer2DModel from the UNet config.

    DreamLite's UNet follows the standard diffusers pattern:
      * down_blocks[0..n-1] each downsample-by-2 *except* the last one
      * mid_block is at the same resolution as down_blocks[-1]'s output
      * up_blocks mirror down_blocks: up_blocks[i] runs at the resolution of
        down_blocks[n-1-i]'s output and (except the last) upsamples by 2
    """
    cfg = unet.config
    n_down = len(cfg.down_block_types)
    n_up = len(cfg.up_block_types)

    sizes: Dict[str, int] = {}
    cur = sample_size
    # down_blocks
    down_resolutions: List[int] = []
    for i in range(n_down):
        down_resolutions.append(cur)
        # all but the last down-block downsamples by 2
        if i < n_down - 1:
            cur //= 2
    # mid block at smallest
    mid_res = cur
    # up_blocks: reverse order
    up_resolutions: List[int] = []
    cur = mid_res
    for i in range(n_up):
        up_resolutions.append(cur)
        if i < n_up - 1:
            cur *= 2

    # Map every Transformer2DModel inside (down|mid|up) to its size
    for name, module in unet.named_modules():
        if type(module).__name__ != "Transformer2DModel":
            continue
        head = name.split(".")[0]
        if head == "down_blocks":
            idx = int(name.split(".")[1])
            sizes[name] = down_resolutions[idx]
        elif head == "up_blocks":
            idx = int(name.split(".")[1])
            sizes[name] = up_resolutions[idx]
        elif head == "mid_block":
            sizes[name] = mid_res
        else:
            raise RuntimeError(f"unexpected transformer host: {name}")
    return sizes


def _probe_feature_sizes(
    unet: nn.Module,
    sample_size: int,
    in_channels: int,
    cross_attention_dim: int,
    encoder_hidden_dim: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, int]:
    """Run one no-grad forward and record (H, W) at each Transformer2DModel input.

    Returns: { transformer_container_name: feature_hw } where feature_hw is
    int(sqrt(num_tokens)) — DreamLite is square-only so this is exact.
    """
    sizes: Dict[str, int] = {}
    handles = []

    transformer_names = [
        n for n, m in unet.named_modules() if type(m).__name__ == "Transformer2DModel"
    ]
    expected = len(transformer_names)

    class _ProbeDone(Exception):
        """Raised after all transformer sizes have been recorded."""

    def make_hook(container_name: str):
        def hook(_mod, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            if hs is not None and hs.dim() == 4:
                sizes[container_name] = int(hs.shape[-1])
            if len(sizes) >= expected:
                raise _ProbeDone()
            return None
        return hook

    for name in transformer_names:
        # Resolve the module by name
        mod = unet.get_submodule(name)
        handles.append(mod.register_forward_pre_hook(make_hook(name), with_kwargs=True))

    try:
        was_training = unet.training
        unet.eval()
        with torch.no_grad():
            B = 1
            sample = torch.randn(B, in_channels, sample_size, sample_size, device=device, dtype=dtype)
            t = torch.zeros(B, device=device, dtype=dtype)
            # DreamLite uses Qwen3-VL features fed into encoder_hid_proj
            # (Linear(qwen_hidden -> cross_attention_dim)). encoder_hidden_dim
            # must match the input side of that projection. If not provided,
            # try to read it from encoder_hid_proj.
            enc_dim = encoder_hidden_dim
            if enc_dim is None:
                proj = getattr(unet, "encoder_hid_proj", None)
                # encoder_hid_proj may be a Sequential; find its first Linear
                if proj is not None:
                    for sub in (proj.modules() if hasattr(proj, "modules") else [proj]):
                        if isinstance(sub, nn.Linear):
                            enc_dim = sub.in_features
                            break
            if enc_dim is None:
                enc_dim = cross_attention_dim  # last-ditch fallback
            enc_hidden = torch.zeros(B, 1, enc_dim, device=device, dtype=dtype)
            added = {
                "text_embeds": torch.zeros(B, 1280, device=device, dtype=dtype),
                "time_ids": torch.zeros(B, 6, device=device, dtype=dtype),
            }
            try:
                unet(
                    sample,
                    t,
                    encoder_hidden_states=enc_hidden,
                    added_cond_kwargs=added,
                    return_dict=False,
                )
            except _ProbeDone:
                pass
            except Exception as e:
                # Forward failed midway (often because we cannot synthesise
                # a valid encoder_hidden_states for cross-attention). The
                # hook may still have collected enough sizes — only re-raise
                # if it didn't.
                if len(sizes) < expected:
                    raise
        if was_training:
            unet.train()
    finally:
        for h in handles:
            h.remove()
    return sizes


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def apply_lllite(
    unet: nn.Module,
    cond_emb_dim: int = 32,
    mlp_dim: int = 64,
    cond_image_size: int = 1024,
    multiplier: float = 1.0,
    width_concat_factor: int = 2,
    sample_size: int = 128,  # latent spatial size (1024 / 8 for TAESDXL)
    in_channels: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    encoder_hidden_dim: Optional[int] = None,
    feature_sizes: Optional[Dict[str, int]] = None,
) -> LLLiteController:
    """Attach LLLite adapters to every eligible Linear in `unet`.

    Returns a LLLiteController whose state_dict represents all adapter
    parameters. Call `controller.set_cond_image(...)` before each generation
    and pass the resulting controller to your training loop / inference.

    `feature_sizes` lets you skip the probing forward by passing
    pre-recorded transformer-block sizes (see `_probe_feature_sizes`).
    """
    if in_channels is None:
        in_channels = getattr(unet.config, "in_channels", 4)
    if cross_attention_dim is None:
        cross_attention_dim = getattr(unet.config, "cross_attention_dim", 2304)

    targets = list_lllite_targets(unet)
    if not targets:
        raise RuntimeError("No LLLite-eligible Linear layers found in this UNet")

    # Block resolutions: prefer the static derivation from UNet config; fall
    # back to a forward probe only if the config is incomplete.
    if feature_sizes is None:
        try:
            feature_sizes = _static_feature_sizes(unet, sample_size=sample_size)
        except Exception:
            sample_param = next(unet.parameters())
            feature_sizes = _probe_feature_sizes(
                unet,
                sample_size=sample_size,
                in_channels=in_channels,
                cross_attention_dim=cross_attention_dim,
                encoder_hidden_dim=encoder_hidden_dim,
                device=sample_param.device,
                dtype=sample_param.dtype,
            )

    controller = LLLiteController(
        cond_emb_dim=cond_emb_dim,
        mlp_dim=mlp_dim,
        cond_image_size=cond_image_size,
        width_concat_factor=width_concat_factor,
        multiplier=multiplier,
    )

    # Attach adapters
    name_to_module = dict(unet.named_modules())
    for t in targets:
        feat_hw = feature_sizes.get(t.container)
        if feat_hw is None:
            # Fallback: derive from container's residual conv if present
            raise RuntimeError(
                f"Could not determine feature HW for {t.container}; "
                f"either pass feature_sizes= or call probe with the right sample_size"
            )
        # Conditioning encoder needs cond_image_size to be a multiple of feat_hw.
        # If the user passes a non-aligned size we round to the nearest valid.
        if cond_image_size % feat_hw != 0:
            raise ValueError(
                f"cond_image_size ({cond_image_size}) must be a multiple of "
                f"block feature size ({feat_hw}) for {t.container}"
            )
        t.feature_hw = feat_hw

        host = name_to_module[t.name]
        # Use safe key: replace dots with double-underscore
        key = t.name.replace(".", "__")
        adapter = LLLiteModule(
            name=key,
            host_linear=host,  # type: ignore[arg-type]
            cond_emb_dim=cond_emb_dim,
            mlp_dim=mlp_dim,
            cond_image_size=cond_image_size,
            block_feature_size=feat_hw,
            width_concat_factor=width_concat_factor,
            multiplier=multiplier,
        )
        controller.add_module_for(key, adapter)

    controller.apply_to()
    return controller


def remove_lllite(controller: LLLiteController) -> None:
    controller.remove_from()
