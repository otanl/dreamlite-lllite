"""ControlNet-LLLite adapter modules for DreamLite-mobile.

Architecture is compatible with kohya-ss LLLite v1
(https://github.com/kohya-ss/sd-scripts/blob/main/networks/control_net_lllite.py),
re-targeted to diffusers-style UNet block names (`down_blocks`/`mid_block`/
`up_blocks`) used by DreamLite.

A single LLLiteModule is attached to a Linear layer (`to_q`/`to_k`/`to_v` of
either self- or cross-attention). It contains:

  * `conditioning1`: a small CNN that maps the conditioning image
    (B, 3, H_img, W_img) to (B, cond_emb_dim, H_block, W_block) where
    (H_block, W_block) matches the feature-map size at the block where this
    Linear lives. Called once per `set_cond_image()`, NOT per timestep.
  * `down`, `mid`, `up`: per-timestep LoRA-like path. `down` reduces hidden
    dim, `mid` mixes with the conditioning embedding (channel concat), `up`
    is zero-initialised and projects back to the original dim.

The forward replaces the host Linear's forward so the adapter delta is added
to its INPUT (equivalent to adding a Linear-of-cx to its output).
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn

ORIGINAL_LINEAR = nn.Linear


def _build_conditioning_encoder(
    cond_emb_dim: int,
    in_hw: int,
    out_hw: int,
) -> nn.Sequential:
    """Build a small CNN that downsamples (3, in_hw, in_hw) -> (cond_emb_dim, out_hw, out_hw).

    Mirrors kohya-ss LLLite's `conditioning1` shape (3 -> cond_emb_dim/2 -> cond_emb_dim)
    but picks strides/kernels dynamically so it works for any block resolution.
    The total downsample factor must be an integer.
    """
    if in_hw % out_hw != 0:
        raise ValueError(
            f"cond image size {in_hw} must be an integer multiple of block size {out_hw}"
        )
    total_stride = in_hw // out_hw
    if total_stride < 1:
        raise ValueError(f"in_hw {in_hw} < out_hw {out_hw}")

    # Factor total_stride into a sequence of strides in {2, 4, 8} so each
    # conv stays well-conditioned. Fall back to repeated 2x.
    strides: List[int] = []
    rem = total_stride
    while rem > 1:
        if rem % 4 == 0 and rem >= 4:
            strides.append(4)
            rem //= 4
        else:
            strides.append(2)
            rem //= 2
    if not strides:
        strides = [1]

    layers: List[nn.Module] = []
    in_ch = 3
    mid_ch = cond_emb_dim // 2
    for i, s in enumerate(strides):
        is_last = i == len(strides) - 1
        out_ch = cond_emb_dim if is_last else mid_ch
        # kernel = stride to make a clean strided conv (no overlap, no padding)
        k = s if s > 1 else 3
        p = 0 if s > 1 else 1
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p))
        if not is_last:
            layers.append(nn.ReLU(inplace=True))
        in_ch = out_ch
    return nn.Sequential(*layers)


class LLLiteModule(nn.Module):
    """Adapter for one Linear inside a Transformer attention.

    The host Linear's forward is monkey-patched in `apply_to()`. The original
    forward is preserved on `self.org_forward` so the adapter can call it.
    """

    def __init__(
        self,
        name: str,
        host_linear: nn.Linear,
        cond_emb_dim: int,
        mlp_dim: int,
        cond_image_size: int,
        block_feature_size: int,
        width_concat_factor: int = 2,
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(host_linear, ORIGINAL_LINEAR):
            raise TypeError(f"LLLiteModule expects nn.Linear, got {type(host_linear)}")

        self.lllite_name = name
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.cond_image_size = cond_image_size
        self.block_feature_size = block_feature_size
        self.width_concat_factor = width_concat_factor
        self.dropout = dropout
        self.multiplier = multiplier
        # Hold the host as a non-submodule so it doesn't appear in our
        # state_dict; we only manage the adapter parameters here.
        self._host_ref: List[nn.Linear] = [host_linear]

        in_dim = host_linear.in_features

        self.conditioning1 = _build_conditioning_encoder(
            cond_emb_dim, cond_image_size, block_feature_size
        )

        self.down = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.ReLU(inplace=True),
        )
        self.mid = nn.Sequential(
            nn.Linear(mlp_dim + cond_emb_dim, mlp_dim),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(
            nn.Linear(mlp_dim, in_dim),
        )
        # Zero init the up projection (zero-conv) so an untrained adapter
        # produces zero delta and the model behaves identically to vanilla.
        nn.init.zeros_(self.up[0].weight)
        nn.init.zeros_(self.up[0].bias)

        self.cond_emb: Optional[torch.Tensor] = None  # cached embedding
        self._org_forward = None

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------
    def set_cond_image(self, cond_image: Optional[torch.Tensor]) -> None:
        """Precompute and cache the conditioning embedding once per generation.

        cond_image: float tensor of shape (B, 3, H, W) in [-1, 1].
        Pass None to disable.
        """
        if cond_image is None:
            self.cond_emb = None
            return
        # Place encoder on the same device/dtype as cond_image
        device, dtype = cond_image.device, cond_image.dtype
        self.conditioning1.to(device=device, dtype=dtype)
        cx = self.conditioning1(cond_image)  # (B, cond_emb_dim, h, w)
        # Account for DreamLite's "spatial concat" inputs: the UNet feature
        # tensor at this block is width-doubled (`width_concat_factor=2`),
        # because the pipeline does `cat([latent, cond_image_latent], dim=W)`.
        # Pad cond_emb along W with zeros so the conditioning signal aligns
        # with the LATENT half of each row and the reference-image half
        # gets no LLLite contribution.
        if self.width_concat_factor > 1:
            n, c, h, w = cx.shape
            pad = torch.zeros(
                n, c, h, w * (self.width_concat_factor - 1),
                device=cx.device, dtype=cx.dtype,
            )
            cx = torch.cat([cx, pad], dim=3)  # (B, c, h, w*factor)
        n, c, h, w = cx.shape
        cx = cx.view(n, c, h * w).permute(0, 2, 1).contiguous()  # (B, h*w, c)
        self.cond_emb = cx

    # ------------------------------------------------------------------
    # Hooking
    # ------------------------------------------------------------------
    def apply_to(self) -> None:
        host = self._host_ref[0]
        if hasattr(host, "_lllite_attached"):
            raise RuntimeError(
                f"Linear {self.lllite_name} already has an LLLite adapter attached"
            )
        self._org_forward = host.forward
        host.forward = self.forward  # type: ignore[assignment]
        host._lllite_attached = self  # type: ignore[attr-defined]

    def remove_from(self) -> None:
        host = self._host_ref[0]
        if self._org_forward is not None:
            host.forward = self._org_forward  # type: ignore[assignment]
        if hasattr(host, "_lllite_attached"):
            delattr(host, "_lllite_attached")
        self._org_forward = None

    # ------------------------------------------------------------------
    # Forward (replaces host Linear's forward)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._org_forward is None:
            raise RuntimeError("LLLiteModule.forward called before apply_to()")
        if self.multiplier == 0.0 or self.cond_emb is None:
            return self._org_forward(x)

        cx = self.cond_emb
        # Match dtype/device of x (e.g. when running pipeline in bf16 but
        # adapter trained in fp32).
        if cx.dtype != x.dtype or cx.device != x.device:
            cx = cx.to(device=x.device, dtype=x.dtype)

        # Handle CFG-style 2x batch (uncond + cond stacked). LLLite was set
        # with cond_image of size B; if x is 2B (uncond, cond), repeat cond.
        if x.shape[0] != cx.shape[0] and x.shape[0] == 2 * cx.shape[0]:
            cx = cx.repeat(2, 1, 1)

        # Some attention pre-norms may flatten differently; we expect
        # x: (B, N, in_dim). Sanity check.
        if x.dim() != 3:
            # Unsupported input rank; bypass adapter to avoid silent garbage.
            return self._org_forward(x)
        if cx.shape[1] != x.shape[1]:
            # Sequence length mismatch (e.g. cross-attn keys from text). The
            # injection logic should have skipped these, but bail safely.
            return self._org_forward(x)

        # down: (B, N, in_dim) -> (B, N, mlp_dim). cast adapter to x dtype.
        d = self.down(x)
        merged = torch.cat([cx, d], dim=2)  # (B, N, mlp_dim + cond_emb_dim)
        merged = self.mid(merged)
        if self.dropout is not None and self.training:
            merged = torch.nn.functional.dropout(merged, p=self.dropout)
        delta = self.up(merged) * self.multiplier  # (B, N, in_dim)

        return self._org_forward(x + delta)


class LLLiteController(nn.Module):
    """Holds all LLLiteModule instances for a UNet, plus convenience methods.

    The controller is also the unit of save/load — its state_dict contains
    every adapter's parameters keyed by `lllite_name`.
    """

    def __init__(
        self,
        modules_dict: Optional[dict] = None,
        cond_emb_dim: int = 32,
        mlp_dim: int = 64,
        cond_image_size: int = 1024,
        width_concat_factor: int = 2,
        multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.cond_image_size = cond_image_size
        self.width_concat_factor = width_concat_factor
        self.multiplier = multiplier
        # Use ModuleDict so DDP / state_dict work cleanly. Names use '/' to
        # avoid clashing with nn.Module's '.' separator.
        self.modules_dict = nn.ModuleDict(modules_dict or {})

    # ------------------------------------------------------------------
    def add_module_for(self, key: str, module: LLLiteModule) -> None:
        if key in self.modules_dict:
            raise KeyError(f"duplicate LLLite module key: {key}")
        self.modules_dict[key] = module

    def __len__(self) -> int:
        return len(self.modules_dict)

    def names(self) -> List[str]:
        return list(self.modules_dict.keys())

    # ------------------------------------------------------------------
    def set_cond_image(self, cond_image: Optional[torch.Tensor]) -> None:
        for m in self.modules_dict.values():
            m.set_cond_image(cond_image)

    def set_multiplier(self, value: float) -> None:
        self.multiplier = value
        for m in self.modules_dict.values():
            m.multiplier = value

    def apply_to(self) -> None:
        for m in self.modules_dict.values():
            m.apply_to()

    def remove_from(self) -> None:
        for m in self.modules_dict.values():
            m.remove_from()

    # ------------------------------------------------------------------
    def trainable_parameters(self):
        return self.parameters()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
