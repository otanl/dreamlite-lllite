"""DreamLite pipeline + LLLite controller wrapper.

Auto-detects the underlying DreamLite variant (mobile vs base) by reading
`<model_path>/model_index.json`'s ``_class_name``, so the same wrapper works
for both:

  * ``DreamLiteMobilePipeline`` — 4-step distilled, no CFG
  * ``DreamLitePipeline``       — 28-step, CFG + IMG_CFG

The two share the same UNet architecture, VAE, and Qwen3-VL text encoder
(in fact the VAE / TE weights are byte-identical), so an LLLite adapter
only differs in the UNet weights it has seen during training. We therefore
route both variants through one wrapper.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch

from .inject import apply_lllite, remove_lllite
from .lllite import LLLiteController


def _detect_pipeline_class(model_path: str):
    """Return the DreamLite pipeline class to use for the given model dir.

    Falls back to ``DreamLiteMobilePipeline`` if model_index.json is missing
    or unreadable, since mobile is the more common case for LLLite.
    """
    idx = os.path.join(model_path, "model_index.json")
    name = "DreamLiteMobilePipeline"
    try:
        with open(idx, "r", encoding="utf-8") as f:
            name = json.load(f).get("_class_name", name)
    except FileNotFoundError:
        pass
    if name == "DreamLitePipeline":
        from dreamlite import DreamLitePipeline as Cls
    elif name == "DreamLiteMobilePipeline":
        from dreamlite import DreamLiteMobilePipeline as Cls
    else:
        # Try generic import; will raise an informative error if missing.
        import importlib
        mod = importlib.import_module("dreamlite")
        Cls = getattr(mod, name)
    return Cls, name


class DreamLiteLLLitePipeline:
    """Wrap a DreamLite{,Mobile}Pipeline + LLLite controller behind one object.

    Usage:
        pipe = DreamLiteLLLitePipeline.from_pretrained(
            "models/DreamLite-mobile",   # or "models/DreamLite-base"
            cond_emb_dim=32, mlp_dim=64,
        )
        pipe.load_lllite_weights("path/to/canny.safetensors")
        pipe.set_cond_image(cond_image_tensor)  # (B,3,H,W) in [-1,1]
        image = pipe(prompt="...", num_inference_steps=4).images[0]
    """

    def __init__(self, base_pipeline, controller: LLLiteController, variant: str):
        self.base = base_pipeline
        self.controller = controller
        self.variant = variant  # "DreamLiteMobilePipeline" or "DreamLitePipeline"

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        cond_emb_dim: int = 32,
        mlp_dim: int = 64,
        cond_image_size: int = 1024,
        multiplier: float = 1.0,
    ) -> "DreamLiteLLLitePipeline":
        Cls, variant = _detect_pipeline_class(model_path)
        base = Cls.from_pretrained(model_path, torch_dtype=torch_dtype).to(device)
        controller = apply_lllite(
            base.unet,
            cond_emb_dim=cond_emb_dim,
            mlp_dim=mlp_dim,
            cond_image_size=cond_image_size,
            multiplier=multiplier,
        )
        # Move adapter modules to UNet's device/dtype (parameters of the
        # adapter were created on CPU/fp32 by default).
        controller.to(device=device, dtype=torch_dtype)
        return cls(base, controller, variant)

    # ------------------------------------------------------------------
    # LLLite weight management
    # ------------------------------------------------------------------
    def load_lllite_weights(self, path: str, strict: bool = True) -> None:
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(path)
        else:
            sd = torch.load(path, map_location="cpu")
        info = self.controller.load_state_dict(sd, strict=strict)
        if not strict:
            print(f"loaded LLLite weights with {len(info.missing_keys)} missing, "
                  f"{len(info.unexpected_keys)} unexpected keys")

    def save_lllite_weights(
        self,
        path: str,
        dtype: Optional[torch.dtype] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        sd = self.controller.state_dict()
        if dtype is not None:
            sd = {k: v.to(dtype).cpu() for k, v in sd.items()}
        if path.endswith(".safetensors"):
            from safetensors.torch import save_file
            md = {k: str(v) for k, v in (metadata or {}).items()}
            save_file(sd, path, md)
        else:
            torch.save(sd, path)

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------
    def set_cond_image(self, cond_image: Optional[torch.Tensor]) -> None:
        if cond_image is None:
            self.controller.set_cond_image(None)
            return
        param = next(self.base.unet.parameters())
        cond_image = cond_image.to(device=param.device, dtype=param.dtype)
        self.controller.set_cond_image(cond_image)

    def set_multiplier(self, value: float) -> None:
        self.controller.set_multiplier(value)

    def detach_lllite(self) -> None:
        remove_lllite(self.controller)

    # ------------------------------------------------------------------
    # Generation passthrough
    # ------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        return self.base(*args, **kwargs)


# Backwards-compatible alias for v0.1 callers.
DreamLiteMobileLLLitePipeline = DreamLiteLLLitePipeline
