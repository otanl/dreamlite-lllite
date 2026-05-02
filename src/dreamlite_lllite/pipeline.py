"""DreamLite-mobile pipeline + LLLite controller helpers.

We do not subclass `DreamLiteMobilePipeline` — DreamLite's pipeline is a
thin wrapper around `from_pretrained` and we only need to bolt LLLite onto
the loaded UNet. Keeping the wrapper minimal also avoids tying us to a
specific DreamLite revision.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .inject import apply_lllite, remove_lllite
from .lllite import LLLiteController


class DreamLiteMobileLLLitePipeline:
    """Wrap a DreamLiteMobilePipeline + LLLite controller behind one object.

    Usage:
        pipe = DreamLiteMobileLLLitePipeline.from_pretrained(
            "models/DreamLite-mobile",
            cond_emb_dim=32, mlp_dim=64,
        )
        pipe.load_lllite_weights("path/to/canny.safetensors")
        pipe.set_cond_image(cond_image_tensor)  # (B,3,H,W) in [-1,1]
        image = pipe(prompt="...", num_inference_steps=4).images[0]
    """

    def __init__(self, base_pipeline, controller: LLLiteController):
        self.base = base_pipeline
        self.controller = controller

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
    ) -> "DreamLiteMobileLLLitePipeline":
        # Late import so this module is importable without DreamLite present.
        from dreamlite import DreamLiteMobilePipeline

        base = DreamLiteMobilePipeline.from_pretrained(
            model_path, torch_dtype=torch_dtype
        ).to(device)
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
        return cls(base, controller)

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
            # safetensors metadata must be str -> str
            md = {k: str(v) for k, v in (metadata or {}).items()}
            save_file(sd, path, md)
        else:
            torch.save(sd, path)

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------
    def set_cond_image(self, cond_image: Optional[torch.Tensor]) -> None:
        """cond_image: (B, 3, H, W) float in [-1, 1] on any device."""
        if cond_image is None:
            self.controller.set_cond_image(None)
            return
        # Move to UNet device/dtype to keep encoder math consistent.
        param = next(self.base.unet.parameters())
        cond_image = cond_image.to(device=param.device, dtype=param.dtype)
        self.controller.set_cond_image(cond_image)

    def set_multiplier(self, value: float) -> None:
        self.controller.set_multiplier(value)

    def detach_lllite(self) -> None:
        """Restore original Linear forwards (turn LLLite off completely)."""
        remove_lllite(self.controller)

    # ------------------------------------------------------------------
    # Generation passthrough
    # ------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        return self.base(*args, **kwargs)
