"""Preprocessors that convert RGB images to LLLite conditioning images.

All preprocessors return a torch.Tensor of shape (3, H, W) in [-1, 1] suitable
for `LLLiteModule.set_cond_image()`. CPU-side processing; the caller moves
to GPU.

Three conditioning types are supported:

  * ``canny`` — edge map. Cheap, no extra model.
  * ``depth`` — relative depth via Depth Anything V2 (lazy-loaded).
  * ``pose`` — OpenPose keypoint visualisation via controlnet-aux (lazy).

If a heavy dependency is not installed, the relevant preprocessor raises
ImportError with a hint.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple, Union

import numpy as np
import torch
from PIL import Image


def _to_pil_rgb(image: Union[Image.Image, np.ndarray, str]) -> Image.Image:
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


def _resize_square(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.LANCZOS)


def _np_to_tensor_neg1_1(arr: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 [0,255] (or HW for grey) to CHW float [-1, 1]."""
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    t = torch.from_numpy(arr).float() / 255.0
    t = t * 2.0 - 1.0
    return t.permute(2, 0, 1).contiguous()


# ----------------------------------------------------------------------
# Canny
# ----------------------------------------------------------------------
def preprocess_canny(
    image: Union[Image.Image, np.ndarray, str],
    size: int = 1024,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> torch.Tensor:
    try:
        import cv2  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "canny preprocessor needs opencv: pip install opencv-python-headless"
        ) from e
    import cv2

    pil = _resize_square(_to_pil_rgb(image), size)
    arr = np.array(pil)
    grey = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(grey, low_threshold, high_threshold)
    return _np_to_tensor_neg1_1(edges)


# ----------------------------------------------------------------------
# Depth (Depth Anything V2 via transformers)
# ----------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_depth_pipeline(device: str):
    try:
        from transformers import pipeline
    except ImportError as e:
        raise ImportError(
            "depth preprocessor needs transformers (already a dep)."
        ) from e
    return pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )


def preprocess_depth(
    image: Union[Image.Image, np.ndarray, str],
    size: int = 1024,
    device: str = "cuda",
) -> torch.Tensor:
    pil = _resize_square(_to_pil_rgb(image), size)
    pipe = _get_depth_pipeline(device)
    out = pipe(pil)
    depth: Image.Image = out["depth"]  # PIL grayscale
    arr = np.array(depth)
    # Normalise to full 0..255 range so the cond signal is well-conditioned
    if arr.max() > arr.min():
        arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255.0).astype(np.uint8)
    return _np_to_tensor_neg1_1(arr)


# ----------------------------------------------------------------------
# Pose (OpenPose via controlnet-aux)
# ----------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_openpose_detector():
    try:
        from controlnet_aux import OpenposeDetector
    except ImportError as e:
        raise ImportError(
            "pose preprocessor needs controlnet-aux: pip install controlnet-aux"
        ) from e
    return OpenposeDetector.from_pretrained("lllyasviel/Annotators")


def preprocess_pose(
    image: Union[Image.Image, np.ndarray, str],
    size: int = 1024,
    include_hand: bool = True,
    include_face: bool = True,
) -> torch.Tensor:
    det = _get_openpose_detector()
    pil = _resize_square(_to_pil_rgb(image), size)
    out = det(pil, hand_and_face=include_hand and include_face)
    if isinstance(out, Image.Image):
        out = out.resize((size, size), Image.Resampling.NEAREST)
    arr = np.array(out)
    return _np_to_tensor_neg1_1(arr)


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------
PREPROCESSORS = {
    "canny": preprocess_canny,
    "depth": preprocess_depth,
    "pose": preprocess_pose,
}


def preprocess(
    cond_type: str,
    image: Union[Image.Image, np.ndarray, str],
    size: int = 1024,
    **kwargs,
) -> torch.Tensor:
    if cond_type not in PREPROCESSORS:
        raise ValueError(f"unknown cond_type {cond_type!r}; expected one of {list(PREPROCESSORS)}")
    return PREPROCESSORS[cond_type](image, size=size, **kwargs)
