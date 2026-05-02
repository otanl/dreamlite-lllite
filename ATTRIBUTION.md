# Attribution

This project is a derivative work that builds on the following projects.

## DreamLite

ControlNet-LLLite for DreamLite is **Adapted Material** (CC BY-NC 4.0 §1(a)) of
[DreamLite](https://github.com/ByteVisionLab/DreamLite) by Kailai Feng et al.
(ByteDance Ltd.).

- Paper: *DreamLite: A Lightweight On-Device Unified Model for Image Generation
  and Editing.* arXiv:2603.28713 (2026).
- License: Model weights are released under
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
  See [`WEIGHTS_LICENSE`](./WEIGHTS_LICENSE).
- Modifications: We add a ControlNet-LLLite adapter and the conditioning
  encoder that is trained on top of the DreamLite-mobile UNet. No DreamLite
  weights are redistributed by this repository; consumers must download the
  base weights from the original distribution.

## ControlNet-LLLite (kohya-ss)

The adapter architecture and naming convention are derived from
[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) (Apache-2.0,
files under `networks/`) and
[kohya-ss/ControlNet-LLLite-ComfyUI](https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI).

- License: Apache-2.0
- Modifications: Re-targeted from SDXL UNet (`input_blocks`/`output_blocks`)
  to DreamLite-mobile's diffusers-style UNet (`down_blocks`/`up_blocks`).
  Conditioning depth is computed from feature-map size rather than hardcoded
  block indices. Cross-attention is to Qwen3-VL features (dim 2304) rather
  than CLIP, but LLLite never targets cross-attention K/V so the change is
  irrelevant in practice.

## Qwen3-VL (text encoder, indirect dependency)

DreamLite-mobile uses [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) as its
text encoder. Adapter weights produced by this repository condition on
features that flow through Qwen3-VL during inference. Users of these adapter
weights are bound by the relevant Qwen3-VL license terms in addition to the
DreamLite license.

## Diffusers / Transformers

Pipeline integration uses [`diffusers`](https://github.com/huggingface/diffusers)
and [`transformers`](https://github.com/huggingface/transformers), both
Apache-2.0.

## SDXL / SnapGen / TAESDXL

DreamLite itself acknowledges SDXL, SnapGen, and TAESDXL. The compact VAE in
DreamLite-mobile is `AutoencoderTiny` (TAESDXL).

---

If you redistribute the trained LLLite weights produced by this code, you must
preserve all attributions in this file and clearly state that the weights
inherit DreamLite's CC BY-NC 4.0 license (non-commercial use only).
