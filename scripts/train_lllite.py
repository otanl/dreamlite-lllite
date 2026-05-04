"""Train a ControlNet-LLLite adapter for DreamLite-mobile.

Loads DreamLite components (VAE, text encoder, scheduler, UNet), freezes
all of them, attaches LLLite, and trains only the adapter parameters with
the flow-matching velocity loss that DreamLite was originally trained with:

    x_0 = VAE.encode(image)
    eps = randn_like(x_0)
    t   ~ U[0, 1)            # sampled per-step
    x_t = (1 - t) * x_0 + t * eps
    v_target = eps - x_0
    v_pred   = UNet(model_input, t, prompt_embeds, ...)
    loss     = MSE(v_pred, v_target)

`model_input` follows DreamLite's "In-Context Spatial Concatenation" — for
T2I training we concatenate `x_t` with zeros along the width axis, matching
the inference-time `task='generate'` path.

Notes:
  * Only the LLLite controller's parameters are trained (~13 M for the
    default config). Everything else is frozen and put in eval mode.
  * Text encoder and VAE are kept on CPU by default (lazy-moved to GPU
    only during the forward) to save VRAM. Toggle with --offload_to_cpu.
  * Gradient checkpointing on the UNet is recommended for batch_size > 1.
  * The flow-matching scheduler is shared with inference, but at training
    time we sample t directly rather than from the discrete sigma grid;
    this matches typical rectified-flow recipes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, os.environ.get("DREAMLITE_ROOT", "../DreamLite"))

from dreamlite_lllite import apply_lllite  # noqa: E402


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class LLLiteDataset(Dataset):
    """Reads a manifest JSONL produced by `prepare_dataset.py`.

    Returns:
      image:   (3, H, W) float in [-1, 1]      — VAE input
      cond:    (3, H, W) float in [-1, 1]      — adapter input
      caption: str
    """

    def __init__(self, manifest_path: str, size: int = 1024):
        self.size = size
        rows = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        self.rows = rows
        self.t = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(size),
            transforms.ToTensor(),  # -> (3, H, W) in [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1,1]
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        img = Image.open(row["image"]).convert("RGB")
        cond = Image.open(row["cond"]).convert("RGB")
        return {
            "image": self.t(img),
            "cond": self.t(cond),
            "caption": row.get("caption", ""),
        }


class LLLiteCachedDataset(Dataset):
    """Reads cache_latents.py output: each shard contains pre-computed
    latent / prompt_embeds / prompt_mask. Cond image is still loaded from disk
    (it's small and cheap to read) so the conditioning encoder gets fresh
    data each step.
    """

    def __init__(self, manifest_path: str, size: int = 1024):
        self.size = size
        rows = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        self.rows = rows
        self.t = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        shard = torch.load(row["shard"], map_location="cpu", weights_only=True)
        cond = Image.open(row["cond"]).convert("RGB")
        return {
            "latent": shard["latent"],
            "prompt_embeds": shard["prompt_embeds"],
            "prompt_mask": shard["prompt_mask"],
            "cond": self.t(cond),
            "caption": row.get("caption", ""),
        }


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "image" in batch[0]:
        out["images"] = torch.stack([b["image"] for b in batch], 0)
    if "latent" in batch[0]:
        out["latents"] = torch.stack([b["latent"] for b in batch], 0)
        # prompt embeds may have varying length; pad to max
        L = max(b["prompt_embeds"].shape[0] for b in batch)
        D = batch[0]["prompt_embeds"].shape[1]
        embeds = torch.zeros(len(batch), L, D, dtype=batch[0]["prompt_embeds"].dtype)
        masks = torch.zeros(len(batch), L, dtype=batch[0]["prompt_mask"].dtype)
        for i, b in enumerate(batch):
            n = b["prompt_embeds"].shape[0]
            embeds[i, :n] = b["prompt_embeds"]
            masks[i, :n] = b["prompt_mask"]
        out["prompt_embeds"] = embeds
        out["prompt_mask"] = masks
    out["conds"] = torch.stack([b["cond"] for b in batch], 0)
    out["captions"] = [b["caption"] for b in batch]
    return out


# ----------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------
def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _freeze(*modules: torch.nn.Module) -> None:
    for m in modules:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)


# ----------------------------------------------------------------------
# Main train loop
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/DreamLite-mobile")
    ap.add_argument("--manifest", default=None,
                    help="raw manifest.jsonl from prepare_dataset.py "
                         "(omit if you pass --cached_manifest)")
    ap.add_argument("--out_dir", required=True, help="checkpoint dir")
    ap.add_argument("--cond_type", choices=["canny", "depth", "pose"], required=True,
                    help="for metadata only — actual data comes from manifest")
    ap.add_argument("--size", type=int, default=1024)
    # Defaults track kohya-ss's published LLLite training config
    # (sd-scripts/docs/train_lllite_README.md): lr=2e-4, adamw8bit, batch=8,
    # 12 epochs, bf16, gradient checkpointing.
    ap.add_argument("--batch_size", type=int, default=2,
                    help="per-step batch (kohya recommends 8; combine with --gradient_accumulation_steps)")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4,
                    help="effective batch = batch_size * grad_accum (kohya recommends 8 effective)")
    ap.add_argument("--learning_rate", type=float, default=2e-4,
                    help="kohya recommended 2e-4 for canny; for depth, halve it")
    ap.add_argument("--max_steps", type=int, default=0,
                    help="optimizer steps; 0 = use --max_epochs instead")
    ap.add_argument("--max_epochs", type=int, default=12,
                    help="kohya default 12 epochs (used only if --max_steps == 0)")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--sample_every", type=int, default=0,
                    help="if >0, run a single inference sample every N optimizer steps")
    ap.add_argument("--sample_prompt", default="a photo of a dog",
                    help="prompt used when --sample_every > 0")
    ap.add_argument("--cond_emb_dim", type=int, default=32)
    ap.add_argument("--mlp_dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mixed_precision", choices=["bf16", "fp16", "no"], default="bf16")
    ap.add_argument("--optimizer", choices=["adamw", "adamw8bit"], default="adamw8bit",
                    help="kohya recommends adamw8bit (bitsandbytes) for memory savings")
    ap.add_argument("--lr_scheduler", choices=["constant", "cosine", "constant_with_warmup"],
                    default="constant_with_warmup")
    ap.add_argument("--lr_warmup_steps", type=int, default=100)
    ap.add_argument("--gradient_checkpointing", action="store_true",
                    help="enable on the UNet to save VRAM")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--resume", default=None, help="path to LLLite .safetensors to resume")
    ap.add_argument("--cap_drop_prob", type=float, default=0.1,
                    help="probability of dropping caption (CFG-style)")
    ap.add_argument("--cached_manifest", default=None,
                    help="manifest.jsonl produced by cache_latents.py; "
                         "when set, skip VAE+text encoder forward at training time")
    args = ap.parse_args()

    _seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device("cuda")
    weight_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "no": torch.float32,
    }[args.mixed_precision]

    # ------------------------------------------------------------------
    # Load the DreamLite pipeline (auto-detects mobile vs base from
    # model_index.json). We only use sub-modules from it.
    # ------------------------------------------------------------------
    from dreamlite_lllite.pipeline import _detect_pipeline_class
    PipelineCls, variant = _detect_pipeline_class(args.model)
    print(f"loading {variant} from {args.model}…")
    pipe = PipelineCls.from_pretrained(args.model, torch_dtype=weight_dtype).to(device)
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    processor = pipe.processor
    unet = pipe.unet
    scheduler = pipe.scheduler

    # We mirror the inference-time encode_prompt() helper but call it
    # through the pipe object — it already handles padding/embedding
    # extraction.
    def encode_text(captions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        embeds, mask = pipe.encode_prompt(
            mode="generate",
            prompts=[f"[Generate]: {c}" for c in captions],
            device=device,
            dtype=weight_dtype,
        )
        return embeds, mask

    _freeze(vae, text_encoder, unet)
    if args.gradient_checkpointing:
        # DreamLite ships an older `_set_gradient_checkpointing(self, module, value)`
        # signature that's incompatible with diffusers' newer
        # `enable_gradient_checkpointing()`. Apply it manually.
        for m in unet.modules():
            if hasattr(m, "gradient_checkpointing"):
                m.gradient_checkpointing = True
        unet.train()  # checkpointing requires train() but params remain frozen

    # ------------------------------------------------------------------
    # Attach LLLite
    # ------------------------------------------------------------------
    print("attaching LLLite controller…")
    # Latent spatial size for this resolution (TAESDXL = 8x downsample)
    vae_downsample = 2 ** (len(vae.config.encoder_block_out_channels) - 1)
    latent_hw = args.size // vae_downsample
    controller = apply_lllite(
        unet,
        cond_emb_dim=args.cond_emb_dim,
        mlp_dim=args.mlp_dim,
        cond_image_size=args.size,
        sample_size=latent_hw,
    )
    # Train adapter in fp32 for stable gradients; cast on the fly
    controller.to(device=device, dtype=torch.float32)
    controller.train()
    n_train = controller.num_parameters()
    print(f"  trainable params: {n_train:,} ({n_train/1e6:.2f} M)")

    if args.resume:
        from safetensors.torch import load_file
        sd = load_file(args.resume)
        controller.load_state_dict(sd, strict=True)
        print(f"  resumed weights from {args.resume}")

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------
    if args.cached_manifest:
        ds = LLLiteCachedDataset(args.cached_manifest, size=args.size)
        print(f"cached dataset size: {len(ds)}")
    else:
        ds = LLLiteDataset(args.manifest, size=args.size)
        print(f"dataset size: {len(ds)}")
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
        drop_last=True,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    if args.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                controller.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.0,
                eps=1e-8,
            )
        except ImportError:
            print("WARN: bitsandbytes not available, falling back to torch AdamW")
            args.optimizer = "adamw"
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            controller.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            eps=1e-8,
        )

    # Resolve total optimizer steps from --max_epochs when --max_steps==0
    if args.max_steps <= 0:
        steps_per_epoch = max(1, len(ds) // (args.batch_size * args.gradient_accumulation_steps))
        args.max_steps = steps_per_epoch * args.max_epochs
        print(f"  max_steps from {args.max_epochs} epochs × {steps_per_epoch} steps/ep = {args.max_steps}")

    # LR scheduler
    if args.lr_scheduler == "constant":
        lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    elif args.lr_scheduler == "cosine":
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.max_steps - args.lr_warmup_steps)
        )
    else:  # constant_with_warmup
        warmup = max(1, args.lr_warmup_steps)

        def _warmup_then_const(step: int) -> float:
            if step < warmup:
                return step / warmup
            return 1.0
        lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup_then_const)

    grad_scaler = (
        torch.amp.GradScaler("cuda")
        if args.mixed_precision == "fp16" else None
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    vae_scaling = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0.0)
    # DreamLite UNet's `addition_embed_type == "time"` consumes
    # time_ids = [[width, height]] (per-sample), projecting each value via
    # add_time_proj and reshaping to (B, projection_class_embeddings_input_dim).
    add_time_ids = torch.tensor(
        [[args.size, args.size]] * args.batch_size,
        device=device, dtype=weight_dtype,
    )

    step = 0
    micro_step = 0
    t_start = time.time()
    pbar = tqdm(total=args.max_steps, desc="train")
    losses_window: List[float] = []

    use_cache = args.cached_manifest is not None
    while step < args.max_steps:
        for batch in dl:
            conds = batch["conds"].to(device=device, dtype=weight_dtype)
            captions = list(batch["captions"])

            if use_cache:
                latents = batch["latents"].to(device=device, dtype=weight_dtype)
                prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=weight_dtype)
                prompt_mask = batch["prompt_mask"].to(device=device)
                # CFG-style caption drop: zero out the embeds for dropped rows
                if args.cap_drop_prob > 0:
                    drops = torch.rand(prompt_embeds.shape[0]) < args.cap_drop_prob
                    if drops.any():
                        prompt_embeds[drops] = 0
                        prompt_mask[drops] = 0
            else:
                images = batch["images"].to(device=device, dtype=weight_dtype)
                if args.cap_drop_prob > 0:
                    captions = ["" if random.random() < args.cap_drop_prob else c for c in captions]
                with torch.no_grad():
                    latents = vae.encode(images).latents
                    latents = (latents - vae_shift) * vae_scaling
                    prompt_embeds, prompt_mask = encode_text(captions)

            # 3. Set conditioning on adapter (precompute embed once per batch)
            controller.set_cond_image(conds)

            # 4. Sample timestep & noise (rectified-flow scheduling)
            B = latents.shape[0]
            # Logit-normal sampling (matches FLUX / SD3 training); DreamLite
            # uses FlowMatchEuler at inference, the t distribution at train
            # time can differ. Logit-normal puts more weight on mid-range t.
            u = torch.randn(B, device=device)
            t = torch.sigmoid(u)  # (B,) in (0,1)
            t = t.to(weight_dtype)

            noise = torch.randn_like(latents)
            t_b = t.view(B, 1, 1, 1)
            x_t = (1 - t_b) * latents + t_b * noise
            v_target = noise - latents

            # 5. Build DreamLite "spatial concat" model input — for T2I,
            #    the conditioning half is zeros.
            zero_half = torch.zeros_like(x_t)
            model_input = torch.cat([x_t, zero_half], dim=3)

            # 6. UNet forward with adapter active
            with torch.amp.autocast("cuda", dtype=weight_dtype, enabled=args.mixed_precision != "no"):
                v_pred = unet(
                    model_input,
                    timestep=(t * 1000.0).to(weight_dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                    added_cond_kwargs={"time_ids": add_time_ids[:B]},
                    return_dict=False,
                )[0]
                # DreamLite UNet outputs the same width as input; only the
                # "left half" (the latents portion) is the prediction.
                v_pred = v_pred[..., : latents.shape[-1]]
                loss = F.mse_loss(v_pred.float(), v_target.float()) / args.gradient_accumulation_steps

            # 7. Backward & step
            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()
            micro_step += 1

            losses_window.append(loss.item() * args.gradient_accumulation_steps)

            if micro_step % args.gradient_accumulation_steps == 0:
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
                if grad_scaler is not None:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
                lr_sched.step()
                optimizer.zero_grad(set_to_none=True)

                step += 1
                pbar.update(1)

                if step % args.log_every == 0:
                    avg = sum(losses_window) / max(1, len(losses_window))
                    losses_window.clear()
                    elapsed = time.time() - t_start
                    pbar.set_postfix(loss=f"{avg:.4f}", sec=f"{elapsed:.0f}")

                if step % args.save_every == 0 or step == args.max_steps:
                    out = out_dir / f"lllite_{args.cond_type}_step{step:06d}.safetensors"
                    from safetensors.torch import save_file
                    sd = {k: v.detach().cpu().to(torch.float32) for k, v in controller.state_dict().items()}
                    md = {
                        "cond_type": args.cond_type,
                        "cond_emb_dim": str(args.cond_emb_dim),
                        "mlp_dim": str(args.mlp_dim),
                        "step": str(step),
                        "model": args.model,
                    }
                    save_file(sd, str(out), md)
                    print(f"  saved {out}")

                if args.sample_every > 0 and step % args.sample_every == 0:
                    sample_path = out_dir / f"sample_step{step:06d}.png"
                    try:
                        controller.eval()
                        with torch.no_grad():
                            controller.set_cond_image(conds[:1])
                            out_img = pipe(
                                prompt=args.sample_prompt,
                                num_inference_steps=4,
                                height=args.size,
                                width=args.size,
                                generator=torch.Generator("cpu").manual_seed(0),
                            ).images[0]
                            out_img.save(sample_path)
                            print(f"  sample -> {sample_path}")
                        controller.set_cond_image(None)
                    finally:
                        controller.train()

                if step >= args.max_steps:
                    break
        if step >= args.max_steps:
            break

    pbar.close()
    print("done")


if __name__ == "__main__":
    main()
