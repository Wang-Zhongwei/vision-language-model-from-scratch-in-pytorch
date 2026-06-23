"""Two-stage LLaVA-style trainer. Multi-GPU via `accelerate launch` / torchrun.

    accelerate launch -m vlm.train --config configs/stage1_align.yaml
    accelerate launch -m vlm.train --config configs/stage2_finetune.yaml

Stage 1 trains only the projector (vision + LM frozen). Stage 2 unfreezes the LM
(optionally LoRA) and trains on instruction data, initialized from the stage-1 projector.
"""

from __future__ import annotations

import argparse
import math
import os

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer, get_cosine_schedule_with_warmup

from .data import Collator, Stage1CaptionDataset, Stage2InstructDataset, num_image_tokens_for
from .modeling import LlavaConfig, LlavaVLM, add_image_token


def load_records(source: dict):
    """Return an indexable of dicts. Supports a local JSONL manifest or an HF dataset.

    JSONL  : {"source":{"type":"jsonl","path":"data/stage1.jsonl"}}
             each line: {"image": "<path>", "caption": "..."}  (or "conversations": [...])
    HF      : {"source":{"type":"hf","id":"nlphuji/flickr30k","split":"test",
                          "image_col":"image","caption_col":"caption"}}
    """
    t = source["type"]
    if t == "jsonl":
        import json
        with open(source["path"]) as f:
            return [json.loads(line) for line in f]
    if t == "hf":
        from datasets import load_dataset
        ds = load_dataset(source["id"], split=source.get("split", "train"))
        if "caption_col" in source:  # stage 1 shape
            ic, cc = source["image_col"], source["caption_col"]
            return [{"image": r[ic], "caption": r[cc] if isinstance(r[cc], str) else r[cc][0]} for r in ds]
        return list(ds)  # already in {image, conversations} shape
    raise ValueError(f"unknown source type {t!r}")


def build(cfg):
    mc = cfg["model"]
    model_cfg = LlavaConfig(
        vision_model=mc["vision_model"],
        language_model=mc["language_model"],
        freeze_vision=mc.get("freeze_vision", True),
        freeze_language=mc.get("freeze_language", True),
        projector_hidden_mult=mc.get("projector_hidden_mult", 4),
        attn_implementation=mc.get("attn_implementation", "sdpa"),
        dtype=mc.get("dtype", "bfloat16"),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.language_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    add_image_token(tokenizer, model_cfg.image_token)
    image_processor = AutoProcessor.from_pretrained(model_cfg.vision_model)

    model = LlavaVLM(model_cfg, tokenizer)
    if cfg.get("init_projector_from"):  # stage 2 resumes the aligned projector
        sd = torch.load(cfg["init_projector_from"], map_location="cpu")
        model.projector.load_state_dict(sd)

    n_img = num_image_tokens_for(model.vision_tower.config)
    DatasetCls = Stage1CaptionDataset if cfg["stage"] == 1 else Stage2InstructDataset
    records = load_records(cfg["data"]["source"])
    dataset = DatasetCls(records, tokenizer, image_processor, n_img, model_cfg.image_token)
    return model, tokenizer, dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg["train"]
    mp = {"bfloat16": "bf16", "float16": "fp16", "float32": "no"}[cfg["model"].get("dtype", "bfloat16")]
    accelerator = Accelerator(
        gradient_accumulation_steps=tcfg.get("grad_accum", 1),
        mixed_precision=mp,
    )
    set_seed(tcfg.get("seed", 0))

    model, tokenizer, dataset = build(cfg)
    loader = DataLoader(
        dataset,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        collate_fn=Collator(tokenizer.pad_token_id),
        num_workers=tcfg.get("num_workers", 8),
        pin_memory=True,
    )

    params = model.trainable_parameters()
    n_trainable = sum(p.numel() for p in params)
    accelerator.print(f"Trainable params: {n_trainable/1e6:.1f}M  | stage {cfg['stage']}")

    optim = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0))
    total_steps = math.ceil(len(loader) / tcfg.get("grad_accum", 1)) * tcfg["epochs"]
    sched = get_cosine_schedule_with_warmup(optim, int(total_steps * tcfg.get("warmup_ratio", 0.03)), total_steps)

    model, optim, loader, sched = accelerator.prepare(model, optim, loader, sched)

    # --- Optional Weights & Biases logging (main process only). ---
    wcfg = tcfg.get("wandb")
    use_wandb = bool(wcfg) and accelerator.is_main_process
    if use_wandb:
        import wandb
        wandb.init(
            project=wcfg.get("project", "vlm-llava"),
            name=wcfg.get("run_name", f"stage{cfg['stage']}"),
            config={**cfg["model"], **tcfg, "stage": cfg["stage"], "trainable_params_M": n_trainable / 1e6},
        )

    model.train()
    step = 0
    for epoch in range(tcfg["epochs"]):
        for batch in loader:
            with accelerator.accumulate(model):
                out = model(**batch)
                accelerator.backward(out.loss)
                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(params, tcfg.get("max_grad_norm", 1.0))
                optim.step()
                sched.step()
                optim.zero_grad()
            if accelerator.sync_gradients:
                step += 1
                if step % tcfg.get("log_every", 10) == 0:
                    loss = out.loss.item()
                    lr = sched.get_last_lr()[0]
                    accelerator.print(f"epoch {epoch} step {step}/{total_steps} loss {loss:.4f} lr {lr:.2e}")
                    if use_wandb:
                        log = {"train/loss": loss, "train/lr": lr, "train/epoch": epoch}
                        if grad_norm is not None:
                            log["train/grad_norm"] = float(grad_norm)
                        wandb.log(log, step=step)

    # --- Save. Stage 1 saves just the projector; stage 2 saves the full model. ---
    accelerator.wait_for_everyone()
    out_dir = tcfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    if cfg["stage"] == 1:
        accelerator.save(unwrapped.projector.state_dict(), os.path.join(out_dir, "projector.pt"))
    else:
        if accelerator.is_main_process:
            unwrapped.language_model.save_pretrained(os.path.join(out_dir, "language_model"))
            accelerator.save(unwrapped.projector.state_dict(), os.path.join(out_dir, "projector.pt"))
            tokenizer.save_pretrained(out_dir)
    accelerator.print(f"Saved to {out_dir}")
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
