"""Caption / chat with a trained VLM.

    python -m vlm.infer --ckpt checkpoints/stage2 --image cat.jpg --prompt "What is in this image?"
"""

from __future__ import annotations

import argparse

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from .data import num_image_tokens_for
from .modeling import LlavaConfig, LlavaVLM, add_image_token


def load_model(ckpt_dir, base_lm=None, vision_model="google/siglip-so400m-patch14-384"):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir if base_lm is None else base_lm)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    add_image_token(tokenizer)
    cfg = LlavaConfig(
        vision_model=vision_model,
        language_model=f"{ckpt_dir}/language_model" if base_lm is None else base_lm,
        freeze_vision=True,
        freeze_language=True,
    )
    model = LlavaVLM(cfg, tokenizer)
    model.projector.load_state_dict(torch.load(f"{ckpt_dir}/projector.pt", map_location="cpu"))
    model.eval()
    return model, tokenizer, AutoProcessor.from_pretrained(vision_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="Describe this image.")
    ap.add_argument("--base-lm", default=None, help="set for stage-1 ckpt (projector only)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(args.ckpt, base_lm=args.base_lm)
    model.to(device)
    n_img = num_image_tokens_for(model.vision_tower.config)

    image = Image.open(args.image).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device, model.projector.fc1.weight.dtype)

    # Render the prompt with the chat template; expand <image> into n_img slots.
    messages = [{"role": "user", "content": f"<image>\n{args.prompt}"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    img_id = tokenizer.convert_tokens_to_ids("<image>")
    pos = ids.index(img_id)
    ids = ids[:pos] + [img_id] * n_img + ids[pos + 1:]
    input_ids = torch.tensor([ids], device=device)

    out = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
