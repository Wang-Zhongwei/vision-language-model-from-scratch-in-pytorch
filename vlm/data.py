"""Datasets + collator for the two-stage LLaVA recipe.

Stage 1 (alignment): image -> short caption. Only the caption tokens are supervised.
Stage 2 (instruction): multi-turn conversation about the image, rendered with the
LM's chat template; only assistant turns are supervised.

The key trick (standard LLaVA): each <image> placeholder is expanded into exactly
`num_image_tokens` copies *before* tokenization, so the sequence length already has a
slot for every projected patch feature. modeling.LlavaVLM._merge then scatters the
vision features into those slots.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


def num_image_tokens_for(vision_config) -> int:
    """Patches produced by the SigLIP tower = (image_size // patch_size) ** 2."""
    return (vision_config.image_size // vision_config.patch_size) ** 2


class Stage1CaptionDataset(Dataset):
    """Alignment data: (image, caption). Supervise the caption only.

    `records` is any indexable of dicts with keys: 'image' (PIL.Image or path) and
    'caption' (str). Works with a streamed HF dataset wrapped in a list, a local
    JSON manifest, etc. — kept backbone-agnostic on purpose.
    """

    def __init__(self, records, tokenizer, image_processor, num_image_tokens, image_token="<image>"):
        self.records = records
        self.tok = tokenizer
        self.img = image_processor
        self.n_img = num_image_tokens
        self.image_token = image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    def __len__(self):
        return len(self.records)

    def _load_image(self, image):
        if isinstance(image, str):
            from PIL import Image
            image = Image.open(image).convert("RGB")
        return self.img(images=image, return_tensors="pt")["pixel_values"][0]

    def __getitem__(self, i):
        rec = self.records[i]
        pixel_values = self._load_image(rec["image"])

        # Prompt = N image-token placeholders; target = the caption.
        image_chunk = [self.image_token_id] * self.n_img
        caption_ids = self.tok(rec["caption"].strip(), add_special_tokens=False)["input_ids"]
        eos = [self.tok.eos_token_id] if self.tok.eos_token_id is not None else []

        input_ids = image_chunk + caption_ids + eos
        # Mask the image tokens (-100) so loss is only on the caption + eos.
        labels = [IGNORE_INDEX] * len(image_chunk) + caption_ids + eos

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values,
        }


class Stage2InstructDataset(Dataset):
    """Instruction data: list of {'image', 'conversations':[{role,content}, ...]}.

    Renders with the LM chat template and supervises assistant turns only. The first
    user turn must contain the <image> placeholder, which we expand to num_image_tokens.
    """

    def __init__(self, records, tokenizer, image_processor, num_image_tokens, image_token="<image>"):
        self.records = records
        self.tok = tokenizer
        self.img = image_processor
        self.n_img = num_image_tokens
        self.image_token = image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    def __len__(self):
        return len(self.records)

    def _load_image(self, image):
        if isinstance(image, str):
            from PIL import Image
            image = Image.open(image).convert("RGB")
        return self.img(images=image, return_tensors="pt")["pixel_values"][0]

    def __getitem__(self, i):
        rec = self.records[i]
        pixel_values = self._load_image(rec["image"])
        msgs = rec["conversations"]

        # Build input_ids + labels by tokenizing turn-by-turn so we can mask user turns.
        input_ids, labels = [], []
        for turn in msgs:
            rendered = self.tok.apply_chat_template(
                [turn], tokenize=False, add_generation_prompt=False
            )
            ids = self.tok(rendered, add_special_tokens=False)["input_ids"]
            supervise = turn["role"] == "assistant"
            input_ids += ids
            labels += ids if supervise else [IGNORE_INDEX] * len(ids)

        # Expand the single <image> placeholder into n_img slots (and mask them).
        if self.image_token_id in input_ids:
            pos = input_ids.index(self.image_token_id)
            fill = [self.image_token_id] * self.n_img
            input_ids = input_ids[:pos] + fill + input_ids[pos + 1:]
            labels = labels[:pos] + [IGNORE_INDEX] * self.n_img + labels[pos + 1:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values,
        }


class Collator:
    """Right-pad input_ids/labels, stack pixel_values, build the attention mask."""

    def __init__(self, pad_token_id):
        self.pad = pad_token_id

    def __call__(self, batch):
        maxlen = max(b["input_ids"].size(0) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = b["input_ids"].size(0)
            padlen = maxlen - n
            input_ids.append(torch.cat([b["input_ids"], torch.full((padlen,), self.pad, dtype=torch.long)]))
            labels.append(torch.cat([b["labels"], torch.full((padlen,), IGNORE_INDEX, dtype=torch.long)]))
            attn.append(torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(padlen, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn),
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        }
