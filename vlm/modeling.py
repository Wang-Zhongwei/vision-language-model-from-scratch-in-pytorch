"""LLaVA-style VLM = frozen SigLIP vision tower + MLP projector + causal LM.

The projector mirrors the from-scratch design in `from_scratch/model.py`:

    projector_first_layer:  h = GELU(x @ w1 + b1)        # (N, d_vision) -> (N, d_hidden)
    projector_second_layer: y = h @ w2 + b2              # (N, d_hidden) -> (N, d_text)

Here it becomes a trainable `nn.Module` so it can learn by backprop on real data,
but the architecture is identical to what you implemented by hand.

Forward pass (the LLaVA recipe):
  1. Run the (frozen) SigLIP tower on pixel_values -> patch features (B, N_patch, d_vision)
  2. Project patch features into the LM embedding space -> image tokens (B, N_patch, d_text)
  3. Embed the text input_ids with the LM's own embedding table
  4. Splice the image tokens into the sequence at <image> placeholder positions
     (the trainable analogue of build_multimodal_embeddings / insert_image_tokens)
  5. Run the causal LM on the merged inputs_embeds and compute next-token loss
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM


@dataclass
class LlavaConfig:
    vision_model: str = "google/siglip-so400m-patch14-384"
    language_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    projector_hidden_mult: int = 4          # d_hidden = mult * d_text (LLaVA uses a wide MLP)
    image_token: str = "<image>"            # placeholder token spliced with vision features
    freeze_vision: bool = True
    freeze_language: bool = True            # stage 1: True (align only). stage 2: False.
    attn_implementation: str = "sdpa"       # "flash_attention_2" on the H100 box if installed
    dtype: str = "bfloat16"
    # Filled in at build time from the loaded backbones:
    extra: dict = field(default_factory=dict)


class MLPProjector(nn.Module):
    """Two-layer GELU MLP, identical in shape to the from-scratch projector.

    d_vision -> d_hidden -> d_text, with GELU after the first layer only.
    """

    def __init__(self, d_vision: int, d_text: int, hidden_mult: int = 4):
        super().__init__()
        d_hidden = hidden_mult * d_text
        self.fc1 = nn.Linear(d_vision, d_hidden)   # <-> projector w1, b1
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_hidden, d_text)     # <-> projector w2, b2

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(patch_features)))


def _dtype_of(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


class LlavaVLM(nn.Module):
    def __init__(self, config: LlavaConfig, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        dtype = _dtype_of(config.dtype)

        # --- Vision tower (frozen): SigLIP has no CLS token; every output token is a patch. ---
        vision = AutoModel.from_pretrained(config.vision_model, torch_dtype=dtype)
        self.vision_tower = getattr(vision, "vision_model", vision)
        d_vision = self.vision_tower.config.hidden_size

        # --- Language model (modern: RoPE/RMSNorm/SwiGLU/GQA already inside). ---
        self.language_model = AutoModelForCausalLM.from_pretrained(
            config.language_model,
            torch_dtype=dtype,
            attn_implementation=config.attn_implementation,
        )
        d_text = self.language_model.config.hidden_size

        # --- The bridge: your projector design, now trainable. ---
        self.projector = MLPProjector(d_vision, d_text, config.projector_hidden_mult).to(dtype)

        # Resolve the <image> placeholder id (added to the tokenizer in data.py).
        self.image_token_id = tokenizer.convert_tokens_to_ids(config.image_token)
        if self.image_token_id is None or self.image_token_id == tokenizer.unk_token_id:
            raise ValueError(
                f"Image token {config.image_token!r} not in tokenizer. "
                "Call add_image_token(tokenizer) before building the model."
            )
        # Embedding table may have grown when the image token was added.
        self.language_model.resize_token_embeddings(len(tokenizer))

        self.config.extra = {"d_vision": d_vision, "d_text": d_text}
        self._apply_freezing()

    def _apply_freezing(self) -> None:
        if self.config.freeze_vision:
            self.vision_tower.requires_grad_(False)
            self.vision_tower.eval()
        if self.config.freeze_language:
            self.language_model.requires_grad_(False)
        # Projector is always trainable.
        self.projector.requires_grad_(True)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # --- Vision -> image tokens -----------------------------------------------------------
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values (B, 3, H, W) -> image tokens (B, N_patch, d_text)."""
        ctx = torch.no_grad() if self.config.freeze_vision else torch.enable_grad()
        with ctx:
            out = self.vision_tower(pixel_values=pixel_values)
            patch_features = out.last_hidden_state  # (B, N_patch, d_vision)
        return self.projector(patch_features.to(self.projector.fc1.weight.dtype))

    # --- Multimodal merge: scatter image tokens into the <image> slots --------------------
    def _merge(self, input_ids, image_embeds, attention_mask, labels):
        """Replace embeddings at <image> positions with projected patch features.

        Assumes data.py expanded each <image> placeholder into exactly N_patch copies
        so the sequence length already accounts for all image tokens (standard LLaVA prep).
        """
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        image_mask = input_ids == self.image_token_id  # (B, S)
        # Flatten-scatter keeps it simple and batch-shape agnostic.
        merged = text_embeds.clone()
        merged[image_mask] = image_embeds.reshape(-1, image_embeds.size(-1)).to(text_embeds.dtype)
        return merged, attention_mask, labels

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        image_embeds = self.encode_images(pixel_values)
        inputs_embeds, attention_mask, labels = self._merge(
            input_ids, image_embeds, attention_mask, labels
        )
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, input_ids, pixel_values, attention_mask=None, **gen_kwargs):
        image_embeds = self.encode_images(pixel_values)
        inputs_embeds, attention_mask, _ = self._merge(input_ids, image_embeds, attention_mask, None)
        return self.language_model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_kwargs
        )


def add_image_token(tokenizer, image_token: str = "<image>"):
    """Register the <image> placeholder as a single special token. Idempotent."""
    if image_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
    return tokenizer
