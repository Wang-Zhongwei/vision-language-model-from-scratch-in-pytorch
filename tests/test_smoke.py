"""End-to-end smoke test for the real-track VLM, runnable on CPU with no downloads.

Builds tiny random-init stand-ins for the SigLIP vision tower and the Qwen2 LM, injects
them into LlavaVLM, and exercises the full path: vision -> projector -> image-token merge
-> LM -> loss -> backward. This de-risks the pipeline before spending H100 hours; the only
thing it can't catch is behavior specific to the real checkpoints (weights, tokenizer).

Run:  python -m pytest tests/test_smoke.py -q      (or)      python tests/test_smoke.py
"""

import os
import sys

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm.data import Collator, IGNORE_INDEX
from vlm.modeling import LlavaConfig, LlavaVLM

IMAGE_ID = 7
VOCAB = 32


class StubTokenizer:
    """Minimal tokenizer surface used by LlavaVLM.__init__ / _merge."""
    pad_token_id = 0
    unk_token_id = 1

    def convert_tokens_to_ids(self, tok):
        return IMAGE_ID if tok == "<image>" else self.unk_token_id

    def __len__(self):
        return VOCAB


def build_tiny_model():
    vision = SiglipVisionModel(SiglipVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, image_size=32, patch_size=16, num_channels=3,
    ))
    lm = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        attn_implementation="eager",
    ))
    cfg = LlavaConfig(freeze_vision=True, freeze_language=True, dtype="float32")
    model = LlavaVLM(cfg, StubTokenizer(), vision_tower=vision, language_model=lm)
    n_patch = (vision.config.image_size // vision.config.patch_size) ** 2  # 4
    return model, n_patch


def make_batch(n_patch, batch_size=2):
    """Each example: [<image> * n_patch] + a few text tokens; supervise only the text."""
    items = []
    for _ in range(batch_size):
        text = [5, 6, 9]
        input_ids = torch.tensor([IMAGE_ID] * n_patch + text)
        labels = torch.tensor([IGNORE_INDEX] * n_patch + text)
        items.append({
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": torch.randn(3, 32, 32),
        })
    return Collator(pad_token_id=0)(items)


def test_forward_and_backward():
    torch.manual_seed(0)
    model, n_patch = build_tiny_model()
    batch = make_batch(n_patch)

    out = model(**batch)
    assert out.loss.dim() == 0 and torch.isfinite(out.loss), out.loss
    out.loss.backward()

    # Stage-1 freezing: only the projector accumulates gradients.
    proj_grad = all(p.grad is not None for p in model.projector.parameters())
    vision_frozen = all(p.grad is None for p in model.vision_tower.parameters())
    lm_frozen = all(p.grad is None for p in model.language_model.parameters())
    assert proj_grad and vision_frozen and lm_frozen


def test_image_token_count_matches_patches():
    """The #1 thing to get right on the real run: <image> placeholders == vision patches."""
    model, n_patch = build_tiny_model()
    batch = make_batch(n_patch)
    n_image_tokens = int((batch["input_ids"] == IMAGE_ID).sum())
    n_vision = model.encode_images(batch["pixel_values"]).reshape(-1, 32).shape[0]
    assert n_image_tokens == n_vision, (n_image_tokens, n_vision)


if __name__ == "__main__":
    test_forward_and_backward()
    test_image_token_count_matches_patches()
    print("OK — forward/backward/freezing + image-token-count match.")
