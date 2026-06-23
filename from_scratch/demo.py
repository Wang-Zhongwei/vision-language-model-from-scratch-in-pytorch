"""
Vision-Language Model from Scratch in PyTorch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

import numpy as np
import torch

from model import (
    build_label_tensor,
    build_token_vocabulary,
    collect_parameters,
    encode_text_to_ids,
    generate_caption,
    initialize_vlm_parameters,
    run_training_loop,
    vision_language_forward,
)


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    # --- Toy config: tiny everything so this runs on CPU in seconds. ---
    image_size = 8
    patch_size = 4
    num_patches = (image_size // patch_size) ** 2  # 4 image tokens
    d_model = 16
    num_heads = 2

    texts = [
        "<image> a red square",
        "<image> a blue dot",
        "<image> tiny image here",
    ]
    vocab = build_token_vocabulary(texts, image_token="<image>", pad_token="<pad>")
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")
    print(f"Vocab: {vocab}")

    config = {
        "image_size": image_size,
        "patch_size": patch_size,
        "num_patches": num_patches,
        "in_channels": 3,
        "d_model": d_model,
        "d_vision": d_model,
        "d_text": d_model,
        "num_heads": num_heads,
        "num_vision_heads": num_heads,
        "num_decoder_heads": num_heads,
        "num_vision_layers": 2,
        "num_decoder_layers": 2,
        "mlp_hidden": 4 * d_model,
        "mlp_hidden_vision": 4 * d_model,
        "mlp_hidden_text": 4 * d_model,
        "vocab_size": vocab_size,
        "max_text_len": 32,
        "num_image_tokens": num_patches,
    }

    params = initialize_vlm_parameters(config, seed=0)
    parameter_list = collect_parameters(params)
    print(f"Initialized {len(parameter_list)} parameter tensors.")

    # --- Build one toy training example. ---
    image = torch.randn(config["in_channels"], image_size, image_size)
    caption = "<image> a red square"
    token_ids = torch.tensor(encode_text_to_ids(caption, vocab), dtype=torch.long)
    image_token_id = vocab["<image>"]
    pad_token_id = vocab["<pad>"]
    labels = build_label_tensor(
        token_ids, image_token_id, pad_token_id,
        num_image_tokens=config["num_image_tokens"],
    )
    print(f"Token ids: {token_ids.tolist()}")
    print(f"Labels:    {labels.tolist()}")

    # --- One forward pass before training, to sanity-check shapes. ---
    with torch.no_grad():
        logits = vision_language_forward(image, token_ids, params)
    print(f"Forward logits shape: {tuple(logits.shape)} (expect (S, V={vocab_size}))")

    # --- Short overfit loop on this single example. ---
    batch = {"image": image, "token_ids": token_ids, "labels": labels}
    losses = run_training_loop(params, batch, num_steps=5, learning_rate=0.05)
    print("Loss curve:", [round(float(l), 4) for l in losses])

    # --- Greedy generation from a prompt. ---
    prompt = "<image>"
    prompt_ids = torch.tensor(encode_text_to_ids(prompt, vocab), dtype=torch.long)
    inv_vocab = {i: t for t, i in vocab.items()}
    generated = generate_caption(
        image, prompt_ids, params,
        max_new_tokens=4, temperature=1.0, top_k=0, do_sample=False,
    )
    gen_ids = generated.tolist() if hasattr(generated, "tolist") else list(generated)
    gen_tokens = [inv_vocab.get(i, "<unk>") for i in gen_ids]
    print(f"Greedy generated ids:    {gen_ids}")
    print(f"Greedy generated tokens: {gen_tokens}")


if __name__ == "__main__":
    main()
