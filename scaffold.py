"""
Vision-Language Model from Scratch in PyTorch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""End-to-end demo of a tiny vision-language model: build toy data, run a
short training loop, then greedily generate a caption."""

import numpy as np
import torch

from solution import (
    split_image_into_patches, flatten_patches, linear_projection,
    project_patches_to_embeddings, prepend_class_token, add_position_embeddings,
    compute_attention_scores, scale_attention_scores, apply_attention_mask,
    attention_softmax, attention_context, scaled_dot_product_attention,
    split_into_heads, merge_heads, project_qkv, split_qkv_into_heads,
    multi_head_attention_scores, merge_and_output_project,
    multi_head_self_attention, gelu_activation, mlp_first_layer,
    mlp_second_layer, mlp_block, compute_layernorm_stats, layer_norm,
    residual_add, pre_norm_sublayer, vision_encoder_block, vision_encoder,
    extract_patch_features, projector_first_layer, projector_second_layer,
    vision_language_projector, build_token_vocabulary, encode_text_to_ids,
    embed_token_ids, add_text_position_embeddings,
    find_image_placeholder_positions, insert_image_tokens,
    build_multimodal_embeddings, build_label_tensor, build_causal_mask,
    decoder_block, language_model_decoder, final_layer_norm,
    language_model_head, encode_image_to_tokens, vision_language_forward,
    shift_logits_and_labels, per_position_cross_entropy, masked_mean_loss,
    greedy_next_token, apply_temperature, top_k_filter, sample_from_logits,
    generate_caption, initialize_vlm_parameters, collect_parameters,
    zero_gradients, training_step, apply_gradient_update, run_training_loop,
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
