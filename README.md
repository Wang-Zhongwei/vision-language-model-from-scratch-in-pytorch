# Vision-Language Model: from-scratch internals → a trained, chat-capable VLM

A two-track multimodal project:

1. **From-scratch track** (`from_scratch/`) — every VLM component implemented from raw
   tensor ops: a ViT image encoder, a vision→language projector, a causal text decoder,
   multimodal fusion, the training loop, and sampling-based generation. This proves the
   internals are understood end to end.
2. **Real-track** (`vlm/`) — the same projector + multimodal-fusion design, now wired to
   production backbones (frozen **SigLIP** vision tower + **Qwen2.5** LM) and trained
   **LLaVA-style** on real image-text data across multiple H100s. The hand-derived
   projector (`from_scratch/model.py`) becomes the trainable bridge
   (`vlm/modeling.py:MLPProjector`).

## Real-track: train a VLM on real images

```bash
pip install -r requirements-train.txt

# Stage 1 — alignment: freeze vision + LM, train only the projector
./scripts/launch.sh configs/stage1_align.yaml

# Stage 2 — instruction tuning: unfreeze the LM, resume the aligned projector
./scripts/launch.sh configs/stage2_finetune.yaml

# Chat with the result
python -m vlm.infer --ckpt checkpoints/stage2 --image cat.jpg --prompt "What is in this image?"
```

Develop locally on CPU, then `REMOTE=user@h100-box ./scripts/sync_to_cluster.sh` to train.

Before burning GPU hours, sanity-check the whole forward/merge/loss path on CPU with tiny
random-init backbones (no downloads):

```bash
python -m pytest tests/test_smoke.py -q
```

## From-scratch track

```bash
python from_scratch/demo.py
```

### Steps

- [x] **1.** split_image_into_patches
- [x] **2.** flatten_patches
- [x] **3.** linear_projection
- [x] **4.** project_patches_to_embeddings
- [x] **5.** prepend_class_token
- [x] **6.** add_position_embeddings
- [x] **7.** compute_attention_scores
- [x] **8.** scale_attention_scores
- [x] **9.** apply_attention_mask
- [x] **10.** attention_softmax
- [x] **11.** attention_context
- [x] **12.** scaled_dot_product_attention
- [x] **13.** split_into_heads
- [x] **14.** merge_heads
- [x] **15.** project_qkv
- [x] **16.** split_qkv_into_heads
- [x] **17.** multi_head_attention_scores
- [x] **18.** merge_and_output_project
- [x] **19.** multi_head_self_attention
- [x] **20.** gelu_activation
- [x] **21.** mlp_first_layer
- [x] **22.** mlp_second_layer
- [x] **23.** mlp_block
- [x] **24.** compute_layernorm_stats
- [x] **25.** layer_norm
- [x] **26.** residual_add
- [x] **27.** pre_norm_sublayer
- [x] **28.** vision_encoder_block
- [x] **29.** vision_encoder
- [x] **30.** extract_patch_features
- [x] **31.** projector_first_layer
- [x] **32.** projector_second_layer
- [x] **33.** vision_language_projector
- [x] **34.** build_token_vocabulary
- [x] **35.** encode_text_to_ids
- [x] **36.** embed_token_ids
- [x] **37.** add_text_position_embeddings
- [x] **38.** find_image_placeholder_positions
- [x] **39.** insert_image_tokens
- [x] **40.** build_multimodal_embeddings
- [x] **41.** build_label_tensor
- [x] **42.** build_causal_mask
- [x] **43.** decoder_block
- [x] **44.** language_model_decoder
- [x] **45.** final_layer_norm
- [x] **46.** language_model_head
- [x] **47.** encode_image_to_tokens
- [x] **48.** vision_language_forward
- [x] **49.** shift_logits_and_labels
- [x] **50.** per_position_cross_entropy
- [x] **51.** masked_mean_loss
- [x] **52.** greedy_next_token
- [x] **53.** apply_temperature
- [x] **54.** top_k_filter
- [x] **55.** sample_from_logits
- [x] **56.** generate_caption
- [x] **57.** initialize_vlm_parameters
- [x] **58.** collect_parameters
- [x] **59.** zero_gradients
- [x] **60.** training_step
- [x] **61.** apply_gradient_update
- [x] **62.** run_training_loop

---

The from-scratch component breakdown follows the Deep-ML VLM curriculum; the real-track
training pipeline (`vlm/`), backbone integration, and two-stage recipe are original.
