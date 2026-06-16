# Vision-Language Model from Scratch in PyTorch

Build an end-to-end multimodal vision-language model that ingests an image plus a text prompt and autoregressively generates a caption. You will implement every component from raw tensor operations: a ViT image encoder, a vision-to-language projector, a causal text decoder, multimodal fusion, the training loop, and sampling-based generation.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** split_image_into_patches
- [x] **2.** flatten_patches
- [x] **3.** linear_projection
- [x] **4.** project_patches_to_embeddings
- [ ] **5.** prepend_class_token
- [ ] **6.** add_position_embeddings
- [ ] **7.** compute_attention_scores
- [ ] **8.** scale_attention_scores
- [ ] **9.** apply_attention_mask
- [ ] **10.** attention_softmax
- [ ] **11.** attention_context
- [ ] **12.** scaled_dot_product_attention
- [ ] **13.** split_into_heads
- [ ] **14.** merge_heads
- [ ] **15.** project_qkv
- [ ] **16.** split_qkv_into_heads
- [ ] **17.** multi_head_attention_scores
- [ ] **18.** merge_and_output_project
- [ ] **19.** multi_head_self_attention
- [ ] **20.** gelu_activation
- [ ] **21.** mlp_first_layer
- [ ] **22.** mlp_second_layer
- [ ] **23.** mlp_block
- [ ] **24.** compute_layernorm_stats
- [ ] **25.** layer_norm
- [ ] **26.** residual_add
- [ ] **27.** pre_norm_sublayer
- [ ] **28.** vision_encoder_block
- [ ] **29.** vision_encoder
- [ ] **30.** extract_patch_features
- [ ] **31.** projector_first_layer
- [ ] **32.** projector_second_layer
- [ ] **33.** vision_language_projector
- [ ] **34.** build_token_vocabulary
- [ ] **35.** encode_text_to_ids
- [ ] **36.** embed_token_ids
- [ ] **37.** add_text_position_embeddings
- [ ] **38.** find_image_placeholder_positions
- [ ] **39.** insert_image_tokens
- [ ] **40.** build_multimodal_embeddings
- [ ] **41.** build_label_tensor
- [ ] **42.** build_causal_mask
- [ ] **43.** decoder_block
- [ ] **44.** language_model_decoder
- [ ] **45.** final_layer_norm
- [ ] **46.** language_model_head
- [ ] **47.** encode_image_to_tokens
- [ ] **48.** vision_language_forward
- [ ] **49.** shift_logits_and_labels
- [ ] **50.** per_position_cross_entropy
- [ ] **51.** masked_mean_loss
- [ ] **52.** greedy_next_token
- [ ] **53.** apply_temperature
- [ ] **54.** top_k_filter
- [ ] **55.** sample_from_logits
- [ ] **56.** generate_caption
- [ ] **57.** initialize_vlm_parameters
- [ ] **58.** collect_parameters
- [ ] **59.** zero_gradients
- [ ] **60.** training_step
- [ ] **61.** apply_gradient_update
- [ ] **62.** run_training_loop

---

Built on Deep-ML.
