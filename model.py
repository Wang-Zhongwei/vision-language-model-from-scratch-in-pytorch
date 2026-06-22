"""
Vision-Language Model from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - split_image_into_patches
import torch
def split_image_into_patches(image, patch_size):
    B, C, H, W = image.shape
    P = patch_size
    x = image.reshape(B, C, H // P, P, W // P, P)
    x = x.permute(0, 2, 4, 1, 3, 5) # (B, H//P, W//P, C, P, P)
    return x.reshape(B, (H // P) * (W // P), C, P, P)

# Step 2 - flatten_patches
def flatten_patches(patches):
    # TODO: flatten each patch's channel and spatial dims into one vector, keep (B, N) leading dims.
    B, N, C, _, _ = patches.shape
    return torch.reshape(patches, (B, N, -1))

# Step 3 - linear_projection
import torch

def linear_projection(x, weight, bias):
    """Apply y = x @ weight.T + bias with arbitrary leading dims on x."""
    # TODO: compute the affine map y = x @ weight.T + bias
    return x @ weight.T + bias

# Step 4 - project_patches_to_embeddings
import torch

def project_patches_to_embeddings(flat_patches, patch_proj_weight, patch_proj_bias):
    # TODO: Linearly project flattened image patches into the ViT embedding dimension.
    return flat_patches @ patch_proj_weight.T + patch_proj_bias # (B, N, D)

# Step 5 - prepend_class_token
import torch

def prepend_class_token(patch_embeddings, class_token):
    """Prepend a learnable [CLS] token to the patch embedding sequence.

    patch_embeddings: (B, num_patches, embed_dim)
    class_token:      (1, 1, embed_dim)
    returns:          (B, num_patches+1, embed_dim)
    """
    # TODO: prepend the [CLS] token to every sequence in the batch
    B, N, D = patch_embeddings.shape
    return torch.cat((class_token.expand(B, -1, -1), patch_embeddings), dim=1)

# Step 6 - add_position_embeddings
import torch

def add_position_embeddings(tokens, position_embeddings):
    """Add learnable position embeddings to a (B, S, D) token sequence."""
    # TODO: combine tokens (B, S, D) with position_embeddings (1, S, D) via broadcasting.
    return tokens + position_embeddings

# Step 7 - compute_attention_scores
import torch

def compute_attention_scores(q, k):
    """Compute raw attention scores Q @ K^T.

    q: (..., Sq, d_head)
    k: (..., Sk, d_head)
    returns: (..., Sq, Sk)
    """
    # TODO: compute the raw attention scores as Q times K-transpose
    return torch.matmul(q, torch.transpose(k, -1, -2))

# Step 8 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_head):
    """Scale raw attention scores so softmax inputs stay well-conditioned."""
    # TODO: Divide raw attention scores by a constant derived from d_head.
    return scores / math.sqrt(d_head)

# Step 9 - apply_attention_mask
def apply_attention_mask(scores, mask):
    # TODO: add an additive mask (0 = allowed, -inf = blocked) to attention scores.
    if mask is None:
        return scores
    return scores + mask

# Step 10 - attention_softmax
import torch

def attention_softmax(masked_scores):
    """Softmax over the last (key) axis of attention scores."""
    # TODO: convert masked attention scores into normalized weights over the key axis
    masked_scores -= torch.max(masked_scores, dim=-1, keepdim=True).values
    masked_scores_exp = masked_scores.exp()
    return masked_scores_exp / masked_scores_exp.sum(dim=-1, keepdim=True)

# Step 11 - attention_context
import torch

def attention_context(attn_weights, v):
    """Combine attention weights with values to produce context vectors."""
    # TODO: return a tensor of shape (..., Sq, d_head) from attn_weights and v
    return attn_weights @ v

# Step 12 - scaled_dot_product_attention
import torch
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    """Compose score, scale, mask, softmax, and context into full attention."""
    # TODO: compose the five attention primitives into a single forward pass.
    d_head = k.shape[-1]
    scores = q @ torch.transpose(k, -1, -2) / math.sqrt(d_head)
    if mask is not None:
        scores += mask
    scores -= scores.max(dim=-1, keepdim=True).values
    scores_exp = scores.exp()
    attention_weights = scores_exp / scores_exp.sum(dim=-1, keepdim=True)
    return attention_weights @ v

# Step 13 - split_into_heads
import torch

def split_into_heads(x, num_heads):
    """Reshape (B, S, d_model) into (B, num_heads, S, d_head)."""
    # TODO: split the last dim into (num_heads, d_head) and move heads next to batch
    *B, S, d_model = x.shape
    return x.reshape(*B, S, num_heads, -1).transpose(-2, -3)

# Step 14 - merge_heads
import torch

def merge_heads(x):
    """Merge (B, num_heads, S, d_head) back to (B, S, num_heads*d_head)."""
    # TODO: merge the multi-head dimension back into the model dimension
    *B, num_heads, S, d_head = x.shape
    return x.transpose(-2, -3).reshape(*B, S, -1)

# Step 15 - project_qkv
def project_qkv(x, wq, bq, wk, bk, wv, bv):
    # TODO: project x into separate query, key, and value tensors using three linear layers.
    return x @ wq.T + bq, x @ wk.T + bk, x @ wv.T + bv

# Step 16 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: reshape q, k, v from (B, S, d_model) into (B, num_heads, S, d_head) each
    return split_into_heads(q, num_heads), split_into_heads(k, num_heads), split_into_heads(v, num_heads)

# Step 17 - multi_head_attention_scores
import torch

def multi_head_attention_scores(q_h, k_h, v_h, mask=None):
    """Run scaled dot-product attention in parallel across all heads.

    q_h, k_h, v_h: (B, num_heads, S, d_head)
    mask: broadcastable to (B, num_heads, S, S) or None
    returns: (B, num_heads, S, d_head)
    """
    # TODO: run scaled dot-product attention across the head axis
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 18 - merge_and_output_project
import torch

def merge_and_output_project(context_heads, wo, bo):
    """Merge heads back to d_model and apply the output projection."""
    # TODO: merge multi-head context to (B, S, d_model) then apply linear projection with wo, bo
    return merge_heads(context_heads) @ wo.transpose(-1, -2) + bo

# Step 19 - multi_head_self_attention
import torch

def multi_head_self_attention(x, params, num_heads, mask=None):
    """Run full multi-head self-attention: QKV proj, head split, attention, merge, output proj."""
    wq, bq = params['wq'], params['bq']
    wk, bk = params['wk'], params['bk']
    wv, bv = params['wv'], params['bv']
    wo, bo = params['wo'], params['bo']

    q, k, v = project_qkv(x, wq, bq, wk, bk, wv, bv)
    q_h, k_h, v_h = split_qkv_into_heads(q, k, v, num_heads)
    context_heads = multi_head_attention_scores(q_h, k_h, v_h, mask)
    x = merge_and_output_project(context_heads, wo, bo)

    return x

# Step 20 - gelu_activation
import torch
import math

def gelu_activation(x):
    """Apply the exact (erf-based) GELU activation elementwise to x."""
    # TODO: implement GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))

# Step 21 - mlp_first_layer
import torch

def mlp_first_layer(x, w1, b1):
    """Apply the first linear layer of the MLP block followed by GELU."""
    # TODO: project x to the feed-forward dimension and apply GELU
    return gelu_activation(linear_projection(x, w1, b1))

# Step 22 - mlp_second_layer
import torch

def mlp_second_layer(h, w2, b2):
    # TODO: project the MLP hidden activations back down to d_model using w2 and b2
    return linear_projection(h, w2, b2)

# Step 23 - mlp_block
import torch

def mlp_block(x, params):
    """Two-layer position-wise MLP with GELU between the layers."""
    # TODO: Assemble the position-wise two-layer MLP block with GELU between layers.
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']

    h = mlp_first_layer(x, w1, b1)
    return mlp_second_layer(h, w2, b2)

# Step 24 - compute_layernorm_stats
import torch

def compute_layernorm_stats(x, eps=1e-5):
    # TODO: return (mean, var) along the last dim, each with shape (..., 1).
    return x.mean(dim=-1, keepdim=True), x.var(dim=-1, keepdim=True, unbiased=False)

# Step 25 - layer_norm
import torch

def layer_norm(x, gamma, beta, eps=1e-5):
    # TODO: normalize the last dim of x and apply learnable scale gamma and shift beta
    mu, var = compute_layernorm_stats(x, eps)
    x = (x - mu) / torch.sqrt(var + eps)
    return gamma * x + beta

# Step 26 - residual_add
import torch

def residual_add(residual, sublayer_output):
    """Add residual skip connection to a sublayer's output."""
    # TODO: return the element-wise sum of residual and sublayer_output
    return residual + sublayer_output

# Step 27 - pre_norm_sublayer
import torch

def pre_norm_sublayer(x, gamma, beta, sublayer_fn):
    """Apply pre-norm: LN(x) -> sublayer -> add residual x."""
    # TODO: layer-normalize x, run sublayer_fn on it, then add the residual
    return residual_add(x, sublayer_fn(layer_norm(x, gamma, beta)))

# Step 28 - vision_encoder_block
import torch

def vision_encoder_block(x, block_params, num_heads):
    # TODO: pre-norm MHSA sublayer, then pre-norm MLP sublayer, both with residuals.
    gamma1 = block_params['ln1_gamma']
    beta1 = block_params['ln1_beta']
    gamma2 = block_params['ln2_gamma']
    beta2 = block_params['ln2_beta']
    y = pre_norm_sublayer(x, gamma1, beta1, lambda x: multi_head_self_attention(x, block_params['attn'], num_heads))
    z = pre_norm_sublayer(y, gamma2, beta2, lambda x: mlp_block(x, block_params['mlp']))
    return z

# Step 29 - vision_encoder
import torch

def vision_encoder(patch_sequence, encoder_params, num_heads):
    """Stack ViT encoder blocks then apply a final layer norm to the patch sequence."""
    # TODO: run patch_sequence through every block in encoder_params['blocks'], then final layer norm.
    x = patch_sequence
    for block_param in encoder_params['blocks']:
        x = vision_encoder_block(x, block_param, num_heads)
    
    final_gamma = encoder_params['final_ln_gamma']
    final_beta = encoder_params['final_ln_beta']

    return layer_norm(x, final_gamma, final_beta)

# Step 30 - extract_patch_features
import torch

def extract_patch_features(encoder_output):
    """Drop the [CLS] token from a ViT encoder output of shape (B, num_patches+1, d_model)."""
    # TODO: drop the class token and return only patch feature tokens
    return encoder_output[:, 1: , :]

# Step 31 - projector_first_layer
import torch

def projector_first_layer(patch_features, w1, b1):
    # TODO: apply the first projector linear layer followed by GELU
    return gelu_activation(patch_features @ w1 + b1)

# Step 32 - projector_second_layer
import torch

def projector_second_layer(hidden, w2, b2):
    """Map hidden activations (N, D_hidden) into the language space (N, D_lang)."""
    # TODO: apply the second linear layer of the projector (no activation).
    return hidden @ w2 + b2

# Step 33 - vision_language_projector
import torch

def vision_language_projector(patch_features, params):
    """Map (N, D_vision) patch features to (N, D_lang) image tokens."""
    # TODO: chain the two projector layers using params 'w1','b1','w2','b2'.
    return projector_second_layer(projector_first_layer(patch_features, params['w1'], params['b1']), params['w2'], params['b2'])

# Step 34 - build_token_vocabulary
def build_token_vocabulary(texts, image_token='<image>', pad_token='<pad>'):
    # TODO: Build a whitespace token-to-id vocabulary with pad at 0 and image token at 1.
    tokens_2_id = {pad_token: 0, image_token: 1}

    tokens = set()
    for text in texts:
        tokens.update(text.split(' '))

    tokens.discard(pad_token)
    tokens.discard(image_token)

    for i, token in enumerate(sorted(tokens), start=2):
        tokens_2_id[token] = i
    
    return tokens_2_id

# Step 35 - encode_text_to_ids
def encode_text_to_ids(text, vocab):
    # TODO: split text on whitespace and map each token to its vocab id
    ids = []
    for tok in text.split(' '):
        if tok in vocab:
            ids.append(vocab[tok])
    return ids

# Step 36 - embed_token_ids
import torch

def embed_token_ids(token_ids, embedding_matrix):
    """Look up embedding vectors for each token id.

    Args:
        token_ids: Long tensor of shape (T,) with values in [0, V).
        embedding_matrix: Tensor of shape (V, D_lang).

    Returns:
        Tensor of shape (T, D_lang).
    """
    # TODO: select the row of embedding_matrix for each token id
    return embedding_matrix[token_ids]

# Step 37 - add_text_position_embeddings
import torch

def add_text_position_embeddings(text_embeddings, position_embeddings):
    """Add learnable position embeddings to text token embeddings.

    text_embeddings: (T, D_lang)
    position_embeddings: (T_max, D_lang) with T_max >= T
    returns: (T, D_lang)
    """
    # TODO: add the first T rows of position_embeddings to text_embeddings
    return text_embeddings + position_embeddings[:text_embeddings.shape[0]]

# Step 38 - find_image_placeholder_positions
import torch

def find_image_placeholder_positions(token_ids, image_token_id):
    """Return a list of indices where token_ids == image_token_id."""
    # TODO: scan token_ids and return every position whose value equals image_token_id
    return torch.nonzero(token_ids == image_token_id).flatten().tolist()

# Step 39 - insert_image_tokens
import torch

def insert_image_tokens(text_embeddings, image_tokens, placeholder_position):
    """Splice image tokens into the text embedding sequence at the placeholder slot."""
    # TODO: replace text_embeddings[placeholder_position] with the N image_tokens rows
    return torch.cat((text_embeddings[:placeholder_position], image_tokens, text_embeddings[placeholder_position+1:]))

# Step 40 - build_multimodal_embeddings
import torch

def build_multimodal_embeddings(token_ids, image_tokens, embedding_matrix, position_embeddings, image_token_id):
    # TODO: build fused multimodal embeddings by embedding text, adding positions, and splicing image tokens.
    text_embedding = embed_token_ids(token_ids, embedding_matrix)
    text_embedding = add_text_position_embeddings(text_embedding, position_embeddings)

    placeholder_position = find_image_placeholder_positions(token_ids, image_token_id)
    multimodal_embedding = insert_image_tokens(text_embedding, image_tokens, placeholder_position[0])

    return multimodal_embedding

# Step 41 - build_label_tensor
import torch

def build_label_tensor(token_ids, image_token_id, pad_token_id, num_image_tokens, ignore_index=-100):
    """Build the label tensor aligned to the fused multimodal sequence."""
    # TODO: expand image placeholders, mask image and pad positions with ignore_index
    label_tensors = []
    for token_id in token_ids.tolist():
        if token_id == pad_token_id:
            label_tensors.append(ignore_index)
        elif token_id == image_token_id:
            label_tensors.extend([ignore_index] * num_image_tokens)
        else:
            label_tensors.append(token_id)
    return torch.tensor(label_tensors)

# Step 42 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (seq_len, seq_len) additive causal mask: 0 on/under diag, -inf above."""
    # TODO: build a lower-triangular additive mask with 0 allowed and -inf blocked
    mask = torch.ones(seq_len, seq_len) * (-float('inf')) 
    return torch.triu(mask, diagonal=1)

# Step 43 - decoder_block
def decoder_block(x, params, causal_mask):
    # TODO: run a pre-norm masked self-attention sublayer then a pre-norm MLP sublayer over x.
    x = pre_norm_sublayer(x, params['ln1']['gamma'], params['ln1']['beta'], lambda x: multi_head_self_attention(x, params['attn'], params['num_heads'], causal_mask))
    x = pre_norm_sublayer(x, params['ln2']['gamma'], params['ln2']['beta'], lambda x: mlp_block(x, params['mlp']))
    return x

# Step 44 - language_model_decoder
import torch

def language_model_decoder(x, blocks_params, causal_mask):
    # TODO: apply every decoder block in blocks_params sequentially to x and return the result
    for block_params in blocks_params:
        x = decoder_block(x, block_params, causal_mask)
    
    return x

# Step 45 - final_layer_norm
import torch

def final_layer_norm(x, gamma, beta):
    # TODO: apply the existing layer_norm primitive to x using gamma and beta.
    return layer_norm(x, gamma, beta)

# Step 46 - language_model_head
def language_model_head(x, w_out, b_out):
    # TODO: project hidden states (L, D) to vocabulary logits (L, V) using w_out and b_out
    return x @ w_out + b_out

# Step 47 - encode_image_to_tokens
# Step 47 - encode_image_to_tokens (not yet solved)
def initialize_vlm_parameters(config, seed=0):
    torch.manual_seed(seed)
    
    # 1. Simplify config extraction with a quick helper
    def get(*keys, default=None):
        for k in keys:
            if k in config:
                return config[k]
        return default

    # 2. Extract hyperparameters
    d_vision = get('d_vision')
    d_lang = get('d_lang', 'd_text')
    img_size = get('image_size')
    p_size = get('patch_size')
    in_c = get('in_channels', default=3)
    num_patches = get('num_patches', default=(img_size // p_size)**2)
    
    v_heads = get('num_vision_heads', 'n_heads', 'n_vision_heads')
    l_heads = get('num_decoder_heads', 'n_heads', 'n_decoder_heads')
    v_layers = get('num_vision_layers', 'n_layers_vision', 'n_vision_layers')
    l_layers = get('num_decoder_layers', 'n_layers_decoder', 'n_decoder_layers')
    
    v_mlp = get('mlp_hidden_vision', default=d_vision * 4)
    l_mlp = get('mlp_hidden_text', default=d_lang * 4)
    
    vocab_size = get('vocab_size')
    max_seq_len = get('max_seq_len', 'max_text_len')

    # 3. Helpers to instantiate standard parameter dictionaries
    def param(*shape):
        # Create a leaf tensor with requires_grad=True
        return torch.randn(*shape, requires_grad=True)

    # def mk_ln(dim):
    #     return {'gamma': param(dim), 'beta': param(dim)}

    # (d_out, d_in) convention for mlp
    def mk_mlp(d_in, d_hid, d_out):
        return {
            'w1': param(d_hid, d_in), 'b1': param(d_hid),
            'w2': param(d_out, d_hid), 'b2': param(d_out)
        }

    def mk_attn(dim):
        return {
            'wq': param(dim, dim),
            'wk': param(dim, dim),
            'wv': param(dim, dim),
            'wo': param(dim, dim),
            'bq': param(dim),
            'bk': param(dim),
            'bv': param(dim),
            'bo': param(dim),
        }

    # 4. Assemble Encoder and Decoder Blocks
    vision_blocks = [{
        'num_heads': v_heads, # Kept as an integer, the test explicitly looks for this!
        'ln1_gamma': param(d_vision),
        'ln1_beta': param(d_vision),
        'attn': mk_attn(d_vision),
        'ln2_gamma': param(d_vision),
        'ln2_beta': param(d_vision),
        'mlp': mk_mlp(d_vision, v_mlp, d_vision),
    } for _ in range(v_layers)]

    decoder_blocks = [{
        'num_heads': l_heads,
        'ln1_gamma': param(d_lang),
        'ln1_beta': param(d_lang),
        'attn': mk_attn(d_lang),
        'ln2_gamma': param(d_lang),
        'ln2_beta': param(d_lang),
        'mlp': mk_mlp(d_lang, l_mlp, d_lang),
    } for _ in range(l_layers)]

    # 5. Return the top-level dictionary matching consumer expectations
    return {
        'vision': {
            "num_heads": v_heads,
            # (d_out, d_in) convention
            'patch_proj': {
                'w': param(d_vision, in_c * p_size * p_size),
                'b': param(d_vision)
            },
            'cls_token': param(1, d_vision),
            'pos_embedding': param(num_patches + 1, d_vision),
            'blocks': vision_blocks,
            'final_ln_gamma': param(d_vision),
            'final_ln_beta': param(d_vision),
            'patch_size': p_size,
        },
        # (d_in, d_out) notation
        'projector': {
            'w1': param(d_vision, d_lang), 'b1': param(d_lang),
            'w2': param(d_lang, d_lang),   'b2': param(d_lang)
        },
        'embedding': param(vocab_size, d_lang),
        'pos_embedding': param(max_seq_len, d_lang),
        'decoder_blocks': decoder_blocks,
        'final_ln_gamma': param(d_lang),
        'final_ln_beta': param(d_lang),
        'lm_head': {
            'w_out': param(d_lang, vocab_size), # test expects [d_lang, vocab_size]
            'b_out': param(vocab_size)
        },
        'image_token_id': get('image_token_id', default=1),
        'num_image_tokens': get('num_image_tokens', default=4)
    }

def encode_image_to_tokens(image, vision_params, projector_params):
    # TODO: run the vision encoder, drop the class token, and apply the projector.
    image = image.unsqueeze(0)
    image_patches = split_image_into_patches(image, vision_params['patch_size'])
    flattened_patches = flatten_patches(image_patches)
    projected_patches = project_patches_to_embeddings(flattened_patches, vision_params['patch_proj']['w'], vision_params['patch_proj']['b'])
    projected_patches = prepend_class_token(projected_patches, vision_params['cls_token'])
    vision_embedding = vision_encoder(projected_patches, vision_params, vision_params['num_heads']) # (B, N, D_vision)
    # drop class token
    return vision_language_projector(vision_embedding[:, 1:, :], projector_params).squeeze(0)

# Step 48 - vision_language_forward
def vision_language_forward(image, token_ids, params):
    # TODO: route image + token_ids through the full vision-language model and return (L, V) logits.
    image_tokens = encode_image_to_tokens(image, params['vision'], params['projector'])
    multimodal_embedding = build_multimodal_embeddings(token_ids, image_tokens, params['embedding'], params['pos_embedding'], params['image_token_id'])
    causal_mask = build_causal_mask(multimodal_embedding.shape[0])
    x = language_model_decoder(multimodal_embedding, params['decoder_blocks'], causal_mask)

    if params.get('final_ln_gamma') is not None and params.get('final_ln_beta') is not None:
        x = final_layer_norm(x, params['final_ln_gamma'], params['final_ln_beta'])
    elif params.get('final_ln') is not None:
        x = final_layer_norm(x, params['final_ln']['gamma'], params['final_ln']['beta'])

    return language_model_head(x, params['lm_head']['w_out'], params['lm_head']['b_out'])

# Step 49 - shift_logits_and_labels
import torch

def shift_logits_and_labels(logits, labels):
    # TODO: align each logit with the next-position label and return (shifted_logits, shifted_labels).
    return logits[:-1], labels[1:]

# Step 50 - per_position_cross_entropy
import torch

def per_position_cross_entropy(shifted_logits, shifted_labels, ignore_index=-100):
    """Per-position next-token cross-entropy with 0 at ignored positions."""
    log_prob = torch.log_softmax(shifted_logits, dim=-1)
    ignored_mask = (shifted_labels == ignore_index)
    safe_labels = shifted_labels.masked_fill(ignored_mask, 0)
    seq_len = shifted_logits.shape[0]
    per_item_loss = -log_prob[torch.arange(seq_len), safe_labels]
    per_item_loss.masked_fill_(ignored_mask, 0)
    return per_item_loss

# Step 51 - masked_mean_loss
import torch

def masked_mean_loss(per_position_losses, shifted_labels, ignore_index=-100):
    """Average per-position losses over positions whose label != ignore_index."""
    # TODO: average per_position_losses over positions where shifted_labels != ignore_index
    n = (shifted_labels != ignore_index).sum()
    if n == 0:
        return per_position_losses.sum()
    return per_position_losses.sum() / n

# Step 52 - greedy_next_token
def greedy_next_token(logits):
    # TODO: return the int token id with the highest logit at the final position
    return logits[-1].argmax().item()

# Step 53 - apply_temperature
import torch

def apply_temperature(logits, temperature):
    """Scale logits by dividing by temperature."""
    # TODO: return a tensor of logits rescaled by the temperature value
    return logits / temperature

# Step 54 - top_k_filter
import torch

def top_k_filter(logits, k):
    """Keep only the top-k logits; set all others to -inf."""
    # no filtering
    if k == 0 or k > logits.shape[0]:
        return logits
        
    vals, indices = logits.topk(k, dim=-1)
    
    filtered_logits = torch.full_like(logits, float('-inf'))

    filtered_logits[indices] = vals

    return filtered_logits

# Step 55 - sample_from_logits
import torch

def sample_from_logits(logits):
    """Sample a token id from softmax(logits).

    Args:
        logits: 1D tensor of shape (V,)
    Returns:
        int token id
    """
    # TODO: turn logits into a categorical distribution and draw one token id
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1).item()

# Step 56 - generate_caption (not yet solved)
# TODO: implement

# Step 57 - initialize_vlm_parameters
import torch

def initialize_vlm_parameters(config, seed=0):
    torch.manual_seed(seed)
    
    # 1. Simplify config extraction with a quick helper
    def get(*keys, default=None):
        for k in keys:
            if k in config:
                return config[k]
        return default

    # 2. Extract hyperparameters
    d_vision = get('d_vision')
    d_lang = get('d_lang', 'd_text')
    img_size = get('image_size')
    p_size = get('patch_size')
    in_c = get('in_channels', default=3)
    num_patches = get('num_patches', default=(img_size // p_size)**2)
    
    v_heads = get('num_vision_heads', 'n_heads', 'n_vision_heads')
    l_heads = get('num_decoder_heads', 'n_heads', 'n_decoder_heads')
    v_layers = get('num_vision_layers', 'n_layers_vision', 'n_vision_layers')
    l_layers = get('num_decoder_layers', 'n_layers_decoder', 'n_decoder_layers')
    
    v_mlp = get('mlp_hidden_vision', default=d_vision * 4)
    l_mlp = get('mlp_hidden_text', default=d_lang * 4)
    
    vocab_size = get('vocab_size')
    max_seq_len = get('max_seq_len', 'max_text_len')

    # 3. Helpers to instantiate standard parameter dictionaries
    def param(*shape):
        # Create a leaf tensor with requires_grad=True
        return torch.randn(*shape, requires_grad=True)

    def mk_ln(dim):
        return {'gamma': param(dim), 'beta': param(dim)}

    def mk_mlp(d_in, d_hid, d_out):
        return {
            'w1': param(d_in, d_hid), 'b1': param(d_hid),
            'w2': param(d_hid, d_out), 'b2': param(d_out)
        }

    def mk_attn(dim):
        return {
            'w_qkv': param(dim, 3 * dim), 'b_qkv': param(3 * dim),
            'w_out': param(dim, dim),     'b_out': param(dim)
        }

    # 4. Assemble Encoder and Decoder Blocks
    vision_blocks = [{
        'num_heads': v_heads, # Kept as an integer, the test explicitly looks for this!
        'attn': mk_attn(d_vision),
        'ln_1': mk_ln(d_vision),
        'mlp': mk_mlp(d_vision, v_mlp, d_vision),
        'ln_2': mk_ln(d_vision)
    } for _ in range(v_layers)]

    decoder_blocks = [{
        'num_heads': l_heads,
        'attn': mk_attn(d_lang),
        'ln_1': mk_ln(d_lang),
        'mlp': mk_mlp(d_lang, l_mlp, d_lang),
        'ln_2': mk_ln(d_lang)
    } for _ in range(l_layers)]

    # 5. Return the top-level dictionary matching consumer expectations
    return {
        'vision': {
            'patch_proj': {
                'w': param(in_c * p_size * p_size, d_vision),
                'b': param(d_vision)
            },
            'cls_token': param(1, d_vision),
            'pos_embedding': param(num_patches + 1, d_vision),
            'blocks': vision_blocks,
            'ln_post': mk_ln(d_vision)
        },
        'projector': {
            'w1': param(d_vision, d_lang), 'b1': param(d_lang),
            'w2': param(d_lang, d_lang),   'b2': param(d_lang)
        },
        'embedding': param(vocab_size, d_lang),
        'pos_embedding': param(max_seq_len, d_lang),
        'decoder_blocks': decoder_blocks,
        'final_ln': mk_ln(d_lang),
        'lm_head': {
            'w_out': param(d_lang, vocab_size), # test expects [d_lang, vocab_size]
            'b_out': param(vocab_size)
        },
        'image_token_id': get('image_token_id', default=1),
        'num_image_tokens': get('num_image_tokens', default=4)
    }

# Step 58 - collect_parameters (not yet solved)
# TODO: implement

# Step 59 - zero_gradients (not yet solved)
# TODO: implement

# Step 60 - training_step (not yet solved)
# TODO: implement

# Step 61 - apply_gradient_update (not yet solved)
# TODO: implement

# Step 62 - run_training_loop (not yet solved)
# TODO: implement

