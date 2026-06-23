"""Vision-Language Model implemented from raw tensor ops (educational toy track).

Every component — ViT image encoder, vision->language projector, causal text decoder,
multimodal fusion, manual-autograd training loop, and sampling — is built here from
scratch with no nn.Module, to demonstrate the internals end to end. It runs on CPU in
seconds on a tiny toy config (see demo.py).

This is the toy counterpart to the real-track package in `vlm/`, which realizes the same
projector design as a trainable nn.Module and trains it LLaVA-style on real data.
"""

import math
import torch


def split_image_into_patches(image, patch_size):
    B, C, H, W = image.shape
    P = patch_size
    x = image.reshape(B, C, H // P, P, W // P, P)
    x = x.permute(0, 2, 4, 1, 3, 5) # (B, H//P, W//P, C, P, P)
    return x.reshape(B, (H // P) * (W // P), C, P, P)

def flatten_patches(patches):
    B, N, C, _, _ = patches.shape
    return torch.reshape(patches, (B, N, -1))

def linear_projection(x, weight, bias):
    """Apply y = x @ weight.T + bias with arbitrary leading dims on x."""
    return x @ weight.T + bias

def project_patches_to_embeddings(flat_patches, patch_proj_weight, patch_proj_bias):
    return flat_patches @ patch_proj_weight.T + patch_proj_bias # (B, N, D)

def prepend_class_token(patch_embeddings, class_token):
    """Prepend a learnable [CLS] token to the patch embedding sequence.

    patch_embeddings: (B, num_patches, embed_dim)
    class_token:      (1, 1, embed_dim)
    returns:          (B, num_patches+1, embed_dim)
    """
    B, N, D = patch_embeddings.shape
    return torch.cat((class_token.expand(B, -1, -1), patch_embeddings), dim=1)

def add_position_embeddings(tokens, position_embeddings):
    """Add learnable position embeddings to a (B, S, D) token sequence."""
    return tokens + position_embeddings

def compute_attention_scores(q, k):
    """Compute raw attention scores Q @ K^T.

    q: (..., Sq, d_head)
    k: (..., Sk, d_head)
    returns: (..., Sq, Sk)
    """
    return torch.matmul(q, torch.transpose(k, -1, -2))

def scale_attention_scores(scores, d_head):
    """Scale raw attention scores so softmax inputs stay well-conditioned."""
    return scores / math.sqrt(d_head)

def apply_attention_mask(scores, mask):
    if mask is None:
        return scores
    return scores + mask

def attention_softmax(masked_scores):
    """Softmax over the last (key) axis of attention scores."""
    masked_scores -= torch.max(masked_scores, dim=-1, keepdim=True).values
    masked_scores_exp = masked_scores.exp()
    return masked_scores_exp / masked_scores_exp.sum(dim=-1, keepdim=True)

def attention_context(attn_weights, v):
    """Combine attention weights with values to produce context vectors."""
    return attn_weights @ v

def scaled_dot_product_attention(q, k, v, mask=None):
    """Compose score, scale, mask, softmax, and context into full attention."""
    d_head = k.shape[-1]
    scores = q @ torch.transpose(k, -1, -2) / math.sqrt(d_head)
    if mask is not None:
        scores += mask
    scores -= scores.max(dim=-1, keepdim=True).values
    scores_exp = scores.exp()
    attention_weights = scores_exp / scores_exp.sum(dim=-1, keepdim=True)
    return attention_weights @ v

def split_into_heads(x, num_heads):
    """Reshape (B, S, d_model) into (B, num_heads, S, d_head)."""
    *B, S, d_model = x.shape
    return x.reshape(*B, S, num_heads, -1).transpose(-2, -3)

def merge_heads(x):
    """Merge (B, num_heads, S, d_head) back to (B, S, num_heads*d_head)."""
    *B, num_heads, S, d_head = x.shape
    return x.transpose(-2, -3).reshape(*B, S, -1)

def project_qkv(x, wq, bq, wk, bk, wv, bv):
    return x @ wq.T + bq, x @ wk.T + bk, x @ wv.T + bv

def split_qkv_into_heads(q, k, v, num_heads):
    return split_into_heads(q, num_heads), split_into_heads(k, num_heads), split_into_heads(v, num_heads)

def multi_head_attention_scores(q_h, k_h, v_h, mask=None):
    """Run scaled dot-product attention in parallel across all heads.

    q_h, k_h, v_h: (B, num_heads, S, d_head)
    mask: broadcastable to (B, num_heads, S, S) or None
    returns: (B, num_heads, S, d_head)
    """
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

def merge_and_output_project(context_heads, wo, bo):
    """Merge heads back to d_model and apply the output projection."""
    return merge_heads(context_heads) @ wo.transpose(-1, -2) + bo

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

def gelu_activation(x):
    """Apply the exact (erf-based) GELU activation elementwise to x."""
    return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))

def mlp_first_layer(x, w1, b1):
    """Apply the first linear layer of the MLP block followed by GELU."""
    return gelu_activation(linear_projection(x, w1, b1))

def mlp_second_layer(h, w2, b2):
    return linear_projection(h, w2, b2)

def mlp_block(x, params):
    """Two-layer position-wise MLP with GELU between the layers."""
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']

    h = mlp_first_layer(x, w1, b1)
    return mlp_second_layer(h, w2, b2)

def compute_layernorm_stats(x, eps=1e-5):
    return x.mean(dim=-1, keepdim=True), x.var(dim=-1, keepdim=True, unbiased=False)

def layer_norm(x, gamma, beta, eps=1e-5):
    mu, var = compute_layernorm_stats(x, eps)
    x = (x - mu) / torch.sqrt(var + eps)
    return gamma * x + beta

def residual_add(residual, sublayer_output):
    """Add residual skip connection to a sublayer's output."""
    return residual + sublayer_output

def pre_norm_sublayer(x, gamma, beta, sublayer_fn):
    """Apply pre-norm: LN(x) -> sublayer -> add residual x."""
    return residual_add(x, sublayer_fn(layer_norm(x, gamma, beta)))

def vision_encoder_block(x, block_params, num_heads):
    gamma1 = block_params['ln1_gamma']
    beta1 = block_params['ln1_beta']
    gamma2 = block_params['ln2_gamma']
    beta2 = block_params['ln2_beta']
    y = pre_norm_sublayer(x, gamma1, beta1, lambda x: multi_head_self_attention(x, block_params['attn'], num_heads))
    z = pre_norm_sublayer(y, gamma2, beta2, lambda x: mlp_block(x, block_params['mlp']))
    return z

def vision_encoder(patch_sequence, encoder_params, num_heads):
    """Stack ViT encoder blocks then apply a final layer norm to the patch sequence."""
    x = patch_sequence
    for block_param in encoder_params['blocks']:
        x = vision_encoder_block(x, block_param, num_heads)

    final_gamma = encoder_params['final_ln_gamma']
    final_beta = encoder_params['final_ln_beta']

    return layer_norm(x, final_gamma, final_beta)

def extract_patch_features(encoder_output):
    """Drop the [CLS] token from a ViT encoder output of shape (B, num_patches+1, d_model)."""
    return encoder_output[:, 1: , :]

def projector_first_layer(patch_features, w1, b1):
    return gelu_activation(patch_features @ w1 + b1)

def projector_second_layer(hidden, w2, b2):
    """Map hidden activations (N, D_hidden) into the language space (N, D_lang)."""
    return hidden @ w2 + b2

def vision_language_projector(patch_features, params):
    """Map (N, D_vision) patch features to (N, D_lang) image tokens."""
    return projector_second_layer(projector_first_layer(patch_features, params['w1'], params['b1']), params['w2'], params['b2'])

def build_token_vocabulary(texts, image_token='<image>', pad_token='<pad>'):
    """Build a whitespace token-to-id vocabulary with pad at 0 and image token at 1."""
    tokens_2_id = {pad_token: 0, image_token: 1}

    tokens = set()
    for text in texts:
        tokens.update(text.split(' '))

    tokens.discard(pad_token)
    tokens.discard(image_token)

    for i, token in enumerate(sorted(tokens), start=2):
        tokens_2_id[token] = i

    return tokens_2_id

def encode_text_to_ids(text, vocab):
    """Split text on whitespace and map each known token to its vocab id."""
    ids = []
    for tok in text.split(' '):
        if tok in vocab:
            ids.append(vocab[tok])
    return ids

def embed_token_ids(token_ids, embedding_matrix):
    """Look up embedding vectors for each token id.

    Args:
        token_ids: Long tensor of shape (T,) with values in [0, V).
        embedding_matrix: Tensor of shape (V, D_lang).

    Returns:
        Tensor of shape (T, D_lang).
    """
    return embedding_matrix[token_ids]

def add_text_position_embeddings(text_embeddings, position_embeddings):
    """Add learnable position embeddings to text token embeddings.

    text_embeddings: (T, D_lang)
    position_embeddings: (T_max, D_lang) with T_max >= T
    returns: (T, D_lang)
    """
    return text_embeddings + position_embeddings[:text_embeddings.shape[0]]

def find_image_placeholder_positions(token_ids, image_token_id):
    """Return a list of indices where token_ids == image_token_id."""
    return torch.nonzero(token_ids == image_token_id).flatten().tolist()

def insert_image_tokens(text_embeddings, image_tokens, placeholder_position):
    """Splice image tokens into the text embedding sequence at the placeholder slot."""
    return torch.cat((text_embeddings[:placeholder_position], image_tokens, text_embeddings[placeholder_position+1:]))

def build_multimodal_embeddings(token_ids, image_tokens, embedding_matrix, position_embeddings, image_token_id):
    """Build fused multimodal embeddings: embed text, add positions, splice image tokens."""
    text_embedding = embed_token_ids(token_ids, embedding_matrix)
    text_embedding = add_text_position_embeddings(text_embedding, position_embeddings)

    placeholder_position = find_image_placeholder_positions(token_ids, image_token_id)
    multimodal_embedding = insert_image_tokens(text_embedding, image_tokens, placeholder_position[0])

    return multimodal_embedding

def build_label_tensor(token_ids, image_token_id, pad_token_id, num_image_tokens, ignore_index=-100):
    """Build the label tensor aligned to the fused multimodal sequence.

    Expands image placeholders and masks image and pad positions with ignore_index.
    """
    label_tensors = []
    for token_id in token_ids.tolist():
        if token_id == pad_token_id:
            label_tensors.append(ignore_index)
        elif token_id == image_token_id:
            label_tensors.extend([ignore_index] * num_image_tokens)
        else:
            label_tensors.append(token_id)
    return torch.tensor(label_tensors)

def build_causal_mask(seq_len):
    """Return a (seq_len, seq_len) additive causal mask: 0 on/under diag, -inf above."""
    mask = torch.ones(seq_len, seq_len) * (-float('inf'))
    return torch.triu(mask, diagonal=1)

def decoder_block(x, params, causal_mask):
    """Pre-norm masked self-attention sublayer then a pre-norm MLP sublayer."""
    x = pre_norm_sublayer(x, params['ln1_gamma'], params['ln1_beta'], lambda x: multi_head_self_attention(x, params['attn'], params['num_heads'], causal_mask))
    x = pre_norm_sublayer(x, params['ln2_gamma'], params['ln2_beta'], lambda x: mlp_block(x, params['mlp']))
    return x

def language_model_decoder(x, blocks_params, causal_mask):
    """Apply every decoder block sequentially to x."""
    for block_params in blocks_params:
        x = decoder_block(x, block_params, causal_mask)

    return x

def final_layer_norm(x, gamma, beta):
    return layer_norm(x, gamma, beta)

def language_model_head(x, w_out, b_out):
    """Project hidden states (L, D) to vocabulary logits (L, V)."""
    return x @ w_out + b_out

# Parameter initialization
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
    """Run the vision encoder, drop the class token, and apply the projector."""
    image = image.unsqueeze(0)
    image_patches = split_image_into_patches(image, vision_params['patch_size'])
    flattened_patches = flatten_patches(image_patches)
    projected_patches = project_patches_to_embeddings(flattened_patches, vision_params['patch_proj']['w'], vision_params['patch_proj']['b'])
    projected_patches = prepend_class_token(projected_patches, vision_params['cls_token'])
    vision_embedding = vision_encoder(projected_patches, vision_params, vision_params['num_heads']) # (B, N, D_vision)
    # drop class token
    return vision_language_projector(vision_embedding[:, 1:, :], projector_params).squeeze(0)

def vision_language_forward(image, token_ids, params):
    """Route image + token_ids through the full vision-language model -> (L, V) logits."""
    image_tokens = encode_image_to_tokens(image, params['vision'], params['projector'])
    multimodal_embedding = build_multimodal_embeddings(token_ids, image_tokens, params['embedding'], params['pos_embedding'], params['image_token_id'])
    causal_mask = build_causal_mask(multimodal_embedding.shape[0])
    x = language_model_decoder(multimodal_embedding, params['decoder_blocks'], causal_mask)

    if params.get('final_ln_gamma') is not None and params.get('final_ln_beta') is not None:
        x = final_layer_norm(x, params['final_ln_gamma'], params['final_ln_beta'])
    elif params.get('final_ln') is not None:
        x = final_layer_norm(x, params['final_ln']['gamma'], params['final_ln']['beta'])

    return language_model_head(x, params['lm_head']['w_out'], params['lm_head']['b_out'])

def shift_logits_and_labels(logits, labels):
    """Align each logit with the next-position label."""
    return logits[:-1], labels[1:]

def per_position_cross_entropy(shifted_logits, shifted_labels, ignore_index=-100):
    """Per-position next-token cross-entropy with 0 at ignored positions."""
    log_prob = torch.log_softmax(shifted_logits, dim=-1)
    ignored_mask = (shifted_labels == ignore_index)
    safe_labels = shifted_labels.masked_fill(ignored_mask, 0)
    seq_len = shifted_logits.shape[0]
    per_item_loss = -log_prob[torch.arange(seq_len), safe_labels]
    per_item_loss.masked_fill_(ignored_mask, 0)
    return per_item_loss

def masked_mean_loss(per_position_losses, shifted_labels, ignore_index=-100):
    """Average per-position losses over positions whose label != ignore_index."""
    n = (shifted_labels != ignore_index).sum()
    if n == 0:
        return per_position_losses.sum()
    return per_position_losses.sum() / n

def greedy_next_token(logits):
    """Return the int token id with the highest logit at the final position."""
    return logits[-1].argmax().item()

def apply_temperature(logits, temperature):
    """Scale logits by dividing by temperature."""
    return logits / temperature

def top_k_filter(logits, k):
    """Keep only the top-k logits; set all others to -inf."""
    # no filtering
    if k == 0 or k > logits.shape[0]:
        return logits

    vals, indices = logits.topk(k, dim=-1)

    filtered_logits = torch.full_like(logits, float('-inf'))

    filtered_logits.scatter_(-1, indices, vals)

    return filtered_logits

def sample_from_logits(logits):
    """Sample a token id from softmax(logits).

    Args:
        logits: 1D tensor of shape (V,)
    Returns:
        int token id
    """
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1).item()

def generate_caption(image, prompt_ids, params, max_new_tokens, temperature=1.0, top_k=0, do_sample=False):

    for i in range(max_new_tokens):
        logits = vision_language_forward(image, prompt_ids, params)
        logits = apply_temperature(logits, temperature)

        # Only apply the filter if top_k is greater than 0.
        if top_k > 0:
            logits = top_k_filter(logits, top_k)

        if do_sample:
            next_token_id = sample_from_logits(logits[-1])
        else:
            next_token_id = greedy_next_token(logits)

        # Create the new tensor on the same device as the prompt.
        new_token_tensor = torch.tensor([next_token_id], device=prompt_ids.device)
        prompt_ids = torch.cat([prompt_ids, new_token_tensor])

    return prompt_ids.tolist()

def collect_parameters(params):
    collected_params = []

    # If it's a dictionary, recurse into its values
    if isinstance(params, dict):
        for value in params.values():
            collected_params.extend(collect_parameters(value))

    # If it's a list or tuple, recurse into its items
    elif isinstance(params, (list, tuple)):
        for item in params:
            collected_params.extend(collect_parameters(item))

    # If it's a tensor, check if it requires gradients
    elif isinstance(params, torch.Tensor):
        if params.requires_grad:
            collected_params.append(params)

    # Non-tensor scalars (ints/floats) are naturally ignored.
    return collected_params

def zero_gradients(parameter_list):
    """Resets the .grad attribute of every parameter tensor to zero in place."""
    for param in parameter_list:
        # Only zero out the gradient if it has already been allocated
        if param.grad is not None:
            param.grad.zero_()

def training_step(image, token_ids, labels, params, parameter_list, learning_rate):
    """Run one optimization step and return the detached scalar loss tensor."""
    zero_gradients(parameter_list)

    logits = vision_language_forward(image, token_ids, params)
    shifted_logits, shifted_labels = shift_logits_and_labels(logits, labels)
    losses = per_position_cross_entropy(shifted_logits, shifted_labels)
    loss = masked_mean_loss(losses, shifted_labels)
    loss.backward()

    apply_gradient_update(parameter_list, learning_rate)
    return loss.detach()

def apply_gradient_update(parameters, learning_rate):
    """Apply an in-place SGD update to parameters with gradients."""
    with torch.no_grad():
        for param in parameters:
            if param.grad is not None:
                param.sub_(learning_rate * param.grad)
    return parameters

def run_training_loop(params, batch, num_steps, learning_rate):
    parameter_list = collect_parameters(params)
    losses = []

    for _ in range(num_steps):
        loss = training_step(
            batch['image'],
            batch['token_ids'],
            batch['labels'],
            params,
            parameter_list,
            learning_rate,
        )
        losses.append(float(loss))

    return losses


def _smoke_test():
    torch.manual_seed(0)
    config = {
        'image_size': 8,
        'patch_size': 4,
        'num_patches': 4,
        'in_channels': 3,
        'd_vision': 8,
        'd_text': 8,
        'num_vision_heads': 2,
        'num_decoder_heads': 2,
        'num_vision_layers': 1,
        'num_decoder_layers': 1,
        'mlp_hidden_vision': 16,
        'mlp_hidden_text': 16,
        'vocab_size': 5,
        'max_text_len': 8,
        'num_image_tokens': 4,
    }
    params = initialize_vlm_parameters(config, seed=0)
    parameter_list = collect_parameters(params)
    image = torch.randn(3, 8, 8)
    token_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    labels = torch.tensor([-100, -100, -100, -100, 2, 3], dtype=torch.long)
    loss = training_step(image, token_ids, labels, params, parameter_list, 0.01)
    print(isinstance(loss, torch.Tensor), loss.dim())


if __name__ == '__main__':
    _smoke_test()
