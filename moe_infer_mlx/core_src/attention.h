#ifndef ATTENTION_H
#define ATTENTION_H

// ============================================================================
// KV Cache for full attention layers
//
// With USE_KV_CACHE_BF16: stored as bfloat16 (uint16_t) to halve memory bandwidth.
// Types (KVCache, LinearAttnState) are in model_types.h.
// ============================================================================

#include "model_internal.h"

#if USE_KV_CACHE_BF16
typedef uint16_t kv_elem_t;
#define KV_ELEM_SIZE sizeof(uint16_t)
static inline float kv_elem_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}
static inline uint16_t f32_to_kv_elem(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(uint32_t));
    return (uint16_t)(bits >> 16);
}
#else
typedef float kv_elem_t;
#define KV_ELEM_SIZE sizeof(float)
static inline float kv_elem_to_f32(float v) { return v; }
static inline float f32_to_kv_elem(float v) { return v; }
#endif

static KVCache *kv_cache_new(FlashMoE_Model *m) {
    KVCache *c = calloc(1, sizeof(KVCache));
    c->k_cache = calloc(MAX_SEQ_LEN * m->cfg.num_kv_heads * HEAD_DIM, KV_ELEM_SIZE);
    c->v_cache = calloc(MAX_SEQ_LEN * m->cfg.num_kv_heads * HEAD_DIM, KV_ELEM_SIZE);
    c->len = 0;
    return c;
}

static void kv_cache_free(KVCache *c) {
    if (c) {
        free(c->k_cache);
        free(c->v_cache);
        free(c);
    }
}

// ============================================================================
// Linear attention state (GatedDeltaNet recurrent state)
// ============================================================================

static LinearAttnState *linear_attn_state_new(FlashMoE_Model *m) {
    LinearAttnState *s = calloc(1, sizeof(LinearAttnState));
    s->conv_state = calloc((CONV_KERNEL_SIZE - 1) * m->cfg.linear_conv_dim, sizeof(float));
    s->ssm_state = calloc(m->cfg.linear_num_v_heads * LINEAR_VALUE_DIM * LINEAR_KEY_DIM, sizeof(float));
    return s;
}

static void linear_attn_state_free(LinearAttnState *s) {
    if (s) {
        free(s->conv_state);
        free(s->ssm_state);
        free(s);
    }
}

// ============================================================================
// Full attention layer forward (single token, incremental)
// ============================================================================

static float vec_rms(const float *v, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += v[i] * v[i];
    return sqrtf(sum / n);
}

__attribute__((unused))
static void full_attention_forward(
    FlashMoE_Model *m,
    WeightFile *wf,
    int layer_idx,
    float *hidden,       // [m->cfg.hidden_dim] in/out
    KVCache *kv,
    int pos              // position in sequence
) {
    m->fa_debug_count++;
    int do_debug = 0;

    char name[256];
    float *normed = malloc(m->cfg.hidden_dim * sizeof(float));
    float *residual = malloc(m->cfg.hidden_dim * sizeof(float));
    cpu_vec_copy(residual, hidden, m->cfg.hidden_dim);

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] layer=%d pos=%d hidden_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, pos, vec_rms(hidden, m->cfg.hidden_dim),
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    // ---- Input LayerNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer_idx);
    uint16_t *norm_w = get_tensor_ptr(m, wf, name);
    cpu_rms_norm(hidden, norm_w, normed, m->cfg.hidden_dim, RMS_NORM_EPS);

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] normed_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(normed, m->cfg.hidden_dim), normed[0], normed[1], normed[2], normed[3], normed[4]);
    }

    // ---- QKV Projection ----
    int q_proj_dim = m->cfg.num_attn_heads * HEAD_DIM * 2;
    int q_dim = m->cfg.num_attn_heads * HEAD_DIM;
    int kv_dim = m->cfg.num_kv_heads * HEAD_DIM;

    float *q_proj_out = calloc(q_proj_dim, sizeof(float));
    float *k = calloc(kv_dim, sizeof(float));
    float *v = calloc(kv_dim, sizeof(float));

    // Batch Q/K/V projections into a single GPU command buffer
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", layer_idx);
    uint32_t *qw = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.scales", layer_idx);
    uint16_t *qs = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.biases", layer_idx);
    uint16_t *qb = get_tensor_ptr(m, wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", layer_idx);
    uint32_t *kw = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.scales", layer_idx);
    uint16_t *ks = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.biases", layer_idx);
    uint16_t *kb = get_tensor_ptr(m, wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", layer_idx);
    uint32_t *vw = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.scales", layer_idx);
    uint16_t *vs = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.biases", layer_idx);
    uint16_t *vb = get_tensor_ptr(m, wf, name);

    // Batch Q/K/V into one command buffer (3 dispatches, 1 commit)
    if (qw && qs && qb && kw && ks && kb && vw && vs && vb) {
        BatchMatvecSpec qkv_specs[3] = {
            { qw, qs, qb, q_proj_out, (uint32_t)q_proj_dim, m->cfg.hidden_dim, GROUP_SIZE, 0 },
            { kw, ks, kb, k,          (uint32_t)kv_dim,     m->cfg.hidden_dim, GROUP_SIZE, 1 },
            { vw, vs, vb, v,          (uint32_t)kv_dim,     m->cfg.hidden_dim, GROUP_SIZE, 2 },
        };
        fast_batch_matvec(m, normed, m->cfg.hidden_dim, qkv_specs, 3);
    }

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] q_proj first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                q_proj_out[0], q_proj_out[1], q_proj_out[2], q_proj_out[3], q_proj_out[4]);
    }

    // Split q_proj_out into queries and gate
    float *q = calloc(q_dim, sizeof(float));
    float *q_gate = calloc(q_dim, sizeof(float));
    for (int h = 0; h < m->cfg.num_attn_heads; h++) {
        float *src = q_proj_out + h * (2 * HEAD_DIM);
        memcpy(q + h * HEAD_DIM, src, HEAD_DIM * sizeof(float));
        memcpy(q_gate + h * HEAD_DIM, src + HEAD_DIM, HEAD_DIM * sizeof(float));
    }
    free(q_proj_out);

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] v_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(v, kv_dim), v[0], v[1], v[2], v[3], v[4]);
        fprintf(stderr, "[FA-DBG] q_gate_rms=%.6f gate_first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(q_gate, q_dim), q_gate[0], q_gate[1], q_gate[2], q_gate[3], q_gate[4]);
        float gate_sigmoid_sum = 0.0f;
        for (int i = 0; i < q_dim; i++) {
            gate_sigmoid_sum += 1.0f / (1.0f + expf(-q_gate[i]));
        }
        fprintf(stderr, "[FA-DBG] gate_sigmoid_mean=%.6f\n", gate_sigmoid_sum / q_dim);
    }

    // ---- Q/K RMSNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", layer_idx);
    uint16_t *qnorm_w = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", layer_idx);
    uint16_t *knorm_w = get_tensor_ptr(m, wf, name);

    // Apply per-head Q norm
    if (qnorm_w) {
        for (int h = 0; h < m->cfg.num_attn_heads; h++) {
            float *qh = q + h * HEAD_DIM;
            float sum_sq = 0.0f;
            for (int i = 0; i < HEAD_DIM; i++) sum_sq += qh[i] * qh[i];
            float inv_rms = 1.0f / sqrtf(sum_sq / HEAD_DIM + RMS_NORM_EPS);
            for (int i = 0; i < HEAD_DIM; i++) {
                qh[i] = qh[i] * inv_rms * bf16_to_f32(qnorm_w[i]);
            }
        }
    }
    // Apply per-head K norm
    if (knorm_w) {
        for (int h = 0; h < m->cfg.num_kv_heads; h++) {
            float *kh = k + h * HEAD_DIM;
            float sum_sq = 0.0f;
            for (int i = 0; i < HEAD_DIM; i++) sum_sq += kh[i] * kh[i];
            float inv_rms = 1.0f / sqrtf(sum_sq / HEAD_DIM + RMS_NORM_EPS);
            for (int i = 0; i < HEAD_DIM; i++) {
                kh[i] = kh[i] * inv_rms * bf16_to_f32(knorm_w[i]);
            }
        }
    }


    // ---- RoPE ----
    apply_rotary_emb(q, k, pos, m->cfg.num_attn_heads, m->cfg.num_kv_heads, HEAD_DIM, m->cfg.rotary_dim);

    // ---- Update KV cache ----
    int cache_pos = kv->len;
#if USE_KV_CACHE_BF16
    for (int i = 0; i < kv_dim; i++) {
        kv->k_cache[cache_pos * kv_dim + i] = f32_to_kv_elem(k[i]);
        kv->v_cache[cache_pos * kv_dim + i] = f32_to_kv_elem(v[i]);
    }
#else
    memcpy(kv->k_cache + cache_pos * kv_dim, k, kv_dim * sizeof(float));
    memcpy(kv->v_cache + cache_pos * kv_dim, v, kv_dim * sizeof(float));
#endif
    kv->len++;

    // ---- Scaled dot-product attention ----
    int heads_per_kv = m->cfg.num_attn_heads / m->cfg.num_kv_heads;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    float *attn_out = calloc(q_dim, sizeof(float));

    for (int h = 0; h < m->cfg.num_attn_heads; h++) {
        int kv_h = h / heads_per_kv;
        float *qh = q + h * HEAD_DIM;

        float *scores = malloc(kv->len * sizeof(float));
        for (int p = 0; p < kv->len; p++) {
            kv_elem_t *kp = kv->k_cache + p * kv_dim + kv_h * HEAD_DIM;
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += qh[d] * kv_elem_to_f32(kp[d]);
            }
            scores[p] = dot * scale;
        }

        cpu_softmax(scores, kv->len);

        float *oh = attn_out + h * HEAD_DIM;
        for (int p = 0; p < kv->len; p++) {
            kv_elem_t *vp = kv->v_cache + p * kv_dim + kv_h * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) {
                oh[d] += scores[p] * kv_elem_to_f32(vp[d]);
            }
        }
        free(scores);
    }


    // ---- Apply sigmoid gate to attention output ----
    for (int i = 0; i < q_dim; i++) {
        float g = 1.0f / (1.0f + expf(-q_gate[i]));
        attn_out[i] *= g;
    }

    // ---- Output projection ----
    float *attn_projected = calloc(m->cfg.hidden_dim, sizeof(float));
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", layer_idx);
    uint32_t *ow = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.scales", layer_idx);
    uint16_t *os_ptr = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.biases", layer_idx);
    uint16_t *ob = get_tensor_ptr(m, wf, name);
    if (ow && os_ptr && ob) fast_dequant_matvec(m, ow, os_ptr, ob, attn_out, attn_projected, m->cfg.hidden_dim, q_dim, GROUP_SIZE);

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] attn_out_rms=%.6f o_proj first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(attn_out, q_dim),
                attn_projected[0], attn_projected[1], attn_projected[2], attn_projected[3], attn_projected[4]);
    }

    // ---- Residual connection ----
    for (int i = 0; i < m->cfg.hidden_dim; i++) {
        hidden[i] = residual[i] + attn_projected[i];
    }

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] AFTER layer=%d hidden_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, vec_rms(hidden, m->cfg.hidden_dim),
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    free(normed);
    free(residual);
    free(q);
    free(q_gate);
    free(k);
    free(v);
    free(attn_out);
    free(attn_projected);
}

// ============================================================================
// Linear attention layer forward (GatedDeltaNet, single token, incremental)
// ============================================================================

// RMS norm without weights (just normalize)
static void cpu_rms_norm_bare(const float *x, float *out, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * inv_rms;
}

// RMSNormGated: out = rms_norm(x) * silu(z)
static void cpu_rms_norm_gated(const float *x, const float *z, const uint16_t *w_bf16,
                                float *out, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) {
        float w = bf16_to_f32(w_bf16[i]);
        float silu_z = z[i] / (1.0f + expf(-z[i]));
        out[i] = x[i] * inv_rms * w * silu_z;
    }
}

__attribute__((unused))
static void linear_attention_forward(
    FlashMoE_Model *m,
    WeightFile *wf,
    int layer_idx,
    float *hidden,           // [m->cfg.hidden_dim] in/out
    LinearAttnState *state
) {
    // If bypass is enabled, just pass through (identity)
    if (m->linear_attn_bypass) {
        (void)wf; (void)layer_idx; (void)state;
        return;
    }

    static int la_debug_count = 0;
    la_debug_count++;
    int la_debug = 0;

    if (la_debug) {
        fprintf(stderr, "[LA-DBG] layer=%d hidden_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, vec_rms(hidden, m->cfg.hidden_dim),
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    char name[256];
    float *normed = malloc(m->cfg.hidden_dim * sizeof(float));
    float *residual = malloc(m->cfg.hidden_dim * sizeof(float));
    cpu_vec_copy(residual, hidden, m->cfg.hidden_dim);

    // ---- Input LayerNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer_idx);
    uint16_t *norm_w = get_tensor_ptr(m, wf, name);
    cpu_rms_norm(hidden, norm_w, normed, m->cfg.hidden_dim, RMS_NORM_EPS);

    // ---- Batch QKV + Z + B + A projections (4 matmuls, 1 command buffer) ----
    int qkv_dim = m->cfg.linear_conv_dim;
    float *qkv = calloc(qkv_dim, sizeof(float));
    int z_dim = m->cfg.linear_total_value;
    float *z = calloc(z_dim, sizeof(float));
    float *beta = calloc(m->cfg.linear_num_v_heads, sizeof(float));
    float *alpha = calloc(m->cfg.linear_num_v_heads, sizeof(float));

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.weight", layer_idx);
    uint32_t *qkv_w = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.scales", layer_idx);
    uint16_t *qkv_s = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.biases", layer_idx);
    uint16_t *qkv_b = get_tensor_ptr(m, wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.weight", layer_idx);
    uint32_t *z_w = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.scales", layer_idx);
    uint16_t *z_s = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.biases", layer_idx);
    uint16_t *z_b = get_tensor_ptr(m, wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.weight", layer_idx);
    uint32_t *b_w = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.scales", layer_idx);
    uint16_t *b_s = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.biases", layer_idx);
    uint16_t *b_b = get_tensor_ptr(m, wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.weight", layer_idx);
    uint32_t *a_w = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.scales", layer_idx);
    uint16_t *a_s = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.biases", layer_idx);
    uint16_t *a_b = get_tensor_ptr(m, wf, name);

    if (qkv_w && qkv_s && qkv_b && z_w && z_s && z_b &&
        b_w && b_s && b_b && a_w && a_s && a_b) {
        BatchMatvecSpec la_specs[4] = {
            { qkv_w, qkv_s, qkv_b, qkv,   (uint32_t)qkv_dim,         m->cfg.hidden_dim, GROUP_SIZE, 0 },
            { z_w,   z_s,   z_b,   z,      (uint32_t)z_dim,           m->cfg.hidden_dim, GROUP_SIZE, 1 },
            { b_w,   b_s,   b_b,   beta,   (uint32_t)m->cfg.linear_num_v_heads, m->cfg.hidden_dim, GROUP_SIZE, 2 },
            { a_w,   a_s,   a_b,   alpha,  (uint32_t)m->cfg.linear_num_v_heads, m->cfg.hidden_dim, GROUP_SIZE, 3 },
        };
        fast_batch_matvec(m, normed, m->cfg.hidden_dim, la_specs, 4);
    }

    // ---- Conv1d step ----
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.conv1d.weight", layer_idx);
    uint16_t *conv_w = get_tensor_ptr(m, wf, name);

    float *conv_out = calloc(qkv_dim, sizeof(float));
    if (conv_w) {
        cpu_conv1d_step(state->conv_state, qkv, conv_w, conv_out,
                        qkv_dim, CONV_KERNEL_SIZE);
    }

    // Update conv state: shift left, append new input
    memmove(state->conv_state, state->conv_state + qkv_dim,
            (CONV_KERNEL_SIZE - 2) * qkv_dim * sizeof(float));
    memcpy(state->conv_state + (CONV_KERNEL_SIZE - 2) * qkv_dim, qkv,
           qkv_dim * sizeof(float));

    // ---- Split conv_out into q, k, v ----
    float *lin_q = conv_out;
    float *lin_k = conv_out + m->cfg.linear_total_key;
    float *lin_v = conv_out + 2 * m->cfg.linear_total_key;

    // ---- RMS normalize q and k (bare, no weights) ----
    float inv_scale = 1.0f / sqrtf((float)LINEAR_KEY_DIM);

    for (int h = 0; h < m->cfg.linear_num_k_heads; h++) {
        float *qh = lin_q + h * LINEAR_KEY_DIM;
        cpu_rms_norm_bare(qh, qh, LINEAR_KEY_DIM, 1e-6f);
        float q_scale = inv_scale * inv_scale;
        for (int d = 0; d < LINEAR_KEY_DIM; d++) qh[d] *= q_scale;
    }
    for (int h = 0; h < m->cfg.linear_num_k_heads; h++) {
        float *kh = lin_k + h * LINEAR_KEY_DIM;
        cpu_rms_norm_bare(kh, kh, LINEAR_KEY_DIM, 1e-6f);
        for (int d = 0; d < LINEAR_KEY_DIM; d++) kh[d] *= inv_scale;
    }

    // ---- Gated delta net recurrence ----
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.A_log", layer_idx);
    float *A_log = get_tensor_ptr(m, wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.dt_bias", layer_idx);
    uint16_t *dt_bias_bf16 = get_tensor_ptr(m, wf, name);

    float *out_values = calloc(m->cfg.linear_total_value, sizeof(float));

    int k_heads_per_v = m->cfg.linear_num_v_heads / m->cfg.linear_num_k_heads;

    // Precompute per-head decay (g) and beta
    float g_decay[m->cfg.linear_num_v_heads];
    float beta_gate[m->cfg.linear_num_v_heads];
    for (int vh = 0; vh < m->cfg.linear_num_v_heads; vh++) {
        float a_val = alpha[vh];
        float dt_b = dt_bias_bf16 ? bf16_to_f32(dt_bias_bf16[vh]) : 0.0f;
        float A_val = A_log ? expf(A_log[vh]) : 1.0f;
        float softplus_val = logf(1.0f + expf(a_val + dt_b));
        g_decay[vh] = expf(-A_val * softplus_val);

        beta_gate[vh] = cpu_sigmoid(beta[vh]);
    }

    for (int vh = 0; vh < m->cfg.linear_num_v_heads; vh++) {
        int kh = vh / k_heads_per_v;

        float g = g_decay[vh];
        float b_gate = beta_gate[vh];

        float *S = state->ssm_state + vh * LINEAR_VALUE_DIM * LINEAR_KEY_DIM;
        float *v_h = lin_v + vh * LINEAR_VALUE_DIM;
        float *k_h = lin_k + kh * LINEAR_KEY_DIM;

        // Step 1: Decay state
        for (int vi = 0; vi < LINEAR_VALUE_DIM; vi++) {
            for (int ki = 0; ki < LINEAR_KEY_DIM; ki++) {
                S[vi * LINEAR_KEY_DIM + ki] *= g;
            }
        }

        // Step 2: Compute kv_mem, delta, and update state
        for (int vi = 0; vi < LINEAR_VALUE_DIM; vi++) {
            float kv_mem = 0.0f;
            for (int ki = 0; ki < LINEAR_KEY_DIM; ki++) {
                kv_mem += S[vi * LINEAR_KEY_DIM + ki] * k_h[ki];
            }
            float delta = (v_h[vi] - kv_mem) * b_gate;
            for (int ki = 0; ki < LINEAR_KEY_DIM; ki++) {
                S[vi * LINEAR_KEY_DIM + ki] += k_h[ki] * delta;
            }
        }

        // Step 3: Output
        float *q_h = lin_q + kh * LINEAR_KEY_DIM;
        float *o_h = out_values + vh * LINEAR_VALUE_DIM;
        for (int vi = 0; vi < LINEAR_VALUE_DIM; vi++) {
            float sum = 0.0f;
            for (int ki = 0; ki < LINEAR_KEY_DIM; ki++) {
                sum += S[vi * LINEAR_KEY_DIM + ki] * q_h[ki];
            }
            o_h[vi] = sum;
        }
    }

    // ---- RMSNormGated ----
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.norm.weight", layer_idx);
    uint16_t *gated_norm_w = get_tensor_ptr(m, wf, name);

    float *gated_out = calloc(m->cfg.linear_total_value, sizeof(float));
    for (int vh = 0; vh < m->cfg.linear_num_v_heads; vh++) {
        float *oh = out_values + vh * LINEAR_VALUE_DIM;
        float *zh = z + vh * LINEAR_VALUE_DIM;
        float *gh = gated_out + vh * LINEAR_VALUE_DIM;
        if (gated_norm_w) {
            cpu_rms_norm_gated(oh, zh, gated_norm_w, gh, LINEAR_VALUE_DIM, RMS_NORM_EPS);
        } else {
            memcpy(gh, oh, LINEAR_VALUE_DIM * sizeof(float));
        }
    }

    // ---- Output projection ----
    float *attn_out = calloc(m->cfg.hidden_dim, sizeof(float));
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.weight", layer_idx);
    uint32_t *out_w = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.scales", layer_idx);
    uint16_t *out_s = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.biases", layer_idx);
    uint16_t *out_b = get_tensor_ptr(m, wf, name);
    if (out_w && out_s && out_b) {
        fast_dequant_matvec(m, out_w, out_s, out_b, gated_out, attn_out, m->cfg.hidden_dim,
                            m->cfg.linear_total_value, GROUP_SIZE);
    }

    // ---- Residual ----
    for (int i = 0; i < m->cfg.hidden_dim; i++) {
        hidden[i] = residual[i] + attn_out[i];
    }

    if (la_debug) {
        fprintf(stderr, "[LA-DBG] AFTER layer=%d out_proj_rms=%.6f gated_rms=%.6f hidden_rms=%.6f\n",
                layer_idx, vec_rms(attn_out, m->cfg.hidden_dim),
                vec_rms(gated_out, m->cfg.linear_total_value),
                vec_rms(hidden, m->cfg.hidden_dim));
    }

    free(normed);
    free(residual);
    free(qkv);
    free(z);
    free(beta);
    free(alpha);
    free(conv_out);
    free(out_values);
    free(gated_out);
    free(attn_out);
}


#endif // ATTENTION_H
