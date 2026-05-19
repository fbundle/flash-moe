#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

// ============================================================================
// Embedding lookup (4-bit quantized)
// ============================================================================

#include "model_internal.h"

static void embed_lookup(FlashMoE_Model *m, WeightFile *wf, int token_id, float *out) {
    TensorInfo *w_info = get_tensor_info(m, wf, "model.embed_tokens.weight");
    TensorInfo *s_info = get_tensor_info(m, wf, "model.embed_tokens.scales");
    TensorInfo *b_info = get_tensor_info(m, wf, "model.embed_tokens.biases");

    if (!w_info || !s_info || !b_info) {
        fprintf(stderr, "ERROR: embedding tensors not found\n");
        memset(out, 0, m->cfg.hidden_dim * sizeof(float));
        return;
    }

    int packed_cols = w_info->shape[1];
    int num_groups = s_info->shape[1];

    uint32_t *W = (uint32_t *)((char *)wf->data + w_info->offset);
    uint16_t *S = (uint16_t *)((char *)wf->data + s_info->offset);
    uint16_t *B = (uint16_t *)((char *)wf->data + b_info->offset);

    const uint32_t *w_row = W + (size_t)token_id * packed_cols;
    const uint16_t *s_row = S + (size_t)token_id * num_groups;
    const uint16_t *b_row = B + (size_t)token_id * num_groups;

    int group_size = m->cfg.hidden_dim / num_groups;
    int packed_per_group = group_size / 8;

    for (int g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(s_row[g]);
        float bias = bf16_to_f32(b_row[g]);

        for (int p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[g * packed_per_group + p];
            int base = g * group_size + p * 8;

            for (int n = 0; n < 8; n++) {
                uint32_t nibble = (packed >> (n * 4)) & 0xF;
                out[base + n] = (float)nibble * scale + bias;
            }
        }
    }
}

// ============================================================================
// LM head (logits projection)
// ============================================================================

static void lm_head_forward(FlashMoE_Model *m, WeightFile *wf, const float *hidden, float *logits) {
    TensorInfo *w_info = get_tensor_info(m, wf, "lm_head.weight");
    TensorInfo *s_info = get_tensor_info(m, wf, "lm_head.scales");
    TensorInfo *b_info = get_tensor_info(m, wf, "lm_head.biases");

    if (!w_info || !s_info || !b_info) {
        fprintf(stderr, "ERROR: lm_head tensors not found\n");
        return;
    }

    uint32_t *W = (uint32_t *)((char *)wf->data + w_info->offset);
    uint16_t *S = (uint16_t *)((char *)wf->data + s_info->offset);
    uint16_t *B = (uint16_t *)((char *)wf->data + b_info->offset);

    fast_dequant_matvec(m, W, S, B, hidden, logits, m->cfg.vocab_size, m->cfg.hidden_dim, GROUP_SIZE);
}


#endif // EMBEDDINGS_H
