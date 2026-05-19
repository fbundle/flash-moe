#ifndef LAYER_FORWARD_H
#define LAYER_FORWARD_H

#include "common.h"

// ============================================================================
// Per-layer weight pointer cache — built once, eliminates 40+ snprintf+lookup
// per layer per token. With 60 layers and 15 tokens = 36,000 lookups saved.
// ============================================================================


static void build_layer_cache(FlashMoE_Context *m, WeightFile *wf) {
    if (m->layer_cache_built) return;
    m->layer_cache = calloc(m->cfg.num_layers, sizeof(LayerWeightCache));
    char name[256];

    for (int i = 0; i < m->cfg.num_layers; i++) {
        LayerWeightCache *lc = &m->layer_cache[i];
        int is_full = ((i + 1) % m->cfg.full_attn_interval == 0);

        // Norms
        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", i);
        lc->input_norm_w = get_tensor_ptr(m, wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", i);
        lc->post_attn_norm_w = get_tensor_ptr(m, wf, name);

        if (is_full) {
            // Full attention
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", i);
            lc->q_w = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.scales", i);
            lc->q_s = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.biases", i);
            lc->q_b = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", i);
            lc->k_w = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.scales", i);
            lc->k_s = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.biases", i);
            lc->k_b = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", i);
            lc->v_w = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.scales", i);
            lc->v_s = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.biases", i);
            lc->v_b = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", i);
            lc->o_w = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.scales", i);
            lc->o_s = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.biases", i);
            lc->o_b = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", i);
            lc->q_norm_w = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", i);
            lc->k_norm_w = get_tensor_ptr(m, wf, name);
        } else {
            // Linear attention
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.weight", i);
            lc->qkv_w = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.scales", i);
            lc->qkv_s = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.biases", i);
            lc->qkv_b = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.weight", i);
            lc->z_w = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.scales", i);
            lc->z_s = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.biases", i);
            lc->z_b = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.weight", i);
            lc->b_w = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.scales", i);
            lc->b_s = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.biases", i);
            lc->b_b = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.weight", i);
            lc->a_w = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.scales", i);
            lc->a_s = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.biases", i);
            lc->a_b = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.conv1d.weight", i);
            lc->conv1d_w = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.A_log", i);
            lc->A_log = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.dt_bias", i);
            lc->dt_bias = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.norm.weight", i);
            lc->gated_norm_w = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.weight", i);
            lc->out_proj_w = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.scales", i);
            lc->out_proj_s = get_tensor_ptr(m, wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.biases", i);
            lc->out_proj_b = get_tensor_ptr(m, wf, name);
        }

        // MoE weights (same for all layers)
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.weight", i);
        lc->gate_w = get_tensor_ptr(m, wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.scales", i);
        lc->gate_s = get_tensor_ptr(m, wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.biases", i);
        lc->gate_b = get_tensor_ptr(m, wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.weight", i);
        lc->sg_w = get_tensor_ptr(m, wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.scales", i);
        lc->sg_s = get_tensor_ptr(m, wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.biases", i);
        lc->sg_b = get_tensor_ptr(m, wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.weight", i);
        lc->su_w = get_tensor_ptr(m, wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.scales", i);
        lc->su_s = get_tensor_ptr(m, wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.biases", i);
        lc->su_b = get_tensor_ptr(m, wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.weight", i);
        lc->sd_w = get_tensor_ptr(m, wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.scales", i);
        lc->sd_s = get_tensor_ptr(m, wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.biases", i);
        lc->sd_b = get_tensor_ptr(m, wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.weight", i);
        lc->seg_w = get_tensor_ptr(m, wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.scales", i);
        lc->seg_s = get_tensor_ptr(m, wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.biases", i);
        lc->seg_b = get_tensor_ptr(m, wf, name);
    }

    m->layer_cache_built = 1;
    printf("[cache] Pre-computed weight pointers for %d layers\n", m->cfg.num_layers);
}

// ============================================================================
// Deferred expert state: holds state for async GPU expert compute.
// GPU experts are submitted async (commit without wait), and the wait+combine
// happens at the start of the NEXT layer. This overlaps ~1ms of GPU expert
// compute with the next layer's attention+routing CPU/GPU work.
// ============================================================================


// Wait for the deferred GPU expert command buffer to complete.
// Split from finalize so timing can be measured independently.
static void wait_deferred_experts_gpu(FlashMoE_Context *m) {
    if (!m->deferred.active) return;
#if USE_EVENT_PIPELINE
    // MTLSharedEvent non-blocking fast path (llama.cpp pattern)
    id<MTLSharedEvent> ev = m->metal->pipeline_event;
    if (ev && [ev signaledValue] >= m->deferred.expert_event_value) {
        return;
    }
#endif
    [m->deferred.cmd_experts waitUntilCompleted];
}

// CPU readback + accumulate + combine after GPU is done.
// Must be called after wait_deferred_experts_gpu(m).
// When gpu_combined=1, the GPU already computed the combine+residual+norm
// in CMD3, so we just need to read back the hidden state from buf_moe_hidden.
static void finalize_deferred_experts(FlashMoE_Context *m) {
    if (!m->deferred.active) return;

    if (m->deferred.gpu_combined) {
        // GPU-side combine: hidden state is already in buf_moe_hidden.
        // buf_input already has the normalized input for the next layer's CMD1.
        // Just read back hidden (needed for the residual connection in future layers).
        memcpy(m->deferred.hidden, [m->metal->buf_moe_hidden contents],
               m->cfg.hidden_dim * sizeof(float));
    } else {
        // CPU-side combine (original path)
        // Read back and accumulate routed expert outputs
        float moe_out[m->cfg.hidden_dim];
        memset(moe_out, 0, sizeof(moe_out));
        for (int k = 0; k < m->deferred.actual_K; k++) {
            if (!m->deferred.valid[k]) continue;
            float *expert_result = (float *)[m->metal->buf_multi_expert_out[k] contents];
            cpu_vec_madd(moe_out, expert_result, m->deferred.expert_weights[k], m->cfg.hidden_dim);
        }

        // Read shared expert result
        float shared_out[m->cfg.hidden_dim];
        memcpy(shared_out, [m->metal->buf_shared_out contents], m->cfg.hidden_dim * sizeof(float));

        // Apply shared expert gate
        float shared_weight = cpu_sigmoid(m->deferred.shared_gate_score);
        for (int i = 0; i < m->cfg.hidden_dim; i++) {
            shared_out[i] *= shared_weight;
        }

        // Final combine: hidden = h_mid + moe_out + shared_out
        for (int i = 0; i < m->cfg.hidden_dim; i++) {
            m->deferred.hidden[i] = m->deferred.h_mid[i] + moe_out[i] + shared_out[i];
        }
    }

    m->deferred.active = 0;
    m->deferred.gpu_combined = 0;
    m->deferred.cmd_experts = nil;
}

// Complete the deferred GPU expert compute: wait for GPU, read back, accumulate, combine.
// Must be called before the next layer modifies static scratch buffers.
static void complete_deferred_experts(FlashMoE_Context *m) {
    wait_deferred_experts_gpu(m);
    finalize_deferred_experts(m);
}

// Discard the deferred GPU expert result: wait for GPU to finish (for buffer safety)
// but skip the CPU readback/combine. Used during prefill for intermediate tokens
// where the hidden state will be immediately overwritten by the next token's embedding.
// This saves ~0.1-0.2ms per prefill token (avoids unnecessary memcpy + combine work).
void discard_deferred_experts(FlashMoE_Context *m) {
    wait_deferred_experts_gpu(m);
    // Clear deferred state without reading back results
    if (m->deferred.active) {
        m->deferred.active = 0;
        m->deferred.gpu_combined = 0;
        m->deferred.cmd_experts = nil;
    }
}

// ============================================================================
// Fused layer forward: GPU/CPU overlap + deferred expert pipeline
//
// Pipeline per layer (3 cmd buffers, GPU-side combine in CMD3):
//
//   FAST PATH (when previous CMD3 did GPU-side combine):
//     CMD1: submit immediately (buf_input already populated by CMD3(N-1))
//     WAIT: CMD1 complete (implies CMD3(N-1) also done, queue is serial)
//     CPU:  finalize deferred (read back hidden from buf_moe_hidden)
//
//   SLOW PATH (first layer, or last layer's CMD3 without GPU combine):
//     [DEFERRED] Wait for PREVIOUS layer's CMD3 (if any) + CPU combine
//     CPU:  input_norm(hidden) -> normed -> buf_input
//     CMD1: attention projections (commit)
//     WAIT: CMD1 complete
//
//   Then (both paths):
//     CPU:  attention compute (RoPE/softmax/delta-net)
//     CMD2: o_proj + residual + norm + routing + shared expert projs (8 encoders, 1 commit)
//     WAIT: CMD2 complete
//     CPU:  softmax + top-K routing
//     I/O:  parallel pread K experts (4 pthreads)
//     CMD3: K expert forwards + shared SwiGLU + shared down
//           + moe_combine_residual + rms_norm -> buf_input (ASYNC commit, NO wait)
//     RETURN: GPU experts + combine running async
//
// GPU-side combine eliminates the 0.83ms deferred_wait + CPU combine + input_norm
// at the start of each layer, allowing CMD1 to be submitted immediately.
//
// Key optimizations:
//   1. Parallel pread (4 threads) instead of sequential: ~4x I/O speedup
//   2. o_proj fused into CMD2 with routing (saves 1 commit+wait)
//   3. Deferred CMD3 (expert GPU compute overlapped with next layer)
//   4. GPU-side combine in CMD3 (eliminates CPU deferred_wait + combine + norm)
// ============================================================================

// Static scratch buffers — allocated once, reused across all layers per token.
// Full attention scratch
// Linear attention scratch

static void init_layer_scratch(FlashMoE_Context *m) {
    m->s_normed          = calloc(m->cfg.hidden_dim, sizeof(float));
    m->s_residual        = calloc(m->cfg.hidden_dim, sizeof(float));
    m->s_attn_proj       = calloc(m->cfg.hidden_dim, sizeof(float));
    m->s_h_post          = calloc(m->cfg.hidden_dim, sizeof(float));
    m->s_h_mid           = calloc(m->cfg.hidden_dim, sizeof(float));
    m->s_gate_scores     = calloc(m->cfg.num_experts, sizeof(float));
    m->s_spec_gate_scores = calloc(m->cfg.num_experts, sizeof(float));
    m->s_shared_gate     = calloc(m->cfg.shared_intermediate, sizeof(float));
    m->s_shared_up       = calloc(m->cfg.shared_intermediate, sizeof(float));
    m->s_moe_out         = calloc(m->cfg.hidden_dim, sizeof(float));
    m->s_shared_out      = calloc(m->cfg.hidden_dim, sizeof(float));
    m->s_q_proj_out      = calloc(m->cfg.num_attn_heads * m->cfg.head_dim * 2, sizeof(float));
    m->s_k_proj_out      = calloc(m->cfg.num_kv_heads * m->cfg.head_dim, sizeof(float));
    m->s_v_proj_out      = calloc(m->cfg.num_kv_heads * m->cfg.head_dim, sizeof(float));
    m->s_q               = calloc(m->cfg.num_attn_heads * m->cfg.head_dim, sizeof(float));
    m->s_q_gate          = calloc(m->cfg.num_attn_heads * m->cfg.head_dim, sizeof(float));
    m->s_attn_out        = calloc(m->cfg.num_attn_heads * m->cfg.head_dim, sizeof(float));
    m->s_qkv_proj_out    = calloc(m->cfg.linear_conv_dim, sizeof(float));
    m->s_z_proj_out      = calloc(m->cfg.linear_total_value, sizeof(float));
    m->s_beta_proj_out   = calloc(m->cfg.linear_num_v_heads, sizeof(float));
    m->s_alpha_proj_out  = calloc(m->cfg.linear_num_v_heads, sizeof(float));
    m->s_conv_out        = calloc(m->cfg.linear_conv_dim, sizeof(float));
    m->s_out_vals        = calloc(m->cfg.linear_total_value, sizeof(float));
    m->s_gated_out       = calloc(m->cfg.linear_total_value, sizeof(float));
    m->deferred.h_mid  = calloc(m->cfg.hidden_dim, sizeof(float));
}

// ============================================================================
// Synchronous MoE forward — simple CPU/GPU path, no async I/O, no deferring.
// Used by fused_layer_forward_debug for debugging/validation.
// ============================================================================


static void moe_forward(
    FlashMoE_Context *m,
    WeightFile *wf,
    int layer_idx,
    float *hidden,
    int K,
    int packed_fd
) {
    m->moe_sync_debug_count++;
    int moe_debug = 0;
    int moe_dump = 0;

    char name[256];
    float *h_post = malloc(m->cfg.hidden_dim * sizeof(float));
    float *h_mid = malloc(m->cfg.hidden_dim * sizeof(float));
    cpu_vec_copy(h_mid, hidden, m->cfg.hidden_dim);

    // ---- Post-attention LayerNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", layer_idx);
    uint16_t *norm_w = get_tensor_ptr(m, wf, name);
    cpu_rms_norm(hidden, norm_w, h_post, m->cfg.hidden_dim, m->cfg.rms_norm_eps);

    // ---- Batch routing gate + shared expert gate/up + shared_expert_gate ----
    float *gate_scores = calloc(m->cfg.num_experts, sizeof(float));
    float *shared_gate = calloc(m->cfg.shared_intermediate, sizeof(float));
    float *shared_up = calloc(m->cfg.shared_intermediate, sizeof(float));
    float shared_gate_score = 0.0f;

    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.weight", layer_idx);
    uint32_t *gate_w = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.scales", layer_idx);
    uint16_t *gate_s = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.biases", layer_idx);
    uint16_t *gate_b = get_tensor_ptr(m, wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.weight", layer_idx);
    uint32_t *sgw = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.scales", layer_idx);
    uint16_t *sgs = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.biases", layer_idx);
    uint16_t *sgb = get_tensor_ptr(m, wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.weight", layer_idx);
    uint32_t *suw = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.scales", layer_idx);
    uint16_t *sus = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.biases", layer_idx);
    uint16_t *sub = get_tensor_ptr(m, wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.weight", layer_idx);
    uint32_t *seg_w = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.scales", layer_idx);
    uint16_t *seg_s = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.biases", layer_idx);
    uint16_t *seg_b = get_tensor_ptr(m, wf, name);

    if (gate_w && gate_s && gate_b && sgw && sgs && sgb &&
        suw && sus && sub && seg_w && seg_s && seg_b) {
        BatchMatvecSpec moe_specs[4] = {
            { gate_w, gate_s, gate_b, gate_scores,        (uint32_t)m->cfg.num_experts,        m->cfg.hidden_dim, m->cfg.group_size, 0 },
            { sgw,    sgs,    sgb,    shared_gate,         (uint32_t)m->cfg.shared_intermediate, m->cfg.hidden_dim, m->cfg.group_size, 1 },
            { suw,    sus,    sub,    shared_up,           (uint32_t)m->cfg.shared_intermediate, m->cfg.hidden_dim, m->cfg.group_size, 2 },
            { seg_w,  seg_s,  seg_b,  &shared_gate_score,  1,                            m->cfg.hidden_dim, m->cfg.group_size, 3 },
        };
        fast_batch_matvec(m, h_post, m->cfg.hidden_dim, moe_specs, 4);
    }

    cpu_softmax(gate_scores, m->cfg.num_experts);

    int expert_indices[64];
    float expert_weights[64];
    cpu_topk(gate_scores, m->cfg.num_experts, K, expert_indices, expert_weights);
    cpu_normalize_weights(expert_weights, K);

    if (moe_dump) {
        fprintf(stderr, "[MOE-DUMP] routing: K=%d experts=[", K);
        for (int k = 0; k < K; k++) fprintf(stderr, "%d(%.4f)%s", expert_indices[k], expert_weights[k], k<K-1?",":"");
        fprintf(stderr, "]\n");
    }

    // ---- Routed expert computation ----
    float *moe_out = calloc(m->cfg.hidden_dim, sizeof(float));

    if (packed_fd >= 0) {
        float *expert_out = malloc(m->cfg.hidden_dim * sizeof(float));

        size_t esz = active_expert_size(m);
        for (int k = 0; k < K; k++) {
            int eidx = expert_indices[k];
            off_t expert_offset = (off_t)eidx * esz;

            if (m->metal && m->metal->buf_expert_data) {
                void *expert_buf_ptr = [m->metal->buf_expert_data contents];
                ssize_t nread = pread(packed_fd, expert_buf_ptr, esz, expert_offset);
                if (nread != (ssize_t)esz) {
                    fprintf(stderr, "WARNING: layer %d expert %d pread: %zd/%zu\n",
                            layer_idx, eidx, nread, esz);
                    continue;
                }
                gpu_expert_forward(m, m->metal, expert_buf_ptr, h_post, expert_out, 1);
            } else {
                void *expert_data = malloc(esz);
                ssize_t nread = pread(packed_fd, expert_data, esz, expert_offset);
                if (nread != (ssize_t)esz) {
                    fprintf(stderr, "WARNING: layer %d expert %d pread: %zd/%zu\n",
                            layer_idx, eidx, nread, esz);
                    free(expert_data);
                    continue;
                }

                uint32_t *gw = (uint32_t *)expert_data;
                uint16_t *gs_p = (uint16_t *)((char *)expert_data + (m->use_2bit ? m->cfg.layout_2bit.gate_s_off : m->cfg.layout_4bit.gate_s_off));
                uint16_t *gb_p = (uint16_t *)((char *)expert_data + (m->use_2bit ? m->cfg.layout_2bit.gate_b_off : m->cfg.layout_4bit.gate_b_off));
                uint32_t *uw = (uint32_t *)((char *)expert_data + (m->use_2bit ? m->cfg.layout_2bit.up_w_off : m->cfg.layout_4bit.up_w_off));
                uint16_t *us_p = (uint16_t *)((char *)expert_data + (m->use_2bit ? m->cfg.layout_2bit.up_s_off : m->cfg.layout_4bit.up_s_off));
                uint16_t *ub_p = (uint16_t *)((char *)expert_data + (m->use_2bit ? m->cfg.layout_2bit.up_b_off : m->cfg.layout_4bit.up_b_off));
                uint32_t *dw = (uint32_t *)((char *)expert_data + (m->use_2bit ? m->cfg.layout_2bit.down_w_off : m->cfg.layout_4bit.down_w_off));
                uint16_t *ds_p = (uint16_t *)((char *)expert_data + (m->use_2bit ? m->cfg.layout_2bit.down_s_off : m->cfg.layout_4bit.down_s_off));
                uint16_t *db_p = (uint16_t *)((char *)expert_data + (m->use_2bit ? m->cfg.layout_2bit.down_b_off : m->cfg.layout_4bit.down_b_off));

                float *gate_proj_out = malloc(m->cfg.moe_intermediate * sizeof(float));
                float *up_proj_out = malloc(m->cfg.moe_intermediate * sizeof(float));
                float *act_out = malloc(m->cfg.moe_intermediate * sizeof(float));

                cpu_dequant_matvec(gw, gs_p, gb_p, h_post, gate_proj_out,
                                   m->cfg.moe_intermediate, m->cfg.hidden_dim, m->cfg.group_size);
                cpu_dequant_matvec(uw, us_p, ub_p, h_post, up_proj_out,
                                   m->cfg.moe_intermediate, m->cfg.hidden_dim, m->cfg.group_size);
                cpu_swiglu(gate_proj_out, up_proj_out, act_out, m->cfg.moe_intermediate);
                cpu_dequant_matvec(dw, ds_p, db_p, act_out, expert_out,
                                   m->cfg.hidden_dim, m->cfg.moe_intermediate, m->cfg.group_size);

                free(gate_proj_out);
                free(up_proj_out);
                free(act_out);
                free(expert_data);
            }

            if (moe_dump) {
                fprintf(stderr, "[MOE-DUMP] expert[%d] out_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        eidx, vec_rms(expert_out, m->cfg.hidden_dim),
                        expert_out[0], expert_out[1], expert_out[2], expert_out[3], expert_out[4]);
            }
            cpu_vec_madd(moe_out, expert_out, expert_weights[k], m->cfg.hidden_dim);
        }

        free(expert_out);
    }

    // ---- Shared expert SwiGLU ----
    float *shared_out = calloc(m->cfg.hidden_dim, sizeof(float));
    float *shared_act = calloc(m->cfg.shared_intermediate, sizeof(float));
    cpu_swiglu(shared_gate, shared_up, shared_act, m->cfg.shared_intermediate);

    if (moe_dump) {
        fprintf(stderr, "[MOE-DUMP] layer=%d h_post_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, vec_rms(h_post, m->cfg.hidden_dim), h_post[0], h_post[1], h_post[2], h_post[3], h_post[4]);
        fprintf(stderr, "[MOE-DUMP] gate_proj_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(shared_gate, m->cfg.shared_intermediate),
                shared_gate[0], shared_gate[1], shared_gate[2], shared_gate[3], shared_gate[4]);
        fprintf(stderr, "[MOE-DUMP] up_proj_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(shared_up, m->cfg.shared_intermediate),
                shared_up[0], shared_up[1], shared_up[2], shared_up[3], shared_up[4]);
        fprintf(stderr, "[MOE-DUMP] swiglu_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(shared_act, m->cfg.shared_intermediate),
                shared_act[0], shared_act[1], shared_act[2], shared_act[3], shared_act[4]);
    }

    // shared_expert down_proj
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.weight", layer_idx);
    uint32_t *sdw = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.scales", layer_idx);
    uint16_t *sds = get_tensor_ptr(m, wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.biases", layer_idx);
    uint16_t *sdb = get_tensor_ptr(m, wf, name);
    if (sdw && sds && sdb) {
        fast_dequant_matvec(m, sdw, sds, sdb, shared_act, shared_out, m->cfg.hidden_dim,
                            m->cfg.shared_intermediate, m->cfg.group_size);
    }

    float shared_weight = cpu_sigmoid(shared_gate_score);
    for (int i = 0; i < m->cfg.hidden_dim; i++) {
        shared_out[i] *= shared_weight;
    }

    // ---- Combine: hidden = h_mid + moe_out + shared_out ----
    for (int i = 0; i < m->cfg.hidden_dim; i++) {
        hidden[i] = h_mid[i] + moe_out[i] + shared_out[i];
    }

    if (moe_debug) {
        fprintf(stderr, "[MOE-DBG] layer=%d h_mid_rms=%.4f moe_rms=%.4f shared_rms=%.4f shared_gate=%.4f hidden_rms=%.4f\n",
                layer_idx, vec_rms(h_mid, m->cfg.hidden_dim), vec_rms(moe_out, m->cfg.hidden_dim),
                vec_rms(shared_out, m->cfg.hidden_dim), shared_weight,
                vec_rms(hidden, m->cfg.hidden_dim));
    }

    free(h_post);
    free(h_mid);
    free(gate_scores);
    free(moe_out);
    free(shared_out);
    free(shared_gate);
    free(shared_up);
    free(shared_act);
}

// Debug version of fused_layer_forward: simple CPU attention + synchronous MoE.
// Same signature as fused_layer_forward.
static void fused_layer_forward_debug(
    FlashMoE_Context *m,
    WeightFile *wf,
    int layer_idx,
    float *hidden,
    KVCache *kv,
    LinearAttnState *la_state,
    int pos,
    const void *mmap_base,
    int K,
    int packed_fd
) {
    (void)mmap_base;

    if (kv) {
        full_attention_forward(m, wf, layer_idx, hidden, kv, pos);
    } else if (la_state) {
        linear_attention_forward(m, wf, layer_idx, hidden, la_state);
    }

    moe_forward(m, wf, layer_idx, hidden, K, packed_fd);
}

static void fused_layer_forward(
    FlashMoE_Context *m,
    WeightFile *wf,
    int layer_idx,
    float *hidden,           // [m->cfg.hidden_dim] in/out
    KVCache *kv,             // non-NULL for full attention layers
    LinearAttnState *la_state, // non-NULL for linear attention layers
    int pos,                 // position for RoPE
    const void *mmap_base,   // mmap'd layer file (NULL if not available)
    int K,                   // number of active experts
    int packed_fd            // fd for packed expert file
) {
    double t_layer_start = 0, t0 = 0, t1 = 0;
    if (m->timing_enabled) { t_layer_start = now_ms(); }
    int pred_started = 0;  // set to 1 if we started prediction preads during CMD1_wait

    if (!m->layer_cache_built) build_layer_cache(m, wf);
    LayerWeightCache *lc = &m->layer_cache[layer_idx];
    int is_full = (kv != NULL);

    // =====================================================================
    // PHASE 1: Deferred completion + CMD1 (attention projections)
    // =====================================================================

    // ---- Prepare attention projection specs (doesn't depend on hidden) ----
    int num_attn_specs = 0;
    BatchMatvecSpec attn_specs[5];
    float *q_proj_out = NULL, *k_out = NULL, *v_out = NULL;
    float *qkv_out = NULL, *z_out = NULL, *beta_out = NULL, *alpha_out = NULL;

    if (is_full) {
        int q_proj_dim = m->cfg.num_attn_heads * m->cfg.head_dim * 2;
        int kv_dim = m->cfg.num_kv_heads * m->cfg.head_dim;

        q_proj_out = m->s_q_proj_out;
        k_out = m->s_k_proj_out;
        v_out = m->s_v_proj_out;

        if (lc->q_w && lc->q_s && lc->q_b && lc->k_w && lc->k_s && lc->k_b &&
            lc->v_w && lc->v_s && lc->v_b) {
            attn_specs[0] = (BatchMatvecSpec){ lc->q_w, lc->q_s, lc->q_b, q_proj_out, (uint32_t)q_proj_dim, m->cfg.hidden_dim, m->cfg.group_size, 0 };
            attn_specs[1] = (BatchMatvecSpec){ lc->k_w, lc->k_s, lc->k_b, k_out,      (uint32_t)kv_dim,     m->cfg.hidden_dim, m->cfg.group_size, 1 };
            attn_specs[2] = (BatchMatvecSpec){ lc->v_w, lc->v_s, lc->v_b, v_out,      (uint32_t)kv_dim,     m->cfg.hidden_dim, m->cfg.group_size, 2 };
            num_attn_specs = 3;
        }
    } else {
        int qkv_dim = m->cfg.linear_conv_dim;
        int z_dim = m->cfg.linear_total_value;

        qkv_out = m->s_qkv_proj_out;
        z_out = m->s_z_proj_out;
        beta_out = m->s_beta_proj_out;
        alpha_out = m->s_alpha_proj_out;

        if (lc->qkv_w && lc->qkv_s && lc->qkv_b && lc->z_w && lc->z_s && lc->z_b &&
            lc->b_w && lc->b_s && lc->b_b && lc->a_w && lc->a_s && lc->a_b) {
            attn_specs[0] = (BatchMatvecSpec){ lc->qkv_w, lc->qkv_s, lc->qkv_b, qkv_out,   (uint32_t)qkv_dim,            m->cfg.hidden_dim, m->cfg.group_size, 0 };
            attn_specs[1] = (BatchMatvecSpec){ lc->z_w,   lc->z_s,   lc->z_b,   z_out,      (uint32_t)z_dim,              m->cfg.hidden_dim, m->cfg.group_size, 1 };
            attn_specs[2] = (BatchMatvecSpec){ lc->b_w,   lc->b_s,   lc->b_b,   beta_out,   (uint32_t)m->cfg.linear_num_v_heads, m->cfg.hidden_dim, m->cfg.group_size, 2 };
            attn_specs[3] = (BatchMatvecSpec){ lc->a_w,   lc->a_s,   lc->a_b,   alpha_out,  (uint32_t)m->cfg.linear_num_v_heads, m->cfg.hidden_dim, m->cfg.group_size, 3 };
            num_attn_specs = 4;
        }
    }

    // ---- Deferred completion + CMD1 (sequential) ----
    float *normed = m->s_normed;
    float *residual = m->s_residual;
    id<MTLCommandBuffer> cmd1 = nil;
    int gpu_linear_attn = 0;  // set to 1 if GPU handles entire linear attention pipeline

    // Pre-compute linear_layer_idx for GPU linear attention encoding in CMD1
    int linear_layer_idx = -1;
    if (!is_full) {
        linear_layer_idx = layer_idx - (layer_idx + 1) / m->cfg.full_attn_interval;
    }
    // Can we run the full linear attention pipeline on GPU in CMD1?
    int can_gpu_linear = (m->gpu_linear_attn_enabled &&
                          !is_full && m->metal && m->metal->delta_net_step &&
                          m->metal->conv1d_step && m->metal->rms_norm_qk &&
                          m->metal->compute_decay_beta && m->metal->gated_rms_norm &&
                          m->metal->wf_buf &&
                          linear_layer_idx >= 0 && linear_layer_idx < m->cfg.num_linear_layers &&
                          lc->conv1d_w && lc->A_log && lc->dt_bias && lc->gated_norm_w &&
                          !m->linear_attn_bypass);

    // Check if previous layer's CMD3 already computed combine+residual+norm on GPU.
    // If so, buf_input already contains the normalized input for this layer's CMD1.
    // We can submit CMD1 immediately — the GPU queue serializes CMD3(N-1) then CMD1(N).
    int prev_gpu_combined = (m->deferred.active && m->deferred.gpu_combined);

    if (prev_gpu_combined && m->metal && m->metal->wf_buf && num_attn_specs > 0) {
        // ---- FAST PATH: GPU-combined previous CMD3 ----
        // buf_input already has the normalized hidden state from CMD3(N-1).
        // Submit CMD1 immediately — GPU runs CMD3(N-1) then CMD1(N) back-to-back.
        if (m->timing_enabled) { t0 = now_ms(); }

        cmd1 = [m->metal->queue commandBuffer];
        gpu_encode_batch_matvec(m->metal, cmd1, attn_specs, num_attn_specs);

        // GPU linear attention: encode conv1d + normalize + decay/beta + delta-net + gated_norm into CMD1
        if (can_gpu_linear && num_attn_specs == 4) {
            // batch_out[0]=qkv(12288), [1]=z(8192), [2]=beta(64), [3]=alpha(64)
            uint32_t conv_dim = m->cfg.linear_conv_dim;
            NSUInteger conv_w_off = (NSUInteger)((const char *)lc->conv1d_w - (const char *)[m->metal->wf_buf contents]);

            // Enc L1: conv1d_step — input=batch_out[0], weights=conv1d_w, state=buf_conv_state, output=buf_conv_output
            {
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:m->metal->conv1d_step];
                [enc setBuffer:m->metal->buf_conv_state[linear_layer_idx] offset:0 atIndex:0];
                [enc setBuffer:m->metal->batch_out[0]    offset:0            atIndex:1]; // qkv projection output
                [enc setBuffer:m->metal->wf_buf          offset:conv_w_off   atIndex:2]; // conv weights (bf16)
                [enc setBuffer:m->metal->buf_conv_output offset:0            atIndex:3]; // conv output
                [enc setBytes:&conv_dim length:4 atIndex:4];
                uint32_t tgs = (conv_dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            // Enc L2: rms_norm_qk — normalize q and k in conv_output in-place
            {
                uint32_t key_dim = m->cfg.linear_key_dim;  // 128
                float inv_scale = 1.0f / sqrtf((float)m->cfg.linear_key_dim);
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:m->metal->rms_norm_qk];
                [enc setBuffer:m->metal->buf_conv_output offset:0 atIndex:0];  // q at offset 0
                [enc setBuffer:m->metal->buf_conv_output offset:m->cfg.linear_total_key * sizeof(float) atIndex:1];  // k at offset 2048 floats
                [enc setBytes:&key_dim   length:4 atIndex:2];
                [enc setBytes:&inv_scale length:4 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(m->cfg.linear_num_k_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(m->cfg.linear_key_dim, 1, 1)];
                [enc endEncoding];
            }

            // Enc L3: compute_decay_beta — alpha=batch_out[3], beta=batch_out[2], A_log+dt_bias from wf_buf
            {
                NSUInteger a_log_off   = (NSUInteger)((const char *)lc->A_log   - (const char *)[m->metal->wf_buf contents]);
                NSUInteger dt_bias_off = (NSUInteger)((const char *)lc->dt_bias  - (const char *)[m->metal->wf_buf contents]);
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:m->metal->compute_decay_beta];
                [enc setBuffer:m->metal->batch_out[3]       offset:0          atIndex:0]; // alpha
                [enc setBuffer:m->metal->batch_out[2]       offset:0          atIndex:1]; // beta
                [enc setBuffer:m->metal->wf_buf             offset:a_log_off  atIndex:2]; // A_log
                [enc setBuffer:m->metal->wf_buf             offset:dt_bias_off atIndex:3]; // dt_bias (bf16)
                [enc setBuffer:m->metal->buf_delta_g_decay  offset:0          atIndex:4]; // g_decay output
                [enc setBuffer:m->metal->buf_delta_beta     offset:0          atIndex:5]; // beta_gate output
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(m->cfg.linear_num_v_heads, 1, 1)];
                [enc endEncoding];
            }

            // Enc L4: gated_delta_net_step — the main recurrence
            {
                uint32_t khpv = m->cfg.linear_num_v_heads / m->cfg.linear_num_k_heads;  // 4
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:m->metal->delta_net_step];
                [enc setBuffer:m->metal->buf_delta_state[linear_layer_idx] offset:0 atIndex:0]; // persistent state
                [enc setBuffer:m->metal->buf_conv_output offset:0 atIndex:1]; // q (first 2048 floats)
                [enc setBuffer:m->metal->buf_conv_output offset:m->cfg.linear_total_key * sizeof(float) atIndex:2]; // k (next 2048)
                [enc setBuffer:m->metal->buf_conv_output offset:2 * m->cfg.linear_total_key * sizeof(float) atIndex:3]; // v (next 8192)
                [enc setBuffer:m->metal->buf_delta_g_decay offset:0 atIndex:4];
                [enc setBuffer:m->metal->buf_delta_beta    offset:0 atIndex:5];
                [enc setBuffer:m->metal->buf_delta_output  offset:0 atIndex:6]; // output [8192]
                [enc setBytes:&khpv length:sizeof(khpv) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(m->cfg.linear_num_v_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding];
            }

            // Enc L5: gated_rms_norm — normalize+gate delta-net output -> batch_out[6] for CMD2 o_proj
            {
                NSUInteger gnorm_w_off = (NSUInteger)((const char *)lc->gated_norm_w - (const char *)[m->metal->wf_buf contents]);
                uint32_t value_dim = m->cfg.linear_value_dim;  // 128
                float eps = m->cfg.rms_norm_eps;
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:m->metal->gated_rms_norm];
                [enc setBuffer:m->metal->buf_delta_output offset:0          atIndex:0]; // values [8192]
                [enc setBuffer:m->metal->batch_out[1]     offset:0          atIndex:1]; // z (z projection output) [8192]
                [enc setBuffer:m->metal->wf_buf           offset:gnorm_w_off atIndex:2]; // weight (bf16)
                [enc setBuffer:m->metal->batch_out[6]     offset:0          atIndex:3]; // output -> batch_out[6] for CMD2
                [enc setBytes:&value_dim length:4 atIndex:4];
                [enc setBytes:&eps       length:4 atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(m->cfg.linear_num_v_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(m->cfg.linear_value_dim, 1, 1)];
                [enc endEncoding];
            }

            gpu_linear_attn = 1;
        }

        [cmd1 commit];

        if (m->timing_enabled) { t1 = now_ms(); m->timing.cmd1_submit += t1 - t0; }

        // Wait for CMD1 (implies CMD3(N-1) also done, since queue is serial)
        if (m->timing_enabled) { t0 = now_ms(); }
        [cmd1 waitUntilCompleted];
        if (!gpu_linear_attn) {
            gpu_flush_batch_results(m->metal, attn_specs, num_attn_specs);
        }
        if (m->timing_enabled) { t1 = now_ms(); m->timing.cmd1_wait += t1 - t0; }

        // Now CMD3(N-1) is done. Read back hidden state from GPU.
        if (m->timing_enabled) { t0 = now_ms(); }
        finalize_deferred_experts(m);  // reads buf_moe_hidden -> hidden

        // Start predicted expert preads AFTER CMD1_wait.
        // CMD3(N-1) is guaranteed done (serial queue), so buf_B is safe to overwrite.
        // Predictions overlap with CPU attn + CMD2 + routing (~0.6ms head start).
        // Predicted experts that hit page cache (same as previous token) complete in ~0.1ms.
        if (m->pred_enabled && m->pred_generating && m->pred_valid && packed_fd >= 0 &&
            m->metal->buf_multi_expert_data_B[0] && m->pred_count[layer_idx] > 0) {
            async_pread_start(m, packed_fd, &m->pred_experts[(layer_idx) * MAX_K],
                              m->pred_count[layer_idx],
                              m->metal->buf_multi_expert_data_B, mmap_base);
            pred_started = 1;
        }
        // Set up residual for CMD2 (residual = hidden before this layer's attention)
        cpu_vec_copy(residual, hidden, m->cfg.hidden_dim);
        if (m->timing_enabled) { t1 = now_ms(); m->timing.deferred_cpu += t1 - t0; }

        // No input_norm needed — CMD3 already computed it into buf_input.
        // normed is only needed if speculative routing is enabled (currently disabled).
        // Skip the readback to avoid unnecessary overhead.
    } else {
        // ---- ORIGINAL PATH: CPU deferred completion + input norm ----
        // Complete deferred experts from previous layer
        if (m->timing_enabled) { t0 = now_ms(); }
        wait_deferred_experts_gpu(m);
        if (m->timing_enabled) { t1 = now_ms(); m->timing.deferred_wait += t1 - t0; }

        if (m->timing_enabled) { t0 = now_ms(); }
        finalize_deferred_experts(m);
        if (m->timing_enabled) { t1 = now_ms(); m->timing.deferred_cpu += t1 - t0; }

        // Input norm
        if (m->timing_enabled) { t0 = now_ms(); }
        cpu_vec_copy(residual, hidden, m->cfg.hidden_dim);
        cpu_rms_norm(hidden, lc->input_norm_w, normed, m->cfg.hidden_dim, m->cfg.rms_norm_eps);
        if (m->timing_enabled) { t1 = now_ms(); m->timing.input_norm += t1 - t0; }

        // Submit CMD1: attention projections
        if (m->timing_enabled) { t0 = now_ms(); }
        if (m->metal && m->metal->wf_buf && num_attn_specs > 0) {
            memcpy([m->metal->buf_input contents], normed, m->cfg.hidden_dim * sizeof(float));
            cmd1 = [m->metal->queue commandBuffer];
            gpu_encode_batch_matvec(m->metal, cmd1, attn_specs, num_attn_specs);

            // GPU linear attention: encode conv1d + normalize + decay/beta + delta-net + gated_norm into CMD1
            if (can_gpu_linear && num_attn_specs == 4) {
                uint32_t conv_dim = m->cfg.linear_conv_dim;
                NSUInteger conv_w_off = (NSUInteger)((const char *)lc->conv1d_w - (const char *)[m->metal->wf_buf contents]);

                // Enc L1: conv1d_step
                {
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:m->metal->conv1d_step];
                    [enc setBuffer:m->metal->buf_conv_state[linear_layer_idx] offset:0 atIndex:0];
                    [enc setBuffer:m->metal->batch_out[0]    offset:0            atIndex:1];
                    [enc setBuffer:m->metal->wf_buf          offset:conv_w_off   atIndex:2];
                    [enc setBuffer:m->metal->buf_conv_output offset:0            atIndex:3];
                    [enc setBytes:&conv_dim length:4 atIndex:4];
                    uint32_t tgs = (conv_dim + 255) / 256;
                    [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }

                // Enc L2: rms_norm_qk
                {
                    uint32_t key_dim = m->cfg.linear_key_dim;
                    float inv_scale = 1.0f / sqrtf((float)m->cfg.linear_key_dim);
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:m->metal->rms_norm_qk];
                    [enc setBuffer:m->metal->buf_conv_output offset:0 atIndex:0];
                    [enc setBuffer:m->metal->buf_conv_output offset:m->cfg.linear_total_key * sizeof(float) atIndex:1];
                    [enc setBytes:&key_dim   length:4 atIndex:2];
                    [enc setBytes:&inv_scale length:4 atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(m->cfg.linear_num_k_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(m->cfg.linear_key_dim, 1, 1)];
                    [enc endEncoding];
                }

                // Enc L3: compute_decay_beta
                {
                    NSUInteger a_log_off   = (NSUInteger)((const char *)lc->A_log   - (const char *)[m->metal->wf_buf contents]);
                    NSUInteger dt_bias_off = (NSUInteger)((const char *)lc->dt_bias  - (const char *)[m->metal->wf_buf contents]);
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:m->metal->compute_decay_beta];
                    [enc setBuffer:m->metal->batch_out[3]       offset:0          atIndex:0];
                    [enc setBuffer:m->metal->batch_out[2]       offset:0          atIndex:1];
                    [enc setBuffer:m->metal->wf_buf             offset:a_log_off  atIndex:2];
                    [enc setBuffer:m->metal->wf_buf             offset:dt_bias_off atIndex:3];
                    [enc setBuffer:m->metal->buf_delta_g_decay  offset:0          atIndex:4];
                    [enc setBuffer:m->metal->buf_delta_beta     offset:0          atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(m->cfg.linear_num_v_heads, 1, 1)];
                    [enc endEncoding];
                }

                // Enc L4: gated_delta_net_step
                {
                    uint32_t khpv = m->cfg.linear_num_v_heads / m->cfg.linear_num_k_heads;
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:m->metal->delta_net_step];
                    [enc setBuffer:m->metal->buf_delta_state[linear_layer_idx] offset:0 atIndex:0];
                    [enc setBuffer:m->metal->buf_conv_output offset:0 atIndex:1];
                    [enc setBuffer:m->metal->buf_conv_output offset:m->cfg.linear_total_key * sizeof(float) atIndex:2];
                    [enc setBuffer:m->metal->buf_conv_output offset:2 * m->cfg.linear_total_key * sizeof(float) atIndex:3];
                    [enc setBuffer:m->metal->buf_delta_g_decay offset:0 atIndex:4];
                    [enc setBuffer:m->metal->buf_delta_beta    offset:0 atIndex:5];
                    [enc setBuffer:m->metal->buf_delta_output  offset:0 atIndex:6];
                    [enc setBytes:&khpv length:sizeof(khpv) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(m->cfg.linear_num_v_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [enc endEncoding];
                }

                // Enc L5: gated_rms_norm -> batch_out[6]
                {
                    NSUInteger gnorm_w_off = (NSUInteger)((const char *)lc->gated_norm_w - (const char *)[m->metal->wf_buf contents]);
                    uint32_t value_dim = m->cfg.linear_value_dim;
                    float eps = m->cfg.rms_norm_eps;
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:m->metal->gated_rms_norm];
                    [enc setBuffer:m->metal->buf_delta_output offset:0          atIndex:0];
                    [enc setBuffer:m->metal->batch_out[1]     offset:0          atIndex:1];
                    [enc setBuffer:m->metal->wf_buf           offset:gnorm_w_off atIndex:2];
                    [enc setBuffer:m->metal->batch_out[6]     offset:0          atIndex:3];
                    [enc setBytes:&value_dim length:4 atIndex:4];
                    [enc setBytes:&eps       length:4 atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(m->cfg.linear_num_v_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(m->cfg.linear_value_dim, 1, 1)];
                    [enc endEncoding];
                }

                gpu_linear_attn = 1;
            }

            [cmd1 commit];
        } else {
            for (int i = 0; i < num_attn_specs; i++) {
                BatchMatvecSpec *s = &attn_specs[i];
                cpu_dequant_matvec(s->W, s->scales, s->biases, normed, s->out_cpu,
                                   s->out_dim, s->in_dim, s->group_size);
            }
        }
        if (m->timing_enabled) { t1 = now_ms(); m->timing.cmd1_submit += t1 - t0; }

        // Wait for CMD1
        if (m->timing_enabled) { t0 = now_ms(); }
        if (cmd1) {
            [cmd1 waitUntilCompleted];
            if (!gpu_linear_attn) {
                gpu_flush_batch_results(m->metal, attn_specs, num_attn_specs);
            }
        }
        if (m->timing_enabled) { t1 = now_ms(); m->timing.cmd1_wait += t1 - t0; }
    }

    // =====================================================================
    // SPECULATIVE EARLY ROUTING — overlap expert I/O with CPU attention
    // =====================================================================
    // Compute approximate routing using the PRE-attention normed hidden state.
    // The real routing (in CMD2/PHASE 3) uses the POST-attention state, so this
    // is an approximation. Fire off async pread for predicted cache misses via
    // dispatch_group so the I/O runs concurrently with CPU attention compute.
    // After CPU attention, we wait for the group to finish. When the real routing
    // happens later, predicted experts are already in the LRU cache as hits.

    dispatch_group_t spec_group = NULL;
    int spec_preload_count = 0;
    int spec_routing_enabled = 0;  // DISABLED: cache pollution + overhead makes it slower

    if (m->timing_enabled) { t0 = now_ms(); }
    m->s_spec_count = 0;

    if (spec_routing_enabled && (m->expert_cache || m->malloc_cache) && packed_fd >= 0 && lc->gate_w) {
        float *spec_scores = m->s_spec_gate_scores;
        memset(spec_scores, 0, m->cfg.num_experts * sizeof(float));

        // Gate projection matvec on pre-attention normed input (CPU, ~0.1ms for 512x4096)
        cpu_dequant_matvec(lc->gate_w, lc->gate_s, lc->gate_b,
                           normed, spec_scores,
                           m->cfg.num_experts, m->cfg.hidden_dim, m->cfg.group_size);
        cpu_softmax(spec_scores, m->cfg.num_experts);

        int spec_K = (K > MAX_K) ? MAX_K : K;
        float spec_weights[MAX_K];
        cpu_topk(spec_scores, m->cfg.num_experts, spec_K, m->s_spec_indices, spec_weights);
        m->s_spec_count = spec_K;

        m->spec_route_attempts += spec_K;

        // Initialize GCD queue if needed
        if (!m->io_gcd_queue)
            m->io_gcd_queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

        // Check cache for each predicted expert, start async I/O for misses
        size_t spec_esz = active_expert_size(m);
        if (m->malloc_cache) {
            spec_group = dispatch_group_create();
            for (int k = 0; k < spec_K; k++) {
                int eidx = m->s_spec_indices[k];
                id<MTLBuffer> cached = malloc_cache_lookup(m, m->malloc_cache, layer_idx, eidx);
                if (!cached) {
                    int cidx = -1;
                    id<MTLBuffer> buf = malloc_cache_insert(m, m->malloc_cache, layer_idx, eidx, &cidx);
                    if (buf && cidx >= 0) {
                        int fd_copy = packed_fd;
                        void *dst = m->malloc_cache->data[cidx];
                        off_t offset = (off_t)eidx * spec_esz;
                        size_t sz = spec_esz;
                        dispatch_group_async(spec_group, m->io_gcd_queue, ^{
                            pread(fd_copy, dst, sz, offset);
                        });
                        spec_preload_count++;
                        m->spec_route_preloads++;
                    }
                }
            }
        } else if (m->expert_cache) {
            spec_group = dispatch_group_create();
            for (int k = 0; k < spec_K; k++) {
                int eidx = m->s_spec_indices[k];
                id<MTLBuffer> cached = expert_cache_lookup(m, m->expert_cache, layer_idx, eidx);
                if (!cached) {
                    id<MTLBuffer> buf = expert_cache_insert(m, m->expert_cache, layer_idx, eidx);
                    if (buf) {
                        int fd_copy = packed_fd;
                        void *dst = [buf contents];
                        off_t offset = (off_t)eidx * spec_esz;
                        size_t sz = spec_esz;
                        dispatch_group_async(spec_group, m->io_gcd_queue, ^{
                            pread(fd_copy, dst, sz, offset);
                        });
                        spec_preload_count++;
                        m->spec_route_preloads++;
                    }
                }
            }
        }
    }
    (void)spec_preload_count;  // tracked via m->spec_route_preloads

    if (m->timing_enabled) { t1 = now_ms(); m->timing.spec_route += t1 - t0; }

    // =====================================================================
    // PHASE 2: CPU attention compute
    // =====================================================================

    if (m->timing_enabled) { t0 = now_ms(); }

    float *attn_projected = m->s_attn_proj;
    memset(attn_projected, 0, m->cfg.hidden_dim * sizeof(float));

    // Pre-lookup o_proj / out_proj weights (used after attention compute)
    // These are looked up NOW to avoid repeated snprintf later.
    uint32_t *oproj_w = NULL;
    uint16_t *oproj_s = NULL, *oproj_b = NULL;
    int oproj_in_dim = 0;

    if (is_full) {
        oproj_w = lc->o_w; oproj_s = lc->o_s; oproj_b = lc->o_b;
        oproj_in_dim = m->cfg.num_attn_heads * m->cfg.head_dim;
    } else if (!m->linear_attn_bypass) {
        oproj_w = lc->out_proj_w; oproj_s = lc->out_proj_s; oproj_b = lc->out_proj_b;
        oproj_in_dim = m->cfg.linear_total_value;
    }

    // All MoE weight pointers from cache (zero snprintf overhead)
    uint32_t *gate_w = lc->gate_w; uint16_t *gate_s = lc->gate_s, *gate_b = lc->gate_b;
    uint32_t *sgw = lc->sg_w;     uint16_t *sgs = lc->sg_s,       *sgb = lc->sg_b;
    uint32_t *suw = lc->su_w;     uint16_t *sus = lc->su_s,       *sub = lc->su_b;
    uint32_t *seg_w = lc->seg_w;  uint16_t *seg_s = lc->seg_s,   *seg_b = lc->seg_b;
    uint32_t *sdw = lc->sd_w;     uint16_t *sds = lc->sd_s,       *sdb = lc->sd_b;

    // ---- CPU attention compute (produces attn_out for o_proj) ----
    float *attn_out_for_oproj = NULL;

    if (is_full) {
        // ---- Full attention CPU compute ----
        int q_proj_dim = m->cfg.num_attn_heads * m->cfg.head_dim * 2;
        int q_dim = m->cfg.num_attn_heads * m->cfg.head_dim;
        int kv_dim = m->cfg.num_kv_heads * m->cfg.head_dim;
        (void)q_proj_dim;

        float *q = m->s_q;
        float *q_gate = m->s_q_gate;
        for (int h = 0; h < m->cfg.num_attn_heads; h++) {
            float *src = q_proj_out + h * (2 * m->cfg.head_dim);
            memcpy(q + h * m->cfg.head_dim, src, m->cfg.head_dim * sizeof(float));
            memcpy(q_gate + h * m->cfg.head_dim, src + m->cfg.head_dim, m->cfg.head_dim * sizeof(float));
        }

        // Q/K RMSNorm
        uint16_t *qnorm_w = lc->q_norm_w;
        uint16_t *knorm_w = lc->k_norm_w;
        if (qnorm_w) {
            for (int h = 0; h < m->cfg.num_attn_heads; h++) {
                float *qh = q + h * m->cfg.head_dim;
                float sum_sq = 0.0f;
                for (int i = 0; i < m->cfg.head_dim; i++) sum_sq += qh[i] * qh[i];
                float inv_rms = 1.0f / sqrtf(sum_sq / m->cfg.head_dim + m->cfg.rms_norm_eps);
                for (int i = 0; i < m->cfg.head_dim; i++) qh[i] = qh[i] * inv_rms * bf16_to_f32(qnorm_w[i]);
            }
        }
        if (knorm_w) {
            for (int h = 0; h < m->cfg.num_kv_heads; h++) {
                float *kh = k_out + h * m->cfg.head_dim;
                float sum_sq = 0.0f;
                for (int i = 0; i < m->cfg.head_dim; i++) sum_sq += kh[i] * kh[i];
                float inv_rms = 1.0f / sqrtf(sum_sq / m->cfg.head_dim + m->cfg.rms_norm_eps);
                for (int i = 0; i < m->cfg.head_dim; i++) kh[i] = kh[i] * inv_rms * bf16_to_f32(knorm_w[i]);
            }
        }

        // RoPE
        apply_rotary_emb(q, k_out, pos, m->cfg.num_attn_heads, m->cfg.num_kv_heads, m->cfg.head_dim, m->cfg.rotary_dim, m->cfg.rope_theta);

        // Update KV cache (CPU + GPU mirror)
        int cache_pos = kv->len;
        for (int i = 0; i < kv_dim; i++) {
            kv->k_cache[cache_pos * kv_dim + i] = f32_to_kv_elem(k_out[i]);
            kv->v_cache[cache_pos * kv_dim + i] = f32_to_kv_elem(v_out[i]);
        }

        int fa_idx = (layer_idx + 1) / m->cfg.full_attn_interval - 1;
        if (m->metal && m->metal->attn_scores_pipe && fa_idx >= 0 && fa_idx < m->cfg.num_full_attn_layers) {
            memcpy((kv_elem_t *)[m->metal->buf_kv_k[fa_idx] contents] + cache_pos * kv_dim,
                   kv->k_cache + cache_pos * kv_dim, kv_dim * KV_ELEM_SIZE);
            memcpy((kv_elem_t *)[m->metal->buf_kv_v[fa_idx] contents] + cache_pos * kv_dim,
                   kv->v_cache + cache_pos * kv_dim, kv_dim * KV_ELEM_SIZE);
        }
        kv->len++;

        // Scaled dot-product attention (GQA) — GPU or CPU
        int heads_per_kv = m->cfg.num_attn_heads / m->cfg.num_kv_heads;
        float scale = 1.0f / sqrtf((float)m->cfg.head_dim);
        float *attn_out = m->s_attn_out;
        memset(attn_out, 0, q_dim * sizeof(float));

        // GPU attention: defer dispatches to CMD2 (fused into single cmd buffer).
        // Only enabled when seq_len >= 32 (below that, CPU is faster).
        int gpu_attn_ready = (m->metal && m->metal->attn_scores_pipe &&
                              fa_idx >= 0 && fa_idx < m->cfg.num_full_attn_layers &&
                              kv->len >= 32 && kv->len < m->cfg.gpu_kv_seq);

        if (gpu_attn_ready) {
            // Copy Q and gate to GPU; attention dispatches will be in CMD2
            memcpy([m->metal->buf_attn_q contents], q, q_dim * sizeof(float));
            memcpy([m->metal->buf_attn_gate contents], q_gate, q_dim * sizeof(float));
            // attn_out_for_oproj will be set to NULL below — CMD2 reads buf_attn_out
        } else {
            // CPU fallback
            for (int h = 0; h < m->cfg.num_attn_heads; h++) {
                int kv_h = h / heads_per_kv;
                float *qh = q + h * m->cfg.head_dim;
                float *scores = malloc(kv->len * sizeof(float));
                for (int p = 0; p < kv->len; p++) {
                    kv_elem_t *kp = kv->k_cache + p * kv_dim + kv_h * m->cfg.head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < m->cfg.head_dim; d++) {
                        dot += qh[d] * kv_elem_to_f32(kp[d]);
                    }
                    scores[p] = dot * scale;
                }
                cpu_softmax(scores, kv->len);
                float *oh = attn_out + h * m->cfg.head_dim;
                for (int p = 0; p < kv->len; p++) {
                    kv_elem_t *vp = kv->v_cache + p * kv_dim + kv_h * m->cfg.head_dim;
                    for (int d = 0; d < m->cfg.head_dim; d++) {
                        oh[d] += scores[p] * kv_elem_to_f32(vp[d]);
                    }
                }
                free(scores);
            }
            for (int i = 0; i < q_dim; i++) {
                float g = 1.0f / (1.0f + expf(-q_gate[i]));
                attn_out[i] *= g;
            }
        }

        if (gpu_attn_ready) {
            attn_out_for_oproj = NULL;  // signal CMD2 to use GPU buf_attn_out
        } else {
            attn_out_for_oproj = attn_out;
        }
        // q_proj_out, k_out, v_out, q, q_gate, attn_out are static scratch.
    } else if (gpu_linear_attn) {
        // ---- GPU linear attention: already computed in CMD1 ----
        // batch_out[6] already contains gated_rms_norm output (8192 floats)
        // Set a non-NULL sentinel so CMD2 enters fused path, but skip the memcpy
        static float gpu_linear_sentinel;
        attn_out_for_oproj = &gpu_linear_sentinel;
    } else {
        // ---- Linear attention CPU compute ----
        if (!m->linear_attn_bypass) {
            int qkv_dim = m->cfg.linear_conv_dim;

            // Conv1d step
            uint16_t *conv_w = lc->conv1d_w;
            float *conv_out = m->s_conv_out;
            memset(conv_out, 0, qkv_dim * sizeof(float));
            if (conv_w) {
                cpu_conv1d_step(la_state->conv_state, qkv_out, conv_w, conv_out,
                                qkv_dim, m->cfg.conv_kernel_size);
            }
            // Update conv state
            memmove(la_state->conv_state, la_state->conv_state + qkv_dim,
                    (m->cfg.conv_kernel_size - 2) * qkv_dim * sizeof(float));
            memcpy(la_state->conv_state + (m->cfg.conv_kernel_size - 2) * qkv_dim, qkv_out,
                   qkv_dim * sizeof(float));

            // Split into q, k, v
            float *lin_q = conv_out;
            float *lin_k = conv_out + m->cfg.linear_total_key;
            float *lin_v = conv_out + 2 * m->cfg.linear_total_key;

            // RMS normalize q and k
            float inv_scale = 1.0f / sqrtf((float)m->cfg.linear_key_dim);
            for (int h = 0; h < m->cfg.linear_num_k_heads; h++) {
                float *qh = lin_q + h * m->cfg.linear_key_dim;
                cpu_rms_norm_bare(qh, qh, m->cfg.linear_key_dim, 1e-6f);
                float q_scale = inv_scale * inv_scale;
                for (int d = 0; d < m->cfg.linear_key_dim; d++) qh[d] *= q_scale;
            }
            for (int h = 0; h < m->cfg.linear_num_k_heads; h++) {
                float *kh = lin_k + h * m->cfg.linear_key_dim;
                cpu_rms_norm_bare(kh, kh, m->cfg.linear_key_dim, 1e-6f);
                for (int d = 0; d < m->cfg.linear_key_dim; d++) kh[d] *= inv_scale;
            }

            // Gated delta net recurrence
            float *A_log = lc->A_log;
            uint16_t *dt_bias_bf16 = lc->dt_bias;

            float *out_values = m->s_out_vals;
            memset(out_values, 0, m->cfg.linear_total_value * sizeof(float));
            int k_heads_per_v = m->cfg.linear_num_v_heads / m->cfg.linear_num_k_heads;

            float g_decay[m->cfg.linear_num_v_heads];
            float beta_gate_arr[m->cfg.linear_num_v_heads];
            for (int vh = 0; vh < m->cfg.linear_num_v_heads; vh++) {
                float a_val = alpha_out[vh];
                float dt_b = dt_bias_bf16 ? bf16_to_f32(dt_bias_bf16[vh]) : 0.0f;
                float A_val = A_log ? expf(A_log[vh]) : 1.0f;
                float softplus_val = logf(1.0f + expf(a_val + dt_b));
                g_decay[vh] = expf(-A_val * softplus_val);
                beta_gate_arr[vh] = cpu_sigmoid(beta_out[vh]);
            }

            // Compute linear_layer_idx: count of non-full-attention layers before this one.
            // Full attention at (layer_idx+1) % 4 == 0, i.e. layers 3,7,11,...
            // linear_layer_idx = layer_idx - number_of_full_layers_at_or_before
            //                  = layer_idx - (layer_idx + 1) / m->cfg.full_attn_interval
            int linear_layer_idx = layer_idx - (layer_idx + 1) / m->cfg.full_attn_interval;

            // GPU delta-net path (falls back to CPU if pipeline unavailable)
            if (m->metal && m->metal->delta_net_step &&
                linear_layer_idx >= 0 && linear_layer_idx < m->cfg.num_linear_layers) {
                // Upload CPU-computed data to GPU scratch buffers
                memcpy([m->metal->buf_delta_q contents], lin_q, m->cfg.linear_total_key * sizeof(float));
                memcpy([m->metal->buf_delta_k contents], lin_k, m->cfg.linear_total_key * sizeof(float));
                memcpy([m->metal->buf_delta_v contents], lin_v, m->cfg.linear_total_value * sizeof(float));
                memcpy([m->metal->buf_delta_g_decay contents], g_decay, m->cfg.linear_num_v_heads * sizeof(float));
                memcpy([m->metal->buf_delta_beta contents], beta_gate_arr, m->cfg.linear_num_v_heads * sizeof(float));

                id<MTLCommandBuffer> cmd_dn = [m->metal->queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd_dn computeCommandEncoder];
                [enc setComputePipelineState:m->metal->delta_net_step];
                [enc setBuffer:m->metal->buf_delta_state[linear_layer_idx] offset:0 atIndex:0];
                [enc setBuffer:m->metal->buf_delta_q       offset:0 atIndex:1];
                [enc setBuffer:m->metal->buf_delta_k       offset:0 atIndex:2];
                [enc setBuffer:m->metal->buf_delta_v       offset:0 atIndex:3];
                [enc setBuffer:m->metal->buf_delta_g_decay offset:0 atIndex:4];
                [enc setBuffer:m->metal->buf_delta_beta    offset:0 atIndex:5];
                [enc setBuffer:m->metal->buf_delta_output  offset:0 atIndex:6];
                uint32_t khpv = (uint32_t)k_heads_per_v;
                [enc setBytes:&khpv length:sizeof(khpv) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(m->cfg.linear_num_v_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding];
                [cmd_dn commit];
                [cmd_dn waitUntilCompleted];

                // Read back GPU result
                memcpy(out_values, [m->metal->buf_delta_output contents], m->cfg.linear_total_value * sizeof(float));
            } else {
                // CPU delta-net with Accelerate BLAS
                for (int vh = 0; vh < m->cfg.linear_num_v_heads; vh++) {
                    int kh = vh / k_heads_per_v;
                    float g = g_decay[vh];
                    float b_gate = beta_gate_arr[vh];
                    float *S = la_state->ssm_state + vh * m->cfg.linear_value_dim * m->cfg.linear_key_dim;
                    float *v_h = lin_v + vh * m->cfg.linear_value_dim;
                    float *k_h = lin_k + kh * m->cfg.linear_key_dim;

                    // Step 1: Decay S *= g (BLAS sscal on entire state matrix)
                    cblas_sscal(m->cfg.linear_value_dim * m->cfg.linear_key_dim, g, S, 1);

                    // Step 2: kv_mem = S @ k (each row dot k)
                    // S is [VALUE_DIM x KEY_DIM] row-major, k is [KEY_DIM]
                    // kv_mem[vi] = sum_ki(S[vi,ki] * k[ki]) = matrix-vector: S @ k
                    float kv_mem_vec[m->cfg.linear_value_dim];
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                m->cfg.linear_value_dim, m->cfg.linear_key_dim,
                                1.0f, S, m->cfg.linear_key_dim, k_h, 1,
                                0.0f, kv_mem_vec, 1);

                    // Step 3: delta = (v - kv_mem) * beta, then rank-1 update S += k * delta^T
                    // delta[vi] = (v[vi] - kv_mem[vi]) * beta
                    float delta_vec[m->cfg.linear_value_dim];
                    for (int vi = 0; vi < m->cfg.linear_value_dim; vi++) {
                        delta_vec[vi] = (v_h[vi] - kv_mem_vec[vi]) * b_gate;
                    }
                    // S += delta @ k^T (rank-1 update: sger)
                    // S[vi,ki] += delta[vi] * k[ki]
                    cblas_sger(CblasRowMajor, m->cfg.linear_value_dim, m->cfg.linear_key_dim,
                               1.0f, delta_vec, 1, k_h, 1, S, m->cfg.linear_key_dim);

                    // Step 4: output = S @ q (matrix-vector multiply)
                    float *q_h = lin_q + kh * m->cfg.linear_key_dim;
                    float *o_h = out_values + vh * m->cfg.linear_value_dim;
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                m->cfg.linear_value_dim, m->cfg.linear_key_dim,
                                1.0f, S, m->cfg.linear_key_dim, q_h, 1,
                                0.0f, o_h, 1);
                }
            }

            // RMSNormGated
            uint16_t *gated_norm_w = lc->gated_norm_w;
            float *gated_out = m->s_gated_out;
            memset(gated_out, 0, m->cfg.linear_total_value * sizeof(float));
            for (int vh = 0; vh < m->cfg.linear_num_v_heads; vh++) {
                float *oh = out_values + vh * m->cfg.linear_value_dim;
                float *zh = z_out + vh * m->cfg.linear_value_dim;
                float *gh = gated_out + vh * m->cfg.linear_value_dim;
                if (gated_norm_w) {
                    cpu_rms_norm_gated(oh, zh, gated_norm_w, gh, m->cfg.linear_value_dim, m->cfg.rms_norm_eps);
                } else {
                    memcpy(gh, oh, m->cfg.linear_value_dim * sizeof(float));
                }
            }

            attn_out_for_oproj = gated_out;

            // conv_out, out_values are static — no free needed
            // gated_out is static — freed/released after CMD2 submission below
        }
        // else: m->linear_attn_bypass — attn_projected stays zero
        // qkv_out, z_out, beta_out, alpha_out are static scratch.
    }

    // =====================================================================
    // PHASE 3: FULLY FUSED CMD2 — o_proj + residual + norm + routing (1 cmd buffer)
    //   Eliminates 1 GPU round-trip vs old 2-buffer approach.
    //   GPU handles residual_add + rms_norm between o_proj and routing,
    //   so no CPU intervention is needed. 8 encoders, 1 commit+wait.
    //   Buffer flow: batch_out[6]->buf_output->buf_h_mid->buf_input->batch_out[0-3]
    // =====================================================================

    if (m->timing_enabled) { t1 = now_ms(); m->timing.cpu_attn += t1 - t0; }

    // Wait for speculative expert I/O to complete (overlapped with CPU attention)
    if (spec_group) {
        dispatch_group_wait(spec_group, DISPATCH_TIME_FOREVER);
        spec_group = NULL;  // ARC releases the group
    }

    if (m->timing_enabled) { t0 = now_ms(); }

    float *h_post = m->s_h_post;
    float *h_mid = m->s_h_mid;
    float *gate_scores = m->s_gate_scores;
    memset(gate_scores, 0, m->cfg.num_experts * sizeof(float));
    float *shared_gate = m->s_shared_gate;
    memset(shared_gate, 0, m->cfg.shared_intermediate * sizeof(float));
    float *shared_up = m->s_shared_up;
    memset(shared_up, 0, m->cfg.shared_intermediate * sizeof(float));
    float shared_gate_score = 0.0f;

    int have_moe_weights = (gate_w && gate_s && gate_b && sgw && sgs && sgb &&
                            suw && sus && sub && seg_w && seg_s && seg_b);

    // gpu_attn_fuse: attention dispatches fused into CMD2 (full-attn layers only).
    // Only enabled when seq_len >= 32 — below that, CPU attention is faster
    // because GPU command encoder overhead dominates at short sequences.
    int gpu_attn_fuse = (is_full && !attn_out_for_oproj && m->metal && m->metal->attn_scores_pipe
                         && kv && kv->len >= 32 && kv->len < m->cfg.gpu_kv_seq);

    if ((attn_out_for_oproj || gpu_attn_fuse) && oproj_w && oproj_s && oproj_b &&
        m->metal && m->metal->wf_buf && have_moe_weights &&
        m->metal->residual_add && m->metal->rms_norm_sum &&
        m->metal->rms_norm_apply_bf16 && lc->post_attn_norm_w) {
        // ---- FULLY FUSED CMD2 ----
        // For GPU attention (full-attn layers): attention dispatches are prepended,
        //   o_proj reads from buf_attn_out instead of batch_out[6].
        // For CPU attention / linear attn: o_proj reads from batch_out[6] as before.
        //
        // GPU attn path (12 encoders):
        //   Enc 1-4: attn_scores + softmax + values + sigmoid -> buf_attn_out
        //   Enc 5:   o_proj (buf_attn_out -> buf_output)
        //   Enc 6-8: residual + norm -> buf_input
        //   Enc 9-12: routing + shared expert
        //
        // CPU attn path (8 encoders, unchanged):
        //   Enc 1:   o_proj (batch_out[6] -> buf_output)
        //   Enc 2-4: residual + norm -> buf_input
        //   Enc 5-8: routing + shared expert

        if (!gpu_attn_fuse && !gpu_linear_attn) {
            // CPU/linear attn: copy attention output to GPU input buffer
            memcpy([m->metal->batch_out[6] contents], attn_out_for_oproj,
                   oproj_in_dim * sizeof(float));
        }
        // gpu_linear_attn: batch_out[6] already has the result from CMD1 gated_rms_norm
        // Copy residual into GPU buffer for residual_add kernel
        memcpy([m->metal->buf_residual contents], residual, m->cfg.hidden_dim * sizeof(float));

        attn_out_for_oproj = NULL;

        id<MTLCommandBuffer> cmd_fused = [m->metal->queue commandBuffer];

        // ---- GPU attention dispatches (only for full-attn layers with GPU path) ----
        if (gpu_attn_fuse) {
            int fa_idx = (layer_idx + 1) / m->cfg.full_attn_interval - 1;
            int kv_dim = m->cfg.num_kv_heads * m->cfg.head_dim;
            int heads_per_kv = m->cfg.num_attn_heads / m->cfg.num_kv_heads;
            float scale = 1.0f / sqrtf((float)m->cfg.head_dim);
            uint32_t hd = m->cfg.head_dim;
            uint32_t kvd = (uint32_t)kv_dim;
            uint32_t sl = (uint32_t)kv->len;
            uint32_t seq_stride = m->cfg.gpu_kv_seq;
            uint32_t hpkv = (uint32_t)heads_per_kv;

            // Enc A1: attn_scores_batched
            {
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:m->metal->attn_scores_pipe];
                [enc setBuffer:m->metal->buf_attn_q          offset:0 atIndex:0];
                [enc setBuffer:m->metal->buf_kv_k[fa_idx]    offset:0 atIndex:1];
                [enc setBuffer:m->metal->buf_attn_scores     offset:0 atIndex:2];
                [enc setBytes:&hd        length:4 atIndex:3];
                [enc setBytes:&kvd       length:4 atIndex:4];
                [enc setBytes:&sl        length:4 atIndex:5];
                [enc setBytes:&seq_stride length:4 atIndex:6];
                [enc setBytes:&scale     length:4 atIndex:7];
                [enc setBytes:&hpkv      length:4 atIndex:8];
                [enc setBytes:&sl        length:4 atIndex:9];
                uint32_t total_tgs = sl * m->cfg.num_attn_heads;
                [enc dispatchThreadgroups:MTLSizeMake(total_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // Enc A2: attn_softmax_batched
            {
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:m->metal->attn_softmax_pipe];
                [enc setBuffer:m->metal->buf_attn_scores offset:0 atIndex:0];
                [enc setBytes:&sl         length:4 atIndex:1];
                [enc setBytes:&seq_stride  length:4 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(m->cfg.num_attn_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // Enc A3: attn_values_batched
            {
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:m->metal->attn_values_pipe];
                [enc setBuffer:m->metal->buf_attn_scores   offset:0 atIndex:0];
                [enc setBuffer:m->metal->buf_kv_v[fa_idx]  offset:0 atIndex:1];
                [enc setBuffer:m->metal->buf_attn_out      offset:0 atIndex:2];
                [enc setBytes:&hd        length:4 atIndex:3];
                [enc setBytes:&kvd       length:4 atIndex:4];
                [enc setBytes:&sl        length:4 atIndex:5];
                [enc setBytes:&seq_stride length:4 atIndex:6];
                [enc setBytes:&hpkv      length:4 atIndex:7];
                uint32_t total_threads = m->cfg.head_dim * m->cfg.num_attn_heads;
                uint32_t tgs = (total_threads + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // Enc A4: sigmoid_gate
            {
                uint32_t qdim = m->cfg.num_attn_heads * m->cfg.head_dim;
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:m->metal->sigmoid_gate_pipe];
                [enc setBuffer:m->metal->buf_attn_out  offset:0 atIndex:0];
                [enc setBuffer:m->metal->buf_attn_gate offset:0 atIndex:1];
                [enc setBytes:&qdim length:4 atIndex:2];
                uint32_t tgs = (qdim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
        }

        // ---- o_proj matvec ----
        {
            NSUInteger w_off = (NSUInteger)((const char *)oproj_w - (const char *)[m->metal->wf_buf contents]);
            NSUInteger s_off = (NSUInteger)((const char *)oproj_s - (const char *)[m->metal->wf_buf contents]);
            NSUInteger b_off = (NSUInteger)((const char *)oproj_b - (const char *)[m->metal->wf_buf contents]);

            // For GPU attention: o_proj reads from buf_attn_out
            // For CPU attention: o_proj reads from batch_out[6]
            id<MTLBuffer> oproj_input = gpu_attn_fuse ? m->metal->buf_attn_out : m->metal->batch_out[6];

            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            uint32_t o_out_dim = m->cfg.hidden_dim;
            uint32_t o_in_dim = (uint32_t)oproj_in_dim;
            uint32_t o_gs = m->cfg.group_size;
            [enc setComputePipelineState:m->metal->matvec_fast];
            [enc setBuffer:m->metal->wf_buf  offset:w_off atIndex:0];
            [enc setBuffer:m->metal->wf_buf  offset:s_off atIndex:1];
            [enc setBuffer:m->metal->wf_buf  offset:b_off atIndex:2];
            [enc setBuffer:oproj_input      offset:0    atIndex:3];
            [enc setBuffer:m->metal->buf_output offset:0 atIndex:4];
            [enc setBytes:&o_out_dim  length:4 atIndex:5];
            [enc setBytes:&o_in_dim   length:4 atIndex:6];
            [enc setBytes:&o_gs       length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(o_out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
        }

        // ---- Enc 2: residual_add (buf_output + buf_residual -> buf_h_mid) ----
        {
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            uint32_t dim = m->cfg.hidden_dim;
            [enc setComputePipelineState:m->metal->residual_add];
            [enc setBuffer:m->metal->buf_residual offset:0 atIndex:0];  // a = residual
            [enc setBuffer:m->metal->buf_output   offset:0 atIndex:1];  // b = o_proj result
            [enc setBuffer:m->metal->buf_h_mid    offset:0 atIndex:2];  // out = h_mid
            [enc setBytes:&dim length:4 atIndex:3];
            uint32_t tgs = (dim + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // ---- Enc 3: rms_norm_sum_sq (buf_h_mid -> buf_sum_sq) ----
        {
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            uint32_t dim = m->cfg.hidden_dim;
            [enc setComputePipelineState:m->metal->rms_norm_sum];
            [enc setBuffer:m->metal->buf_h_mid  offset:0 atIndex:0];
            [enc setBuffer:m->metal->buf_sum_sq offset:0 atIndex:1];
            [enc setBytes:&dim length:4 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // ---- Enc 4: rms_norm_apply_bf16 (buf_h_mid + norm_w -> buf_input) ----
        {
            NSUInteger norm_off = (NSUInteger)((const char *)lc->post_attn_norm_w -
                                               (const char *)[m->metal->wf_buf contents]);
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            uint32_t dim = m->cfg.hidden_dim;
            float eps = m->cfg.rms_norm_eps;
            [enc setComputePipelineState:m->metal->rms_norm_apply_bf16];
            [enc setBuffer:m->metal->buf_h_mid  offset:0       atIndex:0];  // x
            [enc setBuffer:m->metal->wf_buf     offset:norm_off atIndex:1]; // weight (bf16)
            [enc setBuffer:m->metal->buf_sum_sq offset:0       atIndex:2];  // sum_sq
            [enc setBuffer:m->metal->buf_input  offset:0       atIndex:3];  // out = h_post
            [enc setBytes:&dim length:4 atIndex:4];
            [enc setBytes:&eps length:4 atIndex:5];
            uint32_t tgs = (dim + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // ---- Enc 5-8: routing + shared expert projections (read buf_input) ----
        BatchMatvecSpec moe_specs[4] = {
            { gate_w, gate_s, gate_b, gate_scores,        (uint32_t)m->cfg.num_experts,        m->cfg.hidden_dim, m->cfg.group_size, 0 },
            { sgw,    sgs,    sgb,    shared_gate,         (uint32_t)m->cfg.shared_intermediate, m->cfg.hidden_dim, m->cfg.group_size, 1 },
            { suw,    sus,    sub,    shared_up,           (uint32_t)m->cfg.shared_intermediate, m->cfg.hidden_dim, m->cfg.group_size, 2 },
            { seg_w,  seg_s,  seg_b,  &shared_gate_score,  1,                            m->cfg.hidden_dim, m->cfg.group_size, 3 },
        };
        // buf_input already contains h_post from Enc 4 output -- no memcpy needed
        gpu_encode_batch_matvec(m->metal, cmd_fused, moe_specs, 4);

        if (m->timing_enabled) { t1 = now_ms(); m->timing.cmd2_encode += t1 - t0; }

        // ---- Single commit+wait for all 8 encoders ----
        if (m->timing_enabled) { t0 = now_ms(); }
        [cmd_fused commit];
        [cmd_fused waitUntilCompleted];

        // Read back results
        gpu_flush_batch_results(m->metal, moe_specs, 4);
        // Read h_mid from GPU buffer (needed for final combine)
        memcpy(h_mid, [m->metal->buf_h_mid contents], m->cfg.hidden_dim * sizeof(float));
        // Read h_post from buf_input (needed for expert input)
        memcpy(h_post, [m->metal->buf_input contents], m->cfg.hidden_dim * sizeof(float));
        // Update hidden state to h_mid (= residual + o_proj)
        memcpy(hidden, h_mid, m->cfg.hidden_dim * sizeof(float));
        if (m->timing_enabled) { t1 = now_ms(); m->timing.cmd2_wait += t1 - t0; }

    } else {
        // ---- Non-fused fallback path ----
        // O projection
        if (attn_out_for_oproj && oproj_w && oproj_s && oproj_b) {
            fast_dequant_matvec(m, oproj_w, oproj_s, oproj_b, attn_out_for_oproj,
                                attn_projected, m->cfg.hidden_dim, oproj_in_dim, m->cfg.group_size);
        }
        // attn_out_for_oproj is static — no free needed
        attn_out_for_oproj = NULL;

        // Residual connection
        for (int i = 0; i < m->cfg.hidden_dim; i++) {
            hidden[i] = residual[i] + attn_projected[i];
        }
        // attn_projected, normed, residual are static — no free needed

        cpu_vec_copy(h_mid, hidden, m->cfg.hidden_dim);

        // Post-attention norm
        cpu_rms_norm(hidden, lc->post_attn_norm_w, h_post, m->cfg.hidden_dim, m->cfg.rms_norm_eps);

        // Routing + shared expert batch
        if (have_moe_weights) {
            BatchMatvecSpec moe_specs[4] = {
                { gate_w, gate_s, gate_b, gate_scores,        (uint32_t)m->cfg.num_experts,        m->cfg.hidden_dim, m->cfg.group_size, 0 },
                { sgw,    sgs,    sgb,    shared_gate,         (uint32_t)m->cfg.shared_intermediate, m->cfg.hidden_dim, m->cfg.group_size, 1 },
                { suw,    sus,    sub,    shared_up,           (uint32_t)m->cfg.shared_intermediate, m->cfg.hidden_dim, m->cfg.group_size, 2 },
                { seg_w,  seg_s,  seg_b,  &shared_gate_score,  1,                            m->cfg.hidden_dim, m->cfg.group_size, 3 },
            };
            fast_batch_matvec(m, h_post, m->cfg.hidden_dim, moe_specs, 4);
        }
        if (m->timing_enabled) { t1 = now_ms(); m->timing.cmd2_encode += t1 - t0; }
    }

    // ---- Softmax + top-K (CPU) ----
    if (m->timing_enabled) { t0 = now_ms(); }
    cpu_softmax(gate_scores, m->cfg.num_experts);
    int expert_indices[64];
    float expert_weights[64];
    cpu_topk(gate_scores, m->cfg.num_experts, K, expert_indices, expert_weights);
    cpu_normalize_weights(expert_weights, K);
    if (m->freq_tracking) {
        for (int k = 0; k < K; k++) {
            m->expert_freq[(layer_idx) * m->cfg.num_experts + (expert_indices[k])]++;
        }
        if (layer_idx == 0) m->freq_total_tokens++;
    }

    // Track speculative routing prediction accuracy
    if (m->s_spec_count > 0) {
        int cmp_K = (K > MAX_K) ? MAX_K : K;
        for (int s = 0; s < m->s_spec_count; s++) {
            for (int r = 0; r < cmp_K; r++) {
                if (m->s_spec_indices[s] == expert_indices[r]) {
                    m->spec_route_hits++;
                    break;
                }
            }
        }
    }

    if (m->timing_enabled) { t1 = now_ms(); m->timing.routing_cpu += t1 - t0; }

    // Log routing data for predictor training
    if (m->routing_log) {
        int32_t li = layer_idx;
        int32_t ki = (K > MAX_K) ? MAX_K : K;
        fwrite(&li, sizeof(int32_t), 1, m->routing_log);
        fwrite(&ki, sizeof(int32_t), 1, m->routing_log);
        fwrite(hidden, sizeof(float), m->cfg.hidden_dim, m->routing_log);
        fwrite(expert_indices, sizeof(int32_t), ki, m->routing_log);
        m->routing_log_samples++;
    }

    // ---- Parallel pread + GPU experts ----
    if (m->timing_enabled) { t0 = now_ms(); }
    float *moe_out = m->s_moe_out;
    memset(moe_out, 0, m->cfg.hidden_dim * sizeof(float));
    float *shared_out = m->s_shared_out;
    memset(shared_out, 0, m->cfg.hidden_dim * sizeof(float));

    int actual_K = (K > MAX_K) ? MAX_K : K;

    if (packed_fd >= 0 && m->metal && m->metal->buf_multi_expert_data[0]) {
        // GPU multi-expert path with LRU cache + parallel I/O:
        // For each expert:
        //   - Cache HIT:  dispatch directly from cached Metal buffer (skip pread)
        //   - Cache MISS: pread into cache buffer, then dispatch from it
        // Falls back to original parallel_pread_experts when cache is disabled.

        int valid[MAX_K];
        id<MTLBuffer> expert_bufs[MAX_K];  // buffer to dispatch from per expert

        if (m->malloc_cache) {
            // ---- Malloc cache path (zero-copy Metal buffer wrappers) ----
            // Phase 1: check cache for each expert, collect misses
            int miss_indices[MAX_K];
            int miss_cache_idx[MAX_K];  // cache entry index for each miss
            int num_misses = 0;

            for (int k = 0; k < actual_K; k++) {
                id<MTLBuffer> cached = malloc_cache_lookup(m, m->malloc_cache, layer_idx, expert_indices[k]);
                if (cached) {
                    // Cache hit: zero-copy dispatch directly from cache buffer
                    expert_bufs[k] = cached;
                    valid[k] = 1;
                } else {
                    // Cache miss: insert entry (get buffer to pread into)
                    int cidx = -1;
                    id<MTLBuffer> buf = malloc_cache_insert(m, m->malloc_cache, layer_idx, expert_indices[k], &cidx);
                    expert_bufs[k] = buf;
                    miss_indices[num_misses] = k;
                    miss_cache_idx[num_misses] = cidx;
                    num_misses++;
                    valid[k] = 0;
                }
            }

            // Phase 2: parallel pread misses directly into cache buffers (zero-copy)
            if (num_misses > 0) {
                size_t esz = active_expert_size(m);
                InferPreadTask tasks[MAX_K];
                for (int t = 0; t < num_misses; t++) {
                    int k = miss_indices[t];
                    int cidx = miss_cache_idx[t];
                    tasks[t].fd = expert_pick_fd(m, layer_idx, expert_indices[k], packed_fd);
                    tasks[t].dst = m->malloc_cache->data[cidx];
                    tasks[t].offset = (off_t)expert_indices[k] * esz;
                    tasks[t].size = esz;
                    tasks[t].result = 0;
                    tasks[t].mmap_base = NULL;  // always pread for cache population
                }

                io_pool_dispatch(m, tasks, num_misses);

                // Mark valid
                for (int t = 0; t < num_misses; t++) {
                    int k = miss_indices[t];
                    valid[k] = (tasks[t].result == (ssize_t)esz);
                    if (!valid[k]) {
                        fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                                expert_indices[k], tasks[t].result, esz);
                    }
                }
            }
        } else if (m->expert_cache) {
            // ---- Metal buffer LRU cache path ----
            // Phase 1: check cache for each expert, collect misses
            int miss_indices[MAX_K];       // indices into expert_indices[] for misses
            id<MTLBuffer> miss_bufs[MAX_K]; // cache buffers to pread into
            int num_misses = 0;

            for (int k = 0; k < actual_K; k++) {
                id<MTLBuffer> cached = expert_cache_lookup(m, m->expert_cache, layer_idx, expert_indices[k]);
                if (cached) {
                    // Cache hit: use this buffer directly for GPU dispatch
                    expert_bufs[k] = cached;
                    valid[k] = 1;
                } else {
                    // Cache miss: insert into cache (allocates or evicts), will pread below
                    id<MTLBuffer> buf = expert_cache_insert(m, m->expert_cache, layer_idx, expert_indices[k]);
                    if (buf) {
                        expert_bufs[k] = buf;
                        miss_indices[num_misses] = k;
                        miss_bufs[num_misses] = buf;
                        num_misses++;
                        valid[k] = 0;  // not yet loaded
                    } else {
                        expert_bufs[k] = nil;
                        valid[k] = 0;
                    }
                }
            }

            // Phase 2: parallel pread all cache misses
            if (num_misses > 0) {
                size_t esz = active_expert_size(m);
                InferPreadTask tasks[MAX_K];
                for (int t = 0; t < num_misses; t++) {
                    int k = miss_indices[t];
                    tasks[t].fd = expert_pick_fd(m, layer_idx, expert_indices[k], packed_fd);
                    tasks[t].dst = [miss_bufs[t] contents];
                    tasks[t].offset = (off_t)expert_indices[k] * esz;
                    tasks[t].size = esz;
                    tasks[t].result = 0;
                    tasks[t].mmap_base = mmap_base;
                }

                io_pool_dispatch(m, tasks, num_misses);

                // Mark successfully loaded misses as valid
                for (int t = 0; t < num_misses; t++) {
                    int k = miss_indices[t];
                    valid[k] = (tasks[t].result == (ssize_t)esz);
                    if (!valid[k]) {
                        fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                                expert_indices[k], tasks[t].result, esz);
                    }
                }
            }
        } else if (pred_started) {
            // ---- Prediction path: predicted experts already loading into buf_B ----
            // Wait for predicted preads (they've had ~1.6ms: CMD1_wait + attn + CMD2 + routing)
            async_pread_wait(m);
            m->pred_layers++;

            // Match predictions against actual routing
            int miss_ei[MAX_K];       // actual expert indices for misses
            int miss_k_slots[MAX_K];  // which k-slot each miss maps to
            int miss_count = 0;
            int hit_count = 0;

            for (int k = 0; k < actual_K; k++) {
                int found = 0;
                for (int p = 0; p < m->pred_count[layer_idx]; p++) {
                    if (expert_indices[k] == m->pred_experts[(layer_idx) * MAX_K + (p)] &&
                        m->async_pread.valid[p]) {
                        // Hit! This expert was pre-loaded into buf_B[p]
                        expert_bufs[k] = m->metal->buf_multi_expert_data_B[p];
                        valid[k] = 1;
                        found = 1;
                        hit_count++;
                        break;
                    }
                }
                if (!found) {
                    miss_ei[miss_count] = expert_indices[k];
                    miss_k_slots[miss_count] = k;
                    expert_bufs[k] = m->metal->buf_multi_expert_data[k];
                    miss_count++;
                }
            }
            m->pred_hits += hit_count;
            m->pred_misses += miss_count;

            // Parallel sync-pread misses into buf_A
            if (miss_count > 0) {
                InferPreadTask tasks[MAX_K];
                size_t esz = active_expert_size(m);
                for (int t = 0; t < miss_count; t++) {
                    int k = miss_k_slots[t];
                    tasks[t].fd = packed_fd;
                    tasks[t].dst = [m->metal->buf_multi_expert_data[k] contents];
                    tasks[t].offset = (off_t)miss_ei[t] * esz;
                    tasks[t].size = esz;
                    tasks[t].result = 0;
                }
                io_pool_dispatch(m, tasks, miss_count);
                for (int t = 0; t < miss_count; t++) {
                    int k = miss_k_slots[t];
                    valid[k] = (tasks[t].result == (ssize_t)active_expert_size(m));
                }
            }
        } else if (m->use_lz4 && m->lz4_index[layer_idx]) {
            // ---- LZ4 compressed path: read compressed + decompress via io_pool ----
            size_t esz = active_expert_size(m);
            InferPreadTask tasks[MAX_K];
            for (int k = 0; k < actual_K; k++) {
                LZ4IndexEntry *ie = &m->lz4_index[layer_idx][expert_indices[k]];
                tasks[k].fd = packed_fd;
                tasks[k].dst = [m->metal->buf_multi_expert_data[k] contents];
                tasks[k].offset = ie->offset;
                tasks[k].size = esz;
                tasks[k].result = 0;
                tasks[k].mmap_base = NULL;
                tasks[k].lz4_comp_buf = m->lz4_comp_bufs[k];
                tasks[k].lz4_comp_size = ie->comp_size;
                expert_bufs[k] = m->metal->buf_multi_expert_data[k];
            }
            io_pool_dispatch(m, tasks, actual_K);
            for (int k = 0; k < actual_K; k++) {
                valid[k] = (tasks[k].result == (ssize_t)esz);
            }
        } else {
            // ---- No cache, no prediction, no LZ4: ASYNC parallel pread ----
            async_pread_start(m, packed_fd, expert_indices, actual_K,
                              m->metal->buf_multi_expert_data, mmap_base);
            for (int k = 0; k < actual_K; k++) {
                expert_bufs[k] = m->metal->buf_multi_expert_data[k];
            }
        }

        // Shared expert prep (doesn't need expert data — can overlap with async pread)
        memcpy([m->metal->buf_multi_expert_input contents], h_post, m->cfg.hidden_dim * sizeof(float));
        memcpy([m->metal->buf_shared_gate contents], shared_gate,
               m->cfg.shared_intermediate * sizeof(float));
        memcpy([m->metal->buf_shared_up contents], shared_up,
               m->cfg.shared_intermediate * sizeof(float));

        // Wait for non-prediction async pread to complete
        if (!pred_started && m->async_pread.active) {
            async_pread_wait(m);
            for (int k = 0; k < actual_K; k++) {
                valid[k] = m->async_pread.valid[k];
            }
        }

        if (m->timing_enabled) { t1 = now_ms(); m->timing.expert_io += t1 - t0; }

        // Store this layer's routing for next token's temporal prediction.
        // MUST happen AFTER the prediction hit check above (which reads m->pred_experts).
        if (m->pred_enabled && m->pred_generating) {
            for (int k = 0; k < actual_K; k++) {
                m->pred_experts[(layer_idx) * MAX_K + (k)] = expert_indices[k];
            }
            m->pred_count[layer_idx] = actual_K;
            if (layer_idx == m->cfg.num_layers - 1) {
                m->pred_valid = 1;
            }
        }

        if (m->timing_enabled) { t0 = now_ms(); }

        // Step 3: encode ALL experts + shared expert into ONE command buffer.
        // Batched encoding: 4 encoders for K experts + 2 for shared = 6 total
        // (vs. 4*K + 2 = 18 with old per-expert encoding).
        id<MTLCommandBuffer> cmd_experts = [m->metal->queue commandBuffer];

        gpu_encode_experts_batched(m, m->metal, cmd_experts, actual_K, valid, expert_bufs);

        // Shared expert SwiGLU + down_proj (2 more encoders)
        // Note: shared_gate/up already copied to GPU buffers above (before async pread wait)

        // SwiGLU dispatch
        {
            id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
            [enc setComputePipelineState:m->metal->swiglu];
            [enc setBuffer:m->metal->buf_shared_gate offset:0 atIndex:0];
            [enc setBuffer:m->metal->buf_shared_up   offset:0 atIndex:1];
            [enc setBuffer:m->metal->buf_shared_act  offset:0 atIndex:2];
            uint32_t dim = m->cfg.shared_intermediate;
            [enc setBytes:&dim length:4 atIndex:3];
            uint32_t swiglu_tgs = (dim + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // Shared down_proj dispatch
        if (sdw && sds && sdb) {
            gpu_encode_dequant_matvec_with_io_bufs(
                m->metal, cmd_experts, sdw, sds, sdb,
                m->metal->buf_shared_act, m->metal->buf_shared_out,
                m->cfg.hidden_dim, m->cfg.shared_intermediate, m->cfg.group_size);
        }

        // Step 4: GPU-side combine + residual + norm (if not last layer)
        // Appends dispatches to CMD3 so the next layer's CMD1 can submit immediately
        // without waiting for CMD3 to complete + CPU readback.
        //
        // For non-last layers with the combine pipeline available:
        //   Enc C1: moe_combine_residual (expert_outs + h_mid + shared_out -> buf_moe_hidden)
        //   Enc C2: rms_norm_sum_sq (buf_moe_hidden -> buf_cmd3_sum_sq)
        //   Enc C3: rms_norm_apply_bf16 (buf_moe_hidden + next_layer_norm_w -> buf_input)
        //
        // This makes CMD3 self-contained: it produces buf_input for the next layer's CMD1.
        // The next layer skips deferred_wait + finalize + input_norm entirely at layer start.

        int gpu_combine = (m->metal->moe_combine_residual &&
                           m->metal->rms_norm_sum &&
                           m->metal->rms_norm_apply_bf16 &&
                           m->metal->wf_buf &&
                           layer_idx < m->cfg.num_layers - 1 &&
                           m->layer_cache[layer_idx + 1].input_norm_w != NULL);

        if (gpu_combine) {
            // Copy h_mid from buf_h_mid (populated by CMD2) — it's still valid on GPU.
            // h_mid is already in buf_h_mid from CMD2's residual_add dispatch.

            // Prepare combine params: expert_weights[0..K-1] + shared_gate_score
            {
                float *params = (float *)[m->metal->buf_combine_params contents];
                // Zero all 10 slots first (unused experts get weight=0)
                memset(params, 0, 10 * sizeof(float));
                for (int k = 0; k < actual_K; k++) {
                    params[k] = valid[k] ? expert_weights[k] : 0.0f;
                }
                params[8] = shared_gate_score;
            }

            // Enc C1: moe_combine_residual
            {
                id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
                [enc setComputePipelineState:m->metal->moe_combine_residual];
                [enc setBuffer:m->metal->buf_h_mid         offset:0 atIndex:0];   // h_mid
                [enc setBuffer:m->metal->buf_shared_out    offset:0 atIndex:1];   // shared_out
                [enc setBuffer:m->metal->buf_moe_hidden    offset:0 atIndex:2];   // output: hidden
                // Bind all 8 expert output buffers (unused ones have weight=0 in params)
                for (int k = 0; k < MAX_K; k++) {
                    [enc setBuffer:m->metal->buf_multi_expert_out[k] offset:0 atIndex:(3 + k)];
                }
                [enc setBuffer:m->metal->buf_combine_params offset:0 atIndex:11]; // params
                uint32_t dim = m->cfg.hidden_dim;
                uint32_t k_val = (uint32_t)actual_K;
                [enc setBytes:&dim   length:4 atIndex:12];
                [enc setBytes:&k_val length:4 atIndex:13];
                uint32_t tgs = (dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            // Enc C2: rms_norm_sum_sq (buf_moe_hidden -> buf_cmd3_sum_sq)
            {
                id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
                uint32_t dim = m->cfg.hidden_dim;
                [enc setComputePipelineState:m->metal->rms_norm_sum];
                [enc setBuffer:m->metal->buf_moe_hidden  offset:0 atIndex:0];
                [enc setBuffer:m->metal->buf_cmd3_sum_sq offset:0 atIndex:1];
                [enc setBytes:&dim length:4 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            // Enc C3: rms_norm_apply_bf16 (buf_moe_hidden + next_norm_w -> buf_input)
            {
                uint16_t *next_norm_w = m->layer_cache[layer_idx + 1].input_norm_w;
                NSUInteger norm_off = (NSUInteger)((const char *)next_norm_w -
                                                   (const char *)[m->metal->wf_buf contents]);
                id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
                uint32_t dim = m->cfg.hidden_dim;
                float eps = m->cfg.rms_norm_eps;
                [enc setComputePipelineState:m->metal->rms_norm_apply_bf16];
                [enc setBuffer:m->metal->buf_moe_hidden  offset:0       atIndex:0]; // x
                [enc setBuffer:m->metal->wf_buf          offset:norm_off atIndex:1]; // weight (bf16)
                [enc setBuffer:m->metal->buf_cmd3_sum_sq offset:0       atIndex:2]; // sum_sq
                [enc setBuffer:m->metal->buf_input       offset:0       atIndex:3]; // out = normed
                [enc setBytes:&dim length:4 atIndex:4];
                [enc setBytes:&eps length:4 atIndex:5];
                uint32_t tgs = (dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
        }

        // DEFERRED commit — submit async, don't wait.
#if USE_EVENT_PIPELINE
        // Signal shared event (llama.cpp pattern) for non-blocking CPU/GPU sync.
        m->metal->event_value++;
        [cmd_experts encodeSignalEvent:m->metal->pipeline_event value:m->metal->event_value];
#endif
        [cmd_experts commit];
        if (m->timing_enabled) {
            t1 = now_ms();
            m->timing.cmd3_encode += t1 - t0;
            m->timing.count++;
            m->timing.total += t1 - t_layer_start;
        }

        // Save state for deferred completion
        m->deferred.active = 1;
        m->deferred.gpu_combined = gpu_combine;
        m->deferred.cmd_experts = cmd_experts;
#if USE_EVENT_PIPELINE
        m->deferred.expert_event_value = m->metal->event_value;    // for non-blocking wait
#endif
        m->deferred.actual_K = actual_K;
        m->deferred.shared_gate_score = shared_gate_score;
        m->deferred.hidden = hidden;
        m->deferred.layer_idx = layer_idx;
        if (!gpu_combine) {
            // Only need to save h_mid for CPU-side combine path
            memcpy(m->deferred.h_mid, h_mid, m->cfg.hidden_dim * sizeof(float));
        }
        for (int k = 0; k < actual_K; k++) {
            m->deferred.expert_weights[k] = expert_weights[k];
            m->deferred.valid[k] = valid[k];
        }

        // Return immediately — GPU experts are running async.
        // The next call to fused_layer_forward() or complete_deferred_experts(m)
        // will wait for the GPU and apply the final combine.
        return;

    } else if (packed_fd >= 0) {
        // CPU fallback for experts
        size_t esz = active_expert_size(m);
        float *expert_out_cpu = malloc(m->cfg.hidden_dim * sizeof(float));
        for (int k = 0; k < K; k++) {
            int eidx = expert_indices[k];
            off_t expert_offset = (off_t)eidx * esz;
            void *expert_data = malloc(esz);
            ssize_t nread = pread(packed_fd, expert_data, esz, expert_offset);
            if (nread != (ssize_t)esz) {
                fprintf(stderr, "WARNING: layer %d expert %d pread: %zd/%zu\n",
                        layer_idx, eidx, nread, esz);
                free(expert_data);
                continue;
            }

            // CPU fallback offsets — use 4-bit layout (2-bit CPU path not yet implemented)
            uint32_t *gw = (uint32_t *)expert_data;
            uint16_t *gs_p = (uint16_t *)((char *)expert_data + (m->use_2bit ? m->cfg.layout_2bit.gate_s_off : m->cfg.layout_4bit.gate_s_off));
            uint16_t *gb_p = (uint16_t *)((char *)expert_data + (m->use_2bit ? m->cfg.layout_2bit.gate_b_off : m->cfg.layout_4bit.gate_b_off));
            uint32_t *uw = (uint32_t *)((char *)expert_data + (m->use_2bit ? m->cfg.layout_2bit.up_w_off : m->cfg.layout_4bit.up_w_off));
            uint16_t *us_p = (uint16_t *)((char *)expert_data + (m->use_2bit ? m->cfg.layout_2bit.up_s_off : m->cfg.layout_4bit.up_s_off));
            uint16_t *ub_p = (uint16_t *)((char *)expert_data + (m->use_2bit ? m->cfg.layout_2bit.up_b_off : m->cfg.layout_4bit.up_b_off));
            uint32_t *dw = (uint32_t *)((char *)expert_data + (m->use_2bit ? m->cfg.layout_2bit.down_w_off : m->cfg.layout_4bit.down_w_off));
            uint16_t *ds_p = (uint16_t *)((char *)expert_data + (m->use_2bit ? m->cfg.layout_2bit.down_s_off : m->cfg.layout_4bit.down_s_off));
            uint16_t *db_p = (uint16_t *)((char *)expert_data + (m->use_2bit ? m->cfg.layout_2bit.down_b_off : m->cfg.layout_4bit.down_b_off));

            float *gate_proj_out = malloc(m->cfg.moe_intermediate * sizeof(float));
            float *up_proj_out = malloc(m->cfg.moe_intermediate * sizeof(float));
            float *act_out = malloc(m->cfg.moe_intermediate * sizeof(float));

            cpu_dequant_matvec(gw, gs_p, gb_p, h_post, gate_proj_out,
                               m->cfg.moe_intermediate, m->cfg.hidden_dim, m->cfg.group_size);
            cpu_dequant_matvec(uw, us_p, ub_p, h_post, up_proj_out,
                               m->cfg.moe_intermediate, m->cfg.hidden_dim, m->cfg.group_size);
            cpu_swiglu(gate_proj_out, up_proj_out, act_out, m->cfg.moe_intermediate);
            cpu_dequant_matvec(dw, ds_p, db_p, act_out, expert_out_cpu,
                               m->cfg.hidden_dim, m->cfg.moe_intermediate, m->cfg.group_size);

            free(gate_proj_out);
            free(up_proj_out);
            free(act_out);
            free(expert_data);

            cpu_vec_madd(moe_out, expert_out_cpu, expert_weights[k], m->cfg.hidden_dim);
        }
        free(expert_out_cpu);

        // CPU shared expert
        float *shared_act = calloc(m->cfg.shared_intermediate, sizeof(float));
        cpu_swiglu(shared_gate, shared_up, shared_act, m->cfg.shared_intermediate);
        if (sdw && sds && sdb) {
            cpu_dequant_matvec(sdw, sds, sdb, shared_act, shared_out,
                               m->cfg.hidden_dim, m->cfg.shared_intermediate, m->cfg.group_size);
        }
        free(shared_act);
    } else {
        // No experts available -- still need shared expert
        float *shared_act = calloc(m->cfg.shared_intermediate, sizeof(float));
        cpu_swiglu(shared_gate, shared_up, shared_act, m->cfg.shared_intermediate);
        if (sdw && sds && sdb) {
            fast_dequant_matvec(m, sdw, sds, sdb, shared_act, shared_out,
                                m->cfg.hidden_dim, m->cfg.shared_intermediate, m->cfg.group_size);
        }
        free(shared_act);
    }

    // ---- Shared expert gate ----
    float shared_weight = cpu_sigmoid(shared_gate_score);
    for (int i = 0; i < m->cfg.hidden_dim; i++) {
        shared_out[i] *= shared_weight;
    }

    // ---- Final combine: hidden = h_mid + moe_out + shared_out ----
    for (int i = 0; i < m->cfg.hidden_dim; i++) {
        hidden[i] = h_mid[i] + moe_out[i] + shared_out[i];
    }

    if (m->timing_enabled) {
        t1 = now_ms();
        m->timing.cmd3_encode += t1 - t0;  // includes CPU expert compute for non-GPU paths
        m->timing.count++;
        m->timing.total += t1 - t_layer_start;
    }

    // h_post, h_mid, gate_scores, moe_out, shared_out, shared_gate, shared_up
    // are all static scratch buffers — no free needed.
}


#endif // LAYER_FORWARD_H
