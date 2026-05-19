#ifndef MODEL_INTERNAL_H
#define MODEL_INTERNAL_H

// ============================================================================
// Complete FlashMoE_Model struct — ALL inference state in one allocation.
// Opaque in the public C API (moe_infer_c.h), fully defined here.
// Every function that needs state takes FlashMoE_Model *m.
// ============================================================================

#include "model_types.h"

struct FlashMoE_Model {
    // ---- Model config (from model_config.h) ----
    ModelConfig cfg;

    // ---- Model path ----
    char model_path[1024];

    // ---- Weight file (from model_weights.h) ----
    WeightFile *wf;
    TensorHTEntry tensor_ht[TENSOR_HT_SIZE];
    int tensor_ht_built;

    // ---- Metal context (from metal_setup.h) ----
    MetalCtx *metal;

    // ---- Expert file I/O ----
    int   *layer_fds;
    void **layer_mmaps;
    size_t *layer_mmap_sizes;

    // ---- Working buffers ----
    float    *hidden;
    float    *logits;
    uint16_t *final_norm_w;
    int K;
    int initialized;

    // ---- Timing (from util.h) ----
    LayerTimingAccum timing;
    int timing_enabled;

    // ---- Temporal prediction pipeline (from util.h) ----
    int pred_enabled;
    int pred_generating;
    uint64_t pred_hits;
    uint64_t pred_misses;
    uint64_t pred_layers;
    int *pred_experts;
    int *pred_count;

    // ---- Routing data collection ----
    FILE *routing_log;
    int routing_log_samples;

    // ---- LZ4 compressed expert support ----
    LZ4IndexEntry **lz4_index;
    void *lz4_comp_bufs[8];
    int use_lz4;

    // ---- Expert frequency tracking ----
    int *expert_freq;
    int freq_tracking;
    int freq_total_tokens;

    // ---- Quantization / feature flags ----
    int use_2bit;
    int cache_telemetry_enabled;
    int think_budget;

    // ---- Tiered I/O ----
    int *layer_fds_cold;
    uint8_t *expert_seen;

    // ---- Cache telemetry ----
    CacheTelemetry cache_telemetry;
    uint8_t  *cache_seen;
    uint64_t *cache_last_touch_token;
    uint64_t *cache_last_evict_token;

    // ---- I/O thread pool (from expert_io.h) ----
    IOThreadPool io_pool;
    int io_pool_initialized;
    dispatch_queue_t io_gcd_queue;

    // ---- Async expert pread ----
    AsyncPreadState async_pread;

    // ---- Expert LRU cache ----
    ExpertLRUCache *expert_cache;

    // ---- Speculative routing stats ----
    uint64_t spec_route_attempts;
    uint64_t spec_route_hits;
    uint64_t spec_route_preloads;

    // ---- Temporal prediction state ----
    int pred_valid;

    // ---- Malloc-based expert cache ----
    MallocExpertCache *malloc_cache;

    // ---- Background prefetch thread ----
    InferPrefetchCtx *prefetch;
    pthread_t prefetch_tid;

    // ---- Attention debug / bypass (from attention.h) ----
    int fa_debug_count;
    int linear_attn_bypass;
    int gpu_linear_attn_enabled;

    // ---- Layer weight cache (from layer_forward.h) ----
    LayerWeightCache *layer_cache;
    int layer_cache_built;

    // ---- Deferred expert state ----
    DeferredExpertState deferred;

    // ---- Layer scratch buffers ----
    float *s_normed;
    float *s_residual;
    float *s_attn_proj;
    float *s_h_post;
    float *s_h_mid;
    float *s_gate_scores;
    float *s_spec_gate_scores;
    int s_spec_indices[8];
    int s_spec_count;
    float *s_shared_gate;
    float *s_shared_up;
    float *s_moe_out;
    float *s_shared_out;
    float *s_q_proj_out;
    float *s_k_proj_out;
    float *s_v_proj_out;
    float *s_q;
    float *s_q_gate;
    float *s_attn_out;
    float *s_qkv_proj_out;
    float *s_z_proj_out;
    float *s_beta_proj_out;
    float *s_alpha_proj_out;
    float *s_conv_out;
    float *s_out_vals;
    float *s_gated_out;
    int moe_sync_debug_count;
};

#endif // MODEL_INTERNAL_H
