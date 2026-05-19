#ifndef COMMON_H
#define COMMON_H

// ============================================================================
// All type definitions for the Flash-MoE inference engine.
// Sub-structs first, then the main FlashMoE_Context container at the end.
// Opaque in the public C API (moe_infer_c.h), fully defined here.
// Every function that needs state takes FlashMoE_Context *m.
// ============================================================================

// ============================================================================
// Consolidated type definitions for the Flash-MoE inference engine.
// Included by common.h before the FlashMoE_Context struct definition.
// Contains ONLY types — no static variables, no function definitions.
// ============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <math.h>
#include <getopt.h>
#include <pthread.h>
#include <errno.h>
#include <dispatch/dispatch.h>
#include <Accelerate/Accelerate.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/wait.h>
#include <compression.h>

#include "config.h"

// ---- Forward declarations for types defined later in this header ----
typedef struct FlashMoE_Context FlashMoE_Context;
typedef struct FlashMoE_Cache FlashMoE_Cache;
typedef struct WeightFile WeightFile;
typedef struct TensorInfo TensorInfo;
typedef struct TensorManifest TensorManifest;
typedef struct MetalCtx MetalCtx;
typedef struct ExpertLRUCache ExpertLRUCache;
typedef struct MallocExpertCache MallocExpertCache;
typedef struct InferPrefetchCtx InferPrefetchCtx;
typedef struct LayerWeightCache LayerWeightCache;

// ============================================================================
// From model_config.h — Model dimensions and expert layout
// ============================================================================

typedef struct {
    int gate_w_off, gate_s_off, gate_b_off;
    int up_w_off, up_s_off, up_b_off;
    int down_w_off, down_s_off, down_b_off;
    int gate_w_size, gate_s_size, gate_b_size;
    int up_w_size, up_s_size, up_b_size;
    int down_w_size, down_s_size, down_b_size;
} ExpertLayout;

typedef struct {
    int hidden_dim, num_layers, num_attn_heads, num_kv_heads;
    int vocab_size, num_experts, num_experts_per_tok;
    int moe_intermediate, shared_intermediate;
    int linear_num_v_heads, linear_num_k_heads;
    int rotary_dim, linear_total_key, linear_total_value, linear_conv_dim;
    int num_full_attn_layers, num_linear_layers;
    int expert_size_4bit, expert_size_2bit;
    ExpertLayout layout_4bit, layout_2bit;
    // Architectural constants — read from model_config.json at runtime
    int head_dim, group_size, full_attn_interval;
    int conv_kernel_size, max_seq_len, gpu_kv_seq;
    int max_k, linear_key_dim, linear_value_dim;
    float rms_norm_eps, rope_theta;
} ModelConfig;

// ============================================================================
// From util.h — Timing, cache telemetry, LZ4
// ============================================================================

typedef struct {
    double deferred_wait;
    double deferred_cpu;
    double input_norm;
    double cmd1_submit;
    double cmd1_wait;
    double cpu_attn;
    double cmd2_encode;
    double cmd2_wait;
    double routing_cpu;
    double spec_route;
    double expert_io;
    double cmd3_encode;
    double total;
    int count;
} LayerTimingAccum;

typedef struct {
    uint64_t offset;
    uint32_t comp_size;
    uint32_t raw_size;
} LZ4IndexEntry;

typedef struct {
    uint64_t token_clock;
    uint64_t unique_experts_touched;
    uint64_t cold_misses;
    uint64_t eviction_misses;
    uint64_t evictions;
    uint64_t reuse_le_1;
    uint64_t reuse_le_4;
    uint64_t reuse_le_16;
    uint64_t reuse_le_64;
    uint64_t reuse_gt_64;
    uint64_t reuse_distance_sum;
    uint64_t reuse_distance_samples;
} CacheTelemetry;

// ============================================================================
// From model_weights.h — Tensor manifest and weight file
// ============================================================================

struct TensorInfo {
    const char *name;
    size_t offset;
    size_t size;
    int ndim;
    int shape[4];
    char dtype[8];
};

struct TensorManifest {
    TensorInfo *tensors;
    int num_tensors;
    int capacity;
};

#define TENSOR_HT_SIZE 8192

typedef struct {
    const char *key;
    TensorInfo *value;
} TensorHTEntry;

struct WeightFile {
    void *data;
    size_t size;
    TensorManifest *manifest;
};

// ============================================================================
// From metal_setup.h — Metal context (type only, no static functions)
// ============================================================================

#define MAX_BATCH_SLOTS 8

struct MetalCtx {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLLibrary>              library;
    id<MTLComputePipelineState> matvec_v3;
    id<MTLComputePipelineState> matvec_v5;
    id<MTLComputePipelineState> matvec_fast;
    id<MTLComputePipelineState> matvec_2bit;
    id<MTLComputePipelineState> rms_norm_sum;
    id<MTLComputePipelineState> rms_norm_apply;
    id<MTLComputePipelineState> rms_norm_apply_bf16;
#if USE_FUSED_RMS_NORM
    id<MTLComputePipelineState> rms_norm_fused;
#endif
    id<MTLComputePipelineState> residual_add;
    id<MTLComputePipelineState> swiglu;
#if USE_FUSED_GATE_UP_SWIGLU
    id<MTLComputePipelineState> fused_gate_up_swiglu;
#endif
    id<MTLComputePipelineState> attn_scores_pipe;
    id<MTLComputePipelineState> attn_softmax_pipe;
    id<MTLComputePipelineState> attn_values_pipe;
    id<MTLComputePipelineState> sigmoid_gate_pipe;
    id<MTLBuffer> buf_input;
    id<MTLBuffer> buf_output;
    id<MTLBuffer> wf_buf;
    id<MTLBuffer> batch_out[MAX_BATCH_SLOTS];
    id<MTLBuffer> buf_expert_data;
    id<MTLBuffer> buf_expert_input;
    id<MTLBuffer> buf_expert_gate;
    id<MTLBuffer> buf_expert_up;
    id<MTLBuffer> buf_expert_act;
    id<MTLBuffer> buf_expert_out;
    id<MTLBuffer> buf_multi_expert_data[8];
    id<MTLBuffer> buf_multi_expert_data_B[8];
    id<MTLBuffer> buf_multi_expert_gate[8];
    id<MTLBuffer> buf_multi_expert_up[8];
    id<MTLBuffer> buf_multi_expert_act[8];
    id<MTLBuffer> buf_multi_expert_out[8];
    id<MTLBuffer> buf_multi_expert_input;
    id<MTLBuffer> buf_shared_gate;
    id<MTLBuffer> buf_shared_up;
    id<MTLBuffer> buf_shared_act;
    id<MTLBuffer> buf_shared_out;
    id<MTLBuffer> buf_residual;
    id<MTLBuffer> buf_h_mid;
    id<MTLBuffer> buf_sum_sq;
    __unsafe_unretained id<MTLBuffer> *buf_kv_k;
    __unsafe_unretained id<MTLBuffer> *buf_kv_v;
    id<MTLBuffer> buf_attn_q;
    id<MTLBuffer> buf_attn_scores;
    id<MTLBuffer> buf_attn_out;
    id<MTLBuffer> buf_attn_gate;
    id<MTLComputePipelineState> moe_combine_residual;
    id<MTLBuffer> buf_moe_hidden;
    id<MTLBuffer> buf_combine_params;
    id<MTLBuffer> buf_cmd3_sum_sq;
#if USE_EVENT_PIPELINE
    id<MTLSharedEvent> pipeline_event;
    uint64_t event_value;
#endif
    id<MTLComputePipelineState> delta_net_step;
    id<MTLComputePipelineState> conv1d_step;
    id<MTLComputePipelineState> rms_norm_qk;
    id<MTLComputePipelineState> compute_decay_beta;
    id<MTLComputePipelineState> gated_rms_norm;
    __unsafe_unretained id<MTLBuffer> *buf_delta_state;
    __unsafe_unretained id<MTLBuffer> *buf_conv_state;
    id<MTLBuffer> buf_delta_q;
    id<MTLBuffer> buf_delta_k;
    id<MTLBuffer> buf_delta_v;
    id<MTLBuffer> buf_delta_g_decay;
    id<MTLBuffer> buf_delta_beta;
    id<MTLBuffer> buf_delta_output;
    id<MTLBuffer> buf_conv_input;
    id<MTLBuffer> buf_conv_output;
};

// ============================================================================
// From attention.h — KV cache and linear attention state
// ============================================================================

#if USE_KV_CACHE_BF16
typedef uint16_t kv_elem_t;
#else
typedef float kv_elem_t;
#endif

typedef struct {
    kv_elem_t *k_cache;
    kv_elem_t *v_cache;
    int len;
} KVCache;

typedef struct {
    float *conv_state;
    float *ssm_state;
} LinearAttnState;

// ============================================================================
// From expert_io.h — I/O infrastructure types
// ============================================================================

#define NUM_IO_THREADS 4

typedef struct {
    int fd;
    void *dst;
    off_t offset;
    size_t size;
    ssize_t result;
    const void *mmap_base;
    void *lz4_comp_buf;
    uint32_t lz4_comp_size;
} InferPreadTask;

typedef struct {
    InferPreadTask *tasks;
    int num_tasks;
    int thread_id;
} InferPreadThreadArg;

typedef struct {
    pthread_t threads[NUM_IO_THREADS];
    pthread_mutex_t mutex;
    pthread_cond_t work_ready;
    pthread_cond_t work_done;
    InferPreadTask *tasks;
    int num_tasks;
    int tasks_completed;
    int generation;
    volatile int shutdown;
} IOThreadPool;

typedef struct {
    InferPreadTask tasks[8];
    int num_tasks;
    int valid[8];
    dispatch_group_t group;
    int active;
} AsyncPreadState;

typedef struct {
    int layer_idx;
    int expert_idx;
    id<MTLBuffer> buffer;
    uint64_t last_used;
} ExpertCacheEntry;

struct ExpertLRUCache {
    ExpertCacheEntry *entries;
    int max_entries;
    int num_entries;
    int used_entries;
    int *entry_idx;
    uint64_t access_counter;
    id<MTLDevice> device;
    uint64_t hits;
    uint64_t misses;
};

struct MallocExpertCache {
    void **data;
    id<MTLBuffer> __strong *metal_bufs;
    int *layer_idx;
    int *expert_idx;
    uint64_t *last_used;
    int max_entries;
    int num_entries;
    int used_entries;
    int *entry_idx;
    uint64_t access_counter;
    uint64_t hits;
    uint64_t misses;
};

typedef struct {
    void *dst[8];
    off_t offset[8];
    int K;
    int fd;
    int valid[8];
    int loaded;
} InferIOPlan;

struct InferPrefetchCtx {
    InferIOPlan plan;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int start;
    int done;
    int shutdown;
    IOThreadPool *io_pool;   // back-pointer for parallel I/O dispatch
    size_t expert_size;      // cached expert size for pread
};

// ============================================================================
// From layer_forward.h — Per-layer cache and deferred expert state
// ============================================================================

struct LayerWeightCache {
    uint16_t *input_norm_w;
    uint16_t *post_attn_norm_w;
    uint32_t *q_w; uint16_t *q_s, *q_b;
    uint32_t *k_w; uint16_t *k_s, *k_b;
    uint32_t *v_w; uint16_t *v_s, *v_b;
    uint32_t *o_w; uint16_t *o_s, *o_b;
    uint16_t *q_norm_w, *k_norm_w;
    uint32_t *qkv_w; uint16_t *qkv_s, *qkv_b;
    uint32_t *z_w;   uint16_t *z_s, *z_b;
    uint32_t *b_w;   uint16_t *b_s, *b_b;
    uint32_t *a_w;   uint16_t *a_s, *a_b;
    uint16_t *conv1d_w;
    float *A_log;
    uint16_t *dt_bias;
    uint16_t *gated_norm_w;
    uint32_t *out_proj_w; uint16_t *out_proj_s, *out_proj_b;
    uint32_t *gate_w; uint16_t *gate_s, *gate_b;
    uint32_t *sg_w;   uint16_t *sg_s, *sg_b;
    uint32_t *su_w;   uint16_t *su_s, *su_b;
    uint32_t *sd_w;   uint16_t *sd_s, *sd_b;
    uint32_t *seg_w;  uint16_t *seg_s, *seg_b;
};

typedef struct {
    int active;
    int gpu_combined;
    id<MTLCommandBuffer> cmd_experts;
#if USE_EVENT_PIPELINE
    uint64_t expert_event_value;
#endif
    float expert_weights[MAX_K];
    int valid[MAX_K];
    int actual_K;
    float *h_mid;
    float shared_gate_score;
    float *hidden;
    int layer_idx;
} DeferredExpertState;

// ============================================================================
// From gpu_ops.h — Batched matvec spec
// ============================================================================

typedef struct {
    const void *W;
    const void *scales;
    const void *biases;
    float *out_cpu;
    uint32_t out_dim;
    uint32_t in_dim;
    uint32_t group_size;
    int batch_slot;
} BatchMatvecSpec;

struct FlashMoE_Context {
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

#endif // COMMON_H
