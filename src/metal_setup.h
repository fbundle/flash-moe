#ifndef METAL_SETUP_H
#define METAL_SETUP_H

// ============================================================================
// Metal context for GPU-accelerated matmuls
// ============================================================================

// Maximum number of batched matmul output slots.
// Used for encoding multiple matmuls into one command buffer.
#define MAX_BATCH_SLOTS 8

typedef struct {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLLibrary>              library;
    id<MTLComputePipelineState> matvec_v3;
    id<MTLComputePipelineState> matvec_v5;  // LUT dequant variant
    id<MTLComputePipelineState> matvec_fast;  // for in_dim > 4096
    id<MTLComputePipelineState> matvec_2bit;  // 2-bit expert dequant kernel
    id<MTLComputePipelineState> rms_norm_sum;
    id<MTLComputePipelineState> rms_norm_apply;
    id<MTLComputePipelineState> rms_norm_apply_bf16;
#if USE_FUSED_RMS_NORM
    id<MTLComputePipelineState> rms_norm_fused;     // single-kernel fused RMS norm (llama.cpp pattern)
#endif
    id<MTLComputePipelineState> residual_add;
    id<MTLComputePipelineState> swiglu;
#if USE_FUSED_GATE_UP_SWIGLU
    id<MTLComputePipelineState> fused_gate_up_swiglu;  // fused gate+up+SwiGLU kernel
#endif
    // GPU attention pipelines
    id<MTLComputePipelineState> attn_scores_pipe;
    id<MTLComputePipelineState> attn_softmax_pipe;
    id<MTLComputePipelineState> attn_values_pipe;
    id<MTLComputePipelineState> sigmoid_gate_pipe;
    // Reusable buffers for attention matmuls
    id<MTLBuffer> buf_input;     // input vector [HIDDEN_DIM or max projection input]
    id<MTLBuffer> buf_output;    // output vector [max projection output]
    id<MTLBuffer> wf_buf;        // the mmap'd weight file as a Metal buffer
    // Batched matmul output slots (preallocated, reused across dispatches)
    id<MTLBuffer> batch_out[MAX_BATCH_SLOTS];
    // Reusable buffers for expert computation (avoids per-expert alloc)
    // Legacy single-expert buffers (kept for gpu_expert_forward compat)
    id<MTLBuffer> buf_expert_data;   // holds one expert's packed weights (EXPERT_SIZE bytes)
    id<MTLBuffer> buf_expert_input;  // h_post input [HIDDEN_DIM floats]
    id<MTLBuffer> buf_expert_gate;   // gate_proj output [MOE_INTERMEDIATE floats]
    id<MTLBuffer> buf_expert_up;     // up_proj output [MOE_INTERMEDIATE floats]
    id<MTLBuffer> buf_expert_act;    // SwiGLU output [MOE_INTERMEDIATE floats]
    id<MTLBuffer> buf_expert_out;    // down_proj output [HIDDEN_DIM floats]
    // Multi-expert buffers: K independent sets so all experts can be encoded
    // into a SINGLE command buffer (no per-expert commit+wait).
    // Each expert k uses slot [k].
    // Double-buffered: set A (data) for GPU compute, set B (data_B) for background pread.
    // Gate/up/act/out only need one set (GPU uses them after pread completes).
    #define MAX_K 8
    id<MTLBuffer> buf_multi_expert_data[MAX_K];   // [EXPERT_SIZE bytes] each — buffer set A
    id<MTLBuffer> buf_multi_expert_data_B[MAX_K]; // [EXPERT_SIZE bytes] each — buffer set B (prefetch)
    id<MTLBuffer> buf_multi_expert_gate[MAX_K];   // [MOE_INTERMEDIATE floats]
    id<MTLBuffer> buf_multi_expert_up[MAX_K];     // [MOE_INTERMEDIATE floats]
    id<MTLBuffer> buf_multi_expert_act[MAX_K];    // [MOE_INTERMEDIATE floats]
    id<MTLBuffer> buf_multi_expert_out[MAX_K];    // [HIDDEN_DIM floats]
    id<MTLBuffer> buf_multi_expert_input;         // [HIDDEN_DIM floats] (shared, read-only during dispatch)
    // Shared expert buffers for fused CMD2 (shared gate/up computed in CMD1,
    // SwiGLU + down_proj in CMD2 alongside routed experts)
    id<MTLBuffer> buf_shared_gate;   // [SHARED_INTERMEDIATE floats]
    id<MTLBuffer> buf_shared_up;     // [SHARED_INTERMEDIATE floats]
    id<MTLBuffer> buf_shared_act;    // [SHARED_INTERMEDIATE floats] (SwiGLU output)
    id<MTLBuffer> buf_shared_out;    // [HIDDEN_DIM floats] (down_proj output)
    // Fused o_proj+norm+routing buffers (eliminates 1 cmd buffer per layer)
    id<MTLBuffer> buf_residual;     // [HIDDEN_DIM floats] holds residual for GPU add
    id<MTLBuffer> buf_h_mid;        // [HIDDEN_DIM floats] residual+oproj result
    id<MTLBuffer> buf_sum_sq;       // [1 float] for RMS norm reduction
    // GPU attention buffers (for full attention layers)
    id<MTLBuffer> buf_kv_k[NUM_FULL_ATTN_LAYERS];  // K cache per full-attn layer
    id<MTLBuffer> buf_kv_v[NUM_FULL_ATTN_LAYERS];  // V cache per full-attn layer
    id<MTLBuffer> buf_attn_q;       // [NUM_ATTN_HEADS * HEAD_DIM floats] all query heads
    id<MTLBuffer> buf_attn_scores;  // [NUM_ATTN_HEADS * MAX_SEQ_LEN floats] all heads' scores
    id<MTLBuffer> buf_attn_out;     // [NUM_ATTN_HEADS * HEAD_DIM floats] full attention output
    id<MTLBuffer> buf_attn_gate;    // [NUM_ATTN_HEADS * HEAD_DIM floats] sigmoid gate
    // CMD3 GPU-side combine buffers (weighted_sum + residual + norm on GPU)
    id<MTLComputePipelineState> moe_combine_residual;  // fused combine kernel
    id<MTLBuffer> buf_moe_hidden;     // [HIDDEN_DIM floats] GPU combine output (hidden state)
    id<MTLBuffer> buf_combine_params; // [10 floats] expert weights[8] + shared_gate_score + padding
    id<MTLBuffer> buf_cmd3_sum_sq;    // [1 float] for RMS norm reduction in CMD3
    // Shared event for CPU-GPU synchronization (async pipeline)
#if USE_EVENT_PIPELINE
    id<MTLSharedEvent> pipeline_event;   // CPU signals when buf_input is ready
    uint64_t event_value;                // monotonically increasing event counter
#endif
    // GPU delta-net (gated_delta_net_step) and conv1d pipelines
    id<MTLComputePipelineState> delta_net_step;  // gated_delta_net_step kernel
    id<MTLComputePipelineState> conv1d_step;     // conv1d_step kernel
    id<MTLComputePipelineState> rms_norm_qk;     // per-head RMS normalize for q and k
    id<MTLComputePipelineState> compute_decay_beta; // g_decay and beta_gate for delta-net
    id<MTLComputePipelineState> gated_rms_norm;  // z-gated output normalization
    // Persistent GPU state buffers for linear attention layers
    id<MTLBuffer> buf_delta_state[NUM_LINEAR_LAYERS];
    id<MTLBuffer> buf_conv_state[NUM_LINEAR_LAYERS];
    // Scratch buffers for delta-net inputs/outputs (sized at allocation time)
    id<MTLBuffer> buf_delta_q;        // [LINEAR_TOTAL_KEY] float
    id<MTLBuffer> buf_delta_k;        // [LINEAR_TOTAL_KEY] float
    id<MTLBuffer> buf_delta_v;        // [LINEAR_TOTAL_VALUE] float
    id<MTLBuffer> buf_delta_g_decay;  // [LINEAR_NUM_V_HEADS] float
    id<MTLBuffer> buf_delta_beta;     // [LINEAR_NUM_V_HEADS] float
    id<MTLBuffer> buf_delta_output;   // [LINEAR_TOTAL_VALUE] float
    id<MTLBuffer> buf_conv_input;     // [LINEAR_CONV_DIM] float
    id<MTLBuffer> buf_conv_output;    // [LINEAR_CONV_DIM] float
} MetalCtx;

static MetalCtx *g_metal = NULL;

static MetalCtx *metal_setup(void) {
    MetalCtx *ctx = calloc(1, sizeof(MetalCtx));
    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        fprintf(stderr, "ERROR: No Metal device\n");
        free(ctx); return NULL;
    }
    printf("[metal] Device: %s\n", [[ctx->device name] UTF8String]);

    ctx->queue = [ctx->device newCommandQueue];
    if (!ctx->queue) {
        fprintf(stderr, "ERROR: No command queue\n");
        free(ctx); return NULL;
    }

    // Compile shaders from source
    NSError *error = nil;
    NSArray *paths = @[@"src/shaders.metal", @"shaders.metal"];
    NSString *src = nil;
    for (NSString *p in paths) {
        src = [NSString stringWithContentsOfFile:p encoding:NSUTF8StringEncoding error:&error];
        if (src) break;
    }
    if (!src) {
        fprintf(stderr, "ERROR: Cannot find shaders.metal\n");
        free(ctx); return NULL;
    }

    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    opts.mathMode = MTLMathModeFast;
    opts.languageVersion = MTLLanguageVersion3_1;
#if USE_KV_CACHE_BF16 || USE_FUSED_RMS_NORM
    opts.preprocessorMacros = @{
        @"USE_KV_CACHE_BF16": @(USE_KV_CACHE_BF16),
        @"USE_FUSED_RMS_NORM": @(USE_FUSED_RMS_NORM),
    };
#endif
    double t0 = now_ms();
    ctx->library = [ctx->device newLibraryWithSource:src options:opts error:&error];
    if (!ctx->library) {
        fprintf(stderr, "ERROR: Shader compile failed: %s\n",
                [[error localizedDescription] UTF8String]);
        free(ctx); return NULL;
    }
    printf("[metal] Shader compile: %.0f ms\n", now_ms() - t0);

    // Create pipelines
    id<MTLComputePipelineState> (^makePipe)(NSString *) = ^(NSString *name) {
        id<MTLFunction> fn = [ctx->library newFunctionWithName:name];
        if (!fn) { fprintf(stderr, "ERROR: shader '%s' not found\n", [name UTF8String]); return (id<MTLComputePipelineState>)nil; }
        NSError *e2 = nil;
        id<MTLComputePipelineState> ps = [ctx->device newComputePipelineStateWithFunction:fn error:&e2];
        if (!ps) { fprintf(stderr, "ERROR: pipeline '%s': %s\n", [name UTF8String], [[e2 localizedDescription] UTF8String]); }
        return ps;
    };

    ctx->matvec_v3     = makePipe(@"dequant_matvec_4bit_v3");
    ctx->matvec_v5     = makePipe(@"dequant_matvec_4bit_v5");  // LUT variant (no uint→float conversions)
    ctx->matvec_fast   = makePipe(@"dequant_matvec_4bit_fast");
    ctx->matvec_2bit   = makePipe(@"dequant_matvec_2bit");
    ctx->rms_norm_sum  = makePipe(@"rms_norm_sum_sq");
    ctx->rms_norm_apply = makePipe(@"rms_norm_apply");
    ctx->rms_norm_apply_bf16 = makePipe(@"rms_norm_apply_bf16");
#if USE_FUSED_RMS_NORM
    ctx->rms_norm_fused  = makePipe(@"rms_norm_fused");
#endif
    ctx->residual_add  = makePipe(@"residual_add");
    ctx->swiglu        = makePipe(@"swiglu_fused");
#if USE_FUSED_GATE_UP_SWIGLU
    ctx->fused_gate_up_swiglu = makePipe(@"fused_gate_up_swiglu");
#endif
    ctx->attn_scores_pipe  = makePipe(@"attn_scores_batched");
    ctx->attn_softmax_pipe = makePipe(@"attn_softmax_batched");
    ctx->attn_values_pipe  = makePipe(@"attn_values_batched");
    ctx->sigmoid_gate_pipe = makePipe(@"sigmoid_gate");
    ctx->moe_combine_residual = makePipe(@"moe_combine_residual");
    ctx->delta_net_step    = makePipe(@"gated_delta_net_step");
    ctx->conv1d_step       = makePipe(@"conv1d_step");
    ctx->rms_norm_qk       = makePipe(@"rms_norm_qk");
    ctx->compute_decay_beta = makePipe(@"compute_decay_beta");
    ctx->gated_rms_norm    = makePipe(@"gated_rms_norm");
    if (!ctx->moe_combine_residual) fprintf(stderr, "[metal] WARNING: moe_combine_residual pipeline failed\n");
    if (!ctx->delta_net_step) fprintf(stderr, "[metal] WARNING: gated_delta_net_step pipeline failed (CPU fallback)\n");
    if (!ctx->conv1d_step)    fprintf(stderr, "[metal] WARNING: conv1d_step pipeline failed (CPU fallback)\n");
    if (!ctx->rms_norm_qk)       fprintf(stderr, "[metal] WARNING: rms_norm_qk pipeline failed (CPU fallback)\n");
    if (!ctx->compute_decay_beta) fprintf(stderr, "[metal] WARNING: compute_decay_beta pipeline failed (CPU fallback)\n");
    if (!ctx->gated_rms_norm)     fprintf(stderr, "[metal] WARNING: gated_rms_norm pipeline failed (CPU fallback)\n");

    if (!ctx->matvec_v3 || !ctx->matvec_fast) {
        fprintf(stderr, "ERROR: Required Metal pipeline missing\n");
        free(ctx); return NULL;
    }

    // Allocate reusable buffers (large enough for biggest projection)
    // Q proj output is 16384 floats, lm_head output is 248320 floats
    // o_proj input is 8192, linear attn out_proj input is 8192
    size_t max_out = VOCAB_SIZE * sizeof(float);  // lm_head is largest
    size_t max_in = LINEAR_TOTAL_VALUE * sizeof(float);  // 8192 floats (linear_attn out_proj)
    if (max_in < (size_t)(NUM_ATTN_HEADS * HEAD_DIM) * sizeof(float)) {
        max_in = (size_t)(NUM_ATTN_HEADS * HEAD_DIM) * sizeof(float);  // o_proj input = 8192
    }
    ctx->buf_input  = [ctx->device newBufferWithLength:max_in  options:MTLResourceStorageModeShared];
    ctx->buf_output = [ctx->device newBufferWithLength:max_out options:MTLResourceStorageModeShared];

    // Batched matmul output slots — each large enough for the biggest projection
    // q_proj = 16384 floats, qkv_proj = 12288, z_proj = 8192, o_proj = 4096
    // lm_head (248320) uses buf_output directly, not batched.
    {
        size_t slot_size = (size_t)(NUM_ATTN_HEADS * HEAD_DIM * 2) * sizeof(float);  // 16384 floats
        if (slot_size < (size_t)LINEAR_CONV_DIM * sizeof(float))
            slot_size = (size_t)LINEAR_CONV_DIM * sizeof(float);  // 12288 floats
        for (int i = 0; i < MAX_BATCH_SLOTS; i++) {
            ctx->batch_out[i] = [ctx->device newBufferWithLength:slot_size
                                                         options:MTLResourceStorageModeShared];
        }
    }

    // Expert computation buffers (reused across all experts and layers)
    ctx->buf_expert_data  = [ctx->device newBufferWithLength:EXPERT_SIZE
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_input = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_gate  = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_up    = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_act   = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_out   = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                     options:MTLResourceStorageModeShared];

    // Multi-expert buffers: K independent slots (double-buffered data)
    // Expert data buffers use 2MB-aligned backing memory for DMA efficiency.
    // The pread DMA controller transfers 3.6x faster with 2MB alignment vs 16KB.
    ctx->buf_multi_expert_input = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    size_t expert_alloc_size = (EXPERT_SIZE + 2*1024*1024 - 1) & ~(2*1024*1024 - 1);  // round up to 2MB
    for (int k = 0; k < MAX_K; k++) {
        // 2MB-aligned allocation for optimal DMA throughput
        void *aligned_data = NULL, *aligned_data_b = NULL;
        posix_memalign(&aligned_data,   2*1024*1024, expert_alloc_size);
        posix_memalign(&aligned_data_b, 2*1024*1024, expert_alloc_size);
        memset(aligned_data, 0, expert_alloc_size);
        memset(aligned_data_b, 0, expert_alloc_size);
        ctx->buf_multi_expert_data[k] = [ctx->device newBufferWithBytesNoCopy:aligned_data
                                                                       length:expert_alloc_size
                                                                      options:MTLResourceStorageModeShared
                                                                  deallocator:nil];
        ctx->buf_multi_expert_data_B[k] = [ctx->device newBufferWithBytesNoCopy:aligned_data_b
                                                                         length:expert_alloc_size
                                                                        options:MTLResourceStorageModeShared
                                                                    deallocator:nil];
        ctx->buf_multi_expert_gate[k] = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_up[k]   = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_act[k]  = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_out[k]  = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
    }

    // Shared expert buffers (for fused CMD2)
    ctx->buf_shared_gate = [ctx->device newBufferWithLength:SHARED_INTERMEDIATE * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_up   = [ctx->device newBufferWithLength:SHARED_INTERMEDIATE * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_act  = [ctx->device newBufferWithLength:SHARED_INTERMEDIATE * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_out  = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                    options:MTLResourceStorageModeShared];

    // Fused o_proj+norm+routing buffers
    ctx->buf_residual = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    ctx->buf_h_mid    = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    ctx->buf_sum_sq   = [ctx->device newBufferWithLength:sizeof(float)
                                                 options:MTLResourceStorageModeShared];

    // CMD3 GPU-side combine buffers
    ctx->buf_moe_hidden    = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    ctx->buf_combine_params = [ctx->device newBufferWithLength:10 * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
    ctx->buf_cmd3_sum_sq    = [ctx->device newBufferWithLength:sizeof(float)
                                                        options:MTLResourceStorageModeShared];

    // GPU attention buffers
    {
        size_t kv_dim = NUM_KV_HEADS * HEAD_DIM;
#if USE_KV_CACHE_BF16
        size_t kv_cache_size = GPU_KV_SEQ * kv_dim * sizeof(uint16_t);
#else
        size_t kv_cache_size = GPU_KV_SEQ * kv_dim * sizeof(float);
#endif
        for (int i = 0; i < NUM_FULL_ATTN_LAYERS; i++) {
            ctx->buf_kv_k[i] = [ctx->device newBufferWithLength:kv_cache_size
                                                        options:MTLResourceStorageModeShared];
            ctx->buf_kv_v[i] = [ctx->device newBufferWithLength:kv_cache_size
                                                        options:MTLResourceStorageModeShared];
        }
        ctx->buf_attn_q      = [ctx->device newBufferWithLength:NUM_ATTN_HEADS * HEAD_DIM * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_scores = [ctx->device newBufferWithLength:(size_t)NUM_ATTN_HEADS * GPU_KV_SEQ * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_out    = [ctx->device newBufferWithLength:NUM_ATTN_HEADS * HEAD_DIM * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_gate   = [ctx->device newBufferWithLength:NUM_ATTN_HEADS * HEAD_DIM * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        printf("[metal] GPU attention buffers: %d KV caches (%.1f MB each), scores buf %.1f MB\n",
               NUM_FULL_ATTN_LAYERS, kv_cache_size / 1e6,
               (double)(NUM_ATTN_HEADS * MAX_SEQ_LEN * sizeof(float)) / 1e6);
    }

    // Persistent GPU state buffers for delta-net (linear attention layers)
    if (ctx->delta_net_step) {
        size_t delta_state_sz = (size_t)LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM;
        size_t conv_state_sz = 3 * (size_t)LINEAR_CONV_DIM;
        for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
            ctx->buf_delta_state[i] = [ctx->device newBufferWithLength:delta_state_sz * sizeof(float)
                                                               options:MTLResourceStorageModeShared];
            memset([ctx->buf_delta_state[i] contents], 0, delta_state_sz * sizeof(float));
            ctx->buf_conv_state[i] = [ctx->device newBufferWithLength:conv_state_sz * sizeof(float)
                                                              options:MTLResourceStorageModeShared];
            memset([ctx->buf_conv_state[i] contents], 0, conv_state_sz * sizeof(float));
        }
        // Scratch buffers for delta-net inputs/outputs (allocated once, reused)
        ctx->buf_delta_q       = [ctx->device newBufferWithLength:LINEAR_TOTAL_KEY * sizeof(float)   options:MTLResourceStorageModeShared];
        ctx->buf_delta_k       = [ctx->device newBufferWithLength:LINEAR_TOTAL_KEY * sizeof(float)   options:MTLResourceStorageModeShared];
        ctx->buf_delta_v       = [ctx->device newBufferWithLength:LINEAR_TOTAL_VALUE * sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_delta_g_decay = [ctx->device newBufferWithLength:LINEAR_NUM_V_HEADS * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->buf_delta_beta    = [ctx->device newBufferWithLength:LINEAR_NUM_V_HEADS * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->buf_delta_output  = [ctx->device newBufferWithLength:LINEAR_TOTAL_VALUE * sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_conv_input    = [ctx->device newBufferWithLength:LINEAR_CONV_DIM * sizeof(float)    options:MTLResourceStorageModeShared];
        ctx->buf_conv_output   = [ctx->device newBufferWithLength:LINEAR_CONV_DIM * sizeof(float)    options:MTLResourceStorageModeShared];
        printf("[metal] Delta-net GPU buffers: %d layers (%.1f MB state + %.1f MB scratch)\n",
               NUM_LINEAR_LAYERS,
               (double)NUM_LINEAR_LAYERS * (delta_state_sz + conv_state_sz) * sizeof(float) / 1e6,
               (double)(2 * LINEAR_TOTAL_KEY + 2 * LINEAR_TOTAL_VALUE + 2 * LINEAR_NUM_V_HEADS + 2 * LINEAR_CONV_DIM) * sizeof(float) / 1e6);
    }

    // Create shared event for CPU-GPU async pipeline
#if USE_EVENT_PIPELINE
    ctx->pipeline_event = [ctx->device newSharedEvent];
    ctx->event_value = 0;
#endif

    printf("[metal] Inference pipelines ready (multi-expert[%d] + shared buffers allocated)\n", MAX_K);
    return ctx;
}

// Reset delta-net and conv GPU state buffers (call at start of new generation)
static void reset_delta_net_state(void) {
    if (!g_metal || !g_metal->delta_net_step) return;
    size_t delta_state_sz = (size_t)LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float);
    size_t conv_state_sz = 3 * (size_t)LINEAR_CONV_DIM * sizeof(float);
    for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
        if (g_metal->buf_delta_state[i])
            memset([g_metal->buf_delta_state[i] contents], 0, delta_state_sz);
        if (g_metal->buf_conv_state[i])
            memset([g_metal->buf_conv_state[i] contents], 0, conv_state_sz);
    }
}

// Wrap the mmap'd weight file as a Metal buffer (zero-copy on unified memory)
// mmap returns page-aligned addresses, Metal requires the same.
// On Apple Silicon, page size is 16KB.
static void metal_set_weights(MetalCtx *ctx, void *data, size_t size) {
    // Round size up to page boundary (16KB)
    size_t page_size = 16384;
    size_t aligned_size = (size + page_size - 1) & ~(page_size - 1);

    ctx->wf_buf = [ctx->device newBufferWithBytesNoCopy:data
                                                 length:aligned_size
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
    if (!ctx->wf_buf) {
        fprintf(stderr, "WARNING: Cannot wrap weight file as Metal buffer (size=%.2f GB)\n",
                size / 1e9);
        fprintf(stderr, "  data=%p, aligned_size=%zu -- GPU matmul will fall back to CPU\n",
                data, aligned_size);
    } else {
        printf("[metal] Weight file wrapped as Metal buffer (%.2f GB)\n",
               aligned_size / 1e9);
    }
}

// GPU dequant matvec: out[out_dim] = W_4bit * x[in_dim]
// W_packed, scales, biases are pointers into mmap'd weight file
// x_f32 is CPU float array, result written back to out_f32
//
// We wrap the ENTIRE mmap'd weight file as a single Metal buffer and use
// byte offsets to point each shader argument at the right tensor.
// This avoids per-tensor buffer creation and the page-alignment constraint.
static void gpu_dequant_matvec(
    MetalCtx *ctx,
    const void *W_packed, const void *scales, const void *biases,
    const float *x_f32, float *out_f32,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size
) {
    // Copy input to Metal buffer
    memcpy([ctx->buf_input contents], x_f32, in_dim * sizeof(float));

    size_t o_size = (size_t)out_dim * sizeof(float);

    // Compute offsets into the mmap'd weight buffer
    NSUInteger w_off = (NSUInteger)((const char *)W_packed - (const char *)[ctx->wf_buf contents]);
    NSUInteger s_off = (NSUInteger)((const char *)scales   - (const char *)[ctx->wf_buf contents]);
    NSUInteger b_off = (NSUInteger)((const char *)biases   - (const char *)[ctx->wf_buf contents]);

    // Ensure output buffer is large enough
    id<MTLBuffer> o_buf = ctx->buf_output;
    if (o_size > [o_buf length]) {
        o_buf = [ctx->device newBufferWithLength:o_size options:MTLResourceStorageModeShared];
    }

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];

    // v3 shader uses x_shared[4096], so can only handle in_dim <= 4096
    // For larger in_dim (e.g. o_proj with in_dim=8192), use matvec_fast
    int use_v3 = (in_dim <= 4096);
    [enc setComputePipelineState: use_v3 ? ctx->matvec_v3 : ctx->matvec_fast];
    [enc setBuffer:ctx->wf_buf  offset:w_off atIndex:0];
    [enc setBuffer:ctx->wf_buf  offset:s_off atIndex:1];
    [enc setBuffer:ctx->wf_buf  offset:b_off atIndex:2];
    [enc setBuffer:ctx->buf_input offset:0   atIndex:3];
    [enc setBuffer:o_buf        offset:0     atIndex:4];
    [enc setBytes:&out_dim      length:4     atIndex:5];
    [enc setBytes:&in_dim       length:4     atIndex:6];
    [enc setBytes:&group_size   length:4     atIndex:7];

    if (use_v3) {
        // v3: tiled threadgroups, 256 threads, 8 rows per TG
        uint32_t num_tgs = (out_dim + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        // fast: one threadgroup per output row, 64 threads per TG
        NSUInteger tg_size = 64;
        [enc dispatchThreadgroups:MTLSizeMake(out_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    }
    [enc endEncoding];
    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    // Copy result back
    memcpy(out_f32, [o_buf contents], o_size);
}

// Wrapper: use GPU if available and weight buffer is set, CPU otherwise
static void fast_dequant_matvec(
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out,
    int out_dim, int in_dim, int group_size
) {
    if (g_metal && g_metal->wf_buf) {
        gpu_dequant_matvec(g_metal, W, scales, biases, x, out,
                           (uint32_t)out_dim, (uint32_t)in_dim, (uint32_t)group_size);
    } else {
#if USE_CPU_DEQUANT_FMA
        cpu_dequant_matvec_fma(W, scales, biases, x, out, out_dim, in_dim, group_size);
#else
        cpu_dequant_matvec(W, scales, biases, x, out, out_dim, in_dim, group_size);
#endif
    }
}


#endif // METAL_SETUP_H
