#ifndef METAL_SETUP_H
#define METAL_SETUP_H

// ============================================================================
// Metal context for GPU-accelerated matmuls.
// MetalCtx type is in model_types.h. State stored in FlashMoE_Model.metal.
// ============================================================================

#include "model_internal.h"
#include "shaders.h"

static int metal_setup(FlashMoE_Model *m) {
    MetalCtx *ctx = calloc(1, sizeof(MetalCtx));
    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        fprintf(stderr, "ERROR: No Metal device\n");
        free(ctx); return -1;
    }
    printf("[metal] Device: %s\n", [[ctx->device name] UTF8String]);

    ctx->queue = [ctx->device newCommandQueue];
    if (!ctx->queue) {
        fprintf(stderr, "ERROR: No command queue\n");
        free(ctx); return -1;
    }

    // Compile shaders from embedded source
    NSError *error = nil;
    NSString *src = [NSString stringWithUTF8String:g_shader_source];
    if (!src) {
        fprintf(stderr, "ERROR: Embedded shader source is null\n");
        free(ctx); return -1;
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
        free(ctx); return -1;
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
    ctx->matvec_v5     = makePipe(@"dequant_matvec_4bit_v5");
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
        free(ctx); return -1;
    }

    // Allocate reusable buffers (large enough for biggest projection)
    size_t max_out = m->cfg.vocab_size * sizeof(float);
    size_t max_in = m->cfg.linear_total_value * sizeof(float);
    if (max_in < (size_t)(m->cfg.num_attn_heads * HEAD_DIM) * sizeof(float)) {
        max_in = (size_t)(m->cfg.num_attn_heads * HEAD_DIM) * sizeof(float);
    }
    ctx->buf_input  = [ctx->device newBufferWithLength:max_in  options:MTLResourceStorageModeShared];
    ctx->buf_output = [ctx->device newBufferWithLength:max_out options:MTLResourceStorageModeShared];

    // Batched matmul output slots
    {
        size_t slot_size = (size_t)(m->cfg.num_attn_heads * HEAD_DIM * 2) * sizeof(float);
        if (slot_size < (size_t)m->cfg.linear_conv_dim * sizeof(float))
            slot_size = (size_t)m->cfg.linear_conv_dim * sizeof(float);
        for (int i = 0; i < MAX_BATCH_SLOTS; i++) {
            ctx->batch_out[i] = [ctx->device newBufferWithLength:slot_size
                                                         options:MTLResourceStorageModeShared];
        }
    }

    // Expert computation buffers
    ctx->buf_expert_data  = [ctx->device newBufferWithLength:m->cfg.expert_size_4bit
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_input = [ctx->device newBufferWithLength:m->cfg.hidden_dim * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_gate  = [ctx->device newBufferWithLength:m->cfg.moe_intermediate * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_up    = [ctx->device newBufferWithLength:m->cfg.moe_intermediate * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_act   = [ctx->device newBufferWithLength:m->cfg.moe_intermediate * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_out   = [ctx->device newBufferWithLength:m->cfg.hidden_dim * sizeof(float)
                                                     options:MTLResourceStorageModeShared];

    // Multi-expert buffers
    ctx->buf_multi_expert_input = [ctx->device newBufferWithLength:m->cfg.hidden_dim * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    size_t expert_alloc_size = (m->cfg.expert_size_4bit + 2*1024*1024 - 1) & ~(2*1024*1024 - 1);
    for (int k = 0; k < MAX_K; k++) {
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
        ctx->buf_multi_expert_gate[k] = [ctx->device newBufferWithLength:m->cfg.moe_intermediate * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_up[k]   = [ctx->device newBufferWithLength:m->cfg.moe_intermediate * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_act[k]  = [ctx->device newBufferWithLength:m->cfg.moe_intermediate * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_out[k]  = [ctx->device newBufferWithLength:m->cfg.hidden_dim * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
    }

    // Shared expert buffers
    ctx->buf_shared_gate = [ctx->device newBufferWithLength:m->cfg.shared_intermediate * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_up   = [ctx->device newBufferWithLength:m->cfg.shared_intermediate * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_act  = [ctx->device newBufferWithLength:m->cfg.shared_intermediate * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_out  = [ctx->device newBufferWithLength:m->cfg.hidden_dim * sizeof(float)
                                                    options:MTLResourceStorageModeShared];

    // Fused o_proj+norm+routing buffers
    ctx->buf_residual = [ctx->device newBufferWithLength:m->cfg.hidden_dim * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    ctx->buf_h_mid    = [ctx->device newBufferWithLength:m->cfg.hidden_dim * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    ctx->buf_sum_sq   = [ctx->device newBufferWithLength:sizeof(float)
                                                 options:MTLResourceStorageModeShared];

    // CMD3 GPU-side combine buffers
    ctx->buf_moe_hidden    = [ctx->device newBufferWithLength:m->cfg.hidden_dim * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    ctx->buf_combine_params = [ctx->device newBufferWithLength:10 * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
    ctx->buf_cmd3_sum_sq    = [ctx->device newBufferWithLength:sizeof(float)
                                                        options:MTLResourceStorageModeShared];

    // GPU attention buffers
    {
        size_t kv_dim = m->cfg.num_kv_heads * HEAD_DIM;
#if USE_KV_CACHE_BF16
        size_t kv_cache_size = GPU_KV_SEQ * kv_dim * sizeof(uint16_t);
#else
        size_t kv_cache_size = GPU_KV_SEQ * kv_dim * sizeof(float);
#endif
        ctx->buf_kv_k = (__unsafe_unretained id<MTLBuffer> *)calloc(m->cfg.num_full_attn_layers, sizeof(id<MTLBuffer>));
        ctx->buf_kv_v = (__unsafe_unretained id<MTLBuffer> *)calloc(m->cfg.num_full_attn_layers, sizeof(id<MTLBuffer>));
        for (int i = 0; i < m->cfg.num_full_attn_layers; i++) {
            id<MTLBuffer> buf_k = [ctx->device newBufferWithLength:kv_cache_size
                                                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_v = [ctx->device newBufferWithLength:kv_cache_size
                                                           options:MTLResourceStorageModeShared];
            ctx->buf_kv_k[i] = buf_k;
            ctx->buf_kv_v[i] = buf_v;
            CFRetain((__bridge CFTypeRef)buf_k);
            CFRetain((__bridge CFTypeRef)buf_v);
        }
        ctx->buf_attn_q      = [ctx->device newBufferWithLength:m->cfg.num_attn_heads * HEAD_DIM * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_scores = [ctx->device newBufferWithLength:(size_t)m->cfg.num_attn_heads * GPU_KV_SEQ * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_out    = [ctx->device newBufferWithLength:m->cfg.num_attn_heads * HEAD_DIM * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_gate   = [ctx->device newBufferWithLength:m->cfg.num_attn_heads * HEAD_DIM * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        printf("[metal] GPU attention buffers: %d KV caches (%.1f MB each), scores buf %.1f MB\n",
               m->cfg.num_full_attn_layers, kv_cache_size / 1e6,
               (double)(m->cfg.num_attn_heads * MAX_SEQ_LEN * sizeof(float)) / 1e6);
    }

    // Persistent GPU state buffers for delta-net
    if (ctx->delta_net_step) {
        size_t delta_state_sz = (size_t)m->cfg.linear_num_v_heads * LINEAR_VALUE_DIM * LINEAR_KEY_DIM;
        size_t conv_state_sz = 3 * (size_t)m->cfg.linear_conv_dim;
        ctx->buf_delta_state = (__unsafe_unretained id<MTLBuffer> *)calloc(m->cfg.num_linear_layers, sizeof(id<MTLBuffer>));
        ctx->buf_conv_state  = (__unsafe_unretained id<MTLBuffer> *)calloc(m->cfg.num_linear_layers, sizeof(id<MTLBuffer>));
        for (int i = 0; i < m->cfg.num_linear_layers; i++) {
            id<MTLBuffer> ds_buf = [ctx->device newBufferWithLength:delta_state_sz * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
            ctx->buf_delta_state[i] = ds_buf;
            CFRetain((__bridge CFTypeRef)ds_buf);
            memset([ds_buf contents], 0, delta_state_sz * sizeof(float));
            id<MTLBuffer> cs_buf = [ctx->device newBufferWithLength:conv_state_sz * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
            ctx->buf_conv_state[i] = cs_buf;
            CFRetain((__bridge CFTypeRef)cs_buf);
            memset([cs_buf contents], 0, conv_state_sz * sizeof(float));
        }
        // Scratch buffers for delta-net inputs/outputs
        ctx->buf_delta_q       = [ctx->device newBufferWithLength:m->cfg.linear_total_key * sizeof(float)   options:MTLResourceStorageModeShared];
        ctx->buf_delta_k       = [ctx->device newBufferWithLength:m->cfg.linear_total_key * sizeof(float)   options:MTLResourceStorageModeShared];
        ctx->buf_delta_v       = [ctx->device newBufferWithLength:m->cfg.linear_total_value * sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_delta_g_decay = [ctx->device newBufferWithLength:m->cfg.linear_num_v_heads * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->buf_delta_beta    = [ctx->device newBufferWithLength:m->cfg.linear_num_v_heads * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->buf_delta_output  = [ctx->device newBufferWithLength:m->cfg.linear_total_value * sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_conv_input    = [ctx->device newBufferWithLength:m->cfg.linear_conv_dim * sizeof(float)    options:MTLResourceStorageModeShared];
        ctx->buf_conv_output   = [ctx->device newBufferWithLength:m->cfg.linear_conv_dim * sizeof(float)    options:MTLResourceStorageModeShared];
        printf("[metal] Delta-net GPU buffers: %d layers (%.1f MB state + %.1f MB scratch)\n",
               m->cfg.num_linear_layers,
               (double)m->cfg.num_linear_layers * (delta_state_sz + conv_state_sz) * sizeof(float) / 1e6,
               (double)(2 * m->cfg.linear_total_key + 2 * m->cfg.linear_total_value + 2 * m->cfg.linear_num_v_heads + 2 * m->cfg.linear_conv_dim) * sizeof(float) / 1e6);
    }

    // Create shared event for CPU-GPU async pipeline
#if USE_EVENT_PIPELINE
    ctx->pipeline_event = [ctx->device newSharedEvent];
    ctx->event_value = 0;
#endif

    printf("[metal] Inference pipelines ready (multi-expert[%d] + shared buffers allocated)\n", MAX_K);
    m->metal = ctx;
    return 0;
}

// Reset delta-net and conv GPU state buffers (call at start of new generation)
static void reset_delta_net_state(FlashMoE_Model *m) {
    if (!m->metal || !m->metal->delta_net_step) return;
    size_t delta_state_sz = (size_t)m->cfg.linear_num_v_heads * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float);
    size_t conv_state_sz = 3 * (size_t)m->cfg.linear_conv_dim * sizeof(float);
    for (int i = 0; i < m->cfg.num_linear_layers; i++) {
        if (m->metal->buf_delta_state[i])
            memset([m->metal->buf_delta_state[i] contents], 0, delta_state_sz);
        if (m->metal->buf_conv_state[i])
            memset([m->metal->buf_conv_state[i] contents], 0, conv_state_sz);
    }
}

// Wrap the mmap'd weight file as a Metal buffer (zero-copy on unified memory)
static void metal_set_weights(MetalCtx *ctx, void *data, size_t size) {
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
        uint32_t num_tgs = (out_dim + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
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
    FlashMoE_Model *m,
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out,
    int out_dim, int in_dim, int group_size
) {
    if (m->metal && m->metal->wf_buf) {
        gpu_dequant_matvec(m->metal, W, scales, biases, x, out,
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
