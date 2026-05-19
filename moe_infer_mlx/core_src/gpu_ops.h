#ifndef GPU_OPS_H
#define GPU_OPS_H

// ============================================================================
// Batched GPU matmul: encode N independent matmuls sharing the same input
// into ONE command buffer, reducing dispatch overhead by N-1 round-trips.
// BatchMatvecSpec type is in model_types.h.
// ============================================================================

#include "common.h"

// Run N matmuls in a single command buffer. All share the same input vector.
static void gpu_batch_matvec(
    MetalCtx *ctx,
    const float *x_f32, uint32_t x_dim,
    BatchMatvecSpec *specs, int num_specs
) {
    // Copy input once
    memcpy([ctx->buf_input contents], x_f32, x_dim * sizeof(float));

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];

    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        NSUInteger w_off = (NSUInteger)((const char *)s->W      - (const char *)[ctx->wf_buf contents]);
        NSUInteger s_off = (NSUInteger)((const char *)s->scales  - (const char *)[ctx->wf_buf contents]);
        NSUInteger b_off = (NSUInteger)((const char *)s->biases  - (const char *)[ctx->wf_buf contents]);

        id<MTLBuffer> o_buf = ctx->batch_out[s->batch_slot];

        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        int use_v3 = (s->in_dim <= 4096);
        [enc setComputePipelineState: use_v3 ? ctx->matvec_v3 : ctx->matvec_fast];
        [enc setBuffer:ctx->wf_buf  offset:w_off atIndex:0];
        [enc setBuffer:ctx->wf_buf  offset:s_off atIndex:1];
        [enc setBuffer:ctx->wf_buf  offset:b_off atIndex:2];
        [enc setBuffer:ctx->buf_input offset:0   atIndex:3];
        [enc setBuffer:o_buf        offset:0     atIndex:4];
        [enc setBytes:&s->out_dim   length:4     atIndex:5];
        [enc setBytes:&s->in_dim    length:4     atIndex:6];
        [enc setBytes:&s->group_size length:4    atIndex:7];

        if (use_v3) {
            uint32_t num_tgs = (s->out_dim + 7) / 8;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake(s->out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        [enc endEncoding];
    }

    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    // Copy results back to CPU
    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        memcpy(s->out_cpu, [ctx->batch_out[s->batch_slot] contents],
               s->out_dim * sizeof(float));
    }
}

// Encode N matmuls into cmdbuf. Input must already be in ctx->buf_input.
static void gpu_encode_batch_matvec(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    BatchMatvecSpec *specs, int num_specs
) {
    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        NSUInteger w_off = (NSUInteger)((const char *)s->W      - (const char *)[ctx->wf_buf contents]);
        NSUInteger s_off = (NSUInteger)((const char *)s->scales  - (const char *)[ctx->wf_buf contents]);
        NSUInteger b_off = (NSUInteger)((const char *)s->biases  - (const char *)[ctx->wf_buf contents]);

        id<MTLBuffer> o_buf = ctx->batch_out[s->batch_slot];

        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        int use_v3 = (s->in_dim <= 4096);
        [enc setComputePipelineState: use_v3 ? ctx->matvec_v3 : ctx->matvec_fast];
        [enc setBuffer:ctx->wf_buf  offset:w_off atIndex:0];
        [enc setBuffer:ctx->wf_buf  offset:s_off atIndex:1];
        [enc setBuffer:ctx->wf_buf  offset:b_off atIndex:2];
        [enc setBuffer:ctx->buf_input offset:0   atIndex:3];
        [enc setBuffer:o_buf        offset:0     atIndex:4];
        [enc setBytes:&s->out_dim   length:4     atIndex:5];
        [enc setBytes:&s->in_dim    length:4     atIndex:6];
        [enc setBytes:&s->group_size length:4    atIndex:7];

        if (use_v3) {
            uint32_t num_tgs = (s->out_dim + 7) / 8;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake(s->out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        [enc endEncoding];
    }
}

// Copy batch results from GPU buffers back to CPU pointers.
static void gpu_flush_batch_results(MetalCtx *ctx, BatchMatvecSpec *specs, int num_specs) {
    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        memcpy(s->out_cpu, [ctx->batch_out[s->batch_slot] contents],
               s->out_dim * sizeof(float));
    }
}

// Encode a single matvec reading from buf_expert_act into buf_expert_out.
static void gpu_encode_dequant_matvec_with_io_bufs(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    const void *W, const void *scales, const void *biases,
    id<MTLBuffer> in_buf, id<MTLBuffer> out_buf,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size
) {
    NSUInteger w_off = (NSUInteger)((const char *)W      - (const char *)[ctx->wf_buf contents]);
    NSUInteger s_off = (NSUInteger)((const char *)scales  - (const char *)[ctx->wf_buf contents]);
    NSUInteger b_off = (NSUInteger)((const char *)biases  - (const char *)[ctx->wf_buf contents]);

    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    int use_v3 = (in_dim <= 4096);
    [enc setComputePipelineState: use_v3 ? ctx->matvec_v3 : ctx->matvec_fast];
    [enc setBuffer:ctx->wf_buf offset:w_off atIndex:0];
    [enc setBuffer:ctx->wf_buf offset:s_off atIndex:1];
    [enc setBuffer:ctx->wf_buf offset:b_off atIndex:2];
    [enc setBuffer:in_buf      offset:0     atIndex:3];
    [enc setBuffer:out_buf     offset:0     atIndex:4];
    [enc setBytes:&out_dim     length:4     atIndex:5];
    [enc setBytes:&in_dim      length:4     atIndex:6];
    [enc setBytes:&group_size  length:4     atIndex:7];

    if (use_v3) {
        uint32_t num_tgs = (out_dim + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        [enc dispatchThreadgroups:MTLSizeMake(out_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    [enc endEncoding];
}

// Encode one expert forward using multi-expert slot k.
void gpu_encode_expert_forward_slot(
    FlashMoE_Context *m,
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int k
) {
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
    if (m->use_2bit) {
        gate_w_off = m->cfg.layout_2bit.gate_w_off; gate_s_off = m->cfg.layout_2bit.gate_s_off; gate_b_off = m->cfg.layout_2bit.gate_b_off;
        up_w_off   = m->cfg.layout_2bit.up_w_off;   up_s_off   = m->cfg.layout_2bit.up_s_off;   up_b_off   = m->cfg.layout_2bit.up_b_off;
        down_w_off = m->cfg.layout_2bit.down_w_off; down_s_off = m->cfg.layout_2bit.down_s_off; down_b_off = m->cfg.layout_2bit.down_b_off;
    } else {
        gate_w_off = m->cfg.layout_4bit.gate_w_off; gate_s_off = m->cfg.layout_4bit.gate_s_off; gate_b_off = m->cfg.layout_4bit.gate_b_off;
        up_w_off   = m->cfg.layout_4bit.up_w_off;   up_s_off   = m->cfg.layout_4bit.up_s_off;   up_b_off   = m->cfg.layout_4bit.up_b_off;
        down_w_off = m->cfg.layout_4bit.down_w_off; down_s_off = m->cfg.layout_4bit.down_s_off; down_b_off = m->cfg.layout_4bit.down_b_off;
    }
    id<MTLComputePipelineState> expert_pipe = m->use_2bit ? ctx->matvec_2bit : ctx->matvec_v3;

    uint32_t gate_up_out = m->cfg.moe_intermediate;
    uint32_t gate_up_in  = m->cfg.hidden_dim;
    uint32_t down_out    = m->cfg.hidden_dim;
    uint32_t down_in     = m->cfg.moe_intermediate;
    uint32_t gs          = m->cfg.group_size;

    // gate_proj: data[k] -> gate[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:gate_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:gate_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_gate[k]   offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // up_proj: data[k] -> up[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:up_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:up_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_up[k]     offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // SwiGLU: gate[k], up[k] -> act[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_up[k]   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // down_proj: act[k] -> out[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:down_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:down_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_out[k]  offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
}

// Encode one expert forward using explicit data buffer (for double buffering).
void gpu_encode_expert_forward_slot_buf(
    FlashMoE_Context *m,
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int k,
    id<MTLBuffer> data_buf
) {
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
    if (m->use_2bit) {
        gate_w_off = m->cfg.layout_2bit.gate_w_off; gate_s_off = m->cfg.layout_2bit.gate_s_off; gate_b_off = m->cfg.layout_2bit.gate_b_off;
        up_w_off   = m->cfg.layout_2bit.up_w_off;   up_s_off   = m->cfg.layout_2bit.up_s_off;   up_b_off   = m->cfg.layout_2bit.up_b_off;
        down_w_off = m->cfg.layout_2bit.down_w_off; down_s_off = m->cfg.layout_2bit.down_s_off; down_b_off = m->cfg.layout_2bit.down_b_off;
    } else {
        gate_w_off = m->cfg.layout_4bit.gate_w_off; gate_s_off = m->cfg.layout_4bit.gate_s_off; gate_b_off = m->cfg.layout_4bit.gate_b_off;
        up_w_off   = m->cfg.layout_4bit.up_w_off;   up_s_off   = m->cfg.layout_4bit.up_s_off;   up_b_off   = m->cfg.layout_4bit.up_b_off;
        down_w_off = m->cfg.layout_4bit.down_w_off; down_s_off = m->cfg.layout_4bit.down_s_off; down_b_off = m->cfg.layout_4bit.down_b_off;
    }
    id<MTLComputePipelineState> expert_pipe = m->use_2bit ? ctx->matvec_2bit : ctx->matvec_v3;

    uint32_t gate_up_out = m->cfg.moe_intermediate;
    uint32_t gate_up_in  = m->cfg.hidden_dim;
    uint32_t down_out    = m->cfg.hidden_dim;
    uint32_t down_in     = m->cfg.moe_intermediate;
    uint32_t gs          = m->cfg.group_size;

    // gate_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:data_buf                        offset:gate_w_off  atIndex:0];
        [enc setBuffer:data_buf                        offset:gate_s_off  atIndex:1];
        [enc setBuffer:data_buf                        offset:gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_gate[k]   offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // up_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:data_buf                        offset:up_w_off  atIndex:0];
        [enc setBuffer:data_buf                        offset:up_s_off  atIndex:1];
        [enc setBuffer:data_buf                        offset:up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_up[k]     offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // SwiGLU
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_up[k]   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // down_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:data_buf                        offset:down_w_off  atIndex:0];
        [enc setBuffer:data_buf                        offset:down_s_off  atIndex:1];
        [enc setBuffer:data_buf                        offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_out[k]    offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
}

// Batched expert encoding: encode K experts using fused or per-expert encoders.
static void gpu_encode_experts_batched(
    FlashMoE_Context *m,
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int K,
    const int *valid,
    id<MTLBuffer> __strong *expert_bufs
) {
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
    if (m->use_2bit) {
        gate_w_off = m->cfg.layout_2bit.gate_w_off; gate_s_off = m->cfg.layout_2bit.gate_s_off; gate_b_off = m->cfg.layout_2bit.gate_b_off;
        up_w_off   = m->cfg.layout_2bit.up_w_off;   up_s_off   = m->cfg.layout_2bit.up_s_off;   up_b_off   = m->cfg.layout_2bit.up_b_off;
        down_w_off = m->cfg.layout_2bit.down_w_off; down_s_off = m->cfg.layout_2bit.down_s_off; down_b_off = m->cfg.layout_2bit.down_b_off;
    } else {
        gate_w_off = m->cfg.layout_4bit.gate_w_off; gate_s_off = m->cfg.layout_4bit.gate_s_off; gate_b_off = m->cfg.layout_4bit.gate_b_off;
        up_w_off   = m->cfg.layout_4bit.up_w_off;   up_s_off   = m->cfg.layout_4bit.up_s_off;   up_b_off   = m->cfg.layout_4bit.up_b_off;
        down_w_off = m->cfg.layout_4bit.down_w_off; down_s_off = m->cfg.layout_4bit.down_s_off; down_b_off = m->cfg.layout_4bit.down_b_off;
    }
    id<MTLComputePipelineState> expert_pipe = m->use_2bit ? ctx->matvec_2bit : ctx->matvec_v3;

    uint32_t gate_up_out = m->cfg.moe_intermediate;
    uint32_t gate_up_in  = m->cfg.hidden_dim;
    uint32_t down_out    = m->cfg.hidden_dim;
    uint32_t down_in     = m->cfg.moe_intermediate;
    uint32_t gs          = m->cfg.group_size;
    uint32_t gate_up_tgs = (gate_up_out + 7) / 8;
    uint32_t down_tgs    = (down_out + 7) / 8;
    uint32_t swiglu_tgs  = (gate_up_out + 255) / 256;

#if USE_FUSED_GATE_UP_SWIGLU
    if (!m->use_2bit) {
        uint32_t fused_tgs = gate_up_out;
        for (int k = 0; k < K; k++) {
            if (!valid[k]) continue;

            // Encoder A: fused_gate_up_swiglu
            {
                id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                [enc setComputePipelineState:ctx->fused_gate_up_swiglu];
                [enc setBuffer:expert_bufs[k]                  offset:gate_w_off  atIndex:0];
                [enc setBuffer:expert_bufs[k]                  offset:gate_s_off  atIndex:1];
                [enc setBuffer:expert_bufs[k]                  offset:gate_b_off  atIndex:2];
                [enc setBuffer:expert_bufs[k]                  offset:up_w_off    atIndex:3];
                [enc setBuffer:expert_bufs[k]                  offset:up_s_off    atIndex:4];
                [enc setBuffer:expert_bufs[k]                  offset:up_b_off    atIndex:5];
                [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:6];
                [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:7];
                [enc setBytes:&gate_up_out length:4 atIndex:8];
                [enc setBytes:&gate_up_in  length:4 atIndex:9];
                [enc setBytes:&gs          length:4 atIndex:10];
                [enc dispatchThreadgroups:MTLSizeMake(fused_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            // Encoder B: down_proj only
            {
                id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                [enc setComputePipelineState:expert_pipe];
                [enc setBuffer:expert_bufs[k]                  offset:down_w_off  atIndex:0];
                [enc setBuffer:expert_bufs[k]                  offset:down_s_off  atIndex:1];
                [enc setBuffer:expert_bufs[k]                  offset:down_b_off  atIndex:2];
                [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:3];
                [enc setBuffer:ctx->buf_multi_expert_out[k]    offset:0           atIndex:4];
                [enc setBytes:&down_out length:4 atIndex:5];
                [enc setBytes:&down_in  length:4 atIndex:6];
                [enc setBytes:&gs       length:4 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(down_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
        }
        return;
    }
#endif
    // Original path: gate+up encoded together, SwiGLU+down encoded together
    for (int k = 0; k < K; k++) {
        if (!valid[k]) continue;

        // Encoder A: gate_proj + up_proj
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:expert_pipe];
            [enc setBuffer:expert_bufs[k]                  offset:gate_w_off  atIndex:0];
            [enc setBuffer:expert_bufs[k]                  offset:gate_s_off  atIndex:1];
            [enc setBuffer:expert_bufs[k]                  offset:gate_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_gate[k]   offset:0           atIndex:4];
            [enc setBytes:&gate_up_out length:4 atIndex:5];
            [enc setBytes:&gate_up_in  length:4 atIndex:6];
            [enc setBytes:&gs          length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(gate_up_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc setBuffer:expert_bufs[k]                  offset:up_w_off  atIndex:0];
            [enc setBuffer:expert_bufs[k]                  offset:up_s_off  atIndex:1];
            [enc setBuffer:expert_bufs[k]                  offset:up_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_up[k]     offset:0          atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(gate_up_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // Encoder B: SwiGLU + down_proj
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:ctx->swiglu];
            [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_multi_expert_up[k]   offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0 atIndex:2];
            [enc setBytes:&gate_up_out length:4 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc setComputePipelineState:expert_pipe];
            [enc setBuffer:expert_bufs[k]                  offset:down_w_off  atIndex:0];
            [enc setBuffer:expert_bufs[k]                  offset:down_s_off  atIndex:1];
            [enc setBuffer:expert_bufs[k]                  offset:down_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_out[k]    offset:0           atIndex:4];
            [enc setBytes:&down_out length:4 atIndex:5];
            [enc setBytes:&down_in  length:4 atIndex:6];
            [enc setBytes:&gs       length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(down_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
    }
}

// Encode one expert forward (gate+up+swiglu+down) into cmdbuf (single expert, legacy).
__attribute__((unused))
static void gpu_encode_expert_forward(
    FlashMoE_Context *m,
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf
) {
    NSUInteger gate_w_off = m->cfg.layout_4bit.gate_w_off;
    NSUInteger gate_s_off = m->cfg.layout_4bit.gate_s_off;
    NSUInteger gate_b_off = m->cfg.layout_4bit.gate_b_off;
    NSUInteger up_w_off   = m->cfg.layout_4bit.up_w_off;
    NSUInteger up_s_off   = m->cfg.layout_4bit.up_s_off;
    NSUInteger up_b_off   = m->cfg.layout_4bit.up_b_off;
    NSUInteger down_w_off = m->cfg.layout_4bit.down_w_off;
    NSUInteger down_s_off = m->cfg.layout_4bit.down_s_off;
    NSUInteger down_b_off = m->cfg.layout_4bit.down_b_off;

    uint32_t gate_up_out = m->cfg.moe_intermediate;
    uint32_t gate_up_in  = m->cfg.hidden_dim;
    uint32_t down_out    = m->cfg.hidden_dim;
    uint32_t down_in     = m->cfg.moe_intermediate;
    uint32_t gs          = m->cfg.group_size;

    // gate_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->matvec_v3];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_gate  offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // up_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->matvec_v3];
        [enc setBuffer:ctx->buf_expert_data  offset:up_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:up_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_expert_up    offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // SwiGLU
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_expert_gate offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_expert_up   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_expert_act  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // down_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->matvec_v3];
        [enc setBuffer:ctx->buf_expert_data offset:down_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data offset:down_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_act  offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_out  offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
}

// Batched wrapper: takes N matmul specs sharing the same input.
static void fast_batch_matvec(
    FlashMoE_Context *m,
    const float *x, uint32_t x_dim,
    BatchMatvecSpec *specs, int num_specs
) {
    if (m->metal && m->metal->wf_buf) {
        gpu_batch_matvec(m->metal, x, x_dim, specs, num_specs);
    } else {
        for (int i = 0; i < num_specs; i++) {
            BatchMatvecSpec *s = &specs[i];
            cpu_dequant_matvec(s->W, s->scales, s->biases, x, s->out_cpu,
                               s->out_dim, s->in_dim, s->group_size);
        }
    }
}

// ============================================================================
// GPU expert forward: gate+up matvec -> SwiGLU -> down matvec
// ============================================================================

__attribute__((unused))
static void gpu_expert_forward(
    FlashMoE_Context *m,
    MetalCtx *ctx,
    const void *expert_data,
    const float *h_post,
    float *expert_out,
    int expert_data_already_in_buffer
) {
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
    if (m->use_2bit) {
        gate_w_off = m->cfg.layout_2bit.gate_w_off; gate_s_off = m->cfg.layout_2bit.gate_s_off; gate_b_off = m->cfg.layout_2bit.gate_b_off;
        up_w_off   = m->cfg.layout_2bit.up_w_off;   up_s_off   = m->cfg.layout_2bit.up_s_off;   up_b_off   = m->cfg.layout_2bit.up_b_off;
        down_w_off = m->cfg.layout_2bit.down_w_off; down_s_off = m->cfg.layout_2bit.down_s_off; down_b_off = m->cfg.layout_2bit.down_b_off;
    } else {
        gate_w_off = m->cfg.layout_4bit.gate_w_off; gate_s_off = m->cfg.layout_4bit.gate_s_off; gate_b_off = m->cfg.layout_4bit.gate_b_off;
        up_w_off   = m->cfg.layout_4bit.up_w_off;   up_s_off   = m->cfg.layout_4bit.up_s_off;   up_b_off   = m->cfg.layout_4bit.up_b_off;
        down_w_off = m->cfg.layout_4bit.down_w_off; down_s_off = m->cfg.layout_4bit.down_s_off; down_b_off = m->cfg.layout_4bit.down_b_off;
    }
    id<MTLComputePipelineState> expert_pipe = m->use_2bit ? ctx->matvec_2bit : ctx->matvec_v3;

    if (!expert_data_already_in_buffer) {
        memcpy([ctx->buf_expert_data contents], expert_data, active_expert_size(m));
    }
    memcpy([ctx->buf_expert_input contents], h_post, m->cfg.hidden_dim * sizeof(float));

    uint32_t gate_up_out = m->cfg.moe_intermediate;
    uint32_t gate_up_in  = m->cfg.hidden_dim;
    uint32_t down_out    = m->cfg.hidden_dim;
    uint32_t down_in     = m->cfg.moe_intermediate;
    uint32_t gs          = m->cfg.group_size;

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];

    // gate_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_gate  offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // up_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_expert_data  offset:up_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:up_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_expert_up    offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // SwiGLU
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_expert_gate offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_expert_up   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_expert_act  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // down_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_expert_data offset:down_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data offset:down_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_act  offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_out  offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    memcpy(expert_out, [ctx->buf_expert_out contents], m->cfg.hidden_dim * sizeof(float));
}

// ============================================================================
// Rotary position embedding (for full attention layers)
// ============================================================================

static void apply_rotary_emb(float *q, float *k, int pos, int num_heads, int num_kv_heads,
                              int head_dim, int rotary_dim, float rope_theta) {
    int half = rotary_dim / 2;
    for (int h = 0; h < num_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(rope_theta, (float)(2 * i) / rotary_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float q0 = qh[i];
            float q1 = qh[i + half];
            qh[i]        = q0 * cos_a - q1 * sin_a;
            qh[i + half]  = q0 * sin_a + q1 * cos_a;
        }
    }
    for (int h = 0; h < num_kv_heads; h++) {
        float *kh = k + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(rope_theta, (float)(2 * i) / rotary_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float k0 = kh[i];
            float k1 = kh[i + half];
            kh[i]        = k0 * cos_a - k1 * sin_a;
            kh[i + half]  = k0 * sin_a + k1 * cos_a;
        }
    }
}


#endif // GPU_OPS_H
