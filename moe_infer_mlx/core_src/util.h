#ifndef UTIL_H
#define UTIL_H

// Utilities: timing, cache telemetry, bf16 conversion, dynamic array alloc.
// Types (LayerTimingAccum, CacheTelemetry, LZ4IndexEntry) are in model_types.h.
// All state accessed via FlashMoE_Model *m.

#include "model_internal.h"

// ============================================================================
// Timing helper
// ============================================================================

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static inline int expert_is_seen(FlashMoE_Model *m, int layer, int expert) {
    return (m->expert_seen[layer * (m->cfg.num_experts / 8) + (expert >> 3)] >> (expert & 7)) & 1;
}
static inline void expert_mark_seen(FlashMoE_Model *m, int layer, int expert) {
    m->expert_seen[layer * (m->cfg.num_experts / 8) + (expert >> 3)] |= (1 << (expert & 7));
}
static inline int expert_pick_fd(FlashMoE_Model *m, int layer, int expert, int warm_fd) {
    (void)m; (void)layer; (void)expert;
    return warm_fd;
}

static inline size_t active_expert_size(FlashMoE_Model *m) {
    return m->use_2bit ? m->cfg.expert_size_2bit : m->cfg.expert_size_4bit;
}

void cache_telemetry_reset(FlashMoE_Model *m) {
    memset(&m->cache_telemetry, 0, sizeof(m->cache_telemetry));
    if (m->cache_seen)
        memset(m->cache_seen, 0, (size_t)m->cfg.num_layers * m->cfg.num_experts * sizeof(uint8_t));
    if (m->cache_last_touch_token)
        memset(m->cache_last_touch_token, 0, (size_t)m->cfg.num_layers * m->cfg.num_experts * sizeof(uint64_t));
    if (m->cache_last_evict_token)
        memset(m->cache_last_evict_token, 0, (size_t)m->cfg.num_layers * m->cfg.num_experts * sizeof(uint64_t));
}

static void cache_telemetry_note_token(FlashMoE_Model *m) {
    if (!m->cache_telemetry_enabled) return;
    m->cache_telemetry.token_clock++;
}

static void cache_telemetry_touch(FlashMoE_Model *m, int layer_idx, int expert_idx) {
    if (!m->cache_telemetry_enabled) return;
    if (layer_idx < 0 || layer_idx >= m->cfg.num_layers || expert_idx < 0 || expert_idx >= m->cfg.num_experts) return;
    if (!m->cache_seen[(size_t)(layer_idx) * m->cfg.num_experts + (expert_idx)]) {
        m->cache_seen[(size_t)(layer_idx) * m->cfg.num_experts + (expert_idx)] = 1;
        m->cache_telemetry.unique_experts_touched++;
    }
    m->cache_last_touch_token[(size_t)(layer_idx) * m->cfg.num_experts + (expert_idx)] = m->cache_telemetry.token_clock;
}

static void cache_telemetry_miss(FlashMoE_Model *m, int layer_idx, int expert_idx) {
    if (!m->cache_telemetry_enabled) return;
    if (layer_idx < 0 || layer_idx >= m->cfg.num_layers || expert_idx < 0 || expert_idx >= m->cfg.num_experts) return;
    if (!m->cache_seen[(size_t)(layer_idx) * m->cfg.num_experts + (expert_idx)]) {
        m->cache_telemetry.cold_misses++;
        m->cache_seen[(size_t)(layer_idx) * m->cfg.num_experts + (expert_idx)] = 1;
        m->cache_telemetry.unique_experts_touched++;
    } else {
        m->cache_telemetry.eviction_misses++;
        uint64_t dist = 0;
        if (m->cache_last_evict_token[(size_t)(layer_idx) * m->cfg.num_experts + (expert_idx)] > 0 &&
            m->cache_telemetry.token_clock >= m->cache_last_evict_token[(size_t)(layer_idx) * m->cfg.num_experts + (expert_idx)]) {
            dist = m->cache_telemetry.token_clock - m->cache_last_evict_token[(size_t)(layer_idx) * m->cfg.num_experts + (expert_idx)];
        }
        if (dist <= 1) m->cache_telemetry.reuse_le_1++;
        else if (dist <= 4) m->cache_telemetry.reuse_le_4++;
        else if (dist <= 16) m->cache_telemetry.reuse_le_16++;
        else if (dist <= 64) m->cache_telemetry.reuse_le_64++;
        else m->cache_telemetry.reuse_gt_64++;
        m->cache_telemetry.reuse_distance_sum += dist;
        m->cache_telemetry.reuse_distance_samples++;
    }
    m->cache_last_touch_token[(size_t)(layer_idx) * m->cfg.num_experts + (expert_idx)] = m->cache_telemetry.token_clock;
}

static void cache_telemetry_evict(FlashMoE_Model *m, int layer_idx, int expert_idx) {
    if (!m->cache_telemetry_enabled) return;
    if (layer_idx < 0 || layer_idx >= m->cfg.num_layers || expert_idx < 0 || expert_idx >= m->cfg.num_experts) return;
    m->cache_telemetry.evictions++;
    m->cache_last_evict_token[(size_t)(layer_idx) * m->cfg.num_experts + (expert_idx)] = m->cache_telemetry.token_clock;
}

void cache_telemetry_print(FlashMoE_Model *m, uint64_t hits, uint64_t misses) {
    if (!m->cache_telemetry_enabled) return;
    uint64_t total = hits + misses;
    fprintf(stderr, "\n=== Cache Telemetry ===\n");
    fprintf(stderr, "Tokens tracked: %llu\n", m->cache_telemetry.token_clock);
    fprintf(stderr, "Unique experts touched: %llu / %d (%.1f%%)\n",
            m->cache_telemetry.unique_experts_touched,
            m->cfg.num_layers * m->cfg.num_experts,
            100.0 * m->cache_telemetry.unique_experts_touched / (m->cfg.num_layers * m->cfg.num_experts));
    fprintf(stderr, "Miss breakdown: cold %llu (%.1f%% of misses), eviction %llu (%.1f%% of misses)\n",
            m->cache_telemetry.cold_misses,
            misses > 0 ? 100.0 * m->cache_telemetry.cold_misses / misses : 0.0,
            m->cache_telemetry.eviction_misses,
            misses > 0 ? 100.0 * m->cache_telemetry.eviction_misses / misses : 0.0);
    fprintf(stderr, "Evictions: %llu\n", m->cache_telemetry.evictions);
    fprintf(stderr, "Eviction reuse distance: <=1 tok %llu, <=4 %llu, <=16 %llu, <=64 %llu, >64 %llu",
            m->cache_telemetry.reuse_le_1,
            m->cache_telemetry.reuse_le_4,
            m->cache_telemetry.reuse_le_16,
            m->cache_telemetry.reuse_le_64,
            m->cache_telemetry.reuse_gt_64);
    if (m->cache_telemetry.reuse_distance_samples > 0) {
        fprintf(stderr, " (avg %.1f tok)\n",
                (double)m->cache_telemetry.reuse_distance_sum / m->cache_telemetry.reuse_distance_samples);
    } else {
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "Effective hit rate: %.1f%%\n",
            total > 0 ? 100.0 * hits / total : 0.0);
}

void timing_reset(FlashMoE_Model *m) {
    memset(&m->timing, 0, sizeof(m->timing));
}

void timing_print(FlashMoE_Model *m) {
    if (m->timing.count == 0) return;
    int n = m->timing.count;
    fprintf(stderr, "\n[timing] Per-layer breakdown (avg of %d layers, ms):\n", n);
    fprintf(stderr, "  deferred_wait:  %6.3f\n", m->timing.deferred_wait / n);
    fprintf(stderr, "  deferred_cpu:   %6.3f\n", m->timing.deferred_cpu / n);
    fprintf(stderr, "  input_norm:     %6.3f\n", m->timing.input_norm / n);
    fprintf(stderr, "  cmd1_submit:    %6.3f\n", m->timing.cmd1_submit / n);
    fprintf(stderr, "  cmd1_wait:      %6.3f\n", m->timing.cmd1_wait / n);
    fprintf(stderr, "  spec_route:     %6.3f\n", m->timing.spec_route / n);
    fprintf(stderr, "  cpu_attn:       %6.3f\n", m->timing.cpu_attn / n);
    fprintf(stderr, "  cmd2_encode:    %6.3f\n", m->timing.cmd2_encode / n);
    fprintf(stderr, "  cmd2_wait:      %6.3f\n", m->timing.cmd2_wait / n);
    fprintf(stderr, "  routing_cpu:    %6.3f\n", m->timing.routing_cpu / n);
    fprintf(stderr, "  expert_io:      %6.3f\n", m->timing.expert_io / n);
    fprintf(stderr, "  cmd3_encode:    %6.3f\n", m->timing.cmd3_encode / n);
    fprintf(stderr, "  total_layer:    %6.3f\n", m->timing.total / n);
    fprintf(stderr, "  sum_phases:     %6.3f\n",
            (m->timing.deferred_wait + m->timing.deferred_cpu + m->timing.input_norm +
             m->timing.cmd1_submit + m->timing.cmd1_wait + m->timing.spec_route +
             m->timing.cpu_attn +
             m->timing.cmd2_encode + m->timing.cmd2_wait + m->timing.routing_cpu +
             m->timing.expert_io + m->timing.cmd3_encode) / n);
    fprintf(stderr, "  cmd_buffers:    %d (3 per layer: CMD1+CMD2+CMD3)\n", n * 3);
    fprintf(stderr, "  sync_waits:     %d (2 per layer: CMD1+CMD2, CMD3 deferred)\n", n * 2);
    fprintf(stderr, "  gpu_encoders:   ~%d per layer (CMD1:3-4, CMD2:8-12, CMD3:~10)\n", 22);
    if (m->pred_enabled && m->pred_layers > 0) {
        uint64_t total = m->pred_hits + m->pred_misses;
        double hit_rate = total > 0 ? (double)m->pred_hits / total * 100.0 : 0;
        fprintf(stderr, "  [predict] hits=%llu misses=%llu rate=%.1f%% layers=%llu\n",
                m->pred_hits, m->pred_misses, hit_rate, m->pred_layers);
    }
}

// ============================================================================
// bf16 <-> f32 conversion (CPU side)
// ============================================================================

static float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

__attribute__((unused))
static uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

// ---- Dynamic array allocation (called after model_config_load) ----

static int util_arrays_alloc(FlashMoE_Model *m) {
    int N = m->cfg.num_layers;
    int E = m->cfg.num_experts;

    m->lz4_index = calloc(N, sizeof(LZ4IndexEntry *));

    m->layer_fds_cold = calloc(N, sizeof(int));
    for (int i = 0; i < N; i++) m->layer_fds_cold[i] = -1;

    m->expert_seen = calloc((size_t)N * (E / 8), 1);

    m->expert_freq = calloc((size_t)N * E, sizeof(int));

    m->cache_seen = calloc((size_t)N * E, 1);
    m->cache_last_touch_token = calloc((size_t)N * E, sizeof(uint64_t));
    m->cache_last_evict_token = calloc((size_t)N * E, sizeof(uint64_t));

    m->pred_experts = calloc((size_t)N * MAX_K, sizeof(int));
    m->pred_count   = calloc(N, sizeof(int));

    return 0;
}

static void util_arrays_free(FlashMoE_Model *m) {
    free(m->lz4_index);      m->lz4_index = NULL;
    free(m->layer_fds_cold); m->layer_fds_cold = NULL;
    free(m->expert_seen);    m->expert_seen = NULL;
    free(m->expert_freq);    m->expert_freq = NULL;
    free(m->cache_seen);     m->cache_seen = NULL;
    free(m->cache_last_touch_token); m->cache_last_touch_token = NULL;
    free(m->cache_last_evict_token); m->cache_last_evict_token = NULL;
    free(m->pred_experts);  m->pred_experts = NULL;
    free(m->pred_count);    m->pred_count = NULL;
}

#endif // UTIL_H
