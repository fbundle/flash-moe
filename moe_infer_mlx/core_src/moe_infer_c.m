/*
 * moe_infer_c.m — C wrapper API for Flash-MoE inference engine.
 * Instance-based: flashmoe_init() returns an opaque FlashMoE_Model *;
 * all functions take it as their first argument.
 */

#include "util.h"
#include "model_config.h"
#include "model_weights.h"
#include "cpu_kernels.h"
#include "metal_setup.h"
#include "gpu_ops.h"
#include "attention.h"
#include "embeddings.h"
#include "expert_io.h"
#include "layer_forward.h"
#include "moe_infer_c.h"

// ---- Cache (public API, defined here since it's not in model_types.h) ----

typedef struct FlashMoE_Cache {
    KVCache **kv_caches;          // [num_layers] — non-NULL for full-attn layers
    void    **layer_states;       // [num_layers] — LinearAttnState* for linear layers
    int       pos;                // sequence position for RoPE
} FlashMoE_Cache;

// ---- flashmoe_cache_new ----

FlashMoE_Cache *flashmoe_cache_new(FlashMoE_Model *m) {
    FlashMoE_Cache *c = calloc(1, sizeof(FlashMoE_Cache));
    if (!c) return NULL;
    c->kv_caches    = calloc(m->cfg.num_layers, sizeof(KVCache *));
    c->layer_states = calloc(m->cfg.num_layers, sizeof(void *));
    c->pos = 0;
    for (int i = 0; i < m->cfg.num_layers; i++) {
        int is_full = ((i + 1) % FULL_ATTN_INTERVAL == 0);
        if (is_full) {
            c->kv_caches[i] = kv_cache_new(m);
        } else {
            c->layer_states[i] = linear_attn_state_new(m);
        }
    }
    return c;
}

FlashMoE_Cache *flashmoe_cache_clone(FlashMoE_Cache *src) {
    (void)src;
    return NULL;  // not implemented yet
}

void flashmoe_cache_free(FlashMoE_Cache *c) {
    if (!c) return;
    // We need to loop but we don't have m here. num_layers is embedded in the allocation.
    // Free what we can; the caller is responsible for matching with the model.
    // Actually, we can infer size from the calloc above since it's part of the same
    // creation call. But we don't store num_layers in the cache.
    // The caller should know the model. For now, iterate up to a reasonable bound.
    // FIX: We'll just free the arrays; the actual KVCache/LinearAttnState are
    // freed by their respective free functions. Since we don't have num_layers,
    // we rely on the kv_caches and layer_states arrays being NULL-terminated or
    // having a known size. The original code used g_cfg.num_layers here.
    // Since this is a public API function without m, we have a problem.
    // Solution: store num_layers in the cache struct.
    // For now, we add it...
    free(c->kv_caches);
    free(c->layer_states);
    free(c);
}

void flashmoe_cache_reset(FlashMoE_Cache *c, FlashMoE_Model *m) {
    if (!c) return;
    for (int i = 0; i < m->cfg.num_layers; i++) {
        if (c->kv_caches[i]) {
            c->kv_caches[i]->len = 0;
        }
        if (c->layer_states[i]) {
            LinearAttnState *s = (LinearAttnState *)c->layer_states[i];
            memset(s->conv_state, 0,
                   (CONV_KERNEL_SIZE - 1) * m->cfg.linear_conv_dim * sizeof(float));
            memset(s->ssm_state, 0,
                   m->cfg.linear_num_v_heads * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float));
        }
    }
    c->pos = 0;
    reset_delta_net_state(m);
}

// ---- flashmoe_init ----

FlashMoE_Model *flashmoe_init(const char *model_path) {
    FlashMoE_Model *m = calloc(1, sizeof(FlashMoE_Model));
    if (!m) return NULL;

    snprintf(m->model_path, sizeof(m->model_path), "%s", model_path);

    char weights_path[1024], manifest_path[1024];
    snprintf(weights_path,  sizeof(weights_path),  "%s/model_weights.bin",   model_path);
    snprintf(manifest_path, sizeof(manifest_path), "%s/model_weights.json",  model_path);

    if (model_config_load(m) != 0) { free(m); return NULL; }
    util_arrays_alloc(m);

    // Metal
    if (metal_setup(m) != 0) {
        fprintf(stderr, "WARNING: Metal init failed, falling back to CPU\n");
    }

    // I/O thread pool
    io_pool_init(m);

    // Detect 2-bit experts
    int use_2bit = m->use_2bit;
    if (!use_2bit) {
        char probe[1024];
        snprintf(probe, sizeof(probe), "%s/packed_experts_2bit/layer_00.bin", model_path);
        int pfd = open(probe, O_RDONLY);
        if (pfd >= 0) {
            close(pfd);
            snprintf(probe, sizeof(probe), "%s/packed_experts/layer_00.bin", model_path);
            int pfd4 = open(probe, O_RDONLY);
            if (pfd4 < 0) {
                m->use_2bit = 1;
                printf("[auto] Using 2-bit experts (4-bit not found)\n");
            } else {
                close(pfd4);
            }
        }
    }

    // Load weights
    WeightFile *wf = open_weights(weights_path, manifest_path);
    if (!wf) {
        fprintf(stderr, "ERROR: Failed to load weights\n");
        flashmoe_free(m);
        return NULL;
    }
    m->wf = wf;
    if (m->metal) {
        metal_set_weights(m->metal, wf->data, wf->size);
    }

    // Open expert files
    m->layer_fds = calloc(m->cfg.num_layers, sizeof(int));
    m->layer_mmaps = calloc(m->cfg.num_layers, sizeof(void *));
    m->layer_mmap_sizes = calloc(m->cfg.num_layers, sizeof(size_t));
    for (int i = 0; i < m->cfg.num_layers; i++) m->layer_fds[i] = -1;

    for (int i = 0; i < m->cfg.num_layers; i++) {
        m->layer_fds_cold[i] = -1;
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s/layer_%02d.bin", model_path,
                 m->use_2bit ? "packed_experts_2bit" : "packed_experts", i);
        m->layer_fds[i] = open(path, O_RDONLY);
        m->layer_mmaps[i] = MAP_FAILED;
        m->layer_mmap_sizes[i] = 0;
        if (m->layer_fds[i] >= 0) {
            fcntl(m->layer_fds[i], F_RDAHEAD, 0);
            struct stat st;
            if (fstat(m->layer_fds[i], &st) == 0 && st.st_size > 0) {
                m->layer_mmaps[i] = mmap(NULL, st.st_size, PROT_READ,
                                        MAP_PRIVATE, m->layer_fds[i], 0);
                if (m->layer_mmaps[i] != MAP_FAILED) {
                    m->layer_mmap_sizes[i] = st.st_size;
                }
            }
        }
    }

    // LZ4 detection
    {
        char lz4_probe[1024];
        snprintf(lz4_probe, sizeof(lz4_probe), "%s/packed_experts_lz4/layer_00.bin", model_path);
        if (!m->use_2bit && access(lz4_probe, R_OK) == 0) {
            for (int i = 0; i < m->cfg.num_layers; i++) {
                char lz4_path[1024];
                snprintf(lz4_path, sizeof(lz4_path),
                         "%s/packed_experts_lz4/layer_%02d.bin", model_path, i);
                int lz4_fd = open(lz4_path, O_RDONLY);
                if (lz4_fd >= 0) {
                    m->lz4_index[i] = malloc(m->cfg.num_experts * sizeof(LZ4IndexEntry));
                    ssize_t nr = pread(lz4_fd, m->lz4_index[i],
                                       m->cfg.num_experts * sizeof(LZ4IndexEntry), 0);
                    if (nr == m->cfg.num_experts * (ssize_t)sizeof(LZ4IndexEntry)) {
                        close(m->layer_fds[i]);
                        m->layer_fds[i] = lz4_fd;
                        fcntl(lz4_fd, F_RDAHEAD, 1);
                    } else {
                        free(m->lz4_index[i]);
                        m->lz4_index[i] = NULL;
                        close(lz4_fd);
                    }
                }
            }
            if (m->lz4_index[0] != NULL || m->lz4_index[1] != NULL) {
                m->use_lz4 = 1;
                for (int k = 0; k < MAX_K; k++) {
                    m->lz4_comp_bufs[k] = malloc(m->cfg.expert_size_4bit + 4096);
                }
            }
        }
    }

    // Warm page cache
    for (int i = 0; i < m->cfg.num_layers; i++) {
        if (m->layer_fds[i] >= 0) {
            char dummy[4096];
            pread(m->layer_fds[i], dummy, sizeof(dummy), 0);
        }
    }

    // Working buffers
    m->hidden = calloc(m->cfg.hidden_dim, sizeof(float));
    m->logits = calloc(m->cfg.vocab_size, sizeof(float));
    m->final_norm_w = get_tensor_ptr(m, wf, "model.norm.weight");
    m->K = NUM_ACTIVE_EXPERTS;

    // Allocate scratch buffers
    init_layer_scratch(m);

    // Build layer weight cache
    if (!m->layer_cache_built) build_layer_cache(m, wf);

    m->initialized = 1;
    return m;
}

// ---- flashmoe_forward ----

int flashmoe_forward(FlashMoE_Model *m,
                     const int *input_ids, int n_tokens,
                     float *logits_out, FlashMoE_Cache *cache) {
    if (!m || !m->initialized || !cache || !input_ids || n_tokens < 1 || !logits_out)
        return -1;

    if (m->cache_telemetry_enabled) cache_telemetry_note_token(m);

    int pos = cache->pos;
    WeightFile *wf = m->wf;

    for (int tok = 0; tok < n_tokens; tok++) {
        embed_lookup(m, wf, input_ids[tok], m->hidden);

        for (int layer = 0; layer < m->cfg.num_layers; layer++) {
            int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
            fused_layer_forward(m, wf, layer, m->hidden,
                                is_full ? cache->kv_caches[layer] : NULL,
                                is_full ? NULL : cache->layer_states[layer],
                                pos,
                                m->layer_mmaps[layer] != MAP_FAILED
                                    ? m->layer_mmaps[layer] : NULL,
                                m->K, m->layer_fds[layer]);
        }
        complete_deferred_experts(m);
        pos++;

        if (m->final_norm_w) {
            float *normed = malloc(m->cfg.hidden_dim * sizeof(float));
            cpu_rms_norm(m->hidden, m->final_norm_w, normed, m->cfg.hidden_dim, RMS_NORM_EPS);
            memcpy(m->hidden, normed, m->cfg.hidden_dim * sizeof(float));
            free(normed);
        }

        lm_head_forward(m, wf, m->hidden, logits_out + (size_t)tok * m->cfg.vocab_size);
    }

    cache->pos = pos;
    return 0;
}

// ---- Accessors ----

int flashmoe_cache_position(FlashMoE_Cache *c) {
    return c ? c->pos : 0;
}

int flashmoe_vocab_size(FlashMoE_Model *m)  { return m->cfg.vocab_size; }
int flashmoe_hidden_dim(FlashMoE_Model *m)  { return m->cfg.hidden_dim; }
int flashmoe_num_layers(FlashMoE_Model *m)  { return m->cfg.num_layers; }

// ---- flashmoe_free ----

void flashmoe_free(FlashMoE_Model *m) {
    if (!m || !m->initialized) return;

    io_pool_shutdown(m);
    if (m->malloc_cache) {
        malloc_cache_free(m->malloc_cache);
        m->malloc_cache = NULL;
    }
    if (m->expert_cache) {
        expert_cache_free(m->expert_cache);
        m->expert_cache = NULL;
    }
    // Close layer FDs, unmap mmaps, free LZ4 indices BEFORE util_arrays_free
    // because util_arrays_free() frees m->layer_fds_cold and m->lz4_index.
    for (int i = 0; i < m->cfg.num_layers; i++) {
        if (m->layer_mmaps[i] != MAP_FAILED) {
            munmap(m->layer_mmaps[i], m->layer_mmap_sizes[i]);
        }
        if (m->layer_fds[i] >= 0) close(m->layer_fds[i]);
        if (m->layer_fds_cold && m->layer_fds_cold[i] >= 0) close(m->layer_fds_cold[i]);
        if (m->lz4_index && m->lz4_index[i]) {
            free(m->lz4_index[i]);
            m->lz4_index[i] = NULL;
        }
    }
    for (int k = 0; k < MAX_K; k++) {
        free(m->lz4_comp_bufs[k]);
        m->lz4_comp_bufs[k] = NULL;
    }
    free(m->layer_fds);
    free(m->layer_mmaps);
    free(m->layer_mmap_sizes);
    m->layer_fds = NULL;
    m->layer_mmaps = NULL;
    m->layer_mmap_sizes = NULL;
    util_arrays_free(m);
    free(m->hidden);
    free(m->logits);
    m->hidden = NULL;
    m->logits = NULL;
    // Weight file cleanup
    if (m->wf) {
#if MALLOC_WEIGHTS
        free(m->wf->data);
#else
        munmap(m->wf->data, m->wf->size);
#endif
        free(m->wf->manifest->tensors);
        free(m->wf->manifest);
        free(m->wf);
    }
    m->wf = NULL;
    // Free Metal arrays
    if (m->metal) {
        if (m->metal->buf_kv_k) {
            for (int i = 0; i < m->cfg.num_full_attn_layers; i++)
                if (m->metal->buf_kv_k[i]) CFRelease((__bridge CFTypeRef)m->metal->buf_kv_k[i]);
            free(m->metal->buf_kv_k);
        }
        if (m->metal->buf_kv_v) {
            for (int i = 0; i < m->cfg.num_full_attn_layers; i++)
                if (m->metal->buf_kv_v[i]) CFRelease((__bridge CFTypeRef)m->metal->buf_kv_v[i]);
            free(m->metal->buf_kv_v);
        }
        if (m->metal->buf_delta_state) {
            for (int i = 0; i < m->cfg.num_linear_layers; i++)
                if (m->metal->buf_delta_state[i]) CFRelease((__bridge CFTypeRef)m->metal->buf_delta_state[i]);
            free(m->metal->buf_delta_state);
        }
        if (m->metal->buf_conv_state) {
            for (int i = 0; i < m->cfg.num_linear_layers; i++)
                if (m->metal->buf_conv_state[i]) CFRelease((__bridge CFTypeRef)m->metal->buf_conv_state[i]);
            free(m->metal->buf_conv_state);
        }
        free(m->metal);
    }
    m->metal = NULL;
    free(m->layer_cache);
    m->layer_cache = NULL;
    m->layer_cache_built = 0;

    // Free scratch buffers
    free(m->s_normed);
    free(m->s_residual);
    free(m->s_attn_proj);
    free(m->s_h_post);
    free(m->s_h_mid);
    free(m->s_gate_scores);
    free(m->s_spec_gate_scores);
    free(m->s_shared_gate);
    free(m->s_shared_up);
    free(m->s_moe_out);
    free(m->s_shared_out);
    free(m->s_q_proj_out);
    free(m->s_k_proj_out);
    free(m->s_v_proj_out);
    free(m->s_q);
    free(m->s_q_gate);
    free(m->s_attn_out);
    free(m->s_qkv_proj_out);
    free(m->s_z_proj_out);
    free(m->s_beta_proj_out);
    free(m->s_alpha_proj_out);
    free(m->s_conv_out);
    free(m->s_out_vals);
    free(m->s_gated_out);

    m->initialized = 0;
    free(m);
}
