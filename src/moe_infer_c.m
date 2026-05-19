/*
 * moe_infer_c.m — C wrapper API for Flash-MoE inference engine.
 */

#include "util.h"
#include "tensors.h"
#include "cpu_kernels.h"
#include "metal_setup.h"
#include "gpu_ops.h"
#include "attention.h"
#include "embeddings.h"
#include "expert_io.h"
#include "layer_forward.h"
#include "moe_infer_c.h"

// ---- Cache ----

typedef struct FlashMoE_Cache {
    KVCache **kv_caches;          // [NUM_LAYERS] — non-NULL for full-attn layers
    void    **layer_states;       // [NUM_LAYERS] — LinearAttnState* for linear layers
    int       pos;                // sequence position for RoPE
} FlashMoE_Cache;

// ---- File-scope state (was local to main.m) ----

static WeightFile *g_wf = NULL;

static int   g_layer_fds[NUM_LAYERS];
static void *g_layer_mmaps[NUM_LAYERS];
static size_t g_layer_mmap_sizes[NUM_LAYERS];

static float    *g_hidden = NULL;
static float    *g_logits = NULL;
static uint16_t *g_final_norm_w = NULL;

static int g_K = 4;
static int g_initialized = 0;

// ---- flashmoe_cache_new ----

FlashMoE_Cache *flashmoe_cache_new(void) {
    FlashMoE_Cache *c = calloc(1, sizeof(FlashMoE_Cache));
    if (!c) return NULL;
    c->kv_caches    = calloc(NUM_LAYERS, sizeof(KVCache *));
    c->layer_states = calloc(NUM_LAYERS, sizeof(void *));
    c->pos = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        int is_full = ((i + 1) % FULL_ATTN_INTERVAL == 0);
        if (is_full) {
            c->kv_caches[i] = kv_cache_new();
        } else {
            c->layer_states[i] = linear_attn_state_new();
        }
    }
    return c;
}

FlashMoE_Cache *flashmoe_cache_clone(FlashMoE_Cache *src) {
    // Only needed if the caller wants snapshot semantics.
    // For now we don't copy — just return src. Python ensures single path.
    (void)src;
    return NULL;  // not implemented yet
}

void flashmoe_cache_free(FlashMoE_Cache *c) {
    if (!c) return;
    for (int i = 0; i < NUM_LAYERS; i++) {
        if (c->kv_caches[i])    kv_cache_free(c->kv_caches[i]);
        if (c->layer_states[i]) linear_attn_state_free(c->layer_states[i]);
    }
    free(c->kv_caches);
    free(c->layer_states);
    free(c);
}

void flashmoe_cache_reset(FlashMoE_Cache *c) {
    if (!c) return;
    for (int i = 0; i < NUM_LAYERS; i++) {
        if (c->kv_caches[i]) {
            c->kv_caches[i]->len = 0;
        }
        if (c->layer_states[i]) {
            LinearAttnState *s = (LinearAttnState *)c->layer_states[i];
            memset(s->conv_state, 0,
                   (CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM * sizeof(float));
            memset(s->ssm_state, 0,
                   LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float));
        }
    }
    c->pos = 0;
    reset_delta_net_state();
}

// ---- flashmoe_init ----

int flashmoe_init(const char *model_path) {
    if (g_initialized) return -1;

    snprintf(g_model_path, sizeof(g_model_path), "%s", model_path);

    char weights_path[1024], manifest_path[1024];
    snprintf(weights_path,  sizeof(weights_path),  "%s/model_weights.bin",   model_path);
    snprintf(manifest_path, sizeof(manifest_path), "%s/model_weights.json",  model_path);

    print_model_config();

    // Metal
    g_metal = metal_setup();
    if (!g_metal) {
        fprintf(stderr, "WARNING: Metal init failed, falling back to CPU\n");
    }

    // I/O thread pool
    io_pool_init();

    // Detect 2-bit experts
    int use_2bit = g_use_2bit;
    if (!use_2bit) {
        char probe[1024];
        snprintf(probe, sizeof(probe), "%s/packed_experts_2bit/layer_00.bin", model_path);
        int pfd = open(probe, O_RDONLY);
        if (pfd >= 0) {
            close(pfd);
            snprintf(probe, sizeof(probe), "%s/packed_experts/layer_00.bin", model_path);
            int pfd4 = open(probe, O_RDONLY);
            if (pfd4 < 0) {
                g_use_2bit = 1;
                printf("[auto] Using 2-bit experts (4-bit not found)\n");
            } else {
                close(pfd4);
            }
        }
    }

    // Load weights
    g_wf = open_weights(weights_path, manifest_path);
    if (!g_wf) {
        fprintf(stderr, "ERROR: Failed to load weights\n");
        return -1;
    }
    if (g_metal) {
        metal_set_weights(g_metal, g_wf->data, g_wf->size);
    }

    // Open expert files
    memset(g_layer_fds, 0, sizeof(g_layer_fds));
    memset(g_layer_mmaps, 0, sizeof(g_layer_mmaps));
    memset(g_layer_mmap_sizes, 0, sizeof(g_layer_mmap_sizes));
    memset(g_expert_seen, 0, sizeof(g_expert_seen));

    for (int i = 0; i < NUM_LAYERS; i++) {
        g_layer_fds_cold[i] = -1;
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s/layer_%02d.bin", model_path,
                 g_use_2bit ? "packed_experts_2bit" : "packed_experts", i);
        g_layer_fds[i] = open(path, O_RDONLY);
        g_layer_mmaps[i] = MAP_FAILED;
        g_layer_mmap_sizes[i] = 0;
        if (g_layer_fds[i] >= 0) {
            fcntl(g_layer_fds[i], F_RDAHEAD, 0);
            struct stat st;
            if (fstat(g_layer_fds[i], &st) == 0 && st.st_size > 0) {
                g_layer_mmaps[i] = mmap(NULL, st.st_size, PROT_READ,
                                        MAP_PRIVATE, g_layer_fds[i], 0);
                if (g_layer_mmaps[i] != MAP_FAILED) {
                    g_layer_mmap_sizes[i] = st.st_size;
                }
            }
        }
    }

    // LZ4 detection
    {
        char lz4_probe[1024];
        snprintf(lz4_probe, sizeof(lz4_probe), "%s/packed_experts_lz4/layer_00.bin", model_path);
        if (!g_use_2bit && access(lz4_probe, R_OK) == 0) {
            for (int i = 0; i < NUM_LAYERS; i++) {
                char lz4_path[1024];
                snprintf(lz4_path, sizeof(lz4_path),
                         "%s/packed_experts_lz4/layer_%02d.bin", model_path, i);
                int lz4_fd = open(lz4_path, O_RDONLY);
                if (lz4_fd >= 0) {
                    g_lz4_index[i] = malloc(NUM_EXPERTS * sizeof(LZ4IndexEntry));
                    ssize_t nr = pread(lz4_fd, g_lz4_index[i],
                                       NUM_EXPERTS * sizeof(LZ4IndexEntry), 0);
                    if (nr == NUM_EXPERTS * (ssize_t)sizeof(LZ4IndexEntry)) {
                        close(g_layer_fds[i]);
                        g_layer_fds[i] = lz4_fd;
                        fcntl(lz4_fd, F_RDAHEAD, 1);
                    } else {
                        free(g_lz4_index[i]);
                        g_lz4_index[i] = NULL;
                        close(lz4_fd);
                    }
                }
            }
            if (g_lz4_index[0] != NULL || g_lz4_index[1] != NULL) {  // at least some layers
                g_use_lz4 = 1;
                for (int k = 0; k < MAX_K; k++) {
                    g_lz4_comp_bufs[k] = malloc(EXPERT_SIZE + 4096);
                }
            }
        }
    }

    // Warm page cache
    for (int i = 0; i < NUM_LAYERS; i++) {
        if (g_layer_fds[i] >= 0) {
            char dummy[4096];
            pread(g_layer_fds[i], dummy, sizeof(dummy), 0);
        }
    }

    // Working buffers
    g_hidden = calloc(HIDDEN_DIM, sizeof(float));
    g_logits = calloc(VOCAB_SIZE, sizeof(float));
    g_final_norm_w = get_tensor_ptr(g_wf, "model.norm.weight");
    g_K = NUM_ACTIVE_EXPERTS;

    // Build layer weight cache
    if (!layer_cache_built) build_layer_cache(g_wf);

    g_initialized = 1;
    return 0;
}

// ---- flashmoe_forward ----

int flashmoe_forward(const int *input_ids, int n_tokens,
                     float *logits_out, FlashMoE_Cache *cache) {
    if (!g_initialized || !cache || !input_ids || n_tokens < 1 || !logits_out)
        return -1;

    if (g_cache_telemetry_enabled) cache_telemetry_note_token();

    int pos = cache->pos;

    for (int tok = 0; tok < n_tokens; tok++) {
        embed_lookup(g_wf, input_ids[tok], g_hidden);

        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
            fused_layer_forward(g_wf, layer, g_hidden,
                                is_full ? cache->kv_caches[layer] : NULL,
                                is_full ? NULL : cache->layer_states[layer],
                                pos,
                                g_layer_mmaps[layer] != MAP_FAILED
                                    ? g_layer_mmaps[layer] : NULL,
                                g_K, g_layer_fds[layer]);
        }
        complete_deferred_experts();
        pos++;

        if (g_final_norm_w) {
            float *normed = malloc(HIDDEN_DIM * sizeof(float));
            cpu_rms_norm(g_hidden, g_final_norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);
            memcpy(g_hidden, normed, HIDDEN_DIM * sizeof(float));
            free(normed);
        }

        lm_head_forward(g_wf, g_hidden, logits_out + (size_t)tok * VOCAB_SIZE);
    }

    cache->pos = pos;
    return 0;
}

// ---- Accessors ----

int flashmoe_cache_position(FlashMoE_Cache *c) {
    return c ? c->pos : 0;
}

int flashmoe_vocab_size(void)  { return VOCAB_SIZE; }
int flashmoe_hidden_dim(void)  { return HIDDEN_DIM; }
int flashmoe_num_layers(void)  { return NUM_LAYERS; }

// ---- flashmoe_free ----

void flashmoe_free(void) {
    if (!g_initialized) return;

    io_pool_shutdown();
    if (g_malloc_cache) {
        malloc_cache_free(g_malloc_cache);
        g_malloc_cache = NULL;
    }
    if (g_expert_cache) {
        expert_cache_free(g_expert_cache);
        g_expert_cache = NULL;
    }
    for (int i = 0; i < NUM_LAYERS; i++) {
        if (g_layer_mmaps[i] != MAP_FAILED) {
            munmap(g_layer_mmaps[i], g_layer_mmap_sizes[i]);
        }
        if (g_layer_fds[i] >= 0) close(g_layer_fds[i]);
        if (g_layer_fds_cold[i] >= 0) close(g_layer_fds_cold[i]);
        if (g_lz4_index[i]) {
            free(g_lz4_index[i]);
            g_lz4_index[i] = NULL;
        }
    }
    for (int k = 0; k < MAX_K; k++) {
        free(g_lz4_comp_bufs[k]);
        g_lz4_comp_bufs[k] = NULL;
    }
    free(g_hidden);
    free(g_logits);
    // WeightFile is not freed (it's mmap'd)
    g_hidden = NULL;
    g_logits = NULL;
    g_wf = NULL;
    g_metal = NULL;
    g_initialized = 0;
}
