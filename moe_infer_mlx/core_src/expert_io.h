#ifndef EXPERT_IO_H
#define EXPERT_IO_H

// ============================================================================
// Parallel I/O infrastructure for expert pread (from proven main.m pattern).
// Types (IOThreadPool, AsyncPreadState, ExpertLRUCache, MallocExpertCache,
// InferPrefetchCtx, etc.) are in model_types.h.
// All state accessed via FlashMoE_Model *m.
// ============================================================================

#include "model_internal.h"

void *infer_pread_thread_fn(void *arg) {
    InferPreadThreadArg *ta = (InferPreadThreadArg *)arg;
    for (int i = ta->thread_id; i < ta->num_tasks; i += NUM_IO_THREADS) {
        InferPreadTask *t = &ta->tasks[i];
        t->result = pread(t->fd, t->dst, t->size, t->offset);
    }
    return NULL;
}

// ============================================================================
// Persistent I/O Thread Pool
// ============================================================================

typedef struct {
    IOThreadPool *pool;
    int tid;
} IOWorkerArg;

static void *io_pool_worker(void *arg) {
    IOWorkerArg *wa = (IOWorkerArg *)arg;
    IOThreadPool *pool = wa->pool;
    int tid = wa->tid;
    free(wa);

    int my_gen = 0;
    pthread_mutex_lock(&pool->mutex);
    while (1) {
        while (pool->generation == my_gen && !pool->shutdown)
            pthread_cond_wait(&pool->work_ready, &pool->mutex);
        if (pool->shutdown) break;
        my_gen = pool->generation;

        int num_tasks = pool->num_tasks;
        InferPreadTask *tasks = pool->tasks;
        pthread_mutex_unlock(&pool->mutex);

        for (int i = tid; i < num_tasks; i += NUM_IO_THREADS) {
            InferPreadTask *t = &tasks[i];
            if (t->lz4_comp_buf && t->lz4_comp_size > 0) {
                ssize_t nr = pread(t->fd, t->lz4_comp_buf, t->lz4_comp_size, t->offset);
                if (nr == (ssize_t)t->lz4_comp_size) {
                    size_t dec = compression_decode_buffer(
                        t->dst, t->size, t->lz4_comp_buf, t->lz4_comp_size,
                        NULL, COMPRESSION_LZ4);
                    t->result = (ssize_t)dec;
                } else {
                    t->result = -1;
                }
            } else {
                t->result = pread(t->fd, t->dst, t->size, t->offset);
            }
        }

        pthread_mutex_lock(&pool->mutex);
        pool->tasks_completed++;
        if (pool->tasks_completed == NUM_IO_THREADS)
            pthread_cond_signal(&pool->work_done);
    }
    pthread_mutex_unlock(&pool->mutex);
    return NULL;
}

static void io_pool_init(FlashMoE_Model *m) {
    if (m->io_pool_initialized) return;
    pthread_mutex_init(&m->io_pool.mutex, NULL);
    pthread_cond_init(&m->io_pool.work_ready, NULL);
    pthread_cond_init(&m->io_pool.work_done, NULL);
    m->io_pool.shutdown = 0;
    m->io_pool.generation = 0;
    m->io_pool.tasks = NULL;
    for (int i = 0; i < NUM_IO_THREADS; i++) {
        IOWorkerArg *wa = malloc(sizeof(IOWorkerArg));
        wa->pool = &m->io_pool;
        wa->tid = i;
        pthread_create(&m->io_pool.threads[i], NULL, io_pool_worker, wa);
    }
    m->io_pool_initialized = 1;
}

// Low-level dispatch — takes IOThreadPool* directly (usable from background threads).
static void io_pool_dispatch_raw(IOThreadPool *pool, InferPreadTask *tasks, int num_tasks) {
    if (num_tasks == 0) return;
    pthread_mutex_lock(&pool->mutex);
    pool->tasks = tasks;
    pool->num_tasks = num_tasks;
    pool->tasks_completed = 0;
    pool->generation++;
    pthread_cond_broadcast(&pool->work_ready);
    while (pool->tasks_completed < NUM_IO_THREADS) {
        pthread_cond_wait(&pool->work_done, &pool->mutex);
    }
    pthread_mutex_unlock(&pool->mutex);
}

static void io_pool_dispatch(FlashMoE_Model *m, InferPreadTask *tasks, int num_tasks) {
    io_pool_dispatch_raw(&m->io_pool, tasks, num_tasks);
}

// ---- Async expert pread pipeline ----

static void async_pread_start(FlashMoE_Model *m, int packed_fd, int *expert_indices, int K,
                               id<MTLBuffer> __strong *dst_bufs, const void *mmap_base) {
    size_t esz = active_expert_size(m);
    m->async_pread.num_tasks = K;
    m->async_pread.active = 1;
    if (!m->async_pread.group) m->async_pread.group = dispatch_group_create();

    for (int k = 0; k < K; k++) {
        m->async_pread.tasks[k].fd = packed_fd;
        m->async_pread.tasks[k].dst = [dst_bufs[k] contents];
        m->async_pread.tasks[k].offset = (off_t)expert_indices[k] * esz;
        m->async_pread.tasks[k].size = esz;
        m->async_pread.tasks[k].result = 0;
    }

    if (!m->io_gcd_queue) m->io_gcd_queue = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    for (int k = 0; k < K; k++) {
        InferPreadTask *t = &m->async_pread.tasks[k];
        dispatch_group_async(m->async_pread.group, m->io_gcd_queue, ^{
            t->result = pread(t->fd, t->dst, t->size, t->offset);
        });
    }
}

static void async_pread_wait(FlashMoE_Model *m) {
    if (!m->async_pread.active) return;
    dispatch_group_wait(m->async_pread.group, DISPATCH_TIME_FOREVER);
    for (int k = 0; k < m->async_pread.num_tasks; k++) {
        m->async_pread.valid[k] = (m->async_pread.tasks[k].result == (ssize_t)active_expert_size(m));
    }
    m->async_pread.active = 0;
}

static void io_pool_shutdown(FlashMoE_Model *m) {
    if (!m->io_pool_initialized) return;
    pthread_mutex_lock(&m->io_pool.mutex);
    m->io_pool.shutdown = 1;
    pthread_cond_broadcast(&m->io_pool.work_ready);
    pthread_mutex_unlock(&m->io_pool.mutex);
    for (int i = 0; i < NUM_IO_THREADS; i++)
        pthread_join(m->io_pool.threads[i], NULL);
    pthread_mutex_destroy(&m->io_pool.mutex);
    pthread_cond_destroy(&m->io_pool.work_ready);
    pthread_cond_destroy(&m->io_pool.work_done);
    m->io_pool_initialized = 0;
}

// Parallel pread of K experts into Metal buffers using pthreads.
int parallel_pread_experts(
    FlashMoE_Model *m,
    int packed_fd,
    int *expert_indices,
    int K,
    int *valid,
    const void *mmap_base
) {
    size_t esz = active_expert_size(m);
    InferPreadTask tasks[MAX_K];
    for (int k = 0; k < K; k++) {
        tasks[k].fd = packed_fd;
        tasks[k].dst = [m->metal->buf_multi_expert_data[k] contents];
        tasks[k].offset = (off_t)expert_indices[k] * esz;
        tasks[k].size = esz;
        tasks[k].result = 0;
        tasks[k].mmap_base = mmap_base;
    }

    io_pool_dispatch(m, tasks, K);

    int loaded = 0;
    for (int k = 0; k < K; k++) {
        valid[k] = (tasks[k].result == (ssize_t)esz);
        if (valid[k]) loaded++;
        else {
            fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                    expert_indices[k], tasks[k].result, esz);
        }
    }
    return loaded;
}

// Parallel pread into explicit buffer set (for double buffering).
int parallel_pread_experts_into(
    FlashMoE_Model *m,
    int packed_fd,
    int *expert_indices,
    int K,
    id<MTLBuffer> __strong *dst_bufs,
    int *valid
) {
    size_t esz = active_expert_size(m);
    InferPreadTask tasks[MAX_K];
    for (int k = 0; k < K; k++) {
        tasks[k].fd = packed_fd;
        tasks[k].dst = [dst_bufs[k] contents];
        tasks[k].offset = (off_t)expert_indices[k] * esz;
        tasks[k].size = esz;
        tasks[k].result = 0;
    }

    io_pool_dispatch(m, tasks, K);

    int loaded = 0;
    for (int k = 0; k < K; k++) {
        valid[k] = (tasks[k].result == (ssize_t)esz);
        if (valid[k]) loaded++;
        else {
            fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                    expert_indices[k], tasks[k].result, esz);
        }
    }
    return loaded;
}

// ============================================================================
// Expert LRU Cache
// ============================================================================

ExpertLRUCache *expert_cache_new(FlashMoE_Model *m, id<MTLDevice> device, int max_entries) {
    ExpertLRUCache *cache = calloc(1, sizeof(ExpertLRUCache));
    cache->entry_idx = calloc((size_t)m->cfg.num_layers * m->cfg.num_experts, sizeof(int));
    cache->entries = calloc(max_entries, sizeof(ExpertCacheEntry));
    cache->max_entries = max_entries;
    cache->num_entries = 0;
    cache->used_entries = 0;
    cache->access_counter = 0;
    cache->device = device;
    cache->hits = 0;
    cache->misses = 0;
    for (int l = 0; l < m->cfg.num_layers; l++) {
        for (int e = 0; e < m->cfg.num_experts; e++) {
            cache->entry_idx[(l) * m->cfg.num_experts + (e)] = -1;
        }
    }
    size_t esz = active_expert_size(m);
    double t_prealloc = now_ms();
    for (int i = 0; i < max_entries; i++) {
        cache->entries[i].buffer = [device newBufferWithLength:esz
                                                      options:MTLResourceStorageModeShared];
        cache->entries[i].layer_idx = -1;
        cache->entries[i].expert_idx = -1;
        cache->entries[i].last_used = 0;
        if (!cache->entries[i].buffer) {
            fprintf(stderr, "WARNING: expert_cache: pre-alloc failed at entry %d\n", i);
            max_entries = i;
            cache->max_entries = i;
            break;
        }
    }
    cache->num_entries = max_entries;
    printf("[expert_cache] Initialized: max_entries=%d (%.1f GB budget), pre-alloc %.0f ms\n",
           max_entries, (double)max_entries * esz / 1e9, now_ms() - t_prealloc);
    return cache;
}

static void expert_cache_free(ExpertLRUCache *cache) {
    if (!cache) return;
    printf("[expert_cache] Final stats: %llu hits, %llu misses (%.1f%% hit rate)\n",
           cache->hits, cache->misses,
           (cache->hits + cache->misses) > 0
               ? 100.0 * cache->hits / (cache->hits + cache->misses) : 0.0);
    free(cache->entries);
    free(cache);
}

static id<MTLBuffer> expert_cache_lookup(FlashMoE_Model *m, ExpertLRUCache *cache, int layer_idx, int expert_idx) {
    int idx = cache->entry_idx[(layer_idx) * m->cfg.num_experts + (expert_idx)];
    if (idx >= 0) {
        cache->entries[idx].last_used = ++cache->access_counter;
        cache->hits++;
        cache_telemetry_touch(m, layer_idx, expert_idx);
        return cache->entries[idx].buffer;
    }
    cache->misses++;
    cache_telemetry_miss(m, layer_idx, expert_idx);
    return nil;
}

static id<MTLBuffer> expert_cache_insert(FlashMoE_Model *m, ExpertLRUCache *cache, int layer_idx, int expert_idx) {
    id<MTLBuffer> buf = nil;

    int existing = cache->entry_idx[(layer_idx) * m->cfg.num_experts + (expert_idx)];
    if (existing >= 0) {
        cache->entries[existing].last_used = ++cache->access_counter;
        return cache->entries[existing].buffer;
    }

    int target = -1;
    if (cache->used_entries < cache->num_entries) {
        target = cache->used_entries++;
    }
    if (target >= 0) {
        buf = cache->entries[target].buffer;
        cache->entries[target].layer_idx = layer_idx;
        cache->entries[target].expert_idx = expert_idx;
        cache->entries[target].last_used = ++cache->access_counter;
        cache->entry_idx[(layer_idx) * m->cfg.num_experts + (expert_idx)] = target;
        return buf;
    }

    int lru_idx = 0;
    uint64_t min_used = cache->entries[0].last_used;
    for (int i = 1; i < cache->num_entries; i++) {
        if (cache->entries[i].last_used < min_used) {
            min_used = cache->entries[i].last_used;
            lru_idx = i;
        }
    }

    int old_layer = cache->entries[lru_idx].layer_idx;
    int old_expert = cache->entries[lru_idx].expert_idx;
    cache_telemetry_evict(m, old_layer, old_expert);
    if (old_layer >= 0 && old_expert >= 0) {
        cache->entry_idx[(old_layer) * m->cfg.num_experts + (old_expert)] = -1;
    }
    buf = cache->entries[lru_idx].buffer;
    cache->entries[lru_idx].layer_idx = layer_idx;
    cache->entries[lru_idx].expert_idx = expert_idx;
    cache->entries[lru_idx].last_used = ++cache->access_counter;
    cache->entry_idx[(layer_idx) * m->cfg.num_experts + (expert_idx)] = lru_idx;
    return buf;
}

// ============================================================================
// Malloc-based expert frequency cache
// ============================================================================

MallocExpertCache *malloc_cache_init(FlashMoE_Model *m, int max_entries, id<MTLDevice> device) {
    MallocExpertCache *cache = calloc(1, sizeof(MallocExpertCache));
    cache->entry_idx = calloc((size_t)m->cfg.num_layers * m->cfg.num_experts, sizeof(int));
    cache->data = calloc(max_entries, sizeof(void *));
    cache->metal_bufs = (__strong id<MTLBuffer> *)calloc(max_entries, sizeof(id<MTLBuffer>));
    cache->layer_idx = calloc(max_entries, sizeof(int));
    cache->expert_idx = calloc(max_entries, sizeof(int));
    cache->last_used = calloc(max_entries, sizeof(uint64_t));
    cache->max_entries = max_entries;
    cache->num_entries = 0;
    cache->used_entries = 0;
    cache->access_counter = 0;
    cache->hits = 0;
    cache->misses = 0;
    for (int l = 0; l < m->cfg.num_layers; l++) {
        for (int e = 0; e < m->cfg.num_experts; e++) {
            cache->entry_idx[(l) * m->cfg.num_experts + (e)] = -1;
        }
    }

    size_t esz = active_expert_size(m);
    printf("[malloc_cache] Initializing: %d entries (%.1f GB) with zero-copy Metal wrappers\n",
           max_entries, (double)max_entries * esz / 1e9);
    double t_start = now_ms();

    size_t page_size = (size_t)getpagesize();
    size_t aligned_size = (esz + page_size - 1) & ~(page_size - 1);

    for (int i = 0; i < max_entries; i++) {
        void *buf = NULL;
        if (posix_memalign(&buf, page_size, aligned_size) != 0 || !buf) {
            fprintf(stderr, "WARNING: malloc_cache: alloc failed at entry %d\n", i);
            max_entries = i;
            cache->max_entries = i;
            break;
        }
        memset(buf, 0, aligned_size);
        cache->data[i] = buf;

        cache->metal_bufs[i] = [device newBufferWithBytesNoCopy:buf
                                                         length:aligned_size
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        cache->layer_idx[i] = -1;
        cache->expert_idx[i] = -1;
        cache->last_used[i] = 0;
    }
    cache->num_entries = max_entries;

    printf("[malloc_cache] Pre-allocated %d entries in %.0f ms\n",
           max_entries, now_ms() - t_start);
    return cache;
}

static id<MTLBuffer> malloc_cache_lookup(FlashMoE_Model *m, MallocExpertCache *cache, int layer, int expert) {
    int idx = cache->entry_idx[(layer) * m->cfg.num_experts + (expert)];
    if (idx >= 0) {
        cache->last_used[idx] = ++cache->access_counter;
        cache->hits++;
        cache_telemetry_touch(m, layer, expert);
        return cache->metal_bufs[idx];
    }
    cache->misses++;
    cache_telemetry_miss(m, layer, expert);
    return nil;
}

static id<MTLBuffer> malloc_cache_insert(FlashMoE_Model *m, MallocExpertCache *cache, int layer, int expert, int *out_idx) {
    int existing = cache->entry_idx[(layer) * m->cfg.num_experts + (expert)];
    if (existing >= 0) {
        cache->last_used[existing] = ++cache->access_counter;
        if (out_idx) *out_idx = existing;
        return cache->metal_bufs[existing];
    }

    int target = -1;
    if (cache->used_entries < cache->num_entries) {
        target = cache->used_entries++;
    }

    if (target < 0) {
        target = 0;
        uint64_t min_used = cache->last_used[0];
        for (int i = 1; i < cache->num_entries; i++) {
            if (cache->last_used[i] < min_used) {
                min_used = cache->last_used[i];
                target = i;
            }
        }
        cache_telemetry_evict(m, cache->layer_idx[target], cache->expert_idx[target]);
        if (cache->layer_idx[target] >= 0 && cache->expert_idx[target] >= 0) {
            cache->entry_idx[(cache->layer_idx[target]) * m->cfg.num_experts + (cache->expert_idx[target])] = -1;
        }
    }

    cache->layer_idx[target] = layer;
    cache->expert_idx[target] = expert;
    cache->last_used[target] = ++cache->access_counter;
    cache->entry_idx[(layer) * m->cfg.num_experts + (expert)] = target;
    if (out_idx) *out_idx = target;
    return cache->metal_bufs[target];
}

static void malloc_cache_free(MallocExpertCache *cache) {
    if (!cache) return;
    printf("[malloc_cache] Final stats: %llu hits, %llu misses (%.1f%% hit rate)\n",
           cache->hits, cache->misses,
           (cache->hits + cache->misses) > 0
               ? 100.0 * cache->hits / (cache->hits + cache->misses) : 0.0);
    for (int i = 0; i < cache->num_entries; i++) {
        cache->metal_bufs[i] = nil;
        free(cache->data[i]);
    }
    free(cache->data);
    free(cache->metal_bufs);
    free(cache->layer_idx);
    free(cache->expert_idx);
    free(cache->last_used);
    free(cache);
}

// ============================================================================
// Background prefetch thread for double-buffered expert I/O
// ============================================================================

static void *infer_prefetch_thread_fn(void *arg) {
    InferPrefetchCtx *pf = (InferPrefetchCtx *)arg;

    while (1) {
        pthread_mutex_lock(&pf->mutex);
        while (!pf->start && !pf->shutdown) {
            pthread_cond_wait(&pf->cond, &pf->mutex);
        }
        if (pf->shutdown) {
            pthread_mutex_unlock(&pf->mutex);
            break;
        }
        pf->start = 0;
        pthread_mutex_unlock(&pf->mutex);

        // Execute parallel pread using the stored IO pool
        InferIOPlan *plan = &pf->plan;
        size_t esz = pf->expert_size;
        InferPreadTask tasks[MAX_K];
        for (int k = 0; k < plan->K; k++) {
            tasks[k].fd = plan->fd;
            tasks[k].dst = plan->dst[k];
            tasks[k].offset = plan->offset[k];
            tasks[k].size = esz;
            tasks[k].result = 0;
        }

        io_pool_dispatch_raw(pf->io_pool, tasks, plan->K);

        plan->loaded = 0;
        for (int k = 0; k < plan->K; k++) {
            plan->valid[k] = (tasks[k].result == (ssize_t)esz);
            if (plan->valid[k]) plan->loaded++;
        }

        // Signal completion
        pthread_mutex_lock(&pf->mutex);
        pf->done = 1;
        pthread_cond_signal(&pf->cond);
        pthread_mutex_unlock(&pf->mutex);
    }

    return NULL;
}

// Build I/O plan on main thread (ARC-safe: extracts void* from id<MTLBuffer>),
// then signal background prefetch thread.
void infer_prefetch_start(InferPrefetchCtx *pf, int packed_fd,
                                  int *expert_indices, int K,
                                  id<MTLBuffer> __strong *dst_bufs,
                                  size_t expert_size) {
    pthread_mutex_lock(&pf->mutex);
    InferIOPlan *plan = &pf->plan;
    plan->fd = packed_fd;
    plan->K = K;
    for (int k = 0; k < K; k++) {
        plan->dst[k] = [dst_bufs[k] contents];
        plan->offset[k] = (off_t)expert_indices[k] * expert_size;
        plan->valid[k] = 0;
    }
    plan->loaded = 0;
    pf->expert_size = expert_size;
    pf->done = 0;
    pf->start = 1;
    pthread_cond_signal(&pf->cond);
    pthread_mutex_unlock(&pf->mutex);
}

// Wait for background prefetch to complete.
int infer_prefetch_wait(InferPrefetchCtx *pf, int *valid_out, int K) {
    pthread_mutex_lock(&pf->mutex);
    while (!pf->done) {
        pthread_cond_wait(&pf->cond, &pf->mutex);
    }
    int loaded = pf->plan.loaded;
    for (int k = 0; k < K; k++) {
        valid_out[k] = pf->plan.valid[k];
    }
    pthread_mutex_unlock(&pf->mutex);
    return loaded;
}

void infer_prefetch_init(FlashMoE_Model *m) {
    if (m->prefetch) return;
    m->prefetch = calloc(1, sizeof(InferPrefetchCtx));
    pthread_mutex_init(&m->prefetch->mutex, NULL);
    pthread_cond_init(&m->prefetch->cond, NULL);
    m->prefetch->shutdown = 0;
    m->prefetch->io_pool = &m->io_pool;
    m->prefetch->expert_size = 0;
    pthread_create(&m->prefetch_tid, NULL, infer_prefetch_thread_fn, m->prefetch);
}

void infer_prefetch_shutdown(FlashMoE_Model *m) {
    if (!m->prefetch) return;
    pthread_mutex_lock(&m->prefetch->mutex);
    m->prefetch->shutdown = 1;
    pthread_cond_signal(&m->prefetch->cond);
    pthread_mutex_unlock(&m->prefetch->mutex);
    pthread_join(m->prefetch_tid, NULL);
    pthread_mutex_destroy(&m->prefetch->mutex);
    pthread_cond_destroy(&m->prefetch->cond);
    free(m->prefetch);
    m->prefetch = NULL;
}


#endif // EXPERT_IO_H
