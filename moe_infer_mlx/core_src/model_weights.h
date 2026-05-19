#ifndef TENSORS_H
#define TENSORS_H

// ============================================================================
// JSON parser (minimal, for model_weights.json)
// Types (TensorInfo, TensorManifest, TensorHTEntry, WeightFile) are in model_types.h.
// Hash table state (tensor_ht, tensor_ht_built) lives in FlashMoE_Model.
// ============================================================================

#include "model_internal.h"

static TensorManifest *load_manifest(const char *json_path) {
    @autoreleasepool {
        NSData *data = [NSData dataWithContentsOfFile:
            [NSString stringWithUTF8String:json_path]];
        if (!data) {
            fprintf(stderr, "ERROR: Cannot read %s\n", json_path);
            return NULL;
        }

        NSError *error = nil;
        NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data
                                                             options:0
                                                               error:&error];
        if (!root) {
            fprintf(stderr, "ERROR: JSON parse failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return NULL;
        }

        NSDictionary *tensors = root[@"tensors"];
        if (!tensors) {
            fprintf(stderr, "ERROR: No 'tensors' key in manifest\n");
            return NULL;
        }

        TensorManifest *manifest = calloc(1, sizeof(TensorManifest));
        manifest->capacity = (int)[tensors count] + 16;
        manifest->tensors = calloc(manifest->capacity, sizeof(TensorInfo));
        manifest->num_tensors = 0;

        for (NSString *key in tensors) {
            NSDictionary *info = tensors[key];
            TensorInfo *t = &manifest->tensors[manifest->num_tensors];

            const char *name = [key UTF8String];
            t->name = strdup(name);
            t->offset = [info[@"offset"] unsignedLongLongValue];
            t->size = [info[@"size"] unsignedLongLongValue];

            NSArray *shape = info[@"shape"];
            t->ndim = (int)[shape count];
            for (int i = 0; i < t->ndim && i < 4; i++) {
                t->shape[i] = [shape[i] intValue];
            }

            const char *dtype = [info[@"dtype"] UTF8String];
            strncpy(t->dtype, dtype, 7);

            manifest->num_tensors++;
        }

        printf("[manifest] Loaded %d tensors from %s\n", manifest->num_tensors, json_path);
        return manifest;
    }
}

// Hash table for O(1) tensor lookup (replaces O(N) linear scan).
// FNV-1a hash, open addressing with linear probing.
#define TENSOR_HT_SIZE 8192  // power of 2, > 4x num_tensors (2092)

static uint32_t fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) {
        h ^= (uint8_t)*s;
        h *= 16777619u;
    }
    return h;
}

static void build_tensor_ht(FlashMoE_Model *m, TensorManifest *manifest) {
    if (m->tensor_ht_built) return;
    memset(m->tensor_ht, 0, sizeof(m->tensor_ht));
    for (int i = 0; i < manifest->num_tensors; i++) {
        uint32_t idx = fnv1a(manifest->tensors[i].name) & (TENSOR_HT_SIZE - 1);
        while (m->tensor_ht[idx].key) {
            idx = (idx + 1) & (TENSOR_HT_SIZE - 1);
        }
        m->tensor_ht[idx].key = manifest->tensors[i].name;
        m->tensor_ht[idx].value = &manifest->tensors[i];
    }
    m->tensor_ht_built = 1;
}

static TensorInfo *find_tensor(FlashMoE_Model *m, TensorManifest *manifest, const char *name) {
    if (!m->tensor_ht_built) build_tensor_ht(m, manifest);
    uint32_t idx = fnv1a(name) & (TENSOR_HT_SIZE - 1);
    while (m->tensor_ht[idx].key) {
        if (strcmp(m->tensor_ht[idx].key, name) == 0) {
            return m->tensor_ht[idx].value;
        }
        idx = (idx + 1) & (TENSOR_HT_SIZE - 1);
    }
    return NULL;
}

// ============================================================================
// Weight file: mmap'd binary blob
// ============================================================================

static WeightFile *open_weights(const char *bin_path, const char *json_path) {
    int fd = open(bin_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ERROR: Cannot open %s: %s\n", bin_path, strerror(errno));
        return NULL;
    }

    struct stat st;
    fstat(fd, &st);
    size_t size = st.st_size;

#if MALLOC_WEIGHTS
    size_t page_size = 16384;
    size_t aligned_size = (size + page_size - 1) & ~(page_size - 1);
    void *data = NULL;
    if (posix_memalign(&data, page_size, aligned_size) != 0) {
        fprintf(stderr, "ERROR: posix_memalign failed for %.2f GB\n", size / 1e9);
        close(fd);
        return NULL;
    }
    size_t total = 0;
    while (total < size) {
        ssize_t n = read(fd, (char *)data + total, size - total);
        if (n <= 0) {
            fprintf(stderr, "ERROR: read failed at %zu / %zu: %s\n", total, size, strerror(errno));
            free(data);
            close(fd);
            return NULL;
        }
        total += (size_t)n;
    }
    close(fd);
    printf("[weights] loaded %.2f GB from %s\n", total / 1e9, bin_path);
#else
    void *data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) {
        fprintf(stderr, "ERROR: mmap failed: %s\n", strerror(errno));
        return NULL;
    }
    madvise(data, size, MADV_SEQUENTIAL);
    printf("[weights] mmap'd %.2f GB from %s\n", size / 1e9, bin_path);
#endif

    TensorManifest *manifest = load_manifest(json_path);
    if (!manifest) {
#if MALLOC_WEIGHTS
        free(data);
#else
        munmap(data, size);
#endif
        return NULL;
    }

    WeightFile *wf = calloc(1, sizeof(WeightFile));
    wf->data = data;
    wf->size = size;
    wf->manifest = manifest;
    return wf;
}

static void *get_tensor_ptr(FlashMoE_Model *m, WeightFile *wf, const char *name) {
    TensorInfo *t = find_tensor(m, wf->manifest, name);
    if (!t) {
        fprintf(stderr, "WARNING: tensor '%s' not found\n", name);
        return NULL;
    }
    return (char *)wf->data + t->offset;
}

static TensorInfo *get_tensor_info(FlashMoE_Model *m, WeightFile *wf, const char *name) {
    return find_tensor(m, wf->manifest, name);
}


#endif // TENSORS_H
