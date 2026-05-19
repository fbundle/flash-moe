# cython: language_level=3
"""
Thin Cython mirror of src/flashmoe_c.h — one function per C function.
"""

import numpy as np
cimport numpy as cnp

cdef extern from "stdlib.h":
    void *malloc(size_t size)
    void free(void *ptr)

cdef extern from "flashmoe_c.h":
    ctypedef struct FlashMoE_Cache:
        pass

    int  flashmoe_init(const char *model_path)
    void flashmoe_free()

    FlashMoE_Cache *flashmoe_cache_new()
    void            flashmoe_cache_free(FlashMoE_Cache *c)
    void            flashmoe_cache_reset(FlashMoE_Cache *c)

    int flashmoe_forward(const int *input_ids, int n_tokens,
                         float *logits_out, FlashMoE_Cache *cache)

    int         flashmoe_cache_position(FlashMoE_Cache *c)
    int         flashmoe_vocab_size()
    int         flashmoe_hidden_dim()
    int         flashmoe_num_layers()


def init(str model_path):
    """Initialize inference engine. Raises RuntimeError on failure."""
    cdef bytes path_bytes = model_path.encode('utf-8')
    cdef const char *path = path_bytes
    if flashmoe_init(path) != 0:
        raise RuntimeError(f"Failed to initialize model from {model_path}")


def free_all():
    """Free all resources."""
    flashmoe_free()


def cache_new():
    """Return opaque cache pointer (as integer)."""
    return <unsigned long long>flashmoe_cache_new()


def cache_free(unsigned long long ptr):
    """Free a cache."""
    flashmoe_cache_free(<FlashMoE_Cache *>ptr)


def cache_reset(unsigned long long ptr):
    """Reset cache for fresh session."""
    flashmoe_cache_reset(<FlashMoE_Cache *>ptr)


def forward(list input_ids, unsigned long long cache_ptr):
    """Run forward pass. Returns (logits: np.ndarray[dtype=float32], cache_ptr)."""
    cdef int n = len(input_ids)
    cdef int vocab_size = flashmoe_vocab_size()
    cdef int *ids_ptr
    cdef float *logits_ptr
    cdef int ret, i
    cdef FlashMoE_Cache *cp = <FlashMoE_Cache *>cache_ptr

    cdef cnp.ndarray[float, ndim=2] logits

    ids_ptr = <int*>malloc(n * sizeof(int))
    if ids_ptr == NULL:
        raise MemoryError()

    for i in range(n):
        ids_ptr[i] = input_ids[i]

    logits = np.empty((n, vocab_size), dtype=np.float32)
    logits_ptr = <float*>logits.data

    ret = flashmoe_forward(ids_ptr, n, logits_ptr, cp)
    free(ids_ptr)
    if ret != 0:
        raise RuntimeError("Forward pass failed")

    return (logits, cache_ptr)


def vocab_size():
    return flashmoe_vocab_size()


def hidden_dim():
    return flashmoe_hidden_dim()


def num_layers():
    return flashmoe_num_layers()


def cache_position(unsigned long long ptr):
    """Return number of tokens already in cache."""
    return flashmoe_cache_position(<FlashMoE_Cache *>ptr)
