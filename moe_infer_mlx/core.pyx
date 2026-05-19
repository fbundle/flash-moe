# cython: language_level=3
"""
Thin Cython mirror of moe_infer_c.h — one function per C function.
"""

import numpy as np
cimport numpy as cnp

cdef extern from "stdlib.h":
    void *malloc(size_t size)
    void free(void *ptr)

cdef extern from "moe_infer_c.h":
    ctypedef struct FlashMoE_Cache:
        pass
    ctypedef struct FlashMoE_Model:
        pass

    FlashMoE_Model *flashmoe_init(const char *model_path)
    void            flashmoe_free(FlashMoE_Model *model)

    FlashMoE_Cache *flashmoe_cache_new(FlashMoE_Model *model)
    void            flashmoe_cache_free(FlashMoE_Cache *c)
    void            flashmoe_cache_reset(FlashMoE_Cache *c, FlashMoE_Model *model)

    int flashmoe_forward(FlashMoE_Model *model,
                         const int *input_ids, int n_tokens,
                         float *logits_out, FlashMoE_Cache *cache)

    int flashmoe_cache_position(FlashMoE_Cache *c)
    int flashmoe_vocab_size(FlashMoE_Model *model)
    int flashmoe_hidden_dim(FlashMoE_Model *model)
    int flashmoe_num_layers(FlashMoE_Model *model)


# ---- Model lifecycle ----

def init(str model_path):
    """Initialize inference engine. Returns opaque model pointer (as int)."""
    cdef bytes path_bytes = model_path.encode('utf-8')
    cdef const char *path = path_bytes
    cdef FlashMoE_Model *m = flashmoe_init(path)
    if m == NULL:
        raise RuntimeError(f"Failed to initialize model from {model_path}")
    return <unsigned long long>m


def free_all(unsigned long long model_ptr):
    """Free all resources."""
    flashmoe_free(<FlashMoE_Model *>model_ptr)


# ---- Cache lifecycle ----

def cache_new(unsigned long long model_ptr):
    """Return opaque cache pointer (as integer)."""
    return <unsigned long long>flashmoe_cache_new(<FlashMoE_Model *>model_ptr)


def cache_free(unsigned long long ptr):
    """Free a cache."""
    flashmoe_cache_free(<FlashMoE_Cache *>ptr)


def cache_reset(unsigned long long cache_ptr, unsigned long long model_ptr):
    """Reset cache for fresh session."""
    flashmoe_cache_reset(<FlashMoE_Cache *>cache_ptr, <FlashMoE_Model *>model_ptr)


# ---- Inference ----

def forward(list input_ids, unsigned long long model_ptr,
            unsigned long long cache_ptr):
    """Run forward pass. Returns (logits: np.ndarray[dtype=float32], cache_ptr)."""
    cdef int n = len(input_ids)
    cdef int vocab_size = flashmoe_vocab_size(<FlashMoE_Model *>model_ptr)
    cdef int *ids_ptr
    cdef float *logits_ptr
    cdef int ret, i
    cdef FlashMoE_Model *mp = <FlashMoE_Model *>model_ptr
    cdef FlashMoE_Cache *cp = <FlashMoE_Cache *>cache_ptr

    cdef cnp.ndarray[float, ndim=2] logits

    ids_ptr = <int*>malloc(n * sizeof(int))
    if ids_ptr == NULL:
        raise MemoryError()

    for i in range(n):
        ids_ptr[i] = input_ids[i]

    logits = np.empty((n, vocab_size), dtype=np.float32)
    logits_ptr = <float*>logits.data

    ret = flashmoe_forward(mp, ids_ptr, n, logits_ptr, cp)
    free(ids_ptr)
    if ret != 0:
        raise RuntimeError("Forward pass failed")

    return (logits, cache_ptr)


# ---- Accessors ----

def vocab_size(unsigned long long model_ptr):
    return flashmoe_vocab_size(<FlashMoE_Model *>model_ptr)


def hidden_dim(unsigned long long model_ptr):
    return flashmoe_hidden_dim(<FlashMoE_Model *>model_ptr)


def num_layers(unsigned long long model_ptr):
    return flashmoe_num_layers(<FlashMoE_Model *>model_ptr)


def cache_position(unsigned long long ptr):
    """Return number of tokens already in cache."""
    return flashmoe_cache_position(<FlashMoE_Cache *>ptr)
