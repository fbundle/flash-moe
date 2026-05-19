#ifndef MOE_INFER_C_H
#define MOE_INFER_C_H

// ============================================================================
// C API for Flash-MoE inference engine.
// Instance-based: flashmoe_init() returns an opaque FlashMoE_Model pointer;
// all functions take it as their first argument.
// Called from Cython/Python.
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// ---- Opaque handles ----

typedef struct FlashMoE_Cache FlashMoE_Cache;
typedef struct FlashMoE_Model FlashMoE_Model;

// ---- Model lifecycle ----

// Initialize the inference engine from model_path.
// Returns an opaque model handle, or NULL on error.
FlashMoE_Model *flashmoe_init(const char *model_path);

// Free all resources (model, caches, Metal, I/O pool).
void flashmoe_free(FlashMoE_Model *model);

// ---- Cache lifecycle ----

FlashMoE_Cache *flashmoe_cache_new(FlashMoE_Model *model);
FlashMoE_Cache *flashmoe_cache_clone(FlashMoE_Cache *src);
void            flashmoe_cache_free(FlashMoE_Cache *c);
void            flashmoe_cache_reset(FlashMoE_Cache *c, FlashMoE_Model *model);

// Number of tokens already cached (position in the sequence).
int flashmoe_cache_position(FlashMoE_Cache *c);

// ---- Inference ----

// Forward pass: process input_ids[0..n_tokens-1] through the model.
// On success: writes n_tokens * vocab_size logits into logits_out.
// logits_out must be pre-allocated with n_tokens * vocab_size floats.
// Updates cache in-place. Returns 0 on success, -1 on error.
int flashmoe_forward(FlashMoE_Model *model,
                     const int *input_ids, int n_tokens,
                     float *logits_out, FlashMoE_Cache *cache);

// ---- Accessors ----

int flashmoe_vocab_size(FlashMoE_Model *model);
int flashmoe_hidden_dim(FlashMoE_Model *model);
int flashmoe_num_layers(FlashMoE_Model *model);

#ifdef __cplusplus
}
#endif

#endif // MOE_INFER_C_H
