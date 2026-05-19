#ifndef MOE_INFER_C_H
#define MOE_INFER_C_H

// ============================================================================
// C API for Flash-MoE inference engine.
// Instance-based: flashmoe_init() returns an opaque FlashMoE_Context pointer;
// all functions take it as their first argument.
// Called from Cython/Python.
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// ---- Opaque handles ----

typedef struct FlashMoE_Cache FlashMoE_Cache;
typedef struct FlashMoE_Context FlashMoE_Context;

// ---- Model lifecycle ----

// Initialize the inference engine from model_path.
// Returns an opaque model handle, or NULL on error.
FlashMoE_Context *flashmoe_init(const char *model_path);

// Free all resources (model, caches, Metal, I/O pool).
void flashmoe_free(FlashMoE_Context *model);

// ---- Cache lifecycle ----

FlashMoE_Cache *flashmoe_cache_new(FlashMoE_Context *model);
FlashMoE_Cache *flashmoe_cache_clone(FlashMoE_Cache *src);
void            flashmoe_cache_free(FlashMoE_Cache *c);
void            flashmoe_cache_reset(FlashMoE_Cache *c, FlashMoE_Context *model);

// Number of tokens already cached (position in the sequence).
int flashmoe_cache_position(FlashMoE_Cache *c);

// ---- Inference ----

// Forward pass: process input_ids[0..n_tokens-1] through the model.
// On success: writes n_tokens * vocab_size logits into logits_out.
// logits_out must be pre-allocated with n_tokens * vocab_size floats.
// Updates cache in-place. Returns 0 on success, -1 on error.
int flashmoe_forward(FlashMoE_Context *model,
                     const int *input_ids, int n_tokens,
                     float *logits_out, FlashMoE_Cache *cache);

// ---- Generation (sampling) ----

// Autoregressive generation from first_token_id.
// Samples with temperature/top_k/top_p/min_p; writes tokens into output_ids.
// Returns number of tokens generated (stops at eos_token_id or max_completion_length).
// output_ids must be pre-allocated with max_completion_length ints.
int flashmoe_generate(FlashMoE_Context *model,
                      FlashMoE_Cache *cache,
                      int first_token_id,
                      int *output_ids, int max_completion_length,
                      int eos_token_id, float temperature,
                      int top_k, float top_p, float min_p);

// ---- Accessors ----

int flashmoe_vocab_size(FlashMoE_Context *model);
int flashmoe_hidden_dim(FlashMoE_Context *model);
int flashmoe_num_layers(FlashMoE_Context *model);

#ifdef __cplusplus
}
#endif

#endif // MOE_INFER_C_H
