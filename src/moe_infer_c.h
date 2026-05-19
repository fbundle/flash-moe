#ifndef MOE_INFER_C_H
#define MOE_INFER_C_H

// C API for Flash-MoE inference engine.
// Single-instance (globals). Called from Cython/Python.

#ifdef __cplusplus
extern "C" {
#endif

// Opaque cache handle — owns KV caches, linear-attn states, and position.
typedef struct FlashMoE_Cache FlashMoE_Cache;

// Initialize the inference engine from model_path.
// Returns 0 on success, -1 on error.
int flashmoe_init(const char *model_path);

// Create / destroy caches. Cache holds recurrent state (KV, delta-net, position).
FlashMoE_Cache *flashmoe_cache_new(void);
FlashMoE_Cache *flashmoe_cache_clone(FlashMoE_Cache *src);
void            flashmoe_cache_free(FlashMoE_Cache *c);

// Forward pass: process input_ids[0..n_tokens-1] through the model.
// On success: writes n_tokens * vocab_size logits into logits_out.
// logits_out must be pre-allocated with n_tokens * flashmoe_vocab_size() floats.
// Updates cache in-place.
// Returns 0 on success, -1 on error.
int flashmoe_forward(const int *input_ids, int n_tokens,
                     float *logits_out, FlashMoE_Cache *cache);

// Reset cache for a fresh conversation.
void flashmoe_cache_reset(FlashMoE_Cache *c);

// Number of tokens already cached (position in the sequence).
int flashmoe_cache_position(FlashMoE_Cache *c);

// Accessors (values are compile-time constants from model_config.h).
int flashmoe_vocab_size(void);
int flashmoe_hidden_dim(void);
int flashmoe_num_layers(void);

// Free all resources (model, caches, Metal, I/O pool).
void flashmoe_free(void);

#ifdef __cplusplus
}
#endif

#endif // MOE_INFER_C_H
