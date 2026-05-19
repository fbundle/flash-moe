#ifndef GENERATE_H
#define GENERATE_H

// ============================================================================
// Autoregressive token generation with sampling.
// Included into moe_infer_c.m after moe_infer_c.h.
// ============================================================================

#include "common.h"
#include "moe_infer_c.h"

typedef struct {
    float prob;
    int idx;
} _ProbIdx;

static int _cmp_prob_desc(const void *a, const void *b) {
    float pa = ((const _ProbIdx *)a)->prob;
    float pb = ((const _ProbIdx *)b)->prob;
    return (pa < pb) - (pa > pb);
}

static int generate(FlashMoE_Context *m, FlashMoE_Cache *cache,
                    int first_token_id,
                    int *output_ids, int max_completion_length,
                    int eos_token_id, float temperature,
                    int top_k, float top_p, float min_p)
{
    int V = m->cfg.vocab_size;
    float *logits = malloc(V * sizeof(float));
    _ProbIdx *sorted = malloc(V * sizeof(_ProbIdx));
    if (!logits || !sorted) {
        free(logits);
        free(sorted);
        return -1;
    }

    int n_gen = 0;
    int next_id = first_token_id;

    for (int step = 0; step < max_completion_length; step++) {
        if (flashmoe_forward(m, &next_id, 1, logits, cache) != 0) {
            free(logits); free(sorted);
            return -1;
        }

        if (temperature <= 0.0f) {
            // Greedy
            float best = logits[0];
            int best_i = 0;
            for (int i = 1; i < V; i++) {
                if (logits[i] > best) { best = logits[i]; best_i = i; }
            }
            next_id = best_i;
        } else {
            // Temperature scaling + softmax (numerically stable)
            float max_l = logits[0];
            for (int i = 1; i < V; i++)
                if (logits[i] > max_l) max_l = logits[i];

            float inv_t = 1.0f / temperature;
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
                logits[i] = expf((logits[i] - max_l) * inv_t);
                sum += logits[i];
            }

            if (sum <= 0.0f) {
                // Degenerate case: all zero probs, fall back to argmax
                float best = logits[0];
                int best_i = 0;
                for (int i = 1; i < V; i++) {
                    if (logits[i] > best) { best = logits[i]; best_i = i; }
                }
                next_id = best_i;
            } else {
                float norm = 1.0f / sum;
                for (int i = 0; i < V; i++) {
                    sorted[i].prob = logits[i] * norm;
                    sorted[i].idx = i;
                }

                // Sort descending by probability
                qsort(sorted, V, sizeof(_ProbIdx), _cmp_prob_desc);

                // Apply top-k
                int cutoff = V;
                if (top_k > 0 && top_k < cutoff) cutoff = top_k;

                // Apply top-p (nucleus)
                if (top_p > 0.0f && top_p < 1.0f) {
                    float cum = 0.0f;
                    for (int i = 0; i < cutoff; i++) {
                        cum += sorted[i].prob;
                        if (cum >= top_p) {
                            cutoff = i + 1;
                            break;
                        }
                    }
                }

                // Apply min-p
                if (min_p > 0.0f) {
                    float max_p = sorted[0].prob;
                    float thresh = min_p * max_p;
                    for (int i = 0; i < cutoff; i++) {
                        if (sorted[i].prob < thresh) {
                            cutoff = i;
                            break;
                        }
                    }
                }

                if (cutoff < 1) cutoff = 1;

                // Renormalize and sample
                float cum = 0.0f;
                for (int i = 0; i < cutoff; i++) cum += sorted[i].prob;
                float inv_cum = cum > 0.0f ? 1.0f / cum : 1.0f;

                float r = (float)arc4random() / (float)UINT32_MAX;
                float acc = 0.0f;
                next_id = sorted[0].idx; // fallback
                for (int i = 0; i < cutoff; i++) {
                    acc += sorted[i].prob * inv_cum;
                    if (r < acc) {
                        next_id = sorted[i].idx;
                        break;
                    }
                }
            }
        }

        if (next_id == eos_token_id) break;
        output_ids[n_gen++] = next_id;
    }

    free(logits);
    free(sorted);
    return n_gen;
}

#endif // GENERATE_H
