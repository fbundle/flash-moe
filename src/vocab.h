#ifndef VOCAB_H
#define VOCAB_H

// ============================================================================
// Vocabulary for token decoding
// ============================================================================

typedef struct {
    char **tokens;   // token_id -> UTF-8 string
    int *lengths;    // token_id -> byte length
    int num_tokens;
} Vocabulary;

static Vocabulary *load_vocab(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open vocab %s\n", path);
        return NULL;
    }

    uint32_t num_entries, max_id;
    fread(&num_entries, 4, 1, f);
    fread(&max_id, 4, 1, f);

    Vocabulary *v = calloc(1, sizeof(Vocabulary));
    v->num_tokens = max_id + 1;
    v->tokens = calloc(max_id + 1, sizeof(char *));
    v->lengths = calloc(max_id + 1, sizeof(int));

    for (uint32_t i = 0; i < num_entries; i++) {
        uint32_t token_id;
        uint16_t byte_len;
        fread(&token_id, 4, 1, f);
        fread(&byte_len, 2, 1, f);
        if (byte_len > 0) {
            v->tokens[token_id] = malloc(byte_len + 1);
            fread(v->tokens[token_id], 1, byte_len, f);
            v->tokens[token_id][byte_len] = '\0';
            v->lengths[token_id] = byte_len;
        }
    }

    fclose(f);
    printf("[vocab] Loaded %d tokens\n", num_entries);
    return v;
}

static const char *decode_token(Vocabulary *v, int token_id) {
    if (token_id < 0 || token_id >= v->num_tokens || !v->tokens[token_id]) {
        return "<unk>";
    }
    return v->tokens[token_id];
}

// ============================================================================
// Prompt tokens loader
// ============================================================================

typedef struct {
    uint32_t *ids;
    int count;
} PromptTokens;

static PromptTokens *load_prompt_tokens(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    PromptTokens *pt = calloc(1, sizeof(PromptTokens));
    fread(&pt->count, 4, 1, f);
    pt->ids = malloc(pt->count * sizeof(uint32_t));
    fread(pt->ids, 4, pt->count, f);
    fclose(f);
    return pt;
}

// ============================================================================
// C BPE tokenizer (replaces Python encode_prompt.py)
// ============================================================================
#define TOKENIZER_IMPL
#include "tokenizer.h"

static bpe_tokenizer g_tokenizer;
static int g_tokenizer_loaded = 0;

static void init_tokenizer(void) {
    if (g_tokenizer_loaded) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/tokenizer.bin", g_model_path);
    if (access(path, R_OK) == 0 && bpe_load(&g_tokenizer, path) == 0) {
        g_tokenizer_loaded = 1;
        return;
    }
    fprintf(stderr, "WARNING: tokenizer.bin not found at %s, tokenization will fail\n", path);
}

static PromptTokens *encode_prompt_text_to_tokens(const char *text) {
    init_tokenizer();
    if (!g_tokenizer_loaded) return NULL;

    // Allocate output buffer (generous: 4 tokens per character worst case)
    int max_ids = (int)strlen(text) * 4 + 256;
    uint32_t *ids = malloc(max_ids * sizeof(uint32_t));
    if (!ids) return NULL;

    int n = bpe_encode(&g_tokenizer, text, ids, max_ids);
    if (n < 0) { free(ids); return NULL; }

    PromptTokens *pt = calloc(1, sizeof(PromptTokens));
    pt->ids = ids;
    pt->count = n;

    fprintf(stderr, "Tokens (%d): [", n);
    for (int i = 0; i < n && i < 20; i++) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%u", ids[i]);
    }
    if (n > 20) fprintf(stderr, ", ...");
    fprintf(stderr, "]\n");

    return pt;
}


#endif // VOCAB_H
