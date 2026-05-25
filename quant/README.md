# BQ4: Block Quantization for MoE

BQ4 classifies every weight tensor by its **block** — the dot-separated path
before the last segment — and assigns a quantization format per block.
Sensitive blocks stay BF16; large, redundant blocks use affine INT4;
the lm_head uses per-channel symmetric INT8.

## Code structure

| File | Concern |
|------|---------|
| `moe_infer_rs/src/quant.rs` | `Quant` enum + dtype strings + BF16/INT4/INT8 encode/decode |
| `moe_infer_rs/src/quantize/qwen35_moe/bq4.rs` | Model-specific: `bq4()` classification, HF→BQ4 pipeline, `Bq4` struct |
| `moe_infer_rs/src/engine/qwen35_moe/metal_context.rs` | Reads dtype → dispatches GPU kernel via `string_to_quant()` |
| `quant/quantize.py` | CLI, calls `moe_infer.qwen35_moe_bq4_quantize()` |

`quant.rs` only deals with binary format concerns.  Model-specific naming
conventions, sanitization (Qwen3.6 norm shift), and the `bq4()` classifier
live in `bq4.rs`.

## Quantization rules

```haskell
data Quant = FP32 | BF16 | INT4 | INT8

matrixTable :: String -> Quant
matrixTable "self_attn.q_proj" = BF16
matrixTable "self_attn.k_proj" = BF16
matrixTable "self_attn.v_proj" = BF16
matrixTable "self_attn.o_proj" = BF16
matrixTable "mlp.gate"         = BF16
matrixTable "attn.qkv"         = BF16
matrixTable "attn.proj"        = BF16
matrixTable "patch_embed.proj" = BF16
matrixTable "pos_embed"        = BF16
matrixTable "lm_head"          = INT8    -- per-channel symmetric
matrixTable _                  = INT4

bq4 :: String -> Quant
bq4 name
  | kind == "A_log"   = FP32
  | kind == "weight"  = matrixTable block
  | kind == "scales"  = BF16
  | kind == "biases"  = BF16
  | kind == "bias"    = BF16
  | kind == "dt_bias" = BF16
  where
    (prefix, kind) = splitOnLastDot name
    block          = stripLayerPrefix prefix
```

1. **Scalars** (`A_log`) → FP32
2. **Vectors** (`scales`, `biases`, `bias`, `dt_bias`, and `weight` with ndim ≠ 2) → BF16
3. **Matrices** (`weight` with ndim = 2) → look up the block in the matrix table.
   If found, use the table format; otherwise INT4.

### Rationale

**BF16 matrices** — attention projections (`q_proj`, `k_proj`, `v_proj`,
`o_proj`, `qkv`, `proj`), router (`mlp.gate`), projection embeddings
(`patch_embed.proj`), and positional embeddings (`pos_embed`).  Attention Q·Kᵀ
amplifies quantization noise quadratically; router error misroutes tokens across
expert passes.  Sanitization (norm shift, conv1d moveaxis) is skipped for these
since they are pure 2D weight matrices — the pipeline checks `is_norm_key()`
and `--qwen36` only for vectors.

**INT4 matrices** — experts (`mlp.switch_mlp.*`, `mlp.shared_expert.*`),
linear attention projections (`linear_attn.in_proj_*`, `out_proj`), embeddings
(`embed_tokens`), vision FFN (`mlp.linear_fc*`), and MTP projection (`fc`).
These are the bulk of the model (256 experts × 3 matrices × 40 layers).
Affine INT4 with per-group (64) scale + bias.

**INT8 matrix** — `lm_head` only.  Per-channel symmetric quantization: one
float32 scale per output channel, signed int8 weights centered on zero.
Motivation: the lm_head is the single largest matrix (~947 MB BF16 → ~484 MB
INT8 + 0.97 MB scales), applied once at the final layer so quantization error
does not compound.

**BF16 vectors** — norms, conv1d, dt_bias, and all quantization metadata
(`scales`, `biases`, `bias`).

### Naming convention and dtype strings

`Quant::as_str()` defines the manifest dtype written into `model_weights.json`.
`string_to_quant()` does the reverse at runtime.

| Quant | `as_str()` | Storage |
|-------|-----------|---------|
| `Fp32` | `"f32"` | raw float32 |
| `Bf16` | `"bf16"` | raw bfloat16 |
| `Int4` | `"u32"` | packed uint32 + `{name}.scales` (bf16) + `{name}.biases` (bf16) |
| `Int8` | `"u8"` | packed int8 + `{name}.scales` (f32) |

Both the pipeline (writer) and the engine dispatch (reader) import from
`crate::quant` — the strings are never duplicated.

## Affine INT4

Standard affine per-group quantization: each group of 64 contiguous weights
is quantized independently.

```
scale = (max - min) / 15
bias  = min
w_q   = round((w_f32 - bias) / scale)  clamped to [0, 15]
```

Dequant: `w_f32 = nibble × scale + bias`

**Storage per group (64 weights):**
- 32 bytes packed nibbles (4-bit, 8 per uint32, LSB-first)
- 2 bytes BF16 scale
- 2 bytes BF16 bias
- Total: 36 bytes per group = 4.5 bits per weight

## INT8 (lm_head)

Per-channel symmetric quantization: signed int8 weights, one float32 scale per
output channel (vocab entry).

```
scale[i] = max(|w[i,:]|) / 127
w_q      = round(w_f32 / scale[i])   clamped to [-127, 127]
```

Dequant: `w_f32 = int8(w_q) × scale[i]`

No zero-point — symmetric around zero.  The `matvec_int8` Metal kernel
computes `sum += float(w_q) * scale[row] * x[col]` in one pass.

**Storage for lm_head [248320, 2048]:**
- 484 MB packed int8 weights
- 0.97 MB float32 scales (one per output channel)
- Total: ~485 MB vs 947 MB BF16 (49% reduction)

## Kernel dispatch

Dispatch lives in `WeightBuffer::encode_matvec_into()`.  Each tensor's dtype
(from the weight manifest JSON) is parsed via `string_to_quant()` and matched:

| `Quant` variant | Metal kernel | Dequant |
|-----------------|-------------|---------|
| `Bf16` | `matvec_bf16` | direct dot product |
| `Int8` | `matvec_int8` | `int8(w) × scale[row]` |
| `Int4` (default) | `dequant_matvec_4bit_v3` | `nibble × scale + bias` |

No engine variant needed — one engine dispatches per-tensor.  Mixed
quantization is a property of the weight file, not the runtime.

## Weight conversion

Split on the last dot to get the block and kind, then feed the name through
`bq4` above.  The resulting `Quant` is written as the `dtype` in the manifest
JSON.

The `.weight` suffix is **preserved** in the manifest for BF16 and INT8
weight tensors (e.g. `language_model.model.layers.0.input_layernorm.weight`).
For INT4 tensors, the suffix is stripped to form a base name, then three
separate entries are written: `{.weight, .scales, .biases}`.

## Qwen3.5 vs Qwen3.6 norm weight convention

Qwen3.6 changed the convention for RMS norm weights: they are shifted by -1.0
relative to Qwen3.5.  MLX-LM's sanitizer bakes a +1.0 correction into the
quantized weights so the runtime formula `y = x * w` works for both.

Our engines follow the **Qwen3.5 convention** (no runtime shift).  To quantize
a Qwen3.6 model, pass `--qwen36` to `quantize.py`.  This normalizes the norm
weights to Qwen3.5 convention at quantization time.

Without `--qwen36` on a Qwen3.6 model, norm weights will be ~1.0 too low,
causing all RMS norm operations to produce near-zero outputs and the model
to generate garbage.

## Adding a new model family

1. Add a `Quantize` implementation in `src/quantize/<family>/` (struct with
   model-specific `new()` params + a `quantize(input, output)` method).
2. Add arch dispatch in `python_bindings::qwen35_moe_bq4_quantize()` (or add
   a new Python-facing function).
3. Add a `matrix_table` mapping for the model's PyTorch module names.
