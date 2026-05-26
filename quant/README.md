# BQ4: Block Quantization for MoE

BQ4 classifies every weight tensor by its **block** — the dot-separated path
before the last segment — and assigns a quantization format per block.
Sensitive blocks stay BF16; large, redundant blocks use affine INT4;
the lm_head uses per-channel symmetric INT8.

Vision encoder weights are skipped — they're loaded directly from HF
safetensors at runtime by `vision_demo.py`.

## Code structure

| File | Concern |
|------|---------|
| `moe_infer_rs/src/quant.rs` | `Quant` enum + dtype strings + BF16/INT4/INT8 encode/decode |
| `moe_infer_rs/src/quantize/qwen35_moe/bq4.rs` | `bq4()` classification, HF→BQ4 pipeline, `Bq4` struct |
| `moe_infer_rs/src/engine/qwen35_moe/metal_context.rs` | Reads dtype → dispatches GPU kernel via `string_to_quant()` |
| `quantize.py` | CLI entry point, calls `moe_infer.qwen35_moe_bq4_quantize()` |

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
expert passes.

**INT4 matrices** — experts (`mlp.switch_mlp.*`, `mlp.shared_expert.*`),
linear attention projections (`linear_attn.in_proj_*`, `out_proj`), embeddings
(`embed_tokens`), and MTP projection (`fc`).  These are the bulk of the
model (256 experts × 3 matrices × 40 layers).  Affine INT4 with per-group
(64) scale + bias.

**INT8 matrix** — `lm_head` only.  Per-channel symmetric quantization: one
float32 scale per output channel, signed int8 weights centered on zero.
Applied once at the final layer so quantization error does not compound.

**BF16 vectors** — norms, conv1d, dt_bias, and all quantization metadata
(`scales`, `biases`, `bias`).

## Naming convention and dtype strings

| Quant | `as_str()` | Storage |
|-------|-----------|---------|
| `Fp32` | `"f32"` | raw float32 |
| `Bf16` | `"bf16"` | raw bfloat16 |
| `Int4` | `"u32"` | packed uint32 + `{name}.scales` (bf16) + `{name}.biases` (bf16) |
| `Int8` | `"u8"` | packed int8 + `{name}.scales` (f32) |

The `.weight` suffix is **preserved** for BF16 and INT8 tensors.  For INT4,
the suffix is stripped to form a base name, then three entries are written:
`{.weight, .scales, .biases}`.

## Affine INT4

Standard affine per-group quantization (group size = 64):

```
scale = (max - min) / 15
bias  = min
w_q   = round((w_f32 - bias) / scale)  clamped to [0, 15]
```

Dequant: `w_f32 = nibble × scale + bias`

**Storage per group (64 weights):**
- 32 bytes packed nibbles (4-bit, LSB-first)
- 2 bytes BF16 scale
- 2 bytes BF16 bias
- Total: 36 bytes per group = 4.5 bits per weight

## INT8 (lm_head)

Per-channel symmetric quantization:

```
scale[i] = max(|w[i,:]|) / 127
w_q      = round(w_f32 / scale[i])   clamped to [-127, 127]
```

Dequant: `w_f32 = int8(w_q) × scale[i]`

**Storage for lm_head [248320, 2048]:**
- 484 MB packed int8 weights + 0.97 MB float32 scales
- ~485 MB vs 947 MB BF16 (49% reduction)

## Kernel dispatch

Each tensor's dtype (from the weight manifest JSON) is parsed via
`string_to_quant()` in `WeightBuffer::encode_matvec_into()`:

| `Quant` variant | Metal kernel | Dequant |
|-----------------|-------------|---------|
| `Bf16` | `matvec_bf16` | direct dot product |
| `Int8` | `matvec_int8` | `int8(w) × scale[row]` |
| `Int4` (default) | `dequant_matvec_4bit_v3` | `nibble × scale + bias` |

## Qwen3.5 vs Qwen3.6 norm weight convention

Qwen3.6 shifts RMS norm weights by -1.0 relative to Qwen3.5.  Our engines
follow the **Qwen3.5 convention** (no runtime shift).  Pass `--qwen36` to
`quantize.py` to normalize at quantization time.

Without `--qwen36` on a Qwen3.6 model, norm weights will be ~1.0 too low,
producing near-zero outputs.
