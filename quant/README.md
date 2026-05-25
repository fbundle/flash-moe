# BQ4: Block Quantization 4-bit for MoE

BQ4 classifies every weight tensor by its **block** — the dot-separated path
before the last segment — and assigns a quantization format per block.
Sensitive blocks stay full-precision; large, redundant blocks use NormalFloat
4-bit (NF4).

## Naming convention

Tensors follow the MLX convention.  Split on the **last dot**: everything
before is the block, the last segment is the kind.  Blocks may contain dots;
kinds never do.

```
language_model.model.layers.{L}. self_attn.q_proj.  weight
└─────────── prefix ───────────┘└──── block ─────┘└─ kind ─┘

language_model.model.layers.{L}.  mlp.switch_mlp.gate_proj.  weight
└─────────── prefix ───────────┘└────────── block ────────┘ └ kind ┘

language_model. lm_head. weight
└── prefix ───┘└ block ┘└ kind ┘
```

Kinds are one of: `weight`, `scales`, `biases`, `bias`, `A_log`, `dt_bias`.

The prefix (`language_model.model.layers.{L}.`, `vision_tower.blocks.{B}.`,
`mtp.layers.{L}.`, etc.) is stripped to get the **relative block** used for
classification.

## Quantization rules

```haskell
data Quant = FP16 | NF4 | BF16 | FP32

matrixTable :: String -> Quant
matrixTable "self_attn.q_proj" = FP16
matrixTable "self_attn.k_proj" = FP16
matrixTable "self_attn.v_proj" = FP16
matrixTable "self_attn.o_proj" = FP16
matrixTable "mlp.gate"         = FP16
matrixTable "lm_head"          = FP16
matrixTable "attn.qkv"         = FP16
matrixTable "attn.proj"        = FP16
matrixTable "patch_embed.proj" = FP16
matrixTable _                  = NF4

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

1. **Scalar** (`A_log`) → FP32
2. **Metadata** (`scales`, `biases`, `bias`, `dt_bias`) → BF16
3. **Weight** — look up the block in the table.  If found, use the
   table format; otherwise NF4.

### Rationale

**FP16 matrices** — attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`,
`qkv`, `proj`) and routers (`mlp.gate`, `lm_head`).  Attention Q·Kᵀ amplifies
quantization noise quadratically; router error misroutes tokens across 8
expert passes; lm_head directly produces logits.

**NF4 matrices** — experts (`mlp.switch_mlp.*`, `mlp.shared_expert.*`), linear
attention projections (`linear_attn.in_proj_*`, `out_proj`), embeddings
(`embed_tokens`), vision FFN (`mlp.linear_fc*`), and MTP projection (`fc`).

**BF16** — everything else: all vectors (norms, conv1d, pos_embed, dt_bias)
and all quantization metadata (`scales`, `biases`, `bias`).

## NF4 encoding

NF4 (4-bit NormalFloat) uses 16 non-uniform quantization levels optimised for
a normal distribution.  The levels are equally probable under N(0,1) rather
than equally spaced:

```
[-1.0, -0.696, -0.525, -0.395, -0.284, -0.188, -0.101, -0.02,
  0.02,  0.101,  0.188,  0.284,  0.395,  0.525,  0.696,  1.0]
```

**Storage per group (64 weights):**
- 32 bytes packed weights (4-bit indices into the NF4 LUT)
- 2 bytes FP16 scale (no bias — NF4 is zero-centred)
- Total: 34 bytes per group = 4.25 bits per weight

Compare to the current INT4 scheme: 32 + 2 + 2 = 36 bytes = 4.5 bits per
weight.  NF4 saves ~0.25 bits per weight *and* reduces quantization error.

## Kernel dispatch

Dispatch lives in `WeightBuffer::encode_matvec_into()`.  Each tensor's dtype
(from the weight manifest JSON) determines the Metal pipeline:

| dtype   | Kernel                    | Dequant                                     |
|---------|---------------------------|---------------------------------------------|
| `"u32"` | `matvec_int4` (current)   | `nibble * scale + bias`                     |
| `"nf4"` | `matvec_nf4`              | `NF4_LUT[nibble] * scale`                   |
| `"f16"` | `matvec_fp16`             | direct dot product, no dequant              |

No engine variant needed — one engine dispatches per-tensor.  Mixed
quantization is a property of the weight file, not the runtime.

## Weight conversion

Split on the last dot to get the block and kind, then feed the name through
`bq4` above.  The resulting `Quant` is written as the `dtype` in the manifest
JSON.

## Expert router

The expert router is a single linear projection `W_gate ∈ R[num_experts × hidden_dim]`
stored as `language_model.model.layers.{L}.mlp.gate.weight`.

**Forward pass:**
1. Post-attention hidden state is RMS-normed to produce `h ∈ R[hidden_dim]`
2. GPU: `scores = W_gate · h` (into `buf_gate_scores`), executed inside the
   op1 encoder alongside attention projections
3. CPU: softmax → top-k → normalize → select expert buffers

**Why FP16.**  The gate is `[256 × 2048]` = ~500K floats.  Quantizing it saves
~1.5MB total across all 40 layers — negligible.  But a single bit-flip can
reroute a token from expert 47 to expert 231, wasting all subsequent expert
computation.  The error multiplier makes this the most expensive quantization
in the model per byte saved.
