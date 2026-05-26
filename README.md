# MoE-Infer

Fast Mixture-of-Experts inference on Apple Silicon.  Pure Rust engine with
hand-tuned Metal shaders — no Python ML frameworks at runtime.  Expert
weights stream from SSD on demand via mmap.

## Quick Start

### 1. Build

```bash
maturin develop --release -m moe_infer_rs/Cargo.toml
```

### 2. Quantize

Download the HF model to `hub/models--Qwen--Qwen3.6-35B-A3B`, then:

```bash
python quantize.py \
    --model hub/models--Qwen--Qwen3.6-35B-A3B \
    --output data/models--Qwen--Qwen3.6-35B-A3B-bq4 \
    --qwen36
```

The `--qwen36` flag corrects Qwen3.6 norm weights to the Qwen3.5 convention
used by the engine.  Quantization takes ~10 minutes and produces:


### 3. Chat

```bash
python chat.py
python vision_demo.py
```
