# MoE-Infer

Run the Qwen3.6-35B-A3B MoE model on your Mac.  It chats, streams
word-by-word, and understands images — all on-device, no internet needed.

## What you need

- A Mac with Apple Silicon (M1 or newer) and at least 16 GB of RAM
- About 100 GB of free disk space (70 GB download + ~20 GB per quantized model)
- The original HuggingFace model downloaded to `hub/models--Qwen--Qwen3.6-35B-A3B`

## Setup (do this once)

Open Terminal and run these commands from the project folder:

```bash
# 1. Build and install
uv add moe_infer[all]  # or `uv sync --reinstall --extra vision` if you live inside this project

# 2. Download the model (about 70 GB)
hf download Qwen/Qwen3.6-35B-A3B \
  --local-dir hub/models--Qwen--Qwen3.6-35B-A3B

```

If you live in a cave, you can install with pip `pip install moe_infer[all]`

```python
# 3. Convert the model (takes ~5 minutes per scheme)
from moe_infer.qwen35_moe import convert


# BQ4 only (selective quantization — smaller, faster)
convert('hub/models--Qwen--Qwen3.6-35B-A3B', 'data/Qwen3.6-35B-A3B', version='3.6')

# INT4 only (all weights quantized)
convert('hub/models--Qwen--Qwen3.6-35B-A3B', 'data/Qwen3.6-35B-A3B', version='3.6', scheme='int4')

# Both at once
convert('hub/models--Qwen--Qwen3.6-35B-A3B', 'data/Qwen3.6-35B-A3B', version='3.6', scheme=['bq4', 'int4'])
```

When it's done you'll see:

```
data/Qwen3.6-35B-A3B/
├── model_bq4/        ← BQ4 compressed model (~19 GB)
├── model_int4/       ← INT4 compressed model (~18 GB)
├── tokenizer/        ← turns words into numbers (and back)
└── vision_encoder/   ← lets it "see" images
```

## Chat

Start Python and load the model:

```python
from moe_infer import Qwen35MoEPipeline

# Default: uses model_bq4/
pipe = Qwen35MoEPipeline("data/Qwen3.6-35B-A3B")

# INT4 quantization:
pipe = Qwen35MoEPipeline("data/Qwen3.6-35B-A3B", quantize_mode="int4")
```

### Text

```python
pipe.chat("Hello!")
# → 'Hello! How can I help you today?'

pipe.chat("What is the capital of France?")
# → 'The capital of France is **Paris**.'
```

The model remembers your conversation.  Call `pipe.reset()` to start fresh.

### Streaming

See words appear as the model generates them:

```python
for word in pipe.chat("Write a haiku about cats", stream=True):
    print(word, end="", flush=True)
```

### Images

Point it at a photo and ask about it:

```python
pipe.chat("What is in this photo?", images=["data/crycat-crying-cat.gif"], max_image_pixels=65536)
```

## Tips

| Tip | What to do |
|---|---|
| Text-only chat | `Qwen35MoEPipeline("data/...")` — it finds everything automatically |
| Reset conversation | `pipe.reset()` |
| Conversation history | `pipe.messages` — see what was said |
| Engine timing | `pipe.telemetry` — how long each step took |
| Switch quantization | `quantize_mode="int4"` or `quantize_mode="bq4"` (default) |
