# MoE-Infer

Run the Qwen3.6-35B-A3B MoE model on your Mac.  It chats, streams
word-by-word, and understands images — all on-device, no internet needed.

## What you need

- A Mac with Apple Silicon (M1 or newer) and at least 16 GB of RAM
- About 30 GB of free disk space for the model
- The original HuggingFace model downloaded to `hub/models--Qwen--Qwen3.6-35B-A3B`

## Setup (do this once)

Open Terminal and run these commands from the project folder:

```bash
# 1. Build and install
uv add moe_infer[all]  # or `uv sync --reinstall --extra vision` if you live inside this project

# 2. Download the model (about 30 GB)
hf download Qwen/Qwen3.6-35B-A3B \
  --local-dir hub/models--Qwen--Qwen3.6-35B-A3B

```

If you live in a cave, you can install with pip `pip install moe_infer[all]`

```python
# 3. Convert the model (takes ~5 minutes)
from moe_infer.qwen35_moe import bq4_convert
bq4_convert('hub/models--Qwen--Qwen3.6-35B-A3B', 'data/models--Qwen--Qwen3.6-35B-A3B', version='3.6')
```

When it's done you'll see a new folder with three things inside:

```
data/models--Qwen--Qwen3.6-35B-A3B/
├── model_bq4/       ← the model, compressed
├── tokenizer/       ← turns words into numbers (and back)
└── vision_encoder/  ← lets it "see" images
```

## Chat

Start Python and load the model:

```python
from moe_infer import Qwen35MoEPipeline

pipe = Qwen35MoEPipeline("data/models--Qwen--Qwen3.6-35B-A3B")
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

