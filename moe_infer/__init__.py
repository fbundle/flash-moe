"""MoE-Infer — fast Mixture-of-Experts inference on Apple Silicon.

Pure Rust engine with hand-tuned Metal shaders.  No Python ML
frameworks at runtime.  Expert weights stream from SSD on demand.

Quick start (high-level)
-------------------------
>>> from moe_infer import Pipeline
>>> pipe = Pipeline(
...     model="data/models--Qwen--Qwen3.6-35B-A3B-bq4",
...     hub="hub/models--Qwen--Qwen3.6-35B-A3B",
... )
>>> pipe.chat("Hello!")
>>> pipe.chat("What's in this image?", images=["cat.jpg"])

Low-level API
-------------
>>> from moe_infer import Model, Engine, Cache
>>> model = Model("data/models--Qwen--Qwen3.6-35B-A3B-bq4")
>>> engine = Engine(model)
>>> cache = Cache(model)
>>> import numpy as np
>>> ids = np.array([1, 2, 3], dtype=np.int64)
>>> logits = engine.forward(ids, cache)
"""

from moe_infer._core import (
    Cache,
    Engine,
    HfRepo,
    Model,
    qwen35_moe_convert,
    qwen35_moe_extract_tokenizer,
    qwen35_moe_extract_vision,
    qwen35_moe_quantize,
    record_engine_telemetry,
)
from moe_infer.generation import generate_from
from moe_infer.hub import load_tokenizer, load_vision_encoder
from moe_infer.pipeline import Pipeline
from moe_infer.qwen35_moe.pipeline import Qwen35MoEPipeline
from moe_infer.sampling import sample, softmax

__all__ = [
    "Cache",
    "Engine",
    "HfRepo",
    "Model",
    "Pipeline",
    "Qwen35MoEPipeline",
    "generate_from",
    "load_tokenizer",
    "load_vision_encoder",
    "qwen35_moe_convert",
    "qwen35_moe_extract_tokenizer",
    "qwen35_moe_extract_vision",
    "qwen35_moe_quantize",
    "record_engine_telemetry",
    "sample",
    "softmax",
]
