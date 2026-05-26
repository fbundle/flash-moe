"""
MoE-Infer: High-performance MoE inference engine for Apple Silicon.

This package wraps the native Rust module (moe_infer_rs) and re-exports
all public symbols for convenience.
"""

from moe_infer_rs import (  # type: ignore
    Model,
    Engine,
    Cache,
    record_engine_telemetry,
    qwen35_moe_bq4_quantize,
)

__all__ = [
    "Model",
    "Engine",
    "Cache",
    "record_engine_telemetry",
    "qwen35_moe_bq4_quantize",
]
