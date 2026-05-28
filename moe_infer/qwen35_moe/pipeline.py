"""Qwen3.5/3.6 MoE pipeline — Qwen-specific defaults and vision config."""

from __future__ import annotations

from typing import Any

from moe_infer.pipeline import Pipeline

# ── Mode → vision processor mapping ──────────────────────────────────────────

_VISION_CONFIG: dict[str, dict[str, str]] = {
    "Qwen35MoEFusedExp1": {
        "processor_class": "Qwen3VLProcessor",
        "image_processor_type": "Qwen2VLImageProcessorFast",
    },
    "Qwen35MoEFusedExp2": {
        "processor_class": "Qwen3VLProcessor",
        "image_processor_type": "Qwen2VLImageProcessorFast",
    },
}


class Qwen35MoEPipeline(Pipeline):
    """Pipeline pre-configured for Qwen3.5/3.6 MoE models.

    Sets Qwen-specific EOS tokens and response extraction.
    Auto-discovery of ``model_bq4/``, ``tokenizer/``, and
    ``vision_encoder/`` is handled by the base :class:`Pipeline`.

    Usage::

        # After convert:
        pipe = Qwen35MoEPipeline("data/models--Qwen--Qwen3.6-35B-A3B")
        pipe.chat("Hello!")

        # Or point directly at the model directory:
        pipe = Qwen35MoEPipeline("data/.../model_bq4", tokenizer=my_tok)
    """

    eos_ids: tuple[int, ...] = (248046, 248044)

    @classmethod
    def _extract_response(cls, raw: str) -> str:
        text = raw.removesuffix("<|im_end|>")
        parts = text.split("</think>")
        return parts[-1]

    @staticmethod
    def _load_vision_encoder(hub_path: str) -> Any:
        from moe_infer.qwen35_moe.vision import load_vision_encoder

        return load_vision_encoder(hub_path)
