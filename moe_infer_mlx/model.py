"""
Python API for Flash-MoE inference engine.

    import moe_infer_mlx as fm

    with fm.Model("data") as model:
        cache = fm.Cache()
        logits, cache = model.forward(prompt_token_ids, cache)
"""

from __future__ import annotations

import moe_infer_mlx._core as _core


class Cache:
    """KV cache + recurrent state for a generation session."""

    def __init__(self):
        self._ptr = _core.cache_new()

    def __del__(self):
        if self._ptr:
            _core.cache_free(self._ptr)
            self._ptr = 0

    @property
    def position(self) -> int:
        return _core.cache_position(self._ptr)

    def reset(self) -> None:
        _core.cache_reset(self._ptr)


class Model:
    """Flash-MoE inference model loaded on Metal GPU."""

    def __init__(self, model_path: str):
        self._path = model_path
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        _core.init(self._path)
        self._loaded = True

    def unload(self) -> None:
        if not self._loaded:
            return
        _core.free_all()
        self._loaded = False

    def __enter__(self) -> Model:
        self.load()
        return self

    def __exit__(self, *args) -> None:
        self.unload()

    def forward(self, input_ids: list[int], cache: Cache):
        """Forward pass. Returns (logits, cache) — logits is float32[n_tokens, n_vocab]."""
        if not self._loaded:
            raise RuntimeError("Model not loaded — call .load() or use as context manager.")
        if not isinstance(input_ids, list) or not all(isinstance(t, int) for t in input_ids):
            raise TypeError("input_ids must be list[int]")
        logits, _ = _core.forward(input_ids, cache._ptr)
        return (logits, cache)
