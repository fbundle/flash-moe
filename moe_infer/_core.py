"""Private bindings over _moe_infer_rs — not part of the public API.

These thin wrappers delegate to the Rust extension via __getattr__,
adding docstrings and type annotations.  User code imports from
``moe_infer``, never from here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import _moe_infer_rs as _rs  # type: ignore[import-untyped]


# ── Model ─────────────────────────────────────────────────────────────────────

class Model:
    """A quantized MoE language model loaded from disk.

    Parameters
    ----------
    path : str
        Directory containing ``config.json``, ``model_weights.bin``,
        ``model_weights.json``, and ``packed_experts/``.
    """

    def __init__(self, path: str) -> None:
        self._inner = _rs.Model(path)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return self._inner.__repr__()


# ── Engine ────────────────────────────────────────────────────────────────────

class Engine:
    """GPU inference engine backed by hand-tuned Metal compute shaders.

    Parameters
    ----------
    model : Model
        A loaded :class:`Model` instance.
    pipeline_mode : str
        One of ``"Qwen35MoEFusedExp1"`` or ``"Qwen35MoEFusedExp2"``.
    k : int
        Active experts per token.  0 means "use model default" (8 for Qwen3.6).
    """

    def __init__(
        self,
        model: Model,
        pipeline_mode: str = "Qwen35MoEFusedExp2",
        k: int = 0,
    ) -> None:
        self._inner = _rs.Engine(model._inner, pipeline_mode, k)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return self._inner.__repr__()


# ── Cache ─────────────────────────────────────────────────────────────────────

class Cache:
    """KV-cache and linear-attention (DeltaNet) state for a conversation.

    Parameters
    ----------
    model : Model
        A loaded :class:`Model` instance used to size the caches.
    """

    def __init__(self, model: Model) -> None:
        self._inner = _rs.Cache(model._inner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return self._inner.__repr__()


# ── HfRepo ────────────────────────────────────────────────────────────────────

class HfRepo:
    """HuggingFace repo file downloader (local or remote).

    Parameters
    ----------
    repo_id : str
        HuggingFace repo ID (e.g. ``"Qwen/Qwen3.6-35B-A3B"``) or a
        local directory path.
    """

    def __init__(self, repo_id: str) -> None:
        self._inner = _rs.PyHfRepo(repo_id)

    def ensure(self, filename: str) -> str:
        """Download *filename* and return its local path."""
        return self._inner.ensure(filename)

    def ensure_batch(self, filenames: list[str]) -> list[str]:
        """Download multiple files in parallel (Rust-side threading).
        Returns local paths in the same order."""
        return self._inner.ensure_batch(filenames)

    def file_size(self, filename: str) -> int:
        """Get a file's expected size in bytes (from HF API or local fs)."""
        return self._inner.file_size(filename)

    def file_sizes(self) -> list[tuple[str, int]]:
        """Get (filename, size_bytes) for all files in the repo."""
        return self._inner.file_sizes()

    def remove(self, filename: str) -> None:
        """Delete a cached file from the staging directory."""
        self._inner.remove(filename)

    def ls(self, dir: str | None = None) -> list[str]:
        """List immediate children of *dir* (defaults to root).  Behaves like
        UNIX ``ls``: returns names of files and directories at that level."""
        return self._inner.ls(dir)

    @property
    def path(self) -> str:
        """Local staging directory path."""
        return self._inner.path

    @property
    def is_hf(self) -> bool:
        """True if this is a remote HF repo (vs a local directory)."""
        return self._inner.is_hf

    def __repr__(self) -> str:
        return f"HfRepo({self.path!r})"


# ── Top-level functions ──────────────────────────────────────────────────────

def record_engine_telemetry(on: bool) -> None:
    """Enable or disable per-layer GPU timing telemetry globally.

    When enabled, :meth:`Engine.telemetry` returns a dict with keys like
    ``"engine.expert_io_ms"``, ``"engine.total_ms"``, etc.
    """
    _rs.record_engine_telemetry(on)


# ── Qwen-specific re-exports (moved to moe_infer.qwen35_moe) ─────────────────

from moe_infer.qwen35_moe import (  # noqa: E402
    convert as qwen35_moe_convert,
    extract_tokenizer as qwen35_moe_extract_tokenizer,
    extract_vision as qwen35_moe_extract_vision,
    quantize as qwen35_moe_quantize,
)
