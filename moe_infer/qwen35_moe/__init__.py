"""Qwen3.5/3.6 MoE — conversion, extraction, and vision utilities."""

from __future__ import annotations

import os as _os

import _moe_infer_rs as _rs  # type: ignore[import-untyped]


def extract_tokenizer(hub_path: str, output_dir: str) -> None:
    """Copy tokenizer files from a HF hub to *output_dir*."""
    import shutil

    _TOKENIZER_FILES = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
    ]

    _os.makedirs(output_dir, exist_ok=True)
    for name in _TOKENIZER_FILES:
        src = _os.path.join(hub_path, name)
        if _os.path.exists(src):
            shutil.copy2(src, _os.path.join(output_dir, name))


def extract_vision(hub_path: str, output_dir: str) -> None:
    """Copy vision-encoder files from a HF hub to *output_dir*."""
    import json
    import shutil

    _os.makedirs(output_dir, exist_ok=True)

    for name in ("config.json", "preprocessor_config.json"):
        src = _os.path.join(hub_path, name)
        if _os.path.exists(src):
            shutil.copy2(src, _os.path.join(output_dir, name))

    index_path = _os.path.join(hub_path, "model.safetensors.index.json")
    if not _os.path.exists(index_path):
        return

    with open(index_path) as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]

    vis_shards = sorted(
        {sn for k, sn in weight_map.items() if k.startswith("model.visual.")}
    )

    if not vis_shards:
        return

    shutil.copy2(index_path, _os.path.join(output_dir, "model.safetensors.index.json"))

    for shard_name in vis_shards:
        src = _os.path.join(hub_path, shard_name)
        if _os.path.exists(src):
            shutil.copy2(src, _os.path.join(output_dir, shard_name))

    print(
        f"[extract] Vision: {len(vis_shards)} shard(s) → {output_dir}",
        flush=True,
    )


def quantize(
    model_path: str,
    output_dir: str,
    *,
    version: str,
    scheme: str = "bq4",
) -> None:
    """Quantize a HF Qwen3.5-MoE model.

    Parameters
    ----------
    version : str
        Qwen generation: ``"3.5"`` or ``"3.6"``.
        Qwen3.6 applies a +1.0 norm-weight correction.
    scheme : str
        Quantization scheme: ``"bq4"`` (selective) or ``"int4"`` (all-INT4).
    """
    if version not in ("3.5", "3.6"):
        raise ValueError(f"version must be '3.5' or '3.6', got {version!r}")
    _rs.qwen35_moe_quantize(
        model_path,
        output_dir,
        version=version,
        scheme=scheme,
    )


def convert(
    input: str,
    output: str | None = None,
    *,
    version: str,
    scheme: str | list[str] = "bq4",
) -> None:
    """Full conversion: HF hub → quantized model + tokenizer + vision_encoder.

    Parameters
    ----------
    input : str
        Path to the HF hub directory.
    output : str or None
        Output root.  Defaults to ``data/<hub-basename>``.
    version : str
        Qwen generation: ``"3.5"`` or ``"3.6"``.
    scheme : str or list of str
        Quantization schemes: ``"bq4"``, ``"int4"``, or both via a list.
    """
    hub_path = input.rstrip("/")
    if output is None:
        output = f"data/{_os.path.basename(hub_path)}"

    schemes = scheme if isinstance(scheme, list) else [scheme]

    for s in schemes:
        model_dir = _os.path.join(output, f"model_{s}")
        print(f"[quantize] {s} → {model_dir}")
        quantize(hub_path, model_dir, version=version, scheme=s)

    print(f"[extract] Tokenizer → {output}/tokenizer")
    extract_tokenizer(hub_path, _os.path.join(output, "tokenizer"))

    print(f"[extract] Vision encoder → {output}/vision_encoder")
    extract_vision(hub_path, _os.path.join(output, "vision_encoder"))

    print(f"\nDone → {output}/")
    for s in schemes:
        print(f"  model_{s}/  ({s})")
    print(f"  tokenizer/")
    print(f"  vision_encoder/")
