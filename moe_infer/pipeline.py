"""High-level MoE inference pipeline — text + vision in, text out.

Handles tokenization, chat templates, KV-cache management, vision
embedding splicing, and autoregressive generation.  Model-specific
behaviour (EOS tokens, response extraction) is set by subclasses.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Iterator

import numpy as np

import _moe_infer_rs as _rs  # type: ignore[import-untyped]

from moe_infer.generation import generate_from
from moe_infer.hub import load_tokenizer


def _preprocess_image(path: str, processor: Any, min_pixels: int, max_pixels: int):
    """Run the HF image processor on a single image path.

    If *max_pixels* is 0, the processor default (16777216) is used.
    If the image is smaller than *max_pixels*, the image's actual
    pixel count is used instead to avoid upscaling.
    """
    from PIL import Image  # type: ignore[import-untyped]

    img = Image.open(path).convert("RGB")
    img_pixels = img.width * img.height

    proc_kwargs: dict[str, Any] = {"images": img, "return_tensors": "pt"}
    if min_pixels > 0 or max_pixels > 0:
        longest = max_pixels or 16777216
        if img_pixels < longest:
            longest = img_pixels
        proc_kwargs["size"] = {
            "shortest_edge": min_pixels or 65536,
            "longest_edge": longest,
        }
    inputs = processor(**proc_kwargs)
    pv = inputs["pixel_values"]
    grid = inputs["image_grid_thw"]
    n_merged = int((grid[0, 1] // 2) * (grid[0, 2] // 2))
    return pv, grid, n_merged


def _check_mtp(model_dir: str) -> bool:
    """Return True if the quantized model has MTP layers."""
    import json
    import os

    cfg_path = os.path.join(model_dir, "config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    tc = cfg.get("text_config", cfg)
    return tc.get("mtp_num_hidden_layers", 0) > 0


def _print_telemetry(t_start: float, t_first_token: float, num_tokens: int) -> None:
    ttft_ms = (t_first_token - t_start) * 1000.0
    tt_ms = (time.time() - t_start) * 1000.0
    num_gen = num_tokens - 1
    gen_s = (tt_ms - ttft_ms) / 1000.0
    tok_s = num_gen / gen_s if gen_s > 0 and num_gen > 0 else 0.0
    print(
        f"\n── TTFT: {ttft_ms:,.0f}ms | TT: {tt_ms:,.0f}ms |"
        f" tokens: {num_tokens} | tok/s: {tok_s:,.1f} ──",
        file=sys.stderr,
    )


def _word_stream(
    tokenizer: Any, token_ids: list[int], cursor: list[int],
) -> str:
    """Return newly-completed whitespace-delimited text since *cursor*.

    Decodes the full accumulated token list and returns the prefix up
    to the last space or newline that hasn't been yielded yet.  The
    incomplete suffix is held back so the next token can complete it.
    *cursor* is a mutable one-element list tracking the byte offset
    already yielded.
    """
    text = tokenizer.decode(token_ids)
    new = text[cursor[0] :]
    idx = max(new.rfind(" "), new.rfind("\n"))
    if idx < 0:
        return ""          # no word boundary yet — hold everything
    chunk = new[: idx + 1]
    cursor[0] += len(chunk)
    return chunk


# ── Pipeline ─────────────────────────────────────────────────────────────────


class Pipeline:
    """High-level MoE inference pipeline.

    Handles text and vision input, tokenization, chat templates,
    KV-cache management, and autoregressive generation.

    Subclass and set :attr:`eos_ids` to customise for a specific model
    family (see :class:`~moe_infer.qwen35_moe.pipeline.Qwen35MoEPipeline`).

    Parameters
    ----------
    model_path : str
        Path to the quantized model directory.
    hub : str or None
        Path to the HF hub for tokenizer and vision encoder.
        If omitted, *tokenizer* must be provided and vision is disabled.
    tokenizer :
        Pre-loaded HF tokenizer.  If omitted, loaded from *hub*.
    mode : str
        Pipeline mode passed to :class:`~moe_infer.Engine`.
    k : int
        Active experts per token.  0 = model default.
    """

    # ── Set in subclasses ───────────────────────────────────────────────

    eos_ids: tuple[int, ...] = ()  # override in subclass

    @classmethod
    def _extract_response(cls, raw: str) -> str:
        """Post-process a raw completion into the final response.

        Override in subclasses to strip model-specific tokens
        (EOS markers, think blocks, etc.).  The default is a no-op.
        """
        return raw

    # ── Init ────────────────────────────────────────────────────────────

    @staticmethod
    def _read_eos_tokens(model_dir: str, tok_dir: str | None) -> tuple[int, ...]:
        """Collect EOS token IDs from model config + generation config.

        Checks (in order):
        1. ``generation_config.json`` in *tok_dir* (definitive list)
        2. ``config.json`` in *model_dir* (single ``eos_token_id``)
        """
        import json
        import os as _os

        eos: set[int] = set()

        # 1. generation_config.json (e.g. [248046, 248044] for Qwen)
        if tok_dir is not None:
            gen_cfg_path = _os.path.join(tok_dir, "generation_config.json")
            try:
                with open(gen_cfg_path) as f:
                    gc = json.load(f)
                gen_eos = gc.get("eos_token_id")
                if isinstance(gen_eos, list):
                    eos.update(gen_eos)
                elif isinstance(gen_eos, int):
                    eos.add(gen_eos)
            except (OSError, json.JSONDecodeError):
                pass

        # 2. model config.json (single eos_token_id)
        config_path = _os.path.join(model_dir, "config.json")
        try:
            with open(config_path) as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            cfg = {}
        cfg_eos = cfg.get("eos_token_id")
        if cfg_eos is None and isinstance(cfg.get("text_config"), dict):
            cfg_eos = cfg["text_config"].get("eos_token_id")
        if isinstance(cfg_eos, int):
            eos.add(cfg_eos)

        return tuple(sorted(eos)) if eos else ()

    def __init__(
        self,
        model_path: str,
        *,
        hub: str | None = None,
        tokenizer: Any = None,
        mode: str = "Qwen35MoEFusedExp2",
        k: int = 0,
        quantize_mode: str = "bq4",
    ) -> None:
        import os

        root = model_path  # saved for sibling lookups (tokenizer/, vision_encoder/)

        # Auto-discover converted format: root/model_bq4/ (or model_int4/)
        model_subdir = os.path.join(root, f"model_{quantize_mode}")
        if os.path.isdir(model_subdir):
            if tokenizer is None:
                tok_dir = os.path.join(root, "tokenizer")
                if os.path.isdir(tok_dir):
                    tokenizer = load_tokenizer(tok_dir)
            if hub is None:
                vis_dir = os.path.join(root, "vision_encoder")
                if os.path.isdir(vis_dir):
                    hub = vis_dir
            model_path = model_subdir

        # LM (raw Rust types — wrappers break cross-calls)
        self._model = _rs.Model(model_path)
        self._engine = _rs.Engine(self._model, mode, k)
        self._cache = _rs.Cache(self._model)

        # Detect MTP support
        self._has_mtp = _check_mtp(model_path)

        # Tokenizer
        if tokenizer is not None:
            self._tokenizer = tokenizer
        elif hub is not None:
            self._tokenizer = load_tokenizer(hub)
        else:
            raise ValueError("Either hub or tokenizer must be provided")

        # EOS token IDs — read from config + generation_config if not overridden
        if not self.eos_ids:
            tok_dir = os.path.join(root, "tokenizer") if os.path.isdir(os.path.join(root, "tokenizer")) else None
            self.eos_ids = self._read_eos_tokens(model_path, tok_dir)

        # Vision (lazy)
        self._hub = hub
        self._vision_encoder: Any = None
        self._image_processor: Any = None

        # Conversation state
        self._messages: list[dict[str, str]] = []

    # ── Public API ──────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        *,
        images: list[str] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        min_image_pixels: int = 0,
        max_image_pixels: int = 0,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Send a message and get the assistant's response.

        Multi-turn conversation state is preserved via the KV cache.
        Call :meth:`reset` to start fresh.
        """
        # Build message content — use structured image items when images
        # are present so the chat template emits <|image_pad|> tokens.
        t_start = time.time()

        if images:
            content: list[dict[str, str]] = []
            for _ in images:
                content.append({"type": "image"})
            content.append({"type": "text", "text": message})
            self._messages.append({"role": "user", "content": content})
            embeds = self._build_vision_input(
                images, min_image_pixels, max_image_pixels,
            )
        else:
            self._messages.append({"role": "user", "content": message})
            input_ids = np.array(
                self._tokenizer.apply_chat_template(
                    self._messages,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ).input_ids,
                dtype=np.int64,
            )[self._cache.pos :]
            embeds = self._engine.embed_lookup(input_ids)

        logits = self._engine.forward_hidden(embeds, self._cache)
        t_first_token = time.time()

        if stream:
            return self._stream_chat(
                logits[-1], t_start, t_first_token,
                max_tokens, temperature, top_k, top_p, min_p,
            )

        tokens: list[int] = []

        def _on_token(tok: int) -> None:
            tokens.append(tok)

        completion, _stats = generate_from(
            logits[-1],
            self._engine,
            self._cache,
            self._tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            eos_ids=self.eos_ids,
            on_token=_on_token,
        )

        _print_telemetry(t_start, t_first_token, len(tokens))

        response = self._extract_response(completion)
        self._messages.append({"role": "assistant", "content": response})
        return response

    def _stream_chat(
        self,
        first_logits: np.ndarray,
        t_start: float,
        t_first_token: float,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
    ) -> Iterator[str]:
        """Run generation inline, yielding whitespace-delimited word chunks."""
        from moe_infer.sampling import sample

        last = np.asarray(first_logits)
        token_ids: list[int] = []
        cursor = [0]  # byte offset already yielded

        for _ in range(max_tokens):
            tok = sample(last, temperature, top_k, top_p, min_p)
            if tok in self.eos_ids:
                break
            token_ids.append(tok)
            chunk = _word_stream(self._tokenizer, token_ids, cursor)
            if chunk:
                yield chunk
            emb = self._engine.embed_lookup(
                np.array([tok], dtype=np.int64),
            )
            last = self._engine.forward_hidden(emb, self._cache)[0]

        # Flush remainder
        text = self._tokenizer.decode(token_ids)
        remainder = text[cursor[0] :]
        if remainder:
            yield remainder

        _print_telemetry(t_start, t_first_token, len(token_ids))

        response = self._extract_response(text)
        self._messages.append({"role": "assistant", "content": response})

    def reset(self) -> None:
        """Clear conversation history and reset the KV cache."""
        self._messages.clear()
        self._cache.reset()

    @property
    def messages(self) -> list[dict[str, str]]:
        """Current conversation history."""
        return list(self._messages)

    @property
    def telemetry(self) -> dict[str, Any]:
        """Engine timing telemetry."""
        return self._engine.telemetry()

    # ── Vision (subclasses must override) ───────────────────────────────

    @staticmethod
    def _load_vision_encoder(hub_path: str) -> Any:
        """Load a vision encoder from *hub_path*.

        Subclasses must override this — the base class has no default
        vision encoder.
        """
        raise NotImplementedError(
            f"{Pipeline.__name__}._load_vision_encoder — "
            "use a subclass that provides a vision encoder"
        )

    def _ensure_vision(self) -> None:
        """Lazy-load vision encoder and image processor."""
        if self._vision_encoder is not None:
            return
        if self._hub is None:
            raise RuntimeError(
                "Vision requires hub path at Pipeline construction time"
            )
        from transformers import AutoImageProcessor

        self._vision_encoder = self._load_vision_encoder(self._hub)
        self._image_processor = AutoImageProcessor.from_pretrained(self._hub)

    def _build_vision_input(
        self,
        images: list[str],
        min_pixels: int,
        max_pixels: int,
    ) -> np.ndarray:
        """Build embeddings via ``apply_chat_template``, splicing vision
        features at ``<|image_pad|>`` token positions."""
        import torch

        self._ensure_vision()
        assert self._vision_encoder is not None
        assert self._image_processor is not None

        # 1. Run vision encoder on all images
        vis_feats: list[tuple[np.ndarray, int]] = []
        for img_path in images:
            pv, grid, n_merged = _preprocess_image(
                img_path, self._image_processor, min_pixels, max_pixels,
            )
            with torch.no_grad():
                out = self._vision_encoder(pv, grid)
            feats = out.pooler_output.numpy().astype(np.float32)
            vis_feats.append((feats, n_merged))

        # 2. Apply chat template — emits <|image_pad|> per image
        input_ids = np.array(
            self._tokenizer.apply_chat_template(
                self._messages,
                add_generation_prompt=True,
                enable_thinking=False,
            ).input_ids,
            dtype=np.int64,
        )[self._cache.pos :]

        # 3. Find <|image_pad|> token positions
        pad_id = self._tokenizer.convert_tokens_to_ids("<|image_pad|>")
        pad_positions = np.where(input_ids == pad_id)[0]
        n_pads = len(pad_positions)
        n_imgs = len(vis_feats)
        if n_pads < n_imgs:
            raise RuntimeError(
                f"Expected at least {n_imgs} <|image_pad|> tokens "
                f"but found only {n_pads}"
            )

        # 4. Embed text tokens and splice vision features at pad positions.
        #    Each image's features replace ONE <|image_pad|> token.  Extra
        #    pad tokens (e.g. from multi-turn history) stay as text embeds.
        hidden_dim = vis_feats[0][0].shape[1]
        total_len = len(input_ids) - min(n_pads, n_imgs) + sum(
            n for _, n in vis_feats
        )
        embeds = np.empty((total_len, hidden_dim), dtype=np.float32)

        src = 0
        dst = 0
        for img_idx, pad_pos in enumerate(pad_positions):
            seg = input_ids[src:pad_pos]
            if len(seg) > 0:
                seg_emb = self._engine.embed_lookup(seg)
                embeds[dst : dst + len(seg)] = seg_emb
                dst += len(seg)
            if img_idx < n_imgs:
                feats, n = vis_feats[img_idx]
                embeds[dst : dst + len(feats)] = feats
                dst += len(feats)
            else:
                # Extra pad token — embed as regular text
                pad_emb = self._engine.embed_lookup(
                    np.array([pad_id], dtype=np.int64),
                )
                embeds[dst : dst + 1] = pad_emb
                dst += 1
            src = pad_pos + 1

        seg = input_ids[src:]
        if len(seg) > 0:
            seg_emb = self._engine.embed_lookup(seg)
            embeds[dst : dst + len(seg)] = seg_emb

        return embeds
