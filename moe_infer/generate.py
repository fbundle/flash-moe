"""Shared generation helpers used by chat.py and vision_demo.py."""

import time
from collections.abc import Callable

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def sample(logits: np.ndarray, temperature: float = 0.0,
           top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0) -> int:
    """Sample a token from logits. Modifies logits in-place."""
    n = len(logits)
    if abs(temperature - 1.0) > 1e-7:
        logits /= max(temperature, 1e-8)
    if temperature < 0.01:
        return int(np.argmax(logits))
    probs = softmax(logits)
    if top_k > 0 and top_k < n:
        indices = np.argpartition(probs, -top_k)[-top_k:]
        mask = np.ones(n, dtype=bool)
        mask[indices] = False
        probs[mask] = 0.0
    if top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_idx])
        cutoff_idx = np.searchsorted(cumsum, top_p)
        if cutoff_idx < n:
            probs[sorted_idx[cutoff_idx + 1:]] = 0.0
    if min_p > 0.0:
        threshold = probs.max() * min_p
        probs[probs < threshold] = 0.0
    total = probs.sum()
    if total <= 0:
        return 0
    probs /= total
    return int(np.random.choice(n, p=probs))


def generate_from(first_logits: np.ndarray,
                  engine, cache, tokenizer, *,
                  max_tokens: int = 256,
                  temperature: float = 0.0,
                  top_k: int = 0,
                  top_p: float = 1.0,
                  min_p: float = 0.0,
                  eos_ids: tuple[int, ...] = (248046, 248044),
                  on_token: Callable[[int], None] | None = None,
                  ) -> tuple[str, dict]:
    """Generate tokens autoregressively from pre-computed first_logits.

    Args:
        first_logits: [vocab_size] or [1, vocab_size] from prefill.
        on_token: called with each token id as it's sampled (for streaming).

    Returns (decoded_text, stats_dict).
    """
    t0 = time.time()
    last = np.asarray(first_logits[-1] if first_logits.ndim == 2 else first_logits)
    generated: list[int] = []

    for _ in range(max_tokens):
        tok = sample(last, temperature, top_k, top_p, min_p)
        if tok in eos_ids:
            break
        generated.append(tok)
        if on_token:
            on_token(tok)
        emb = engine.embed_lookup(np.array([tok], dtype=np.int64))
        last = engine.forward_hidden(emb, cache)[0]

    dt = time.time() - t0
    n = len(generated)
    text = tokenizer.decode(generated)
    stats = {"tokens": n, "seconds": dt, "tok_per_s": n / dt if n > 0 else 0}
    return text, stats
