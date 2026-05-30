#!/usr/bin/env python3
"""bench_prefill.py — Compare forward (token-serial) vs forward_batched
(layer-batched, see src/engine/qwen35_moe/batched.rs).

Reports wall time and tok/s at each prompt length, plus the speedup.
Numerics are equivalent (max_diff ~2e-5, top-1 match — see verify_nway.py
and verify_vs_original.py).

Usage:
    python helpers/bench_prefill.py [--model PATH] [--ns 4,8,16,32,64,128] [--runs N]
"""

import argparse
import time

import numpy as np
import moe_infer
from moe_infer import Model


def bench(model, method: str, n: int, n_runs: int) -> float:
    e = moe_infer._core._rs.Engine(model._inner, "Qwen35MoEFusedExp2", 0)
    # Warmup — first call always slower (shader compile + GPU warm-up).
    warmup = np.array(np.random.RandomState(1).randint(4, 50000, size=4), dtype=np.int64)
    getattr(e, method)(warmup, moe_infer._core._rs.Cache(model._inner))
    times = []
    rng = np.random.RandomState(0)
    for r in range(n_runs):
        c = moe_infer._core._rs.Cache(model._inner)
        toks = np.array(rng.randint(4, 50000, size=n), dtype=np.int64)
        t0 = time.perf_counter()
        getattr(e, method)(toks, c)
        times.append(time.perf_counter() - t0)
    return min(times) * 1000.0  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/Qwen3.6-35B-A3B/model_bq4")
    ap.add_argument("--ns", default="4,8,16,32,64,128",
                    help="Comma-separated prompt lengths")
    ap.add_argument("--runs", type=int, default=3,
                    help="Measured runs per (method, n); reports min")
    args = ap.parse_args()

    ns = [int(x) for x in args.ns.split(",")]
    print(f"Loading {args.model}...")
    m = Model(args.model)
    print()
    print(f"{'N':>6s} {'seq ms':>10s} {'seq tok/s':>10s} {'bat ms':>10s} {'bat tok/s':>10s} {'speedup':>8s}")
    print("-" * 62)
    for n in ns:
        t_seq = bench(m, "forward", n, args.runs)
        t_bat = bench(m, "forward_batched", n, args.runs)
        print(f"{n:>6d} {t_seq:>10.0f} {n*1000/t_seq:>10.2f} "
              f"{t_bat:>10.0f} {n*1000/t_bat:>10.2f} {t_seq/t_bat:>7.2f}x")


if __name__ == "__main__":
    main()
