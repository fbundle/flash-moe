#!/usr/bin/env python3
"""Benchmark Fused4bit engine stages using built-in telemetry with mean/std across runs."""
import subprocess, sys, os, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "data", "models--mlx-community--Qwen3.6-35B-A3B-4bit")
RS_DIR = os.path.join(ROOT, "moe_infer_rs")

TOKEN_COUNTS = [20, 50, 100]
WARMUP_TOKENS = 32
N_RUNS = 5


def random_tokens(n: int, seed: int = 42) -> list[int]:
    rng = np.random.RandomState(seed)
    return (rng.randint(4, 50000, size=n)).tolist()


def main():
    # Build
    print("Building Rust module...")
    subprocess.run(
        [sys.executable, "-m", "maturin", "develop", "--release"],
        cwd=RS_DIR, check=True, capture_output=True,
    )

    import moe_infer
    from moe_infer import Model, Engine, Cache # type: ignore

    moe_infer.record_engine_telemetry(True) # type: ignore

    print(f"\nModel: {MODEL_DIR}")
    print(f"Mode: Fused4bit | Runs per config: {N_RUNS}\n")

    # Warmup
    print(f"Warmup ({WARMUP_TOKENS} tokens)...")
    model = Model(MODEL_DIR)
    engine = Engine(model, pipeline_mode="Fused4bit")
    warm = random_tokens(WARMUP_TOKENS, seed=1)
    warm_ids = np.array(warm, dtype=np.int64)
    warm_cache = Cache(model)
    engine.forward(warm_ids, warm_cache)

    for n in TOKEN_COUNTS:
        print(f"\n{'='*70}")
        print(f"Token count: {n}")
        print(f"{'='*70}")

        # Collect per-run telemetry for each stage
        all_telem: dict[str, list[float]] = {}
        wall_times: list[float] = []

        for run in range(N_RUNS):
            tokens = random_tokens(n, seed=100 + run)
            ids = np.array(tokens, dtype=np.int64)
            cache = Cache(model)

            t0 = time.perf_counter()
            engine.forward(ids, cache)
            wall = (time.perf_counter() - t0) * 1000.0
            wall_times.append(wall)

            telem = engine.telemetry()
            for key, val in telem.items():
                if isinstance(val, list):
                    # Per-token timing: aggregate as mean across tokens
                    agg = float(np.mean(val))
                else:
                    agg = float(val)
                all_telem.setdefault(key, []).append(agg)

        # ── Report ──
        mean_wall = np.mean(wall_times)
        std_wall = np.std(wall_times)
        print(f"\n{'Stage':<40} {'Mean (ms)':>10} {'Std (ms)':>10}")
        print("-" * 62)
        print(f"{'Wall time':<40} {mean_wall:>10.3f} {std_wall:>10.3f}")

        for key in sorted(all_telem.keys()):
            vals = all_telem[key]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            print(f"{key:<40} {mean_v:>10.3f} {std_v:>10.3f}")


if __name__ == "__main__":
    main()
