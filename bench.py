#!/usr/bin/env python3
"""Benchmark C and Rust GPU pipelines on the full model at various prompt lengths."""
import subprocess, sys, os, struct, tempfile, time
import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = "data/models--Qwen--Qwen3.6-35B-A3B-bq4"
C_DIR = "moe_infer_c"
RS_DIR = "."

TOKEN_COUNTS = [20, 50, 100, 200, 500]
from helpers.avail_models import ACTIVE_ENGINES as RUST_MODES


def random_tokens(n: int, seed: int = 42) -> list[int]:
    rng = np.random.RandomState(seed)
    return (rng.randint(4, 50000, size=n)).tolist()


def write_tokens_binary(tokens: list[int], path: str):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(tokens)))
        for t in tokens:
            f.write(struct.pack("<I", t))


def bench_rust(mode: str, tokens: list[int]) -> tuple[float, float, float]:
    from moe_infer import Model, Engine, Cache

    model = Model(MODEL_DIR)
    engine = Engine(model, pipeline_mode=mode)
    cache = Cache(model)
    ids = np.array(tokens, dtype=np.int64)

    t0 = time.perf_counter()
    logits_all = engine.forward(ids, cache)
    elapsed = (time.perf_counter() - t0) * 1000.0

    logits_last = np.array(logits_all[-1], dtype=np.float32)
    return elapsed, float(logits_last.min()), float(logits_last.max())


def bench_c(tokens: list[int]) -> tuple[float, float, float]:
    bench_bin = os.path.join(C_DIR, "bench")

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    tf.close()
    write_tokens_binary(tokens, tf.name)

    logits_fd, logits_path = tempfile.mkstemp(suffix=".bin")
    os.close(logits_fd)

    t0 = time.perf_counter()
    result = subprocess.run(
        [
            bench_bin,
            "--model", MODEL_DIR,
            "--weights", os.path.join(MODEL_DIR, "model_weights.bin"),
            "--manifest", os.path.join(MODEL_DIR, "model_weights.json"),
            "--prompt-tokens", tf.name,
            "--verify",
            "--verify-output", logits_path,
            "--k", "8",
        ],
        cwd=C_DIR,
        capture_output=True,
        text=True,
    )
    elapsed = (time.perf_counter() - t0) * 1000.0

    if result.returncode != 0:
        tqdm.write(f"  C bench FAILED (exit={result.returncode})")
        tqdm.write(f"  stderr: {result.stderr[-1000:]}")
        os.unlink(tf.name)
        os.unlink(logits_path)
        raise RuntimeError("C bench failed")

    logits = np.fromfile(logits_path, dtype=np.float32)
    os.unlink(tf.name)
    os.unlink(logits_path)
    return elapsed, float(logits.min()), float(logits.max())


def main():
    # Build
    tqdm.write("Building C bench (full model)...")
    subprocess.run(["make", "bench"], cwd=C_DIR, check=True, capture_output=True)


    tqdm.write(f"\nModel: {MODEL_DIR}")
    tqdm.write(f"Modes: {', '.join(RUST_MODES)} (Rust) + C\n")

    # Results: dict[mode][n_tokens] -> (elapsed_ms, tok_s)
    results: dict[str, dict[int, tuple[float, float]]] = {}

    all_modes = [f"Rust {m}" for m in RUST_MODES] # + ["C"]
    for n in tqdm(TOKEN_COUNTS, desc="Token counts", unit="len"):
        for label in all_modes:
            tokens = random_tokens(n, seed=int(time.time()))

            if label == "C":
                elapsed, mn, mx = bench_c(tokens)
            else:
                mode = label.replace("Rust ", "")
                elapsed, mn, mx = bench_rust(mode, tokens)

            
            tok_s = n / (elapsed / 1000.0)
            results.setdefault(label, {})[n] = (elapsed, tok_s)
            tqdm.write(f"  {label:<18}: {elapsed:6.0f} ms  {tok_s:6.1f} tok/s  "
                       f"[{mn:.2f}, {mx:.2f}]")

    # ── Table ──
    print("\n" + "=" * 70)
    header = " ".join(f"{n:<4d} tok".rjust(14) for n in TOKEN_COUNTS)
    sub_header = " ".join(f"{'ms':>6} {'tok/s':>7}" for _ in TOKEN_COUNTS)
    print(f"{'Engine':<18} {header}")
    print(f"{'':>18} {sub_header}")
    print("-" * 70)

    for label in all_modes:
        row = f"{label:<18}"
        for n in TOKEN_COUNTS:
            ms, tok_s = results[label][n]
            row += f" {ms:>6.0f} {tok_s:>7.1f}"
        print(row)

    print("-" * 70)

    # Speedup vs C
    print("\nSpeedup vs C:")
    for label in all_modes:
        if label == "C":
            continue
        for n in TOKEN_COUNTS:
            c_ms = results["C"][n][0]
            r_ms = results[label][n][0]
            ratio = c_ms / r_ms
            print(f"  {n:3d} tok: {label:<14} {ratio:.2f}x")


if __name__ == "__main__":
    main()
