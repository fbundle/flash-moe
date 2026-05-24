#!/usr/bin/env python3
"""N-way logit verification: mlx-lm vs Rust pipelines vs C bench on stripped model."""
import subprocess, sys, os, json, struct, tempfile
import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(ROOT, "data", "models--mlx-community--Qwen3.6-35B-A3B-4bit-stripped")
MLX_MODEL_DIR = os.path.join(ROOT, "hub", "models--mlx-community--Qwen3.6-35B-A3B-4bit-stripped")

RUST_ENGINES = ["FusedExpStripped"]
ENGINES = RUST_ENGINES + ["C", "mlx-lm"]

C_DIR = os.path.join(ROOT, "moe_infer_c")
RS_DIR = os.path.join(ROOT, "moe_infer_rs")

TOKENS = [248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 593,
          26003, 248046, 198, 248045, 846, 198, 9419, 11, 1204, 513,
          488, 30, 248046, 198, 248045, 74455, 198, 248068, 198]


def run_rust(mode):
    """Run Rust engine with given pipeline mode, return last-position logits."""
    from moe_infer import Model, Engine, Cache # type: ignore

    t0 = __import__('time').time()

    model = Model(MODEL_DIR)
    engine = Engine(model, pipeline_mode=mode)
    cache = Cache(model)

    ids_arr = np.array(TOKENS, dtype=np.int64)
    logits_all = engine.forward(ids_arr, cache)

    elapsed = __import__('time').time() - t0
    logits = np.array(logits_all[-1], dtype=np.float32)

    tqdm.write(f"[nway] Rust {mode:<12}: {elapsed*1000:5.0f} ms  "
               f"min={logits.min():.4f} max={logits.max():.4f} "
               f"mean={logits.mean():.4f} NaNs={np.isnan(logits).sum()}")
    return logits


def run_c():
    """Run C engine (bench.m compiled with stripped model constants),
    return last-position logits."""
    print("[nway] Compiling C bench-stripped...")
    subprocess.run(["make", "bench-stripped"], cwd=C_DIR,
                   check=True, capture_output=True)

    bench_bin = os.path.join(C_DIR, "bench-stripped")

    # Write tokens to temp binary file (format: count u32 + N * token u32)
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    tf.write(struct.pack("<I", len(TOKENS)))
    for t in TOKENS:
        tf.write(struct.pack("<I", t))
    tokens_path = tf.name
    tf.close()

    # Temp file for logit output
    logits_fd, logits_path = tempfile.mkstemp(suffix=".bin")
    os.close(logits_fd)

    print(f"[nway] Running C bench...")
    t0 = __import__('time').time()

    result = subprocess.run(
        [
            bench_bin,
            "--model", MODEL_DIR,
            "--weights", os.path.join(MODEL_DIR, "model_weights.bin"),
            "--manifest", os.path.join(MODEL_DIR, "model_weights.json"),
            "--prompt-tokens", tokens_path,
            "--verify",
            "--verify-output", logits_path,
            "--k", "4",
        ],
        cwd=C_DIR,
        capture_output=True,
        text=True,
    )

    elapsed = __import__('time').time() - t0

    if result.returncode != 0:
        print(f"[nway] C bench FAILED (exit={result.returncode})")
        print(f"  stdout: {result.stdout[-2000:]}")
        print(f"  stderr: {result.stderr[-2000:]}")
        os.unlink(tokens_path)
        os.unlink(logits_path)
        raise RuntimeError("C bench failed")

    logits = np.fromfile(logits_path, dtype=np.float32)

    tqdm.write(f"[nway] C bench        : {elapsed*1000:5.0f} ms  "
               f"min={logits.min():.4f} max={logits.max():.4f} "
               f"mean={logits.mean():.4f} NaNs={np.isnan(logits).sum()}")

    os.unlink(tokens_path)
    os.unlink(logits_path)
    return logits


def run_mlx():
    """Run MLX-LM on stripped model, return last-position logits."""
    print("[nway] Running MLX-LM...")
    t0 = __import__('time').time()

    from pathlib import Path
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm import tokenizer_utils

    model_path = Path(MLX_MODEL_DIR)
    model, _ = load(str(model_path))
    tokenizer = tokenizer_utils.load(model_path)

    input_ids = mx.array(TOKENS, dtype=mx.int32)[None, :]
    outputs = model(input_ids)
    logits = np.array(mx.array(outputs[0, -1, :]).astype(mx.float32))

    elapsed = __import__('time').time() - t0
    tqdm.write(f"[nway] MLX-LM          : {elapsed*1000:5.0f} ms  "
               f"min={logits.min():.4f} max={logits.max():.4f} "
               f"mean={logits.mean():.4f} NaNs={np.isnan(logits).sum()}")
    return logits


def compare(label1, logits1, label2, logits2, eps=1e-3):
    """Compare two logit arrays."""
    min_len = min(len(logits1), len(logits2))
    a = logits1[:min_len].astype(np.float64)
    b = logits2[:min_len].astype(np.float64)
    diff = np.abs(a - b)
    max_diff = diff.max()
    mean_diff = diff.mean()
    idx_max = diff.argmax()

    # Relative diff: |a-b| / max(|a|, |b|, 1e-8)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-8)
    rel_diff = diff / denom
    max_rel = rel_diff.max()
    idx_rel = rel_diff.argmax()

    # Cosine similarity
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos_sim = float(np.dot(a, b) / max(a_norm * b_norm, 1e-12))

    matching = (diff < eps).sum()
    pct = 100.0 * matching / len(diff)

    print(f"\n  {label1} vs {label2}:")
    print(f"    max_diff={max_diff:.6f} at idx {idx_max} ({label1}={a[idx_max]:.6f}, {label2}={b[idx_max]:.6f})")
    print(f"    mean_diff={mean_diff:.6f}")
    print(f"    max_rel_diff={max_rel:.6f} at idx {idx_rel} ({label1}={a[idx_rel]:.6f}, {label2}={b[idx_rel]:.6f})")
    print(f"    cosine_sim={cos_sim:.8f}")
    print(f"    within {eps}: {matching}/{len(diff)} ({pct:.2f}%)")
    return max_diff


def _lookup_pair(diffs, e1, e2):
    """Symmetric lookup in max_diffs dict."""
    key = f"{e1}_vs_{e2}"
    if key in diffs:
        return diffs[key]
    key = f"{e2}_vs_{e1}"
    if key in diffs:
        return diffs[key]
    return float('nan')


def print_nway_table(max_diffs, engines):
    """Print an N×N summary table: engines on rows, engines on columns.
    Each cell shows the max_diff between the pair."""
    n = len(engines)
    # Build an N×N matrix (symmetric)
    matrix = {}
    for i, e1 in enumerate(engines):
        for j, e2 in enumerate(engines):
            if e1 == e2:
                matrix[(i, j)] = 0.0
            else:
                matrix[(i, j)] = _lookup_pair(max_diffs, e1, e2)

    # Column widths: label col + N data cols
    label_w = max(len(e) for e in engines)
    col_w = max(10, label_w)

    # Header row
    header = " " * (label_w + 2) + "".join(f"{e:>{col_w}}" for e in engines)
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("Pairwise max_diff matrix")
    print("=" * len(header))
    print(header)
    print(sep)

    for i, e1 in enumerate(engines):
        row = f"{e1:>{label_w}} |"
        for j, e2 in enumerate(engines):
            if i == j:
                row += f"{'—':>{col_w}}"
            else:
                val = matrix[(i, j)]
                if np.isnan(val):
                    row += f"{'N/A':>{col_w}}"
                elif val < 1e-5:
                    row += f"{val:>{col_w}.2e}"
                elif val < 1e-3:
                    row += f"{val:>{col_w}.6f}"
                else:
                    row += f"{val:>{col_w}.6f}"
        # Status: worst match against non-mlx engines (mlx uses bf16, not a correctness target)
        worst = 0.0
        for j in range(n):
            if i != j and engines[j] != "mlx-lm":
                worst = max(worst, matrix[(i, j)])
        if worst == 0.0:
            status = " BIT-EXACT"
        elif worst < 1e-5:
            status = " NEAR-MATCH"
        elif worst < 1e-3:
            status = " MATCH"
        elif worst < 0.01:
            status = " CLOSE"
        else:
            status = " DIVERGE"
        row += status
        print(row)

    print(sep)
    print("Legend: ==0 = BIT-EXACT, < 1e-5 = NEAR-MATCH, < 1e-3 = MATCH, < 0.01 = CLOSE, >= 0.01 = DIVERGE")


def main():
    print("=" * 60)
    print("N-Way Verification: mlx-lm vs Rust vs C")
    print(f"Model: stripped  (4 layers, 4 experts)")
    print(f"Tokens: {len(TOKENS)}")
    print("=" * 60)

    # Ensure Rust module is built and installed
    print("[nway] Building Rust module (maturin develop)...")
    subprocess.run(
        [sys.executable, "-m", "maturin", "develop", "--release"],
        cwd=RS_DIR, check=True, capture_output=True,
    )
    print()

    engines = ENGINES

    results = {}
    for mode in tqdm(RUST_ENGINES,
                     desc="Rust pipelines", unit="mode"):
        results[mode] = run_rust(mode)

    results["mlx-lm"] = run_mlx()
    results["C"] = run_c()

    print("\n" + "=" * 60)
    print("Pairwise comparisons")
    print("=" * 60)

    max_diffs = {}
    for i, e1 in enumerate(engines):
        for e2 in engines[i+1:]:
            key = f"{e1}_vs_{e2}"
            max_diffs[key] = compare(e1, results[e1], e2, results[e2])

    print_nway_table(max_diffs, engines)



if __name__ == "__main__":
    main()
