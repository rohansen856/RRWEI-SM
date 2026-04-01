#!/usr/bin/env python3
"""
benchmark.py - time encryption, decryption and embedding for the paper's
O(n) complexity claim (Section V-E).

The paper states: "Additive secret sharing and block-level scrambling
are developed to encrypt the image, and thus, the complexity of the
encryption and decryption is O(n)."  This script runs encrypt/decrypt/
embed on a geometric series of image sizes and reports per-pixel time
as a function of n.  An approximately flat per-pixel time confirms the
O(n) behaviour; the raw times can also be saved for plotting.

Usage
-----

    python benchmark.py                       # default sizes
    python benchmark.py --sizes 64 128 256 512 1024
    python benchmark.py --dump-pickle bench.pkl
"""

from __future__ import annotations

import argparse
import pickle
import time

import numpy as np

from rrwei_sm import ModifiedRRWEISM, RRWEISM


def _make_cover(size: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(30, 220, size=(size, size), dtype=np.uint8)


def _timeit(fn, n_repeats: int) -> float:
    """Return median wall time in seconds."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def benchmark_sizes(
    sizes: list[int],
    n_repeats: int = 3,
    scrambler: str = "random",
) -> list[dict]:
    basic = RRWEISM(n_lsb=3, max_layers=2, scrambler=scrambler)
    results: list[dict] = []
    for sz in sizes:
        cover = _make_cover(sz)
        n = cover.size
        # Encryption: scramble + additive share.
        t_enc = _timeit(
            lambda: basic.encrypt(cover, key_scramble=1, key_share=2), n_repeats
        )
        s1, s2, keys = basic.encrypt(cover, key_scramble=1, key_share=2)
        # Decryption: combine + unscramble.
        t_dec = _timeit(lambda: basic.decrypt(s1, s2, keys), n_repeats)
        # Embedding: PEE on combined image.
        bits = np.random.default_rng(0).integers(
            0, 2, size=min(n // 64, 1024), dtype=np.uint8
        )
        t_emb = _timeit(lambda: basic.embed(s1, s2, bits), n_repeats)
        results.append(
            {
                "size": sz,
                "n_pixels": n,
                "encrypt_sec": t_enc,
                "decrypt_sec": t_dec,
                "embed_sec": t_emb,
                "encrypt_ns_per_px": t_enc / n * 1e9,
                "decrypt_ns_per_px": t_dec / n * 1e9,
                "embed_ns_per_px": t_emb / n * 1e9,
            }
        )
    return results


def print_table(results: list[dict], scrambler: str) -> None:
    print(f"\n==== Benchmark (scrambler='{scrambler}') ====")
    print(
        f"{'size':>6}  {'n_px':>8}  "
        f"{'enc (ms)':>10}  {'dec (ms)':>10}  {'emb (ms)':>10}  "
        f"{'enc ns/px':>10}  {'dec ns/px':>10}  {'emb ns/px':>10}"
    )
    for r in results:
        print(
            f"{r['size']:>6}  {r['n_pixels']:>8}  "
            f"{r['encrypt_sec'] * 1000:>10.2f}  "
            f"{r['decrypt_sec'] * 1000:>10.2f}  "
            f"{r['embed_sec'] * 1000:>10.2f}  "
            f"{r['encrypt_ns_per_px']:>10.1f}  "
            f"{r['decrypt_ns_per_px']:>10.1f}  "
            f"{r['embed_ns_per_px']:>10.1f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024],
        help="Image side lengths to benchmark (must be even).",
    )
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument(
        "--scramblers", type=str, nargs="+", default=["random", "hua"],
        help="Which scrambling methods to benchmark.",
    )
    ap.add_argument("--dump-pickle", type=str, default=None)
    args = ap.parse_args()

    all_results: dict[str, list[dict]] = {}
    for s in args.scramblers:
        rs = benchmark_sizes(args.sizes, n_repeats=args.n_repeats, scrambler=s)
        print_table(rs, s)
        all_results[s] = rs

    if args.dump_pickle:
        with open(args.dump_pickle, "wb") as f:
            pickle.dump(all_results, f)
        print(f"\nSaved benchmarks -> {args.dump_pickle}")


if __name__ == "__main__":
    main()
