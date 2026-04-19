#!/usr/bin/env python3
"""Benchmark the modern RRWEI-SM pipeline.

Compares per-pixel wall time of encrypt / decrypt / embed (STDM) /
embed (PVO) across a geometric series of image sizes.  A flat
per-pixel time confirms the O(n) claim; the absolute time is the
interesting number for the 3x speedup criterion of Step 8.

Usage::

    python benchmark_modern.py
    python benchmark_modern.py --sizes 64 128 256 512 --n-repeats 2
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from rrwei_sm.crypto_scrambler import block_scramble, block_unscramble
from rrwei_sm.orchestrator import ModernScheme
from rrwei_sm.pvo import capacity_estimate, embed as pvo_embed
from rrwei_sm.secret_sharing import (
    additive_combine_shares,
    additive_share_image,
)
from rrwei_sm.stdm import STDMConfig, embed as stdm_embed


def _make_cover(size: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(30, 220, size=(size, size), dtype=np.uint8)


def _timeit(fn, n_repeats: int) -> float:
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _pvo_embed_python(image: np.ndarray, bits: np.ndarray):
    """Pure-Python PVO reference implementation (Step-7 state) for the
    Numba speedup comparison.  Mirrors the 2-pass algorithm in pvo.py
    but without JIT.
    """
    if image.ndim != 2:
        raise ValueError
    h, w = image.shape
    blocks = image.astype(np.int64).reshape(h // 2, 2, w // 2, 2).swapaxes(1, 2).reshape(-1, 4).copy()
    n_blocks = blocks.shape[0]
    bit_idx = 0

    # Max pass.
    for i in range(n_blocks):
        order = np.argsort(blocks[i], kind="stable")
        argmax = int(order[-1])
        sorted_vals = blocks[i][order]
        u = int(sorted_vals[-1] - sorted_vals[-2])
        if u == 0 or blocks[i, argmax] >= 255 or bit_idx >= bits.size:
            continue
        if u == 1:
            new_val = blocks[i, argmax] + int(bits[bit_idx])
            if new_val > 255:
                continue
            blocks[i, argmax] = new_val
            bit_idx += 1
        else:
            new_val = blocks[i, argmax] + 1
            if new_val > 255:
                continue
            blocks[i, argmax] = new_val

    # Min pass.
    for i in range(n_blocks):
        order = np.argsort(blocks[i], kind="stable")
        argmin = int(order[0])
        sorted_vals = blocks[i][order]
        v = int(sorted_vals[1] - sorted_vals[0])
        if v == 0 or blocks[i, argmin] <= 0 or bit_idx >= bits.size:
            continue
        if v == 1:
            new_val = blocks[i, argmin] - int(bits[bit_idx])
            if new_val < 0:
                continue
            blocks[i, argmin] = new_val
            bit_idx += 1
        else:
            new_val = blocks[i, argmin] - 1
            if new_val < 0:
                continue
            blocks[i, argmin] = new_val

    marked = blocks.reshape(h // 2, w // 2, 2, 2).swapaxes(1, 2).reshape(h, w)
    return np.clip(marked, 0, 255).astype(np.uint8)


def benchmark(sizes: list[int], n_repeats: int) -> list[dict]:
    results = []
    for sz in sizes:
        cover = _make_cover(sz)
        n = cover.size
        # Warm up PVO JIT.
        _ = pvo_embed(cover, np.zeros(8, dtype=np.uint8))

        t_scramble = _timeit(
            lambda: block_scramble(cover, block_size=2, seed=1), n_repeats
        )
        scrambled = block_scramble(cover, block_size=2, seed=1)

        t_share = _timeit(
            lambda: additive_share_image(scrambled, n_parties=3, seed=2),
            n_repeats,
        )
        sharing = additive_share_image(scrambled, n_parties=3, seed=2)

        t_combine = _timeit(
            lambda: additive_combine_shares(sharing.shares), n_repeats
        )

        t_unscramble = _timeit(
            lambda: block_unscramble(cover, block_size=2, seed=1), n_repeats
        )

        # STDM on 8x8 blocks: at most sz//8 ** 2 bits.
        bits_stdm = np.random.default_rng(0).integers(
            0, 2, size=(sz // 8) * (sz // 8), dtype=np.uint8
        )
        cfg = STDMConfig(delta=40.0, n_coeffs=4, seed=1)
        t_stdm = _timeit(lambda: stdm_embed(cover, bits_stdm, cfg), n_repeats)

        # PVO on 2x2 blocks.
        cap = capacity_estimate(cover)
        n_pvo_bits = cap["max_side"] + cap["min_side"]
        if n_pvo_bits > 0:
            bits_pvo = np.random.default_rng(1).integers(
                0, 2, size=n_pvo_bits, dtype=np.uint8
            )
            t_pvo = _timeit(lambda: pvo_embed(cover, bits_pvo), n_repeats)
            t_pvo_py = _timeit(
                lambda: _pvo_embed_python(cover, bits_pvo), n_repeats
            )
            pvo_speedup = t_pvo_py / max(t_pvo, 1e-9)
        else:
            t_pvo = 0.0
            t_pvo_py = 0.0
            pvo_speedup = 0.0

        results.append(
            {
                "size": sz,
                "n_pixels": n,
                "scramble_sec": t_scramble,
                "unscramble_sec": t_unscramble,
                "share_sec": t_share,
                "combine_sec": t_combine,
                "stdm_sec": t_stdm,
                "pvo_sec": t_pvo,
                "pvo_py_sec": t_pvo_py,
                "pvo_speedup": pvo_speedup,
                "scramble_ns_per_px": t_scramble / n * 1e9,
                "share_ns_per_px": t_share / n * 1e9,
                "stdm_ns_per_px": t_stdm / n * 1e9,
                "pvo_ns_per_px": t_pvo / n * 1e9,
            }
        )
    return results


def print_table(results: list[dict]) -> None:
    print(
        f"\n{'size':>6}  {'n_px':>9}  "
        f"{'scr (ms)':>10}  {'share (ms)':>11}  "
        f"{'stdm (ms)':>11}  {'pvo jit (ms)':>13}  {'pvo py (ms)':>12}  "
        f"{'pvo speedup':>12}  {'pvo ns/px (jit)':>17}"
    )
    for r in results:
        print(
            f"{r['size']:>6}  {r['n_pixels']:>9}  "
            f"{r['scramble_sec']*1000:>10.2f}  "
            f"{r['share_sec']*1000:>11.2f}  "
            f"{r['stdm_sec']*1000:>11.2f}  "
            f"{r['pvo_sec']*1000:>13.2f}  "
            f"{r['pvo_py_sec']*1000:>12.2f}  "
            f"{r['pvo_speedup']:>12.1f}x  "
            f"{r['pvo_ns_per_px']:>16.1f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sizes", type=int, nargs="+", default=[64, 128, 256, 512],
    )
    ap.add_argument("--n-repeats", type=int, default=3)
    args = ap.parse_args()
    results = benchmark(args.sizes, n_repeats=args.n_repeats)
    print_table(results)


if __name__ == "__main__":
    main()
