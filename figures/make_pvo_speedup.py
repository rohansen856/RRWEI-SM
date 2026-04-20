#!/usr/bin/env python3
"""figures/make_pvo_speedup.py -> figures/out/pvo_speedup.png

Bar chart of Numba-JIT vs pure-Python PVO embed throughput at four
image sizes.  The companion number printed in the log is the speedup
ratio ``pvo_py_sec / pvo_jit_sec``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmark_modern import benchmark
from figures._common import ensure_figure_dir, mpl


def main() -> None:
    plt = mpl()
    sizes = [64, 128, 256, 512]
    rs = benchmark(sizes, n_repeats=1)

    labels = [f"{s}x{s}\n({s*s} px)" for s in sizes]
    jit = [r["pvo_ns_per_px"] for r in rs]
    py = [r["pvo_py_sec"] / r["n_pixels"] * 1e9 for r in rs]
    speedups = [r["pvo_speedup"] for r in rs]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(len(sizes))
    w = 0.38
    ax.bar(x - w / 2, py, w, label="pure Python (reference)", color="#c0504d")
    ax.bar(x + w / 2, jit, w, label="Numba JIT (shipping)", color="#4f81bd")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ns / pixel  (log scale)")
    ax.set_title("PVO embed throughput: Numba JIT vs pure Python")
    for i, s in enumerate(speedups):
        ax.text(
            x[i] + w / 2, jit[i] * 1.25, f"{s:.0f}x",
            ha="center", fontsize=10, color="#2a4d75",
        )
    ax.legend()
    fig.tight_layout()

    out = ensure_figure_dir() / "pvo_speedup.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")
    for sz, s in zip(sizes, speedups):
        print(f"  {sz}x{sz}: speedup = {s:.1f}x")


if __name__ == "__main__":
    main()
