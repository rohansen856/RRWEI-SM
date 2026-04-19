#!/usr/bin/env python3
"""figures/make_benchmark.py -> figures/out/benchmark_on.png

Per-pixel wall time for scramble / share / STDM / PVO vs image size.
Flat lines confirm the O(n) claim from the paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figures._common import ensure_figure_dir, mpl

from benchmark_modern import benchmark


def main() -> None:
    plt = mpl()
    sizes = [64, 128, 256, 512]
    rs = benchmark(sizes, n_repeats=2)

    xs = [r["n_pixels"] for r in rs]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(xs, [r["scramble_ns_per_px"] for r in rs], "o-", label="scramble")
    ax.plot(xs, [r["share_ns_per_px"] for r in rs], "s-", label="share")
    ax.plot(xs, [r["stdm_ns_per_px"] for r in rs], "^-", label="STDM embed")
    ax.plot(xs, [r["pvo_ns_per_px"] for r in rs], "d-", label="PVO embed (Numba)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n (pixels)")
    ax.set_ylabel("ns / pixel")
    ax.set_title("Per-pixel wall time (flat = O(n))")
    ax.legend()
    fig.tight_layout()

    out = ensure_figure_dir() / "benchmark_on.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
