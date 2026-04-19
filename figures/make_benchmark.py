#!/usr/bin/env python3
"""
figures/make_benchmark.py -> figures/out/benchmark_on.png

Reproduces the style of the paper's complexity-analysis figure:
per-pixel wall time for encrypt/decrypt/embed across image sizes,
showing the O(n) scaling.  A successful O(n) implementation yields
roughly-flat per-pixel timings.
"""

from __future__ import annotations

from figures._common import ensure_figure_dir, mpl
from benchmark import benchmark_sizes


def main():
    plt = mpl()
    sizes = [64, 128, 256, 512]
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for scrambler, ax, title in [
        ("random", axs[0], "seeded-random 2x2 scrambler"),
        ("hua", axs[1], "Hua et al. 2D-LSCM scrambler"),
    ]:
        rs = benchmark_sizes(sizes, n_repeats=3, scrambler=scrambler)
        xs = [r["n_pixels"] for r in rs]
        ax.plot(xs, [r["encrypt_ns_per_px"] for r in rs], "o-", label="encrypt")
        ax.plot(xs, [r["decrypt_ns_per_px"] for r in rs], "s-", label="decrypt")
        ax.plot(xs, [r["embed_ns_per_px"] for r in rs], "^-", label="embed")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("pixels")
        ax.set_ylabel("ns / pixel")
        ax.set_title(f"Per-pixel wall time  ({title})")
        ax.legend()
    out = ensure_figure_dir() / "benchmark_on.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
