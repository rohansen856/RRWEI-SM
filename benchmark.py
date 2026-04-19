#!/usr/bin/env python3
"""Compatibility shim around the modern benchmark entry point.

The Step-8 benchmark implementation lives in :mod:`benchmark_modern`.
This file exists only so older invocations (``python benchmark.py ...``)
and the figure scripts that rely on a ``benchmark_sizes`` helper keep
working against the modern pipeline.
"""

from __future__ import annotations

import argparse

from benchmark_modern import benchmark, print_table


def benchmark_sizes(sizes: list[int], n_repeats: int = 3, **_ignored) -> list[dict]:
    """Compatibility alias for the old API.  Extra kwargs are ignored."""
    return benchmark(sizes, n_repeats=n_repeats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--n-repeats", type=int, default=3)
    args = ap.parse_args()
    results = benchmark_sizes(args.sizes, n_repeats=args.n_repeats)
    print_table(results)


if __name__ == "__main__":
    main()
