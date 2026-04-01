#!/usr/bin/env python3
"""
figures/make_capacity.py -> figures/out/capacity_psnr.png

Reproduces the style of the paper's capacity-vs-PSNR tables:
sweep the number of embedded bits, plot the resulting PSNR for each
classic-style image.
"""

from __future__ import annotations

import numpy as np

from figures._common import ensure_figure_dir, mpl
from datasets import load_classic_images
from rrwei_sm import RRWEISM
from rrwei_sm.utils import psnr


def _psnr_at_capacity(scheme, cover, n_bits, key_scramble, key_share):
    s1, s2, keys = scheme.encrypt(cover, key_scramble, key_share)
    bits = np.random.default_rng(0).integers(0, 2, size=n_bits, dtype=np.uint8)
    ms1, ms2, side = scheme.embed(s1, s2, bits)
    marked = scheme.decrypt(ms1, ms2, keys)
    return psnr(cover, marked), side.n_embedded


def main():
    plt = mpl()
    images = load_classic_images(size=128)
    # Use n_lsb=2 so each PEE bit shift is smaller -> higher PSNR,
    # easier to sweep many capacities.
    scheme = RRWEISM(n_lsb=2, max_layers=3)
    cap_targets = [64, 128, 256, 512, 1024, 2048, 3072, 4096]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for name, img in images.items():
        xs, ys = [], []
        for n_bits in cap_targets:
            p, actual = _psnr_at_capacity(scheme, img, n_bits, 11, 22)
            xs.append(actual)
            ys.append(p)
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("Embedding capacity (bits)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("RRWEI-SM: PSNR vs embedding capacity  (n_lsb=2)")
    ax.legend(loc="lower left")
    out = ensure_figure_dir() / "capacity_psnr.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
