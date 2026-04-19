#!/usr/bin/env python3
"""figures/make_wgn_ber.py -> figures/out/wgn_ber.png

Plot 1 - BER of the modern STDM robust watermark vs. Gaussian noise
sigma on the three classic-style synthetic covers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from figures._common import ensure_figure_dir, mpl
from datasets import load_classic_images
from rrwei_sm.attacks import gaussian_noise
from rrwei_sm.stdm import STDMConfig, embed, extract


def _ber_for_sigma(marked, bits, side, sigma):
    attacked = gaussian_noise(marked, sigma=float(sigma), seed=1)
    return float((extract(attacked, side) != bits).mean())


def main() -> None:
    plt = mpl()
    images = load_classic_images(size=128)
    cfg = STDMConfig(delta=40.0, n_coeffs=4, seed=0)
    rng = np.random.default_rng(0)
    robust_bits = rng.integers(0, 2, size=128, dtype=np.uint8)
    sigmas = np.arange(1, 41, 2)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for name, img in images.items():
        marked, side = embed(img, robust_bits, cfg)
        bers = [_ber_for_sigma(marked, robust_bits, side, s) for s in sigmas]
        ax.plot(sigmas, [1 - b for b in bers], marker="o", label=name)
    ax.set_xlabel(r"Gaussian noise $\sigma$")
    ax.set_ylabel(r"$1 - \mathrm{BER}$")
    ax.set_title("Modern STDM: 1 - BER vs. Gaussian noise sigma")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()

    out = ensure_figure_dir() / "wgn_ber.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
