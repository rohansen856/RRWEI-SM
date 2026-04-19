#!/usr/bin/env python3
"""figures/make_jpeg_ber.py -> figures/out/jpeg_ber.png

Plot 1 - BER of the modern STDM watermark vs. JPEG quality factor.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from figures._common import ensure_figure_dir, mpl
from datasets import load_classic_images
from rrwei_sm.attacks import jpeg_compress
from rrwei_sm.stdm import STDMConfig, embed, extract


def _ber_for_q(marked, bits, side, q):
    attacked = jpeg_compress(marked, quality=int(q))
    return float((extract(attacked, side) != bits).mean())


def main() -> None:
    plt = mpl()
    images = load_classic_images(size=128)
    cfg = STDMConfig(delta=40.0, n_coeffs=4, seed=0)
    rng = np.random.default_rng(0)
    robust_bits = rng.integers(0, 2, size=128, dtype=np.uint8)
    qualities = list(range(10, 101, 5))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for name, img in images.items():
        marked, side = embed(img, robust_bits, cfg)
        bers = [_ber_for_q(marked, robust_bits, side, q) for q in qualities]
        ax.plot(qualities, [1 - b for b in bers], marker="o", label=name)
    ax.set_xlabel("JPEG quality factor")
    ax.set_ylabel(r"$1 - \mathrm{BER}$")
    ax.set_title("Modern STDM: 1 - BER vs. JPEG quality")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()

    out = ensure_figure_dir() / "jpeg_ber.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
