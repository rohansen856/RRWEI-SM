#!/usr/bin/env python3
"""figures/make_capacity.py -> figures/out/capacity_psnr.png

PSNR vs. embedded bits (cumulative over PVO layers) for each cover.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from figures._common import ensure_figure_dir, mpl
from datasets import load_classic_images
from rrwei_sm.metrics import psnr
from rrwei_sm.pvo import capacity_estimate, embed


def _sweep(cover, max_layers: int = 6) -> list[tuple[float, float]]:
    """Return a list of (bits, psnr) for layer 0 .. max_layers."""
    rng = np.random.default_rng(0)
    current = cover.copy()
    total_bits = 0
    out = [(0, psnr(cover, cover))]
    for _ in range(max_layers):
        cap = capacity_estimate(current)
        room = cap["max_side"] + cap["min_side"]
        if room == 0:
            break
        bits = rng.integers(0, 2, size=room, dtype=np.uint8)
        current, _, n_emb = embed(current, bits)
        total_bits += n_emb
        out.append((total_bits, psnr(cover, current)))
    return out


def main() -> None:
    plt = mpl()
    images = load_classic_images(size=128)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for name, img in images.items():
        series = _sweep(img)
        xs = [s[0] / img.size for s in series]  # bpp
        ys = [s[1] for s in series]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("Embedded capacity (bits / pixel)")
    ax.set_ylabel("PSNR(cover, marked) (dB)")
    ax.set_title("PVO capacity vs. PSNR (modern pipeline)")
    ax.legend()
    fig.tight_layout()

    out = ensure_figure_dir() / "capacity_psnr.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
