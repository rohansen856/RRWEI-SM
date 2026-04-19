#!/usr/bin/env python3
"""
figures/make_wgn_ber.py  ->  figures/out/wgn_ber.png

Reproduces the style of Table II (WGN sigma sweep): 1-BER vs sigma for
the Modified RRWEI-SM on each classic-style image.
"""

from __future__ import annotations

import numpy as np

from figures._common import ensure_figure_dir, mpl
from datasets import load_classic_images
from rrwei_sm import ModifiedRRWEISM
from rrwei_sm.attacks import additive_gaussian_noise
from rrwei_sm.scrambling import block_scramble
from rrwei_sm.utils import ber


def _ber_for_sigma(scheme, cover, robust_bits, sigma, key_scramble, key_share):
    s1, s2, keys = scheme.encrypt(cover, key_scramble, key_share)
    ms1, ms2, side = scheme.embed(s1, s2, robust_bits)
    marked = scheme.decrypt(ms1, ms2, keys)
    atk = additive_gaussian_noise(marked, sigma=float(sigma), seed=1)
    scrambled = block_scramble(atk, keys.block_size, keys.key_scramble)
    a1, a2 = scrambled.astype(np.int32), np.zeros_like(scrambled, dtype=np.int32)
    ext = scheme.extract_robust_after_decrypt(a1, a2, keys, side)
    return float(ber(ext, robust_bits))


def main():
    plt = mpl()
    images = load_classic_images(size=128)
    scheme = ModifiedRRWEISM(
        n_lsb=3, patchwork_m=64, patchwork_T=5, patchwork_seed=5,
        max_pee_layers=4, patchwork_plane="hsb",
    )
    sigmas = np.arange(1, 41, 2)  # 1 to 40 in steps of 2
    robust_bits = np.random.default_rng(0).integers(0, 2, size=64, dtype=np.uint8)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for name, img in images.items():
        bers = [
            _ber_for_sigma(scheme, img, robust_bits, s, 11, 22) for s in sigmas
        ]
        one_minus_ber = [1 - b for b in bers]
        ax.plot(sigmas, one_minus_ber, marker="o", label=name)
    ax.set_xlabel(r"Gaussian noise $\sigma$")
    ax.set_ylabel(r"$1 - \mathrm{BER}$")
    ax.set_title("Modified RRWEI-SM: robustness vs white Gaussian noise")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    out = ensure_figure_dir() / "wgn_ber.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
