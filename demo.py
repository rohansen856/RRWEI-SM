#!/usr/bin/env python3
"""
End-to-end demo for the RRWEI-SM paper (Xiong et al., 2022).

Runs both schemes (basic RRWEI-SM + Modified RRWEI-SM) on a small
synthetic image and prints the key metrics reported in the paper:
  * PSNR between cover and marked image
  * Bit Error Rate of the extracted watermark
  * Exact-recovery check (cover_recovered == cover_original?)
  * Simple encryption security indicator (correlation of a share with
    the cover; ideal ~ 0).

Usage
-----
    python demo.py                       # use the synthetic cover
    python demo.py path/to/grayscale.png # use your own image
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from rrwei_sm import ModifiedRRWEISM, RRWEISM
from rrwei_sm.utils import ber, psnr, ssim_simple


def _load_or_make_cover(argv: list[str]) -> np.ndarray:
    if len(argv) > 1:
        from PIL import Image
        path = Path(argv[1])
        img = np.asarray(Image.open(path).convert("L"))
        # Crop to a multiple of 2 for 2x2-block operations.
        h = (img.shape[0] // 4) * 4
        w = (img.shape[1] // 4) * 4
        return img[:h, :w]
    xs = np.linspace(30, 220, 128)
    ys = np.linspace(20, 200, 128)[:, None]
    return (0.6 * xs + 0.4 * ys).clip(0, 255).astype(np.uint8)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).flatten()
    bf = b.astype(np.float64).flatten()
    if af.std() == 0 or bf.std() == 0:
        return 0.0
    return float(np.corrcoef(af, bf)[0, 1])


def demo_basic(cover: np.ndarray, n_bits: int = 120) -> None:
    print("\n==== Basic RRWEI-SM ====")
    scheme = RRWEISM(n_lsb=3, max_layers=4)
    s1, s2, keys = scheme.encrypt(cover, key_scramble=7, key_share=13)
    bits = np.random.default_rng(0).integers(0, 2, size=n_bits, dtype=np.uint8)
    ms1, ms2, side = scheme.embed(s1, s2, bits)

    marked_plain = scheme.decrypt(ms1, ms2, keys)
    rec_cover_enc, rec2, ext_bits_enc = scheme.extract_before_decrypt(ms1, ms2, side)
    rec_cover_dec, ext_bits_dec = scheme.extract_after_decrypt(ms1, ms2, keys, side)

    print(f"  cover shape          : {cover.shape}")
    print(f"  n_embedded / n_bits  : {side.n_embedded} / {n_bits}")
    print(f"  PSNR(cover, marked)  : {psnr(cover, marked_plain):.2f} dB")
    print(f"  SSIM(cover, marked)  : {ssim_simple(cover, marked_plain):.4f}")
    print(f"  BER (extract-first)  : {ber(ext_bits_enc, bits[:side.n_embedded]):.4f}")
    print(f"  BER (decrypt-first)  : {ber(ext_bits_dec, bits[:side.n_embedded]):.4f}")

    rec_plain = scheme.decrypt(rec_cover_enc, rec2, keys)
    exact_enc = bool((rec_plain == cover).all())
    exact_dec = bool((rec_cover_dec == cover).all())
    print(f"  Exact recovery (E)   : {exact_enc}")
    print(f"  Exact recovery (D)   : {exact_dec}")

    # Security: a single share should look uncorrelated with cover.
    print(f"  Corr(share1, cover)  : {_correlation(s1, cover):+.3f} (ideal: 0)")


def demo_modified(cover: np.ndarray, n_robust_bits: int = 8) -> None:
    print("\n==== Modified RRWEI-SM (two-stage) ====")
    scheme = ModifiedRRWEISM(
        n_lsb=3,
        patchwork_m=64,
        patchwork_T=5,
        patchwork_seed=3,
        max_pee_layers=4,
    )
    s1, s2, keys = scheme.encrypt(cover, key_scramble=1, key_share=2)
    robust = np.random.default_rng(1).integers(0, 2, size=n_robust_bits, dtype=np.uint8)
    ms1, ms2, side = scheme.embed(s1, s2, robust)

    marked = scheme.decrypt(ms1, ms2, keys)
    quick_bits = scheme.extract_robust_after_decrypt(ms1, ms2, keys, side)
    rec, full_bits = scheme.extract_and_recover_after_decrypt(ms1, ms2, keys, side)

    print(f"  cover shape              : {cover.shape}")
    print(f"  robust bits              : {n_robust_bits}")
    print(f"  PSNR(cover, marked)      : {psnr(cover, marked):.2f} dB")
    print(f"  SSIM(cover, marked)      : {ssim_simple(cover, marked):.4f}")
    print(f"  BER (quick robust)       : {ber(quick_bits, robust):.4f}")
    print(f"  BER (full after recover) : {ber(full_bits, robust):.4f}")
    print(f"  Exact recovery           : {bool((rec == cover).all())}")

    # Mini noise-attack demo
    rng = np.random.default_rng(2)
    attacked = np.clip(marked.astype(np.int64) + rng.integers(-3, 4, marked.shape), 0, 255).astype(np.uint8)
    # Re-share the attacked image so the extractor API works.
    ma_s1 = attacked.astype(np.int32)
    ma_s2 = np.zeros_like(ma_s1)
    # The scheme's quick path recombines shares, so re-scramble the attack
    # back into scrambled space so the share reconstruction matches.
    # For a simple noise attack, we reuse the decrypt helpers on the
    # untouched shares but replace with noisy ones *via their sum*:
    from rrwei_sm.scrambling import block_scramble
    noisy_scrambled = block_scramble(attacked, keys.block_size, keys.key_scramble)
    ma_s1 = noisy_scrambled.astype(np.int32)
    ma_s2 = np.zeros_like(ma_s1)
    attacked_bits = scheme.extract_robust_after_decrypt(ma_s1, ma_s2, keys, side)
    print(f"  BER under mild noise     : {ber(attacked_bits, robust):.4f}")


def main():
    cover = _load_or_make_cover(sys.argv)
    print(f"Cover image min/max: {cover.min()}/{cover.max()}, shape: {cover.shape}")
    demo_basic(cover)
    demo_modified(cover)
    print("\nDone.")


if __name__ == "__main__":
    main()
