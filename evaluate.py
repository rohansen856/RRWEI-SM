#!/usr/bin/env python3
"""
evaluate.py - run the RRWEI-SM paper's experiments on a set of images.

For each image in the set, measures:

* Encryption-side security      (Sec. V-A):
    - Share/cover correlation (ideal: 0)
    - Horizontal/Vertical/Diagonal adjacent-pixel correlation of share
    - NPCR & UACI when flipping a single cover pixel
* Marked-image visual quality   (Sec. V-B):
    - PSNR(cover, marked)
    - SSIM(cover, marked)
    - Exact recovery flag
* Reversible capacity           (Eq. 25/28):
    - Per-layer and total PEE capacity in bpp
* Modified-scheme robustness    (Sec. V-C):
    - BER curve vs Gaussian sigma (1..40 in steps of 5)
    - BER vs JPEG quality factor (20..95)
    - BER vs JPEG2000 rate (5..40)
    - BER under median/mean/sharpen/S&P filters

Results are printed as a table and (optionally) pickled for later
plotting by the scripts in ``figures/``.

Usage
-----

    python evaluate.py                           # built-in classic images
    python evaluate.py --size 256                # classic images at 256x256
    python evaluate.py --image path/to/file.png  # a single user image
    python evaluate.py --dump-pickle results.pkl # save raw numbers
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from datasets import load_classic_images
from rrwei_sm import ModifiedRRWEISM, RRWEISM
from rrwei_sm.attacks import (
    additive_gaussian_noise,
    jpeg2000_compress,
    jpeg_compress,
    mean_filter,
    median_filter,
    salt_and_pepper_noise,
    sharpen_filter,
)
from rrwei_sm.metrics import (
    correlation_report,
    npcr_report,
    pee_capacity_bpp,
    pixel_correlation,
)
from rrwei_sm.scrambling import block_scramble
from rrwei_sm.utils import ber, psnr, ssim_simple


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).flatten()
    bf = b.astype(np.float64).flatten()
    if af.std() == 0 or bf.std() == 0:
        return 0.0
    return float(np.corrcoef(af, bf)[0, 1])


def _encrypt_fn(
    scheme: RRWEISM, key_scramble: int, key_share: int
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a closure that encrypts an image deterministically (same keys)."""

    def enc(img):
        s1, s2, _ = scheme.encrypt(img, key_scramble, key_share)
        return np.clip(s1.astype(np.int64) + s2.astype(np.int64), 0, 255).astype(
            np.uint8
        )

    return enc


def evaluate_security(
    cover: np.ndarray, scheme: RRWEISM, key_scramble: int, key_share: int
) -> dict:
    """NPCR / UACI / pixel-correlation analysis of the encrypted image."""
    s1, s2, keys = scheme.encrypt(cover, key_scramble, key_share)
    combined = np.clip(s1.astype(np.int64) + s2.astype(np.int64), 0, 255).astype(
        np.uint8
    )
    return {
        "share_cover_corr": _correlation(s1, cover),
        "combined_corr_H": pixel_correlation(combined, "horizontal", 5000, seed=0),
        "combined_corr_V": pixel_correlation(combined, "vertical", 5000, seed=0),
        "combined_corr_D": pixel_correlation(combined, "diagonal", 5000, seed=0),
        "cover_corr": correlation_report(cover, n_pairs=5000, seed=0),
        "npcr_uaci": npcr_report(
            cover, _encrypt_fn(scheme, key_scramble, key_share), n_samples=10, seed=0
        ),
    }


def evaluate_quality_and_capacity(
    cover: np.ndarray, scheme: RRWEISM, key_scramble: int, key_share: int,
    n_bits: int | None = None,
) -> dict:
    """PSNR/SSIM of the marked image + PEE capacity + exact-recovery flag."""
    s1, s2, keys = scheme.encrypt(cover, key_scramble, key_share)
    if n_bits is None:
        n_bits = max(128, cover.size // 64)
    bits = np.random.default_rng(0).integers(0, 2, size=n_bits, dtype=np.uint8)
    ms1, ms2, side = scheme.embed(s1, s2, bits)
    marked_plain = scheme.decrypt(ms1, ms2, keys)
    rec, ext = scheme.extract_after_decrypt(ms1, ms2, keys, side)
    return {
        "n_embedded": int(side.n_embedded),
        "n_requested": int(n_bits),
        "psnr": psnr(cover, marked_plain),
        "ssim": ssim_simple(cover, marked_plain),
        "exact_recovery": bool((rec == cover).all()),
        "capacity_formula": pee_capacity_bpp(cover, n_lsb=scheme.n_lsb, max_layers=scheme.max_layers),
    }


def _sweep(
    cover: np.ndarray,
    scheme: ModifiedRRWEISM,
    robust_bits: np.ndarray,
    key_scramble: int,
    key_share: int,
    attack_fn: Callable[[np.ndarray], np.ndarray],
) -> float:
    s1, s2, keys = scheme.encrypt(cover, key_scramble, key_share)
    ms1, ms2, side = scheme.embed(s1, s2, robust_bits)
    marked_plain = scheme.decrypt(ms1, ms2, keys)
    attacked = attack_fn(marked_plain)
    scrambled = block_scramble(attacked, keys.block_size, keys.key_scramble)
    a1 = scrambled.astype(np.int32)
    a2 = np.zeros_like(a1)
    extracted = scheme.extract_robust_after_decrypt(a1, a2, keys, side)
    return float(ber(extracted, robust_bits))


def evaluate_robustness(
    cover: np.ndarray,
    modified: ModifiedRRWEISM,
    key_scramble: int,
    key_share: int,
    n_robust_bits: int,
) -> dict:
    """Run the full attack sweep and return per-attack BER values."""
    rng = np.random.default_rng(0)
    robust_bits = rng.integers(0, 2, size=n_robust_bits, dtype=np.uint8)
    results = {"n_robust_bits": n_robust_bits}

    # Gaussian sigma sweep (paper Table II: sigma = 5, 10, 15, 20, 25, 30, 35, 40).
    gaussian = {}
    for sigma in (1, 5, 10, 15, 20, 25, 30, 35, 40):
        gaussian[sigma] = _sweep(
            cover,
            modified,
            robust_bits,
            key_scramble,
            key_share,
            lambda img, s=sigma: additive_gaussian_noise(img, sigma=float(s), seed=1),
        )
    results["gaussian_ber"] = gaussian

    # JPEG quality sweep (paper Figs. 14-17 use q in [20..95]).
    jpeg = {}
    for q in (20, 30, 40, 50, 60, 70, 80, 90, 95):
        jpeg[q] = _sweep(
            cover, modified, robust_bits, key_scramble, key_share,
            lambda img, qq=q: jpeg_compress(img, quality=int(qq)),
        )
    results["jpeg_ber"] = jpeg

    # JPEG2000 rate sweep (larger rate = higher compression).
    j2k = {}
    for rate in (5, 10, 15, 20, 25, 30, 40):
        j2k[rate] = _sweep(
            cover, modified, robust_bits, key_scramble, key_share,
            lambda img, r=rate: jpeg2000_compress(img, quality_layers=(float(r),)),
        )
    results["jpeg2000_ber"] = j2k

    # Other filters.
    results["median_3x3_ber"] = _sweep(
        cover, modified, robust_bits, key_scramble, key_share,
        lambda img: median_filter(img, ksize=3)
    )
    results["mean_3x3_ber"] = _sweep(
        cover, modified, robust_bits, key_scramble, key_share,
        lambda img: mean_filter(img, ksize=3)
    )
    results["sharpen_ber"] = _sweep(
        cover, modified, robust_bits, key_scramble, key_share,
        lambda img: sharpen_filter(img, amount=0.5)
    )
    results["salt_pepper_1pct_ber"] = _sweep(
        cover, modified, robust_bits, key_scramble, key_share,
        lambda img: salt_and_pepper_noise(img, p=0.01, seed=11)
    )
    return results


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _load_images(args) -> dict[str, np.ndarray]:
    if args.image:
        from PIL import Image

        arr = np.asarray(Image.open(args.image).convert("L"))
        # Crop to multiple of 4 for 2x2 blocks.
        h = (arr.shape[0] // 4) * 4
        w = (arr.shape[1] // 4) * 4
        return {Path(args.image).stem: arr[:h, :w]}
    return load_classic_images(size=args.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128,
                    help="Size of synthetic classic images (default 128).")
    ap.add_argument("--image", type=str, default=None,
                    help="Use this single image instead of the classic set.")
    ap.add_argument("--n-robust-bits", type=int, default=64)
    ap.add_argument("--dump-pickle", type=str, default=None)
    ap.add_argument("--json", action="store_true",
                    help="Print results as JSON to stdout.")
    args = ap.parse_args()

    images = _load_images(args)
    basic = RRWEISM(n_lsb=3, block_size=2, max_layers=4)
    modified = ModifiedRRWEISM(
        n_lsb=3, block_size=2, patchwork_m=64, patchwork_T=5,
        patchwork_seed=5, max_pee_layers=4, patchwork_plane="hsb",
    )

    all_results: dict[str, dict] = {}
    for name, img in images.items():
        t0 = time.time()
        sec = evaluate_security(img, basic, key_scramble=11, key_share=22)
        qc = evaluate_quality_and_capacity(img, basic, key_scramble=11, key_share=22)
        rob = evaluate_robustness(
            img, modified, key_scramble=11, key_share=22,
            n_robust_bits=args.n_robust_bits,
        )
        elapsed = time.time() - t0
        all_results[name] = {
            "image_shape": list(img.shape),
            "security": sec,
            "quality_and_capacity": qc,
            "robustness": rob,
            "elapsed_sec": elapsed,
        }
        if not args.json:
            print(f"\n==== {name}  ({img.shape})  (elapsed {elapsed:.1f}s) ====")
            print(f"  share/cover correlation    : {sec['share_cover_corr']:+.4f}")
            print(f"  combined corr (H/V/D)      : "
                  f"{sec['combined_corr_H']:+.4f} / "
                  f"{sec['combined_corr_V']:+.4f} / "
                  f"{sec['combined_corr_D']:+.4f}")
            print(f"  NPCR mean / UACI mean      : "
                  f"{sec['npcr_uaci']['NPCR_mean']:.3f}% / "
                  f"{sec['npcr_uaci']['UACI_mean']:.3f}%")
            print(f"  PSNR(cover, marked)        : {qc['psnr']:.2f} dB")
            print(f"  SSIM(cover, marked)        : {qc['ssim']:.4f}")
            print(f"  PEE capacity (bpp)         : {qc['capacity_formula']['total_bpp']:.3f}")
            print(f"  PEE embedded / requested   : {qc['n_embedded']} / {qc['n_requested']}")
            print(f"  Exact recovery             : {qc['exact_recovery']}")
            print(f"  BER @ Gaussian sigma=10    : {rob['gaussian_ber'][10]:.3f}")
            print(f"  BER @ JPEG q=50            : {rob['jpeg_ber'][50]:.3f}")
            print(f"  BER @ JPEG2000 r=20        : {rob['jpeg2000_ber'][20]:.3f}")
            print(f"  BER median/mean/sharpen/SP : "
                  f"{rob['median_3x3_ber']:.3f} / "
                  f"{rob['mean_3x3_ber']:.3f} / "
                  f"{rob['sharpen_ber']:.3f} / "
                  f"{rob['salt_pepper_1pct_ber']:.3f}")

    if args.json:
        print(json.dumps(all_results, indent=2, default=float))
    if args.dump_pickle:
        with open(args.dump_pickle, "wb") as f:
            pickle.dump(all_results, f)
        print(f"\nSaved raw results -> {args.dump_pickle}")


if __name__ == "__main__":
    main()
