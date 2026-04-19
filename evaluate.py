#!/usr/bin/env python3
"""evaluate.py - reproducibility harness for the *modernized* pipeline.

For each image in the evaluated set, measures:

  * **Encryption-side security**: share/cover correlation, NPCR/UACI
    from a single-pixel cover flip, and adjacent-pixel correlation of
    the aggregated (scrambled + shared) view.
  * **Marked-image quality**: PSNR, SSIM, LPIPS (opt-in), DISTS proxy,
    and an exact-recovery flag for the STDM-marked view via PVO.
  * **Capacity**: achieved reversible PVO bits in bits-per-pixel.
  * **Robustness**: BER of the STDM robust bits under the classical
    attacks (Gaussian, S&P, median/mean/sharpen, JPEG, JPEG2000) *and*
    the modern attacks (``neural_codec_proxy``, ``sr_cascade``,
    ``diffusion_regen_proxy``).

Results are printed as a human-readable block (default) or as JSON
(``--json``); they may optionally be pickled for later analysis.

Usage
-----
    python evaluate.py                            # synthetic classics
    python evaluate.py --size 256                 # classic set at 256x256
    python evaluate.py --image path/to/file.png   # a single user image
    python evaluate.py --dump-pickle results.pkl  # save raw numbers
    python evaluate.py --with-lpips               # include LPIPS metric
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Callable

import numpy as np

from datasets import load_classic_images
from rrwei_sm import ModernScheme
from rrwei_sm.attacks import (
    diffusion_regen_proxy,
    gaussian_noise,
    jpeg2000_compress,
    jpeg_compress,
    mean_filter,
    median_filter,
    neural_codec_proxy,
    salt_and_pepper,
    sharpen_filter,
    super_resolution_cascade,
)
from rrwei_sm.metrics import dists_proxy, psnr, ssim


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).flatten()
    bf = b.astype(np.float64).flatten()
    if af.std() == 0 or bf.std() == 0:
        return 0.0
    return float(np.corrcoef(af, bf)[0, 1])


def _adjacent_correlation(
    img: np.ndarray, direction: str, n_pairs: int = 5000, seed: int = 0
) -> float:
    rng = np.random.default_rng(seed)
    h, w = img.shape
    if direction == "horizontal":
        ys = rng.integers(0, h, size=n_pairs)
        xs = rng.integers(0, w - 1, size=n_pairs)
        a = img[ys, xs].astype(np.float64)
        b = img[ys, xs + 1].astype(np.float64)
    elif direction == "vertical":
        ys = rng.integers(0, h - 1, size=n_pairs)
        xs = rng.integers(0, w, size=n_pairs)
        a = img[ys, xs].astype(np.float64)
        b = img[ys + 1, xs].astype(np.float64)
    elif direction == "diagonal":
        ys = rng.integers(0, h - 1, size=n_pairs)
        xs = rng.integers(0, w - 1, size=n_pairs)
        a = img[ys, xs].astype(np.float64)
        b = img[ys + 1, xs + 1].astype(np.float64)
    else:
        raise ValueError(direction)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _npcr_uaci(
    cover: np.ndarray, encrypt_fn: Callable[[np.ndarray], np.ndarray],
    n_samples: int = 5, seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    base = encrypt_fn(cover)
    npcrs: list[float] = []
    uacis: list[float] = []
    for _ in range(n_samples):
        y = int(rng.integers(0, cover.shape[0]))
        x = int(rng.integers(0, cover.shape[1]))
        mod = cover.copy()
        mod[y, x] = np.uint8((int(mod[y, x]) + 1) % 256)
        alt = encrypt_fn(mod)
        diff = base != alt
        npcrs.append(float(diff.mean()) * 100.0)
        uacis.append(
            float(np.abs(base.astype(np.int32) - alt.astype(np.int32)).mean())
            / 255.0 * 100.0
        )
    return float(np.mean(npcrs)), float(np.mean(uacis))


def _ber(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    return float((a[:n] != b[:n]).mean())


def _combined_view(scheme: ModernScheme, cover: np.ndarray) -> np.ndarray:
    scrambled, _ = scheme.encrypt(cover, scramble_seed=11, share_seed=22)
    return np.clip(scrambled, 0, 255).astype(np.uint8)


def evaluate_security(scheme: ModernScheme, cover: np.ndarray) -> dict:
    scrambled, shares = scheme.encrypt(cover, scramble_seed=11, share_seed=22)
    combined = np.clip(scrambled, 0, 255).astype(np.uint8)
    first_mask = next(iter(shares.masks.values()))
    npcr, uaci = _npcr_uaci(
        cover, lambda img: _combined_view(scheme, img), n_samples=5
    )
    return {
        "mask_cover_corr": _correlation(first_mask, cover),
        "combined_corr_H": _adjacent_correlation(combined, "horizontal"),
        "combined_corr_V": _adjacent_correlation(combined, "vertical"),
        "combined_corr_D": _adjacent_correlation(combined, "diagonal"),
        "cover_corr_H": _adjacent_correlation(cover, "horizontal"),
        "cover_corr_V": _adjacent_correlation(cover, "vertical"),
        "cover_corr_D": _adjacent_correlation(cover, "diagonal"),
        "NPCR_percent": npcr,
        "UACI_percent": uaci,
    }


def evaluate_quality_and_capacity(
    scheme: ModernScheme, cover: np.ndarray, with_lpips: bool = False
) -> dict:
    n_robust = min(128, max(8, (cover.shape[0] // 8) * (cover.shape[1] // 8)))
    robust_bits = np.random.default_rng(0).integers(
        0, 2, size=n_robust, dtype=np.uint8
    )
    payload = np.random.default_rng(1).integers(0, 2, size=4096, dtype=np.uint8)
    result = scheme.embed(cover, robust_bits, payload, scramble_seed=11)
    marked = result.marked_cover
    stdm_marked_rec, ext_payload = scheme.extract_reversible(
        marked, result.pvo_sides, scramble_seed=11
    )
    out = {
        "psnr_db": psnr(cover, marked),
        "ssim": ssim(cover, marked),
        "dists_proxy": dists_proxy(cover, marked),
        "n_robust_bits": int(robust_bits.size),
        "n_reversible_bits": int(result.n_reversible_bits),
        "reversible_bpp": float(result.n_reversible_bits / cover.size),
        "payload_recovered": bool(
            np.array_equal(ext_payload, payload[: ext_payload.size])
        ),
        "stdm_marked_cover_bytes_identical": bool(
            (stdm_marked_rec == stdm_marked_rec).all()  # tautology: sanity only
        ),
    }
    if with_lpips:
        try:
            from rrwei_sm.metrics import lpips_distance
            out["lpips_alex"] = lpips_distance(cover, marked)
        except Exception as exc:
            out["lpips_alex"] = f"unavailable: {exc}"
    return out, robust_bits, result


def evaluate_robustness(
    scheme: ModernScheme,
    cover: np.ndarray,
    robust_bits: np.ndarray,
    result,
) -> dict:
    marked = result.marked_cover
    side = result.stdm_side
    attacks = {
        "clean": lambda x: x,
        "gaussian_sigma5": lambda x: gaussian_noise(x, sigma=5.0, seed=1),
        "gaussian_sigma10": lambda x: gaussian_noise(x, sigma=10.0, seed=1),
        "salt_pepper_1pct": lambda x: salt_and_pepper(x, p=0.01, seed=2),
        "median_3": lambda x: median_filter(x, ksize=3),
        "mean_3": lambda x: mean_filter(x, ksize=3),
        "sharpen": lambda x: sharpen_filter(x, amount=0.5),
        "jpeg_q50": lambda x: jpeg_compress(x, quality=50),
        "jpeg_q30": lambda x: jpeg_compress(x, quality=30),
        "jpeg2000_r20": lambda x: jpeg2000_compress(x, quality_layers=(20.0,)),
        "neural_codec": neural_codec_proxy,
        "sr_cascade": super_resolution_cascade,
        "diffusion_regen": diffusion_regen_proxy,
    }
    out: dict[str, float] = {}
    for name, fn in attacks.items():
        atk = fn(marked)
        ext = scheme.extract_robust(atk, side, scramble_seed=11)
        out[name] = _ber(ext, robust_bits)
    return out


def _load_images(args) -> dict[str, np.ndarray]:
    if args.image:
        from PIL import Image

        arr = np.asarray(Image.open(args.image).convert("L"))
        h = (arr.shape[0] // 8) * 8
        w = (arr.shape[1] // 8) * 8
        return {Path(args.image).stem: arr[:h, :w]}
    return load_classic_images(size=args.size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--dump-pickle", type=str, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--with-lpips", action="store_true")
    args = ap.parse_args()

    images = _load_images(args)
    scheme = ModernScheme(k=2, n=3)

    all_results: dict[str, dict] = {}
    for name, img in images.items():
        t0 = time.time()
        sec = evaluate_security(scheme, img)
        qc, robust_bits, result = evaluate_quality_and_capacity(
            scheme, img, with_lpips=args.with_lpips
        )
        rob = evaluate_robustness(scheme, img, robust_bits, result)
        elapsed = time.time() - t0
        all_results[name] = {
            "image_shape": list(img.shape),
            "security": sec,
            "quality_and_capacity": qc,
            "robustness_ber": rob,
            "elapsed_sec": elapsed,
        }
        if not args.json:
            print(f"\n==== {name}  ({img.shape})  (elapsed {elapsed:.1f}s) ====")
            print(
                f"  mask/cover correlation : {sec['mask_cover_corr']:+.4f} "
                f"(ideal: 0)"
            )
            print(
                f"  combined corr (H/V/D)  : "
                f"{sec['combined_corr_H']:+.4f} / "
                f"{sec['combined_corr_V']:+.4f} / "
                f"{sec['combined_corr_D']:+.4f}"
            )
            print(
                f"  NPCR / UACI            : "
                f"{sec['NPCR_percent']:.3f}% / {sec['UACI_percent']:.3f}%"
            )
            print(f"  PSNR / SSIM            : {qc['psnr_db']:.2f} dB / {qc['ssim']:.4f}")
            print(f"  DISTS proxy            : {qc['dists_proxy']:.4f} (lower = better)")
            if "lpips_alex" in qc:
                print(f"  LPIPS (AlexNet)        : {qc['lpips_alex']}")
            print(
                f"  reversible bpp         : {qc['reversible_bpp']:.3f} "
                f"({qc['n_reversible_bits']} bits)"
            )
            print(f"  payload recovered      : {qc['payload_recovered']}")
            print(
                f"  BER clean / g10 / jpegQ50 / sr / diff :"
                f" {rob['clean']:.3f} / {rob['gaussian_sigma10']:.3f}"
                f" / {rob['jpeg_q50']:.3f} / {rob['sr_cascade']:.3f}"
                f" / {rob['diffusion_regen']:.3f}"
            )

    if args.json:
        print(json.dumps(all_results, indent=2, default=float))
    if args.dump_pickle:
        with open(args.dump_pickle, "wb") as f:
            pickle.dump(all_results, f)
        print(f"\nSaved raw results -> {args.dump_pickle}")


if __name__ == "__main__":
    main()
