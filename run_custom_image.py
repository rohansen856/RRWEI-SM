#!/usr/bin/env python3
"""Run the modernized RRWEI-SM pipeline on an arbitrary user-supplied image.

Converts the input to 8-bit grayscale, center-crops to a square whose
side is a multiple of 8 (required by the STDM block grid), runs the
full ``ModernScheme`` pipeline (encrypt -> (k, n) share -> STDM robust
watermark -> PVO reversible payload), then reports PSNR / SSIM / DISTS
and robust-bit BER under clean / Gaussian / JPEG attacks.  Saves a
cover / marked / diff panel to ``figures/out/custom_image.png``.

Usage
-----
    python run_custom_image.py <path-to-image> [--size 512]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from real_image_harness import (
    check_acceptance,
    load_watermark_bits,
    run_pipeline,
    save_panel,
)


def save_artifacts(
    cover: np.ndarray,
    res: dict,
    out_dir: Path,
    *,
    stem: str,
) -> dict[str, Path]:
    """Save cover / scrambled / shares / marked / attacked / recovered
    as individual 8-bit PNGs alongside the composite panel."""
    from PIL import Image

    from rrwei_sm import crypto_scrambler
    from rrwei_sm.attacks import gaussian_noise, jpeg_compress

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    def _save(name: str, arr: np.ndarray) -> None:
        p = out_dir / f"{stem}__{name}.png"
        Image.fromarray(arr.astype(np.uint8)).save(p)
        saved[name] = p

    marked = res["marked"].astype(np.uint8)
    _save("01_cover", cover)
    _save("03_marked", marked)

    scrambled = crypto_scrambler.block_scramble(cover, block_size=8, seed=11)
    _save("02_scrambled", scrambled)

    _save("04_attacked_gaussian_s10", gaussian_noise(marked, sigma=10.0, seed=1))
    _save("05_attacked_jpeg_q40", jpeg_compress(marked, quality=40))

    diff = np.abs(cover.astype(np.int32) - marked.astype(np.int32)).astype(np.uint8)
    _save("06_diff_abs_x25", np.clip(diff * 25, 0, 255).astype(np.uint8))
    return saved


def load_gray(path: Path, size: int | None) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("L")
    arr = np.asarray(img)
    h, w = arr.shape
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    arr = arr[y0 : y0 + s, x0 : x0 + s]
    if size is not None and size != s:
        arr = np.asarray(
            Image.fromarray(arr).resize((size, size), Image.BICUBIC)
        )
    h = (arr.shape[0] // 8) * 8
    w = (arr.shape[1] // 8) * 8
    return arr[:h, :w].astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=str, help="Path to the input image")
    ap.add_argument(
        "--size", type=int, default=512,
        help="Resize the (center-cropped) square to this side (default 512). "
             "Use 0 to keep native size (truncated to multiple of 8).",
    )
    ap.add_argument("--n-robust", type=int, default=128)
    ap.add_argument("--payload-bits", type=int, default=4096)
    ap.add_argument(
        "--panel", type=str, default="figures/out/custom_image.png"
    )
    ap.add_argument(
        "--watermark", type=str, default="watermark.png",
        help=("Path to a watermark image to embed and visualise. Set to "
              "'' or 'none' to use random bits + diff-map panel."),
    )
    args = ap.parse_args()

    path = Path(args.image)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    size = None if args.size == 0 else args.size
    cover = load_gray(path, size)
    print(f"input               : {path.name}")
    print(f"processed shape     : {cover.shape} (grayscale, multiple of 8)")

    wm_path = None
    if args.watermark and args.watermark.lower() != "none":
        wm_candidate = Path(args.watermark)
        if wm_candidate.exists():
            wm_path = wm_candidate

    robust_bits = None
    wm_shape = None
    n_robust = min(args.n_robust, (cover.shape[0] // 8) * (cover.shape[1] // 8))
    if wm_path is not None:
        robust_bits, wm_shape = load_watermark_bits(wm_path, cover.shape)
        n_robust = int(robust_bits.size)
        print(f"watermark           : {wm_path} -> {wm_shape} ({n_robust} bits)")
    else:
        print("watermark           : (none, using random bits)")

    res = run_pipeline(
        cover, n_robust=n_robust, payload_bits=args.payload_bits,
        robust_bits=robust_bits, watermark_shape=wm_shape,
    )
    res["_cover"] = cover
    res["source"] = str(path)
    failures = check_acceptance(res)
    res["acceptance_failures"] = failures

    print(f"n robust bits       : {n_robust}")
    print(f"PSNR (dB)           : {res['psnr_db']:.2f}")
    print(f"SSIM                : {res['ssim']:.4f}")
    print(f"DISTS proxy         : {res['dists_proxy']:.4f}")
    print(
        f"reversible bpp      : {res['reversible_bpp']:.3f} "
        f"({res['n_reversible_bits']} bits)"
    )
    print(f"BER clean           : {res['ber_clean']:.3f}")
    print(f"BER gaussian s10    : {res['ber_gaussian_sigma10']:.3f}")
    print(f"BER jpeg q40        : {res['ber_jpeg_q40']:.3f}")
    print(f"payload recovered   : {res['payload_recovered']}")
    print(
        f"acceptance          : {'PASS' if not failures else 'FAIL'}"
        + ("\n  - " + "\n  - ".join(failures) if failures else "")
    )

    out_path = Path(args.panel)
    save_panel({path.stem: res}, out_path)
    print(f"\nsaved visual panel -> {out_path}")

    artifacts_dir = out_path.parent / f"{path.stem}_artifacts"
    saved = save_artifacts(cover, res, artifacts_dir, stem=path.stem)
    print(f"saved individual artifacts -> {artifacts_dir}/")
    for name, p in saved.items():
        print(f"  - {name:28s} {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
