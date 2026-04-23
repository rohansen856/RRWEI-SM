#!/usr/bin/env python3
"""Real-image harness for the modernized RRWEI-SM pipeline.

Downloads (or loads from a local cache) the three canonical test
images used by the paper -- Lena, Baboon, Peppers -- from a public
GitHub mirror, runs the full ``ModernScheme`` pipeline on each, and
checks a set of acceptance thresholds:

  * Clean robust-bit extraction BER == 0
  * PSNR(cover, marked) >= 35 dB
  * SSIM(cover, marked) >= 0.85
  * Reversible payload recovered exactly
  * Gaussian sigma=10 BER <= 0.15
  * JPEG q=40 BER <= 0.15

The harness also emits a JSON per-image summary to stdout and
(optionally) saves a visual diff panel of the cover / marked / diff
images to ``figures/out/real_images.png``.

Usage
-----
    python real_image_harness.py
    python real_image_harness.py --size 256 --strict
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

_MIRRORS: dict[str, list[str]] = {
    "lena": [
        "https://raw.githubusercontent.com/mohammadimtiazz/standard-test-images-for-Image-Processing/master/standard_test_images/lena_gray_512.tif",
        "https://github.com/opencv/opencv/raw/master/samples/data/lena.jpg",
    ],
    "baboon": [
        "https://raw.githubusercontent.com/mohammadimtiazz/standard-test-images-for-Image-Processing/master/standard_test_images/mandril_gray.tif",
        "https://github.com/opencv/opencv/raw/master/samples/data/baboon.jpg",
    ],
    "peppers": [
        "https://raw.githubusercontent.com/mohammadimtiazz/standard-test-images-for-Image-Processing/master/standard_test_images/peppers_gray.tif",
        "https://homepages.cae.wisc.edu/~ece533/images/peppers.png",
        "https://raw.githubusercontent.com/scikit-image/skimage-tutorials/main/images/color.png",
    ],
}


def _skimage_fallback(name: str, size: int) -> np.ndarray | None:
    """Last-resort fallback using real photographs from skimage.data."""
    try:
        from skimage import data
        from skimage.color import rgb2gray
    except Exception:
        return None
    mapping = {
        "lena": data.astronaut,   # portrait-style, smooth-to-textured
        "baboon": data.chelsea,   # high-frequency fur texture
        "peppers": data.coffee,   # smooth blobs of different means
    }
    fn = mapping.get(name)
    if fn is None:
        return None
    arr = fn()
    if arr.ndim == 3:
        arr = (rgb2gray(arr) * 255).astype(np.uint8)
    return _crop_square(arr, size)

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "assets" / "test_images"


def _download(url: str, out: Path) -> bool:
    import requests

    try:
        resp = requests.get(url, timeout=15, stream=True)
        resp.raise_for_status()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 14):
                f.write(chunk)
        return True
    except Exception as exc:
        print(f"  ! download failed: {url} ({exc})", file=sys.stderr)
        return False


def fetch_cover(
    name: str, size: int, *, cache_dir: Path = _DEFAULT_CACHE_DIR,
    urls: Iterable[str] | None = None, allow_fallback: bool = True,
) -> tuple[np.ndarray, str]:
    """Return ``(image, source)`` tuple.  ``source`` is either 'cache',
    'downloaded:<url>', or 'synthetic' if we had to fall back."""
    from PIL import Image

    cached = cache_dir / f"{name}.png"
    if cached.exists():
        arr = np.asarray(Image.open(cached).convert("L"))
    else:
        arr = None
        for url in urls if urls is not None else _MIRRORS.get(name, []):
            tmp = cache_dir / f"{name}_raw.{url.rsplit('.', 1)[-1]}"
            if _download(url, tmp):
                try:
                    arr = np.asarray(Image.open(tmp).convert("L"))
                    # Save normalized PNG for quick reloading.
                    Image.fromarray(arr).save(cached)
                    source = f"downloaded:{url}"
                    break
                except Exception as exc:
                    print(f"  ! could not open {tmp}: {exc}", file=sys.stderr)
                    arr = None
        if arr is None:
            if not allow_fallback:
                raise RuntimeError(f"could not obtain '{name}' from any mirror")
            ski = _skimage_fallback(name, size)
            if ski is not None:
                # Cache the skimage-derived image too.
                try:
                    from PIL import Image as _PIL

                    _PIL.fromarray(ski).save(cached)
                except Exception:
                    pass
                return ski, "skimage_fallback"
            from datasets import baboon_like, lena_like, peppers_like

            synth_map = {
                "lena": lena_like,
                "baboon": baboon_like,
                "peppers": peppers_like,
            }
            arr = synth_map[name](size=max(size, 256))
            return _crop_square(arr, size), "synthetic"
        source = "cache" if cached.exists() and arr is not None else source  # noqa
    else_source = "cache"
    # Compile the source string for the cached branch.
    if cached.exists() and "source" not in locals():
        source = else_source
    arr = _crop_square(arr, size)
    return arr, source


def _crop_square(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    arr = arr[y0 : y0 + s, x0 : x0 + s]
    if size != s:
        from PIL import Image

        arr = np.asarray(
            Image.fromarray(arr).resize((size, size), Image.BICUBIC)
        )
    h = (arr.shape[0] // 8) * 8
    w = (arr.shape[1] // 8) * 8
    return arr[:h, :w].astype(np.uint8)


def _ber(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    return 0.0 if n == 0 else float((a[:n] != b[:n]).mean())


def pick_watermark_shape(
    cover_hw: tuple[int, int],
    native_wh: tuple[int, int],
    *,
    budget_frac: float = 0.9,
) -> tuple[int, int]:
    """Pick (h, w) for the watermark thumbnail such that h*w fits in
    ``budget_frac`` * (number of 8x8 STDM blocks in the cover) and the
    aspect ratio is as close as possible to the native watermark."""
    n_blocks = (cover_hw[0] // 8) * (cover_hw[1] // 8)
    budget = int(n_blocks * budget_frac)
    native_w, native_h = native_wh
    ratio = native_w / max(native_h, 1)
    h = max(8, int(round(np.sqrt(budget / max(ratio, 1e-6)))))
    w = int(round(h * ratio))
    while h * w > budget and (h > 1 or w > 1):
        if w >= h:
            w -= 1
        else:
            h -= 1
    return h, w


def load_watermark_bits(
    wm_path: Path,
    cover_hw: tuple[int, int],
    *,
    budget_frac: float = 0.9,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Load ``wm_path``, binarise (mean threshold), downsample to fit
    the cover's STDM budget, and return (bits, (wm_h, wm_w))."""
    from PIL import Image

    img = Image.open(wm_path)
    wm_h, wm_w = pick_watermark_shape(cover_hw, img.size)
    gray = img.convert("L").resize((wm_w, wm_h), Image.LANCZOS)
    arr = np.asarray(gray, dtype=np.uint8)
    bits = (arr < arr.mean()).astype(np.uint8).flatten()
    return bits, (wm_h, wm_w)


def run_pipeline(
    cover: np.ndarray,
    *,
    n_robust: int,
    payload_bits: int,
    robust_bits: np.ndarray | None = None,
    watermark_shape: tuple[int, int] | None = None,
) -> dict:
    from rrwei_sm import ModernScheme
    from rrwei_sm.attacks import gaussian_noise, jpeg_compress
    from rrwei_sm.metrics import dists_proxy, psnr, ssim

    scheme = ModernScheme(k=2, n=3)
    rng = np.random.default_rng(0)
    if robust_bits is None:
        robust_bits = rng.integers(0, 2, size=n_robust, dtype=np.uint8)
    else:
        robust_bits = robust_bits.astype(np.uint8, copy=False)
    payload = rng.integers(0, 2, size=payload_bits, dtype=np.uint8)
    result = scheme.embed(cover, robust_bits, payload, scramble_seed=11)
    marked = result.marked_cover

    ext_clean = scheme.extract_robust(marked, result.stdm_side, scramble_seed=11)
    _, ext_payload = scheme.extract_reversible(
        marked, result.pvo_sides, scramble_seed=11
    )

    atk_g = gaussian_noise(marked, sigma=10.0, seed=1)
    ext_g = scheme.extract_robust(atk_g, result.stdm_side, scramble_seed=11)

    atk_j = jpeg_compress(marked, quality=40)
    ext_j = scheme.extract_robust(atk_j, result.stdm_side, scramble_seed=11)

    return {
        "cover_sha1": hashlib.sha1(cover.tobytes()).hexdigest()[:12],
        "shape": list(cover.shape),
        "n_robust_bits": int(robust_bits.size),
        "n_reversible_bits": int(result.n_reversible_bits),
        "reversible_bpp": float(result.n_reversible_bits / cover.size),
        "psnr_db": psnr(cover, marked),
        "ssim": ssim(cover, marked),
        "dists_proxy": dists_proxy(cover, marked),
        "ber_clean": _ber(ext_clean, robust_bits),
        "ber_gaussian_sigma10": _ber(ext_g, robust_bits),
        "ber_jpeg_q40": _ber(ext_j, robust_bits),
        "payload_recovered": bool(
            np.array_equal(ext_payload, payload[: ext_payload.size])
        ),
        "marked": marked,
        "robust_bits": robust_bits,
        "ext_clean_bits": ext_clean,
        "watermark_shape": watermark_shape,
    }


ACCEPTANCE = {
    "psnr_db_min": 35.0,
    "ssim_min": 0.85,
    "ber_clean_max": 0.0,
    "ber_gaussian_sigma10_max": 0.15,
    "ber_jpeg_q40_max": 0.15,
    "payload_recovered": True,
}


def check_acceptance(res: dict) -> list[str]:
    failures: list[str] = []
    if res["psnr_db"] < ACCEPTANCE["psnr_db_min"]:
        failures.append(f"PSNR {res['psnr_db']:.2f} < {ACCEPTANCE['psnr_db_min']}")
    if res["ssim"] < ACCEPTANCE["ssim_min"]:
        failures.append(f"SSIM {res['ssim']:.3f} < {ACCEPTANCE['ssim_min']}")
    if res["ber_clean"] > ACCEPTANCE["ber_clean_max"]:
        failures.append(f"BER(clean) {res['ber_clean']:.3f} > 0")
    if res["ber_gaussian_sigma10"] > ACCEPTANCE["ber_gaussian_sigma10_max"]:
        failures.append(
            f"BER(sigma10) {res['ber_gaussian_sigma10']:.3f} > "
            f"{ACCEPTANCE['ber_gaussian_sigma10_max']}"
        )
    if res["ber_jpeg_q40"] > ACCEPTANCE["ber_jpeg_q40_max"]:
        failures.append(
            f"BER(jpegQ40) {res['ber_jpeg_q40']:.3f} > "
            f"{ACCEPTANCE['ber_jpeg_q40_max']}"
        )
    if not res["payload_recovered"]:
        failures.append("payload not recovered")
    return failures


def save_panel(results: dict[str, dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    fig, axs = plt.subplots(len(names), 3, figsize=(9, 3 * len(names)))
    if len(names) == 1:
        axs = axs[None, :]
    for row, name in enumerate(names):
        r = results[name]
        cover = r["_cover"]
        marked = r["marked"]
        axs[row, 0].imshow(cover, cmap="gray", vmin=0, vmax=255)
        axs[row, 0].set_title(f"{name} cover")
        axs[row, 1].imshow(marked, cmap="gray", vmin=0, vmax=255)
        axs[row, 1].set_title(
            f"{name} marked\nPSNR={r['psnr_db']:.1f} SSIM={r['ssim']:.3f}"
        )
        if r.get("watermark_shape") is not None and r.get("ext_clean_bits") is not None:
            wm_h, wm_w = r["watermark_shape"]
            extracted = (r["ext_clean_bits"][: wm_h * wm_w]
                         .reshape(wm_h, wm_w) * 255).astype(np.uint8)
            axs[row, 2].imshow(extracted, cmap="gray", vmin=0, vmax=255)
            axs[row, 2].set_title(
                f"extracted watermark\n"
                f"BER clean={r['ber_clean']:.3f} jpegQ40={r['ber_jpeg_q40']:.3f}"
            )
        else:
            diff = np.abs(
                cover.astype(np.int32) - marked.astype(np.int32)
            ).astype(np.uint8)
            axs[row, 2].imshow(diff, cmap="inferno", vmin=0, vmax=10)
            axs[row, 2].set_title(
                f"|cover - marked| (clip to 10)\n"
                f"BER g10={r['ber_gaussian_sigma10']:.2f} jpegQ40={r['ber_jpeg_q40']:.2f}"
            )
        for c in range(3):
            axs[row, c].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n-robust", type=int, default=128)
    ap.add_argument("--payload-bits", type=int, default=4096)
    ap.add_argument(
        "--panel", type=str, default="figures/out/real_images.png"
    )
    ap.add_argument(
        "--strict", action="store_true",
        help="Exit nonzero if any acceptance check fails.",
    )
    ap.add_argument(
        "--no-fallback", action="store_true",
        help="Do not fall back to synthetic covers if the download fails.",
    )
    ap.add_argument(
        "--watermark", type=str, default="watermark.png",
        help=("Path to a watermark image to embed and visualise in the "
              "third panel. Set to '' or 'none' to fall back to the "
              "random-bits diff-map panel."),
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    wm_path = None
    if args.watermark and args.watermark.lower() != "none":
        p = Path(args.watermark)
        if p.exists():
            wm_path = p
        else:
            print(f"  ! watermark file not found: {p}, falling back to random bits",
                  file=sys.stderr)

    all_results: dict[str, dict] = {}
    any_failed = False
    for name in ("lena", "baboon", "peppers"):
        print(f"\n=== {name} ===")
        cover, source = fetch_cover(
            name, args.size, allow_fallback=not args.no_fallback
        )
        print(f"  source              : {source}")
        print(f"  shape               : {cover.shape}")
        n_robust = min(args.n_robust, (cover.shape[0] // 8) * (cover.shape[1] // 8))
        robust_bits = None
        wm_shape = None
        if wm_path is not None:
            robust_bits, wm_shape = load_watermark_bits(wm_path, cover.shape)
            n_robust = int(robust_bits.size)
            print(f"  watermark           : {wm_path} -> {wm_shape} ({n_robust} bits)")
        res = run_pipeline(
            cover, n_robust=n_robust, payload_bits=args.payload_bits,
            robust_bits=robust_bits, watermark_shape=wm_shape,
        )
        res["_cover"] = cover
        res["source"] = source
        failures = check_acceptance(res)
        res["acceptance_failures"] = failures
        all_results[name] = res
        print(f"  PSNR (dB)           : {res['psnr_db']:.2f}")
        print(f"  SSIM                : {res['ssim']:.4f}")
        print(f"  DISTS proxy         : {res['dists_proxy']:.4f}")
        print(
            f"  reversible bpp      : {res['reversible_bpp']:.3f} "
            f"({res['n_reversible_bits']} bits)"
        )
        print(f"  BER clean           : {res['ber_clean']:.3f}")
        print(f"  BER gaussian s10    : {res['ber_gaussian_sigma10']:.3f}")
        print(f"  BER jpeg q40        : {res['ber_jpeg_q40']:.3f}")
        print(f"  payload recovered   : {res['payload_recovered']}")
        print(
            f"  acceptance          : {'PASS' if not failures else 'FAIL'}"
            + ("\n  - " + "\n  - ".join(failures) if failures else "")
        )
        if failures:
            any_failed = True

    out_path = Path(args.panel)
    save_panel(all_results, out_path)
    print(f"\nSaved visual panel -> {out_path}")

    if args.json:
        dumpable = {
            k: {kk: (vv if not isinstance(vv, np.ndarray) else None)
                for kk, vv in v.items()
                if kk not in ("_cover", "marked")}
            for k, v in all_results.items()
        }
        print(json.dumps(dumpable, indent=2, default=float))

    return 2 if (args.strict and any_failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
