#!/usr/bin/env python3
"""Embed an arbitrary image as the robust STDM watermark.

Usage
-----
    python embed_watermark.py              # cover=input.png, watermark=watermark.png
    python embed_watermark.py COVER WM     # custom paths

The watermark is binarised (Otsu-style mean threshold), down-sampled to
a small thumbnail that fits the cover's STDM block-count budget, flattened
into a bitstream, and embedded via :class:`rrwei_sm.ModernScheme`.

Outputs land in ``figures/out/watermark_demo/``:

    01_cover.png                 preprocessed cover (grayscale, cropped)
    02_watermark_input.png       the binarised watermark thumbnail
    03_marked.png                the marked cover
    04_extracted_clean.png       watermark recovered from the clean marked image
    05_extracted_gaussian_s10.png           ... after Gaussian sigma=10
    05_extracted_jpeg_q75.png               ... after JPEG q=75
    05_extracted_jpeg_q40.png               ... after JPEG q=40
    panel.png                    side-by-side summary (matplotlib)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from rrwei_sm import ModernScheme
from rrwei_sm.attacks import gaussian_noise, jpeg_compress
from rrwei_sm.metrics import psnr, ssim
from rrwei_sm.stdm import STDMConfig


def load_gray_square(path: Path, block: int = 8) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    h, w = arr.shape
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    arr = arr[y0 : y0 + s, x0 : x0 + s]
    s2 = (s // block) * block
    return arr[:s2, :s2]


def load_binary_watermark(path: Path, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    ):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba).convert("RGB")
    img = img.convert("L").resize((w, h), Image.LANCZOS)
    a = np.asarray(img, dtype=np.uint8)
    return (a < a.mean()).astype(np.uint8)


def save_bits_as_png(bits: np.ndarray, shape_hw: tuple[int, int], path: Path) -> None:
    arr = (bits.reshape(shape_hw) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def pick_wm_shape(cover_hw: tuple[int, int], target_ratio_wh: float) -> tuple[int, int]:
    """Choose (h, w) so h*w <= 0.9 * #blocks and w/h ~= target_ratio_wh."""
    n_blocks = (cover_hw[0] // 8) * (cover_hw[1] // 8)
    budget = int(n_blocks * 0.9)
    h = max(8, int(round(np.sqrt(budget / target_ratio_wh))))
    w = int(round(h * target_ratio_wh))
    while h * w > budget:
        if w > h:
            w -= 1
        else:
            h -= 1
    return h, w


def main() -> int:
    cover_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("input.png")
    wm_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("watermark.png")
    out_dir = Path("figures/out/watermark_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    cover = load_gray_square(cover_path)
    bh, bw = cover.shape[0] // 8, cover.shape[1] // 8
    n_blocks = bh * bw

    wm_native = Image.open(wm_path)
    native_w, native_h = wm_native.size
    ratio = native_w / native_h
    wm_h, wm_w = pick_wm_shape(cover.shape, ratio)

    wm = load_binary_watermark(wm_path, (wm_h, wm_w))
    bits = wm.flatten().astype(np.uint8)

    print(f"cover   : {cover_path} -> shape {cover.shape}, {n_blocks} STDM blocks")
    print(f"watermk : {wm_path} -> thumbnail {wm.shape} ({bits.size} bits)")
    print(f"budget  : {bits.size} / {n_blocks} blocks used ({100*bits.size/n_blocks:.1f}%)")

    scheme = ModernScheme(stdm_config=STDMConfig(delta=60.0, n_coeffs=4, seed=0))
    empty_payload = np.zeros(0, dtype=np.uint8)
    res = scheme.embed(cover, bits, empty_payload, scramble_seed=11)
    marked = res.marked_cover.astype(np.uint8)

    psnr_db = psnr(cover, marked)
    ssim_val = ssim(cover, marked)
    print(f"PSNR    : {psnr_db:.2f} dB")
    print(f"SSIM    : {ssim_val:.4f}")

    Image.fromarray(cover).save(out_dir / "01_cover.png")
    Image.fromarray((wm * 255).astype(np.uint8)).save(out_dir / "02_watermark_input.png")
    Image.fromarray(marked).save(out_dir / "03_marked.png")

    ext_clean = scheme.extract_robust(marked, res.stdm_side, scramble_seed=11)
    ber_clean = float((ext_clean != bits).mean())
    save_bits_as_png(ext_clean, (wm_h, wm_w), out_dir / "04_extracted_clean.png")
    print(f"BER     : clean           = {ber_clean:.4f}")

    attacks = {
        "gaussian_s10": gaussian_noise(marked, sigma=10.0, seed=1),
        "jpeg_q75":     jpeg_compress(marked, quality=75),
        "jpeg_q40":     jpeg_compress(marked, quality=40),
    }
    bers: dict[str, float] = {}
    for name, atk in attacks.items():
        ext = scheme.extract_robust(atk, res.stdm_side, scramble_seed=11)
        bers[name] = float((ext != bits).mean())
        save_bits_as_png(ext, (wm_h, wm_w), out_dir / f"05_extracted_{name}.png")
        print(f"BER     : {name:15s} = {bers[name]:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5))
    axes[0, 0].imshow(cover, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"cover\n{cover.shape[0]}x{cover.shape[1]}")
    axes[0, 1].imshow(marked, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"marked\nPSNR={psnr_db:.2f} dB, SSIM={ssim_val:.4f}")
    axes[0, 2].imshow(wm * 255, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title(f"watermark (input)\n{wm_h}x{wm_w} ({bits.size} bits)")

    axes[1, 0].imshow(
        ext_clean.reshape(wm_h, wm_w) * 255, cmap="gray", vmin=0, vmax=255
    )
    axes[1, 0].set_title(f"recovered, clean\nBER={ber_clean:.3f}")
    ext_g10 = scheme.extract_robust(
        attacks["gaussian_s10"], res.stdm_side, scramble_seed=11
    )
    axes[1, 1].imshow(
        ext_g10.reshape(wm_h, wm_w) * 255, cmap="gray", vmin=0, vmax=255
    )
    axes[1, 1].set_title(
        f"recovered, Gaussian sigma=10\nBER={bers['gaussian_s10']:.3f}"
    )
    ext_j40 = scheme.extract_robust(
        attacks["jpeg_q40"], res.stdm_side, scramble_seed=11
    )
    axes[1, 2].imshow(
        ext_j40.reshape(wm_h, wm_w) * 255, cmap="gray", vmin=0, vmax=255
    )
    axes[1, 2].set_title(
        f"recovered, JPEG q=40\nBER={bers['jpeg_q40']:.3f}"
    )

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    panel = out_dir / "panel.png"
    fig.savefig(panel, dpi=160)
    plt.close(fig)
    print(f"panel   : {panel}")
    print(f"artifacts written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
