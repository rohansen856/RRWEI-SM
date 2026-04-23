#!/usr/bin/env python3
"""End-to-end demo for the *modernized* RRWEI-SM pipeline.

Runs the :class:`rrwei_sm.ModernScheme` orchestrator on a small cover
(synthetic by default, user image otherwise) and prints:

  * PSNR / SSIM between cover and marked image
  * BER of robust bits extracted from the plaintext marked image
  * BER of the same bits extracted from a mild noise-attacked image
  * Exact-recovery flag for the STDM-marked cover via PVO unrolling
  * Share-cover correlation (ideal ~0) as a sanity check

Usage
-----
    python demo.py                        # use the synthetic cover
    python demo.py path/to/grayscale.png  # use your own image
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from rrwei_sm import ModernScheme
from rrwei_sm.attacks import gaussian_noise
from rrwei_sm.metrics import psnr, ssim

try:
    from real_image_harness import load_watermark_bits
except Exception:  # pragma: no cover
    load_watermark_bits = None


def _load_or_make_cover(argv: list[str]) -> np.ndarray:
    if len(argv) > 1:
        from PIL import Image

        img = np.asarray(Image.open(Path(argv[1])).convert("L"))
        h = (img.shape[0] // 8) * 8
        w = (img.shape[1] // 8) * 8
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


def _ber(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    return float((a[:n] != b[:n]).mean())


def _save_panel(
    cover: np.ndarray,
    marked: np.ndarray,
    out_path: Path,
    *,
    extracted_watermark: np.ndarray | None = None,
    ber_clean: float | None = None,
) -> None:
    """Write a cover / marked / (extracted watermark | |diff|*25) triptych."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    axes[0].imshow(cover, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("cover")
    axes[1].imshow(marked.astype(np.uint8), cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("marked")
    if extracted_watermark is not None:
        axes[2].imshow(extracted_watermark, cmap="gray", vmin=0, vmax=255)
        title = "extracted watermark"
        if ber_clean is not None:
            title += f"\nBER clean={ber_clean:.3f}"
        axes[2].set_title(title)
    else:
        diff = (
            np.abs(cover.astype(np.int16) - marked.astype(np.int16)).clip(0, 10) * 25
        )
        axes[2].imshow(diff.astype(np.uint8), cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("|cover - marked| x25")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    cover = _load_or_make_cover(sys.argv)
    print(f"Cover shape: {cover.shape}, min/max: {cover.min()}/{cover.max()}")

    scheme = ModernScheme(k=2, n=3)
    rng = np.random.default_rng(0)

    wm_path = Path("watermark.png")
    wm_shape: tuple[int, int] | None = None
    if wm_path.exists() and load_watermark_bits is not None:
        robust_bits, wm_shape = load_watermark_bits(wm_path, cover.shape)
        print(
            f"Watermark: {wm_path} -> {wm_shape} ({robust_bits.size} bits)"
        )
    else:
        n_robust = min(128, (cover.shape[0] // 8) * (cover.shape[1] // 8))
        robust_bits = rng.integers(0, 2, size=n_robust, dtype=np.uint8)
        print(f"Watermark: random bits ({n_robust})")

    reversible_payload = rng.integers(0, 2, size=512, dtype=np.uint8)

    scrambled, shares = scheme.encrypt(cover, scramble_seed=11, share_seed=22)
    result = scheme.embed(cover, robust_bits, reversible_payload, scramble_seed=11)
    marked = result.marked_cover

    ext_clean = scheme.extract_robust(marked, result.stdm_side, scramble_seed=11)
    stdm_marked, ext_payload = scheme.extract_reversible(
        marked, result.pvo_sides, scramble_seed=11
    )

    attacked = gaussian_noise(marked, sigma=3.0, seed=1)
    ext_noisy = scheme.extract_robust(attacked, result.stdm_side, scramble_seed=11)

    ber_clean = _ber(ext_clean, robust_bits)
    print("\n== Modern pipeline ==")
    print(f"  robust bits              : {robust_bits.size}")
    print(f"  reversible bits embedded : {result.n_reversible_bits}")
    print(f"  PSNR(cover, marked)      : {psnr(cover, marked):.2f} dB")
    print(f"  SSIM(cover, marked)      : {ssim(cover, marked):.4f}")
    print(f"  BER (clean extract)      : {ber_clean:.4f}")
    print(f"  BER (sigma=3 noise)      : {_ber(ext_noisy, robust_bits):.4f}")
    print(
        "  payload recovered exactly:"
        f" {bool(np.array_equal(ext_payload, reversible_payload[: ext_payload.size]))}"
    )
    first_mask = next(iter(shares.masks.values()))
    print(f"  Corr(mask_0, cover)      : {_correlation(first_mask, cover):+.3f}")

    extracted_img: np.ndarray | None = None
    if wm_shape is not None:
        wh, ww = wm_shape
        extracted_img = (
            ext_clean[: wh * ww].reshape(wh, ww) * 255
        ).astype(np.uint8)

    out_path = Path("figures/out/demo_cover_marked.png")
    _save_panel(cover, marked, out_path,
                extracted_watermark=extracted_img, ber_clean=ber_clean)
    print(f"  panel written to         : {out_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
