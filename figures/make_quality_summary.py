#!/usr/bin/env python3
"""figures/make_quality_summary.py -> figures/out/quality_summary.png

Consolidated bar chart of imperceptibility metrics on the real-image
harness covers (Lena / Baboon / Peppers at 256x256).  Shows PSNR,
SSIM, and 1 - DISTS_proxy side by side so the three covers can be
compared at a glance.  LPIPS is plotted on a second axis if the lpips
package + torch backend are available; otherwise it's skipped silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from figures._common import ensure_figure_dir, mpl
from real_image_harness import fetch_cover, run_pipeline


def _maybe_lpips(cover: np.ndarray, marked: np.ndarray) -> float | None:
    try:
        from rrwei_sm.metrics import lpips_distance
        return float(lpips_distance(cover, marked))
    except Exception as exc:
        print(f"  ! lpips unavailable: {type(exc).__name__}: {exc}")
        return None


def main() -> None:
    plt = mpl()
    names = ["lena", "baboon", "peppers"]
    psnrs, ssims, dists_inv, lpipses = [], [], [], []
    for n in names:
        cover, _ = fetch_cover(n, 256)
        res = run_pipeline(cover, n_robust=128, payload_bits=4096)
        psnrs.append(res["psnr_db"])
        ssims.append(res["ssim"])
        dists_inv.append(1.0 - res["dists_proxy"])
        lp = _maybe_lpips(cover, res["marked"].astype(np.uint8))
        lpipses.append(lp)
        print(
            f"{n:8s}  PSNR={res['psnr_db']:.2f}dB  SSIM={res['ssim']:.4f}"
            f"  DISTS={res['dists_proxy']:.4f}"
            f"  LPIPS={'NA' if lp is None else f'{lp:.4f}'}"
        )

    have_lpips = all(x is not None for x in lpipses)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    x = np.arange(len(names))
    w = 0.35
    ax_a.bar(x - w / 2, psnrs, w, label="PSNR (dB)", color="#4f81bd")
    ax_a.set_ylabel("PSNR (dB)")
    ax_a.set_title("PSNR per cover (higher = better)")
    for i, v in enumerate(psnrs):
        ax_a.text(x[i] - w / 2, v + 0.3, f"{v:.2f}", ha="center", fontsize=9)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(names)
    ax_a.set_ylim(0, max(psnrs) * 1.15)

    ax_b.bar(x - w / 2, ssims, w, label="SSIM", color="#9bbb59")
    ax_b.bar(x + w / 2, dists_inv, w, label="1 - DISTS_proxy", color="#c0504d")
    if have_lpips:
        ax2 = ax_b.twinx()
        ax2.plot(x, lpipses, "kD--", label="LPIPS (right axis)")
        ax2.set_ylabel("LPIPS (lower = better)")
        ax2.legend(loc="upper right")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(names)
    ax_b.set_ylabel("score (higher = better)")
    ax_b.set_title("SSIM / 1 - DISTS_proxy (and LPIPS if available)")
    ax_b.set_ylim(0, 1.1)
    ax_b.legend(loc="upper left")
    fig.suptitle(
        "Imperceptibility on real covers (256x256 Lena / Baboon / Peppers, "
        "128 robust + 4096 payload bits)"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = ensure_figure_dir() / "quality_summary.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
