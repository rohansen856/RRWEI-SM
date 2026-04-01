#!/usr/bin/env python3
"""
figures/make_encrypted_visuals.py -> figures/out/encrypted_visuals.png

Reproduces the style of Fig. 9: a panel showing, for each classic-style
image, the original cover, one share, the combined encrypted image and
a histogram comparison.  Useful for visually confirming that the
encryption produces noise-like shares.
"""

from __future__ import annotations

import numpy as np

from figures._common import ensure_figure_dir, mpl
from datasets import load_classic_images
from rrwei_sm import RRWEISM


def main():
    plt = mpl()
    images = load_classic_images(size=128)
    scheme = RRWEISM(n_lsb=3)
    fig, axs = plt.subplots(
        len(images), 4, figsize=(10, 2.6 * len(images))
    )
    if len(images) == 1:
        axs = axs[None, :]

    for row_idx, (name, cover) in enumerate(images.items()):
        s1, s2, _keys = scheme.encrypt(cover, key_scramble=11, key_share=22)
        combined = np.clip(
            s1.astype(np.int64) + s2.astype(np.int64), 0, 255
        ).astype(np.uint8)

        axs[row_idx, 0].imshow(cover, cmap="gray", vmin=0, vmax=255)
        axs[row_idx, 0].set_title(f"{name}\ncover")
        axs[row_idx, 0].axis("off")

        share_u8 = np.clip(s1, 0, 255).astype(np.uint8)
        axs[row_idx, 1].imshow(share_u8, cmap="gray", vmin=0, vmax=255)
        axs[row_idx, 1].set_title("share 1  (looks like noise)")
        axs[row_idx, 1].axis("off")

        axs[row_idx, 2].imshow(combined, cmap="gray", vmin=0, vmax=255)
        axs[row_idx, 2].set_title("combined (scrambled)")
        axs[row_idx, 2].axis("off")

        axs[row_idx, 3].hist(cover.flatten(), bins=32, alpha=0.5, label="cover")
        axs[row_idx, 3].hist(
            combined.flatten(), bins=32, alpha=0.5, label="combined"
        )
        axs[row_idx, 3].set_title("histogram")
        axs[row_idx, 3].legend(fontsize=8)

    out = ensure_figure_dir() / "encrypted_visuals.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
