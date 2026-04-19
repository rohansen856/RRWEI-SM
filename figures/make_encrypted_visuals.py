#!/usr/bin/env python3
"""figures/make_encrypted_visuals.py -> figures/out/encrypted_visuals.png

Visual confirmation of the modern encryption pipeline: for each
classic cover, show the cover, one share, the combined encrypted
image (scrambled sum), and a histogram comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from figures._common import ensure_figure_dir, mpl
from datasets import load_classic_images
from rrwei_sm.crypto_scrambler import block_scramble
from rrwei_sm.secret_sharing import additive_share_image, additive_combine_shares


def main() -> None:
    plt = mpl()
    images = load_classic_images(size=128)

    fig, axs = plt.subplots(len(images), 4, figsize=(10, 2.6 * len(images)))
    if len(images) == 1:
        axs = axs[None, :]

    for row_idx, (name, cover) in enumerate(images.items()):
        scrambled = block_scramble(cover, block_size=2, seed=11)
        sharing = additive_share_image(scrambled, n_parties=3, seed=22)
        share0 = np.mod(sharing.shares[0], 256).astype(np.uint8)
        combined = np.mod(
            additive_combine_shares(sharing.shares), 256
        ).astype(np.uint8)

        axs[row_idx, 0].imshow(cover, cmap="gray", vmin=0, vmax=255)
        axs[row_idx, 0].set_title(f"{name} cover")
        axs[row_idx, 1].imshow(share0, cmap="gray", vmin=0, vmax=255)
        axs[row_idx, 1].set_title("share 0 (mod 256)")
        axs[row_idx, 2].imshow(combined, cmap="gray", vmin=0, vmax=255)
        axs[row_idx, 2].set_title("combined (scrambled)")
        axs[row_idx, 3].hist(
            cover.ravel(), bins=32, alpha=0.5, label="cover", density=True
        )
        axs[row_idx, 3].hist(
            share0.ravel(), bins=32, alpha=0.5, label="share 0", density=True
        )
        axs[row_idx, 3].legend(fontsize=8)
        axs[row_idx, 3].set_title("histograms")
        for c in range(3):
            axs[row_idx, c].axis("off")

    fig.tight_layout()
    out = ensure_figure_dir() / "encrypted_visuals.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
