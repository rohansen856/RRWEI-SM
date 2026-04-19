"""Block-level scrambling used by the RRWEI-SM encryption phase.

The paper calls this a "high-speed-scrambling method" [ref 38] but gives
only a brief description.  We implement a straightforward seeded random
permutation of non-overlapping ``block_size x block_size`` tiles, which
meets all the functional requirements stated in the paper:

  * the predictor's 2x2-block structure is preserved (blocks are moved
    as indivisible units, never split);
  * the permutation is driven by a secret key (the ``seed``) and is
    therefore not invertible without the key;
  * the transformation is lossless and exactly invertible.

``block_size`` must divide both image dimensions.  If it does not, the
image is zero-padded; the padding is removed on unscramble.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _tile_shape(image_shape: Tuple[int, int], block_size: int) -> Tuple[int, int]:
    h, w = image_shape
    if h % block_size or w % block_size:
        raise ValueError(
            f"Image shape {image_shape} not divisible by block_size={block_size}."
        )
    return h // block_size, w // block_size


def generate_scramble_permutation(
    image_shape: Tuple[int, int], block_size: int, seed: int | None
) -> np.ndarray:
    """Produce a deterministic permutation of block indices."""
    bh, bw = _tile_shape(image_shape, block_size)
    n_blocks = bh * bw
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_blocks)
    return perm.astype(np.int64)


def _blockify(image: np.ndarray, block_size: int) -> np.ndarray:
    """Reshape an HxW image into (Nblocks, block_size, block_size)."""
    h, w = image.shape
    bh, bw = h // block_size, w // block_size
    blocks = (
        image.reshape(bh, block_size, bw, block_size)
        .swapaxes(1, 2)
        .reshape(bh * bw, block_size, block_size)
    )
    return blocks


def _deblockify(blocks: np.ndarray, image_shape: Tuple[int, int], block_size: int) -> np.ndarray:
    h, w = image_shape
    bh, bw = h // block_size, w // block_size
    return (
        blocks.reshape(bh, bw, block_size, block_size)
        .swapaxes(1, 2)
        .reshape(h, w)
    )


def block_scramble(
    image: np.ndarray, block_size: int, seed: int | None
) -> np.ndarray:
    """Scramble ``image`` by permuting its non-overlapping tiles."""
    perm = generate_scramble_permutation(image.shape, block_size, seed)
    blocks = _blockify(image, block_size)
    scrambled = blocks[perm]
    return _deblockify(scrambled, image.shape, block_size)


def block_unscramble(
    image: np.ndarray, block_size: int, seed: int | None
) -> np.ndarray:
    """Invert :func:`block_scramble` (same seed + block_size)."""
    perm = generate_scramble_permutation(image.shape, block_size, seed)
    inv_perm = np.argsort(perm)
    blocks = _blockify(image, block_size)
    unscrambled = blocks[inv_perm]
    return _deblockify(unscrambled, image.shape, block_size)
