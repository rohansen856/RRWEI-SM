"""
Patchwork robust watermarking (first stage of the Modified RRWEI-SM).

Implements Eqs. (23)-(27) of the paper.

Each non-overlapping ``2m``-pixel block is split into two m-element sets
by a key-driven random permutation.  One robust bit ``w`` is embedded per
block by forcing the mean-difference between the two halves to be
``+2T`` (for ``w = 1``) or ``-2T`` (for ``w = 0``).  The original
difference ``d_i`` is saved as side information so the block pixels can
be exactly recovered after extraction.

Notes
-----
* The paper embeds patchwork on the HSB plane of the shares; we expose
  the primitive in plaintext space and let the caller decide whether to
  apply it to the full pixel values or to just the HSB plane.  The
  plaintext-space embedding is mathematically equivalent for the
  Modified RRWEI-SM because the additive shares split across HSB / LSB.
* Overflow/underflow blocks (those where any pixel would leave [0, 255]
  after the shift) are skipped; their index is recorded in the returned
  skip list.
* Block size ``2m`` must divide the number of pixels of the image.  We
  process the image in row-major order reshaped into (N_blocks, 2m).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class PatchworkSideInfo:
    block_size: int                 # == 2m
    m: int
    T: int
    seed_permutation: int | None
    differences: np.ndarray         # shape (N_blocks,) int,  original d_i
    skipped: np.ndarray             # bool mask of skipped blocks
    n_embedded: int                 # number of non-skipped blocks / bits
    shape: Tuple[int, int]


class PatchworkEmbedder:
    """Embed a robust binary watermark via the patchwork technique."""

    def __init__(self, m: int = 4, T: int = 2, seed: int | None = 123):
        if m < 1:
            raise ValueError("m must be >= 1.")
        self.m = m
        self.block_size = 2 * m
        self.T = T
        self.seed = seed

    def _plan(self, n_pixels: int) -> np.ndarray:
        """Per-block permutation of length ``2m`` (Eq. 23)."""
        rng = np.random.default_rng(self.seed)
        return rng.permutation(self.block_size)

    def embed(
        self, image: np.ndarray, bits: Iterable[int]
    ) -> Tuple[np.ndarray, PatchworkSideInfo]:
        img = image.astype(np.int64).copy().flatten()
        if img.size % self.block_size:
            raise ValueError(
                f"Image size {img.size} not divisible by block_size={self.block_size}"
            )
        n_blocks = img.size // self.block_size
        perm = self._plan(img.size)

        blocks = img.reshape(n_blocks, self.block_size)
        # Apply permutation so first m are "set A" and next m are "set B".
        permuted = blocks[:, perm]

        bit_list = list(bits)
        differences = np.zeros(n_blocks, dtype=np.int64)
        skipped = np.ones(n_blocks, dtype=bool)  # default: skipped
        bit_ptr = 0

        T = self.T
        m = self.m
        for i in range(n_blocks):
            if bit_ptr >= len(bit_list):
                # Out of bits -- remaining blocks stay unchanged & skipped.
                break

            block = permuted[i]
            w = int(bit_list[bit_ptr]) & 1
            # Exact sum-difference (Eq. 23 without /m) -> exact recovery.
            delta = int(block[:m].sum() - block[m:].sum())
            d_i = delta // m

            shift_a = (2 * w - 1) * T - d_i // 2
            shift_b = -(2 * w - 1) * T + d_i // 2

            new_block = block.copy()
            new_block[:m] = block[:m] + shift_a
            new_block[m:] = block[m:] + shift_b

            # Overflow check -- this block is skipped and we try the next
            # block with the *same* bit, so no bits are ever lost.
            if np.any(new_block < 0) or np.any(new_block > 255):
                continue

            permuted[i] = new_block
            differences[i] = delta
            skipped[i] = False
            bit_ptr += 1
        n_bits = bit_ptr

        # Unpermute to recover the original ordering
        inv_perm = np.argsort(perm)
        out_blocks = permuted[:, inv_perm]
        out = out_blocks.flatten().reshape(image.shape)

        side = PatchworkSideInfo(
            block_size=self.block_size,
            m=self.m,
            T=self.T,
            seed_permutation=self.seed,
            differences=differences,
            skipped=skipped,
            n_embedded=n_bits,
            shape=image.shape,
        )
        out = np.clip(out, 0, 255).astype(image.dtype) if np.issubdtype(image.dtype, np.unsignedinteger) else out.astype(image.dtype)
        return out, side


class PatchworkExtractor:
    """Robust-bit extractor + exact-recovery routine for patchwork."""

    def extract_bits(
        self, marked: np.ndarray, side: PatchworkSideInfo
    ) -> np.ndarray:
        """Robust extraction: one bit per block, via sign of d_i^w (Eq. 26)."""
        img = marked.astype(np.int64).flatten()
        n_blocks = img.size // side.block_size
        rng = np.random.default_rng(side.seed_permutation)
        perm = rng.permutation(side.block_size)
        blocks = img.reshape(n_blocks, side.block_size)[:, perm]
        m = side.m
        bits = np.zeros(n_blocks, dtype=np.uint8)
        for i in range(n_blocks):
            if side.skipped[i]:
                bits[i] = 0  # skipped blocks have no meaningful bit
                continue
            d_w = (blocks[i, :m].sum() - blocks[i, m:].sum()) // m
            bits[i] = 1 if d_w > 0 else 0
        # Only non-skipped entries correspond to embedded bits.
        return bits[~side.skipped]

    def recover(
        self, marked: np.ndarray, side: PatchworkSideInfo, bits: np.ndarray
    ) -> np.ndarray:
        """Reverse the patchwork embedding using stored ``differences``.

        ``bits`` is the sequence of embedded bits (length == n_embedded).
        """
        img = marked.astype(np.int64).flatten()
        n_blocks = img.size // side.block_size
        rng = np.random.default_rng(side.seed_permutation)
        perm = rng.permutation(side.block_size)
        inv_perm = np.argsort(perm)
        blocks = img.reshape(n_blocks, side.block_size)[:, perm]
        m = side.m
        T = side.T

        b_iter = iter(bits.tolist())
        for i in range(n_blocks):
            if side.skipped[i]:
                continue
            try:
                w = next(b_iter)
            except StopIteration:
                break
            delta = int(side.differences[i])
            d_i = delta // m
            # Undo Eq. (24): subtract the embedding shift.
            shift_a = (2 * w - 1) * T - d_i // 2
            shift_b = -(2 * w - 1) * T + d_i // 2
            blocks[i, :m] -= shift_a
            blocks[i, m:] -= shift_b
        rec = blocks[:, inv_perm].flatten().reshape(side.shape)
        return np.clip(rec, 0, 255).astype(np.uint8) if np.issubdtype(marked.dtype, np.unsignedinteger) else rec.astype(marked.dtype)
