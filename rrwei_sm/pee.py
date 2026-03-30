"""
Prediction-Error Expansion (PEE) on significant bits.

Implements the block-level predictor of Eq. (6) / Eq. (15) with a 2x2 block
and three-identical-weights configuration, and the PEE embedding /
extraction rules of Eqs. (20)-(22).

We parameterise the scheme by ``n_lsb`` (number of LSB bits, i.e. the
amount to add/subtract when shifting the histogram in pixel space is
2**n_lsb) and work on the *HSB plane* of the image.  This matches the
paper's observation that "only the HSB plane of the image shares is used
to calculate the prediction error" (Section IV-B).

Scaled prediction error (integer-exact, SMC-friendly)
----------------------------------------------------
The paper defines the prediction error as

    e_HSB(i,j) = x_HSB(i,j) - (x_HSB(i,j+1) + x_HSB(i+1,j) + x_HSB(i+1,j+1)) / 3

which is not in general an integer.  Following the SMC formulation of
Eqs. (17)-(19), we work internally with the *scaled* error

    e_scaled(i,j) = 3 * x_HSB(i,j) - sum(context_HSB)

which is an integer that each party can compute from its own share:

    e1 = 3*s1_HSB(i,j) - sum of s1_HSB in context
    e2 = 3*s2_HSB(i,j) - sum of s2_HSB in context
    e_scaled = e1 + e2       (exact, integer, matches (e1+e2)/3 * 3)

Adding 1 to x_HSB(i,j) (i.e. adding 2**n in pixel space to the target
pixel) increases ``e_scaled`` by exactly 3.  The histogram-shift operation
therefore works with a step of ``3`` in the scaled space instead of ``1``.

This is mathematically equivalent to the paper's formulation, avoids any
rounding ambiguity, and collapses to exactly the same embedding operation
in pixel space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

import numpy as np


_SCALE = 3  # matches the "3 * x_HSB(i,j) - sum of 3 neighbours" predictor


@dataclass
class PEELayerInfo:
    """Side information recorded by one PEE layer.

    Parameters
    ----------
    ME_scaled : int
        The mode of the scaled prediction error histogram (chosen as the
        bin to embed into).
    n_lsb : int
        The LSB split used; the shift amount in pixel space is 2**n_lsb.
    skip_mask : np.ndarray of shape (Bh, Bw) bool
        True for blocks that were *not* used for embedding (either because
        of overflow/underflow protection, or because they were not part of
        the payload region).
    n_embedded : int
        Number of bits actually embedded in this layer.
    """

    ME_scaled: int
    n_lsb: int
    skip_mask: np.ndarray
    n_embedded: int = 0


@dataclass
class PEESideInfo:
    """All side-information needed to invert one or more PEE layers."""

    layers: List[PEELayerInfo] = field(default_factory=list)


def _extract_block_targets_and_context(hsb_plane: np.ndarray):
    """For a 2x2 block grid, return (target, c1, c2, c3) arrays.

    target = x_HSB at (2i,   2j),
    c1     = x_HSB at (2i,   2j+1),
    c2     = x_HSB at (2i+1, 2j),
    c3     = x_HSB at (2i+1, 2j+1).
    """
    if hsb_plane.ndim != 2:
        raise ValueError("hsb_plane must be 2D.")
    h, w = hsb_plane.shape
    if h % 2 or w % 2:
        raise ValueError("HSB plane must have even dimensions for 2x2 blocks.")
    t = hsb_plane[0::2, 0::2]
    c1 = hsb_plane[0::2, 1::2]
    c2 = hsb_plane[1::2, 0::2]
    c3 = hsb_plane[1::2, 1::2]
    return t, c1, c2, c3


def compute_scaled_error(hsb_plane: np.ndarray) -> np.ndarray:
    """Return ``e_scaled`` (Bh x Bw) for an HSB plane using 2x2 blocks."""
    t, c1, c2, c3 = _extract_block_targets_and_context(hsb_plane)
    return _SCALE * t.astype(np.int64) - (c1 + c2 + c3).astype(np.int64)


def compute_share_contribution_to_error(
    share_hsb_plane: np.ndarray,
) -> np.ndarray:
    """Per-share scaled error contribution for the SMC protocol (Eqs. 17/18).

    Each party computes this locally on its own share; the sum of the two
    parties' outputs equals :func:`compute_scaled_error` applied to the
    combined HSB plane.
    """
    return compute_scaled_error(share_hsb_plane)


def _find_mode(errors: np.ndarray, exclude_blocks: np.ndarray | None = None) -> int:
    """Find the most common value of ``errors`` (optionally excluding blocks)."""
    flat = errors.flatten()
    if exclude_blocks is not None:
        flat = flat[~exclude_blocks.flatten()]
    if flat.size == 0:
        return 0
    vals, counts = np.unique(flat, return_counts=True)
    return int(vals[np.argmax(counts)])


class PEEEmbedder:
    """Embed bits into an image using the HSB-plane PEE scheme.

    Operates in pixel space: given a uint / int image (representing either
    a plaintext image or a *combined* encrypted image, which has the same
    HSB plane as the plaintext), it embeds bits by shifting the target
    pixel of each 2x2 block by ``+2**n_lsb`` in pixel space.

    The SMC variant (:class:`~rrwei_sm.rrwei_sm.RRWEISM`) does the
    equivalent modification on *one* of the shares, which achieves the
    same net effect after recombination.
    """

    def __init__(self, n_lsb: int = 3, max_layers: int = 4):
        if not (1 <= n_lsb <= 7):
            raise ValueError("n_lsb must be in [1, 7]")
        self.n_lsb = n_lsb
        self.max_layers = max_layers

    @property
    def shift(self) -> int:
        return 1 << self.n_lsb

    def overflow_mask(self, image: np.ndarray) -> np.ndarray:
        """Per-block mask: True if embedding would over/underflow that block's target."""
        t, _, _, _ = _extract_block_targets_and_context(image)
        return t + self.shift > 255

    def embed(
        self,
        image: np.ndarray,
        bits: Iterable[int],
    ) -> Tuple[np.ndarray, PEESideInfo, int]:
        """Embed as many bits from ``bits`` as possible.

        Returns
        -------
        marked_image : np.ndarray
            The watermarked image (same dtype as input when possible).
        side : PEESideInfo
            Layer-by-layer side information needed for extraction.
        n_embedded : int
            Total number of bits embedded.
        """
        bits_arr = np.asarray(list(bits), dtype=np.uint8)
        if bits_arr.ndim != 1:
            raise ValueError("bits must be a 1D iterable of 0/1")
        total = len(bits_arr)
        side = PEESideInfo()
        bit_ptr = 0
        work = image.astype(np.int64).copy()

        for _ in range(self.max_layers):
            if bit_ptr >= total:
                break
            hsb = _pixels_to_hsb(work, self.n_lsb)
            e_scaled = compute_scaled_error(hsb)
            overflow = (
                _extract_block_targets_and_context(work)[0] + self.shift > 255
            )
            # ME chosen over non-overflow blocks to avoid bias.
            ME = _find_mode(e_scaled, exclude_blocks=overflow)

            shift_mask = (e_scaled > ME) & ~overflow
            embed_mask = (e_scaled == ME) & ~overflow

            # How many bits can we embed in this layer?
            remaining = total - bit_ptr
            n_this_layer = int(embed_mask.sum())
            if n_this_layer == 0 and int(shift_mask.sum()) == 0:
                # This layer would do nothing; stop.
                break
            use_bits = bits_arr[bit_ptr : bit_ptr + min(n_this_layer, remaining)]

            # If we have more embed-candidate blocks than remaining bits,
            # only embed into the first ``len(use_bits)`` of them; the rest
            # keep their target pixels unchanged (they stay at bin ME).
            emb_coords = np.argwhere(embed_mask)
            if len(use_bits) < len(emb_coords):
                used = emb_coords[: len(use_bits)]
                unused = emb_coords[len(use_bits) :]
                # Drop the unused coords from embed_mask so side info matches.
                embed_mask = np.zeros_like(embed_mask)
                embed_mask[used[:, 0], used[:, 1]] = True
                # The ``unused`` blocks stay at ME with bit 0 implicit.
                # This is fine for extraction: they will simply be read as
                # extra bit-0 reads, so we record n_embedded to cap the
                # reconstruction correctly.
                # (No modification needed here.)

            # Apply shift to (shift_mask) blocks: target pixel += self.shift
            _apply_target_shift(work, shift_mask, self.shift)
            # Apply embedding to (embed_mask) blocks with bits use_bits.
            _apply_target_embed(work, embed_mask, self.shift, use_bits)

            skip_mask = overflow.copy()
            side.layers.append(
                PEELayerInfo(
                    ME_scaled=ME,
                    n_lsb=self.n_lsb,
                    skip_mask=skip_mask,
                    n_embedded=len(use_bits),
                )
            )
            bit_ptr += len(use_bits)

        # Clip back to uint8 domain only if original dtype suggests it.
        if np.issubdtype(image.dtype, np.unsignedinteger):
            work = np.clip(work, 0, 255).astype(image.dtype)
        else:
            work = work.astype(image.dtype)
        return work, side, bit_ptr


class PEEExtractor:
    """Inverse of :class:`PEEEmbedder`.  Uses the layer-by-layer side info."""

    def extract(
        self, marked: np.ndarray, side: PEESideInfo
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (recovered_image, extracted_bits)."""
        work = marked.astype(np.int64).copy()
        bits_layers: List[np.ndarray] = []
        # Invert in reverse order (last-in, first-out).
        for layer in reversed(side.layers):
            shift = 1 << layer.n_lsb
            hsb = _pixels_to_hsb(work, layer.n_lsb)
            e_scaled = compute_scaled_error(hsb)
            ME = layer.ME_scaled
            skip = layer.skip_mask

            # Bits live in blocks where e_scaled is ME (bit 0) or ME + 3 (bit 1);
            # all blocks with e_scaled > ME + 3 were shifted and must be shifted back.
            bit0 = (e_scaled == ME) & ~skip
            bit1 = (e_scaled == ME + _SCALE) & ~skip
            shifted = (e_scaled > ME + _SCALE) & ~skip

            # Reconstruct the embed-bit sequence in row-major block order.
            block_flat_order = bit0 | bit1
            bits_this_layer = np.zeros(block_flat_order.sum(), dtype=np.uint8)
            coords = np.argwhere(block_flat_order)
            for idx, (bi, bj) in enumerate(coords):
                bits_this_layer[idx] = 1 if bit1[bi, bj] else 0
            # Truncate to recorded count and add to list.
            bits_this_layer = bits_this_layer[: layer.n_embedded]
            bits_layers.append(bits_this_layer)

            # Undo embedding: blocks at e = ME+3 had target += shift, undo.
            _apply_target_shift(work, bit1, -shift)
            # Undo shift: blocks at e > ME+3 had target += shift, undo.
            _apply_target_shift(work, shifted, -shift)
        bits_all = np.concatenate(list(reversed(bits_layers))) if bits_layers else np.zeros(0, dtype=np.uint8)

        if np.issubdtype(marked.dtype, np.unsignedinteger):
            work = np.clip(work, 0, 255).astype(marked.dtype)
        else:
            work = work.astype(marked.dtype)
        return work, bits_all


def _pixels_to_hsb(image: np.ndarray, n_lsb: int) -> np.ndarray:
    """Signed-aware HSB extraction (arithmetic right shift)."""
    return image.astype(np.int64) >> n_lsb


def _apply_target_shift(work: np.ndarray, block_mask: np.ndarray, delta: int) -> None:
    """Add ``delta`` to the target pixel of each 2x2 block where mask is True."""
    coords = np.argwhere(block_mask)
    for bi, bj in coords:
        work[2 * bi, 2 * bj] += delta


def _apply_target_embed(
    work: np.ndarray, block_mask: np.ndarray, shift: int, bits: np.ndarray
) -> None:
    """For each True block, add ``bit * shift`` to its target pixel."""
    coords = np.argwhere(block_mask)
    for idx, (bi, bj) in enumerate(coords):
        if idx >= len(bits):
            break
        if bits[idx]:
            work[2 * bi, 2 * bj] += shift
