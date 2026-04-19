"""Pixel Value Ordering (PVO) reversible data hiding.

Replaces the legacy 3-neighbour PEE predictor with the PVO scheme of
Li-Li 2013, "Efficient Reversible Data Hiding Based on Multiple
Histograms Modification", extended with a **pairwise** embedding
(both the max-side and the min-side of each block carry a bit).

Algorithm, per non-overlapping 2x2 block
----------------------------------------
Sort the four pixels ascending: ``x_0 <= x_1 <= x_2 <= x_3``.

Max side (uses the position of ``x_3``):

* ``u = x_3 - x_2``.
* If ``u == 1``: embed bit ``b_max`` by setting ``x_3' = x_3 + b_max``.
* If ``u >  1``: shift ``x_3' = x_3 + 1`` (histogram shift; no bit
  carried, but the shift is reversible given the marked image).
* If ``u == 0``: block is ambiguous on the max side; skip it and
  record in the location map.

Min side (symmetric, uses ``x_0``):

* ``v = x_1 - x_0`` (>= 0).
* If ``v == 1``: embed ``b_min`` by ``x_0' = x_0 - b_min``.
* If ``v >  1``: shift ``x_0' = x_0 - 1``.
* If ``v == 0``: skip.

Overflow / underflow handling:

* ``x_3 == 255`` -> cannot apply the max rule; skip max side.
* ``x_0 == 0``   -> cannot apply the min rule; skip min side.

Reversibility
-------------
Given the marked image and the skip masks, each side can be inverted
deterministically: sort, check ``u'`` / ``v'``, peel the bit, unshift.

Capacity
--------
Theoretical maximum per 2x2 block: 2 bits (one max-side + one
min-side).  Per layer that is ``0.5`` bpp.  In practice 0.25-0.4 bpp
per layer on smooth images; multiple layers stack linearly.

Side information (per side, per layer)
--------------------------------------
* ``skip_mask`` : bool array of length ``N_blocks`` -- blocks that
  contributed neither a bit nor a shift.
* ``n_embedded`` : int -- how many bits the side carried.

These go through the rANS coder (:mod:`rrwei_sm_modern.coding`) in the
orchestrator stage, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "PVOSideInfo",
    "embed",
    "extract",
    "capacity_estimate",
]


_BLOCK = 2  # 2x2 blocks


@dataclass
class PVOSideInfo:
    """Per-layer side information for reversible inversion."""

    shape: tuple[int, int]
    n_blocks: int
    # Per-block boolean flags, all length n_blocks:
    skipped_max: np.ndarray   # True where max side was not processed
    skipped_min: np.ndarray   # True where min side was not processed
    n_embedded_max: int
    n_embedded_min: int


def _block_view(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = image.shape
    if h % _BLOCK or w % _BLOCK:
        raise ValueError(
            f"shape {image.shape} not divisible by {_BLOCK}"
        )
    bh = h // _BLOCK
    bw = w // _BLOCK
    # (bh, bw, 2, 2) -> (n_blocks, 4)
    return (
        image.reshape(bh, _BLOCK, bw, _BLOCK)
        .swapaxes(1, 2)
        .reshape(bh * bw, 4)
        .copy(),
        (bh, bw),
    )


def _unblock(blocks: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    bh, bw = grid
    return (
        blocks.reshape(bh, bw, _BLOCK, _BLOCK)
        .swapaxes(1, 2)
        .reshape(bh * _BLOCK, bw * _BLOCK)
    )


def _sorted_with_argmax_argmin(block: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return (sorted_vals, argmax_position, argmin_position).

    Ties are broken by position (stable sort), matching standard PVO.
    """
    order = np.argsort(block, kind="stable")
    sorted_vals = block[order]
    argmin = int(order[0])
    argmax = int(order[-1])
    return sorted_vals, argmax, argmin


def _apply_max_rule(block: np.ndarray, bit: int | None) -> tuple[np.ndarray, bool, bool]:
    """Max-side PVO rule on a single 4-pixel block.

    Returns ``(new_block, did_embed, skipped)``.  If ``bit`` is None the
    block is NOT shifted -- instead it is marked as skipped so that
    blocks past the end of the payload are left unchanged.  This keeps
    extraction unambiguous (a shifted block is always the result of
    either embed-1 or shift-by-1, never "I had no bit left so I
    shifted anyway").
    """
    sorted_vals, argmax, _ = _sorted_with_argmax_argmin(block)
    u = int(sorted_vals[-1] - sorted_vals[-2])
    if u == 0:
        return block, False, True
    if block[argmax] >= 255:
        return block, False, True
    if bit is None:
        # No bit to embed and we choose not to shift -> skip the block.
        return block, False, True
    new_block = block.copy()
    if u == 1:
        new_block[argmax] = block[argmax] + bit
        if new_block[argmax] > 255:
            return block, False, True
        return new_block, True, False
    # u > 1 -> histogram shift so extraction can still separate
    # "originally u=1, bit=1 (u'=2)" from "originally u>=2 (u'>=3)".
    new_block[argmax] = block[argmax] + 1
    if new_block[argmax] > 255:
        return block, False, True
    return new_block, False, False


def _apply_min_rule(block: np.ndarray, bit: int | None) -> tuple[np.ndarray, bool, bool]:
    """Min-side PVO rule on a single 4-pixel block."""
    sorted_vals, _, argmin = _sorted_with_argmax_argmin(block)
    v = int(sorted_vals[1] - sorted_vals[0])
    if v == 0:
        return block, False, True
    if block[argmin] <= 0:
        return block, False, True
    if bit is None:
        return block, False, True
    new_block = block.copy()
    if v == 1:
        new_block[argmin] = block[argmin] - bit
        if new_block[argmin] < 0:
            return block, False, True
        return new_block, True, False
    new_block[argmin] = block[argmin] - 1
    if new_block[argmin] < 0:
        return block, False, True
    return new_block, False, False


def _invert_max_rule(block: np.ndarray) -> tuple[np.ndarray, int | None]:
    """Invert the max-side rule.  Returns ``(orig_block, bit_or_None)``.

    Caller must have already verified that the block is *not* in the
    skip set for this side.
    """
    sorted_vals, argmax, _ = _sorted_with_argmax_argmin(block)
    u = int(sorted_vals[-1] - sorted_vals[-2])
    orig = block.copy()
    if u == 1:
        # Original u was 1 (bit=0 case): nothing to undo.
        return orig, 0
    if u == 2:
        # Could be u=1 with bit=1, OR u=2 shifted from u=1 (same outcome
        # in terms of new values).  In PVO these ARE distinguishable
        # because the shift rule only applies when u >= 2 originally.
        # Since we skipped u==0 blocks, and the only way to end up with
        # u'==2 after embedding is (u=1, bit=1).  After a "shift" step
        # (u >= 2 originally) the output u' = u + 1 >= 3.  So u' == 2
        # unambiguously means "was 1, embedded a 1".
        orig[argmax] = block[argmax] - 1
        return orig, 1
    # u >= 3: histogram shift -- undo by subtracting 1.
    orig[argmax] = block[argmax] - 1
    return orig, None


def _invert_min_rule(block: np.ndarray) -> tuple[np.ndarray, int | None]:
    sorted_vals, _, argmin = _sorted_with_argmax_argmin(block)
    v = int(sorted_vals[1] - sorted_vals[0])
    orig = block.copy()
    if v == 1:
        return orig, 0
    if v == 2:
        orig[argmin] = block[argmin] + 1
        return orig, 1
    orig[argmin] = block[argmin] + 1
    return orig, None


def embed(
    image: np.ndarray, bits: np.ndarray
) -> tuple[np.ndarray, PVOSideInfo, int]:
    """Embed ``bits`` (1D uint8 0/1) via pairwise PVO.

    Returns ``(marked_image, side_info, n_embedded)``.

    Bits are consumed greedily -- first all max-side embeddings across
    the block grid, then all min-side embeddings on the (already
    max-modified) blocks.  This interleaving is important for
    reversibility: extraction must invert in reverse order.
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D")
    if image.dtype != np.uint8:
        # Allow int-typed covers (e.g. HSB plane after sharing) but
        # clamp at the end.
        pass
    if bits.ndim != 1:
        raise ValueError("bits must be 1D")

    blocks, grid = _block_view(image.astype(np.int64))
    n_blocks = blocks.shape[0]
    skipped_max = np.zeros(n_blocks, dtype=bool)
    skipped_min = np.zeros(n_blocks, dtype=bool)
    bit_idx = 0
    total = int(bits.size)

    # Pass 1: max side.
    n_max_embedded = 0
    for i in range(n_blocks):
        bit = int(bits[bit_idx]) if bit_idx < total else None
        new_block, did_embed, skipped = _apply_max_rule(blocks[i], bit)
        blocks[i] = new_block
        skipped_max[i] = skipped
        if did_embed:
            n_max_embedded += 1
            bit_idx += 1

    # Pass 2: min side on the (updated) blocks.
    n_min_embedded = 0
    for i in range(n_blocks):
        bit = int(bits[bit_idx]) if bit_idx < total else None
        new_block, did_embed, skipped = _apply_min_rule(blocks[i], bit)
        blocks[i] = new_block
        skipped_min[i] = skipped
        if did_embed:
            n_min_embedded += 1
            bit_idx += 1

    marked = np.clip(_unblock(blocks, grid), 0, 255).astype(np.uint8)
    side = PVOSideInfo(
        shape=image.shape,
        n_blocks=n_blocks,
        skipped_max=skipped_max,
        skipped_min=skipped_min,
        n_embedded_max=n_max_embedded,
        n_embedded_min=n_min_embedded,
    )
    return marked, side, n_max_embedded + n_min_embedded


def extract(
    marked: np.ndarray, side: PVOSideInfo
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`embed`.

    Returns ``(recovered_image, bits)`` where ``bits`` has length
    ``side.n_embedded_max + side.n_embedded_min`` in the embedding
    order (all max bits followed by all min bits).
    """
    if marked.shape != side.shape:
        raise ValueError(f"shape mismatch: {marked.shape} vs {side.shape}")
    blocks, grid = _block_view(marked.astype(np.int64))
    n_blocks = blocks.shape[0]
    assert n_blocks == side.n_blocks, (n_blocks, side.n_blocks)

    min_bits_reverse: list[int] = []
    # Invert min side first (reverse order within the pass).
    for i in range(n_blocks - 1, -1, -1):
        if side.skipped_min[i]:
            continue
        orig, bit = _invert_min_rule(blocks[i])
        blocks[i] = orig
        if bit is not None:
            min_bits_reverse.append(bit)
    min_bits = list(reversed(min_bits_reverse))

    max_bits_reverse: list[int] = []
    for i in range(n_blocks - 1, -1, -1):
        if side.skipped_max[i]:
            continue
        orig, bit = _invert_max_rule(blocks[i])
        blocks[i] = orig
        if bit is not None:
            max_bits_reverse.append(bit)
    max_bits = list(reversed(max_bits_reverse))

    if len(max_bits) != side.n_embedded_max:
        raise RuntimeError(
            f"max-side bit count mismatch: extracted {len(max_bits)}, "
            f"expected {side.n_embedded_max}"
        )
    if len(min_bits) != side.n_embedded_min:
        raise RuntimeError(
            f"min-side bit count mismatch: extracted {len(min_bits)}, "
            f"expected {side.n_embedded_min}"
        )

    recovered = np.clip(_unblock(blocks, grid), 0, 255).astype(np.uint8)
    out_bits = np.array(max_bits + min_bits, dtype=np.uint8)
    return recovered, out_bits


def capacity_estimate(image: np.ndarray) -> dict[str, int]:
    """Quickly estimate how many bits can be embedded in one PVO layer.

    This is the number of 2x2 blocks whose (sorted) max-max-next-max
    difference is exactly 1, plus the same count for the min side.
    """
    blocks, _ = _block_view(image.astype(np.int64))
    sorted_blocks = np.sort(blocks, axis=1)
    u = sorted_blocks[:, -1] - sorted_blocks[:, -2]
    v = sorted_blocks[:, 1] - sorted_blocks[:, 0]
    # Overflow / underflow safety (max=255 or min=0).
    safe_max = (sorted_blocks[:, -1] < 255)
    safe_min = (sorted_blocks[:, 0] > 0)
    return {
        "max_side": int(np.sum((u == 1) & safe_max)),
        "min_side": int(np.sum((v == 1) & safe_min)),
        "n_blocks": int(blocks.shape[0]),
    }
