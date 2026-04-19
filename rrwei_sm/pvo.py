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

These go through the rANS coder (:mod:`rrwei_sm.coding`) in the
orchestrator stage, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np

__all__ = [
    "PVOSideInfo",
    "embed",
    "extract",
    "capacity_estimate",
]


_BLOCK = 2  # 2x2 blocks


# ---------------------------------------------------------------------------
# Numba-accelerated hot loops.
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _pvo_embed_inner(
    blocks: np.ndarray,      # shape (n_blocks, 4) int64
    bits: np.ndarray,        # shape (B,) int8 (0/1) or uint8
    n_bits: int,
    do_max: bool,            # True -> max-side pass, False -> min-side pass
) -> tuple:
    """JIT-compiled PVO inner loop for one pass (max or min).

    Updates ``blocks`` in-place and returns (n_embedded, skipped_mask).
    """
    n_blocks = blocks.shape[0]
    skipped = np.ones(n_blocks, dtype=np.bool_)
    bit_idx = 0
    n_embedded = 0

    for i in range(n_blocks):
        # 4-way insertion sort over the block's pixel values to find max / min.
        a = blocks[i, 0]
        b = blocks[i, 1]
        c = blocks[i, 2]
        d = blocks[i, 3]
        # Find argmax / argmin with ties broken by position.
        argmax = 0
        max_val = a
        if b > max_val:
            max_val = b
            argmax = 1
        if c > max_val:
            max_val = c
            argmax = 2
        if d > max_val:
            max_val = d
            argmax = 3
        argmin = 0
        min_val = a
        if b < min_val:
            min_val = b
            argmin = 1
        if c < min_val:
            min_val = c
            argmin = 2
        if d < min_val:
            min_val = d
            argmin = 3

        if do_max:
            # Find second-max (excluding argmax position).
            second = -10**9
            for j in range(4):
                if j != argmax and blocks[i, j] > second:
                    second = blocks[i, j]
            u = blocks[i, argmax] - second
            if u == 0 or blocks[i, argmax] >= 255:
                continue
            if bit_idx >= n_bits:
                continue
            if u == 1:
                bit = bits[bit_idx]
                new_val = blocks[i, argmax] + bit
                if new_val > 255:
                    continue
                blocks[i, argmax] = new_val
                bit_idx += 1
                n_embedded += 1
                skipped[i] = False
            else:
                new_val = blocks[i, argmax] + 1
                if new_val > 255:
                    continue
                blocks[i, argmax] = new_val
                skipped[i] = False
        else:
            # Find second-min (excluding argmin position).
            second = 10**9
            for j in range(4):
                if j != argmin and blocks[i, j] < second:
                    second = blocks[i, j]
            v = second - blocks[i, argmin]
            if v == 0 or blocks[i, argmin] <= 0:
                continue
            if bit_idx >= n_bits:
                continue
            if v == 1:
                bit = bits[bit_idx]
                new_val = blocks[i, argmin] - bit
                if new_val < 0:
                    continue
                blocks[i, argmin] = new_val
                bit_idx += 1
                n_embedded += 1
                skipped[i] = False
            else:
                new_val = blocks[i, argmin] - 1
                if new_val < 0:
                    continue
                blocks[i, argmin] = new_val
                skipped[i] = False

    return n_embedded, skipped


@numba.njit(cache=True)
def _pvo_extract_inner(
    blocks: np.ndarray,       # shape (n_blocks, 4) int64
    skipped: np.ndarray,      # bool per block
    do_max: bool,
) -> np.ndarray:
    """JIT-compiled inverse of one PVO pass.  Updates ``blocks`` in place.

    Returns an int64 array of extracted bits, in ascending block order.
    """
    n_blocks = blocks.shape[0]
    out = np.empty(n_blocks, dtype=np.int64)
    n_out = 0

    # Traverse in reverse (mirror of embed pass) to match the forward
    # order of the embedded bits.
    for ii in range(n_blocks - 1, -1, -1):
        i = ii
        if skipped[i]:
            continue
        argmax = 0
        max_val = blocks[i, 0]
        for j in range(1, 4):
            if blocks[i, j] > max_val:
                max_val = blocks[i, j]
                argmax = j
        argmin = 0
        min_val = blocks[i, 0]
        for j in range(1, 4):
            if blocks[i, j] < min_val:
                min_val = blocks[i, j]
                argmin = j
        if do_max:
            second = -10**9
            for j in range(4):
                if j != argmax and blocks[i, j] > second:
                    second = blocks[i, j]
            u = blocks[i, argmax] - second
            if u == 1:
                out[n_out] = 0
                n_out += 1
            elif u == 2:
                blocks[i, argmax] -= 1
                out[n_out] = 1
                n_out += 1
            else:
                blocks[i, argmax] -= 1
        else:
            second = 10**9
            for j in range(4):
                if j != argmin and blocks[i, j] < second:
                    second = blocks[i, j]
            v = second - blocks[i, argmin]
            if v == 1:
                out[n_out] = 0
                n_out += 1
            elif v == 2:
                blocks[i, argmin] += 1
                out[n_out] = 1
                n_out += 1
            else:
                blocks[i, argmin] += 1

    # Reverse so that out[0..n_out) is in forward embed order.
    final = np.empty(n_out, dtype=np.int64)
    for j in range(n_out):
        final[j] = out[n_out - 1 - j]
    return final


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
    """Embed ``bits`` via pairwise PVO (Numba-JIT accelerated).

    Returns ``(marked_image, side_info, n_embedded)``.
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D")
    if bits.ndim != 1:
        raise ValueError("bits must be 1D")

    blocks, grid = _block_view(image.astype(np.int64))
    bits_i8 = np.ascontiguousarray(bits.astype(np.int8))

    # Pass 1: max side consumes bits[0..); the JIT function returns how
    # many bits it actually embedded.
    n_max_embedded, skipped_max = _pvo_embed_inner(
        blocks, bits_i8, bits_i8.size, True,
    )
    # Pass 2: min side consumes bits[n_max_embedded..).
    remaining = bits_i8[n_max_embedded:]
    n_min_embedded, skipped_min = _pvo_embed_inner(
        blocks, remaining, remaining.size, False,
    )

    marked = np.clip(_unblock(blocks, grid), 0, 255).astype(np.uint8)
    side = PVOSideInfo(
        shape=image.shape,
        n_blocks=blocks.shape[0],
        skipped_max=np.asarray(skipped_max, dtype=bool),
        skipped_min=np.asarray(skipped_min, dtype=bool),
        n_embedded_max=int(n_max_embedded),
        n_embedded_min=int(n_min_embedded),
    )
    return marked, side, int(n_max_embedded + n_min_embedded)


def extract(
    marked: np.ndarray, side: PVOSideInfo
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`embed` (Numba-JIT accelerated)."""
    if marked.shape != side.shape:
        raise ValueError(f"shape mismatch: {marked.shape} vs {side.shape}")
    blocks, grid = _block_view(marked.astype(np.int64))
    assert blocks.shape[0] == side.n_blocks

    # Invert passes in reverse order: min first, then max.
    min_bits = _pvo_extract_inner(blocks, side.skipped_min, False)
    max_bits = _pvo_extract_inner(blocks, side.skipped_max, True)

    if min_bits.size != side.n_embedded_min:
        raise RuntimeError(
            f"min-side bit count mismatch: extracted {min_bits.size}, "
            f"expected {side.n_embedded_min}"
        )
    if max_bits.size != side.n_embedded_max:
        raise RuntimeError(
            f"max-side bit count mismatch: extracted {max_bits.size}, "
            f"expected {side.n_embedded_max}"
        )

    recovered = np.clip(_unblock(blocks, grid), 0, 255).astype(np.uint8)
    out_bits = np.concatenate(
        [max_bits.astype(np.uint8), min_bits.astype(np.uint8)]
    )
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
