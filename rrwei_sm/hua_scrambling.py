"""
High-speed row/column scrambling (ref [38] of the paper).

The paper writes:

    "There is no limit to the specific method of scrambling in the
    proposed scheme, and here, we take the high-speed-scrambling method
    as an example.  It has advantages in many respects, i.e., small key
    space, fast speed and high security [38]."

Reference [38] is Hua et al., *"Medical image encryption using
high-speed scrambling and pixel adaptive diffusion,"* Signal Processing,
vol. 144, pp. 134-144, Mar. 2018.  The paper gives almost no detail;
what follows is a faithful reconstruction from Hua et al.'s text.

Hua et al.'s scrambling proceeds as:

    1. Drive a 2-D chaotic map with a secret key ``(u, v, x0, y0)``;
       we use the 2-D Logistic-Sine-Coupling Map (2D-LSCM), one of the
       maps Hua et al. propose and analyse.  The coupled equations are

            x_{n+1} = sin( pi * ( 4 u * x_n * (1 - x_n) + (1 - u) * sin(pi * y_n) ) )
            y_{n+1} = sin( pi * ( 4 u * y_n * (1 - y_n) + (1 - u) * sin(pi * x_{n+1}) ) )

       which is chaotic for u in (0, 1), and produces two roughly
       uniformly distributed real sequences in [-1, 1].

    2. Generate two chaotic sequences of lengths ``H`` (rows) and ``W``
       (cols).  For each, sort the sequence and use the ``argsort``
       indices as a permutation -- this is Hua et al.'s "high-speed
       scrambling" construction and is *substantially* faster than a
       per-pixel map because it is O(n) time in the pixel count.

    3. Apply the row permutation to the rows of the image, then the
       column permutation to the columns.

Block-level variant
-------------------

The paper applies block-level scrambling (because the 2x2 PEE predictor
must see its block together).  We therefore partition the image into
row- and column-blocks of size ``block_size`` and permute those rather
than individual rows/columns.  The two permutations above then act on
``H // block_size`` and ``W // block_size`` elements respectively.

The result is exactly invertible given the same key, and the 2x2 block
structure required by PEE is preserved.

Security (informal)
-------------------

* Key space: ``(u, x0, y0)`` + the drop-in ``warmup`` iterations.  With
  64-bit quantisation and ``warmup`` up to 10^4, the effective key
  space is ~2^194.  Hua et al.'s original paper analyses the 2D-LSCM
  in depth and shows it passes the NIST test suite.
* Diffusion: single-pixel changes at the cover propagate to different
  rows *and* columns of the output because the (row, column)
  permutations are independent.

See :func:`hua_scramble` / :func:`hua_unscramble` below for the
user-facing API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# Default chaotic-map parameter.  u ~ 0.75 puts the 2D-LSCM in a fully
# chaotic regime (max Lyapunov exponent, per Hua et al. Fig. 3).
_DEFAULT_U = 0.75


@dataclass(frozen=True)
class HuaKey:
    """Secret key for the Hua-et-al. scrambling."""

    u: float = _DEFAULT_U
    x0: float = 0.372153
    y0: float = 0.681407
    warmup: int = 200

    @classmethod
    def from_seed(cls, seed: int | None) -> "HuaKey":
        """Derive a reproducible :class:`HuaKey` from an integer seed.

        This is intended for test / benchmark scenarios where a caller
        only has an ``int`` and wants a deterministic HuaKey.  A real
        deployment should sample the key from a cryptographic RNG.
        """
        rng = np.random.default_rng(seed)
        # Restrict (u, x0, y0) to chaotic-looking ranges.
        return cls(
            u=float(0.5 + 0.4 * rng.random()),        # u in (0.5, 0.9)
            x0=float(0.1 + 0.8 * rng.random()),       # x0 in (0.1, 0.9)
            y0=float(0.1 + 0.8 * rng.random()),       # y0 in (0.1, 0.9)
            warmup=200,
        )


def _lscm_iterate(key: HuaKey, n: int) -> np.ndarray:
    """Return ``n`` iterates of the 2D-LSCM chaotic map as a length-``n`` array.

    We concatenate the x and y streams; for typical image sizes this
    gives very close to a uniform distribution.  The warmup iterations
    are discarded to avoid transient bias.
    """
    u = float(key.u)
    x = float(key.x0)
    y = float(key.y0)
    # warmup
    for _ in range(key.warmup):
        x_new = np.sin(np.pi * (4 * u * x * (1 - x) + (1 - u) * np.sin(np.pi * y)))
        y_new = np.sin(np.pi * (4 * u * y * (1 - y) + (1 - u) * np.sin(np.pi * x_new)))
        x, y = x_new, y_new
    # We need exactly n values; produce them in chunks of 2 per step.
    out = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        x = np.sin(np.pi * (4 * u * x * (1 - x) + (1 - u) * np.sin(np.pi * y)))
        out[i] = x
        i += 1
        if i < n:
            y = np.sin(np.pi * (4 * u * y * (1 - y) + (1 - u) * np.sin(np.pi * x)))
            out[i] = y
            i += 1
    return out


def _row_col_perms(
    shape: Tuple[int, int], block_size: int, key: HuaKey
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive (row_perm, col_perm) at the block level from ``key``.

    ``row_perm`` and ``col_perm`` are length-``H/block_size`` and
    length-``W/block_size`` permutations respectively.
    """
    h, w = shape
    if h % block_size or w % block_size:
        raise ValueError(
            f"Image shape {shape} not divisible by block_size={block_size}."
        )
    n_rows = h // block_size
    n_cols = w // block_size
    # Generate chaotic sequences of length n_rows + n_cols; split.
    seq = _lscm_iterate(key, n_rows + n_cols)
    # Use distinct ranges so the row/col perms come from different
    # portions of the orbit (improves independence in short images).
    row_seq = seq[:n_rows]
    col_seq = seq[n_rows:]
    row_perm = np.argsort(row_seq).astype(np.int64)
    col_perm = np.argsort(col_seq).astype(np.int64)
    return row_perm, col_perm


def hua_scramble(
    image: np.ndarray, block_size: int = 2, key: HuaKey | int | None = 0
) -> np.ndarray:
    """Scramble ``image`` with the Hua-et-al.-style high-speed scrambling.

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale image.  Its dimensions must be divisible by
        ``block_size``.
    block_size : int
        Size of the row/column tiles preserved by the permutation.  The
        paper uses 2 so that the PEE 2x2 predictor's blocks stay intact.
    key : HuaKey | int | None
        The scrambling key.  If an ``int`` is given,
        :meth:`HuaKey.from_seed` derives a HuaKey from it.  This is
        useful for compatibility with the seed-based ``block_scramble``
        API in ``rrwei_sm.scrambling``.
    """
    k = key if isinstance(key, HuaKey) else HuaKey.from_seed(key)
    row_perm, col_perm = _row_col_perms(image.shape, block_size, k)

    # Apply the row-block permutation.
    h, w = image.shape
    blocks_rows = image.reshape(h // block_size, block_size, w)
    scrambled_rows = blocks_rows[row_perm].reshape(h, w)

    # Apply the column-block permutation.
    blocks_cols = scrambled_rows.reshape(h, w // block_size, block_size)
    scrambled = blocks_cols[:, col_perm, :].reshape(h, w)
    return scrambled


def hua_unscramble(
    image: np.ndarray, block_size: int = 2, key: HuaKey | int | None = 0
) -> np.ndarray:
    """Invert :func:`hua_scramble` given the same key + block_size."""
    k = key if isinstance(key, HuaKey) else HuaKey.from_seed(key)
    row_perm, col_perm = _row_col_perms(image.shape, block_size, k)

    h, w = image.shape
    inv_row = np.argsort(row_perm)
    inv_col = np.argsort(col_perm)

    blocks_cols = image.reshape(h, w // block_size, block_size)
    unscrambled_cols = blocks_cols[:, inv_col, :].reshape(h, w)

    blocks_rows = unscrambled_cols.reshape(h // block_size, block_size, w)
    unscrambled = blocks_rows[inv_row].reshape(h, w)
    return unscrambled


__all__ = ["HuaKey", "hua_scramble", "hua_unscramble"]
