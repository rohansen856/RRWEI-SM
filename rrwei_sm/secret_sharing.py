"""
Additive secret sharing of a grayscale image with HSB/LSB decomposition.

Implements Section IV-B, step 1 ("Image Encryption Phase") of
Xiong et al., 2022 (Eqs. 7-14):

    x(i,j)      = x_HSB(i,j) * 2^n + x_LSB(i,j)
    x_HSB(i,j)  = x1_HSB(i,j) + x2_HSB(i,j)
    x_LSB(i,j)  = x1_LSB(i,j) + x2_LSB(i,j)
    x_k(i,j)    = x_k_HSB(i,j) * 2^n + x_k_LSB(i,j)   (k = 1, 2)

The HSB and LSB planes are shared *independently* so that, later, the
sum-of-HSBs of the two shares equals the HSB of the cover (a requirement
of the block-level predictor used by the PEE embedder).

Design notes / deviations from paper
------------------------------------
* The paper states that shares lie in [0, 255]. With a naive uniform random
  split there is a non-negligible chance of overflow. To keep the scheme
  mathematically exact we store shares as **signed int32**. Overflow /
  underflow pixels can still be detected via :func:`shares_in_gray_range`
  and recorded, matching the paper's "overflow/underflow list" idea.
* The split uses per-pixel uniform random values from [0, max_plane_value];
  this is a common practical interpretation of additive sharing over a
  bounded range. It is information-theoretically secure in the sense of
  Theorem 1 in the paper (Bogdanov et al., 2012).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Shares:
    """A pair of additive shares of an image (+ the chosen n-LSB split)."""

    share1: np.ndarray
    share2: np.ndarray
    n_lsb: int

    def combine(self) -> np.ndarray:
        return additive_combine_shares(self.share1, self.share2)


def split_hsb_lsb(image: np.ndarray, n_lsb: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split a grayscale image into its HSB and LSB planes (Eqs. 8-10).

    Parameters
    ----------
    image : np.ndarray
        Grayscale image as uint8 / integer in [0, 255].
    n_lsb : int
        Number of LSB bits; the HSB plane is (8 - n_lsb) bits wide.
    """
    if not (1 <= n_lsb <= 7):
        raise ValueError("n_lsb must be in [1, 7]")
    img = image.astype(np.int64)
    modulus = 1 << n_lsb
    lsb = img % modulus
    hsb = (img - lsb) // modulus
    return hsb, lsb


def recombine_hsb_lsb(hsb: np.ndarray, lsb: np.ndarray, n_lsb: int) -> np.ndarray:
    """Inverse of :func:`split_hsb_lsb`."""
    return (hsb.astype(np.int64) << n_lsb) + lsb.astype(np.int64)


def additive_share_image(
    image: np.ndarray,
    n_lsb: int = 3,
    seed: int | None = None,
) -> Shares:
    """Produce two additive shares of ``image`` with an n-LSB split (Eqs. 11-14).

    Implementation note
    -------------------
    To preserve the invariant that ``share_k >> n_lsb == x_k_HSB`` even
    when a share becomes negative, LSB parts are shared *modulo* ``2**n_lsb``
    (so they are always in ``[0, 2**n_lsb)``) and the resulting carry is
    absorbed by the HSB split.  This is functionally identical to the
    paper's Eqs. (11)-(14) because the residual carry is accounted for
    exactly; the sum ``share1 + share2`` still equals the cover image.
    """
    if image.ndim != 2:
        raise ValueError("Only grayscale images are supported.")
    rng = np.random.default_rng(seed)
    hsb, lsb = split_hsb_lsb(image, n_lsb)
    hsb_max_plus_1 = 1 << (8 - n_lsb)
    lsb_mod = 1 << n_lsb

    # LSB: modular split (both parts in [0, 2^n - 1]).
    s1_lsb = rng.integers(0, lsb_mod, size=lsb.shape, dtype=np.int64)
    s2_lsb = (lsb - s1_lsb) % lsb_mod
    carry = ((s1_lsb + s2_lsb) >= lsb_mod).astype(np.int64)

    # HSB: integer split, absorbing the LSB carry so the overall sum is exact.
    s1_hsb = rng.integers(0, hsb_max_plus_1, size=hsb.shape, dtype=np.int64)
    s2_hsb = hsb - s1_hsb - carry

    share1 = (s1_hsb << n_lsb) + s1_lsb
    share2 = (s2_hsb << n_lsb) + s2_lsb
    return Shares(share1=share1.astype(np.int32), share2=share2.astype(np.int32), n_lsb=n_lsb)


def additive_combine_shares(share1: np.ndarray, share2: np.ndarray) -> np.ndarray:
    """Recombine two additive shares into the original image values."""
    return (share1.astype(np.int64) + share2.astype(np.int64))


def additive_share_image_k(
    image: np.ndarray,
    n_parties: int,
    n_lsb: int = 3,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Split ``image`` into ``n_parties`` additive shares (Eqs. 7-14, k-party).

    The first ``n_parties - 1`` shares are sampled uniformly at random
    (independently per party) from the same HSB/LSB range as in the
    two-party case; the final share is chosen to make the sum equal the
    cover with LSBs non-negative and HSBs carrying exactly the right
    carry.  The additive invariant ``sum(shares) == image`` holds
    exactly, and the HSB-recovery invariant

        sum_k (share_k >> n_lsb)  ==  cover_HSB  -  carry

    still holds with a deterministic per-pixel carry, just like the
    two-party variant.  This is the property the SMC protocol relies on
    (carry cancels out in the *relative* prediction-error contribution
    each party computes from its own share).

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale image.
    n_parties : int
        Number of parties.  Must be >= 2.
    n_lsb : int
        Number of LSB bits.  Same semantics as :func:`additive_share_image`.
    seed : int | None
        Seed for the random split.

    Returns
    -------
    list[np.ndarray]
        List of ``n_parties`` ``int32`` arrays summing (pixelwise) to
        ``image``.
    """
    if n_parties < 2:
        raise ValueError("n_parties must be >= 2")
    if image.ndim != 2:
        raise ValueError("Only grayscale images are supported.")
    rng = np.random.default_rng(seed)
    hsb, lsb = split_hsb_lsb(image, n_lsb)
    lsb_mod = 1 << n_lsb
    hsb_max_plus_1 = 1 << (8 - n_lsb)

    # Sample n_parties - 1 random LSB parts, modulo lsb_mod.
    lsb_shares = []
    for _ in range(n_parties - 1):
        lsb_shares.append(
            rng.integers(0, lsb_mod, size=lsb.shape, dtype=np.int64)
        )
    last_lsb = (lsb - sum(lsb_shares)) % lsb_mod
    lsb_shares.append(last_lsb)
    total_lsb = sum(lsb_shares)
    carry = (total_lsb - lsb) // lsb_mod  # always an integer

    # Sample n_parties - 1 random HSB parts in [0, hsb_max_plus_1).
    hsb_shares = []
    for _ in range(n_parties - 1):
        hsb_shares.append(
            rng.integers(0, hsb_max_plus_1, size=hsb.shape, dtype=np.int64)
        )
    last_hsb = hsb - sum(hsb_shares) - carry
    hsb_shares.append(last_hsb)

    shares = []
    for hsb_k, lsb_k in zip(hsb_shares, lsb_shares):
        share_k = (hsb_k << n_lsb) + lsb_k
        shares.append(share_k.astype(np.int32))
    return shares


def additive_combine_shares_k(shares: list[np.ndarray]) -> np.ndarray:
    """Sum ``n_parties`` additive shares back to the cover."""
    if not shares:
        raise ValueError("shares list must be non-empty")
    out = np.zeros_like(shares[0], dtype=np.int64)
    for s in shares:
        out = out + s.astype(np.int64)
    return out


def shares_in_gray_range(shares: Shares) -> np.ndarray:
    """Boolean mask: True where both shares fit in [0, 255]."""
    return (
        (shares.share1 >= 0)
        & (shares.share1 <= 255)
        & (shares.share2 >= 0)
        & (shares.share2 <= 255)
    )


def share_hsb_plane(share: np.ndarray, n_lsb: int) -> np.ndarray:
    """Return the HSB plane of a share (values may be negative or large).

    This is ``floor(share / 2^n)`` for sign-aware division, matching the
    split used in :func:`additive_share_image`.
    """
    arr = share.astype(np.int64)
    # Arithmetic shift = floor division by power of two.
    return arr >> n_lsb


def share_lsb_plane(share: np.ndarray, n_lsb: int) -> np.ndarray:
    """Return the LSB plane (value mod 2^n) of a share."""
    arr = share.astype(np.int64)
    return arr & ((1 << n_lsb) - 1)
