"""Additive secret sharing driven by a ChaCha20 keystream.

Modernization of ``rrwei_sm.secret_sharing`` that uses ChaCha20 for the
per-pixel random masks instead of NumPy's default PRNG.  The math is
unchanged:

    cover = sum(shares)   (exact, over Z)
    HSB split is modular in 2^n_lsb; carry absorbed into HSB share 0

so PEE / PVO / STDM continue to work as linear operations on the
combined plaintext view.

Key benefits over the legacy implementation
--------------------------------------------
* **Cryptographic randomness.**  Each share is indistinguishable from
  uniform random bytes under IND-CPA of ChaCha20 (a 2-of-2 additive
  share with a keystream-derived mask is a standard one-time pad).
* **Key derivation.**  Integer seeds are passed through HKDF-SHA256,
  so weak seeds cannot leak structure; two different integer seeds
  produce statistically independent keystreams.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crypto_scrambler import (
    CryptoScramblerKey,
    chacha20_keystream,
    derive_key,
)

__all__ = [
    "SharingResult",
    "additive_share_image",
    "additive_combine_shares",
    "split_hsb_lsb",
    "recombine_hsb_lsb",
]


_INFO_MASK = b"rrwei_sm_modern/additive_mask/v1"


@dataclass
class SharingResult:
    """Container for the additive shares of one cover image."""

    shares: list[np.ndarray]        # length >= 2, each int32 with same shape as cover
    n_lsb: int
    seed: int                       # the integer seed actually used


def _random_mask(shape: tuple[int, int], seed: int, stream_idx: int) -> np.ndarray:
    """Return a uint8 mask of ``shape`` keyed by (seed, stream_idx)."""
    # stream_idx makes successive shares use disjoint subkeys so they
    # are statistically independent even for the same seed.
    info = _INFO_MASK + f"/stream={stream_idx}".encode("ascii")
    key = derive_key(seed, info=info)
    n_bytes = int(np.prod(shape))
    raw = chacha20_keystream(n_bytes, key)
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()


def split_hsb_lsb(image: np.ndarray, n_lsb: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (HSB, LSB) planes.  Matches the legacy semantics."""
    image = image.astype(np.int64)
    lsb = image & ((1 << n_lsb) - 1)
    hsb = image >> n_lsb
    return hsb, lsb


def recombine_hsb_lsb(hsb: np.ndarray, lsb: np.ndarray, n_lsb: int) -> np.ndarray:
    """Inverse of :func:`split_hsb_lsb`."""
    return (hsb.astype(np.int64) << n_lsb) | lsb.astype(np.int64)


def additive_share_image(
    image: np.ndarray,
    *,
    n_parties: int = 2,
    n_lsb: int = 3,
    seed: int = 0,
) -> SharingResult:
    """Split ``image`` into ``n_parties`` additive shares over Z.

    The shares are int32 arrays whose **exact** sum equals the input
    image.  No share alone leaks information about the cover (modular
    one-time-pad argument, IND-CPA of ChaCha20).

    The LSB split is modular in ``2**n_lsb``; the carry that this
    induces on the HSB plane is absorbed into share ``0`` so the pure
    additive identity ``sum(shares) == cover`` holds in Z.
    """
    if n_parties < 2:
        raise ValueError("n_parties must be >= 2")
    if image.dtype != np.uint8:
        raise TypeError("image must be uint8")
    if image.ndim != 2:
        raise ValueError("Only 2D grayscale images supported")

    hsb, lsb = split_hsb_lsb(image, n_lsb)
    mod = 1 << n_lsb

    # --- LSB sharing (modular in 2^n_lsb) ---------------------------
    lsb_shares = np.zeros((n_parties,) + image.shape, dtype=np.int64)
    running_sum_lsb = np.zeros_like(lsb, dtype=np.int64)
    for p in range(n_parties - 1):
        mask = _random_mask(image.shape, seed=seed, stream_idx=p).astype(np.int64) % mod
        lsb_shares[p] = mask
        running_sum_lsb += mask
    lsb_shares[-1] = (lsb - running_sum_lsb) % mod

    # Carry of the LSB sum that crosses into HSB:
    #   lsb_true = lsb_shares.sum() % mod = lsb (by construction)
    # but lsb_shares.sum() (over Z) may exceed lsb by k*mod for some
    # integer k >= 0; absorb that into HSB share 0 below.
    carry_lsb_sum = lsb_shares.sum(axis=0) - lsb  # multiple of mod

    # --- HSB sharing (plain additive over Z) ------------------------
    hsb_shares = np.zeros_like(lsb_shares)
    running_sum_hsb = np.zeros_like(hsb, dtype=np.int64)
    for p in range(n_parties - 1):
        mask_stream_idx = p + n_parties  # disjoint from LSB streams
        mask = _random_mask(image.shape, seed=seed, stream_idx=mask_stream_idx).astype(np.int64)
        hsb_shares[p] = mask
        running_sum_hsb += mask
    hsb_shares[-1] = hsb - running_sum_hsb

    # Absorb LSB carry into HSB share 0: this subtraction is scaled by
    # ``mod`` in pixel space, because a unit-LSB carry equals mod in
    # the combined integer.
    carry_k = (carry_lsb_sum // mod).astype(np.int64)
    hsb_shares[0] -= carry_k

    # --- Recombine each party's HSB + LSB --------------------------
    shares: list[np.ndarray] = []
    for p in range(n_parties):
        combined = recombine_hsb_lsb(hsb_shares[p], lsb_shares[p], n_lsb)
        shares.append(combined.astype(np.int32))

    return SharingResult(shares=shares, n_lsb=n_lsb, seed=seed)


def additive_combine_shares(shares: list[np.ndarray]) -> np.ndarray:
    """Exact sum of ``shares`` as int64."""
    if len(shares) < 2:
        raise ValueError("need >= 2 shares")
    acc = shares[0].astype(np.int64)
    for s in shares[1:]:
        acc = acc + s.astype(np.int64)
    return acc
