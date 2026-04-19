"""Replicated secret sharing with a ``(k, n)`` reconstruction threshold.

The legacy scheme used k-of-k additive sharing: lose any share, lose
the cover.  This module provides the ``k``-of-``n`` generalization by
replicated secret sharing (Cramer-Damgård-Ishai 2005), which is:

* **Additive on Z** -- so PEE / PVO / STDM stay linear over the
  combined view and the SMC contribution identity still holds.
* **Threshold**: any ``k`` parties can reconstruct; fewer than ``k``
  cannot distinguish the cover from uniform.

Construction
------------
Let ``p = k - 1`` be the privacy threshold and ``m = C(n, p)`` the
number of subsets of parties of size ``p``.  Generate ``m`` random
masks ``r_S`` (one per size-``p`` subset ``S``) such that
``sum_S r_S (mod 2**32) == cover``.

Party ``i`` holds *every* ``r_S`` with ``i not in S``.  Each party
therefore stores ``C(n-1, p)`` mask pieces per pixel.

Reconstruction
--------------
Any subset of ``k = p + 1`` parties covers every ``r_S`` (because for
any size-``p`` ``S``, a set of size ``p + 1`` has at least one member
outside ``S``).  Summing the union of their mask pieces (deduplicated)
recovers the cover.

Embedding (PEE owner-writes-a-delta pattern)
--------------------------------------------
The embedding modifies one party's share.  In RSS, that party's
``C(n-1, p)`` mask pieces all get a candidate pool; we update
exactly one arbitrarily-chosen piece (the first in ``party_masks[i]``)
by the full delta.  Any ``k`` reconstructing parties still sum to the
marked cover.

Notes
-----
* For ``(k, n) = (2, 2)`` this degenerates to two-party additive
  sharing with a single shared mask per pixel.
* For ``(k, n) = (2, 3)`` the default: ``m = 3`` masks; each party
  holds 2 of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np

from .crypto_scrambler import chacha20_keystream, derive_key

__all__ = [
    "ReplicatedShares",
    "share",
    "combine",
    "apply_owner_delta",
]

_INFO_RSS = b"rrwei_sm_modern/rss/mask/v1"


@dataclass
class ReplicatedShares:
    """All data needed to run a ``(k, n)`` RSS session.

    ``masks`` is a dict mapping a size-``p`` tuple ``S`` (party indices
    in ascending order) to the int64 mask array ``r_S``.  The total is
    ``sum(r_S) == cover``.

    ``party_masks[i]`` is the list of subsets ``S`` whose mask party
    ``i`` holds.
    """

    k: int
    n: int
    n_lsb: int
    seed: int
    masks: dict[tuple[int, ...], np.ndarray]
    party_masks: list[list[tuple[int, ...]]]


def _enumerate_subsets(n: int, size: int) -> list[tuple[int, ...]]:
    return [tuple(s) for s in combinations(range(n), size)]


def _mask_from_keystream(shape, seed: int, tag: bytes) -> np.ndarray:
    key = derive_key(seed, info=_INFO_RSS + b"/" + tag)
    n_bytes = int(np.prod(shape)) * 4
    raw = chacha20_keystream(n_bytes, key)
    return np.frombuffer(raw, dtype=np.int32).reshape(shape).astype(np.int64).copy()


def share(
    cover: np.ndarray,
    *,
    k: int = 2,
    n: int = 3,
    n_lsb: int = 3,
    seed: int = 0,
) -> ReplicatedShares:
    """Create ``(k, n)`` replicated shares of ``cover`` (uint8, 2D)."""
    if not (2 <= k <= n):
        raise ValueError(f"require 2 <= k <= n, got k={k}, n={n}")
    if cover.dtype != np.uint8:
        raise TypeError("cover must be uint8")
    if cover.ndim != 2:
        raise ValueError("cover must be 2D")

    p = k - 1
    subsets = _enumerate_subsets(n, p)  # length C(n, p)
    m = len(subsets)

    # Generate m - 1 random masks; the last absorbs the rest so the sum equals cover.
    masks: dict[tuple[int, ...], np.ndarray] = {}
    acc = np.zeros(cover.shape, dtype=np.int64)
    for S in subsets[:-1]:
        tag = ",".join(map(str, S)).encode("ascii")
        r = _mask_from_keystream(cover.shape, seed, tag)
        masks[S] = r
        acc += r
    masks[subsets[-1]] = cover.astype(np.int64) - acc
    assert sum(m_.sum() for m_ in masks.values()) == int(cover.sum()), "mask invariant"

    party_masks: list[list[tuple[int, ...]]] = []
    for i in range(n):
        party_masks.append([S for S in subsets if i not in S])
    # Each party holds C(n-1, p) masks.
    expected = comb(n - 1, p)
    assert all(len(ms) == expected for ms in party_masks), "RSS distribution invariant"

    return ReplicatedShares(
        k=k, n=n, n_lsb=n_lsb, seed=seed, masks=masks, party_masks=party_masks,
    )


def combine(shares: ReplicatedShares, participating_parties: list[int]) -> np.ndarray:
    """Reconstruct the cover from ``k`` or more cooperating parties.

    Uses only the mask pieces held by the provided parties; if the set
    is too small to cover every subset ``S``, raises ``ValueError``.
    """
    if len(set(participating_parties)) < shares.k:
        raise ValueError(
            f"need >= {shares.k} cooperating parties, got {len(participating_parties)}"
        )
    have = set()
    for i in participating_parties:
        have.update(shares.party_masks[i])
    missing = set(shares.masks.keys()) - have
    if missing:
        raise ValueError(
            f"cooperating parties cannot cover all masks; missing {missing}"
        )
    total = np.zeros(next(iter(shares.masks.values())).shape, dtype=np.int64)
    for S in shares.masks:
        total += shares.masks[S]
    return total.astype(np.int64)


def apply_owner_delta(
    shares: ReplicatedShares,
    owner_idx: int,
    delta: np.ndarray,
) -> ReplicatedShares:
    """Owner ``owner_idx`` adds ``delta`` to one of its mask pieces.

    This is the SMC-equivalent of "add delta to one of the k additive
    shares" that the watermark owner performs during embedding.
    """
    if not (0 <= owner_idx < shares.n):
        raise ValueError(f"owner_idx out of range: {owner_idx}")
    if not shares.party_masks[owner_idx]:
        raise ValueError(f"party {owner_idx} holds no mask pieces")
    target_subset = shares.party_masks[owner_idx][0]
    new_masks = dict(shares.masks)
    new_masks[target_subset] = new_masks[target_subset] + delta.astype(np.int64)
    return ReplicatedShares(
        k=shares.k,
        n=shares.n,
        n_lsb=shares.n_lsb,
        seed=shares.seed,
        masks=new_masks,
        party_masks=shares.party_masks,
    )
