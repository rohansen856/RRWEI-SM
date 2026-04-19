"""Tests for (k, n) replicated secret sharing."""

from __future__ import annotations

import numpy as np
import pytest

from rrwei_sm_modern.threshold_sharing import (
    ReplicatedShares,
    apply_owner_delta,
    combine,
    share,
)


@pytest.mark.parametrize("k,n", [(2, 2), (2, 3), (3, 4), (2, 5)])
def test_k_parties_reconstruct_exactly(textured_cover, k, n):
    shares = share(textured_cover, k=k, n=n, seed=7)
    parties = list(range(k))  # first k parties
    recovered = combine(shares, parties)
    assert (recovered == textured_cover.astype(np.int64)).all()


@pytest.mark.parametrize("k,n", [(2, 3), (3, 4)])
def test_fewer_than_k_cannot_reconstruct(textured_cover, k, n):
    shares = share(textured_cover, k=k, n=n, seed=11)
    with pytest.raises(ValueError):
        combine(shares, list(range(k - 1)))


def test_two_different_k_subsets_agree(textured_cover):
    shares = share(textured_cover, k=2, n=3, seed=21)
    r_01 = combine(shares, [0, 1])
    r_02 = combine(shares, [0, 2])
    r_12 = combine(shares, [1, 2])
    assert np.array_equal(r_01, r_02)
    assert np.array_equal(r_01, r_12)


def test_apply_owner_delta_preserves_additive_homomorphism(textured_cover):
    shares = share(textured_cover, k=2, n=3, seed=42)
    delta = np.zeros_like(textured_cover, dtype=np.int64)
    delta[0, 0] = 3
    delta[10, 12] = -7
    new_shares = apply_owner_delta(shares, owner_idx=0, delta=delta)
    recovered = combine(new_shares, [0, 1])
    assert (recovered == textured_cover.astype(np.int64) + delta).all()


def test_each_party_holds_c_n_minus_1_p_masks():
    import numpy as np

    from math import comb

    cover = np.zeros((16, 16), dtype=np.uint8)
    for k, n in [(2, 3), (3, 5), (2, 4)]:
        sh = share(cover, k=k, n=n, seed=0)
        p = k - 1
        expected = comb(n - 1, p)
        for holdings in sh.party_masks:
            assert len(holdings) == expected


def test_rejects_invalid_parameters(textured_cover):
    with pytest.raises(ValueError):
        share(textured_cover, k=1, n=2)
    with pytest.raises(ValueError):
        share(textured_cover, k=4, n=3)
    with pytest.raises(TypeError):
        share(textured_cover.astype(np.int16), k=2, n=3)
