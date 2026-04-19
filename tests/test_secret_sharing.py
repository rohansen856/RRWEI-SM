"""Tests for the ChaCha20-keystream additive secret sharing."""

from __future__ import annotations

import numpy as np
import pytest

from rrwei_sm.secret_sharing import (
    additive_combine_shares,
    additive_share_image,
    recombine_hsb_lsb,
    split_hsb_lsb,
)


@pytest.mark.parametrize("n_parties", [2, 3, 5])
def test_shares_sum_to_cover(textured_cover, n_parties):
    res = additive_share_image(textured_cover, n_parties=n_parties, seed=17)
    combined = additive_combine_shares(res.shares)
    assert (combined == textured_cover.astype(np.int64)).all()


def test_hsb_lsb_split_round_trip(textured_cover):
    hsb, lsb = split_hsb_lsb(textured_cover, n_lsb=3)
    rec = recombine_hsb_lsb(hsb, lsb, n_lsb=3)
    assert (rec == textured_cover.astype(np.int64)).all()


def test_any_single_share_viewed_mod256_has_low_correlation_with_cover(textured_cover):
    """Shares viewed as uint8 (mod 256) must be indistinguishable from uniform.

    This is the information-theoretic view that makes the scheme a
    one-time pad.  Signed int32 shares can correlate with the cover
    because one share contains (cover - uniform_mask) as a signed
    integer, but the mod-256 view cancels that.
    """
    res = additive_share_image(textured_cover, n_parties=2, seed=31337)
    cover = textured_cover.astype(np.float64).ravel()
    cover -= cover.mean()
    for s in res.shares:
        s_mod = (s.astype(np.int64) % 256).astype(np.float64).ravel()
        s_mod -= s_mod.mean()
        denom = np.linalg.norm(cover) * np.linalg.norm(s_mod)
        corr = float((cover @ s_mod) / denom)
        assert abs(corr) < 0.05, corr


def test_different_seeds_give_different_shares(textured_cover):
    a = additive_share_image(textured_cover, n_parties=2, seed=1)
    b = additive_share_image(textured_cover, n_parties=2, seed=2)
    assert not np.array_equal(a.shares[0], b.shares[0])


def test_npcr_under_random_key_regime_is_near_ideal(textured_cover):
    """With two independent random keys the shares look like fresh random bytes.

    This is the 'random-key' NPCR protocol used by some papers to
    report ~99.6% NPCR.  It is trivially satisfied by any
    keystream-based OTP, and serves as a sanity check that our
    keystream is not re-using state across seeds.
    """
    a = additive_share_image(textured_cover, n_parties=2, seed=100)
    b = additive_share_image(textured_cover, n_parties=2, seed=200)
    diffs_share0 = (a.shares[0] != b.shares[0]).mean() * 100.0
    assert diffs_share0 > 99.0


def test_rejects_non_uint8(textured_cover):
    with pytest.raises(TypeError):
        additive_share_image(textured_cover.astype(np.int16))
