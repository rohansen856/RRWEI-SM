"""Tests for the additive-secret-sharing module (Eqs. 7-14)."""

import numpy as np

from rrwei_sm.secret_sharing import (
    Shares,
    additive_combine_shares,
    additive_share_image,
    recombine_hsb_lsb,
    share_hsb_plane,
    share_lsb_plane,
    shares_in_gray_range,
    split_hsb_lsb,
)


def test_split_recombine_is_identity():
    img = np.arange(256, dtype=np.uint8).reshape(16, 16)
    hsb, lsb = split_hsb_lsb(img, n_lsb=3)
    assert (recombine_hsb_lsb(hsb, lsb, n_lsb=3) == img).all()


def test_additive_share_sum_equals_cover(synthetic_random_image):
    shares = additive_share_image(synthetic_random_image, n_lsb=3, seed=1234)
    recovered = additive_combine_shares(shares.share1, shares.share2)
    assert (recovered == synthetic_random_image).all(), "Shares must sum to cover"


def test_share_hsb_and_lsb_reconstruct_cover(synthetic_random_image):
    """Eqs. 11-14 (carry-aware):  shares reconstruct the cover exactly.

    Because we share LSBs modularly (to keep arithmetic-shift-based HSB
    recovery well-defined for negative shares), the HSB parts sum to
    ``cover_HSB - carry`` and the LSB parts sum to ``cover_LSB + carry * 2^n``.
    These are consistent -- i.e. ``share1 + share2 == cover`` exactly --
    which is the *functional* invariant the SMC embedding depends on.
    """
    n = 3
    shares = additive_share_image(synthetic_random_image, n_lsb=n, seed=42)
    h1 = share_hsb_plane(shares.share1, n)
    h2 = share_hsb_plane(shares.share2, n)
    l1 = share_lsb_plane(shares.share1, n)
    l2 = share_lsb_plane(shares.share2, n)
    hsb, lsb = split_hsb_lsb(synthetic_random_image, n)
    carry = ((l1 + l2) >= (1 << n)).astype(np.int64)
    assert (h1 + h2 + carry == hsb).all(), "HSB equality must hold up to the LSB carry"
    assert ((l1 + l2) % (1 << n) == lsb).all(), "LSB sum is correct mod 2^n"
    # Overall additive invariant (Eq. 7).
    assert (shares.share1.astype(np.int64) + shares.share2.astype(np.int64)
            == synthetic_random_image).all()


def test_share_is_random_not_cover():
    """A single share alone must not leak the cover (additive OTP property)."""
    img = np.full((16, 16), 100, dtype=np.uint8)
    shares_a = additive_share_image(img, n_lsb=3, seed=1)
    shares_b = additive_share_image(img, n_lsb=3, seed=2)
    # Two independent share-1's of the same cover should differ heavily.
    diff_fraction = np.mean(shares_a.share1 != shares_b.share1)
    assert diff_fraction > 0.5


def test_shares_in_gray_range_mask(synthetic_random_image):
    shares = additive_share_image(synthetic_random_image, n_lsb=3, seed=0)
    mask = shares_in_gray_range(shares)
    # Many shares will overflow this by design; we just check that the
    # in-range shares really are in [0,255].
    assert shares.share1[mask].min() >= 0
    assert shares.share1[mask].max() <= 255
    assert shares.share2[mask].min() >= 0
    assert shares.share2[mask].max() <= 255
