"""Tests for k-party additive sharing + the k-party RRWEI-SM API."""

import numpy as np
import pytest

from rrwei_sm import RRWEISM, ModifiedRRWEISM
from rrwei_sm.secret_sharing import (
    additive_combine_shares_k,
    additive_share_image_k,
)


@pytest.mark.parametrize("n_parties", [2, 3, 4, 5])
def test_kshare_sum_equals_cover(synthetic_random_image, n_parties):
    shares = additive_share_image_k(
        synthetic_random_image, n_parties=n_parties, n_lsb=3, seed=1234
    )
    assert len(shares) == n_parties
    combined = additive_combine_shares_k(shares)
    assert (combined == synthetic_random_image).all()


def test_kshare_single_share_looks_random(synthetic_random_image):
    """Any proper subset of <k shares must be uninformative about the cover."""
    shares = additive_share_image_k(
        synthetic_random_image, n_parties=4, n_lsb=3, seed=1
    )
    for s in shares[:3]:  # first 3 alone should look random
        corr = np.corrcoef(s.flatten(), synthetic_random_image.flatten())[0, 1]
        assert abs(corr) < 0.3


@pytest.mark.parametrize("n_parties", [2, 3, 4])
def test_rrweism_kparty_round_trip(synthetic_smooth_image, n_parties):
    scheme = RRWEISM(n_lsb=3, max_layers=2, n_parties=n_parties)
    shares, keys = scheme.encrypt_k(synthetic_smooth_image, key_scramble=1, key_share=2)
    recovered = scheme.decrypt_k(shares, keys)
    assert (recovered == synthetic_smooth_image).all()


def test_rrweism_kparty_embed_and_extract(synthetic_smooth_image):
    scheme = RRWEISM(n_lsb=3, max_layers=2, n_parties=3)
    shares, keys = scheme.encrypt_k(synthetic_smooth_image, key_scramble=5, key_share=6)
    bits = np.random.default_rng(7).integers(0, 2, size=80, dtype=np.uint8)
    marked_shares, side = scheme.embed_k(shares, owner_idx=1, bits=bits)
    rec_img, ext_bits = scheme.extract_and_recover_k(marked_shares, keys, side)
    assert (rec_img == synthetic_smooth_image).all()
    assert np.array_equal(ext_bits, bits[: side.n_embedded])


def test_modified_rrweism_kparty_round_trip(synthetic_smooth_image):
    scheme = ModifiedRRWEISM(
        n_lsb=3, patchwork_m=8, patchwork_T=3, patchwork_seed=101,
        max_pee_layers=4, n_parties=3,
    )
    shares, keys = scheme.encrypt_k(
        synthetic_smooth_image, key_scramble=1, key_share=2
    )
    robust = np.random.default_rng(99).integers(0, 2, size=8, dtype=np.uint8)
    marked_shares, side = scheme.embed_k(shares, owner_idx=0, robust_bits=robust)

    quick = scheme.extract_robust_k(marked_shares, keys, side)
    assert np.array_equal(quick, robust)

    rec, full_bits = scheme.extract_and_recover_k(marked_shares, keys, side)
    assert (rec == synthetic_smooth_image).all()
    assert np.array_equal(full_bits, robust)


def test_kparty_share_is_visually_noise_like(synthetic_smooth_image):
    shares = additive_share_image_k(
        synthetic_smooth_image, n_parties=3, n_lsb=3, seed=0
    )
    clip0 = np.clip(shares[0], 0, 255).astype(np.uint8)
    # A share of a smooth gradient should look much less smooth than
    # the gradient itself -- its pixel-to-pixel variance is much higher.
    assert clip0.std() > synthetic_smooth_image.std()
