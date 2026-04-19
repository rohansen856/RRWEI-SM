"""Tests for the Prediction-Error-Expansion embedder/extractor."""

import numpy as np

from rrwei_sm.pee import (
    PEEEmbedder,
    PEEExtractor,
    compute_scaled_error,
    compute_share_contribution_to_error,
)
from rrwei_sm.secret_sharing import additive_share_image, share_hsb_plane


def test_scaled_error_is_integer_and_centered_near_zero(synthetic_smooth_image):
    e = compute_scaled_error(synthetic_smooth_image.astype(np.int64) >> 3)
    assert e.dtype == np.int64
    assert -50 < e.mean() < 50


def test_smc_contribution_additivity(synthetic_random_image):
    """Eqs. (17)-(19): e1 + e2 == e_scaled of the combined HSB plane."""
    shares = additive_share_image(synthetic_random_image, n_lsb=3, seed=999)
    h1 = share_hsb_plane(shares.share1, 3)
    h2 = share_hsb_plane(shares.share2, 3)
    e1 = compute_share_contribution_to_error(h1)
    e2 = compute_share_contribution_to_error(h2)
    combined_e = compute_scaled_error(h1 + h2)
    assert np.array_equal(e1 + e2, combined_e)


def test_pee_round_trip_on_smooth_image(synthetic_smooth_image):
    """Embed random bits -> extract -> recover original exactly."""
    bits = np.random.default_rng(1).integers(0, 2, size=200, dtype=np.uint8)
    embedder = PEEEmbedder(n_lsb=3, max_layers=2)
    marked, side, n_emb = embedder.embed(synthetic_smooth_image, bits)
    assert n_emb > 0
    extractor = PEEExtractor()
    recovered, ext_bits = extractor.extract(marked, side)
    assert (recovered == synthetic_smooth_image).all(), "PEE must be reversible"
    assert np.array_equal(ext_bits[:n_emb], bits[:n_emb])


def test_pee_preserves_non_target_pixels(synthetic_smooth_image):
    """Embedding only modifies target pixels of 2x2 blocks."""
    bits = np.random.default_rng(0).integers(0, 2, size=100, dtype=np.uint8)
    marked, _, _ = PEEEmbedder(n_lsb=3, max_layers=1).embed(synthetic_smooth_image, bits)
    diff = marked.astype(int) - synthetic_smooth_image.astype(int)
    # Non-target pixels (odd row OR odd col) must be unchanged in layer 1.
    assert (diff[0::2, 1::2] == 0).all()
    assert (diff[1::2, 0::2] == 0).all()
    assert (diff[1::2, 1::2] == 0).all()


def test_pee_respects_overflow_mask():
    """Saturated pixels (255) should never be target-shifted to 256+."""
    img = np.full((16, 16), 255, dtype=np.uint8)
    embedder = PEEEmbedder(n_lsb=3, max_layers=1)
    marked, side, n_emb = embedder.embed(img, np.zeros(100, dtype=np.uint8))
    assert marked.max() <= 255
    # No bits should be embeddable into an all-saturated image.
    assert n_emb == 0


def test_pee_extracted_bit_count_matches_embedded(synthetic_smooth_image):
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1] * 10, dtype=np.uint8)
    marked, side, n_emb = PEEEmbedder(n_lsb=3, max_layers=1).embed(
        synthetic_smooth_image, bits
    )
    _, extracted = PEEExtractor().extract(marked, side)
    assert len(extracted) == n_emb
    assert np.array_equal(extracted, bits[:n_emb])
