"""Tests for patchwork robust watermarking (Eqs. 23-27)."""

import numpy as np

from rrwei_sm.patchwork import PatchworkEmbedder, PatchworkExtractor


def test_patchwork_embed_extract_robust_bits():
    rng = np.random.default_rng(0)
    img = rng.integers(30, 220, size=(16, 16), dtype=np.uint8)
    bits = rng.integers(0, 2, size=16 * 16 // 8, dtype=np.uint8)
    embedder = PatchworkEmbedder(m=4, T=5, seed=123)
    marked, side = embedder.embed(img, bits.tolist())
    extractor = PatchworkExtractor()
    out_bits = extractor.extract_bits(marked, side)
    # Unattacked image: robust bits must match.
    assert np.array_equal(out_bits, bits[: side.n_embedded])


def test_patchwork_exact_recovery_intact():
    rng = np.random.default_rng(1)
    img = rng.integers(40, 200, size=(16, 32), dtype=np.uint8)
    bits = rng.integers(0, 2, size=16 * 32 // 8, dtype=np.uint8).tolist()
    embedder = PatchworkEmbedder(m=4, T=3, seed=5)
    marked, side = embedder.embed(img, bits)
    ext = PatchworkExtractor()
    got_bits = ext.extract_bits(marked, side)
    recovered = ext.recover(marked, side, got_bits)
    assert (recovered == img).all()


def test_patchwork_robust_to_small_noise():
    rng = np.random.default_rng(2)
    img = rng.integers(60, 180, size=(16, 16), dtype=np.uint8)
    bits = rng.integers(0, 2, size=32, dtype=np.uint8).tolist()
    marked, side = PatchworkEmbedder(m=4, T=5, seed=7).embed(img, bits)
    # Add small Gaussian noise (|noise| <= ~3).
    noise = rng.integers(-3, 4, size=marked.shape)
    attacked = np.clip(marked.astype(int) + noise, 0, 255).astype(np.uint8)
    ext_bits = PatchworkExtractor().extract_bits(attacked, side)
    # Expect BER <= ~0.1 (robustness is the whole point).
    n_cmp = min(len(ext_bits), len(bits))
    ber = np.mean(ext_bits[:n_cmp] != np.array(bits[:n_cmp], dtype=np.uint8))
    assert ber <= 0.15


def test_patchwork_skipped_blocks_are_not_modified():
    # Make a block full of 255 so any positive-T shift would overflow.
    img = np.zeros((4, 8), dtype=np.uint8)
    img[:] = 255
    bits = [1, 0, 1, 0]
    marked, side = PatchworkEmbedder(m=4, T=5, seed=1).embed(img, bits)
    # All blocks should have been skipped; image unchanged.
    assert (marked == img).all()
    assert side.skipped.all()
