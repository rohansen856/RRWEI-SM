"""Tests for block-level scrambling."""

import numpy as np

from rrwei_sm.scrambling import block_scramble, block_unscramble


def test_scramble_then_unscramble_is_identity(synthetic_random_image):
    scrambled = block_scramble(synthetic_random_image, block_size=2, seed=123)
    recovered = block_unscramble(scrambled, block_size=2, seed=123)
    assert (recovered == synthetic_random_image).all()


def test_scramble_actually_changes_image(synthetic_smooth_image):
    scrambled = block_scramble(synthetic_smooth_image, block_size=2, seed=5)
    # Smooth gradient should look different after scrambling (at least for
    # a meaningful fraction of pixels).
    n_diff = np.sum(scrambled != synthetic_smooth_image)
    assert n_diff > synthetic_smooth_image.size // 4


def test_scramble_block_preserved():
    """2x2 blocks must move as indivisible units (matters for PEE)."""
    img = np.arange(64, dtype=np.uint8).reshape(8, 8)
    scr = block_scramble(img, block_size=2, seed=0)
    # Every 2x2 block of scr should be a 2x2 block of img (possibly moved).
    src_blocks = {
        tuple(img[i : i + 2, j : j + 2].flatten())
        for i in range(0, 8, 2)
        for j in range(0, 8, 2)
    }
    out_blocks = {
        tuple(scr[i : i + 2, j : j + 2].flatten())
        for i in range(0, 8, 2)
        for j in range(0, 8, 2)
    }
    assert out_blocks == src_blocks


def test_wrong_seed_fails_to_invert():
    img = np.arange(256, dtype=np.uint8).reshape(16, 16)
    scr = block_scramble(img, block_size=2, seed=1)
    wrong = block_unscramble(scr, block_size=2, seed=2)
    assert not (wrong == img).all()
