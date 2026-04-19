"""Tests for the Hua-et-al. 2D-LSCM-based high-speed scrambling."""

import numpy as np
import pytest

from rrwei_sm.hua_scrambling import HuaKey, hua_scramble, hua_unscramble


def test_hua_scramble_unscramble_is_identity(synthetic_random_image):
    key = HuaKey.from_seed(42)
    scrambled = hua_scramble(synthetic_random_image, block_size=2, key=key)
    recovered = hua_unscramble(scrambled, block_size=2, key=key)
    assert (recovered == synthetic_random_image).all()


def test_hua_with_int_seed_is_deterministic(synthetic_random_image):
    a = hua_scramble(synthetic_random_image, block_size=2, key=7)
    b = hua_scramble(synthetic_random_image, block_size=2, key=7)
    c = hua_scramble(synthetic_random_image, block_size=2, key=8)
    assert (a == b).all()
    assert not (a == c).all()


def test_hua_preserves_2x2_blocks():
    img = np.arange(64, dtype=np.uint8).reshape(8, 8)
    scr = hua_scramble(img, block_size=2, key=0)
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


def test_hua_changes_image_meaningfully(synthetic_smooth_image):
    scr = hua_scramble(synthetic_smooth_image, block_size=2, key=3)
    n_diff = np.sum(scr != synthetic_smooth_image)
    assert n_diff > synthetic_smooth_image.size // 4


def test_hua_wrong_key_does_not_invert(synthetic_random_image):
    scr = hua_scramble(synthetic_random_image, block_size=2, key=1)
    wrong = hua_unscramble(scr, block_size=2, key=2)
    assert not (wrong == synthetic_random_image).all()


def test_hua_requires_divisible_shape():
    with pytest.raises(ValueError):
        hua_scramble(np.zeros((5, 5), dtype=np.uint8), block_size=2, key=0)


def test_hua_decorrelates_adjacent_pixels():
    """Scrambling should break the horizontal correlation of a textured image.

    The paper reports horizontal correlation ~0.05-0.15 after 2x2 block
    scrambling on Lena/Baboon.  We use a textured synthetic image here
    (pure-gradient images still have surprisingly high residual
    correlation after 2x2 block permutation because neighbouring blocks
    that used to be in the same column can end up adjacent again).
    """
    from rrwei_sm.metrics import pixel_correlation
    rng = np.random.default_rng(0)
    base = np.linspace(40, 210, 64).astype(np.int64)
    img = np.tile(base, (64, 1)) + rng.integers(-20, 21, size=(64, 64))
    img = np.clip(img, 0, 255).astype(np.uint8)
    before = pixel_correlation(img, "horizontal", n_pairs=2000, seed=0)
    scr = hua_scramble(img, block_size=2, key=5)
    after = pixel_correlation(scr, "horizontal", n_pairs=2000, seed=0)
    # Require the correlation magnitude to be noticeably lower.
    assert abs(after) < max(abs(before) - 0.2, 0.5)
