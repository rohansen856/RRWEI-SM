"""Tests for the synthetic classic-image generators."""

import numpy as np

from datasets import baboon_like, lena_like, load_classic_images, peppers_like


def test_all_classic_images_are_uint8_and_correct_size():
    imgs = load_classic_images(size=64)
    assert set(imgs.keys()) == {"lena_like", "baboon_like", "peppers_like"}
    for name, img in imgs.items():
        assert img.shape == (64, 64), name
        assert img.dtype == np.uint8, name


def test_lena_like_is_smooth():
    img = lena_like(size=128)
    # Average absolute horizontal gradient should be small (smooth image).
    grad = np.abs(np.diff(img.astype(int), axis=1))
    assert grad.mean() < 25.0


def test_baboon_like_is_textured():
    img = baboon_like(size=128)
    grad = np.abs(np.diff(img.astype(int), axis=1))
    # Baboon should have large horizontal gradient on average.
    assert grad.mean() > 10.0


def test_peppers_like_has_multiple_regions():
    img = peppers_like(size=128)
    # Histogram should not be unimodal: multiple distinct peaks expected.
    hist, _ = np.histogram(img, bins=16)
    assert hist[hist > hist.max() * 0.3].size >= 2


def test_classic_images_are_deterministic():
    a = lena_like(size=64)
    b = lena_like(size=64)
    assert (a == b).all()
