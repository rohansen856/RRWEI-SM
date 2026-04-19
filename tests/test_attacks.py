"""Tests for the image attack primitives."""

import numpy as np
import pytest

from rrwei_sm.attacks import (
    additive_gaussian_noise,
    jpeg2000_compress,
    jpeg_compress,
    mean_filter,
    median_filter,
    salt_and_pepper_noise,
    sharpen_filter,
)


@pytest.fixture
def simple_img():
    rng = np.random.default_rng(0)
    return rng.integers(30, 220, size=(32, 32), dtype=np.uint8)


def test_gaussian_noise_distribution(simple_img):
    attacked = additive_gaussian_noise(simple_img, sigma=5.0, seed=1)
    assert attacked.shape == simple_img.shape
    assert attacked.dtype == np.uint8
    diff = attacked.astype(int) - simple_img.astype(int)
    # Standard deviation should be close to sigma (5), allow for clipping.
    assert 2.0 < diff.std() < 10.0


def test_gaussian_noise_zero_sigma_is_identity(simple_img):
    attacked = additive_gaussian_noise(simple_img, sigma=0.0, seed=0)
    assert (attacked == simple_img).all()


def test_salt_and_pepper_changes_about_p_pixels(simple_img):
    attacked = salt_and_pepper_noise(simple_img, p=0.1, seed=0)
    changed = np.sum(attacked != simple_img)
    # At least half of the requested ~10% should differ (some random pixels
    # may receive their own value).
    assert changed > int(simple_img.size * 0.05)


def test_salt_and_pepper_values_are_extremes(simple_img):
    attacked = salt_and_pepper_noise(simple_img, p=0.2, seed=0)
    diff_mask = attacked != simple_img
    values = attacked[diff_mask]
    assert set(values.tolist()).issubset({0, 255})


def test_median_filter_removes_salt_spikes():
    img = np.full((16, 16), 100, dtype=np.uint8)
    img[4, 4] = 255  # single salt pixel
    filtered = median_filter(img, ksize=3)
    assert filtered[4, 4] == 100


def test_mean_filter_smooths(simple_img):
    filtered = mean_filter(simple_img, ksize=3)
    # Mean filter should reduce pixel-to-pixel variation.
    assert filtered.std() <= simple_img.std()


def test_sharpen_identity_on_constant_image():
    img = np.full((16, 16), 128, dtype=np.uint8)
    s = sharpen_filter(img, amount=1.5)
    assert (s == img).all()


def test_jpeg_compress_returns_uint8(simple_img):
    out = jpeg_compress(simple_img, quality=75)
    assert out.shape == simple_img.shape
    assert out.dtype == np.uint8


def test_jpeg_compress_low_quality_is_lossy(simple_img):
    high = jpeg_compress(simple_img, quality=95)
    low = jpeg_compress(simple_img, quality=20)
    mse_high = np.mean((high.astype(float) - simple_img.astype(float)) ** 2)
    mse_low = np.mean((low.astype(float) - simple_img.astype(float)) ** 2)
    assert mse_low > mse_high


def test_jpeg2000_compress_returns_uint8(simple_img):
    out = jpeg2000_compress(simple_img, quality_layers=(20.0,))
    assert out.shape == simple_img.shape
    assert out.dtype == np.uint8


def test_jpeg_compress_invalid_quality_raises(simple_img):
    with pytest.raises(ValueError):
        jpeg_compress(simple_img, quality=0)
    with pytest.raises(ValueError):
        jpeg_compress(simple_img, quality=150)


def test_median_filter_invalid_ksize(simple_img):
    with pytest.raises(ValueError):
        median_filter(simple_img, ksize=2)
