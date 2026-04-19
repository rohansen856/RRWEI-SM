"""Tests for LPIPS and DISTS-proxy perceptual metrics."""

from __future__ import annotations

import numpy as np
import pytest

from rrwei_sm_modern.metrics import (
    dists_proxy,
    lpips_distance,
    psnr,
    ssim,
)


# Guard LPIPS tests: require torch + lpips + cached weights.
torch = pytest.importorskip("torch")
lpips = pytest.importorskip("lpips")


@pytest.fixture
def random_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 64), dtype=np.uint8)


def test_psnr_identical_is_large(random_image):
    assert psnr(random_image, random_image) >= 80.0


def test_ssim_identical_is_one(random_image):
    assert abs(ssim(random_image, random_image) - 1.0) < 1e-6


def test_lpips_identity_is_zero(random_image):
    d = lpips_distance(random_image, random_image)
    assert abs(d) < 1e-5


def test_lpips_decreases_monotonically_as_noise_grows(random_image):
    rng = np.random.default_rng(7)
    # Use a smoother cover so LPIPS is dominated by added noise.
    yy, xx = np.mgrid[:64, :64]
    smooth = ((yy + xx) * 2).astype(np.uint8)
    distances = []
    for sigma in [0.0, 2.0, 5.0, 10.0, 20.0]:
        if sigma == 0.0:
            noisy = smooth
        else:
            noise = rng.normal(scale=sigma, size=smooth.shape)
            noisy = np.clip(smooth + noise, 0, 255).astype(np.uint8)
        distances.append(lpips_distance(smooth, noisy))
    # Distances should be monotonically non-decreasing.
    for i in range(1, len(distances)):
        assert distances[i] + 1e-3 >= distances[i - 1], distances


def test_dists_identity_is_zero(random_image):
    d = dists_proxy(random_image, random_image)
    assert d < 1e-4, d


def test_dists_increases_with_noise(random_image):
    rng = np.random.default_rng(3)
    d_low = dists_proxy(
        random_image,
        np.clip(random_image + rng.integers(-2, 3, size=random_image.shape), 0, 255).astype(np.uint8),
    )
    d_high = dists_proxy(
        random_image,
        np.clip(random_image + rng.integers(-40, 41, size=random_image.shape), 0, 255).astype(np.uint8),
    )
    assert d_high > d_low, (d_low, d_high)


def test_dists_shape_mismatch_raises():
    import pytest as _pt

    with _pt.raises(ValueError):
        dists_proxy(np.zeros((64, 64), np.uint8), np.zeros((32, 32), np.uint8))
