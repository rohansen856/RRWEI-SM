"""Tests for utility metric helpers."""

import numpy as np

from rrwei_sm.utils import ber, bits_to_bytes, bytes_to_bits, psnr, ssim_simple


def test_psnr_identical_is_inf():
    a = np.zeros((8, 8), dtype=np.uint8)
    assert psnr(a, a) == float("inf")


def test_psnr_monotone_with_noise():
    a = np.full((32, 32), 128, dtype=np.uint8)
    small = a.copy()
    small[0, 0] = 130
    large = a.copy()
    large[0, 0] = 200
    assert psnr(a, small) > psnr(a, large)


def test_ber_basic():
    assert ber(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1])) == 0.0
    assert ber(np.array([1, 1, 1, 1]), np.array([0, 0, 0, 0])) == 1.0
    assert ber(np.array([1, 0, 1, 0]), np.array([0, 0, 1, 0])) == 0.25


def test_bit_byte_round_trip():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=np.uint8)
    back = bytes_to_bits(bits_to_bytes(bits), n_bits=len(bits))
    assert np.array_equal(back, bits)


def test_ssim_identity_is_one():
    img = np.random.default_rng(0).integers(0, 255, (32, 32), dtype=np.uint8)
    assert abs(ssim_simple(img, img) - 1.0) < 1e-9
