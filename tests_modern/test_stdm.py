"""Tests for the STDM robust watermark."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from rrwei_sm_modern.stdm import STDMConfig, embed, extract


def _rng_bits(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, size=n, dtype=np.uint8)


def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("L"), dtype=np.uint8)


def _awgn(image: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=sigma, size=image.shape)
    return np.clip(np.round(image.astype(np.float64) + noise), 0, 255).astype(np.uint8)


def test_round_trip_clean_image(textured_cover):
    bits = _rng_bits(32, seed=1)
    marked, side = embed(textured_cover, bits)
    out = extract(marked, side)
    assert (out == bits).all()


def test_marked_image_shape_and_dtype(textured_cover):
    bits = _rng_bits(32, seed=2)
    marked, _ = embed(textured_cover, bits)
    assert marked.shape == textured_cover.shape
    assert marked.dtype == np.uint8


def test_embedding_raises_when_bits_exceed_blocks(textured_cover):
    n_blocks = (textured_cover.size // 64)
    bits = _rng_bits(n_blocks + 1, seed=3)
    with pytest.raises(ValueError):
        embed(textured_cover, bits)


def test_ber_under_jpeg_q40(textured_cover):
    bits = _rng_bits(32, seed=4)
    # Low-frequency coefficients (larger delta needed for visibility
    # but much more JPEG-robust because JPEG quantizes low freqs least).
    marked, side = embed(textured_cover, bits, STDMConfig(delta=40.0, n_coeffs=4))
    attacked = _jpeg_roundtrip(marked, quality=40)
    out = extract(attacked, side)
    ber = float((out != bits).mean())
    assert ber <= 0.10, ber


def test_ber_under_gaussian_sigma10(textured_cover):
    bits = _rng_bits(32, seed=5)
    marked, side = embed(textured_cover, bits, STDMConfig(delta=40.0, n_coeffs=4))
    attacked = _awgn(marked, sigma=10.0, seed=7)
    out = extract(attacked, side)
    ber = float((out != bits).mean())
    assert ber <= 0.20, ber


def test_marked_image_quality_is_reasonable(textured_cover):
    bits = _rng_bits(32, seed=6)
    marked, _ = embed(textured_cover, bits, STDMConfig(delta=8.0, n_coeffs=8))
    mse = np.mean((marked.astype(np.float64) - textured_cover.astype(np.float64)) ** 2)
    psnr = 10.0 * np.log10(255.0**2 / max(mse, 1e-8))
    assert psnr >= 35.0, psnr
