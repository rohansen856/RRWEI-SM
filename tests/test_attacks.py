"""Tests for the modern attack suite."""

from __future__ import annotations

import numpy as np
import pytest

from rrwei_sm.attacks import (
    AVAILABLE_ATTACKS,
    diffusion_regen_proxy,
    gaussian_noise,
    jpeg2000_compress,
    jpeg_compress,
    mean_filter,
    median_filter,
    neural_codec_proxy,
    salt_and_pepper,
    sharpen_filter,
    super_resolution_cascade,
)


@pytest.mark.parametrize("name", list(AVAILABLE_ATTACKS.keys()))
def test_attack_preserves_shape_and_dtype(textured_cover, name):
    out = AVAILABLE_ATTACKS[name](textured_cover)
    assert out.shape == textured_cover.shape
    assert out.dtype == np.uint8


def test_neural_codec_distorts_image(textured_cover):
    attacked = neural_codec_proxy(textured_cover)
    assert not np.array_equal(attacked, textured_cover)
    # But still within the correct range.
    assert attacked.min() >= 0 and attacked.max() <= 255


def test_super_resolution_cascade_smooths_but_preserves_shape(textured_cover):
    out = super_resolution_cascade(textured_cover)
    assert out.shape == textured_cover.shape
    # Expect lower total variation after the round-trip.
    tv_orig = np.mean(np.abs(np.diff(textured_cover, axis=0))) + np.mean(
        np.abs(np.diff(textured_cover, axis=1))
    )
    tv_attacked = np.mean(np.abs(np.diff(out, axis=0))) + np.mean(
        np.abs(np.diff(out, axis=1))
    )
    assert tv_attacked < tv_orig


def test_diffusion_regen_proxy_fallback(textured_cover):
    out = diffusion_regen_proxy(textured_cover, use_vae=False)
    assert out.shape == textured_cover.shape
    assert out.dtype == np.uint8
    # Should distort (gaussian blur + jpeg).
    assert not np.array_equal(out, textured_cover)


def test_stdm_bits_survive_sr_cascade(classic_covers):
    """STDM survives a 2x bicubic SR round-trip with BER <= 0.25."""
    from datasets import lena_like
    from rrwei_sm.stdm import STDMConfig, embed, extract

    cover = lena_like(size=128)
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=128, dtype=np.uint8)
    marked, side = embed(cover, bits, STDMConfig(delta=60.0, n_coeffs=4, seed=0))
    attacked = AVAILABLE_ATTACKS["sr_cascade"](marked)
    out = extract(attacked, side)
    ber = float((out != bits).mean())
    assert ber <= 0.25, ber


def test_neural_codec_and_diffusion_are_harder_than_sr(classic_covers):
    """Sanity: the neural-codec and diffusion proxies produce strictly
    more distortion (in L2 pixel distance) than the SR cascade.

    Their exact BER against STDM is reported by the Step 7 robustness
    figure; we don't bake a tight bound here because the whole point
    of the modern attack suite is to be adversarial.
    """
    from datasets import lena_like

    cover = lena_like(size=128)
    d_sr = np.linalg.norm(
        AVAILABLE_ATTACKS["sr_cascade"](cover).astype(np.float64) - cover.astype(np.float64)
    )
    d_nc = np.linalg.norm(
        AVAILABLE_ATTACKS["neural_codec"](cover).astype(np.float64) - cover.astype(np.float64)
    )
    d_dr = np.linalg.norm(
        AVAILABLE_ATTACKS["diffusion_regen"](cover).astype(np.float64) - cover.astype(np.float64)
    )
    assert d_nc > d_sr, (d_nc, d_sr)
    assert d_dr > d_sr, (d_dr, d_sr)


def test_neural_codec_runs_on_marked_image(textured_cover):
    """Smoke test: neural_codec proxy does not crash and stays in range."""
    from rrwei_sm.stdm import STDMConfig, embed

    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, size=32, dtype=np.uint8)
    marked, _ = embed(textured_cover, bits, STDMConfig(delta=40.0, n_coeffs=4, seed=0))
    attacked = AVAILABLE_ATTACKS["neural_codec"](marked)
    assert attacked.shape == marked.shape
    assert attacked.dtype == np.uint8
    assert attacked.min() >= 0 and attacked.max() <= 255
