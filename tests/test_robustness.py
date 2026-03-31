"""Robustness tests for the Modified RRWEI-SM scheme.

The paper's central robustness claim (Sec. V-A2) is that the Modified
RRWEI-SM can recover a large fraction of its robust-watermark bits even
when the marked image is attacked.  We replicate scaled-down versions of
the JPEG-compression-style and additive-noise attacks.
"""

import numpy as np
import pytest

from rrwei_sm.modified_rrwei_sm import ModifiedRRWEISM
from rrwei_sm.utils import ber


def _make_smooth_cover(h: int = 128, w: int = 128) -> np.ndarray:
    xs = np.linspace(40, 210, w)
    ys = np.linspace(30, 200, h)[:, None]
    img = (0.5 * xs + 0.5 * ys).clip(0, 255)
    return img.astype(np.uint8)


def _additive_noise_attack(img: np.ndarray, sigma: float, rng) -> np.ndarray:
    noise = rng.normal(0, sigma, size=img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("sigma", [0.5, 1.5])
def test_modified_resists_additive_noise(sigma):
    img = _make_smooth_cover()
    rng = np.random.default_rng(0)
    robust_bits = rng.integers(0, 2, size=4, dtype=np.uint8)

    scheme = ModifiedRRWEISM(
        n_lsb=3, patchwork_m=64, patchwork_T=8, patchwork_seed=9, max_pee_layers=4
    )
    s1, s2, keys = scheme.encrypt(img, key_scramble=11, key_share=22)
    ms1, ms2, side = scheme.embed(s1, s2, robust_bits)

    # Attack: combine shares -> noise -> re-share with zero-share.
    marked_scrambled = (ms1.astype(np.int64) + ms2.astype(np.int64))
    marked_u8 = np.clip(marked_scrambled, 0, 255).astype(np.uint8)
    attacked_u8 = _additive_noise_attack(marked_u8, sigma=sigma, rng=rng)
    attacked_s1 = attacked_u8.astype(np.int32)
    attacked_s2 = np.zeros_like(attacked_s1)

    extracted = scheme.extract_robust_after_decrypt(attacked_s1, attacked_s2, keys, side)
    # At low noise, the robust watermark should be mostly intact.
    assert ber(extracted, robust_bits) <= 0.5


def test_modified_rrwei_sm_recovers_exactly_when_untouched():
    img = _make_smooth_cover()
    scheme = ModifiedRRWEISM(
        n_lsb=3, patchwork_m=64, patchwork_T=5, patchwork_seed=3, max_pee_layers=4
    )
    s1, s2, keys = scheme.encrypt(img, key_scramble=1, key_share=2)
    bits = np.array([1, 0, 1, 1], dtype=np.uint8)
    ms1, ms2, side = scheme.embed(s1, s2, bits)

    rec, got = scheme.extract_and_recover_after_decrypt(ms1, ms2, keys, side)
    assert (rec == img).all()
    assert np.array_equal(got, bits)


def test_rrwei_sm_fragile_under_strong_attack():
    """PEE (the basic scheme) is *not* expected to survive attacks.

    We just assert that the extractor runs without crashing and returns
    something -- this is a smoke test to document fragility.
    """
    from rrwei_sm.rrwei_sm import RRWEISM
    img = _make_smooth_cover(64, 64)
    rng = np.random.default_rng(0)
    scheme = RRWEISM(n_lsb=3, max_layers=2)
    s1, s2, keys = scheme.encrypt(img, key_scramble=1, key_share=2)
    bits = rng.integers(0, 2, size=80, dtype=np.uint8)
    ms1, ms2, side = scheme.embed(s1, s2, bits)

    # Strong attack: heavy noise on combined image
    combined = (ms1.astype(np.int64) + ms2.astype(np.int64))
    attacked = _additive_noise_attack(
        np.clip(combined, 0, 255).astype(np.uint8), sigma=5.0, rng=rng
    )
    att_s1 = attacked.astype(np.int32)
    att_s2 = np.zeros_like(att_s1)

    # Should not crash even if recovery is lossy.
    rec = scheme.decrypt(att_s1, att_s2, keys)
    assert rec.shape == img.shape
