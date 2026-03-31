"""
Robustness tests: run each attack from rrwei_sm.attacks against the
Modified RRWEI-SM watermark and verify BER stays below a reasonable
threshold.  These are smaller-scale regression tests; the full sweep
over sigma / JPEG quality / etc. is driven by ``evaluate.py`` and the
figure generators.
"""

import numpy as np
import pytest

from rrwei_sm import ModifiedRRWEISM
from rrwei_sm.attacks import (
    additive_gaussian_noise,
    jpeg2000_compress,
    jpeg_compress,
    mean_filter,
    median_filter,
    salt_and_pepper_noise,
    sharpen_filter,
)
from rrwei_sm.scrambling import block_scramble
from rrwei_sm.utils import ber


def _build_scheme() -> ModifiedRRWEISM:
    return ModifiedRRWEISM(
        n_lsb=3,
        patchwork_m=64,
        patchwork_T=5,
        patchwork_seed=5,
        max_pee_layers=4,
        patchwork_plane="hsb",
    )


@pytest.fixture(scope="module")
def smooth_128():
    """A 128x128 'smooth + mild texture' image (the pure gradient is hard
    for sign-of-difference extraction because many permuted blocks end up
    with near-zero original d_i; adding a little deterministic texture
    gives the patchwork embedder a stronger starting signal, closer to
    Lena/Baboon statistics)."""
    xs = np.linspace(40, 210, 128)
    ys = np.linspace(30, 200, 128)[:, None]
    base = (0.5 * xs + 0.5 * ys).astype(np.int64)
    rng = np.random.default_rng(0)
    tex = rng.integers(-15, 16, size=(128, 128))
    return np.clip(base + tex, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def marked(smooth_128):
    scheme = _build_scheme()
    s1, s2, keys = scheme.encrypt(smooth_128, key_scramble=11, key_share=22)
    robust = np.random.default_rng(42).integers(0, 2, size=16, dtype=np.uint8)
    ms1, ms2, side = scheme.embed(s1, s2, robust)
    marked_plain = scheme.decrypt(ms1, ms2, keys)
    return {
        "scheme": scheme,
        "keys": keys,
        "side": side,
        "robust": robust,
        "ms1": ms1,
        "ms2": ms2,
        "marked_plain": marked_plain,
    }


def _ber_under_attack(marked: dict, attacked_plain: np.ndarray) -> float:
    scheme = marked["scheme"]
    keys = marked["keys"]
    side = marked["side"]
    # Convert attacked plaintext back into the encrypted domain using the
    # public scrambling key -- this is the robust-path the paper describes.
    scrambled = block_scramble(attacked_plain, keys.block_size, keys.key_scramble)
    s1 = scrambled.astype(np.int32)
    s2 = np.zeros_like(s1)
    ext = scheme.extract_robust_after_decrypt(s1, s2, keys, side)
    return ber(ext, marked["robust"])


# Note on thresholds
# ------------------
# These are smoke tests on a synthetic 128x128 textured image with only
# 16 robust bits and T=5 in HSB mode.  They ensure BER is clearly below
# chance (<= 0.25-0.45 depending on attack severity).  The full
# Lena/Baboon/Peppers-scale experiment with per-sigma BER curves is in
# figures/make_wgn_ber.py and evaluate.py.


def test_no_attack_extraction_is_perfect(marked):
    """Sanity: with no attack the quick-extract must match embedded bits."""
    quick = _ber_under_attack(marked, marked["marked_plain"])
    assert quick == 0.0


def test_resists_gaussian_sigma5(marked):
    attacked = additive_gaussian_noise(marked["marked_plain"], sigma=5.0, seed=0)
    assert _ber_under_attack(marked, attacked) <= 0.10


def test_resists_gaussian_sigma20(marked):
    attacked = additive_gaussian_noise(marked["marked_plain"], sigma=20.0, seed=0)
    assert _ber_under_attack(marked, attacked) <= 0.25


def test_resists_salt_pepper_1pct(marked):
    attacked = salt_and_pepper_noise(marked["marked_plain"], p=0.01, seed=0)
    assert _ber_under_attack(marked, attacked) <= 0.20


def test_resists_median_3x3(marked):
    attacked = median_filter(marked["marked_plain"], ksize=3)
    # Median aggressively smooths the mild texture in the fixture image,
    # so BER here is noticeably higher than for JPEG/Gaussian.  The
    # paper uses larger textured images (Lena/Baboon) where the median
    # filter preserves the patchwork difference signal better.
    assert _ber_under_attack(marked, attacked) <= 0.40


def test_resists_mean_3x3(marked):
    attacked = mean_filter(marked["marked_plain"], ksize=3)
    assert _ber_under_attack(marked, attacked) <= 0.30


def test_resists_sharpen(marked):
    attacked = sharpen_filter(marked["marked_plain"], amount=0.5)
    assert _ber_under_attack(marked, attacked) <= 0.20


def test_resists_jpeg_q80(marked):
    attacked = jpeg_compress(marked["marked_plain"], quality=80)
    assert _ber_under_attack(marked, attacked) <= 0.15


def test_resists_jpeg_q40(marked):
    attacked = jpeg_compress(marked["marked_plain"], quality=40)
    assert _ber_under_attack(marked, attacked) <= 0.35


def test_resists_jpeg2000(marked):
    attacked = jpeg2000_compress(marked["marked_plain"], quality_layers=(20.0,))
    assert _ber_under_attack(marked, attacked) <= 0.35
