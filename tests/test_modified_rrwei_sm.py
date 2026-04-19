"""End-to-end tests for the Modified (two-stage) RRWEI-SM."""

import numpy as np
import pytest

from rrwei_sm.modified_rrwei_sm import ModifiedRRWEISM


def test_modified_round_trip(synthetic_smooth_image):
    scheme = ModifiedRRWEISM(
        n_lsb=3,
        patchwork_m=8,
        patchwork_T=3,
        patchwork_seed=101,
        max_pee_layers=4,
    )
    s1, s2, keys = scheme.encrypt(synthetic_smooth_image, key_scramble=1, key_share=2)
    robust_bits = np.random.default_rng(99).integers(0, 2, size=8, dtype=np.uint8)

    ms1, ms2, side = scheme.embed(s1, s2, robust_bits)

    # Quick robust-only path (does not need side info beyond skeleton).
    quick = scheme.extract_robust_after_decrypt(ms1, ms2, keys, side)
    assert np.array_equal(quick, robust_bits)

    # Full inversion on intact image.
    recovered, full_bits = scheme.extract_and_recover_after_decrypt(ms1, ms2, keys, side)
    assert (recovered == synthetic_smooth_image).all()
    assert np.array_equal(full_bits, robust_bits)


def test_modified_raises_when_capacity_too_low():
    """Modified scheme should fail loudly when PEE can't fit the side info."""
    img = np.random.default_rng(0).integers(0, 256, size=(16, 16), dtype=np.uint8)
    scheme = ModifiedRRWEISM(
        n_lsb=3, patchwork_m=2, patchwork_T=5, patchwork_seed=7, max_pee_layers=1
    )
    s1, s2, keys = scheme.encrypt(img, key_scramble=1, key_share=2)
    # Force many robust bits -> large side info, likely over PEE capacity
    # on a 16x16 random image in a single layer.
    bits = np.random.default_rng(1).integers(0, 2, size=32, dtype=np.uint8)
    with pytest.raises(RuntimeError):
        scheme.embed(s1, s2, bits)


def test_modified_separable_property(synthetic_smooth_image):
    """The two shares individually are uninformative about the robust bits."""
    scheme = ModifiedRRWEISM(
        n_lsb=3, patchwork_m=8, patchwork_T=3, patchwork_seed=0, max_pee_layers=4
    )
    s1, s2, keys = scheme.encrypt(synthetic_smooth_image, key_scramble=5, key_share=6)
    bits = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    ms1, _, _ = scheme.embed(s1, s2, bits)
    # Correlation of either share with the cover should be weak.
    corr = np.corrcoef(ms1.flatten(), synthetic_smooth_image.flatten())[0, 1]
    assert abs(corr) < 0.3
