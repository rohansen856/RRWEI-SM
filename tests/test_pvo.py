"""Tests for PVO + pairwise PEE predictor."""

from __future__ import annotations

import numpy as np
import pytest

from rrwei_sm.pvo import (
    capacity_estimate,
    embed,
    extract,
)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10.0 * np.log10(255.0**2 / mse)


def test_round_trip_empty_bits(textured_cover):
    bits = np.zeros(0, dtype=np.uint8)
    marked, side, n_emb = embed(textured_cover, bits)
    # Embed with no bits -> histogram shifts only.  Still reversible.
    recovered, out_bits = extract(marked, side)
    assert (recovered == textured_cover).all()
    assert out_bits.size == 0


def test_round_trip_random_bits(textured_cover):
    cap = capacity_estimate(textured_cover)
    n_bits = cap["max_side"] + cap["min_side"]
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
    marked, side, n_emb = embed(textured_cover, bits)
    recovered, out_bits = extract(marked, side)
    assert (recovered == textured_cover).all()
    assert out_bits.size == n_emb
    assert (out_bits == bits[:n_emb]).all()


def test_quality_under_small_payload(textured_cover):
    # Embed ~64 bits (small fraction of capacity) and check PSNR.
    rng = np.random.default_rng(4)
    bits = rng.integers(0, 2, size=64, dtype=np.uint8)
    marked, side, n_emb = embed(textured_cover, bits)
    psnr = _psnr(marked, textured_cover)
    assert psnr >= 35.0, psnr


_LEGACY_PEE_BPP = {
    # Legacy 3-neighbour PEE figures (rrwei_sm) measured during the
    # main-branch evaluate.py run, repeated here as the regression
    # bound that the modern predictor must beat.
    "lena_like": 0.100,
    "baboon_like": 0.030,
    "peppers_like": 0.200,
}


@pytest.mark.parametrize("name", list(_LEGACY_PEE_BPP.keys()))
def test_pvo_capacity_beats_legacy_pee(classic_covers, name):
    """Multi-layer PVO must strictly exceed the legacy 3-neighbour PEE bpp."""
    rng = np.random.default_rng(5)
    cover = classic_covers[name]
    current = cover.copy()
    total = 0
    n_layers_used = 0
    for layer in range(6):
        cap = capacity_estimate(current)
        n_bits = cap["max_side"] + cap["min_side"]
        if n_bits == 0:
            break
        bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
        current, _, n_emb = embed(current, bits)
        total += n_emb
        n_layers_used += 1
    bpp = total / cover.size
    threshold = _LEGACY_PEE_BPP[name]
    print(f"{name} bpp after {n_layers_used} layers: {bpp:.3f} (>= {threshold})")
    assert bpp >= threshold, (name, bpp, threshold)


def test_round_trip_multiple_layers(textured_cover):
    rng = np.random.default_rng(6)
    current = textured_cover.copy()
    sides = []
    embedded_bits = []
    for layer in range(3):
        cap = capacity_estimate(current)
        n_bits = cap["max_side"] + cap["min_side"]
        if n_bits == 0:
            break
        bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
        current, side, n_emb = embed(current, bits)
        sides.append(side)
        embedded_bits.append(bits[:n_emb])

    # Invert in reverse.
    for side, bits in zip(reversed(sides), reversed(embedded_bits)):
        current, out_bits = extract(current, side)
        assert (out_bits == bits).all()
    assert (current == textured_cover).all()


def test_capacity_estimate_is_nonnegative(textured_cover):
    cap = capacity_estimate(textured_cover)
    assert cap["max_side"] >= 0
    assert cap["min_side"] >= 0
    assert cap["n_blocks"] == textured_cover.size // 4
