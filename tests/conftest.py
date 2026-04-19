"""Fixtures shared across all modernization tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def smooth_cover() -> np.ndarray:
    """Small smooth grayscale cover image for quick tests."""
    yy, xx = np.mgrid[:64, :64]
    return ((yy + xx) * 2).astype(np.uint8)


@pytest.fixture
def textured_cover() -> np.ndarray:
    """Slightly textured 64x64 cover (smooth + low-amplitude noise)."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:64, :64]
    smooth = ((yy + xx) * 2).astype(np.int32)
    noise = rng.integers(-8, 9, size=smooth.shape, dtype=np.int32)
    return np.clip(smooth + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def classic_covers() -> dict[str, np.ndarray]:
    """Reuse the synthetic lena/baboon/peppers-like covers."""
    from datasets import load_classic_images

    return load_classic_images(size=64)
