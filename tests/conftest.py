"""Pytest bootstrap: adds the repo root to sys.path for ``import rrwei_sm``."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


import numpy as np
import pytest


@pytest.fixture
def synthetic_smooth_image():
    """A smooth 64x64 gradient image -- easy for PEE to embed into."""
    x = np.linspace(30, 200, 64, dtype=np.int64)
    img = np.tile(x, (64, 1))
    return img.astype(np.uint8)


@pytest.fixture
def synthetic_random_image():
    """A 32x32 pseudo-random image -- stresses overflow handling."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
