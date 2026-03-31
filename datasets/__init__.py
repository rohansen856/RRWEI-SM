"""
Small classic test-image stand-ins (Lena / Baboon / Peppers-style).

The paper reports experiments on Lena, Baboon and Peppers from the
USC-SIPI database.  Those images carry their own license; rather than
vendor them directly we generate deterministic synthetic stand-ins
whose statistics approximate the three canonical categories:

* ``lena_like``    - smooth-to-textured portrait-style, single strong
                     brightness gradient and smoothly-varying texture.
* ``baboon_like``  - high-frequency textured image ("fur"), high pixel
                     variance, weak spatial correlation.
* ``peppers_like`` - smooth with a few disjoint regions of different
                     means ("peppers"), low entropy within each region.

Every generator returns a 256x256 uint8 grayscale image.  They are
deterministic (seeded) so regression tests and benchmark scripts
produce identical results on every run.

Usage
-----

    from datasets import load_classic_images

    for name, img in load_classic_images().items():
        ...
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def lena_like(size: int = 256, seed: int = 0) -> np.ndarray:
    """Smooth radial gradient + Gaussian-textured detail."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(xs, xs)
    # Portrait-like oval mask.
    radial = np.exp(-((xx ** 2) / 0.8 + (yy ** 2) / 0.6))
    base = 60 + 150 * radial
    # Add low-frequency shading so regions have different mean luminance.
    shade = 20 * np.cos(2 * np.pi * (xx + 0.3 * yy))
    texture = rng.normal(0, 6, size=(size, size))
    img = np.clip(base + shade + texture, 0, 255)
    return img.astype(np.uint8)


def baboon_like(size: int = 256, seed: int = 1) -> np.ndarray:
    """Dense high-frequency texture (approximates Baboon's fur)."""
    rng = np.random.default_rng(seed)
    low_freq = 100 + 40 * np.sin(
        np.linspace(0, 4 * np.pi, size)
    )[:, None] * np.cos(np.linspace(0, 4 * np.pi, size))
    # Multiple scales of noise added together.
    tex = (
        rng.normal(0, 25, size=(size, size))
        + rng.normal(0, 10, size=(size, size))
    )
    img = np.clip(low_freq + tex, 0, 255)
    return img.astype(np.uint8)


def peppers_like(size: int = 256, seed: int = 2) -> np.ndarray:
    """Smooth, piecewise-constant 'peppers' with gentle shading per region."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 120, dtype=np.float64)
    # Drop a few blobs of different intensity.
    for cx, cy, r, mean in [
        (60, 60, 40, 80),
        (180, 90, 50, 190),
        (140, 180, 40, 60),
        (80, 200, 30, 210),
    ]:
        ys, xs = np.ogrid[:size, :size]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2
        # Soft edges: Gaussian-weighted.
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        weight = np.clip(1 - dist / r, 0, 1)
        img = img * (1 - weight * 0.9) + mean * (weight * 0.9)
    img += rng.normal(0, 3, size=(size, size))
    return np.clip(img, 0, 255).astype(np.uint8)


def load_classic_images(
    size: int = 256,
) -> dict[str, np.ndarray]:
    """Return a name-indexed dict of the three classic-style images."""
    return {
        "lena_like": lena_like(size=size),
        "baboon_like": baboon_like(size=size),
        "peppers_like": peppers_like(size=size),
    }


__all__ = ["lena_like", "baboon_like", "peppers_like", "load_classic_images"]
