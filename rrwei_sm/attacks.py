"""
Image attack primitives for robustness testing (Figs. 14-19 of the paper).

These functions take an 8-bit grayscale ``np.ndarray`` and return an
attacked version with the same shape/dtype.  They are dependency-light:
all filters are implemented in pure numpy; JPEG and JPEG2000 use Pillow.

Covered attacks (paper references):

* :func:`additive_gaussian_noise`         - Fig. 19, Table II (WGN)
* :func:`salt_and_pepper_noise`           - Table III
* :func:`median_filter`                   - Table III
* :func:`mean_filter`                     - Table III
* :func:`sharpen_filter`                  - Table III
* :func:`jpeg_compress`                   - Figs. 14-17
* :func:`jpeg2000_compress`               - Fig. 18
"""

from __future__ import annotations

import io
from typing import Literal

import numpy as np


# --------------------------------------------------------------------------
# Noise attacks
# --------------------------------------------------------------------------

def additive_gaussian_noise(
    image: np.ndarray, sigma: float, seed: int | None = None
) -> np.ndarray:
    """Add zero-mean Gaussian noise with standard deviation ``sigma``."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=image.shape)
    out = np.clip(image.astype(np.float64) + noise, 0, 255)
    return out.astype(image.dtype if np.issubdtype(image.dtype, np.unsignedinteger) else np.uint8)


def salt_and_pepper_noise(
    image: np.ndarray, p: float = 0.01, seed: int | None = None
) -> np.ndarray:
    """Flip a fraction ``p`` of pixels to random 0 (pepper) / 255 (salt)."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    rng = np.random.default_rng(seed)
    out = image.copy()
    h, w = image.shape
    n_pixels = int(round(p * h * w))
    if n_pixels == 0:
        return out
    flat_idx = rng.choice(h * w, size=n_pixels, replace=False)
    values = rng.choice([0, 255], size=n_pixels)
    flat = out.reshape(-1)
    flat[flat_idx] = values.astype(flat.dtype)
    return out


# --------------------------------------------------------------------------
# Linear / non-linear filters
# --------------------------------------------------------------------------

def _pad_reflect(img: np.ndarray, r: int) -> np.ndarray:
    return np.pad(img, r, mode="reflect")


def _windowed(img: np.ndarray, ksize: int) -> np.ndarray:
    """Return a (H, W, ksize, ksize) view of ``img`` with reflective padding."""
    r = ksize // 2
    padded = _pad_reflect(img.astype(np.float64), r)
    from numpy.lib.stride_tricks import sliding_window_view

    return sliding_window_view(padded, (ksize, ksize))


def median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Standard median filter with reflective boundary (Table III)."""
    if ksize % 2 != 1 or ksize < 1:
        raise ValueError("ksize must be a positive odd integer")
    w = _windowed(image, ksize)
    med = np.median(w, axis=(-2, -1))
    return np.clip(med, 0, 255).astype(np.uint8)


def mean_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Uniform mean filter (box filter) with reflective boundary."""
    if ksize % 2 != 1 or ksize < 1:
        raise ValueError("ksize must be a positive odd integer")
    w = _windowed(image, ksize)
    m = np.mean(w, axis=(-2, -1))
    return np.clip(m, 0, 255).astype(np.uint8)


def sharpen_filter(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Simple unsharp-mask style sharpen: ``img + amount * (img - blurred)``."""
    blurred = mean_filter(image, ksize=3)
    out = image.astype(np.float64) + amount * (
        image.astype(np.float64) - blurred.astype(np.float64)
    )
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Lossy compression (requires Pillow)
# --------------------------------------------------------------------------

def jpeg_compress(image: np.ndarray, quality: int = 75) -> np.ndarray:
    """Round-trip ``image`` through JPEG at the given quality factor.

    Requires Pillow (imported lazily so the module loads even in
    stripped-down environments).  ``quality`` must be in ``[1, 100]``.
    """
    if not 1 <= quality <= 100:
        raise ValueError("quality must be in [1, 100]")
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image.astype(np.uint8), mode="L").save(
        buf, format="JPEG", quality=int(quality)
    )
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L"), dtype=np.uint8)


def jpeg2000_compress(
    image: np.ndarray,
    quality_mode: Literal["rates", "dB"] = "rates",
    quality_layers: tuple[float, ...] = (20.0,),
) -> np.ndarray:
    """Round-trip ``image`` through JPEG 2000.

    ``quality_mode`` and ``quality_layers`` are forwarded to Pillow's
    Jpeg2K plugin.  Default ``rates=(20,)`` gives ~20:1 compression.
    """
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image.astype(np.uint8), mode="L").save(
        buf,
        format="JPEG2000",
        quality_mode=quality_mode,
        quality_layers=list(quality_layers),
    )
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L"), dtype=np.uint8)


__all__ = [
    "additive_gaussian_noise",
    "salt_and_pepper_noise",
    "median_filter",
    "mean_filter",
    "sharpen_filter",
    "jpeg_compress",
    "jpeg2000_compress",
]
