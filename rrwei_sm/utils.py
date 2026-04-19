"""Small utility helpers: metrics, bit packing, RNG helpers."""

from __future__ import annotations

import numpy as np


def psnr(original: np.ndarray, test: np.ndarray, peak: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio in dB. Returns inf when images are identical."""
    original = original.astype(np.float64)
    test = test.astype(np.float64)
    mse = np.mean((original - test) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(peak / np.sqrt(mse))


def ber(extracted: np.ndarray, original: np.ndarray) -> float:
    """Bit Error Rate between two binary arrays."""
    extracted = np.asarray(extracted).astype(np.uint8).flatten()
    original = np.asarray(original).astype(np.uint8).flatten()
    n = min(len(extracted), len(original))
    if n == 0:
        return 0.0
    return float(np.sum(extracted[:n] != original[:n]) / n)


def ssim_simple(
    img1: np.ndarray, img2: np.ndarray, k1: float = 0.01, k2: float = 0.03, L: float = 255.0
) -> float:
    """A simple single-window SSIM (not multi-scale).

    Good enough for unit-test-level verification; for rigorous SSIM use
    scikit-image.  This stays dependency-free.
    """
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    mu1, mu2 = a.mean(), b.mean()
    var1, var2 = a.var(), b.var()
    cov = np.mean((a - mu1) * (b - mu2))
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    num = (2 * mu1 * mu2 + c1) * (2 * cov + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2)
    return float(num / den) if den != 0 else 1.0


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack an array of 0/1 bits (MSB-first) into bytes, padding with zeros."""
    bits = np.asarray(bits, dtype=np.uint8).flatten()
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    grouped = bits.reshape(-1, 8)
    vals = (grouped * (1 << np.arange(7, -1, -1))).sum(axis=1).astype(np.uint8)
    return vals.tobytes()


def bytes_to_bits(data: bytes, n_bits: int | None = None) -> np.ndarray:
    """Convert a bytes object into an MSB-first 0/1 numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr)
    if n_bits is not None:
        bits = bits[:n_bits]
    return bits.astype(np.uint8)
