"""
Security + capacity metrics from Section V of Xiong et al. (2022).

Implements the numeric quantities the paper reports in its Experiments
section:

* ``NPCR`` (Eq. 33-34) -- Number of Pixel Change Rate, measures how much
  a single-pixel change to the cover propagates through the encryption.
  For a good encryption the ideal value on 8-bit images is 99.6094%.
* ``UACI`` -- Unified Average Changing Intensity, the companion metric
  typically paired with NPCR.  The paper does not formally define it
  but it is mentioned as a follow-on ([38]) and is computed here for
  completeness.
* ``pixel_correlation`` (Eq. 35) -- Pearson correlation between adjacent
  pixel pairs in the *horizontal*, *vertical* or *diagonal* direction.
  Low (~0) correlation on the encrypted image is a standard security
  indicator.
* ``capacity_formula`` (Eq. 25 / Eq. 28) -- theoretical maximum PEE
  capacity as a function of image histogram and number of layers.

All functions operate on 2-D ``np.ndarray`` inputs with values in
``[0, 255]``; ``dtype`` can be unsigned or signed int and is cast as
needed.
"""

from __future__ import annotations

from typing import Callable, Literal, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Differential security metrics
# --------------------------------------------------------------------------

def npcr(encrypted_a: np.ndarray, encrypted_b: np.ndarray) -> float:
    """Number of Pixel Change Rate between two encrypted images (Eq. 33-34).

    ``encrypted_a`` and ``encrypted_b`` are the encryptions of two covers
    that differ by **exactly one pixel** (the standard NPCR protocol).
    Returns a value in ``[0, 100]`` representing the percent of pixels
    whose value changed between the two encrypted images.

    For an 8-bit image the theoretical ideal value is
    ``(1 - 1/256) * 100% = 99.6094%``.
    """
    a = np.asarray(encrypted_a).astype(np.int64)
    b = np.asarray(encrypted_b).astype(np.int64)
    if a.shape != b.shape:
        raise ValueError(
            f"NPCR inputs must have the same shape (got {a.shape} vs {b.shape})"
        )
    return float(np.mean(a != b) * 100.0)


def uaci(
    encrypted_a: np.ndarray, encrypted_b: np.ndarray, peak: float = 255.0
) -> float:
    """Unified Average Changing Intensity between two encrypted images.

    UACI is defined as ``mean(|A - B|) / peak * 100``.  Ideal value on
    8-bit images is ~33.4635% for a random pair.
    """
    a = np.asarray(encrypted_a).astype(np.float64)
    b = np.asarray(encrypted_b).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError("UACI inputs must have the same shape")
    return float(np.mean(np.abs(a - b)) / peak * 100.0)


def npcr_report(
    cover: np.ndarray,
    encrypt_fn: Callable[[np.ndarray], np.ndarray],
    n_samples: int = 20,
    seed: int | None = 0,
) -> dict[str, float]:
    """Run the NPCR/UACI protocol over ``n_samples`` random single-pixel flips.

    Parameters
    ----------
    cover : np.ndarray
        The original (M, N) grayscale image in ``[0, 255]``.
    encrypt_fn : callable
        A deterministic function ``image -> encrypted_image`` (i.e. the
        same random key is used across all samples).  Typically a closure
        that captures the scrambling + sharing keys.
    n_samples : int
        Number of random single-pixel perturbations to average over.
    seed : int
        Seed for the perturbation RNG.

    Returns
    -------
    dict with keys ``NPCR_mean``, ``NPCR_std``, ``UACI_mean``, ``UACI_std``.
    """
    rng = np.random.default_rng(seed)
    base = encrypt_fn(cover)
    npcrs: list[float] = []
    uacis: list[float] = []
    h, w = cover.shape
    for _ in range(n_samples):
        i = int(rng.integers(0, h))
        j = int(rng.integers(0, w))
        perturbed = cover.copy()
        new_val = int(perturbed[i, j]) ^ 0b0000_0001
        perturbed[i, j] = new_val if new_val != perturbed[i, j] else (int(perturbed[i, j]) + 1) % 256
        enc = encrypt_fn(perturbed)
        npcrs.append(npcr(base, enc))
        uacis.append(uaci(base, enc))
    return {
        "NPCR_mean": float(np.mean(npcrs)),
        "NPCR_std": float(np.std(npcrs)),
        "UACI_mean": float(np.mean(uacis)),
        "UACI_std": float(np.std(uacis)),
        "n_samples": float(n_samples),
    }


# --------------------------------------------------------------------------
# Adjacent-pixel correlation (Eq. 35)
# --------------------------------------------------------------------------

Direction = Literal["horizontal", "vertical", "diagonal"]


def _sample_pairs(
    image: np.ndarray, direction: Direction, n_pairs: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape
    if direction == "horizontal":
        rows = rng.integers(0, h, size=n_pairs)
        cols = rng.integers(0, w - 1, size=n_pairs)
        x = image[rows, cols]
        y = image[rows, cols + 1]
    elif direction == "vertical":
        rows = rng.integers(0, h - 1, size=n_pairs)
        cols = rng.integers(0, w, size=n_pairs)
        x = image[rows, cols]
        y = image[rows + 1, cols]
    elif direction == "diagonal":
        rows = rng.integers(0, h - 1, size=n_pairs)
        cols = rng.integers(0, w - 1, size=n_pairs)
        x = image[rows, cols]
        y = image[rows + 1, cols + 1]
    else:
        raise ValueError(f"Unknown direction {direction!r}")
    return x.astype(np.float64), y.astype(np.float64)


def pixel_correlation(
    image: np.ndarray,
    direction: Direction = "horizontal",
    n_pairs: int = 5000,
    seed: int | None = 0,
) -> float:
    """Pearson correlation of adjacent pixel pairs (Eq. 35).

    Exactly the estimator used in the paper's Table V: sample ``n_pairs``
    random adjacent pairs in the chosen direction and compute their
    Pearson correlation coefficient.
    """
    if image.ndim != 2:
        raise ValueError("pixel_correlation expects a 2-D grayscale image")
    rng = np.random.default_rng(seed)
    x, y = _sample_pairs(image, direction, n_pairs, rng)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def correlation_report(
    image: np.ndarray, n_pairs: int = 5000, seed: int | None = 0
) -> dict[str, float]:
    """Return H/V/D correlation coefficients as a dict."""
    return {
        "corr_horizontal": pixel_correlation(image, "horizontal", n_pairs, seed),
        "corr_vertical": pixel_correlation(image, "vertical", n_pairs, seed),
        "corr_diagonal": pixel_correlation(image, "diagonal", n_pairs, seed),
    }


# --------------------------------------------------------------------------
# PEE capacity formula (Eq. 25 / Eq. 28)
# --------------------------------------------------------------------------

def pee_capacity_bpp(
    cover: np.ndarray,
    n_lsb: int = 3,
    max_layers: int = 4,
) -> dict[str, float]:
    """Theoretical PEE embedding capacity for ``cover`` (Eq. 25 / Eq. 28).

    The paper's Eq. 28 expresses the max capacity as

        EC_max = sum_{i=1..k} h_{E_i}(ME_i)

    where ``ME_i`` is the mode of the scaled prediction-error histogram
    at layer ``i`` and ``h_{E_i}(ME_i)`` is the number of blocks that
    attain that mode.  After each layer the image is re-embedded with the
    mode blocks shifted; we simulate this by repeatedly flattening the
    mode bin.

    Returns a dict with fields:
        bits_per_layer : list of bits each PEE layer would embed
        total_bits     : sum over layers
        total_bpp      : total_bits / (M*N)
    """
    from .pee import compute_scaled_error, _find_mode, _SCALE

    work = cover.astype(np.int64).copy()
    per_layer_bits: list[int] = []
    for _ in range(max_layers):
        hsb = work >> n_lsb
        e = compute_scaled_error(hsb)
        shift = 1 << n_lsb
        # Approximate overflow mask on the target pixel.
        t = hsb[0::2, 0::2] << n_lsb
        overflow = (t + shift) > 255
        ME = _find_mode(e, exclude_blocks=overflow)
        bits_this_layer = int(np.sum((e == ME) & ~overflow))
        per_layer_bits.append(bits_this_layer)
        if bits_this_layer == 0:
            break
        # Simulate: the blocks that would embed are moved to ME + SCALE
        # and the "shift" blocks pushed right; re-computing for the next
        # layer needs the image after those shifts.  For simplicity we
        # apply a +shift to all target pixels of blocks with e >= ME
        # (conservative upper bound that still models the right-ward
        # population draining after each layer).
        shift_mask = (e >= ME) & ~overflow
        coords = np.argwhere(shift_mask)
        for bi, bj in coords:
            work[2 * bi, 2 * bj] += shift
    total = int(sum(per_layer_bits))
    h, w = cover.shape
    return {
        "bits_per_layer": per_layer_bits,
        "total_bits": total,
        "total_bpp": total / (h * w),
    }


__all__ = [
    "npcr",
    "uaci",
    "npcr_report",
    "pixel_correlation",
    "correlation_report",
    "pee_capacity_bpp",
]
