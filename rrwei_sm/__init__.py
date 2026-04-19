"""Robust Reversible Watermarking in Encrypted Image: modern rewrite.

Public surface (import-through re-export):

* :class:`ModernScheme`  -- end-to-end orchestrator
* :mod:`attacks`         -- classical + modern attack suite
* :mod:`coding`          -- rANS side-info coder
* :mod:`crypto_scrambler` -- ChaCha20 block scrambler
* :mod:`metrics`         -- PSNR / SSIM / LPIPS / DISTS
* :mod:`pvo`             -- PVO + pairwise PEE reversible predictor
* :mod:`secret_sharing`  -- keystream-based additive sharing (2-of-2)
* :mod:`stdm`            -- spread-transform dither modulation robust WM
* :mod:`threshold_sharing` -- (k, n) replicated threshold sharing
"""

from __future__ import annotations

from . import (
    attacks,
    coding,
    crypto_scrambler,
    metrics,
    pvo,
    secret_sharing,
    stdm,
    threshold_sharing,
)
from .orchestrator import EmbedResult, ModernScheme

__all__ = [
    "ModernScheme",
    "EmbedResult",
    "attacks",
    "coding",
    "crypto_scrambler",
    "metrics",
    "pvo",
    "secret_sharing",
    "stdm",
    "threshold_sharing",
]
