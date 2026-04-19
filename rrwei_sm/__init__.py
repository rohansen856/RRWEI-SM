"""
RRWEI-SM: Robust Reversible Watermarking in Encrypted Image with Secure Multi-Party
based on Lightweight Cryptography.

Python implementation of:
    Xiong, Han, Yang, Shi. "Robust Reversible Watermarking in Encrypted Image
    With Secure Multi-Party Based on Lightweight Cryptography." IEEE Trans.
    Circuits Syst. Video Technol., vol. 32, no. 1, Jan. 2022.

Public API
----------
- RRWEISM: the basic scheme (additive secret sharing + PEE on HSB).
- ModifiedRRWEISM: the two-stage scheme (patchwork robust + PEE reversible).
- encrypt/decrypt helpers, PEE primitives, scrambling primitives.
- Metric helpers (psnr, ber) exposed via utils.
"""

from .secret_sharing import (
    additive_share_image,
    additive_share_image_k,
    additive_combine_shares,
    additive_combine_shares_k,
    split_hsb_lsb,
    recombine_hsb_lsb,
)
from .scrambling import (
    block_scramble,
    block_unscramble,
    generate_scramble_permutation,
)
from .hua_scrambling import hua_scramble, hua_unscramble, HuaKey
from .pee import PEEEmbedder, PEEExtractor
from .patchwork import PatchworkEmbedder, PatchworkExtractor
from .rrwei_sm import RRWEISM
from .modified_rrwei_sm import ModifiedRRWEISM
from .utils import psnr, ber, ssim_simple, bits_to_bytes, bytes_to_bits
from .metrics import (
    npcr,
    uaci,
    npcr_report,
    pixel_correlation,
    correlation_report,
    pee_capacity_bpp,
)
from . import attacks

__all__ = [
    "RRWEISM",
    "ModifiedRRWEISM",
    "PEEEmbedder",
    "PEEExtractor",
    "PatchworkEmbedder",
    "PatchworkExtractor",
    "additive_share_image",
    "additive_share_image_k",
    "additive_combine_shares",
    "additive_combine_shares_k",
    "split_hsb_lsb",
    "recombine_hsb_lsb",
    "block_scramble",
    "block_unscramble",
    "generate_scramble_permutation",
    "hua_scramble",
    "hua_unscramble",
    "HuaKey",
    "psnr",
    "ber",
    "ssim_simple",
    "bits_to_bytes",
    "bytes_to_bits",
    "npcr",
    "uaci",
    "npcr_report",
    "pixel_correlation",
    "correlation_report",
    "pee_capacity_bpp",
    "attacks",
]
