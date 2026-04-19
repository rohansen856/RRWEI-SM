"""End-to-end smoke test for the modernized RRWEI-SM pipeline.

Run ``python modern_demo.py`` after each modernization step; it imports
every modern module and exercises whichever pieces have already landed.
Each step extends this script with a concrete end-to-end run.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np


_OPTIONAL_MODULES = [
    "rrwei_sm_modern.crypto_scrambler",
    "rrwei_sm_modern.secret_sharing",
    "rrwei_sm_modern.stdm",
    "rrwei_sm_modern.coding",
    "rrwei_sm_modern.threshold_sharing",
    "rrwei_sm_modern.pvo",
    "rrwei_sm_modern.metrics",
    "rrwei_sm_modern.attacks",
    "rrwei_sm_modern.orchestrator",
]


def _import_matrix() -> list[tuple[str, bool]]:
    status: list[tuple[str, bool]] = []
    for name in _OPTIONAL_MODULES:
        try:
            importlib.import_module(name)
            status.append((name, True))
        except ModuleNotFoundError:
            status.append((name, False))
    return status


def _step1_encryption_roundtrip() -> None:
    """Step 1 gate: ChaCha20 scramble + additive shares, round-trip exact."""
    from datasets import load_classic_images
    from rrwei_sm_modern.crypto_scrambler import block_scramble, block_unscramble
    from rrwei_sm_modern.secret_sharing import (
        additive_combine_shares,
        additive_share_image,
    )

    covers = load_classic_images(size=64)
    print("\n== Step 1: ChaCha20 scrambler + additive shares ==")
    for name, cover in covers.items():
        scrambled = block_scramble(cover, block_size=2, seed=0xdeadbeef)
        res = additive_share_image(scrambled, n_parties=3, seed=0xcafebabe)
        combined = additive_combine_shares(res.shares)
        assert (combined == scrambled.astype(np.int64)).all(), name
        recovered = block_unscramble(combined.astype(np.uint8), 2, seed=0xdeadbeef)
        assert (recovered == cover).all(), name
        print(f"  {name:12s} shape={cover.shape} round-trip OK; n_parties={len(res.shares)}")


def main() -> int:
    import rrwei_sm_modern  # noqa: F401

    matrix = _import_matrix()
    print("modern_demo import matrix:")
    for name, ok in matrix:
        flag = "OK " if ok else "-- "
        print(f"  {flag} {name}")
    imported = {name for name, ok in matrix if ok}

    if {
        "rrwei_sm_modern.crypto_scrambler",
        "rrwei_sm_modern.secret_sharing",
    }.issubset(imported):
        _step1_encryption_roundtrip()

    return 0


if __name__ == "__main__":
    sys.exit(main())
