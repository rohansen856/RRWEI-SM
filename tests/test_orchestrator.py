"""End-to-end orchestrator tests."""

from __future__ import annotations

import numpy as np
import pytest

from rrwei_sm.orchestrator import ModernScheme
from rrwei_sm.stdm import STDMConfig


def test_encrypt_decrypt_round_trip(classic_covers):
    scheme = ModernScheme(k=2, n=3)
    for name, cover in classic_covers.items():
        _, shares = scheme.encrypt(cover, scramble_seed=1, share_seed=2)
        rec = scheme.decrypt(shares, [0, 1], scramble_seed=1)
        assert (rec == cover).all(), name


def test_full_pipeline_recovers_robust_and_payload_bits(classic_covers):
    scheme = ModernScheme(
        k=2, n=3,
        stdm_config=STDMConfig(delta=60.0, n_coeffs=4, seed=0),
        max_pvo_layers=3,
    )
    cover = classic_covers["lena_like"]  # 64x64 -> 64 STDM blocks max
    rng = np.random.default_rng(3)
    robust = rng.integers(0, 2, size=32, dtype=np.uint8)
    payload = rng.integers(0, 2, size=256, dtype=np.uint8)
    result = scheme.embed(cover, robust, payload, scramble_seed=7)

    # Robust bits survive via blind extraction.
    out_robust = scheme.extract_robust(
        result.marked_cover, result.stdm_side, scramble_seed=7
    )
    assert (out_robust == robust).all()

    # Reversible payload is exactly recoverable on the intact cover.
    _, out_payload = scheme.extract_reversible(
        result.marked_cover, result.pvo_sides, scramble_seed=7
    )
    assert out_payload.size == result.n_reversible_bits
    assert (out_payload == payload[: result.n_reversible_bits]).all()


def test_robust_bits_survive_jpeg_q40_via_orchestrator(classic_covers):
    import io

    from PIL import Image

    scheme = ModernScheme(
        k=2, n=3,
        stdm_config=STDMConfig(delta=60.0, n_coeffs=4, seed=0),
        max_pvo_layers=3,
    )
    cover = classic_covers["lena_like"]
    rng = np.random.default_rng(4)
    robust = rng.integers(0, 2, size=64, dtype=np.uint8)
    payload = rng.integers(0, 2, size=128, dtype=np.uint8)
    result = scheme.embed(cover, robust, payload, scramble_seed=11)

    buf = io.BytesIO()
    Image.fromarray(result.marked_cover).save(buf, format="JPEG", quality=40)
    buf.seek(0)
    attacked = np.array(Image.open(buf).convert("L"), dtype=np.uint8)

    out = scheme.extract_robust(attacked, result.stdm_side, scramble_seed=11)
    ber = float((out != robust).mean())
    assert ber <= 0.10, ber
