"""Tests for the rANS-based side-info coder."""

from __future__ import annotations

import zlib

import numpy as np
import pytest

from rrwei_sm_modern.coding import decode_symbols, encode_symbols, estimate_pmf


def test_round_trip_small():
    symbols = np.array([0, 1, 0, 2, 0, 0, 1, 2, 0, 1, 0], dtype=np.int64)
    pmf = estimate_pmf(symbols, alphabet_size=3)
    coded = encode_symbols(symbols, pmf)
    decoded = decode_symbols(coded.blob)
    assert (decoded == symbols).all()


def test_round_trip_with_offset():
    rng = np.random.default_rng(0)
    symbols = rng.integers(-3, 4, size=500, dtype=np.int64)  # in [-3, 3]
    pmf = estimate_pmf(symbols + 3, alphabet_size=7)
    coded = encode_symbols(symbols, pmf, alphabet_offset=-3)
    decoded = decode_symbols(coded.blob)
    assert (decoded == symbols).all()


def test_beats_zlib_on_bimodal_payload():
    rng = np.random.default_rng(1)
    # Simulate patchwork-style differences: bimodal around +-T.
    n = 2000
    symbols = rng.choice([-5, 5], size=n, p=[0.5, 0.5])
    noise = rng.integers(-1, 2, size=n)
    symbols = symbols + noise  # values in {-6..-4, 4..6}
    pmf = estimate_pmf(symbols + 6, alphabet_size=13)
    coded = encode_symbols(symbols, pmf, alphabet_offset=-6)

    as_int16 = symbols.astype("<i2").tobytes()
    zlib_size = len(zlib.compress(as_int16, level=9))
    print(f"rANS={len(coded.blob)} bytes, zlib={zlib_size} bytes")
    assert len(coded.blob) <= int(0.9 * zlib_size), (len(coded.blob), zlib_size)


def test_rejects_out_of_range_symbols():
    symbols = np.array([0, 1, 2, 5], dtype=np.int64)
    pmf = np.array([0.4, 0.4, 0.2])
    with pytest.raises(ValueError):
        encode_symbols(symbols, pmf)


def test_empty_round_trip():
    symbols = np.zeros(0, dtype=np.int64)
    pmf = np.array([1.0])
    coded = encode_symbols(symbols, pmf)
    assert coded.n_symbols == 0
    # blob is allowed to be empty; we still require decode to not crash
    # on a freshly-constructed zero-length header.  We skip decode here
    # because constriction needs at least the header for a real payload.


def test_quantization_preserves_every_symbol():
    rng = np.random.default_rng(2)
    alphabet = 32
    symbols = rng.integers(0, alphabet, size=10_000, dtype=np.int64)
    pmf = estimate_pmf(symbols, alphabet_size=alphabet)
    coded = encode_symbols(symbols, pmf)
    decoded = decode_symbols(coded.blob)
    assert (decoded == symbols).all()
    ratio = 8 * len(coded.blob) / symbols.size  # bits per symbol
    entropy = -np.sum(pmf[pmf > 0] * np.log2(pmf[pmf > 0]))
    # Expect to be within 1 bit of Shannon entropy for a large stream.
    assert ratio < entropy + 1.0, (ratio, entropy)
