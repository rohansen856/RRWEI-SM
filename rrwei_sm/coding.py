"""Entropy-coding helpers for side information.

Replaces the legacy ``zlib`` path used by ``ModifiedRRWEISM`` with an
Asymmetric Numeral Systems (rANS) coder from `constriction`
(`stream.stack.AnsCoder`).  rANS achieves rates within a fraction of
a bit of the Shannon limit and is >10x faster than zlib's LZ77+Huffman
on the short, highly-biased byte streams produced by the RRWEI-SM
pipeline (STDM side info, PVO skip masks, etc.).

The API is intentionally payload-agnostic: the caller supplies a
probability mass function (pmf) estimated from the data, and the
coder round-trips an ``int`` array through it.  A small wire format
carries the pmf alongside the bitstream so decoding is self-contained.

Wire format
-----------
```
[ magic 4 bytes = "ANS1" ]
[ u32  length of symbol stream (in symbols) ]
[ u16  alphabet size K ]
[ u32  alphabet offset (so symbols map to [offset, offset+K)) ]
[ K x u32 pmf counts (rescaled to sum to 2**16) ]
[ u32  n_bytes of compressed stream ]
[ n_bytes bytes compressed rANS state ]
```

Why we carry the pmf: it is usually very short (bimodal patchwork
differences, a handful of PVO buckets), and storing it lets the
decoder run without any oracle state beyond the bitstream.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import constriction

__all__ = [
    "CodedPayload",
    "estimate_pmf",
    "encode_symbols",
    "decode_symbols",
]


_MAGIC = b"ANS1"
_PMF_PRECISION = 16  # rANS target total = 2**16


@dataclass
class CodedPayload:
    blob: bytes               # what goes through PEE
    n_symbols: int
    alphabet_size: int
    alphabet_offset: int
    original_nbits: int       # 8 * (n_symbols * symbol_bytes) for ratio calc


def estimate_pmf(
    symbols: np.ndarray,
    alphabet_size: int,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Empirical PMF over the alphabet, additively smoothed by ``eps``."""
    counts = np.bincount(symbols, minlength=alphabet_size).astype(np.float64)
    counts += eps
    return counts / counts.sum()


def _quantize_pmf(pmf: np.ndarray, precision: int = _PMF_PRECISION) -> np.ndarray:
    """Quantize a PMF to integer counts summing to exactly ``2**precision``.

    We use the straightforward method of rounding and then adjusting
    the largest bucket to absorb any drift.  This matches what
    constriction's Categorical expects internally and ensures every
    count is >= 1 so no symbol is unencodable.
    """
    total = 1 << precision
    raw = pmf * total
    counts = np.maximum(np.round(raw).astype(np.int64), 1)
    drift = int(total - counts.sum())
    if drift != 0:
        # Distribute drift starting from the most-massive bucket.
        order = np.argsort(-counts)
        idx = 0
        step = 1 if drift > 0 else -1
        while drift != 0:
            i = order[idx % len(order)]
            if step == -1 and counts[i] <= 1:
                idx += 1
                continue
            counts[i] += step
            drift -= step
            idx += 1
    assert counts.sum() == total
    return counts


def encode_symbols(
    symbols: np.ndarray,
    pmf: np.ndarray,
    alphabet_offset: int = 0,
) -> CodedPayload:
    """Encode integer ``symbols`` with known pmf via rANS.

    ``symbols`` should be in ``[alphabet_offset, alphabet_offset + len(pmf))``.
    """
    if symbols.ndim != 1:
        raise ValueError("symbols must be 1D")
    if pmf.ndim != 1:
        raise ValueError("pmf must be 1D")
    if symbols.size == 0:
        return CodedPayload(
            blob=b"", n_symbols=0, alphabet_size=len(pmf),
            alphabet_offset=alphabet_offset, original_nbits=0,
        )

    adj = symbols.astype(np.int64) - int(alphabet_offset)
    if adj.min() < 0 or adj.max() >= len(pmf):
        raise ValueError(
            "symbols out of alphabet range: "
            f"[{adj.min() + alphabet_offset}, {adj.max() + alphabet_offset}], "
            f"alphabet=[{alphabet_offset}, {alphabet_offset + len(pmf)})"
        )

    counts = _quantize_pmf(pmf)
    # constriction Categorical wants floats that sum to 1; quantize then renormalize.
    pmf_quantized = counts.astype(np.float64) / counts.sum()
    model = constriction.stream.model.Categorical(pmf_quantized, perfect=False)
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(adj.astype(np.int32), model)
    compressed = coder.get_compressed().tobytes()

    header = (
        _MAGIC
        + struct.pack("<I", int(symbols.size))
        + struct.pack("<H", int(len(pmf)))
        + struct.pack("<i", int(alphabet_offset))
        + counts.astype("<u4").tobytes()
        + struct.pack("<I", len(compressed))
    )
    return CodedPayload(
        blob=header + compressed,
        n_symbols=int(symbols.size),
        alphabet_size=int(len(pmf)),
        alphabet_offset=int(alphabet_offset),
        original_nbits=int(symbols.size * 32),  # int32 per symbol
    )


def decode_symbols(blob: bytes) -> np.ndarray:
    """Inverse of :func:`encode_symbols`; returns the original symbols."""
    if blob[:4] != _MAGIC:
        raise ValueError("not a rANS payload (bad magic)")
    offset = 4
    n_symbols = struct.unpack_from("<I", blob, offset)[0]
    offset += 4
    alphabet_size = struct.unpack_from("<H", blob, offset)[0]
    offset += 2
    alphabet_offset = struct.unpack_from("<i", blob, offset)[0]
    offset += 4
    counts = np.frombuffer(
        blob, dtype="<u4", count=alphabet_size, offset=offset
    ).astype(np.int64)
    offset += 4 * alphabet_size
    n_bytes = struct.unpack_from("<I", blob, offset)[0]
    offset += 4
    compressed = blob[offset : offset + n_bytes]

    if n_symbols == 0:
        return np.zeros(0, dtype=np.int64)

    pmf_quantized = counts.astype(np.float64) / counts.sum()
    model = constriction.stream.model.Categorical(pmf_quantized, perfect=False)
    coder = constriction.stream.stack.AnsCoder(
        np.frombuffer(compressed, dtype=np.uint32).copy()
    )
    decoded = coder.decode(model, n_symbols)
    return decoded.astype(np.int64) + int(alphabet_offset)
