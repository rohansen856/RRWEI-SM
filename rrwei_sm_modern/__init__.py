"""Modern RRWEI-SM pipeline.

A cryptographically-grounded rewrite of the paper's scheme with:

* ChaCha20-keystream scrambler + additive masks (replaces Fisher-Yates + NumPy RNG).
* Spread Transform Dither Modulation (STDM) robust watermark on the DCT
  plane (replaces patchwork).
* rANS-coded side information (replaces zlib).
* Replicated (t, n) threshold secret sharing (replaces additive k-of-k).
* PVO + pairwise PEE reversible predictor (replaces 3-neighbour PEE).
* LPIPS / DISTS quality metrics.
* Modern attack suite: neural-codec proxy, SR cascade, diffusion-proxy.

The API is exposed through :class:`ModernScheme`.
"""

from __future__ import annotations

# Public API is populated as modules land during the modernization
# steps; until then, importing the package is a cheap smoke test that
# ensures the package layout is valid.

__all__: list[str] = []
