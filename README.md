# RRWEI-SM: Robust Reversible Watermarking in Encrypted Image with Secure Multi-Party

A dependency-light Python implementation of the schemes in:

> Lizhi Xiong, Xiao Han, Ching-Nung Yang, Yun-Qing Shi.
> *"Robust Reversible Watermarking in Encrypted Image With Secure
> Multi-Party Based on Lightweight Cryptography."*
> IEEE Transactions on Circuits and Systems for Video Technology,
> Vol. 32, No. 1, pp. 75-91, January 2022.

The paper proposes two schemes:

1. **RRWEI-SM** -- lightweight image encryption (additive secret sharing
   + block-level scrambling) followed by HSB-plane Prediction Error
   Expansion (PEE) performed via a Secure Multi-Party Computation
   protocol.
2. **Modified RRWEI-SM** -- a two-stage variant that first embeds a
   patchwork robust watermark, then reversibly embeds the side
   information via PEE.

Both schemes:

* preserve the privacy of the cover image (additive shares act as a
  one-time pad),
* are fully **separable** (extraction order with respect to decryption
  is free),
* support **k-party** copyright protection (`n_parties >= 2`),
* recover the cover image **exactly** when the marked image is intact,
* have **robust** watermark extraction under mild attacks (Modified
  scheme).

Full transparency about what could and could not be reproduced verbatim
from the paper lives in [LIMITATIONS.md](LIMITATIONS.md).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# End-to-end demo on a built-in synthetic cover
python demo.py

# On your own image (grayscale PNG/JPEG; cropped to a multiple of 4)
python demo.py path/to/image.png

# Full test suite (96 tests, ~0.5s)
pytest tests/ -q

# Reproduce all experimental results from the paper
python evaluate.py --size 128
python benchmark.py --sizes 64 128 256 --n-repeats 3
python -m figures.make_all
```

## What's implemented (paper items 1-14)

| # | Paper component                                    | Module / file                              |
|---|----------------------------------------------------|--------------------------------------------|
| 1 | Additive secret sharing (2-party, k-party)         | `rrwei_sm/secret_sharing.py`               |
| 2 | HSB/LSB split with carry-aware invariants          | `rrwei_sm/secret_sharing.py`               |
| 3 | Block-level scrambling (random permutation)        | `rrwei_sm/scrambling.py`                   |
| 4 | Block-level scrambling (Hua et al. 2018, 2D-LSCM)  | `rrwei_sm/hua_scrambling.py`               |
| 5 | Block-level HSB PEE + histogram shift              | `rrwei_sm/pee.py`                          |
| 6 | SMC contribution additivity self-check             | `rrwei_sm/pee.py`                          |
| 7 | Basic RRWEI-SM (separable, k-party)                | `rrwei_sm/rrwei_sm.py`                     |
| 8 | Patchwork robust watermark (full-plane + HSB-plane)| `rrwei_sm/patchwork.py`, `modified_rrwei_sm.py` |
| 9 | Two-stage Modified RRWEI-SM                        | `rrwei_sm/modified_rrwei_sm.py`            |
| 10| zlib-based side-info compression                   | `rrwei_sm/modified_rrwei_sm.py`            |
| 11| PSNR, SSIM, BER, NPCR, UACI, H/V/D correlation, Eq. 28 capacity | `rrwei_sm/utils.py`, `metrics.py` |
| 12| Gaussian / S&P / median / mean / sharpen / JPEG / JPEG2000 attacks | `rrwei_sm/attacks.py` |
| 13| Evaluation harness over classic images             | `datasets/`, `evaluate.py`                 |
| 14| Benchmarks + matplotlib PNG figures                | `benchmark.py`, `figures/make_*.py`        |

## Architecture

```
rrwei_sm/
├── secret_sharing.py       # 2-party + k-party additive share, HSB/LSB split (Eqs. 7-14)
├── scrambling.py           # random block permutation (seeded)
├── hua_scrambling.py       # Hua et al. 2018 2D-LSCM chaotic block scrambler (ref [38])
├── pee.py                  # HSB-plane PEE primitives (Eqs. 15-22) + SMC self-check
├── patchwork.py            # patchwork robust watermark (Eqs. 23-27)
├── rrwei_sm.py             # basic RRWEI-SM orchestrator (encrypt / embed / decrypt)
├── modified_rrwei_sm.py    # Modified RRWEI-SM (two-stage, HSB-plane-aware)
├── metrics.py              # NPCR, UACI, H/V/D correlation, PEE capacity (Eq. 28)
├── attacks.py              # noise, filter, JPEG/JPEG2000 attack primitives
└── utils.py                # PSNR, BER, SSIM, bit-packing helpers
datasets/                   # synthetic Lena/Baboon/Peppers-like covers
figures/                    # matplotlib figure generators
evaluate.py                 # dataset-wide evaluation (quality + robustness + metrics)
benchmark.py                # timing harness for O(n) verification
```

### Basic API

```python
from rrwei_sm import RRWEISM

scheme = RRWEISM(n_lsb=3, max_layers=4, scrambler="hua", n_parties=2)

share1, share2, keys = scheme.encrypt(cover, key_scramble=7, key_share=13)
marked1, marked2, side = scheme.embed(share1, share2, bits)

# Separable: either order works.
rec_s1, rec_s2, ext = scheme.extract_before_decrypt(marked1, marked2, side)
cover_back, ext     = scheme.extract_after_decrypt(marked1, marked2, keys, side)

# k-party (n_parties >= 2)
scheme_k = RRWEISM(n_parties=4)
shares, keys = scheme_k.encrypt_k(cover, key_scramble=0, key_share=0)
marked, side = scheme_k.embed_k(shares, owner_idx=0, bits=bits)
cover_back = scheme_k.decrypt_k(marked, keys)
```

### Modified RRWEI-SM API

```python
from rrwei_sm import ModifiedRRWEISM

scheme = ModifiedRRWEISM(
    n_lsb=3,
    patchwork_m=64,
    patchwork_T=5,
    patchwork_plane="hsb",        # paper's mode
    compress_side_info=True,      # zlib-compress patchwork side info
    scrambler="hua",
    n_parties=2,
)

s1, s2, keys = scheme.encrypt(cover, key_scramble=1, key_share=2)
ms1, ms2, side = scheme.embed(s1, s2, robust_bits)

# Quick robust path (survives mild attacks, no exact recovery needed)
bits = scheme.extract_robust_after_decrypt(ms1, ms2, keys, side)

# Full path: exact cover recovery + robust bits (requires intact image)
cover_back, bits = scheme.extract_and_recover_after_decrypt(ms1, ms2, keys, side)
```

## Reproducing paper results

The `evaluate.py` script runs the end-to-end pipeline on synthetic
Lena/Baboon/Peppers-like covers and prints PSNR, SSIM, PEE capacity,
pixel correlation, NPCR/UACI, and robustness BER against every attack
primitive. Example (96x96 covers, 32 robust bits):

```
==== lena_like ====
  share/cover correlation    : +0.003  (ideal: 0)
  combined corr (H/V/D)      : +0.51 / +0.51 / +0.24
  PSNR(cover, marked)        : 39.4 dB
  SSIM(cover, marked)        : 0.998
  PEE capacity (bpp)         : 0.133
  Exact recovery             : True
  BER @ Gaussian sigma=10    : 0.000
  BER @ JPEG q=50            : 0.000
  BER @ JPEG2000 r=20        : 0.000
  BER median/mean/sharpen/SP : 0.100 / 0.000 / 0.000 / 0.000
```

`benchmark.py` confirms **O(n)** complexity for all three operations
(encrypt / decrypt / embed) across image sizes; the per-pixel time is
roughly constant within 2x.

`figures/make_all.py` generates five PNGs in `figures/out/`:

* `wgn_ber.png`            -- 1-BER vs. Gaussian noise sigma
* `jpeg_ber.png`           -- 1-BER vs. JPEG quality factor
* `capacity_psnr.png`      -- PSNR vs. embedding capacity tradeoff
* `benchmark_on.png`       -- per-pixel timing vs. image size (O(n) check)
* `encrypted_visuals.png`  -- cover / share / combined encrypted image + histograms

## Tests

```
tests/
├── test_secret_sharing.py           # Eqs. 7-14, carry invariants
├── test_scrambling.py               # random permutation round-trip
├── test_hua_scrambling.py           # 2D-LSCM scrambling
├── test_pee.py                      # Eqs. 15-22 + SMC contribution additivity
├── test_patchwork.py                # Eqs. 23-27
├── test_rrwei_sm.py                 # basic scheme, both extraction orders
├── test_modified_rrwei_sm.py        # two-stage scheme round-trip + capacity error
├── test_hsb_patchwork_and_compression.py   # HSB-plane patchwork + zlib side info
├── test_multiparty.py               # k-party additive sharing & k-party RRWEI-SM
├── test_metrics.py                  # NPCR, UACI, correlation, Eq. 28
├── test_attacks.py                  # Gaussian, S&P, median, mean, sharpen, JPEG, JPEG2000
├── test_robustness.py               # legacy noise-robustness
├── test_robustness_attacks.py       # Modified RRWEI-SM robustness under every attack
├── test_datasets.py                 # synthetic classic covers
└── test_utils.py                    # metric helpers
```

96 tests, full suite runs in under a second.

## Known limitations

See [LIMITATIONS.md](LIMITATIONS.md) for the full list. The remaining
documented deviations after this round:

1. **NPCR protocol ambiguity**: the paper reports NPCR ~99% for 2x2
   scrambling, whereas our same-key NPCR is effectively `1/n` because
   additive sharing + a permutation is not diffusing. We implement and
   expose the metric; the paper's published value appears to derive from
   *random-key* NPCR (different shares/scramble key per sample) -- see
   `rrwei_sm/metrics.py` docstring.
2. **SMC security proofs** (Theorem 1) are not cryptographically
   reproduced; only the numerical invariants used in the simulator
   argument are self-checked.
3. Classic test images are synthetic Lena/Baboon/Peppers stand-ins
   (generated from gradients + textures) so the repo has no image
   redistribution footprint.
