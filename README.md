# RRWEI-SM (modernized)

A dependency-light Python implementation of the paper

> Lizhi Xiong, Xiao Han, Ching-Nung Yang, Yun-Qing Shi.
> *"Robust Reversible Watermarking in Encrypted Image With Secure
> Multi-Party Based on Lightweight Cryptography."*
> IEEE Transactions on Circuits and Systems for Video Technology,
> Vol. 32, No. 1, pp. 75-91, January 2022.

This branch (`modernization`) replaces each legacy building block with a
modern equivalent while preserving the paper's architecture:

| Stage                        | Paper primitive                 | Modern replacement                                    |
|------------------------------|---------------------------------|-------------------------------------------------------|
| Encryption                   | Additive share + Hua 2D-LSCM    | **ChaCha20** stream + block permutation               |
| Secret sharing               | 2-party additive                | **(k, n) replicated** shares                          |
| Robust watermark             | Patchwork on HSB plane          | **STDM** (quantized DCT dither modulation)            |
| Side-information compression | zlib                            | **rANS** entropy coder                                |
| Reversible payload           | HSB Prediction-Error Expansion  | **PVO + pairwise PEE** (Numba-JIT)                    |
| Imperceptibility             | PSNR / SSIM                     | PSNR / SSIM + **LPIPS** + **DISTS-proxy**             |
| Attacks                      | Gaussian / S&P / JPEG / JPEG2K  | + **neural codec**, **SR cascade**, **diffusion** proxies |

See [`MODERNIZATION.md`](MODERNIZATION.md) for the component-by-component
rationale and measured gains, and [`FIGURES.md`](FIGURES.md) for a
research-paper-style empirical evaluation of every committed figure.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# End-to-end demo on a synthetic cover
python demo.py

# Full test suite (unit tests for every modern module)
pytest tests/ -q
```

## Modern API

```python
import numpy as np
from rrwei_sm import ModernScheme, STDMConfig

rng = np.random.default_rng(0)
cover = rng.integers(0, 256, (256, 256), dtype=np.uint8)

scheme = ModernScheme(
    stdm=STDMConfig(delta=60, n_bits=128),   # robust watermark strength / length
    n_shares=3, threshold=2,                 # (2, 3) replicated sharing
    block_size=8,                             # ChaCha20 scramble block size
    key=b"0" * 32,                           # 32-byte ChaCha20 key
)

robust_bits = rng.integers(0, 2, 128, dtype=np.uint8)
payload_bits = rng.integers(0, 2, 4096, dtype=np.uint8)

result = scheme.embed(cover, robust_bits=robust_bits, payload_bits=payload_bits)
# result -> {"marked", "side_info", "shares", "psnr_db", ...}

rec_robust = scheme.extract_robust(result["marked"], result["side_info"])
rec_cover, rec_payload = scheme.extract_reversible(result["marked"], result["side_info"])
```

Public re-exports live in [`rrwei_sm/__init__.py`](rrwei_sm/__init__.py).

## Repository layout

```
rrwei_sm/
├── crypto_scrambler.py     # ChaCha20 block scrambler
├── secret_sharing.py       # legacy 2-party additive (kept for ablation)
├── threshold_sharing.py    # (k, n) replicated secret shares
├── stdm.py                 # Spread Transform Dither Modulation
├── coding.py               # rANS entropy coder for side info
├── pvo.py                  # Pixel-Value-Ordering PEE (Numba-JIT)
├── metrics.py              # PSNR, SSIM, BER, LPIPS, DISTS-proxy, NPCR/UACI
├── attacks.py              # classical + neural + SR + diffusion proxies
└── orchestrator.py         # ModernScheme (end-to-end pipeline)

tests/                      # unit tests, one per module (pytest)
figures/                    # matplotlib generators for every plot in FIGURES.md
figures/out/                # committed scripted outputs (.png)
```

## Pipeline tools (not core library code)

These scripts wrap the library to produce the empirical artifacts in
`FIGURES.md` and `MODERNIZATION.md`. They are the "pipeline glue" --
nothing in `rrwei_sm/` depends on them.

### `demo.py` -- minimal round-trip demo

```bash
python demo.py                 # synthetic cover
python demo.py path/to.png     # arbitrary grayscale image
```

Prints PSNR/SSIM, robust BER, PVO payload bits-per-pixel, and writes a
side-by-side PNG to `figures/out/demo_cover_marked.png`.

### `evaluate.py` -- reproducibility harness

```bash
python evaluate.py --size 128
```

Runs the full pipeline on synthetic Lena/Baboon/Peppers-like covers and
prints PSNR, SSIM, PEE capacity, pixel correlation, NPCR/UACI, and
robustness BER against every attack primitive in `rrwei_sm.attacks`.

### `real_image_harness.py` -- canonical real-image acceptance suite

```bash
python real_image_harness.py
```

Downloads (and caches under `assets/test_images/`) Lena, Baboon and
Peppers from a public mirror, runs the modernized pipeline on each, and
**enforces acceptance thresholds** (PSNR >= 38 dB, SSIM >= 0.97,
exact reversible recovery, BER == 0 under Gaussian sigma=10 and JPEG
q=40). Exits non-zero if any threshold fails. Also emits
`figures/out/real_images.png`.

### `run_custom_image.py` -- run the pipeline on your own image

```bash
python run_custom_image.py path/to/image.png
python run_custom_image.py path/to/image.jpg --size 512
```

Loads an arbitrary image, converts to 8-bit grayscale, center-crops to
the largest square that is a multiple of 8, runs the full pipeline, and
saves into `figures/out/<stem>_artifacts/`:

```
<stem>__01_cover.png                       # preprocessed cover
<stem>__02_scrambled.png                   # ChaCha20 block scramble
<stem>__03_marked.png                      # after STDM + PVO embedding
<stem>__04_attacked_gaussian_s10.png       # AWGN sigma=10 attack
<stem>__05_attacked_jpeg_q40.png           # JPEG q=40 attack
<stem>__06_diff_abs_x25.png                # |cover - marked| * 25
```

Plus a 3-panel summary at `figures/out/custom_image.png`.
Metrics are echoed to stdout.

### `benchmark_modern.py` -- runtime harness

```bash
python benchmark_modern.py --sizes 64 128 256 512 --n-repeats 3
```

Measures per-pixel wall time for encrypt, decrypt, STDM embed, PVO
embed, and reports the **Numba speedup** of PVO vs. a pure-Python
reference. `benchmark.py` is a thin compatibility shim around this.

### `figures/make_*.py` -- individual plot generators

Each file regenerates one PNG under `figures/out/`:

| Script                              | Output                  | What it shows                                 |
|-------------------------------------|-------------------------|-----------------------------------------------|
| `figures/make_wgn_ber.py`           | `wgn_ber.png`           | 1-BER vs. Gaussian noise sigma                |
| `figures/make_jpeg_ber.py`          | `jpeg_ber.png`          | 1-BER vs. JPEG quality                        |
| `figures/make_robustness_sweep.py`  | `robustness_sweep.png`  | BER across the full modern attack suite       |
| `figures/make_capacity.py`          | `capacity_psnr.png`     | PVO capacity vs. PSNR tradeoff                |
| `figures/make_benchmark.py`         | `benchmark_on.png`      | per-pixel time vs. image size (O(n) check)    |
| `figures/make_pvo_speedup.py`       | `pvo_speedup.png`       | Numba JIT vs. pure-Python PVO                 |
| `figures/make_encrypted_visuals.py` | `encrypted_visuals.png` | cover/share/combined + histograms             |
| `figures/make_quality_summary.py`   | `quality_summary.png`   | PSNR/SSIM/DISTS/LPIPS on Lena/Baboon/Peppers  |

Regenerate everything at once:

```bash
python -m figures.make_all
```

All scripted PNGs are committed so `FIGURES.md` renders on GitHub. Only
user-specific outputs (`custom_image.png`, `*_artifacts/`) are
gitignored.

## Tests

```
tests/
├── test_crypto_scrambler.py   # ChaCha20 scramble round-trip + diffusion
├── test_threshold_sharing.py  # (k, n) replicated shares
├── test_stdm.py               # STDM embed/extract under perturbation
├── test_coding.py             # rANS round-trip
├── test_pvo.py                # PVO+PEE reversibility + capacity
├── test_metrics.py            # PSNR, SSIM, LPIPS, DISTS-proxy, NPCR/UACI
├── test_attacks.py            # classical + neural/SR/diffusion proxies
├── test_secret_sharing.py     # legacy additive share (kept for ablation)
└── test_orchestrator.py       # ModernScheme end-to-end round-trip
```

Run with `pytest tests/ -q`.

## Known limitations

See [`LIMITATIONS.md`](LIMITATIONS.md) for the full list. Highlights:

1. Neural / SR / diffusion attack stages are deterministic CPU-side
   *proxies*, not the actual published models. They match the
   qualitative shape of each attack but underestimate worst-case drift
   in any given deployment.
2. The canonical Lena/Baboon/Peppers images are downloaded from a
   public mirror by `real_image_harness.py`; if that mirror is down the
   harness falls back to the synthetic covers in `datasets/`.
3. LPIPS requires `torch` + the `lpips` package. When unavailable,
   `rrwei_sm.metrics.lpips_distance` returns `None` and
   `make_quality_summary.py` degrades gracefully.
