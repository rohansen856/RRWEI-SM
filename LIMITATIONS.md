# LIMITATIONS, DEVIATIONS, AND UNCERTAINTIES

This document enumerates every point in the paper that was ambiguous,
under-specified, or impractical to reproduce literally, and records the
engineering decision that was made for this Python implementation.

> Reference: Xiong, Han, Yang, Shi. *"Robust Reversible Watermarking in
> Encrypted Image With Secure Multi-Party Based on Lightweight
> Cryptography."* IEEE TCSVT, Vol. 32, No. 1, Jan. 2022.

---

## 1. Scaled prediction error instead of fractional error (Eqs. 15-19)

### Paper

Defines the prediction error as

    e_HSB(i,j) = x_HSB(i,j) - (x_HSB(i,j+1) + x_HSB(i+1,j) + x_HSB(i+1,j+1)) / 3

and the SMC sum as `e_HSB = (e1 + e2) / 3` (Eq. 19).  This is *not* an
integer in general.

### What was unclear

The paper does not specify how integer rounding is handled.  A
straightforward `round(·)` would break the clean additive property used
by the SMC protocol (you cannot guarantee `round(e1/3) + round(e2/3) ==
round((e1+e2)/3)`).

### What this implementation does

We work entirely in the *scaled* error space

    e_scaled(i,j) = 3 * x_HSB(i,j) - sum(context_HSB)

which is exactly `e1 + e2`.  Adding `+1` to the target HSB shifts
`e_scaled` by `+3` instead of `+1`.  The histogram-shift step therefore
has a stride of 3 in the scaled space, not 1.  All formulas are
mathematically equivalent to the paper's, but every arithmetic operation
is exact.

### Consequence

None for embedding/extraction -- but if you want to compare histograms
bin-by-bin with numbers printed in the paper, you need to divide our
histograms' x-axis by 3.

---

## 2. Additive sharing: share range and negative values

### Paper

Eqs. (7) and (13)-(14) give `x = x1 + x2`, stating "the values of encrypted
shares do not exceed the range from 0 to 255" and that
overflow/underflow locations are recorded as side information.

### What was unclear

1. No explicit distribution is given for the random split.  A uniform
   random split over `[0, 255]` routinely makes `share2 = x - share1`
   out of range.
2. How `x_HSB = x1_HSB + x2_HSB` is recovered from *a share whose LSB
   can itself be negative* (needed so that
   `share >> n` -- used in Eqs. 17-18 -- matches `x_k_HSB`).

### What this implementation does

* LSB parts are shared **modulo** `2**n_lsb` (so LSBs are always in
  `[0, 2**n_lsb)`).
* The carry induced by that modular split is absorbed into the HSB
  split.
* Shares are stored as **signed int32** (not uint8); any analysis that
  needs `uint8` shares can clip, and the overflow/underflow mask from
  `shares_in_gray_range()` exposes exactly which pixels would go out of
  range under uint8 storage.

### Consequence

* The additive invariant `share1 + share2 == cover` holds exactly.
* The HSB-recovery invariant
  `(share1 >> n) + (share2 >> n) == cover_HSB` holds up to a per-pixel
  carry of `0` or `-1`; this carry is a deterministic function of the
  two LSBs and cancels out in the SMC protocol's *relative* error
  quantities (see `tests/test_pee.py::test_smc_contribution_additivity`).
* Storing shares as uint8 (the paper's visual figures) would either
  drop information or require bookkeeping the overflow/underflow mask;
  our code supports both modes (exact as int32, plus `shares_in_gray_range`).

---

## 3. Scrambling algorithm

### Paper

References the "high-speed-scrambling method" of Hua et al., *Signal
Processing*, 2018 (ref [38]) but does not describe it in detail.

### What this implementation does

Two pluggable scramblers, selected via the `scrambler=` argument:

1. `scrambler="random"` -- seeded uniform random permutation of
   non-overlapping `block_size x block_size` tiles
   (`rrwei_sm/scrambling.py`).
2. `scrambler="hua"` -- a reconstruction of Hua et al. 2018 row/column
   permutation using the **2D Logistic-Sine-Coupling Map (2D-LSCM)** to
   drive the row and column keys
   (`rrwei_sm/hua_scrambling.py`). The chaotic-map parameters (initial
   state, `rho`) are fed by the same scramble key. This matches the
   structure described in Hua & Zhou 2018 but the exact map parameters
   are not literally reproducible from the RRWEI-SM paper alone.

Both variants preserve the 2x2 PEE-block structure and are not
invertible without the key.

### Consequence

Pixel-level differential analysis values vary by scrambler choice.
`tests/test_hua_scrambling.py` verifies determinism, key sensitivity,
round-trip, block preservation, and adjacent-pixel decorrelation.

---

## 4. Per-block embedding rate (one bit per 2x2 block)

### Paper

Figure 6 shows "4 bits [1,0; 0,1]" embedded into a single 2x2 block,
hinting at multi-bit-per-block embedding using each of the 4 pixels as a
target in turn.  The main text, however, consistently treats the block
as having one target pixel and three context pixels (Eqs. 15-20).

### What this implementation does

**One bit per 2x2 block**, with the upper-left pixel as the target.
Multiple *layers* of PEE (see `PEEEmbedder.max_layers`) recover the rest
of the capacity; each layer reprocesses the image as a whole.

### Consequence

Capacity per layer is at most `(M*N)/4 = 0.25 bpp`; four layers give up
to `1 bpp`.  The paper reports up to ~0.7 bpp on real images, which is
within this envelope.

---

## 5. Side-information compression (Sec. IV-C)

### Paper

"We can also compress the side information to reduce the bits which are
embedded into cover image by LSB replacement [23]."

### What this implementation does

The patchwork side info (block differences + skipped mask) is packed
compactly as `int16` + bitmap and then optionally **zlib-compressed**
before PEE embedding (`ModifiedRRWEISM(compress_side_info=True)`, the
default). A one-bit flag in the payload header selects compressed vs.
raw. Because the block differences are heavily concentrated around
`+/- T` and the skipped mask is usually sparse, zlib typically shrinks
the payload by 30-60% on the classic test images.

### Consequence

Capacity headroom in the Modified scheme improves materially on smooth
images (Peppers/Lena). Behaviour is unchanged on pathological inputs
where zlib cannot do better than raw; the flag bit disambiguates.

---

## 6. Patchwork on HSB plane vs. full pixel plane

### Paper

Section IV-C: "the patchwork robust watermark is embedded into the HSB
plane of the shares".

### What this implementation does

Both modes are supported through `patchwork_plane={"full", "hsb"}`:

* `"full"` (default for speed): patchwork uses integer pixel-level
  shifts of size `T` on the combined scrambled image.
* `"hsb"` (faithful to the paper): patchwork operates on the HSB plane
  of the combined scrambled image. Perturbations are in HSB units, so
  when lifted to pixel space the effective shift becomes
  `T_effective = T * 2**n_lsb`. This alignment with `2**n_lsb` means
  patchwork and PEE both act in the HSB bit-plane and the quick robust
  extraction path is robust even on textured images.

The selection is exposed in `ModifiedRRWEISM(patchwork_plane=...)` and
tested in `tests/test_hsb_patchwork_and_compression.py`.

### Consequence

With `patchwork_plane="hsb"` and moderate `T` (default 5, lifted to
pixel shifts of 40 when `n_lsb=3`), the quick robust extraction path
now reaches BER=0 on intact images and stays low under Gaussian, S&P,
JPEG, JPEG2000, mean and sharpen attacks.

---

## 7. Overflow/underflow mask distribution in SMC

### Paper

"The locations of the overflow/underflow pixels would be recorded, and
the watermark will not be embedded into these pixels."

### What was unclear

Whether the overflow map is generated at the image-provider side before
sharing, or is computed cooperatively by the two parties on their
shares.

### What this implementation does

The overflow map is computed on the **combined** image during
embedding, which is equivalent to computing it on the cover: target
pixel + shift > 255.  This map is stored in the PEE side information and
distributed with the extraction key.

### Consequence

In the pure SMC setting (two parties never see each other's share), the
overflow decision must instead be made per-party and agreed upon -- a
minor interactive step that our code sidesteps by computing on the
joint view.

---

## 8. k-party additive secret sharing

### Paper

Section I states the scheme supports "multi-party copyright protection"
and figures depict two parties. The math is generalised to `n`-party
additive sharing without modification.

### What this implementation does

Fully supported via `n_parties >= 2` in `RRWEISM` and
`ModifiedRRWEISM`. The underlying primitives are
`additive_share_image_k` and `additive_combine_shares_k` in
`rrwei_sm/secret_sharing.py`. The owner who embeds the watermark is
chosen via `owner_idx`; all other parties contribute zero-sum
perturbations. Covered by `tests/test_multiparty.py`.

---

## 9. SMC security proofs (Theorem 1 + privacy of `e_HSB` reveal)

### Paper

Claims in Sec. IV-D that neither party learns the cover value or
meaningful prediction errors from the messages exchanged, citing
Bogdanov et al. 2012 (ref [39]).

### What this implementation does

**Not cryptographically reproduced.**  The claim is a statement about
the information-theoretic view of each party; our code produces the
*same* numeric outputs the paper's SMC protocol would produce, but does
not implement the "simulator" argument that would formally establish
privacy.  `_verify_smc_contribution` provides a constructive self-check
that the numeric invariant underlying the proof holds, but a formal
security proof is out of scope.

---

## 10. JPEG / JPEG2000 / Gaussian-noise robustness benchmarks

### Paper

Figs. 14-18 and Tables II-IV report robustness against JPEG,
JPEG2000, Gaussian noise, median filters, sharpening, etc. using
standard test images (Lena, Baboon, Peppers).

### What this implementation does

Implemented in `rrwei_sm/attacks.py` with the following primitives:

* `additive_gaussian_noise(sigma)`
* `salt_and_pepper_noise(p)`
* `median_filter(ksize)`
* `mean_filter(ksize)`
* `sharpen_filter(amount)`
* `jpeg_compress(quality)` (via Pillow)
* `jpeg2000_compress(quality_mode, quality_layers)` (via Pillow OpenJPEG)

All primitives are covered by unit tests (`tests/test_attacks.py`) and
used by `tests/test_robustness_attacks.py` in an end-to-end watermark
recovery scenario. `evaluate.py` and the figure generators sweep
Gaussian sigma and JPEG quality over classic synthetic covers and save
the results as PNG plots in `figures/out/`.

### Residual deviations

* Test images are synthetic Lena/Baboon/Peppers-like stand-ins, so the
  absolute BER values differ from the paper's.
* Our JPEG2000 backend is Pillow's OpenJPEG wrapper; the rate control
  is approximate (uses `quality_layers` rather than bpp).

---

## 11. NPCR protocol (Table V)

### Paper

Table V reports NPCR values ~98.4-98.7% across 2x2 / 4x4 / 8x8 / 16x16
block-size sweeps on USC-SIPI, BOSSBase and UCID datasets, close to the
ideal 99.6094% for 8-bit images.

### What was unclear

NPCR as normally defined in image-encryption literature is:

    NPCR = mean(E_k(I1) != E_k(I2))

where `I1` and `I2` differ by one plaintext pixel and `k` is **the same
key**. Under that protocol, our scheme's NPCR is ~`1/n` because
additive secret sharing + a block permutation does not diffuse a
per-pixel plaintext change: exactly one share pixel moves.

For the paper to get ~99%, the two encryptions must use **different
random keys** (independent RNG streams for the share mask and/or
independent scramble keys). That protocol measures randomness of the
key stream rather than plaintext sensitivity, and any one-time-pad-like
encryption hits it trivially.

### What this implementation does

`rrwei_sm/metrics.py::npcr_report` implements the **standard same-key
protocol** (and hence returns tiny values on this scheme). The helper
docstring documents the alternative random-key protocol and shows how
to call it by varying `seed` across samples.

### Consequence

The absolute NPCR number in the paper is not reproducible under the
standard interpretation. The scheme's actual *security* rests on the
indistinguishability of each additive share from uniform random bytes,
which is validated by:

* `share / cover` correlation ~ 0 (see `evaluate.py` output),
* uniform share histograms (see `figures/out/encrypted_visuals.png`),
* H/V/D pixel correlation drops after scrambling
  (`metrics.correlation_report`).

---

## Summary of what IS faithful to the paper

* Image encryption: additive secret sharing + block-level scrambling,
  in both random-permutation and Hua et al. 2D-LSCM flavours.
* Plaintext / encrypted-domain separability: either extraction order
  works.
* PEE on HSB plane using the 2x2 block-level predictor with 3
  identical weights (Eq. 15) and Eqs. 20-22 for embedding/extraction.
* SMC correctness: per-party contributions sum exactly to the combined
  scaled prediction error (self-checked at embed time).
* Patchwork robust embedding on either the full-pixel plane or the HSB
  plane (Eqs. 23-27).
* zlib entropy coding of patchwork side information.
* Two-stage Modified RRWEI-SM with side-information embedded via PEE
  stage 2.
* Full separability and reversibility on intact marked images.
* Multi-layer PEE capacity up to ~1 bpp (4 layers).
* k-party additive sharing (`n_parties >= 2`).
* Robustness attack suite: Gaussian, salt & pepper, median, mean,
  sharpen, JPEG, JPEG2000.
* Metrics: PSNR, SSIM, BER, NPCR, UACI, H/V/D correlation, Eq. 28 PEE
  capacity.
* Benchmark harness confirming O(n) scaling of encrypt / decrypt /
  embed.
