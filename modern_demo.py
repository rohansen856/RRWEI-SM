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


def _step2_stdm_robustness() -> None:
    """Step 2 gate: STDM embeds 256 bits, JPEG q=40 BER <= 0.1 on lena_like."""
    import io

    from PIL import Image

    from datasets import lena_like
    from rrwei_sm_modern.stdm import STDMConfig, embed, extract

    print("\n== Step 2: STDM robust watermark (128x128, 256 bits) ==")
    cover = lena_like(size=128)
    bits = np.random.default_rng(42).integers(0, 2, size=256, dtype=np.uint8)
    cfg = STDMConfig(delta=40.0, n_coeffs=4, seed=7)
    marked, side = embed(cover, bits, cfg)
    mse = np.mean((marked.astype(np.float64) - cover.astype(np.float64)) ** 2)
    psnr = 10.0 * np.log10(255.0**2 / max(mse, 1e-8))

    buf = io.BytesIO()
    Image.fromarray(marked).save(buf, format="JPEG", quality=40)
    buf.seek(0)
    attacked = np.array(Image.open(buf).convert("L"), dtype=np.uint8)

    out_clean = extract(marked, side)
    out_jpeg = extract(attacked, side)
    ber_clean = float((out_clean != bits).mean())
    ber_jpeg = float((out_jpeg != bits).mean())
    print(
        f"  lena_like 128x128: PSNR={psnr:.2f} dB, "
        f"BER(clean)={ber_clean:.3f}, BER(JPEG q=40)={ber_jpeg:.3f}"
    )
    assert ber_clean == 0.0
    assert ber_jpeg <= 0.10, ber_jpeg


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

    if "rrwei_sm_modern.stdm" in imported:
        _step2_stdm_robustness()

    if "rrwei_sm_modern.coding" in imported:
        _step3_rans_roundtrip()

    if "rrwei_sm_modern.threshold_sharing" in imported:
        _step4_threshold_sharing()

    if "rrwei_sm_modern.pvo" in imported:
        _step5_pvo_capacity()

    if "rrwei_sm_modern.metrics" in imported:
        _step6_perceptual_metrics()

    return 0


def _step6_perceptual_metrics() -> None:
    """Step 6 gate: LPIPS + DISTS proxy on marked images."""
    from datasets import load_classic_images
    from rrwei_sm_modern.metrics import dists_proxy, lpips_distance, psnr, ssim
    from rrwei_sm_modern.stdm import STDMConfig, embed as stdm_embed

    print("\n== Step 6: Perceptual metrics (LPIPS + DISTS-proxy) ==")
    covers = load_classic_images(size=128)
    rng = np.random.default_rng(17)
    header = f"{'cover':<14} {'PSNR':>7} {'SSIM':>6} {'LPIPS':>6} {'DISTS':>6}"
    print(f"  {header}")
    for name, cover in covers.items():
        bits = rng.integers(0, 2, size=128, dtype=np.uint8)
        marked, _ = stdm_embed(cover, bits, STDMConfig(delta=16, n_coeffs=4, seed=1))
        vals = [
            f"{psnr(cover, marked):7.2f}",
            f"{ssim(cover, marked):6.3f}",
            f"{lpips_distance(cover, marked):6.3f}",
            f"{dists_proxy(cover, marked):6.3f}",
        ]
        print(f"  {name:<14}" + " ".join(vals))


def _step5_pvo_capacity() -> None:
    """Step 5 gate: PVO reversible round-trip on all three covers."""
    from datasets import load_classic_images
    from rrwei_sm_modern.pvo import capacity_estimate, embed, extract

    covers = load_classic_images(size=128)
    print("\n== Step 5: PVO + pairwise PEE reversible predictor ==")
    for name, cover in covers.items():
        rng = np.random.default_rng(11)
        current = cover.copy()
        sides = []
        bit_trail = []
        n_layers = 0
        total = 0
        for _ in range(4):
            cap = capacity_estimate(current)
            n_bits = cap["max_side"] + cap["min_side"]
            if n_bits == 0:
                break
            bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
            current, side, n_emb = embed(current, bits)
            sides.append(side)
            bit_trail.append(bits[:n_emb])
            total += n_emb
            n_layers += 1
        marked = current.copy()
        mse = np.mean((marked.astype(np.float64) - cover.astype(np.float64)) ** 2)
        psnr = 99.0 if mse == 0 else 10.0 * np.log10(255.0**2 / mse)

        # Reverse traversal.
        for side, bits in zip(reversed(sides), reversed(bit_trail)):
            current, out_bits = extract(current, side)
            assert (out_bits == bits).all(), name
        assert (current == cover).all(), name

        print(
            f"  {name:12s} layers={n_layers} bpp={total/cover.size:.3f} "
            f"PSNR(marked)={psnr:.2f} dB; reversible round-trip OK"
        )


def _step4_threshold_sharing() -> None:
    """Step 4 gate: (2, 3) RSS round-trip + privacy invariant."""
    from datasets import load_classic_images
    from rrwei_sm_modern.threshold_sharing import (
        apply_owner_delta,
        combine,
        share,
    )

    covers = load_classic_images(size=64)
    print("\n== Step 4: (k, n) replicated secret sharing ==")
    for name, cover in covers.items():
        shares = share(cover, k=2, n=3, seed=0xdeadfeed)
        recovered = combine(shares, [0, 1]).astype(np.uint8)
        assert (recovered == cover).all()

        delta = np.zeros(cover.shape, dtype=np.int64)
        delta[5, 5] = 8
        marked = apply_owner_delta(shares, owner_idx=0, delta=delta)
        marked_img = combine(marked, [1, 2]).astype(np.int64)
        assert (marked_img - cover.astype(np.int64) == delta).all()
        print(
            f"  {name:12s} (k=2, n=3) round-trip OK; "
            f"owner-delta propagates via any 2 of 3 parties."
        )


def _step3_rans_roundtrip() -> None:
    """Step 3 gate: rANS round-trip + compression ratio demo."""
    import zlib

    from rrwei_sm_modern.coding import decode_symbols, encode_symbols, estimate_pmf

    rng = np.random.default_rng(0)
    n = 4096
    # Simulate bimodal side information (typical PVO / STDM residuals).
    symbols = rng.choice([-4, 4], size=n, p=[0.5, 0.5])
    symbols = symbols + rng.integers(-1, 2, size=n)
    pmf = estimate_pmf(symbols + 5, alphabet_size=11)
    coded = encode_symbols(symbols, pmf, alphabet_offset=-5)
    decoded = decode_symbols(coded.blob)
    assert (decoded == symbols).all()

    raw_bits = 8 * symbols.size * 2  # int16 baseline
    zlib_bits = 8 * len(zlib.compress(symbols.astype("<i2").tobytes(), level=9))
    rans_bits = 8 * len(coded.blob)
    print("\n== Step 3: rANS side-info coder ==")
    print(
        f"  bimodal payload n={n}: int16={raw_bits} bits, "
        f"zlib={zlib_bits} bits, rANS={rans_bits} bits "
        f"(rANS/zlib = {rans_bits / max(zlib_bits, 1):.2f})"
    )


if __name__ == "__main__":
    sys.exit(main())
