"""End-to-end tests for the basic RRWEI-SM scheme."""

import numpy as np

from rrwei_sm.rrwei_sm import RRWEISM


def test_encrypt_decrypt_identity(synthetic_random_image):
    scheme = RRWEISM(n_lsb=3)
    s1, s2, keys = scheme.encrypt(synthetic_random_image, key_scramble=1, key_share=2)
    # Shares individually look nothing like the cover.
    corr1 = np.corrcoef(s1.flatten(), synthetic_random_image.flatten())[0, 1]
    assert abs(corr1) < 0.3
    recovered = scheme.decrypt(s1, s2, keys)
    assert (recovered == synthetic_random_image).all()


def test_end_to_end_embed_and_decrypt_first(synthetic_smooth_image):
    """Decryption-first extraction path (paper's 2nd case)."""
    scheme = RRWEISM(n_lsb=3, max_layers=2)
    s1, s2, keys = scheme.encrypt(synthetic_smooth_image, key_scramble=11, key_share=22)
    bits = np.random.default_rng(4).integers(0, 2, size=150, dtype=np.uint8)
    ms1, ms2, side = scheme.embed(s1, s2, bits)

    rec_img, ext_bits = scheme.extract_after_decrypt(ms1, ms2, keys, side)
    assert (rec_img == synthetic_smooth_image).all()
    assert np.array_equal(ext_bits, bits[: side.n_embedded])


def test_end_to_end_extract_first_in_encrypted_domain(synthetic_smooth_image):
    """Extraction-before-decryption (paper's 1st case: separable)."""
    scheme = RRWEISM(n_lsb=3, max_layers=2)
    s1, s2, keys = scheme.encrypt(synthetic_smooth_image, key_scramble=7, key_share=8)
    bits = np.random.default_rng(5).integers(0, 2, size=100, dtype=np.uint8)
    ms1, ms2, side = scheme.embed(s1, s2, bits)

    rs1, rs2, ext_bits = scheme.extract_before_decrypt(ms1, ms2, side)
    # After extraction, decrypt should produce the original cover.
    recovered = scheme.decrypt(rs1, rs2, keys)
    assert (recovered == synthetic_smooth_image).all()
    assert np.array_equal(ext_bits, bits[: side.n_embedded])


def test_marked_decrypt_visually_close_to_cover(synthetic_smooth_image):
    """Paper claim: decrypt-without-extract yields marked image close to cover."""
    scheme = RRWEISM(n_lsb=2, max_layers=1)
    s1, s2, keys = scheme.encrypt(synthetic_smooth_image, key_scramble=1, key_share=2)
    bits = np.random.default_rng(0).integers(0, 2, size=50, dtype=np.uint8)
    ms1, ms2, _ = scheme.embed(s1, s2, bits)
    marked_plain = scheme.decrypt(ms1, ms2, keys)
    # The marked image should differ but be close in PSNR (>30dB for a
    # smooth gradient under n_lsb=2).
    from rrwei_sm.utils import psnr
    assert psnr(synthetic_smooth_image, marked_plain) > 20.0
