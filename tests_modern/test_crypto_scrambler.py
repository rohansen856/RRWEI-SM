"""Tests for the ChaCha20-based block scrambler."""

from __future__ import annotations

import numpy as np
import pytest

from rrwei_sm_modern.crypto_scrambler import (
    CryptoScramblerKey,
    block_scramble,
    block_unscramble,
    derive_key,
    generate_permutation,
)


def test_round_trip_preserves_image(textured_cover):
    scrambled = block_scramble(textured_cover, block_size=2, seed=42)
    recovered = block_unscramble(scrambled, block_size=2, seed=42)
    assert (recovered == textured_cover).all()


def test_two_keys_produce_different_permutations():
    p0 = generate_permutation(1024, 0)
    p1 = generate_permutation(1024, 1)
    assert not np.array_equal(p0, p1)


def test_permutation_is_a_bijection():
    perm = generate_permutation(4096, seed=7)
    assert sorted(perm.tolist()) == list(range(4096))


def test_bit_flipped_key_changes_permutation():
    k0 = derive_key(0)
    k1 = derive_key(0, info=b"unrelated/info")
    assert k0 != k1
    p0 = generate_permutation(512, k0)
    p1 = generate_permutation(512, k1)
    assert not np.array_equal(p0, p1)


def test_scrambled_blocks_preserve_2x2_structure(textured_cover):
    scrambled = block_scramble(textured_cover, block_size=2, seed=13)
    h, w = textured_cover.shape
    assert scrambled.shape == (h, w)
    input_blocks = textured_cover.reshape(h // 2, 2, w // 2, 2).swapaxes(1, 2)
    output_blocks = scrambled.reshape(h // 2, 2, w // 2, 2).swapaxes(1, 2)
    # Every output 2x2 block must appear somewhere in the input.
    in_set = {bytes(b) for b in input_blocks.reshape(-1, 4)}
    out_set = {bytes(b) for b in output_blocks.reshape(-1, 4)}
    assert in_set == out_set


def test_key_object_validates_lengths():
    with pytest.raises(ValueError):
        CryptoScramblerKey(key=b"x" * 16, nonce=b"y" * 16)
    with pytest.raises(ValueError):
        CryptoScramblerKey(key=b"x" * 32, nonce=b"y" * 8)
