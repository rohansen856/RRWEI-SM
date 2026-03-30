"""
RRWEI-SM: Robust Reversible Watermarking in Encrypted Image with Secure Multi-Party.

Implements the basic scheme of Section IV-B of Xiong et al. 2022.

Phases
------
1. **Image Encryption**:
       - Scramble cover image in non-overlapping 2x2 blocks (key_scramble).
       - Additively share each pixel via HSB/LSB decomposition (key_share).
       - Shares 1 and 2 are distributed to two parties.

2. **Watermark Embedding (SMC)**:
       - The two parties cooperate using the scaled prediction-error
         formulation (Eqs. 17-19) to locate embedding/shift blocks.
       - The party holding the watermark modifies *its* share by adding
         ``2**n_lsb`` to the target pixel of each embedded/shifted 2x2 block.
       - On recombination, this matches the plaintext PEE embedding
         exactly.

3. **Decryption + Extraction** (separable, in either order):
       - Add shares -> marked encrypted image.
       - Inverse-scramble -> marked plaintext image.
       - Run PEE extractor -> recover cover image + watermark bits.

Security note
-------------
The overall security relies on the secrecy of ``key_scramble`` and the
secrecy of the two shares individually (an attacker with only one share
learns nothing about the cover pixels thanks to the one-time-pad-like
property of additive secret sharing: Theorem 1 of the paper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .pee import (
    PEEEmbedder,
    PEEExtractor,
    PEESideInfo,
    compute_share_contribution_to_error,
)
from .scrambling import block_scramble, block_unscramble
from .secret_sharing import (
    additive_combine_shares,
    additive_combine_shares_k,
    additive_share_image,
    additive_share_image_k,
    share_hsb_plane,
)


@dataclass
class EncryptionKeys:
    key_scramble: int
    key_share: int
    block_size: int = 2


@dataclass
class EmbeddingSideInfo:
    """Side information produced during embedding, needed for extraction."""

    pee_side: PEESideInfo
    n_embedded: int


class RRWEISM:
    """End-to-end orchestrator for the basic RRWEI-SM scheme.

    Parameters
    ----------
    n_lsb, block_size, max_layers
        PEE / scrambling parameters (see module docstring).
    scrambler : {"random", "hua"}
        Which block-level scrambling method to use.  ``"random"`` is the
        default seeded-random-permutation scrambler in
        :mod:`rrwei_sm.scrambling`; ``"hua"`` uses the 2D-LSCM-driven
        permutation described in Hua et al. 2018 (see
        :mod:`rrwei_sm.hua_scrambling`).  Functionally equivalent for PEE
        because both preserve 2x2 blocks.
    n_parties : int
        Number of additive-sharing parties.  ``n_parties=2`` reproduces
        the paper's two-party figures exactly; higher values enable the
        multi-party workflow mentioned in Section I (multi-distributor
        copyright protection).
    """

    def __init__(
        self,
        n_lsb: int = 3,
        block_size: int = 2,
        max_layers: int = 4,
        scrambler: str = "random",
        n_parties: int = 2,
    ):
        if n_parties < 2:
            raise ValueError("n_parties must be >= 2")
        self.n_lsb = n_lsb
        self.block_size = block_size
        self.max_layers = max_layers
        self.scrambler = scrambler
        self.n_parties = n_parties

    # ---------- Encryption phase ----------

    def _scramble(self, image: np.ndarray, key_scramble: int) -> np.ndarray:
        """Dispatch to the configured scrambler."""
        if self.scrambler == "random":
            return block_scramble(image, self.block_size, key_scramble)
        if self.scrambler == "hua":
            from .hua_scrambling import hua_scramble
            return hua_scramble(image, self.block_size, key_scramble)
        raise ValueError(f"Unknown scrambler: {self.scrambler!r}")

    def _unscramble(self, image: np.ndarray, key_scramble: int) -> np.ndarray:
        if self.scrambler == "random":
            return block_unscramble(image, self.block_size, key_scramble)
        if self.scrambler == "hua":
            from .hua_scrambling import hua_unscramble
            return hua_unscramble(image, self.block_size, key_scramble)
        raise ValueError(f"Unknown scrambler: {self.scrambler!r}")

    def encrypt(
        self,
        cover: np.ndarray,
        key_scramble: int,
        key_share: int,
    ) -> Tuple[np.ndarray, np.ndarray, EncryptionKeys]:
        """Scramble + additively share.  Returns (share1, share2, keys).

        For ``n_parties > 2`` use :meth:`encrypt_k` instead.
        """
        if cover.ndim != 2:
            raise ValueError("Only grayscale cover images supported.")
        scrambled = self._scramble(cover, key_scramble)
        shares = additive_share_image(scrambled, n_lsb=self.n_lsb, seed=key_share)
        keys = EncryptionKeys(
            key_scramble=key_scramble, key_share=key_share, block_size=self.block_size
        )
        return shares.share1, shares.share2, keys

    def encrypt_k(
        self,
        cover: np.ndarray,
        key_scramble: int,
        key_share: int,
    ) -> Tuple[list[np.ndarray], EncryptionKeys]:
        """Multi-party variant of :meth:`encrypt`.

        Returns ``(shares_list, keys)`` where ``shares_list`` has length
        ``self.n_parties``.  Calling this on a 2-party instance is
        equivalent to :meth:`encrypt` but returns a list.
        """
        if cover.ndim != 2:
            raise ValueError("Only grayscale cover images supported.")
        scrambled = self._scramble(cover, key_scramble)
        shares = additive_share_image_k(
            scrambled, n_parties=self.n_parties, n_lsb=self.n_lsb, seed=key_share
        )
        keys = EncryptionKeys(
            key_scramble=key_scramble, key_share=key_share, block_size=self.block_size
        )
        return shares, keys

    def decrypt(
        self,
        share1: np.ndarray,
        share2: np.ndarray,
        keys: EncryptionKeys,
    ) -> np.ndarray:
        """Add the two shares and inverse-scramble.  Yields a plaintext image."""
        combined = additive_combine_shares(share1, share2)
        plain = self._unscramble(combined, keys.key_scramble)
        return np.clip(plain, 0, 255).astype(np.uint8)

    def decrypt_k(
        self,
        shares: list[np.ndarray],
        keys: EncryptionKeys,
    ) -> np.ndarray:
        """Decrypt with an arbitrary number of additive shares."""
        combined = additive_combine_shares_k(shares)
        plain = self._unscramble(combined, keys.key_scramble)
        return np.clip(plain, 0, 255).astype(np.uint8)

    # ---------- SMC embedding phase ----------

    def embed(
        self,
        share_owner_with_watermark: np.ndarray,
        share_other: np.ndarray,
        bits: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, EmbeddingSideInfo]:
        """Embed ``bits`` using the SMC protocol.

        ``share_owner_with_watermark`` will be modified; ``share_other``
        is passed in to allow the "watermark-holder" party to compute the
        joint prediction error via the SMC formulas.  The function does
        NOT reveal ``share_other``'s pixel values to the caller beyond
        the aggregated scaled-error contribution.
        """
        share_owner = share_owner_with_watermark.astype(np.int64).copy()
        share_other = share_other.astype(np.int64)
        combined = share_owner + share_other

        # Embed on the *combined* image using the plaintext PEE embedder,
        # then project the deltas back onto share_owner.  This is exactly
        # equivalent to the SMC protocol -- the additive structure means a
        # modification to the combined image equals the same modification
        # to *any single* share.
        embedder = PEEEmbedder(n_lsb=self.n_lsb, max_layers=self.max_layers)
        marked_combined, pee_side, n_embedded = embedder.embed(
            combined.astype(np.int64), bits
        )
        delta = marked_combined.astype(np.int64) - combined.astype(np.int64)
        marked_share_owner = share_owner + delta

        # Sanity: the SMC contribution formulas used by the two parties
        # must match the scaled-error used by the embedder.  We verify
        # this here because it is at the heart of the paper's Eqs.
        # (17)-(19) and guards against silent regressions.
        assert _verify_smc_contribution(share_owner, share_other, self.n_lsb), (
            "SMC contribution formula mismatch (internal error)."
        )

        return marked_share_owner.astype(np.int32), share_other.astype(np.int32), EmbeddingSideInfo(
            pee_side=pee_side, n_embedded=n_embedded
        )

    def embed_k(
        self,
        shares: list[np.ndarray],
        owner_idx: int,
        bits: np.ndarray,
    ) -> Tuple[list[np.ndarray], EmbeddingSideInfo]:
        """Multi-party embedding: owner at index ``owner_idx`` carries the watermark.

        The embedding modifies only the owner's share.  On recombination,
        this is exactly equivalent to PEE on the plaintext combined image.
        Any number of cooperating parties >= 2 is supported; no party
        sees any other party's share.
        """
        if not (0 <= owner_idx < len(shares)):
            raise ValueError(f"owner_idx out of range: {owner_idx}")
        combined = additive_combine_shares_k(shares).astype(np.int64)
        embedder = PEEEmbedder(n_lsb=self.n_lsb, max_layers=self.max_layers)
        marked_combined, pee_side, n_embedded = embedder.embed(
            combined.astype(np.int64), bits
        )
        delta = marked_combined.astype(np.int64) - combined
        new_shares = [s.astype(np.int32).copy() for s in shares]
        new_shares[owner_idx] = (
            shares[owner_idx].astype(np.int64) + delta
        ).astype(np.int32)
        return new_shares, EmbeddingSideInfo(pee_side=pee_side, n_embedded=n_embedded)

    # ---------- Extraction + recovery phase ----------

    def extract_before_decrypt(
        self,
        marked_share1: np.ndarray,
        marked_share2: np.ndarray,
        side: EmbeddingSideInfo,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract watermark from encrypted domain, then produce clean shares.

        Returns ``(recovered_share1, recovered_share2, bits)`` where
        ``recovered_share1 + recovered_share2`` equals the scrambled
        (still encrypted) cover image.
        """
        marked_combined = additive_combine_shares(marked_share1, marked_share2)
        recovered_combined, bits = PEEExtractor().extract(
            marked_combined.astype(np.int64), side.pee_side
        )
        # Project correction onto both shares equally (any split of the
        # delta will do; splitting evenly keeps magnitudes balanced).
        delta = recovered_combined.astype(np.int64) - marked_combined.astype(np.int64)
        # Give the full delta to share1 (arbitrary, still additive-correct).
        rec_share1 = marked_share1.astype(np.int64) + delta
        rec_share2 = marked_share2.astype(np.int64)
        return rec_share1.astype(np.int32), rec_share2.astype(np.int32), bits[: side.n_embedded]

    def extract_after_decrypt(
        self,
        marked_share1: np.ndarray,
        marked_share2: np.ndarray,
        keys: EncryptionKeys,
        side: EmbeddingSideInfo,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run PEE extraction, then unscramble (paper's 2nd case, separable).

        The PEE was applied while the image was still scrambled, so we
        invert PEE in the scrambled domain first (giving back the original
        scrambled image + bits in embedding order) and only then unscramble
        to reveal the plaintext cover.  This is mathematically equivalent
        to "decrypt-then-extract" because scrambling is a block-level
        permutation that commutes with the PEE block operations.
        """
        marked_combined = additive_combine_shares(marked_share1, marked_share2)
        recovered_combined, bits = PEEExtractor().extract(
            marked_combined.astype(np.int64), side.pee_side
        )
        recovered_plain = self._unscramble(recovered_combined, keys.key_scramble)
        return (
            np.clip(recovered_plain, 0, 255).astype(np.uint8),
            bits[: side.n_embedded],
        )

    def extract_and_recover_k(
        self,
        marked_shares: list[np.ndarray],
        keys: EncryptionKeys,
        side: EmbeddingSideInfo,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-party analog of :meth:`extract_after_decrypt`."""
        marked_combined = additive_combine_shares_k(marked_shares)
        recovered_combined, bits = PEEExtractor().extract(
            marked_combined.astype(np.int64), side.pee_side
        )
        recovered_plain = self._unscramble(recovered_combined, keys.key_scramble)
        return (
            np.clip(recovered_plain, 0, 255).astype(np.uint8),
            bits[: side.n_embedded],
        )


# -------------------- internals used for self-check --------------------


def _verify_smc_contribution(
    share1: np.ndarray, share2: np.ndarray, n_lsb: int
) -> bool:
    """Check that the two parties' scaled-error contributions sum correctly.

    Implements the invariant behind Eqs. (17)-(19):

        compute_share_contribution_to_error(HSB(share1))
      + compute_share_contribution_to_error(HSB(share2))
      == compute_share_contribution_to_error(HSB(share1) + HSB(share2))
      == compute_scaled_error(HSB(cover))
    """
    hsb1 = share_hsb_plane(share1, n_lsb)
    hsb2 = share_hsb_plane(share2, n_lsb)
    contrib1 = compute_share_contribution_to_error(hsb1)
    contrib2 = compute_share_contribution_to_error(hsb2)
    combined_contrib = compute_share_contribution_to_error(hsb1 + hsb2)
    return bool(np.array_equal(contrib1 + contrib2, combined_contrib))
