"""
Modified RRWEI-SM: two-stage robust-reversible watermarking.

Stage 1 (robust): embed the *robust* watermark via patchwork (see
:mod:`rrwei_sm.patchwork`).  Patchwork records a large body of per-block
side information (the original ``differences`` and the ``skipped`` mask).

Stage 2 (reversible): embed that side information into the intermediate
image with the PEE reversible watermarking scheme (see :mod:`rrwei_sm.pee`).

The receiver only needs a small :class:`PatchworkSkeleton` (seed, m, T,
shape, n_robust) to drive both extractions.  The per-block differences
and the skipped mask are recovered *from the marked image itself*,
exactly as the paper describes.

Encryption / decryption phases are identical to :class:`RRWEISM`.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .patchwork import PatchworkEmbedder, PatchworkExtractor, PatchworkSideInfo
from .pee import PEEEmbedder, PEEExtractor, PEESideInfo
from .rrwei_sm import EncryptionKeys, RRWEISM
from .secret_sharing import additive_combine_shares, additive_combine_shares_k
from .utils import bits_to_bytes, bytes_to_bits


@dataclass
class PatchworkSkeleton:
    """Non-secret metadata needed for patchwork operations.

    ``skipped_mask`` is included here (rather than hidden inside the PEE
    payload) because the paper's "quick robust extraction" path needs to
    know which blocks were actually embedded even when PEE has been
    destroyed by an attack.  The mask carries no information about the
    robust bits themselves -- it is just the overflow map of the cover
    image -- so making it public is safe.
    """

    block_size: int
    m: int
    T: int
    seed_permutation: int | None
    shape: Tuple[int, int]
    n_robust_bits: int
    patchwork_plane: str = "full"       # "full" or "hsb"
    n_lsb: int = 3                      # relevant only when patchwork_plane == "hsb"
    skipped_mask: np.ndarray | None = None  # shape (n_blocks,) bool


@dataclass
class ModifiedSideInfo:
    """Minimal side information for the two-stage inversion."""

    skeleton: PatchworkSkeleton
    pee_side: PEESideInfo   # enough to invert stage 2
    compressed: bool = False            # if True, side info payload was zlib-compressed


class ModifiedRRWEISM:
    """End-to-end orchestrator for the two-stage scheme."""

    def __init__(
        self,
        n_lsb: int = 3,
        block_size: int = 2,
        patchwork_m: int = 4,
        patchwork_T: int = 2,
        patchwork_seed: int = 42,
        max_pee_layers: int = 4,
        patchwork_plane: str = "full",
        compress_side_info: bool = True,
        scrambler: str = "random",
        n_parties: int = 2,
    ):
        """Create the two-stage scheme.

        Parameters
        ----------
        patchwork_plane : {"full", "hsb"}
            Where to apply the patchwork perturbations.  ``"full"`` applies
            them directly on the combined pixel values (what the original
            implementation did).  ``"hsb"`` applies them on the HSB plane
            of the combined image, matching the paper's Section IV-C:
            *"the patchwork robust watermark is embedded into the HSB
            plane of the shares"*.  The HSB-plane variant tends to be
            more faithful to the paper's robustness claims on textured
            images because it does not interfere with PEE's own HSB-unit
            perturbations.
        compress_side_info : bool
            If True, compress the patchwork side-information payload
            with zlib before PEE-embedding it.  This matches the paper's
            advice in Section IV-C ("we can also compress the side
            information"), at the cost of slightly slower embedding.
        scrambler, n_parties
            Passed through to :class:`RRWEISM`.
        """
        if patchwork_plane not in ("full", "hsb"):
            raise ValueError(f"patchwork_plane must be 'full' or 'hsb'")
        self._base = RRWEISM(
            n_lsb=n_lsb,
            block_size=block_size,
            max_layers=max_pee_layers,
            scrambler=scrambler,
            n_parties=n_parties,
        )
        self.patchwork_m = patchwork_m
        self.patchwork_T = patchwork_T
        self.patchwork_seed = patchwork_seed
        self.n_lsb = n_lsb
        self.max_pee_layers = max_pee_layers
        self.patchwork_plane = patchwork_plane
        self.compress_side_info = compress_side_info
        self.n_parties = n_parties

    # ---------- delegate encryption / decryption ----------

    def encrypt(self, cover, key_scramble, key_share):
        return self._base.encrypt(cover, key_scramble, key_share)

    def encrypt_k(self, cover, key_scramble, key_share):
        return self._base.encrypt_k(cover, key_scramble, key_share)

    def decrypt(self, share1, share2, keys):
        return self._base.decrypt(share1, share2, keys)

    def decrypt_k(self, shares, keys):
        return self._base.decrypt_k(shares, keys)

    # ---------- two-stage embedding ----------

    # ---------- shared embed core (works for 2-party or k-party) ----------

    def _run_embed_core(
        self, combined: np.ndarray, robust_bits: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, ModifiedSideInfo]:
        """Run both stages on ``combined`` (already scrambled).  Returns
        ``(after_pee_u8, delta, modified_side_info)`` where ``delta`` is the
        per-pixel correction to apply to the owner's share.
        """
        combined_u8 = np.clip(combined, 0, 255).astype(np.uint8)
        robust_bits = np.asarray(robust_bits, dtype=np.uint8).flatten()

        # ---- Stage 1: patchwork ----
        # "hsb" mode: perturb by T HSB-units = T * 2**n_lsb pixel-units, so
        # LSBs are preserved and PEE's +/- 2**n_lsb bumps don't mangle the
        # sign of the patchwork difference during the quick-extract path.
        if self.patchwork_plane == "hsb":
            T_effective = self.patchwork_T * (1 << self.n_lsb)
        else:
            T_effective = self.patchwork_T
        pw = PatchworkEmbedder(
            m=self.patchwork_m, T=T_effective, seed=self.patchwork_seed
        )
        after_robust, pw_side = pw.embed(combined_u8, robust_bits.tolist())

        # ---- serialize + optionally compress side info ----
        side_bytes = _pack_patchwork_side_info(pw_side)
        if self.compress_side_info:
            compressed = zlib.compress(side_bytes, level=9)
            # Use the compressed payload only if it's actually smaller;
            # otherwise fall back to the raw payload (fingerprint flag).
            if len(compressed) < len(side_bytes):
                side_bytes = compressed
                compressed_flag = True
            else:
                compressed_flag = False
        else:
            compressed_flag = False

        side_bits = bytes_to_bits(side_bytes)
        # 33-bit header: 1 flag bit + 32 length bits (little-endian).
        flag_bit = np.array([1 if compressed_flag else 0], dtype=np.uint8)
        length_bits = np.unpackbits(
            np.array([len(side_bits)], dtype=np.uint32).view(np.uint8)
        )
        payload_bits = np.concatenate(
            [flag_bit, length_bits.astype(np.uint8), side_bits]
        )

        # ---- Stage 2: PEE-embed the payload ----
        pee_embedder = PEEEmbedder(n_lsb=self.n_lsb, max_layers=self.max_pee_layers)
        after_pee, pee_side, n_embedded = pee_embedder.embed(
            after_robust.astype(np.int64), payload_bits
        )
        if n_embedded < len(payload_bits):
            raise RuntimeError(
                f"PEE capacity too low: need {len(payload_bits)} bits, "
                f"embedded {n_embedded}.  Try compress_side_info=True, "
                f"a larger patchwork_m (fewer blocks), or a lower T."
            )

        delta = after_pee.astype(np.int64) - combined.astype(np.int64)

        skeleton = PatchworkSkeleton(
            block_size=pw_side.block_size,
            m=pw_side.m,
            T=pw_side.T,                        # the effective T used in embedding
            seed_permutation=pw_side.seed_permutation,
            shape=pw_side.shape,
            n_robust_bits=int(robust_bits.size),
            patchwork_plane=self.patchwork_plane,
            n_lsb=self.n_lsb,
            skipped_mask=pw_side.skipped.copy(),
        )
        mod_side = ModifiedSideInfo(
            skeleton=skeleton, pee_side=pee_side, compressed=compressed_flag
        )
        return after_pee, delta, mod_side

    def embed(
        self,
        share1: np.ndarray,
        share2: np.ndarray,
        robust_bits: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, ModifiedSideInfo]:
        """Two-stage embed (2-party).  Returns new shares + side info."""
        combined = additive_combine_shares(share1, share2).astype(np.int64)
        _, delta, side = self._run_embed_core(combined, robust_bits)
        marked_share1 = (share1.astype(np.int64) + delta).astype(np.int32)
        marked_share2 = share2.astype(np.int32)
        return marked_share1, marked_share2, side

    def embed_k(
        self,
        shares: list[np.ndarray],
        owner_idx: int,
        robust_bits: np.ndarray,
    ) -> Tuple[list[np.ndarray], ModifiedSideInfo]:
        """Two-stage embed (k-party)."""
        if not (0 <= owner_idx < len(shares)):
            raise ValueError(f"owner_idx out of range: {owner_idx}")
        combined = additive_combine_shares_k(shares).astype(np.int64)
        _, delta, side = self._run_embed_core(combined, robust_bits)
        new_shares = [s.astype(np.int32).copy() for s in shares]
        new_shares[owner_idx] = (
            shares[owner_idx].astype(np.int64) + delta
        ).astype(np.int32)
        return new_shares, side

    # ---------- extraction / recovery ----------

    # ---------- extraction helpers (shared) ----------

    def _quick_extract_from_combined(
        self, combined_scrambled: np.ndarray, side: ModifiedSideInfo
    ) -> np.ndarray:
        """Sign-of-difference extraction, ignoring PEE's perturbation.

        Uses the public ``skipped_mask`` from :class:`PatchworkSkeleton`
        so the reader knows which blocks actually carry bits.  The
        stored ``side.skeleton.T`` is already the effective pixel-space
        T (i.e. scaled by 2**n_lsb in HSB mode at embed time), so the
        extractor uses it directly on the combined pixel plane in either
        mode.
        """
        combined_u8 = np.clip(combined_scrambled, 0, 255).astype(np.uint8)
        n_blocks = combined_u8.size // side.skeleton.block_size
        if side.skeleton.skipped_mask is not None:
            skipped = side.skeleton.skipped_mask.astype(bool)
            if skipped.size != n_blocks:
                raise RuntimeError("skipped_mask size does not match image size")
        else:
            # Backward compatibility: treat all blocks as non-skipped.
            skipped = np.zeros(n_blocks, dtype=bool)
        sk_side = PatchworkSideInfo(
            block_size=side.skeleton.block_size,
            m=side.skeleton.m,
            T=side.skeleton.T,
            seed_permutation=side.skeleton.seed_permutation,
            differences=np.zeros(n_blocks, dtype=np.int64),
            skipped=skipped,
            n_embedded=int((~skipped).sum()),
            shape=side.skeleton.shape,
        )
        return PatchworkExtractor().extract_bits(combined_u8, sk_side)[
            : side.skeleton.n_robust_bits
        ]

    def _full_extract_from_combined(
        self, combined_scrambled: np.ndarray, side: ModifiedSideInfo
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Invert PEE + patchwork on the combined (still scrambled) image.

        Returns ``(recovered_scrambled_u8, robust_bits)``.
        """
        combined_u8 = np.clip(combined_scrambled, 0, 255).astype(np.uint8)
        pee_extractor = PEEExtractor()
        intermediate, all_bits = pee_extractor.extract(combined_u8, side.pee_side)

        if len(all_bits) < 33:
            raise RuntimeError("Corrupt PEE payload: header too short.")
        compressed_flag = bool(all_bits[0])
        length_bits = all_bits[1:33]
        length_bytes = np.packbits(length_bits).tobytes()
        payload_len = int(np.frombuffer(length_bytes, dtype=np.uint32)[0])
        side_bits = all_bits[33 : 33 + payload_len]
        side_bytes = bits_to_bytes(side_bits)
        if compressed_flag:
            side_bytes = zlib.decompress(side_bytes)
        pw_side = _unpack_patchwork_side_info(side_bytes, side.skeleton)

        pw = PatchworkExtractor()
        robust_bits = pw.extract_bits(intermediate.astype(np.int64), pw_side)[
            : side.skeleton.n_robust_bits
        ]
        recovered = pw.recover(intermediate.astype(np.int64), pw_side, robust_bits)
        return np.clip(recovered, 0, 255).astype(np.uint8), robust_bits

    # ---------- 2-party API ----------

    def extract_robust_after_decrypt(
        self,
        marked_share1: np.ndarray,
        marked_share2: np.ndarray,
        keys: EncryptionKeys,
        side: ModifiedSideInfo,
    ) -> np.ndarray:
        """Quick robust-bit extraction -- works even if the image is attacked."""
        combined_scrambled = additive_combine_shares(marked_share1, marked_share2)
        return self._quick_extract_from_combined(combined_scrambled, side)

    def extract_and_recover_after_decrypt(
        self,
        marked_share1: np.ndarray,
        marked_share2: np.ndarray,
        keys: EncryptionKeys,
        side: ModifiedSideInfo,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Full two-stage inversion: exact cover image + robust bits."""
        combined_scrambled = additive_combine_shares(marked_share1, marked_share2)
        recovered_scrambled, robust_bits = self._full_extract_from_combined(
            combined_scrambled, side
        )
        recovered_plain = self._base._unscramble(recovered_scrambled, keys.key_scramble)
        return np.clip(recovered_plain, 0, 255).astype(np.uint8), robust_bits

    # ---------- k-party API ----------

    def extract_robust_k(
        self,
        marked_shares: list[np.ndarray],
        keys: EncryptionKeys,
        side: ModifiedSideInfo,
    ) -> np.ndarray:
        combined_scrambled = additive_combine_shares_k(marked_shares)
        return self._quick_extract_from_combined(combined_scrambled, side)

    def extract_and_recover_k(
        self,
        marked_shares: list[np.ndarray],
        keys: EncryptionKeys,
        side: ModifiedSideInfo,
    ) -> Tuple[np.ndarray, np.ndarray]:
        combined_scrambled = additive_combine_shares_k(marked_shares)
        recovered_scrambled, robust_bits = self._full_extract_from_combined(
            combined_scrambled, side
        )
        recovered_plain = self._base._unscramble(recovered_scrambled, keys.key_scramble)
        return np.clip(recovered_plain, 0, 255).astype(np.uint8), robust_bits


# --------------------------- helpers ---------------------------

def _pack_patchwork_side_info(pw_side: PatchworkSideInfo) -> bytes:
    """Compact binary serialisation of patchwork side info.

    Layout (little-endian):
        u32  n_blocks
        u32  n_embedded
        byte[ceil(n_blocks/8)]       packed ``skipped`` mask (MSB-first)
        i16[n_embedded]              differences of *non-skipped* blocks in
                                     block-order

    The paper notes (Sec. IV-C) that side information *should* be
    compressed; we use a fixed 16-bit representation because
    ``|sum_A - sum_B| <= 255 * m``, which fits int16 for all practical
    ``m`` values (m <= 128).  A real deployment should use
    entropy coding; we deliberately keep the exact layout to simplify
    reviewer verification.
    """
    n_blocks = int(pw_side.differences.size)
    skipped_packed = bits_to_bytes(pw_side.skipped.astype(np.uint8))
    non_skipped_diffs = pw_side.differences[~pw_side.skipped].astype(np.int16)
    header = struct.pack("<II", n_blocks, int(non_skipped_diffs.size))
    return header + skipped_packed + non_skipped_diffs.tobytes()


def _unpack_patchwork_side_info(
    raw: bytes, skeleton: PatchworkSkeleton
) -> PatchworkSideInfo:
    n_blocks, n_emb = struct.unpack_from("<II", raw, 0)
    pos = 8
    bitmap_len = (n_blocks + 7) // 8
    skipped_bytes = raw[pos : pos + bitmap_len]
    pos += bitmap_len
    skipped = bytes_to_bits(skipped_bytes, n_bits=n_blocks).astype(bool)
    diffs_nonskipped = np.frombuffer(
        raw[pos : pos + n_emb * 2], dtype=np.int16
    ).astype(np.int64)
    # Re-inflate to a per-block differences array (0 for skipped blocks).
    differences = np.zeros(n_blocks, dtype=np.int64)
    differences[~skipped] = diffs_nonskipped
    return PatchworkSideInfo(
        block_size=skeleton.block_size,
        m=skeleton.m,
        T=skeleton.T,
        seed_permutation=skeleton.seed_permutation,
        differences=differences,
        skipped=skipped,
        n_embedded=int(n_emb),
        shape=skeleton.shape,
    )
