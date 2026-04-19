"""End-to-end orchestrator for the modernized RRWEI-SM pipeline.

Ties every modern module into a single ``ModernScheme`` class that
the rest of the codebase (evaluate.py, benchmark.py, the Step 9 real-
image harness) calls as a single unit.

Pipeline
--------

Encrypt:
  1. :func:`crypto_scrambler.block_scramble` -- ChaCha20-keyed
     permutation of 2x2 blocks.
  2. :func:`threshold_sharing.share` -- replicated ``(k, n)`` shares.

Embed:
  3. :func:`stdm.embed` on the combined plaintext view of the
     scrambled cover.  Produces the robust bit plane.
  4. :func:`pvo.embed` on the STDM-marked combined view.  Produces the
     reversible payload (side information for STDM + user payload).
  5. Serialize the side info via :func:`coding.encode_symbols` (rANS).
  6. Owner applies the aggregated delta back onto one of its mask
     pieces via :func:`threshold_sharing.apply_owner_delta`.

Decrypt / extract (any ``t`` cooperating parties):
  7. :func:`threshold_sharing.combine` to recover the marked
     combined view.
  8. :func:`stdm.extract` for blind robust bits (works even after
     attacks).
  9. :func:`pvo.extract` for reversible payload + exact cover.
  10. :func:`crypto_scrambler.block_unscramble` to undo the
      permutation.

The class is deliberately a thin coordinator -- all the algorithms
live in their own modules and can be tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import crypto_scrambler, pvo, stdm, threshold_sharing

__all__ = [
    "ModernScheme",
    "EmbedResult",
]


@dataclass
class EmbedResult:
    """Everything produced by :meth:`ModernScheme.embed`.

    ``marked_cover`` is the plaintext view after both STDM and PVO
    have been applied (unscrambled, as the legitimate owner would
    see it).
    ``stdm_side`` travels alongside the marked image so a blind
    decoder can recover the robust bits under attack.
    ``pvo_sides`` list: one entry per PVO layer used.
    """

    marked_cover: np.ndarray
    stdm_side: stdm.STDMSideInfo
    pvo_sides: list[pvo.PVOSideInfo]
    n_robust_bits: int
    n_reversible_bits: int


class ModernScheme:
    """Coordinator for the modern RRWEI-SM pipeline."""

    def __init__(
        self,
        *,
        k: int = 2,
        n: int = 3,
        n_lsb: int = 3,
        block_size: int = 2,
        stdm_config: stdm.STDMConfig | None = None,
        max_pvo_layers: int = 4,
    ) -> None:
        if block_size != 2:
            raise ValueError("PVO assumes 2x2 blocks")
        self.k = k
        self.n = n
        self.n_lsb = n_lsb
        self.block_size = block_size
        self.stdm_config = stdm_config or stdm.STDMConfig(
            delta=60.0, n_coeffs=4, seed=0,
        )
        self.max_pvo_layers = max_pvo_layers

    # -------- encryption / decryption -------------------------------
    def encrypt(
        self,
        cover: np.ndarray,
        *,
        scramble_seed: int = 0,
        share_seed: int = 0,
    ) -> tuple[np.ndarray, threshold_sharing.ReplicatedShares]:
        """Scramble then replicate-share."""
        scrambled = crypto_scrambler.block_scramble(
            cover, block_size=self.block_size, seed=scramble_seed
        )
        shares = threshold_sharing.share(
            scrambled, k=self.k, n=self.n, n_lsb=self.n_lsb, seed=share_seed
        )
        return scrambled, shares

    def decrypt(
        self,
        shares: threshold_sharing.ReplicatedShares,
        participating_parties: list[int],
        *,
        scramble_seed: int,
    ) -> np.ndarray:
        combined = threshold_sharing.combine(shares, participating_parties)
        combined = np.clip(combined, 0, 255).astype(np.uint8)
        return crypto_scrambler.block_unscramble(
            combined, block_size=self.block_size, seed=scramble_seed
        )

    # -------- embedding / extraction --------------------------------
    def embed(
        self,
        cover: np.ndarray,
        robust_bits: np.ndarray,
        reversible_payload: np.ndarray,
        *,
        scramble_seed: int = 0,
    ) -> EmbedResult:
        """Full two-stage embed on a plaintext cover.

        This helper operates directly on the plaintext view for
        simplicity; the SMC equivalence (owner applies delta to own
        mask) is covered separately by
        :meth:`threshold_sharing.apply_owner_delta`.
        """
        # Stage 1: scramble then STDM on the scrambled view.
        scrambled = crypto_scrambler.block_scramble(
            cover, block_size=self.block_size, seed=scramble_seed
        )
        stdm_marked, stdm_side = stdm.embed(
            scrambled, robust_bits, self.stdm_config
        )

        # Stage 2: PVO layers over reversible payload.
        current = stdm_marked.copy()
        sides: list[pvo.PVOSideInfo] = []
        payload = reversible_payload.copy()
        payload_pos = 0
        for _ in range(self.max_pvo_layers):
            if payload_pos >= payload.size:
                break
            cap = pvo.capacity_estimate(current)
            room = cap["max_side"] + cap["min_side"]
            if room == 0:
                break
            chunk = payload[payload_pos : payload_pos + room]
            current, side, n_emb = pvo.embed(current, chunk)
            sides.append(side)
            payload_pos += n_emb

        marked_scrambled = current
        marked_cover = crypto_scrambler.block_unscramble(
            marked_scrambled, block_size=self.block_size, seed=scramble_seed
        )
        return EmbedResult(
            marked_cover=marked_cover,
            stdm_side=stdm_side,
            pvo_sides=sides,
            n_robust_bits=int(robust_bits.size),
            n_reversible_bits=payload_pos,
        )

    def extract_robust(
        self,
        marked_or_attacked_cover: np.ndarray,
        stdm_side: stdm.STDMSideInfo,
        *,
        scramble_seed: int,
    ) -> np.ndarray:
        """Blindly extract the robust STDM bits.  Works under attacks."""
        scrambled = crypto_scrambler.block_scramble(
            marked_or_attacked_cover, block_size=self.block_size, seed=scramble_seed
        )
        return stdm.extract(scrambled, stdm_side)

    def extract_reversible(
        self,
        marked_cover: np.ndarray,
        pvo_sides: list[pvo.PVOSideInfo],
        *,
        scramble_seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Invert the PVO layers (reversible payload + stdm-marked cover).

        Returns ``(stdm_marked_cover, payload_bits)``.  Requires
        ``marked_cover`` to be unattacked; there is no error
        correction on the reversible side.
        """
        scrambled = crypto_scrambler.block_scramble(
            marked_cover, block_size=self.block_size, seed=scramble_seed
        )
        current = scrambled
        all_bits: list[np.ndarray] = []
        for side in reversed(pvo_sides):
            current, bits = pvo.extract(current, side)
            all_bits.append(bits)
        payload = np.concatenate(list(reversed(all_bits))) if all_bits else np.zeros(0, np.uint8)
        stdm_marked_cover = crypto_scrambler.block_unscramble(
            current, block_size=self.block_size, seed=scramble_seed
        )
        return stdm_marked_cover, payload

    def recover_cover(
        self,
        marked_cover: np.ndarray,
        stdm_side: stdm.STDMSideInfo,
        pvo_sides: list[pvo.PVOSideInfo],
        *,
        scramble_seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full reverse: marked -> STDM-marked -> scrambled cover -> cover.

        Returns ``(recovered_cover, robust_bits, reversible_payload)``.
        Requires an unattacked marked cover.
        """
        stdm_marked, payload = self.extract_reversible(
            marked_cover, pvo_sides, scramble_seed=scramble_seed
        )
        # Re-extract robust bits from the STDM-marked view (identical
        # to a blind extraction on the marked cover).
        robust_bits = self.extract_robust(
            stdm_marked, stdm_side, scramble_seed=scramble_seed
        )
        scrambled = crypto_scrambler.block_scramble(
            stdm_marked, block_size=self.block_size, seed=scramble_seed
        )
        # Now we need to invert STDM.  STDM is NOT exactly invertible
        # (it is a lossy quantization).  To recover the exact original
        # cover we would need PEE side info for the STDM residual too
        # -- that's the same idea as the legacy "Modified" scheme.
        # For now, return the STDM-marked cover: the PVO layer is
        # exactly reversible on top of it, and the STDM residual is
        # <= delta in projection space.
        return stdm_marked, robust_bits, payload
