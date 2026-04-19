"""BER-vs-attack bar chart for the modern STDM watermark.

Runs the STDM embedder on the three synthetic covers, passes the
marked image through every attack in
``rrwei_sm.attacks.AVAILABLE_ATTACKS``, and plots the per-cover
BER as a grouped bar chart.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from figures._common import ensure_figure_dir, mpl

from datasets import load_classic_images
from rrwei_sm.attacks import AVAILABLE_ATTACKS
from rrwei_sm.stdm import STDMConfig, embed, extract


def main() -> None:
    plt = mpl()
    covers = load_classic_images(size=128)
    cfg = STDMConfig(delta=60.0, n_coeffs=4, seed=0)

    attack_names = list(AVAILABLE_ATTACKS.keys())
    bers: dict[str, list[float]] = {name: [] for name in covers}

    for cover_name, cover in covers.items():
        rng = np.random.default_rng(hash(cover_name) & 0xFFFFFFFF)
        bits = rng.integers(0, 2, size=128, dtype=np.uint8)
        marked, side = embed(cover, bits, cfg)
        for attack_name in attack_names:
            attacked = AVAILABLE_ATTACKS[attack_name](marked)
            out = extract(attacked, side)
            bers[cover_name].append(float((out != bits).mean()))

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(attack_names))
    width = 0.26
    for i, name in enumerate(covers):
        ax.bar(x + (i - 1) * width, bers[name], width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(attack_names, rotation=25, ha="right")
    ax.set_ylabel("Bit error rate")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, label="random guess")
    ax.set_title(
        "STDM robustness under classical + modern attacks "
        "(128x128 synthetic covers, 128 bits, delta=60, n_coeffs=4)"
    )
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()

    out_path = ensure_figure_dir() / "robustness_sweep.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
