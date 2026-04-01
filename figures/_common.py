"""Shared helpers for figure scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running scripts directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def ensure_figure_dir() -> Path:
    out = ROOT / "figures" / "out"
    out.mkdir(parents=True, exist_ok=True)
    return out


# Common matplotlib style: force a non-interactive backend by default so
# scripts work on headless CI without any configuration.
def mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )
    return plt
