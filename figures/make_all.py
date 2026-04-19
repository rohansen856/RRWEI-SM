#!/usr/bin/env python3
"""Run every figure generator in this folder and print a summary."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figures._common import ensure_figure_dir


def main():
    here = Path(__file__).resolve().parent
    out_dir = ensure_figure_dir()
    scripts = sorted(
        p for p in here.glob("make_*.py") if p.name != "make_all.py"
    )
    for script in scripts:
        print(f"\n--- Running {script.name} ---")
        # run_path sets __name__=='__main__' -> script's main() fires.
        runpy.run_path(str(script), run_name="__main__")
    print(f"\nAll figures written to {out_dir}")


if __name__ == "__main__":
    main()
