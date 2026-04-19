"""End-to-end smoke test for the modernized RRWEI-SM pipeline.

Run ``python modern_demo.py`` after each modernization step; it imports
every modern module and exercises whichever pieces have already landed.
Before Step 1 it is only a sanity check that the package is importable.
"""

from __future__ import annotations

import importlib
import sys


def main() -> int:
    import rrwei_sm_modern  # noqa: F401

    attempted: list[tuple[str, bool, str]] = []

    optional_modules = [
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

    for name in optional_modules:
        try:
            importlib.import_module(name)
            attempted.append((name, True, ""))
        except ModuleNotFoundError as e:
            attempted.append((name, False, f"not implemented yet: {e}"))
        except Exception as e:  # pragma: no cover - surface real errors
            attempted.append((name, False, f"ERROR: {type(e).__name__}: {e}"))

    print("modern_demo import matrix:")
    for name, ok, note in attempted:
        flag = "OK " if ok else "-- "
        print(f"  {flag} {name:50s} {note}")

    ran = [n for n, ok, _ in attempted if ok]
    print(f"\n{len(ran)}/{len(attempted)} modules importable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
