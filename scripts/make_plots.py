#!/usr/bin/env python3
"""Create figures from experiment CSV results."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ve_ordering.plots import save_all_figures


def main() -> None:
    save_all_figures(ROOT / "results" / "results.csv", ROOT / "figures")
    print("wrote figures/*.pdf")


if __name__ == "__main__":
    main()

