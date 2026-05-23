#!/usr/bin/env python3
"""Run the expanded v2 VE ordering experiments."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ve_ordering.experiments_v2 import run_experiments_v2


def main() -> None:
    df = run_experiments_v2(ROOT / "results" / "results_v2.csv")
    ok = int((df["status"] == "ok").sum())
    total = len(df)
    print(f"wrote results/results_v2.csv with {ok}/{total} successful rows")


if __name__ == "__main__":
    main()
