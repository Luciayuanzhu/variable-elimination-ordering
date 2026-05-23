#!/usr/bin/env python3
"""Run the full VE ordering experiment suite."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ve_ordering.experiments import run_experiments


def main() -> None:
    df = run_experiments(ROOT / "results" / "results.csv")
    ok = int((df["status"] == "ok").sum())
    total = len(df)
    print(f"wrote results/results.csv with {ok}/{total} successful rows")


if __name__ == "__main__":
    main()

