#!/usr/bin/env python3
"""Create v2 figures from expanded experiment CSV results."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TOPOLOGY_ORDER = ["low_treewidth", "random", "grid", "scale_free", "moralized_bn"]
MAIN_METHODS = [
    "min_degree",
    "min_fill",
    "weighted_min_fill",
    "hybrid_degree_lam0.1",
    "fill_pressure_lam1",
    "adaptive_fill_pressure",
]
METHOD_LABELS = {
    "min_degree": "Min-degree",
    "min_fill": "Min-fill",
    "weighted_min_fill": "Weighted",
    "hybrid_degree_lam0.1": "V1 degree",
    "fill_pressure_lam1": "V2 fill-pressure",
    "adaptive_fill_pressure": "Adaptive",
}


def load_results() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results" / "results_v2.csv")
    return df[df["status"] == "ok"].copy()


def metric_bar(df: pd.DataFrame, metric: str, ylabel: str, output: Path) -> None:
    table = (
        df[df["method"].isin(MAIN_METHODS)]
        .groupby(["topology", "method"], observed=True)[metric]
        .median()
        .unstack("method")
        .reindex(TOPOLOGY_ORDER)
    )
    ax = table[MAIN_METHODS].rename(columns=METHOD_LABELS).plot(kind="bar", figsize=(9.4, 4.8))
    ax.set_xlabel("Topology")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " by topology (v2)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def improvement_plot(df: pd.DataFrame, output: Path) -> None:
    methods = ["weighted_min_fill", "hybrid_degree_lam0.1", "fill_pressure_lam1", "adaptive_fill_pressure"]
    wide = df[df["method"].isin(methods)].pivot_table(
        index=["topology", "graph_name"],
        columns="method",
        values="peak_factor_log10",
        aggfunc="first",
    ).dropna()
    summary = pd.DataFrame(
        {
            "V1 degree": wide["weighted_min_fill"] - wide["hybrid_degree_lam0.1"],
            "V2 fill-pressure": wide["weighted_min_fill"] - wide["fill_pressure_lam1"],
            "Adaptive": wide["weighted_min_fill"] - wide["adaptive_fill_pressure"],
        }
    ).groupby("topology").median().reindex(TOPOLOGY_ORDER)
    ax = summary.plot(kind="bar", figsize=(9, 4.5))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Topology")
    ax.set_ylabel("Median reduction in $\\log_{10}$ peak factor entries")
    ax.set_title("Improvement over weighted min-fill (v2)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def lambda_plot(df: pd.DataFrame, output: Path) -> None:
    part = df[df["base_method"] == "fill_pressure"].dropna(subset=["lambda"])
    summary = (
        part.groupby(["topology", "lambda"], observed=True)["peak_factor_log10"]
        .median()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for topology in TOPOLOGY_ORDER:
        p = summary[summary["topology"] == topology].sort_values("lambda")
        if not p.empty:
            ax.plot(p["lambda"], p["peak_factor_log10"], marker="o", label=topology)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("Median $\\log_{10}$ peak factor entries")
    ax.set_title("V2 fill-pressure lambda sensitivity")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def failure_plot(df: pd.DataFrame, output: Path) -> None:
    wide = df[df["method"].isin(["weighted_min_fill", "hybrid_degree_lam0.1", "fill_pressure_lam1"])].pivot_table(
        index=["topology", "graph_name"],
        columns="method",
        values="peak_factor_log10",
        aggfunc="first",
    ).dropna()
    wide["v1_delta"] = wide["hybrid_degree_lam0.1"] - wide["weighted_min_fill"]
    wide["v2_delta"] = wide["fill_pressure_lam1"] - wide["weighted_min_fill"]
    sample = wide.sort_values("v1_delta", ascending=False).head(10).reset_index()
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = range(len(sample))
    ax.bar([i - 0.18 for i in x], sample["v1_delta"], width=0.36, label="V1 degree")
    ax.bar([i + 0.18 for i in x], sample["v2_delta"], width=0.36, label="V2 fill-pressure")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(sample["graph_name"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("$\\Delta$ log peak factor vs weighted")
    ax.set_title("Worst V1 failures and V2 behavior")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def main() -> None:
    df = load_results()
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    metric_bar(df, "induced_width", "Induced width", out / "v2_induced_width_by_topology.pdf")
    metric_bar(df, "peak_factor_log10", "$\\log_{10}$ peak factor entries", out / "v2_peak_factor_by_topology.pdf")
    improvement_plot(df, out / "v2_improvement_over_wmf.pdf")
    lambda_plot(df, out / "v2_lambda_sensitivity.pdf")
    failure_plot(df, out / "v2_failure_cases.pdf")
    print("wrote v2 figures")


if __name__ == "__main__":
    main()
