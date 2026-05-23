"""Plot generation for experiment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MAIN_METHODS = ["min_degree", "min_fill", "weighted_min_fill", "hybrid_lam0.1"]
METHOD_LABELS = {
    "min_degree": "Min-degree",
    "min_fill": "Min-fill",
    "weighted_min_fill": "Weighted min-fill",
    "hybrid_lam0.1": "Hybrid $\\lambda=0.1$",
}
TOPOLOGY_ORDER = ["low_treewidth", "random", "grid", "scale_free", "moralized_bn"]


def load_results(path: str | Path = "results/results.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["status"] == "ok"].copy()


def save_all_figures(
    results_path: str | Path = "results/results.csv",
    figure_dir: str | Path = "figures",
) -> None:
    df = load_results(results_path)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_by_topology(
        df,
        metric="induced_width",
        ylabel="Induced width",
        output=figure_dir / "induced_width_by_topology.pdf",
    )
    plot_metric_by_topology(
        df,
        metric="peak_factor_log10",
        ylabel="$\\log_{10}$ peak factor entries",
        output=figure_dir / "peak_factor_by_topology.pdf",
    )
    plot_hybrid_improvement(df, figure_dir / "hybrid_improvement_over_wmf.pdf")
    plot_lambda_sensitivity(df, figure_dir / "lambda_sensitivity.pdf")
    plot_ordering_time(df, figure_dir / "ordering_time_vs_n.pdf")


def plot_metric_by_topology(
    df: pd.DataFrame, metric: str, ylabel: str, output: Path
) -> None:
    subset = df[df["method"].isin(MAIN_METHODS)].copy()
    grouped = (
        subset.groupby(["topology", "method"], observed=True)[metric]
        .median()
        .unstack("method")
        .reindex(TOPOLOGY_ORDER)
    )
    grouped = grouped[MAIN_METHODS]
    ax = grouped.rename(columns=METHOD_LABELS).plot(kind="bar", figsize=(9, 4.8))
    ax.set_xlabel("Topology")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " by topology")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_hybrid_improvement(df: pd.DataFrame, output: Path) -> None:
    wide = df[df["method"].isin(["weighted_min_fill", "hybrid_lam0.1"])].pivot_table(
        index=["topology", "graph_name"],
        columns="method",
        values="peak_factor_log10",
        aggfunc="first",
    )
    wide = wide.dropna()
    wide["improvement"] = wide["weighted_min_fill"] - wide["hybrid_lam0.1"]
    summary = wide.groupby("topology")["improvement"].median().reindex(TOPOLOGY_ORDER)
    ax = summary.plot(kind="bar", color="#0F766E", figsize=(8, 4.2))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Topology")
    ax.set_ylabel("Median reduction in $\\log_{10}$ peak factor entries")
    ax.set_title("Hybrid improvement over weighted min-fill")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_lambda_sensitivity(df: pd.DataFrame, output: Path) -> None:
    hybrid = df[df["base_method"] == "hybrid"].dropna(subset=["lambda"]).copy()
    summary = (
        hybrid.groupby(["topology", "lambda"], observed=True)["peak_factor_log10"]
        .median()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for topology in TOPOLOGY_ORDER:
        part = summary[summary["topology"] == topology].sort_values("lambda")
        if part.empty:
            continue
        ax.plot(part["lambda"], part["peak_factor_log10"], marker="o", label=topology)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("Median $\\log_{10}$ peak factor entries")
    ax.set_title("Hybrid lambda sensitivity")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_ordering_time(df: pd.DataFrame, output: Path) -> None:
    subset = df[df["method"].isin(MAIN_METHODS)].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for method in MAIN_METHODS:
        part = subset[subset["method"] == method]
        summary = part.groupby("n", observed=True)["ordering_time"].median().reset_index()
        ax.plot(summary["n"], summary["ordering_time"], marker="o", label=METHOD_LABELS[method])
    ax.set_xlabel("Number of variables")
    ax.set_ylabel("Median ordering time (s)")
    ax.set_title("Ordering overhead")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()

