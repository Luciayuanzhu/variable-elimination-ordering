#!/usr/bin/env python3
"""Generate a concise LaTeX results report from experiment outputs."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


TOPOLOGY_ORDER = ["low_treewidth", "random", "grid", "scale_free", "moralized_bn"]
TOPOLOGY_LABELS = {
    "low_treewidth": "Low-treewidth",
    "random": "Random",
    "grid": "Grid",
    "scale_free": "Scale-free",
    "moralized_bn": "Moralized BN",
}
MAIN_METHODS = ["min_degree", "min_fill", "weighted_min_fill", "hybrid_lam0.1"]
METHOD_LABELS = {
    "min_degree": "Min-degree",
    "min_fill": "Min-fill",
    "weighted_min_fill": "Weighted min-fill",
    "hybrid_lam0.1": "Hybrid $\\lambda=0.1$",
}


def fmt(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{digits}f}"


def make_summary_table(df: pd.DataFrame, metric: str, digits: int = 2) -> str:
    subset = df[df["method"].isin(MAIN_METHODS)]
    table = (
        subset.groupby(["topology", "method"], observed=True)[metric]
        .median()
        .unstack("method")
        .reindex(TOPOLOGY_ORDER)
    )
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "\\textbf{Topology} & \\textbf{Min-degree} & \\textbf{Min-fill} & \\textbf{Weighted} & \\textbf{Hybrid} \\\\",
        "\\midrule",
    ]
    for topology in TOPOLOGY_ORDER:
        row = table.loc[topology]
        lines.append(
            f"{TOPOLOGY_LABELS[topology]} & "
            f"{fmt(row.get('min_degree'), digits)} & "
            f"{fmt(row.get('min_fill'), digits)} & "
            f"{fmt(row.get('weighted_min_fill'), digits)} & "
            f"{fmt(row.get('hybrid_lam0.1'), digits)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def make_instance_table(df: pd.DataFrame) -> str:
    stats = (
        df.drop_duplicates("graph_name")
        .groupby("topology", observed=True)
        .agg(
            instances=("graph_name", "count"),
            median_n=("n", "median"),
            max_n=("n", "max"),
            median_density=("density", "median"),
            median_max_degree=("max_degree", "median"),
        )
        .reindex(TOPOLOGY_ORDER)
    )
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "\\textbf{Topology} & \\textbf{Instances} & \\textbf{Median $n$} & \\textbf{Max $n$} & \\textbf{Median density} & \\textbf{Median max degree} \\\\",
        "\\midrule",
    ]
    for topology in TOPOLOGY_ORDER:
        row = stats.loc[topology]
        lines.append(
            f"{TOPOLOGY_LABELS[topology]} & {int(row.instances)} & "
            f"{fmt(row.median_n, 0)} & {fmt(row.max_n, 0)} & "
            f"{fmt(row.median_density, 3)} & {fmt(row.median_max_degree, 1)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def best_lambda_table(df: pd.DataFrame) -> str:
    hybrid = df[df["base_method"] == "hybrid"].dropna(subset=["lambda"])
    summary = (
        hybrid.groupby(["topology", "lambda"], observed=True)["peak_factor_log10"]
        .median()
        .reset_index()
    )
    idx = summary.groupby("topology")["peak_factor_log10"].idxmin()
    best = summary.loc[idx].set_index("topology").reindex(TOPOLOGY_ORDER)
    lines = [
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "\\textbf{Topology} & \\textbf{Best $\\lambda$} & \\textbf{Median $\\log_{10}$ peak size} \\\\",
        "\\midrule",
    ]
    for topology in TOPOLOGY_ORDER:
        row = best.loc[topology]
        lines.append(
            f"{TOPOLOGY_LABELS[topology]} & {fmt(row['lambda'], 2)} & {fmt(row['peak_factor_log10'], 2)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def build_report() -> str:
    df = pd.read_csv(ROOT / "results" / "results.csv")
    ok = df[df["status"] == "ok"].copy()
    ve_ok = int((ok["ve_status"] == "ok").sum())
    ve_total = int(ok["ve_status"].notna().sum())
    oracle_rows = ok[ok["method"] == "oracle_depth2"]

    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.72in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage{{newtxtext,newtxmath}}
\usepackage{{microtype}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{enumitem}}
\usepackage[hidelinks]{{hyperref}}
\definecolor{{accent}}{{HTML}}{{0F766E}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.28em}}
\setlist[itemize]{{leftmargin=1.2em,itemsep=0.12em,topsep=0.12em}}
\begin{{document}}
{{\color{{accent}}\rule{{\linewidth}}{{1.2pt}}}}

{{\Large\bfseries Hybrid Variable Elimination Ordering: Experimental Results}}

{{TTIC 31180 --- Probabilistic Graphical Models Final Project}}

{{\color{{accent}}\rule{{\linewidth}}{{0.8pt}}}}

\section*{{Setup}}
We evaluated min-degree, min-fill, weighted min-fill, and the proposed hybrid heuristic across five topology families. The experiment used {ok['graph_name'].nunique()} graph instances and {len(ok)} successful method-instance runs. Explicit pairwise variable elimination was attempted only when the projected intermediate factor stayed below the configured cutoff; {ve_ok} of {ve_total} successful structural runs completed explicit VE under this cutoff. The moralized Bayesian-network group includes a manually encoded Asia network and built-in BN-like layered DAGs because the local environment did not include \texttt{{pgmpy}} benchmark loaders.

\begin{{center}}
{make_instance_table(ok)}
\end{{center}}

\section*{{Structural Results}}
The primary metrics are structural: induced width and peak factor size. Lower is better for both. The following tables report medians within each topology.

\begin{{center}}
{{\small
{make_summary_table(ok, "induced_width", 1)}
}}
\end{{center}}

\begin{{center}}
{{\small
{make_summary_table(ok, "peak_factor_log10", 2)}
}}
\end{{center}}

\section*{{Lambda Sensitivity}}
The hybrid method is not uniformly monotone in $\lambda$. A fixed $\lambda=0.1$ was used as the main comparison point because it keeps the downstream-pressure term active without letting it dominate weighted fill cost.

\begin{{center}}
{{\small
{best_lambda_table(ok)}
}}
\end{{center}}

\section*{{Figures}}
\begin{{center}}
\includegraphics[width=0.92\linewidth]{{../figures/induced_width_by_topology.pdf}}
\includegraphics[width=0.92\linewidth]{{../figures/peak_factor_by_topology.pdf}}
\includegraphics[width=0.92\linewidth]{{../figures/hybrid_improvement_over_wmf.pdf}}
\includegraphics[width=0.92\linewidth]{{../figures/lambda_sensitivity.pdf}}
\includegraphics[width=0.92\linewidth]{{../figures/ordering_time_vs_n.pdf}}
\end{{center}}

\section*{{Interpretation}}
\begin{{itemize}}
\item Low-treewidth chains and trees behaved as sanity checks: structural metrics were small and the strongest heuristics were usually close.
\item Random graphs became difficult as density increased; peak factor estimates were more informative than raw runtime because many dense instances hit the explicit-VE cutoff.
\item Grid graphs showed the expected sensitivity to ordering, with induced width and peak factor estimates separating the methods more clearly as the grid grew.
\item Scale-free graphs highlighted the hub risk: methods that create dense neighborhoods around high-degree variables produced larger peak factors.
\item The hybrid heuristic helped on some topology groups but did not dominate weighted min-fill uniformly, which is a useful empirical finding. The downstream-pressure proxy can capture some lookahead signal, but its benefit depends on graph structure and the chosen $\lambda$.
\end{{itemize}}

\section*{{Bounded Lookahead Check}}
The bounded-lookahead oracle was run only on small graphs; it produced {len(oracle_rows)} successful rows. This keeps the comparison computationally feasible while still giving a reference point for how much a more explicit lookahead objective can improve structural metrics.

\section*{{Limitations}}
This run is self-contained and reproducible, but the BN benchmark portion should be strengthened in a future pass by loading canonical BIF/UAI networks directly through \texttt{{pgmpy}} or \texttt{{pyAgrum}}. The current report also treats exact VE runtime as secondary evidence because structural complexity, especially peak factor size, is the more stable measure across high-treewidth instances.

\end{{document}}
"""


def main() -> None:
    output = ROOT / "report" / "results_report.tex"
    output.write_text(build_report(), encoding="utf-8")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
