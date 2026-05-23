#!/usr/bin/env python3
"""Generate the expanded v2 LaTeX results report."""

from __future__ import annotations

import math
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
MAIN_METHODS = [
    "min_degree",
    "min_fill",
    "weighted_min_fill",
    "hybrid_degree_lam0.1",
    "fill_pressure_lam1",
    "adaptive_fill_pressure",
]


def fmt(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{digits}f}"


def tex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
    )


def summary_table(df: pd.DataFrame, metric: str, digits: int = 2) -> str:
    table = (
        df[df["method"].isin(MAIN_METHODS)]
        .groupby(["topology", "method"], observed=True)[metric]
        .median()
        .unstack("method")
        .reindex(TOPOLOGY_ORDER)
    )
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "\\textbf{Topology} & \\textbf{Min-degree} & \\textbf{Min-fill} & \\textbf{Weighted} & \\textbf{V1 degree} & \\textbf{V2 fill} & \\textbf{Adaptive} \\\\",
        "\\midrule",
    ]
    for topology in TOPOLOGY_ORDER:
        row = table.loc[topology]
        lines.append(
            f"{TOPOLOGY_LABELS[topology]} & "
            f"{fmt(row.get('min_degree'), digits)} & "
            f"{fmt(row.get('min_fill'), digits)} & "
            f"{fmt(row.get('weighted_min_fill'), digits)} & "
            f"{fmt(row.get('hybrid_degree_lam0.1'), digits)} & "
            f"{fmt(row.get('fill_pressure_lam1'), digits)} & "
            f"{fmt(row.get('adaptive_fill_pressure'), digits)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def best_lambda_table(df: pd.DataFrame) -> str:
    part = df[df["base_method"] == "fill_pressure"].dropna(subset=["lambda"])
    summary = (
        part.groupby(["topology", "lambda"], observed=True)["peak_factor_log10"]
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
        lines.append(f"{TOPOLOGY_LABELS[topology]} & {fmt(row['lambda'], 2)} & {fmt(row['peak_factor_log10'], 2)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def bn_table(df: pd.DataFrame) -> str:
    bns = df[df["topology"] == "moralized_bn"].drop_duplicates("graph_name")
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "\\textbf{Benchmark} & \\textbf{$n$} & \\textbf{Edges} & \\textbf{Max card.} & \\textbf{Source} \\\\",
        "\\midrule",
    ]
    for _, row in bns.sort_values("graph_name").iterrows():
        bench = row.get("benchmark") if isinstance(row.get("benchmark"), str) else row["graph_name"]
        source = row.get("source", "")
        lines.append(f"{tex_escape(bench)} & {int(row.n)} & {int(row.m)} & {int(row.max_cardinality)} & {tex_escape(source)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def failure_table(df: pd.DataFrame) -> str:
    methods = ["weighted_min_fill", "hybrid_degree_lam0.1", "fill_pressure_lam1", "adaptive_fill_pressure"]
    wide = df[df["method"].isin(methods)].pivot_table(
        index=["topology", "graph_name"],
        columns="method",
        values="peak_factor_log10",
        aggfunc="first",
    ).dropna()
    wide["v1_delta"] = wide["hybrid_degree_lam0.1"] - wide["weighted_min_fill"]
    wide["v2_delta"] = wide["fill_pressure_lam1"] - wide["weighted_min_fill"]
    wide["adaptive_delta"] = wide["adaptive_fill_pressure"] - wide["weighted_min_fill"]
    sample = wide.sort_values("v1_delta", ascending=False).head(6).reset_index()
    lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "\\textbf{Topology} & \\textbf{Graph} & \\textbf{V1 $\\Delta$} & \\textbf{V2 $\\Delta$} & \\textbf{Adaptive $\\Delta$} \\\\",
        "\\midrule",
    ]
    for _, row in sample.iterrows():
        lines.append(
            f"{TOPOLOGY_LABELS[row['topology']]} & {tex_escape(row['graph_name'])} & "
            f"{fmt(row['v1_delta'], 2)} & {fmt(row['v2_delta'], 2)} & {fmt(row['adaptive_delta'], 2)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def build_report() -> str:
    df = pd.read_csv(ROOT / "results" / "results_v2.csv")
    ok = df[df["status"] == "ok"].copy()
    ve_ok = int((ok["ve_status"] == "ok").sum())
    ve_total = int(ok["ve_status"].notna().sum())
    oracle_rows = int(ok["method"].str.startswith("oracle").sum())
    next_rows = int((ok["method"] == "next_step_lam0.1").sum())
    canonical_bn = int((ok.drop_duplicates("graph_name")["source"] == "pgmpy_example_model").sum())

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

{{\Large\bfseries Hybrid Variable Elimination Ordering: v2 Results}}

{{TTIC 31180 --- Probabilistic Graphical Models Final Project}}

{{\color{{accent}}\rule{{\linewidth}}{{0.8pt}}}}

\section*{{What Changed in v2}}
Version 2 keeps the v1 baselines and adds three improvements: canonical \texttt{{pgmpy}} Bayesian-network benchmarks with native cardinalities, a redesigned fill-edge-local pressure term, and diagnostic cheap lookahead / bounded-lookahead runs. The original v1 hybrid penalized all high-degree neighbors; the v2 fill-pressure score penalizes only pressure created by actual fill-in edges, which is intended to reduce brittleness on hub-dominated graphs. The adaptive method chooses $\lambda$ from graph statistics rather than from topology labels.

\section*{{Setup}}
The v2 run used {ok['graph_name'].nunique()} graph instances and {len(ok)} successful method-instance runs. It loaded {canonical_bn} canonical \texttt{{pgmpy}} Bayesian networks. Explicit pairwise VE completed for {ve_ok} of {ve_total} rows under the factor-size cutoff. The cheap one-step method was run on {next_rows} medium/small rows; bounded-lookahead oracle diagnostics produced {oracle_rows} rows.

\begin{{center}}
{{\small
{bn_table(ok)}
}}
\end{{center}}

\section*{{Structural Results}}
Lower is better. Tables report topology-level medians.

\begin{{center}}
{{\small
{summary_table(ok, "induced_width", 1)}
}}
\end{{center}}

\begin{{center}}
{{\small
{summary_table(ok, "peak_factor_log10", 2)}
}}
\end{{center}}

\section*{{Lambda Sensitivity}}
The fill-edge-local pressure term shifted the useful $\lambda$ regime. In this run, larger $\lambda$ values helped random and grid graphs more than the original v1 degree-pressure score, while low-treewidth and canonical BN benchmarks still preferred little or no extra pressure.

\begin{{center}}
{{\small
{best_lambda_table(ok)}
}}
\end{{center}}

\section*{{Failure Cases}}
The table reports $\Delta=\log_{{10}}(\text{{method peak}})-\log_{{10}}(\text{{weighted min-fill peak}})$ for the worst v1-degree failures. Negative values are improvements over weighted min-fill; positive values are worse.

\begin{{center}}
{{\small
{failure_table(ok)}
}}
\end{{center}}

\section*{{Figures}}
\begin{{center}}
\includegraphics[width=0.92\linewidth]{{../figures/v2_induced_width_by_topology.pdf}}
\includegraphics[width=0.92\linewidth]{{../figures/v2_peak_factor_by_topology.pdf}}
\includegraphics[width=0.92\linewidth]{{../figures/v2_improvement_over_wmf.pdf}}
\includegraphics[width=0.92\linewidth]{{../figures/v2_lambda_sensitivity.pdf}}
\includegraphics[width=0.92\linewidth]{{../figures/v2_failure_cases.pdf}}
\end{{center}}

\section*{{Findings and Implications}}
\begin{{itemize}}
\item The topology dependence remains the dominant story. No hybrid method uniformly dominates weighted min-fill.
\item The redesigned fill-pressure term reduces one weakness of v1: it no longer broadly penalizes high-degree neighborhoods when the actual new fill-in is small.
\item A fixed v2 $\lambda=1$ gives stronger median peak-factor results on random and grid graphs than the conservative $\lambda=0.1$ choice from v1, but it is still not a universal setting.
\item Canonical BN benchmarks are less favorable to hybrid pressure: weighted min-fill is already strong, and extra pressure often adds little.
\item Adaptive $\lambda$ is useful as a concept, but the current hand-written rule is not yet reliable enough to replace a sweep. It avoids some v1 hub failures but can still misclassify random or BN-like structure.
\end{{itemize}}

\section*{{Limitations}}
The v2 run improves benchmark realism, but it still uses structural pairwise VE instrumentation rather than full CPD-aware BN inference. The next-step heuristic is diagnostic-only on medium/small graphs because full-size one-step simulation was too expensive. The adaptive rule is intentionally simple and should be treated as a prototype rather than a tuned model-selection procedure.

\section*{{Next Steps}}
\begin{{itemize}}
\item Replace the hand-written adaptive rule with a small validation-based selector that chooses $\lambda$ from pilot instances by graph statistics.
\item Add CPD-aware exact inference for canonical BNs, or use pgmpy/pyAgrum inference as a runtime validation layer.
\item Expand oracle diagnostics on selected small graphs to compare cheap proxies against explicit lookahead more systematically.
\item Analyze individual elimination traces for the random and insurance failure cases where adaptive fill-pressure is worse than weighted min-fill.
\item Consider a guarded hybrid rule: use weighted min-fill by default, activate pressure only when pilot graph statistics predict a benefit.
\end{{itemize}}

\end{{document}}
"""


def main() -> None:
    output = ROOT / "report" / "results_report_v2.tex"
    output.write_text(build_report(), encoding="utf-8")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
