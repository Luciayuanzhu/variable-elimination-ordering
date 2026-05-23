"""Experiment runner helpers."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .graphs import GraphInstance, make_experiment_instances
from .inference import run_pairwise_variable_elimination
from .metrics import evaluate_order, graph_statistics
from .orderings import get_order


@dataclass(frozen=True)
class MethodSpec:
    label: str
    method: str
    lambda_: float | None = None
    lookahead_depth: int | None = None
    small_only: bool = False


METHODS = [
    MethodSpec("min_degree", "min_degree"),
    MethodSpec("min_fill", "min_fill"),
    MethodSpec("weighted_min_fill", "weighted_min_fill"),
    MethodSpec("hybrid_lam0", "hybrid", 0.0),
    MethodSpec("hybrid_lam0.01", "hybrid", 0.01),
    MethodSpec("hybrid_lam0.05", "hybrid", 0.05),
    MethodSpec("hybrid_lam0.1", "hybrid", 0.1),
    MethodSpec("hybrid_lam0.5", "hybrid", 0.5),
    MethodSpec("hybrid_lam1", "hybrid", 1.0),
    MethodSpec("hybrid_lam2", "hybrid", 2.0),
    MethodSpec("oracle_depth2", "bounded_lookahead", None, 2, True),
]


def run_experiments(
    output_path: str | Path = "results/results.csv",
    ve_cutoff_entries: int = 250_000,
    oracle_node_limit: int = 35,
) -> pd.DataFrame:
    instances = make_experiment_instances()
    rows: list[dict[str, object]] = []

    for instance in instances:
        graph_stats = graph_statistics(instance.graph, instance.cardinalities)
        for method_spec in METHODS:
            if method_spec.small_only and instance.graph.number_of_nodes() > oracle_node_limit:
                continue
            row = {
                "topology": instance.topology,
                "graph_name": instance.name,
                "method": method_spec.label,
                "base_method": method_spec.method,
                "lambda": math.nan if method_spec.lambda_ is None else method_spec.lambda_,
                **graph_stats,
                **instance.metadata,
            }

            try:
                start = time.perf_counter()
                order = get_order(
                    instance.graph,
                    instance.cardinalities,
                    method_spec.method,
                    lambda_=0.0 if method_spec.lambda_ is None else method_spec.lambda_,
                    lookahead_depth=method_spec.lookahead_depth,
                )
                ordering_time = time.perf_counter() - start
                metrics = evaluate_order(instance.graph, instance.cardinalities, order)
                ve_result = run_pairwise_variable_elimination(
                    instance.graph,
                    instance.cardinalities,
                    order,
                    max_factor_entries=ve_cutoff_entries,
                    seed=stable_seed(instance.name, method_spec.label),
                )
                row.update(metrics)
                row.update(ve_result)
                row["ordering_time"] = ordering_time
                row["total_runtime"] = (
                    ordering_time + ve_result["ve_runtime"]
                    if ve_result["ve_status"] == "ok"
                    else math.nan
                )
                row["status"] = "ok"
            except Exception as exc:  # Keep the batch running and surface failures in CSV.
                row["status"] = "error"
                row["error"] = repr(exc)
            rows.append(row)

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def stable_seed(*parts: str) -> int:
    value = 0
    for text in parts:
        for char in text:
            value = (value * 131 + ord(char)) % (2**32)
    return value

