"""Expanded v2 experiment runner with revised hybrid heuristics."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .experiments import stable_seed
from .graphs import GraphInstance, make_experiment_instances
from .inference import run_pairwise_variable_elimination
from .metrics import evaluate_order, graph_statistics
from .orderings import adaptive_lambda, get_order


@dataclass(frozen=True)
class MethodSpec:
    label: str
    method: str
    lambda_: float | None = None
    lookahead_depth: int | None = None
    oracle_pilot_only: bool = False
    max_nodes: int | None = None


V2_METHODS = [
    MethodSpec("min_degree", "min_degree"),
    MethodSpec("min_fill", "min_fill"),
    MethodSpec("weighted_min_fill", "weighted_min_fill"),
    MethodSpec("hybrid_degree_lam0.1", "hybrid", 0.1),
    MethodSpec("fill_pressure_lam0", "fill_pressure", 0.0),
    MethodSpec("fill_pressure_lam0.01", "fill_pressure", 0.01),
    MethodSpec("fill_pressure_lam0.05", "fill_pressure", 0.05),
    MethodSpec("fill_pressure_lam0.1", "fill_pressure", 0.1),
    MethodSpec("fill_pressure_lam0.5", "fill_pressure", 0.5),
    MethodSpec("fill_pressure_lam1", "fill_pressure", 1.0),
    MethodSpec("fill_pressure_lam2", "fill_pressure", 2.0),
    MethodSpec("next_step_lam0.1", "next_step", 0.1, max_nodes=40),
    MethodSpec("adaptive_fill_pressure", "adaptive_fill_pressure"),
    MethodSpec("oracle_depth2", "bounded_lookahead", None, 2, True, max_nodes=25),
    MethodSpec("oracle_depth3", "bounded_lookahead", None, 3, True, max_nodes=16),
]


def is_oracle_pilot(instance: GraphInstance) -> bool:
    """Run oracle on a broader but still bounded diagnostic subset."""
    n = instance.graph.number_of_nodes()
    if n <= 30:
        return True
    if instance.topology == "random" and n <= 40:
        return True
    if instance.topology == "scale_free" and n <= 50 and instance.metadata.get("seed") == 0:
        return True
    if instance.topology == "moralized_bn" and n <= 40:
        return True
    return False


def run_experiments_v2(
    output_path: str | Path = "results/results_v2.csv",
    ve_cutoff_entries: int = 250_000,
) -> pd.DataFrame:
    instances = make_experiment_instances()
    rows: list[dict[str, object]] = []

    for instance in instances:
        graph_stats = graph_statistics(instance.graph, instance.cardinalities)
        for method_spec in V2_METHODS:
            if (
                method_spec.max_nodes is not None
                and instance.graph.number_of_nodes() > method_spec.max_nodes
            ):
                continue
            if method_spec.oracle_pilot_only and not is_oracle_pilot(instance):
                continue

            row = {
                "version": "v2",
                "topology": instance.topology,
                "graph_name": instance.name,
                "method": method_spec.label,
                "base_method": method_spec.method,
                "lambda": math.nan if method_spec.lambda_ is None else method_spec.lambda_,
                "adaptive_lambda_initial": adaptive_lambda(instance.graph),
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
                row["order_prefix"] = ",".join(map(str, order[:8]))
                row["status"] = "ok"
            except Exception as exc:
                row["status"] = "error"
                row["error"] = repr(exc)
            rows.append(row)

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
