"""Structural metrics for elimination orders."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce
from operator import mul

import networkx as nx

from .orderings import fill_edges


@dataclass(frozen=True)
class OrderMetrics:
    induced_width: int
    max_clique_size: int
    total_fill_edges: int
    peak_factor_entries: int
    peak_factor_log10: float
    final_edge_count: int


def _factor_entries(nodes: list[object], cardinalities: dict[object, int]) -> int:
    if not nodes:
        return 1
    return reduce(mul, (int(cardinalities[node]) for node in nodes), 1)


def evaluate_order(
    graph: nx.Graph, cardinalities: dict[object, int], order: list[object]
) -> dict[str, float | int]:
    """Simulate elimination and compute induced graph/factor-size metrics."""
    if set(order) != set(graph.nodes):
        missing = set(graph.nodes) - set(order)
        extra = set(order) - set(graph.nodes)
        raise ValueError(f"order does not match graph nodes; missing={missing}, extra={extra}")

    work = graph.copy()
    induced_width = 0
    max_clique_size = 0
    total_fill = 0
    peak_entries = 1
    peak_log10 = 0.0

    for node in order:
        neighbors = list(work.neighbors(node))
        clique = [node, *neighbors]
        new_edges = fill_edges(work, node)

        induced_width = max(induced_width, len(neighbors))
        max_clique_size = max(max_clique_size, len(clique))
        total_fill += len(new_edges)

        entries = _factor_entries(clique, cardinalities)
        peak_entries = max(peak_entries, entries)
        peak_log10 = max(
            peak_log10,
            sum(math.log10(cardinalities[item]) for item in clique),
        )

        work.add_edges_from(new_edges)
        work.remove_node(node)

    return {
        "induced_width": induced_width,
        "max_clique_size": max_clique_size,
        "total_fill_edges": total_fill,
        "peak_factor_entries": peak_entries,
        "peak_factor_log10": peak_log10,
        "final_edge_count": graph.number_of_edges() + total_fill,
    }


def graph_statistics(
    graph: nx.Graph, cardinalities: dict[object, int]
) -> dict[str, float | int]:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    degrees = [degree for _, degree in graph.degree()]
    cards = list(cardinalities.values())
    density = 0.0 if n <= 1 else 2.0 * m / (n * (n - 1))
    return {
        "n": n,
        "m": m,
        "density": density,
        "avg_degree": 0.0 if not degrees else sum(degrees) / len(degrees),
        "max_degree": 0 if not degrees else max(degrees),
        "min_cardinality": min(cards),
        "max_cardinality": max(cards),
        "avg_cardinality": sum(cards) / len(cards),
    }

