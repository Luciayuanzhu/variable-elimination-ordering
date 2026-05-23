from __future__ import annotations

import networkx as nx

from ve_ordering.metrics import evaluate_order
from ve_ordering.orderings import get_order


def test_chain_has_width_one_for_min_fill() -> None:
    graph = nx.path_graph(8)
    cards = {node: 2 for node in graph.nodes}
    order = get_order(graph, cards, "min_fill")
    metrics = evaluate_order(graph, cards, order)
    assert metrics["induced_width"] == 1


def test_complete_graph_order_width_is_n_minus_one() -> None:
    graph = nx.complete_graph(6)
    cards = {node: 2 for node in graph.nodes}
    order = get_order(graph, cards, "weighted_min_fill")
    metrics = evaluate_order(graph, cards, order)
    assert metrics["induced_width"] == 5
    assert metrics["total_fill_edges"] == 0


def test_hybrid_lambda_zero_matches_weighted_min_fill_on_small_graph() -> None:
    graph = nx.cycle_graph(6)
    graph.add_edge(0, 3)
    cards = {node: 2 + (node % 3) for node in graph.nodes}
    weighted = get_order(graph, cards, "weighted_min_fill")
    hybrid_zero = get_order(graph, cards, "hybrid", lambda_=0.0)
    assert hybrid_zero == weighted
