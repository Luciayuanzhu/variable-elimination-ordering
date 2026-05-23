"""Elimination ordering heuristics."""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import combinations
from typing import Callable, Iterable

import networkx as nx


Cardinalities = dict[object, int]


def _node_key(node: object) -> str:
    return str(node)


def fill_edges(graph: nx.Graph, node: object) -> list[tuple[object, object]]:
    """Return fill-in edges created by eliminating node in the current graph."""
    neighbors = list(graph.neighbors(node))
    missing: list[tuple[object, object]] = []
    for u, w in combinations(neighbors, 2):
        if not graph.has_edge(u, w):
            missing.append((u, w))
    return missing


def weighted_fill_score(
    graph: nx.Graph, node: object, cardinalities: Cardinalities
) -> float:
    score = 0.0
    for u, w in fill_edges(graph, node):
        score += math.log(cardinalities[u] * cardinalities[w])
    return score


def hybrid_score(
    graph: nx.Graph,
    node: object,
    cardinalities: Cardinalities,
    lambda_: float,
) -> float:
    """Weighted min-fill plus a local downstream clique-pressure proxy."""
    immediate = weighted_fill_score(graph, node, cardinalities)
    neighbors = list(graph.neighbors(node))
    if not neighbors:
        return immediate

    incident_fill_count = {u: 0 for u in neighbors}
    for u, w in fill_edges(graph, node):
        incident_fill_count[u] += 1
        incident_fill_count[w] += 1

    pressure = 0.0
    for u in neighbors:
        deg_after_fill = graph.degree[u] + incident_fill_count[u]
        pressure += deg_after_fill * deg_after_fill
    pressure /= max(1, len(neighbors))
    return immediate + lambda_ * pressure


def fill_pressure_score(
    graph: nx.Graph,
    node: object,
    cardinalities: Cardinalities,
    lambda_: float,
) -> float:
    """Weighted min-fill plus pressure only around newly created fill edges.

    This avoids penalizing a whole high-degree neighborhood when eliminating a
    variable creates little or no new structure. The intent is to make the proxy
    less brittle on hub-dominated graphs than the original degree-squared term.
    """
    immediate = weighted_fill_score(graph, node, cardinalities)
    new_edges = fill_edges(graph, node)
    if not new_edges:
        return immediate

    added_degree = {u: 0 for edge in new_edges for u in edge}
    for u, w in new_edges:
        added_degree[u] += 1
        added_degree[w] += 1

    pressure = 0.0
    for u, w in new_edges:
        edge_weight = math.log(cardinalities[u] * cardinalities[w])
        u_after = graph.degree[u] - 1 + added_degree[u]
        w_after = graph.degree[w] - 1 + added_degree[w]
        pressure += edge_weight * (u_after + w_after)
    pressure /= max(1, len(new_edges))
    return immediate + lambda_ * pressure


def _local_factor_log10(
    graph: nx.Graph, node: object, cardinalities: Cardinalities
) -> float:
    return sum(
        math.log10(cardinalities[u]) for u in [node, *list(graph.neighbors(node))]
    )


def next_step_pressure(
    graph_after: nx.Graph,
    cardinalities: Cardinalities,
    next_candidate_limit: int = 5,
) -> float:
    """Cheap proxy for the best next elimination cost after a candidate move."""
    if not graph_after.nodes:
        return 0.0
    candidates = sorted(
        graph_after.nodes,
        key=lambda u: (
            weighted_fill_score(graph_after, u, cardinalities),
            _local_factor_log10(graph_after, u, cardinalities),
            _node_key(u),
        ),
    )[:next_candidate_limit]
    next_costs = []
    for candidate in candidates:
        next_costs.append(
            weighted_fill_score(graph_after, candidate, cardinalities)
            + _local_factor_log10(graph_after, candidate, cardinalities)
        )
    return min(next_costs)


def next_step_score(
    graph: nx.Graph,
    node: object,
    cardinalities: Cardinalities,
    lambda_: float,
) -> float:
    """Weighted min-fill with a one-step simulated downstream objective."""
    immediate = weighted_fill_score(graph, node, cardinalities)
    work = graph.copy()
    _eliminate_in_place(work, node)
    return immediate + lambda_ * next_step_pressure(work, cardinalities)


def _eliminate_in_place(graph: nx.Graph, node: object) -> None:
    graph.add_edges_from(fill_edges(graph, node))
    graph.remove_node(node)


def _greedy_order(
    graph: nx.Graph,
    score_fn: Callable[[nx.Graph, object], tuple[float, ...]],
) -> list[object]:
    work = graph.copy()
    order: list[object] = []
    while work.nodes:
        best = min(work.nodes, key=lambda node: (score_fn(work, node), _node_key(node)))
        order.append(best)
        _eliminate_in_place(work, best)
    return order


def _limited_candidate_order(
    graph: nx.Graph,
    cardinalities: Cardinalities,
    score_fn: Callable[[nx.Graph, object], tuple[float, ...]],
    candidate_limit: int = 12,
) -> list[object]:
    """Greedy order that evaluates expensive scores on top WMF candidates."""
    work = graph.copy()
    order: list[object] = []
    while work.nodes:
        candidates = sorted(
            work.nodes,
            key=lambda node: (
                weighted_fill_score(work, node, cardinalities),
                len(fill_edges(work, node)),
                _node_key(node),
            ),
        )[:candidate_limit]
        best = min(candidates, key=lambda node: (score_fn(work, node), _node_key(node)))
        order.append(best)
        _eliminate_in_place(work, best)
    return order


def min_degree_order(graph: nx.Graph) -> list[object]:
    return _greedy_order(graph, lambda g, v: (float(g.degree[v]),))


def min_fill_order(graph: nx.Graph) -> list[object]:
    return _greedy_order(graph, lambda g, v: (float(len(fill_edges(g, v))),))


def weighted_min_fill_order(
    graph: nx.Graph, cardinalities: Cardinalities
) -> list[object]:
    return _greedy_order(
        graph, lambda g, v: (weighted_fill_score(g, v, cardinalities),)
    )


def hybrid_order(
    graph: nx.Graph, cardinalities: Cardinalities, lambda_: float
) -> list[object]:
    return _greedy_order(
        graph, lambda g, v: (hybrid_score(g, v, cardinalities, lambda_),)
    )


def fill_pressure_order(
    graph: nx.Graph, cardinalities: Cardinalities, lambda_: float
) -> list[object]:
    return _greedy_order(
        graph, lambda g, v: (fill_pressure_score(g, v, cardinalities, lambda_),)
    )


def next_step_order(
    graph: nx.Graph, cardinalities: Cardinalities, lambda_: float
) -> list[object]:
    return _limited_candidate_order(
        graph,
        cardinalities,
        lambda g, v: (next_step_score(g, v, cardinalities, lambda_),),
        candidate_limit=8,
    )


def adaptive_lambda(graph: nx.Graph) -> float:
    """Simple topology-aware lambda chosen from graph statistics only."""
    n = graph.number_of_nodes()
    if n <= 2 or nx.is_forest(graph):
        return 0.0
    m = graph.number_of_edges()
    density = 0.0 if n <= 1 else 2.0 * m / (n * (n - 1))
    degrees = [degree for _, degree in graph.degree()]
    avg_degree = sum(degrees) / len(degrees)
    max_degree = max(degrees)
    heterogeneity = max_degree / max(1e-9, avg_degree)
    clustering = nx.average_clustering(graph) if n <= 400 else 0.0

    if heterogeneity >= 3.5:
        return 0.0
    if max_degree <= 4 and avg_degree >= 2.4 and clustering < 0.05:
        return 0.5
    if density >= 0.10 or clustering >= 0.20:
        return 0.01
    return 0.05


def adaptive_fill_pressure_order(
    graph: nx.Graph, cardinalities: Cardinalities
) -> list[object]:
    work = graph.copy()
    order: list[object] = []
    while work.nodes:
        lambda_ = adaptive_lambda(work)
        best = min(
            work.nodes,
            key=lambda node: (
                fill_pressure_score(work, node, cardinalities, lambda_),
                _node_key(node),
            ),
        )
        order.append(best)
        _eliminate_in_place(work, best)
    return order


def _graph_signature(graph: nx.Graph) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    nodes = tuple(sorted((_node_key(node) for node in graph.nodes)))
    edges = tuple(
        sorted(
            tuple(sorted((_node_key(u), _node_key(v))))
            for u, v in graph.edges
        )
    )
    return nodes, edges


def _rebuild_graph(
    nodes: Iterable[object], edges: Iterable[tuple[object, object]]
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def bounded_lookahead_order(
    graph: nx.Graph,
    cardinalities: Cardinalities,
    depth: int = 2,
    candidate_limit: int = 8,
) -> list[object]:
    """Expensive oracle-style greedy order with bounded future simulation.

    At each step, this evaluates a limited set of promising candidates by
    recursively simulating future eliminations. It is meant for small pilot
    graphs, not full-scale benchmarking.
    """
    original_nodes_by_name = {_node_key(node): node for node in graph.nodes}
    cards_by_name = {_node_key(node): cardinalities[node] for node in graph.nodes}

    @lru_cache(maxsize=200_000)
    def score_state(
        nodes_sig: tuple[str, ...],
        edges_sig: tuple[tuple[str, str], ...],
        remaining_depth: int,
    ) -> tuple[float, float, float]:
        state_graph = _rebuild_graph(nodes_sig, edges_sig)
        if not state_graph.nodes or remaining_depth <= 0:
            return (0.0, 0.0, 0.0)

        candidates = sorted(
            state_graph.nodes,
            key=lambda v: (
                weighted_fill_score(state_graph, v, cards_by_name),
                len(fill_edges(state_graph, v)),
                _node_key(v),
            ),
        )[:candidate_limit]

        best_score = (float("inf"), float("inf"), float("inf"))
        for candidate in candidates:
            neighbors = list(state_graph.neighbors(candidate))
            local_width = float(len(neighbors))
            local_log_size = sum(
                math.log10(cards_by_name[u]) for u in [candidate, *neighbors]
            )
            local_fill = float(len(fill_edges(state_graph, candidate)))

            next_graph = state_graph.copy()
            _eliminate_in_place(next_graph, candidate)
            next_nodes, next_edges = _graph_signature(next_graph)
            future_width, future_log_size, future_fill = score_state(
                next_nodes, next_edges, remaining_depth - 1
            )
            candidate_score = (
                max(local_width, future_width),
                max(local_log_size, future_log_size),
                local_fill + future_fill,
            )
            if candidate_score < best_score:
                best_score = candidate_score
        return best_score

    work = nx.relabel_nodes(graph.copy(), {node: _node_key(node) for node in graph.nodes})
    order_names: list[str] = []
    while work.nodes:
        candidates = sorted(
            work.nodes,
            key=lambda v: (
                weighted_fill_score(work, v, cards_by_name),
                len(fill_edges(work, v)),
                _node_key(v),
            ),
        )[:candidate_limit]
        best = min(
            candidates,
            key=lambda candidate: (
                _candidate_lookahead_score(
                    work, candidate, cards_by_name, score_state, depth
                ),
                _node_key(candidate),
            ),
        )
        order_names.append(best)
        _eliminate_in_place(work, best)
    return [original_nodes_by_name[name] for name in order_names]


def _candidate_lookahead_score(
    graph: nx.Graph,
    candidate: object,
    cards_by_name: dict[str, int],
    score_state: Callable[
        [tuple[str, ...], tuple[tuple[str, str], ...], int],
        tuple[float, float, float],
    ],
    depth: int,
) -> tuple[float, float, float]:
    neighbors = list(graph.neighbors(candidate))
    local_width = float(len(neighbors))
    local_log_size = sum(math.log10(cards_by_name[u]) for u in [candidate, *neighbors])
    local_fill = float(len(fill_edges(graph, candidate)))
    next_graph = graph.copy()
    _eliminate_in_place(next_graph, candidate)
    next_nodes, next_edges = _graph_signature(next_graph)
    future_width, future_log_size, future_fill = score_state(
        next_nodes, next_edges, depth - 1
    )
    return (
        max(local_width, future_width),
        max(local_log_size, future_log_size),
        local_fill + future_fill,
    )


def get_order(
    graph: nx.Graph,
    cardinalities: Cardinalities,
    method: str,
    lambda_: float = 0.0,
    lookahead_depth: int | None = None,
) -> list[object]:
    """Dispatch ordering method by name."""
    if method == "min_degree":
        return min_degree_order(graph)
    if method == "min_fill":
        return min_fill_order(graph)
    if method == "weighted_min_fill":
        return weighted_min_fill_order(graph, cardinalities)
    if method == "hybrid":
        return hybrid_order(graph, cardinalities, lambda_)
    if method == "fill_pressure":
        return fill_pressure_order(graph, cardinalities, lambda_)
    if method == "next_step":
        return next_step_order(graph, cardinalities, lambda_)
    if method == "adaptive_fill_pressure":
        return adaptive_fill_pressure_order(graph, cardinalities)
    if method == "bounded_lookahead":
        return bounded_lookahead_order(
            graph,
            cardinalities,
            depth=2 if lookahead_depth is None else lookahead_depth,
        )
    raise ValueError(f"unknown ordering method: {method}")
