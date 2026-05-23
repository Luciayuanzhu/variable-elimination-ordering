"""Graph topology generators and built-in benchmark-style moralized BNs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any
import warnings

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class GraphInstance:
    name: str
    topology: str
    graph: nx.Graph
    cardinalities: dict[object, int]
    metadata: dict[str, Any]


def relabel_with_prefix(graph: nx.Graph, prefix: str) -> nx.Graph:
    return nx.relabel_nodes(graph, {node: f"{prefix}_{node}" for node in graph.nodes})


def assign_cardinalities(
    graph: nx.Graph,
    mode: str,
    seed: int = 0,
    hub_stress: bool = False,
) -> dict[object, int]:
    rng = np.random.default_rng(seed)
    if mode == "binary":
        cards = {node: 2 for node in graph.nodes}
    elif mode == "mixed":
        values = np.array([2, 2, 3, 4, 5])
        cards = {node: int(rng.choice(values)) for node in graph.nodes}
    elif mode == "wide":
        values = np.array([2, 3, 4, 5, 8, 10])
        probs = np.array([0.35, 0.2, 0.15, 0.1, 0.1, 0.1])
        cards = {node: int(rng.choice(values, p=probs)) for node in graph.nodes}
    else:
        raise ValueError(f"unknown cardinality mode: {mode}")

    if hub_stress and graph.nodes:
        hubs = sorted(graph.degree, key=lambda item: (-item[1], str(item[0])))[: max(1, graph.number_of_nodes() // 20)]
        for node, _ in hubs:
            cards[node] = max(cards[node], 8)
    return cards


def make_chain(n: int) -> nx.Graph:
    return relabel_with_prefix(nx.path_graph(n), f"chain{n}")


def make_random_tree(n: int, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)
    if n == 1:
        graph = nx.empty_graph(1)
    else:
        prufer = rng.integers(0, n, size=n - 2).tolist()
        graph = nx.from_prufer_sequence(prufer)
    return relabel_with_prefix(graph, f"tree{n}s{seed}")


def make_polytree_like(n: int, seed: int) -> nx.Graph:
    """Generate a sparse moralized DAG that is close to a tree."""
    rng = np.random.default_rng(seed)
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n))
    for child in range(1, n):
        parent = int(rng.integers(0, child))
        dag.add_edge(parent, child)
        if child > 4 and rng.random() < 0.12:
            extra_parent = int(rng.integers(max(0, child - 8), child))
            if extra_parent != parent:
                dag.add_edge(extra_parent, child)
    graph = moralize_dag(dag)
    return relabel_with_prefix(graph, f"poly{n}s{seed}")


def make_grid(rows: int, cols: int) -> nx.Graph:
    graph = nx.grid_2d_graph(rows, cols)
    return nx.relabel_nodes(graph, {(r, c): f"grid{rows}x{cols}_{r}_{c}" for r, c in graph.nodes})


def make_random_graph(n: int, p: float, seed: int) -> nx.Graph:
    graph = nx.gnp_random_graph(n, p, seed=seed)
    return relabel_with_prefix(graph, f"er{n}p{int(p * 1000)}s{seed}")


def make_scale_free(n: int, m: int, seed: int) -> nx.Graph:
    graph = nx.barabasi_albert_graph(n, m, seed=seed)
    return relabel_with_prefix(graph, f"ba{n}m{m}s{seed}")


def moralize_dag(dag: nx.DiGraph) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(dag.nodes)
    graph.add_edges_from((u, v) for u, v in dag.edges)
    for child in dag.nodes:
        parents = list(dag.predecessors(child))
        graph.add_edges_from(combinations(parents, 2))
    return graph


def asia_moralized() -> GraphInstance:
    dag = nx.DiGraph()
    dag.add_edges_from(
        [
            ("VisitAsia", "Tuberculosis"),
            ("Smoking", "LungCancer"),
            ("Smoking", "Bronchitis"),
            ("Tuberculosis", "Either"),
            ("LungCancer", "Either"),
            ("Either", "XRay"),
            ("Either", "Dyspnea"),
            ("Bronchitis", "Dyspnea"),
        ]
    )
    graph = moralize_dag(dag)
    cards = {node: 2 for node in graph.nodes}
    return GraphInstance(
        name="asia_manual",
        topology="moralized_bn",
        graph=graph,
        cardinalities=cards,
        metadata={"source": "manual_asia_structure", "cardinality_mode": "binary"},
    )


def _layered_bn_like(
    name: str,
    layer_sizes: list[int],
    max_parents: int,
    seed: int,
    cardinality_values: list[int],
) -> GraphInstance:
    rng = np.random.default_rng(seed)
    dag = nx.DiGraph()
    layers: list[list[str]] = []
    index = 0
    for layer_id, size in enumerate(layer_sizes):
        layer = [f"{name}_{index + offset}" for offset in range(size)]
        index += size
        layers.append(layer)
        dag.add_nodes_from(layer)

    for layer_id in range(1, len(layers)):
        possible_parents = [node for prev in layers[max(0, layer_id - 2):layer_id] for node in prev]
        for child in layers[layer_id]:
            parent_count = int(rng.integers(1, min(max_parents, len(possible_parents)) + 1))
            parents = rng.choice(possible_parents, size=parent_count, replace=False)
            for parent in parents:
                dag.add_edge(str(parent), child)

    graph = moralize_dag(dag)
    cards = {node: int(rng.choice(cardinality_values)) for node in graph.nodes}
    return GraphInstance(
        name=f"{name}_moralized",
        topology="moralized_bn",
        graph=graph,
        cardinalities=cards,
        metadata={
            "source": "built_in_bn_like",
            "layers": "-".join(map(str, layer_sizes)),
            "max_parents": max_parents,
            "cardinality_mode": "bn_native_like",
            "seed": seed,
        },
    )


def benchmark_bn_instances() -> list[GraphInstance]:
    canonical = load_pgmpy_bn_instances()
    if canonical:
        return canonical
    return [
        asia_moralized(),
        _layered_bn_like("child_like", [3, 5, 6, 6], 2, 101, [2, 2, 3, 4]),
        _layered_bn_like("alarm_like", [5, 8, 10, 8, 6], 3, 102, [2, 2, 3, 4]),
        _layered_bn_like("insurance_like", [4, 7, 8, 8, 6], 3, 103, [2, 3, 4, 5]),
        _layered_bn_like("hailfinder_like", [6, 10, 14, 12, 10, 6], 3, 104, [2, 3, 4, 5]),
    ]


def load_pgmpy_bn_instances() -> list[GraphInstance]:
    """Load canonical pgmpy example Bayesian networks when available."""
    loaders = []
    try:
        from pgmpy.example_models import load_model
        loaders.append(load_model)
    except Exception:
        pass
    try:
        from pgmpy.utils import get_example_model
        loaders.append(get_example_model)
    except Exception:
        pass
    if not loaders:
        return []

    instances: list[GraphInstance] = []
    for name in ["asia", "alarm", "child", "insurance", "hailfinder"]:
        model = None
        for loader in loaders:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    model = loader(name)
                break
            except Exception:
                continue
        if model is None:
            continue
        try:
            dag = nx.DiGraph()
            dag.add_nodes_from(model.nodes())
            dag.add_edges_from(model.edges())
            graph = moralize_dag(dag)
            cardinalities = {node: 2 for node in graph.nodes}
            for cpd in model.get_cpds():
                variable = getattr(cpd, "variable", None)
                variable_card = getattr(cpd, "variable_card", None)
                if variable is not None and variable_card is not None:
                    cardinalities[variable] = int(variable_card)
            instances.append(
                GraphInstance(
                    name=f"pgmpy_{name}",
                    topology="moralized_bn",
                    graph=graph,
                    cardinalities=cardinalities,
                    metadata={
                        "source": "pgmpy_example_model",
                        "benchmark": name,
                        "cardinality_mode": "bn_native",
                    },
                )
            )
        except Exception:
            continue
    return instances


def make_experiment_instances() -> list[GraphInstance]:
    instances: list[GraphInstance] = []

    for n in [25, 50, 100]:
        graph = make_chain(n)
        instances.append(
            GraphInstance(
                name=f"chain_n{n}",
                topology="low_treewidth",
                graph=graph,
                cardinalities=assign_cardinalities(graph, "mixed", seed=n),
                metadata={"family": "chain", "cardinality_mode": "mixed"},
            )
        )

    for n in [50, 100, 200]:
        graph = make_random_tree(n, seed=n)
        instances.append(
            GraphInstance(
                name=f"tree_n{n}",
                topology="low_treewidth",
                graph=graph,
                cardinalities=assign_cardinalities(graph, "mixed", seed=2 * n),
                metadata={"family": "tree", "cardinality_mode": "mixed"},
            )
        )

    for n in [50, 100]:
        graph = make_polytree_like(n, seed=n + 7)
        instances.append(
            GraphInstance(
                name=f"polytree_like_n{n}",
                topology="low_treewidth",
                graph=graph,
                cardinalities=assign_cardinalities(graph, "mixed", seed=3 * n),
                metadata={"family": "polytree_like", "cardinality_mode": "mixed"},
            )
        )

    for n in [40, 80, 120]:
        for p in [0.03, 0.06, 0.10]:
            for seed in [0, 1, 2]:
                graph = make_random_graph(n, p, seed)
                instances.append(
                    GraphInstance(
                        name=f"er_n{n}_p{p:.2f}_s{seed}",
                        topology="random",
                        graph=graph,
                        cardinalities=assign_cardinalities(graph, "mixed", seed=seed + n),
                        metadata={
                            "family": "erdos_renyi",
                            "p": p,
                            "seed": seed,
                            "cardinality_mode": "mixed",
                        },
                    )
                )

    for rows, cols in [(5, 5), (8, 8), (10, 10), (12, 12)]:
        graph = make_grid(rows, cols)
        instances.append(
            GraphInstance(
                name=f"grid_{rows}x{cols}",
                topology="grid",
                graph=graph,
                cardinalities=assign_cardinalities(graph, "mixed", seed=rows * 100 + cols),
                metadata={
                    "family": "grid",
                    "rows": rows,
                    "cols": cols,
                    "cardinality_mode": "mixed",
                },
            )
        )

    for n in [50, 100, 150]:
        for m in [1, 2, 3]:
            for seed in [0, 1, 2]:
                graph = make_scale_free(n, m, seed)
                instances.append(
                    GraphInstance(
                        name=f"ba_n{n}_m{m}_s{seed}",
                        topology="scale_free",
                        graph=graph,
                        cardinalities=assign_cardinalities(
                            graph, "mixed", seed=n + m + seed, hub_stress=True
                        ),
                        metadata={
                            "family": "barabasi_albert",
                            "attachment_m": m,
                            "seed": seed,
                            "cardinality_mode": "mixed_hub_stress",
                        },
                    )
                )

    instances.extend(benchmark_bn_instances())
    return instances
