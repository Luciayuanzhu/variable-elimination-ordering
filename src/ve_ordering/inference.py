"""Small-scale explicit variable elimination for pairwise Markov networks."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import reduce
from operator import mul

import networkx as nx
import numpy as np


@dataclass
class Factor:
    vars: tuple[object, ...]
    values: np.ndarray

    def align(self, target_vars: tuple[object, ...], cardinalities: dict[object, int]) -> np.ndarray:
        present = [var for var in target_vars if var in self.vars]
        if present:
            axes = [self.vars.index(var) for var in present]
            arr = np.transpose(self.values, axes)
        else:
            arr = self.values
        shape = [cardinalities[var] if var in self.vars else 1 for var in target_vars]
        return arr.reshape(shape)

    def multiply(self, other: "Factor", cardinalities: dict[object, int]) -> "Factor":
        new_vars = tuple(dict.fromkeys((*self.vars, *other.vars)))
        values = self.align(new_vars, cardinalities) * other.align(new_vars, cardinalities)
        return Factor(new_vars, values)

    def sum_out(self, var: object) -> "Factor":
        axis = self.vars.index(var)
        new_values = self.values.sum(axis=axis)
        new_vars = tuple(item for item in self.vars if item != var)
        return Factor(new_vars, new_values)


def _entries(vars_: list[object] | tuple[object, ...], cardinalities: dict[object, int]) -> int:
    return reduce(mul, (int(cardinalities[var]) for var in vars_), 1)


def make_pairwise_factors(
    graph: nx.Graph, cardinalities: dict[object, int], seed: int
) -> list[Factor]:
    rng = np.random.default_rng(seed)
    factors: list[Factor] = []
    for node in graph.nodes:
        shape = (cardinalities[node],)
        values = rng.random(shape) + 0.1
        factors.append(Factor((node,), values))
    for u, v in graph.edges:
        shape = (cardinalities[u], cardinalities[v])
        values = rng.random(shape) + 0.1
        factors.append(Factor((u, v), values))
    return factors


def run_pairwise_variable_elimination(
    graph: nx.Graph,
    cardinalities: dict[object, int],
    order: list[object],
    max_factor_entries: int = 250_000,
    seed: int = 0,
) -> dict[str, float | str | int]:
    """Run explicit VE on random unary/pairwise factors when feasible."""
    factors = make_pairwise_factors(graph, cardinalities, seed)
    max_observed_entries = 1
    factor_ops = 0
    start = time.perf_counter()

    for var in order:
        involved = [factor for factor in factors if var in factor.vars]
        if not involved:
            continue
        remaining = [factor for factor in factors if var not in factor.vars]
        union_vars = tuple(dict.fromkeys(var_ for factor in involved for var_ in factor.vars))
        projected_entries = _entries(union_vars, cardinalities)
        if projected_entries > max_factor_entries:
            return {
                "ve_status": "skipped_factor_cutoff",
                "ve_runtime": math.nan,
                "ve_max_observed_entries": max_observed_entries,
                "ve_factor_ops": factor_ops,
            }

        product = involved[0]
        for factor in involved[1:]:
            product = product.multiply(factor, cardinalities)
            factor_ops += 1
        max_observed_entries = max(max_observed_entries, product.values.size)
        marginalized = product.sum_out(var)
        factor_ops += 1
        factors = remaining
        if marginalized.vars:
            factors.append(marginalized)

    runtime = time.perf_counter() - start
    return {
        "ve_status": "ok",
        "ve_runtime": runtime,
        "ve_max_observed_entries": max_observed_entries,
        "ve_factor_ops": factor_ops,
    }

