"""Independent hyperedge and weighted-projection reward implementations."""

import numpy as np
from numpy.typing import NDArray

from .parameters import Parameters
from .state import C, D, P, State


def group_payoffs(counts: NDArray[np.int64], params: Parameters) -> NDArray[np.float64]:
    nc, nd, np_ = np.moveaxis(np.asarray(counts), -1, 0)
    base = params.r * params.c * (nc + np_)
    return np.stack((base - params.c, base - params.beta * np_,
                     base - params.c - params.alpha * nd), axis=-1)


def reward(state: State, node: int, action: int, params: Parameters) -> float:
    graph = state.graph
    incident = graph.incidence[graph.offsets[node]:graph.offsets[node + 1]]
    counts = state.counts[incident].copy()
    counts[:, state.strategy[node]] -= 1
    counts[:, action] += 1
    return float(group_payoffs(counts, params)[:, action].mean())


def projected_reward(state: State, node: int, action: int, params: Parameters) -> float:
    graph = state.graph
    cooccurrences = np.zeros(graph.n, dtype=np.float64)
    for edge in graph.edges:
        if node in edge:
            cooccurrences[edge] += 1
    cooccurrences[node] = 0
    weights = cooccurrences / graph.degree[node]
    contribution = (state.strategy != D).astype(np.float64)
    neighbors = params.r * params.c * contribution
    neighbors -= params.alpha * (action == P) * (state.strategy == D)
    neighbors -= params.beta * (action == D) * (state.strategy == P)
    return float((params.r - 1) * params.c * (action != D) + weights @ neighbors)


def reward_bounds(g: int, params: Parameters) -> tuple[float, float]:
    values: list[float] = []
    for nc in range(g + 1):
        for nd in range(g - nc + 1):
            counts = np.array([nc, nd, g - nc - nd])
            values.extend(group_payoffs(counts, params)[counts > 0].tolist())
    return min(values), max(values)
