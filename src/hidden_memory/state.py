"""Mutable state owned exclusively by one simulation branch."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .hypergraph import Hypergraph

C, D, P = 0, 1, 2


@dataclass
class State:
    graph: Hypergraph
    strategy: NDArray[np.int64]
    q: NDArray[np.float64]
    counts: NDArray[np.int64]
    visit_count: NDArray[np.int64]
    last_update_event: NDArray[np.int64]
    event: int = 0


def group_counts(graph: Hypergraph, strategy: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.stack([(strategy[graph.edges] == a).sum(axis=1) for a in range(3)], axis=1)


def initialize(graph: Hypergraph, seed: int) -> State:
    strategy = np.random.default_rng(seed).integers(0, 3, size=graph.n)
    shape = (graph.n, 3, 3)
    return State(deepcopy(graph), strategy, np.zeros(shape, dtype=np.float64),
                 group_counts(graph, strategy), np.zeros(shape, dtype=np.int64),
                 np.full(shape, -1, dtype=np.int64))


def clone(state: State) -> State:
    return deepcopy(state)


def validate_state(state: State) -> None:
    n = state.graph.n
    if state.strategy.shape != (n,) or state.strategy.dtype != np.int64:
        raise ValueError('strategy must be int64[N]')
    if np.any((state.strategy < C) | (state.strategy > P)):
        raise ValueError('Invalid strategy encoding')
    if state.q.shape != (n, 3, 3) or state.q.dtype != np.float64 or not np.isfinite(state.q).all():
        raise ValueError('Q must be finite float64[N,3,3]')
    if state.event < 0:
        raise ValueError('event must be nonnegative')
    for name in ('visit_count', 'last_update_event'):
        value = getattr(state, name)
        if value.shape != state.q.shape or value.dtype != np.int64:
            raise ValueError(f'Invalid {name} array')
    if np.any(state.visit_count < 0) or np.any(state.last_update_event < -1):
        raise ValueError('Invalid visitation metadata')
    if np.any(state.last_update_event >= state.event):
        raise ValueError('last_update_event must precede the next event')
    if not np.array_equal(state.counts, group_counts(state.graph, state.strategy)):
        raise ValueError('Cached group counts disagree with public strategies')
