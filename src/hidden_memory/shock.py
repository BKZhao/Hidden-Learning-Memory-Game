"""One-time public strategy reset, without learning or forced persistence."""

from math import floor

import numpy as np
from numpy.typing import NDArray

from .state import D, State, group_counts


def sample_shock(state: State, fraction: float, rng: np.random.Generator) -> NDArray[np.int64]:
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Shock fraction must be in [0,1]')
    count = floor(fraction * state.graph.n)
    contributors = np.flatnonzero(state.strategy != D)
    if count > len(contributors):
        raise ValueError('Not enough checkpoint contributors for the requested shock')
    return np.sort(rng.choice(contributors, count, replace=False))


def apply_shock(state: State, nodes: NDArray[np.int64]) -> None:
    if nodes.ndim != 1 or nodes.dtype.kind not in 'iu':
        raise ValueError('Shock nodes must be a one-dimensional integer array')
    if len(np.unique(nodes)) != len(nodes) or np.any(nodes < 0) or np.any(nodes >= state.graph.n):
        raise ValueError('Shock nodes must be distinct and in range')
    if np.any(state.strategy[nodes] == D):
        raise ValueError('Shock nodes must all be checkpoint contributors')
    state.strategy[nodes] = D
    state.counts[:] = group_counts(state.graph, state.strategy)
