"""Simple uniform hypergraphs, projection connectivity and sparse incidence."""

from dataclasses import dataclass
from math import comb

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Hypergraph:
    n: int
    edges: NDArray[np.int64]
    offsets: NDArray[np.int64]
    incidence: NDArray[np.int64]
    degree: NDArray[np.int64]
    rejected_graphs: int = 0

    @property
    def g(self) -> int:
        return self.edges.shape[1]


def from_edges(n: int, edges: object, rejected_graphs: int = 0) -> Hypergraph:
    raw = np.asarray(edges)
    if raw.dtype.kind not in 'iu' or raw.ndim != 2 or raw.shape[0] == 0:
        raise ValueError('edges must be a nonempty rectangular integer array')
    array = np.sort(raw.astype(np.int64), axis=1)
    if n < 2 or array.shape[1] < 2 or array.min() < 0 or array.max() >= n:
        raise ValueError('Invalid graph size or node index')
    if np.any(np.diff(array, axis=1) == 0) or len(np.unique(array, axis=0)) != len(array):
        raise ValueError('Nodes within an edge and hyperedges must be distinct')
    parents = list(range(n))

    def root(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    buckets: list[list[int]] = [[] for _ in range(n)]
    for h, edge in enumerate(array):
        for node in edge:
            buckets[node].append(h)
            parents[root(int(node))] = root(int(edge[0]))
    if len({root(i) for i in range(n)}) != 1:
        raise ValueError('Hypergraph projection is disconnected')
    degree = np.array([len(b) for b in buckets], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(degree)]
    incidence = np.array([h for b in buckets for h in b], dtype=np.int64)
    return Hypergraph(n, array, offsets, incidence, degree, rejected_graphs)


def sample_connected(n: int, g: int, edge_count: int, rng: np.random.Generator,
                     max_attempts: int = 1000) -> Hypergraph:
    if not (2 <= g <= n) or not (1 <= edge_count <= comb(n, g)):
        raise ValueError('Requested simple uniform hypergraph is impossible')
    if edge_count * (g - 1) < n - 1:
        raise ValueError('Too few edges to connect all nodes')
    for attempt in range(max_attempts):
        edges: set[tuple[int, ...]] = set()
        while len(edges) < edge_count:
            edges.add(tuple(sorted(int(x) for x in rng.choice(n, g, replace=False))))
        # Sorting removes dependence on set iteration order from serialized graphs.
        candidate = sorted(edges)
        try:
            return from_edges(n, candidate, attempt)
        except ValueError as exc:
            if str(exc) != 'Hypergraph projection is disconnected':
                raise
    raise RuntimeError(f'No connected graph after {max_attempts} whole-graph attempts')
