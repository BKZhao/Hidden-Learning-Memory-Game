"""Compiled implementation of learner.step without per-event diagnostic objects."""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .parameters import Parameters
from .state import State


@njit(cache=True, fastmath=False)
def _small_sum(values, start, count):
    if count < 8:
        result = -0.0
        for i in range(start, start + count):
            result += values[i]
        return result
    a, b, c, d = values[start], values[start + 1], values[start + 2], values[start + 3]
    e, f, g, h = values[start + 4], values[start + 5], values[start + 6], values[start + 7]
    end = count - count % 8
    for i in range(8, end, 8):
        a += values[start + i]
        b += values[start + i + 1]
        c += values[start + i + 2]
        d += values[start + i + 3]
        e += values[start + i + 4]
        f += values[start + i + 5]
        g += values[start + i + 6]
        h += values[start + i + 7]
    result = ((a + b) + (c + d)) + ((e + f) + (g + h))
    for i in range(end, count):
        result += values[start + i]
    return result


@njit(cache=True, fastmath=False)
def _pairwise_sum(values, start, count):
    if count <= 128:
        return _small_sum(values, start, count)
    # Explicit traversal avoids recursive compiled function pointers in disk caches.
    starts = np.empty(64, dtype=np.int64)
    sizes = np.empty(64, dtype=np.int64)
    phases = np.zeros(64, dtype=np.int64)
    left_values = np.empty(64, dtype=np.float64)
    starts[0], sizes[0] = start, count
    depth = 0
    value = 0.0
    while depth >= 0:
        size = sizes[depth]
        if size <= 128:
            value = _small_sum(values, starts[depth], size)
            depth -= 1
        elif phases[depth] == 2:
            value = left_values[depth] + value
            phases[depth] = 0
            depth -= 1
        else:
            half = size // 2
            half -= half % 8
            if phases[depth] == 0:
                starts[depth + 1], sizes[depth + 1] = starts[depth], half
            else:
                left_values[depth] = value
                starts[depth + 1], sizes[depth + 1] = starts[depth] + half, size - half
            phases[depth] += 1
            depth += 1
            phases[depth] = 0
    return value


@njit(cache=True, fastmath=False)
def _advance(strategy, q, counts, visits, last_update, offsets, incidence,
             tape, event_start, r, alpha, beta, cost, eta, gamma, epsilon, freeze):
    n = len(strategy)
    scratch = np.empty(len(incidence), dtype=np.float64)
    for event_index in range(len(tape)):
        event = tape[event_index]
        node = int(event[0] * n)
        old = strategy[node]
        if event[1] < epsilon:
            action = int(event[2] * 3)
        else:
            maximum = max(q[node, old, 0], q[node, old, 1], q[node, old, 2])
            ties = 0
            for a in range(3):
                if q[node, old, a] == maximum:
                    ties += 1
            rank = int(event[3] * ties)
            action = 0
            for a in range(3):
                if q[node, old, a] == maximum:
                    if rank == 0:
                        action = a
                        break
                    rank -= 1
        begin, end = offsets[node], offsets[node + 1]
        for position in range(begin, end):
            edge = incidence[position]
            nc = counts[edge, 0] - (old == 0) + (action == 0)
            nd = counts[edge, 1] - (old == 1) + (action == 1)
            np_ = counts[edge, 2] - (old == 2) + (action == 2)
            base = r * cost * (nc + np_)
            if action == 0:
                payoff = base - cost
            elif action == 1:
                payoff = base - beta * np_
            else:
                payoff = base - cost - alpha * nd
            scratch[position - begin] = payoff
        reward = _pairwise_sum(scratch, 0, end - begin) / (end - begin)
        target = reward + gamma * max(q[node, action, 0], q[node, action, 1], q[node, action, 2])
        if not freeze:
            previous = q[node, old, action]
            q[node, old, action] = previous + eta * (target - previous)
        visits[node, old, action] += 1
        if not freeze and eta > 0:
            last_update[node, old, action] = event_start + event_index
        for position in range(begin, end):
            edge = incidence[position]
            counts[edge, old] -= 1
            counts[edge, action] += 1
        strategy[node] = action


def advance_mcs(state: State, params: Parameters, tape: NDArray[np.float64], freeze: bool) -> None:
    """Called after runner validates state/tape; update exactly one MCS in sequence."""
    _advance(state.strategy, state.q, state.counts, state.visit_count, state.last_update_event,
             state.graph.offsets, state.graph.incidence, tape, state.event,
             params.r, params.alpha, params.beta, params.c, params.eta,
             params.gamma, params.epsilon, freeze)
    state.event += len(tape)
