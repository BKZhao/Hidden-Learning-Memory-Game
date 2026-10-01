"""One asynchronous event; this is the sole owner of learning-state mutation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .parameters import Parameters
from .payoff import reward
from .state import State


@dataclass(frozen=True)
class EventTrace:
    event: int
    node: int
    old_state: int
    action: int
    explored: bool
    reward: float
    target: float
    old_value: float
    new_value: float
    greedy_before: tuple[int, ...]
    greedy_after: tuple[int, ...]


def policy(q: NDArray[np.float64], epsilon: float) -> NDArray[np.float64]:
    greedy = q == q.max(axis=-1, keepdims=True)
    return epsilon / 3 + (1 - epsilon) * greedy / greedy.sum(axis=-1, keepdims=True)


def choose_action(row: NDArray[np.float64], epsilon: float,
                  event: NDArray[np.float64]) -> tuple[int, bool]:
    if event[1] < epsilon:
        return int(event[2] * 3), True
    greedy = np.flatnonzero(row == row.max())
    return int(greedy[int(event[3] * len(greedy))]), False


def step(state: State, params: Parameters, event: NDArray[np.float64],
         freeze: bool = False) -> EventTrace:
    node = int(event[0] * state.graph.n)
    old_state = int(state.strategy[node])
    row = state.q[node, old_state]
    greedy_before = tuple(int(a) for a in np.flatnonzero(row == row.max()))
    action, explored = choose_action(row, params.epsilon, event)
    payoff = reward(state, node, action, params)
    old_value = float(row[action])
    target = payoff + params.gamma * float(state.q[node, action].max())
    if not freeze:
        row[action] = old_value + params.eta * (target - old_value)
    # Visits include choices with frozen Q; last_update_event records actual updates only.
    state.visit_count[node, old_state, action] += 1
    if not freeze and params.eta > 0:
        state.last_update_event[node, old_state, action] = state.event
    incident = state.graph.incidence[state.graph.offsets[node]:state.graph.offsets[node + 1]]
    state.counts[incident, old_state] -= 1
    state.counts[incident, action] += 1
    state.strategy[node] = action
    trace = EventTrace(state.event, node, old_state, action, explored, payoff,
                       target, old_value, float(row[action]), greedy_before,
                       tuple(int(a) for a in np.flatnonzero(row == row.max())))
    state.event += 1
    return trace
