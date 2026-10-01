"""Coordinate fixed-event runs and six paired branches; no reward logic."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .accelerated import advance_mcs
from .events import validate_tape
from .intervention import intervene
from .learner import EventTrace, step
from .metrics import signed_loss, snapshot
from .parameters import Parameters
from .shock import apply_shock
from .state import State, clone, validate_state


@dataclass
class PairedResult:
    trajectories: dict[tuple[float, int], dict[str, NDArray[np.float64]]]
    losses: dict[float, float]
    theta: float
    eligible_fraction: float


def run_trajectory(state: State, params: Parameters, tape: NDArray[np.float64],
                   freeze: bool = False, traces: list[EventTrace] | None = None,
                   backend: str = 'reference') -> dict[str, NDArray[np.float64]]:
    if backend not in ('reference', 'numba'):
        raise ValueError(f'Unknown backend: {backend}')
    if backend == 'numba' and traces is not None:
        raise ValueError('Per-event traces require the reference backend')
    validate_state(state)
    validate_tape(tape)
    if not len(tape) or len(tape) % state.graph.n:
        raise ValueError('Trajectory must contain a positive whole number of MCS')
    records = [snapshot(state, params)]
    if backend == 'numba':
        for start in range(0, len(tape), state.graph.n):
            advance_mcs(state, params, tape[start:start + state.graph.n], freeze)
            records.append(snapshot(state, params))
    else:
        for index, event in enumerate(tape):
            trace = step(state, params, event, freeze)
            if traces is not None:
                traces.append(trace)
            if (index + 1) % state.graph.n == 0:
                records.append(snapshot(state, params))
    validate_state(state)
    return {key: np.array([row[key] for row in records]) for key in records[0]}


def run_paired(state: State, params: Parameters, tape: NDArray[np.float64],
               shock_nodes: NDArray[np.int64], freeze: bool = False,
               backend: str = 'reference') -> PairedResult:
    branches, eligible = intervene(state, params)
    trajectories = {}
    for scale, edited in branches.items():
        for z in (0, 1):
            branch = clone(edited)
            if z:
                before = branch.q.copy()
                q_before = np.count_nonzero(branch.strategy != 1)
                apply_shock(branch, shock_nodes)
                if not np.array_equal(before, branch.q) or q_before - np.count_nonzero(branch.strategy != 1) != len(shock_nodes):
                    raise AssertionError('T10: shock violated Q invariance or contribution drop')
            trajectories[(scale, z)] = run_trajectory(branch, params, tape, freeze, backend=backend)
    if freeze:
        for z in (0, 1):
            for scale in (0.5, 2.0):
                for metric, reference in trajectories[(1.0, z)].items():
                    if not np.array_equal(reference, trajectories[(scale, z)][metric]):
                        raise AssertionError(f'T08: frozen trajectories differ in {metric}')
    losses = {scale: signed_loss(trajectories[(scale, 0)]['q'][1:], trajectories[(scale, 1)]['q'][1:]) for scale in branches}
    return PairedResult(trajectories, losses, losses[0.5] - losses[2.0], float(eligible.mean()))
