"""One-time suboptimal P-gap changes on a shared eligible set."""

import numpy as np
from numpy.typing import NDArray

from .learner import policy
from .parameters import Parameters
from .payoff import reward_bounds
from .state import P, State, clone, validate_state


def intervene(state: State, params: Parameters,
              lambdas: tuple[float, ...] = (0.5, 1.0, 2.0),
              near_tie: float = 1e-10) -> tuple[dict[float, State], NDArray[np.bool_]]:
    validate_state(state)
    if not lambdas or len(set(lambdas)) != len(lambdas) or any(not np.isfinite(x) or x <= 0 for x in lambdas):
        raise ValueError('lambdas must be distinct finite positive values')
    if not np.isfinite(near_tie) or near_tie < 0:
        raise ValueError('near_tie must be finite and nonnegative')
    maximum = state.q[..., :2].max(axis=-1)
    gap = maximum - state.q[..., P]
    eligible = gap > near_tie
    low, high = reward_bounds(state.graph.g, params)
    low, high = min(0.0, low / (1 - params.gamma)), max(0.0, high / (1 - params.gamma))
    for scale in lambdas:
        candidate = maximum - scale * gap
        # A rounding-induced tie also excludes the row from every branch.
        eligible &= (candidate >= low) & (candidate <= high) & (candidate < maximum)
    result: dict[float, State] = {}
    initial_policy = policy(state.q, params.epsilon)
    for scale in lambdas:
        branch = clone(state)
        if scale != 1.0:
            branch.q[..., P][eligible] = (maximum - scale * gap)[eligible]
        if not np.array_equal(policy(branch.q, params.epsilon), initial_policy):
            raise AssertionError('T07: intervention changed an all-state action policy')
        result[scale] = branch
    return result, eligible
