"""Four fixed slots per event: node, exploration, exploration action, tie draw."""

import numpy as np
from numpy.typing import NDArray


def make_tape(rng: np.random.Generator, events: int) -> NDArray[np.float64]:
    if events < 0:
        raise ValueError('events must be nonnegative')
    tape = rng.random((events, 4))
    tape.flags.writeable = False
    return tape


def validate_tape(tape: NDArray[np.float64]) -> None:
    if tape.ndim != 2 or tape.shape[1] != 4 or not np.isfinite(tape).all():
        raise ValueError('Expected finite event tape with four columns')
    if np.any(tape < 0) or np.any(tape >= 1):
        raise ValueError('Event uniforms must lie in [0,1)')
