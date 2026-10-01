"""Validated scalar model inputs; no simulation state."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Parameters:
    r: float
    alpha: float = 0.1
    beta: float = 0.8
    c: float = 1.0
    eta: float = 0.8
    gamma: float = 0.8
    epsilon: float = 0.02

    def __post_init__(self) -> None:
        if not all(isfinite(v) for v in vars(self).values()):
            raise ValueError('Model parameters must be finite')
        if self.r < 0 or self.alpha < 0 or self.beta < 0 or self.c <= 0:
            raise ValueError('r, alpha, beta must be nonnegative; c must be positive')
        if not (0 <= self.eta <= 1 and 0 <= self.gamma < 1 and 0 <= self.epsilon <= 1):
            raise ValueError('Require eta, epsilon in [0,1] and gamma in [0,1)')
