"""Snapshot accounting and finite-window outcomes, without statistical claims."""

import numpy as np
from numpy.typing import NDArray

from .parameters import Parameters
from .state import C, D, P, State


def snapshot(state: State, params: Parameters) -> dict[str, float]:
    counts = state.counts
    participation = state.graph.g * len(state.graph.edges)
    contributors = counts[:, C] + counts[:, P]
    encounters = int((counts[:, P] * counts[:, D]).sum())
    cost = params.alpha * encounters / participation
    fines = params.beta * encounters / participation
    qk = float(contributors.sum() / participation)
    benefit = (state.graph.g * params.r - 1) * params.c * qk
    return {'q': float(np.mean(state.strategy != D)),
            'fC': float(np.mean(state.strategy == C)), 'fP': float(np.mean(state.strategy == P)),
            'qk': qk, 'investment': params.c * float(contributors.sum()),
            'pd_encounters': float(encounters), 'punishment_cost': cost,
            'fine_loss': fines, 'welfare': benefit - cost - fines}


def signed_loss(unshocked: NDArray[np.float64], shocked: NDArray[np.float64]) -> float:
    if unshocked.shape != shocked.shape or unshocked.ndim != 1 or len(unshocked) == 0:
        raise ValueError('Loss requires aligned nonempty one-dimensional trajectories')
    if not np.isfinite(unshocked).all() or not np.isfinite(shocked).all():
        raise ValueError('Loss trajectories must be finite')
    return float(np.mean(unshocked - shocked))


def recovery_summary(q: NDArray[np.float64], q0: float, smooth: int = 50,
                     sustain: int = 200, shock_end: int = 0) -> dict[str, object]:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or not np.isfinite(q0):
        raise ValueError('Recovery requires a finite nonempty trajectory and baseline')
    if smooth < 1 or sustain < 1 or shock_end < 0 or shock_end >= len(q):
        raise ValueError('Invalid recovery window or shock end')
    # Input contains t=1..T. Only full trailing windows are eligible; no future padding.
    times = np.arange(smooth, len(q) + 1)
    if len(q) < smooth:
        trailing = np.empty(0, dtype=np.longdouble)
    else:
        windows = np.lib.stride_tricks.sliding_window_view(q, smooth)
        trailing = windows.mean(axis=1, dtype=np.longdouble)
    after_end = times > shock_end

    def first_run(condition: NDArray[np.bool_]) -> int | None:
        length = 0
        for time, valid in zip(times, condition & after_end):
            length = length + 1 if valid else 0
            if length == sustain:
                return int(time - sustain + 1)
        return None

    recovered = first_run(trailing >= q0 - 0.05)
    collapsed = first_run(trailing < q0 - 0.20)
    return {'recovery_time': recovered, 'right_censored': recovered is None,
            'censor_time': len(q), 'collapse': collapsed is not None,
            'collapse_time': collapsed,
            'restricted_unrecovered_time': (len(q) if recovered is None else recovered) - shock_end}


def stability(q: NDArray[np.float64], fp: NDArray[np.float64], window: int = 5000) -> dict[str, object]:
    if window < 1 or q.ndim != 1 or q.shape != fp.shape or len(q) < 2 * window:
        raise ValueError('Stability requires two complete aligned windows')
    if not np.isfinite(q).all() or not np.isfinite(fp).all():
        raise ValueError('Nonfinite stability trajectory')
    mean_q = float(q[-window:].mean())
    delta_q = float(abs(mean_q - q[-2 * window:-window].mean()))
    delta_fp = float(abs(fp[-window:].mean() - fp[-2 * window:-window].mean()))
    reasons = []
    if mean_q < 0.8:
        reasons.append('mean_q_below_0.80')
    if delta_q > 0.02:
        reasons.append('q_window_difference_above_0.02')
    if delta_fp > 0.02:
        reasons.append('fP_window_difference_above_0.02')
    return {'passed': not reasons, 'mean_q': mean_q, 'delta_q': delta_q,
            'delta_fP': delta_fp, 'reasons': reasons, 'window_mcs': window}
