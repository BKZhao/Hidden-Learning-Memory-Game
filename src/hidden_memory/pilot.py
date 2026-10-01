"""Sham-only continuation and shock screening; never compare near/far Q interventions."""

from pathlib import Path

import numpy as np

from .checkpoint import array_hash, load_checkpoint, save_checkpoint
from .events import make_tape
from .experiment_io import write_json
from .experiment_specs import Job, job_seeds
from .metrics import recovery_summary, signed_loss
from .runner import run_trajectory
from .shock import apply_shock, sample_shock
from .state import clone


def screen_pilot(job: Job, output: Path, training_summary: dict) -> dict[str, object]:
    if not training_summary['stability']['passed']:
        return {'status': 'training_failed_screen', 'candidate': False, 'shocks': []}
    if job.continuation_mcs != job.response_mcs:
        raise ValueError('Pilot requires aligned continuation and shock response horizons')
    state, params, _, _ = load_checkpoint(output / 'checkpoints' / f'mcs-{job.training_mcs}.npz')
    seeds = job_seeds(job)
    pilot_dir = output / 'pilot'
    pilot_dir.mkdir(exist_ok=False)
    horizon = job.response_mcs

    def branch(fraction: float) -> tuple[dict, dict]:
        current = clone(state)
        nodes = sample_shock(current, fraction, np.random.default_rng(seeds['shock']))
        apply_shock(current, nodes)
        rng = np.random.default_rng(seeds['future'])
        parts: dict[str, list] = {}
        for offset in range(0, horizon, job.chunk_mcs):
            count = min(job.chunk_mcs, horizon - offset)
            trajectory = run_trajectory(current, params, make_tape(rng, job.nodes * count), backend='numba')
            for metric, values in trajectory.items():
                parts.setdefault(metric, []).append(values if offset == 0 else values[1:])
            write_json(output / 'progress.json', {'job_id': job.job_id, 'phase': 'pilot_sham',
                       'delta': fraction, 'completed_mcs': offset + count, 'target_mcs': horizon})
        trajectory = {metric: np.concatenate(values) for metric, values in parts.items()}
        np.savez_compressed(pilot_dir / f'delta-{fraction:.2f}.npz', **trajectory)
        metadata = save_checkpoint(pilot_dir / f'delta-{fraction:.2f}-end.npz', current, params,
                                   {'future': rng.bit_generator.state},
                                   {'training_id': job.job_id, 'parent_checkpoint_id': training_summary['checkpoint_id'],
                                    'delta': fraction, 'lambda': 1.0})
        return trajectory, {'delta': fraction, 'shock_nodes': nodes.tolist(), 'shock_nodes_hash': array_hash(nodes),
                            'future_event_seed': seeds['future'], 'checkpoint_id': metadata['checkpoint_id']}

    baseline, baseline_meta = branch(0.0)
    means = training_summary['tail_means']
    changes = {metric: abs(float(baseline[metric][1:].mean()) - means[metric]) for metric in ('q', 'fP')}
    no_drift = all(value <= 0.02 for value in changes.values())
    shocks = []
    for fraction in (0.05, 0.1, 0.2, 0.3):
        if int(np.floor(fraction * job.nodes)) > np.count_nonzero(state.strategy != 1):
            shocks.append({'delta': fraction, 'status': 'invalid_insufficient_contributors'})
            continue
        trajectory, meta = branch(fraction)
        outcome = recovery_summary(trajectory['q'][1:], means['q'])
        shocks.append({**meta, 'status': 'completed', 'signed_loss': signed_loss(baseline['q'][1:], trajectory['q'][1:]),
                       'minimum_q': float(trajectory['q'][1:].min()), **outcome})
    result = {'status': 'completed', 'candidate': no_drift, 'continuation_mean_changes': changes,
              'baseline': baseline_meta, 'shocks': shocks, 'work_point_locked': False}
    write_json(pilot_dir / 'summary.json', result)
    return result
