"""Stream training trajectories and preserve checkpoint/diagnostic evidence."""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from .checkpoint import save_checkpoint
from .events import make_tape
from .experiment_io import file_hash, write_json
from .experiment_specs import Job, job_seeds
from .hypergraph import sample_connected
from .metrics import snapshot, stability
from .parameters import Parameters
from .runner import run_trajectory
from .state import State, initialize


def hidden_diagnostics(state: State) -> dict[str, object]:
    gaps = state.q[..., :2].max(axis=-1) - state.q[..., 2]
    greedy = state.q == state.q.max(axis=-1, keepdims=True)
    codes = (greedy * np.array([1, 2, 4])).sum(axis=-1)
    visited = state.last_update_event >= 0
    ages = (state.event - state.last_update_event[visited]) / state.graph.n
    return {'gap_quantiles': np.quantile(gaps, [0.01, 0.25, 0.5, 0.75, 0.99]).tolist(),
            'current_gap_quantiles': np.quantile(gaps[np.arange(state.graph.n), state.strategy], [0.01, 0.25, 0.5, 0.75, 0.99]).tolist(),
            'greedy_set_counts': np.bincount(codes.ravel(), minlength=8).tolist(),
            'visited_fraction': float(visited.mean()), 'mean_visits': float(state.visit_count.mean()),
            'age_quantiles_mcs': np.quantile(ages, [0.25, 0.5, 0.75, 0.99]).tolist() if len(ages) else None}


def train_job(job: Job, output: Path, *, network_seed: int | None = None,
              selection_times: tuple[int, ...] = ()) -> dict[str, object]:
    if job.training_mcs < 2 * job.window_mcs or job.chunk_mcs < 1:
        raise ValueError('Training must cover two reference windows with positive chunk size')
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    seeds = job_seeds(job)
    if network_seed is not None:
        seeds['network'] = network_seed
    if selection_times and (tuple(sorted(set(selection_times))) != selection_times
                            or selection_times[-1] != job.training_mcs
                            or selection_times[0] < 2 * job.window_mcs):
        raise ValueError('Selection times must be ordered, cover two windows, and end at the training limit')
    network_rng = np.random.default_rng(seeds['network'])
    graph = sample_connected(job.nodes, job.group_size, job.edges, network_rng)
    state = initialize(graph, seeds['initialization'])
    params = Parameters(r=job.r, alpha=job.alpha, beta=job.beta)
    rng = np.random.default_rng(seeds['training'])
    columns = ['mcs', *snapshot(state, params)]
    write_json(output / 'config.json', {'job': asdict(job), 'seeds': seeds, 'backend': 'numba'})
    write_json(output / 'trajectory_columns.json', columns)
    trajectory = np.lib.format.open_memmap(output / 'trajectory.npy', mode='w+', dtype=np.float64,
                                           shape=(job.training_mcs + 1, len(columns)))
    trajectory[0] = [0, *snapshot(state, params).values()]
    milestones = sorted({0, job.training_mcs, *selection_times, *[m for m in (100, 1000, 5000, 10000, 15000, 20000, 100000, 200000, 500000) if m < job.training_mcs]})
    diagnostics = []

    def checkpoint(mcs: int) -> dict:
        diagnostic = {'training_id': job.job_id, 'stage': job.stage, 'mcs': mcs,
                      'hidden_state': hidden_diagnostics(state)}
        if mcs >= 2 * job.window_mcs:
            diagnostic['stability'] = stability(trajectory[1:mcs + 1, columns.index('q')],
                                                trajectory[1:mcs + 1, columns.index('fP')], job.window_mcs)
        metadata = save_checkpoint(output / 'checkpoints' / f'mcs-{mcs}.npz', state, params,
                                   {'training': rng.bit_generator.state, 'network': network_rng.bit_generator.state,
                                    'seed_recipe': seeds}, diagnostic)
        diagnostics.append(diagnostic)
        write_json(output / 'diagnostics.json', diagnostics)
        return metadata

    checkpoint(0)
    completed = 0
    for milestone in milestones[1:]:
        while completed < milestone:
            count = min(job.chunk_mcs, milestone - completed)
            values = run_trajectory(state, params, make_tape(rng, job.nodes * count), backend='numba')
            trajectory[completed + 1:completed + count + 1, 0] = np.arange(completed + 1, completed + count + 1)
            for index, metric in enumerate(columns[1:], 1):
                trajectory[completed + 1:completed + count + 1, index] = values[metric][1:]
            completed += count
            trajectory.flush()
            write_json(output / 'progress.json', {'job_id': job.job_id, 'phase': 'training',
                       'completed_mcs': completed, 'target_mcs': job.training_mcs,
                       'micro_updates': state.event, 'elapsed_seconds': perf_counter() - started})
        metadata = checkpoint(milestone)
        if milestone in selection_times and diagnostics[-1]['stability']['passed']:
            break
    if completed < job.training_mcs:
        retained = trajectory[:completed + 1].copy()
        del trajectory
        temporary = output / 'trajectory-selected.npy'
        np.save(temporary, retained)
        temporary.replace(output / 'trajectory.npy')
        trajectory = retained
    summary = {'job_id': job.job_id, 'stage': job.stage, 'r': job.r, 'alpha': job.alpha, 'beta': job.beta,
               'replicate': job.replicate, 'training_micro_updates': state.event,
               'network_rejections': graph.rejected_graphs, 'training_seconds': perf_counter() - started,
               'checkpoint_id': metadata['checkpoint_id'], 'checkpoint_checksum': metadata['checksum'],
               'network_hash': metadata['network_hash'],
               'trajectory_sha256': file_hash(output / 'trajectory.npy'),
               'stability': diagnostics[-1]['stability'],
               'tail_means': {metric: float(trajectory[-job.window_mcs:, i].mean()) for i, metric in enumerate(columns) if i},
               'final_hidden_state': diagnostics[-1]['hidden_state']}
    if selection_times:
        summary['selection'] = {'planned_times': selection_times, 'completed_mcs': completed,
                                'selected_mcs': completed if diagnostics[-1]['stability']['passed'] else None,
                                'checked': [{'mcs': d['mcs'], 'stability': d['stability']}
                                            for d in diagnostics if d['mcs'] in selection_times]}
    write_json(output / 'training_summary.json', summary)
    return summary
