"""Run one paired delta=.30 continuation for each legal dense-phase checkpoint."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
import shutil
import traceback

import numpy as np

from hidden_memory.checkpoint import load_checkpoint, provenance
from hidden_memory.events import make_tape
from hidden_memory.experiment_io import write_json
from hidden_memory.metrics import signed_loss
from hidden_memory.runner import run_trajectory
from hidden_memory.shock import apply_shock, sample_shock
from hidden_memory.state import clone


def response_seeds(root_seed: int, point: int, replicate: int) -> tuple[int, int]:
    children = np.random.SeedSequence([root_seed, 52, point, replicate]).spawn(2)
    return tuple(int(child.generate_state(1, dtype=np.uint64)[0]) for child in children)


def run_task(task: dict) -> dict:
    state, params, _, metadata = load_checkpoint(Path(task['checkpoint']))
    if metadata['network_hash'] != task['network_hash']:
        raise ValueError('Checkpoint network differs from response manifest')
    shock_seed, event_seed = response_seeds(task['root_seed'], task['point'], task['replicate'])
    shock_nodes = sample_shock(state, task['delta'], np.random.default_rng(shock_seed))
    tape = make_tape(np.random.default_rng(event_seed), state.graph.n*task['horizon'])
    baseline_state = clone(state)
    shock_state = clone(state)
    apply_shock(shock_state, shock_nodes)
    baseline = run_trajectory(baseline_state, params, tape, backend='numba')
    shocked = run_trajectory(shock_state, params, tape, backend='numba')
    drift = {metric: abs(float(baseline[metric][1:].mean())-task['tail_means'][metric])
             for metric in ('q', 'fP')}
    eligible = all(value <= task['drift_limit'] for value in drift.values())
    loss = signed_loss(baseline['q'][1:], shocked['q'][1:])
    output = Path(task['output'])/f"point-{task['point']:03d}-rep-{task['replicate']}"
    output.mkdir(exist_ok=False)
    arrays = {f'baseline_{metric}': baseline[metric] for metric in ('q', 'fP', 'welfare')}
    arrays.update({f'shocked_{metric}': shocked[metric] for metric in ('q', 'fP', 'welfare')})
    np.savez_compressed(output/'trajectories.npz', **arrays)
    result = {'point': task['point'], 'replicate': task['replicate'], 'r': params.r,
              'alpha': params.alpha, 'beta': params.beta, 'network_hash': metadata['network_hash'],
              'checkpoint_checksum': metadata['checksum'], 'shock_seed': shock_seed,
              'event_seed': event_seed, 'shock_nodes': shock_nodes.tolist(), 'delta': task['delta'],
              'loss': loss, 'baseline_drift': drift, 'eligible': eligible,
              'baseline_mean_q': float(baseline['q'][1:].mean()),
              'shocked_mean_q': float(shocked['q'][1:].mean())}
    write_json(output/'result.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    expected = config['script_sha256'][Path(__file__).name]
    if sha256(Path(__file__).read_bytes()).hexdigest() != expected:
        raise ValueError('Dense response launcher differs from fixed configuration')
    status = json.loads((args.training/'status.json').read_text())
    if status['state'] != 'completed' or status['failed']:
        raise ValueError('Dense training must complete without failures before response')
    manifest = json.loads((args.training/'manifest.json').read_text())
    training = {row['job_id']: row for row in map(json.loads, (args.training/'results.jsonl').read_text().splitlines())}
    tasks, skipped = [], []
    for job in manifest['jobs']:
        job_id = f"E5-p{job['point']:02d}-r{job['replicate']:02d}"
        record = training[job_id]
        reason = None
        if not record['stability']['passed']:
            reason = 'training_failed_public_gate'
        checkpoint = args.training/'jobs'/job_id/'checkpoints'/f"mcs-{config['training_mcs']}.npz"
        if reason is None:
            state, _, _, metadata = load_checkpoint(checkpoint)
            if np.count_nonzero(state.strategy != 1) < int(np.floor(config['delta']*job['nodes'])):
                reason = 'insufficient_checkpoint_contributors'
        if reason is not None:
            skipped.append({'point': job['point'], 'replicate': job['replicate'], 'reason': reason})
            continue
        tasks.append({'output': str(args.output), 'checkpoint': str(checkpoint),
                      'network_hash': metadata['network_hash'], 'point': job['point'],
                      'replicate': job['replicate'], 'root_seed': config['response_seed'],
                      'delta': config['delta'], 'horizon': config['response_mcs'],
                      'drift_limit': config['drift_limit'], 'tail_means': record['tail_means']})
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), args.output/Path(__file__).name)
    shutil.copyfile(args.config, args.output/'config.json')
    shutil.copyfile(args.training/'source.tar.gz', args.output/'training-source.tar.gz')
    write_json(args.output/'manifest.json', {'config': config, 'tasks': tasks, 'skipped': skipped,
                                             'provenance': provenance()})
    run_status = {'state': 'running', 'total': len(tasks), 'completed': 0, 'failed': 0}
    write_json(args.output/'status.json', run_status)
    with ProcessPoolExecutor(config['response_workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
                run_status['completed'] += 1
            except Exception as error:
                task = futures[future]
                result = {'point': task['point'], 'replicate': task['replicate'],
                          'error': repr(error), 'traceback': traceback.format_exc()}
                run_status['failed'] += 1
            with (args.output/'results.jsonl').open('a') as stream:
                stream.write(json.dumps(result, allow_nan=False)+'\n')
            write_json(args.output/'status.json', run_status)
    run_status['state'] = 'failed' if run_status['failed'] else 'completed'
    write_json(args.output/'status.json', run_status)
    if run_status['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
