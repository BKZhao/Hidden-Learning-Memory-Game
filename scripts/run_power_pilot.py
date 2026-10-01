"""Estimate network-level paired variability after a sham-only work-point lock."""

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
from hidden_memory.runner import run_paired
from hidden_memory.shock import sample_shock


def run_network(task: dict) -> dict:
    output = Path(task['output'])/f"network-{task['replicate']:02d}"
    output.mkdir(exist_ok=False)
    state, params, _, meta = load_checkpoint(Path(task['checkpoint']))
    if meta['network_hash'] != task['network_hash']:
        raise ValueError('Source network differs from locked pilot input')
    outcomes = []
    for future in range(task['futures']):
        streams = np.random.SeedSequence([task['seed'], 91, task['replicate'], future]).spawn(2)
        shock_seed, event_seed = [int(s.generate_state(1, dtype=np.uint64)[0]) for s in streams]
        nodes = sample_shock(state, task['delta'], np.random.default_rng(shock_seed))
        tape = make_tape(np.random.default_rng(event_seed), state.graph.n*task['horizon'])
        result = run_paired(state, params, tape, nodes, backend='numba')
        arrays = {f'lambda_{scale}_shock_{z}_{metric}': values
                  for (scale, z), metrics in result.trajectories.items() for metric, values in metrics.items()}
        np.savez_compressed(output/f'future-{future:02d}.npz', **arrays)
        record = {'future': future, 'shock_seed': shock_seed, 'event_seed': event_seed,
                  'shock_nodes': nodes.tolist(), 'losses': result.losses, 'theta': result.theta,
                  'eligible_fraction': result.eligible_fraction}
        if future == 0:
            frozen = run_paired(state, params, tape, nodes, freeze=True, backend='numba')
            if frozen.theta != 0:
                raise AssertionError('Frozen negative control has nonzero theta')
            np.savez_compressed(output/'frozen-control.npz', **{
                f'lambda_{scale}_shock_{z}_{metric}': values
                for (scale, z), metrics in frozen.trajectories.items() for metric, values in metrics.items()})
            record['full_horizon_frozen_theta'] = frozen.theta
        outcomes.append(record)
        write_json(output/'progress.json', {'completed_futures': future+1, 'target': task['futures']})
    result = {'replicate': task['replicate'], 'network_hash': task['network_hash'],
              'checkpoint_id': meta['checkpoint_id'], 'checkpoint_checksum': meta['checksum'],
              'outcomes': outcomes, 'network_mean_theta': float(np.mean([r['theta'] for r in outcomes]))}
    write_json(output/'result.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if sha256(Path(__file__).read_bytes()).hexdigest() != config['launcher_sha256']:
        raise ValueError('Power pilot launcher hash mismatch')
    lock_path = Path(config['work_point_lock'])
    if sha256(lock_path.read_bytes()).hexdigest() != config['work_point_lock_sha256']:
        raise ValueError('Work-point lock changed')
    lock = json.loads(lock_path.read_text())
    if lock['status'] != 'work_point_locked' or lock['point'] != config['point']:
        raise ValueError('A prior sham-only work-point lock is required')
    if config['checkpoint_rule'] != 'public_training_gate_at_100000_all_planned_networks':
        raise ValueError('Unsupported fixed checkpoint inclusion rule')
    source = Path(config['source_campaign'])
    tasks, skipped = [], []
    for rep in range(config['networks']):
        folder = source/'jobs'/f"E1-p{config['point']:02d}-r{rep:02d}"
        record = json.loads((folder/'result.json').read_text())
        if not record['stability']['passed']:
            skipped.append({'replicate': rep, 'reason': 'training_failed_public_gate'})
            continue
        cp = folder/'checkpoints'/f"mcs-{config['training_mcs']}.npz"
        _, _, _, meta = load_checkpoint(cp)
        tasks.append({'output': str(args.output), 'replicate': rep, 'checkpoint': str(cp),
                      'network_hash': meta['network_hash'], 'seed': config['root_seed'],
                      'futures': config['future_replicates'], 'delta': lock['delta'], 'horizon': config['response_mcs']})
    if len({t['network_hash'] for t in tasks}) != len(tasks) or len(tasks) < 3:
        raise ValueError('Need at least three distinct source networks')
    args.output.mkdir(exist_ok=False)
    shutil.copyfile(__file__, args.output/Path(__file__).name)
    shutil.copyfile(args.config, args.output/'config.json')
    shutil.copyfile(lock_path, args.output/'work-point-lock.json')
    shutil.copyfile(source/'source.tar.gz', args.output/'simulation-source.tar.gz')
    write_json(args.output/'manifest.json', {'config': config, 'tasks': tasks, 'skipped': skipped, 'provenance': provenance()})
    status = {'state': 'running', 'total': len(tasks), 'completed': 0, 'failed': 0}
    write_json(args.output/'status.json', status)
    with ProcessPoolExecutor(config['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(run_network, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
                status['completed'] += 1
            except Exception as error:
                result = {'replicate': futures[future]['replicate'], 'error': repr(error),
                          'traceback': traceback.format_exc()}
                status['failed'] += 1
            with (args.output/'results.jsonl').open('a') as f:
                f.write(json.dumps(result)+'\n')
            write_json(args.output/'status.json', status)
    status['state'] = 'failed' if status['failed'] else 'completed'
    write_json(args.output/'status.json', status)
    if status['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
