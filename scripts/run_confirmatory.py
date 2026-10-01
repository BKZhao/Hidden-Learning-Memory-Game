"""Execute a fixed E2 network/history hierarchy and the six paired response arms."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
import shutil
import traceback

import numpy as np

from hidden_memory.audit import invariant_events
from hidden_memory.checkpoint import load_checkpoint, source_hash
from hidden_memory.config import validate_confirmatory
from hidden_memory.events import make_tape
from hidden_memory.experiment_io import archive_source, write_json
from hidden_memory.experiment_specs import Job, job_seeds
from hidden_memory.shock import sample_shock
from hidden_memory.training import train_job
from run_power_pilot import run_network


def run_history(task: dict) -> dict:
    job = Job(**task['job'])
    folder = Path(task['output'])/'histories'/f"network-{task['network']:02d}-history-{task['history']}"
    summary = train_job(job, folder, network_seed=task['network_seed'],
                        selection_times=tuple(task['selection_times']))
    if summary['network_hash'] in task.get('excluded_network_hashes', []):
        raise AssertionError('A confirmatory network duplicates a pilot network')
    selected = summary['selection']['selected_mcs']
    identity = {'network': task['network'], 'history': task['history'], 'training': summary}
    if selected is None:
        result = {**identity, 'status': 'training_not_eligible', 'response': None}
    else:
        checkpoint = folder/'checkpoints'/f'mcs-{selected}.npz'
        state, params, _, meta = load_checkpoint(checkpoint)
        nodes = sample_shock(state, task['delta'], np.random.default_rng(job_seeds(job)['audit']))
        controls = invariant_events(state, params, make_tape(np.random.default_rng(job_seeds(job)['audit']), 32), nodes)
        write_json(folder/'controls.json', controls)
        response_dir = folder/'response'
        response_dir.mkdir()
        response = run_network({'output': str(response_dir), 'replicate': job.replicate,
                                'checkpoint': str(checkpoint), 'network_hash': meta['network_hash'],
                                'seed': task['future_seed'], 'futures': task['future_replicates'],
                                'delta': task['delta'], 'horizon': task['response_mcs']})
        result = {**identity, 'status': 'completed', 'response': response}
    write_json(folder/'confirmatory-result.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_confirmatory(config)
    histories = config['independent_networks']*config['training_histories']
    maximum_updates = histories*config['n']*(config['selection_times'][-1]
                      + 6*config['response_mcs']*(config['future_replicates']+1))+histories*320
    if maximum_updates > config['maximum_micro_updates']:
        raise ValueError('The fixed design exceeds the locked micro-update budget')
    if source_hash() != config['code_version']:
        raise ValueError('Simulation source differs from the locked version')
    for name, digest in config['script_sha256'].items():
        if sha256((Path(__file__).parent/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Script differs from lock: {name}')
    args.output.mkdir(exist_ok=False)
    (args.output/'histories').mkdir()
    shutil.copyfile(args.config, args.output/'config.json')
    for name in config['script_sha256']:
        shutil.copyfile(Path(__file__).parent/name, args.output/name)
    jobs, tasks = [], []
    for network in range(config['independent_networks']):
        seed = int(np.random.SeedSequence([config['root_seed'], 200, network]).generate_state(1, dtype=np.uint64)[0])
        for history in range(config['training_histories']):
            job = Job('E2', 0, network*config['training_histories']+history, config['root_seed'],
                      config['n'], config['edges'], config['r'], config['alpha'], config['beta'], config['selection_times'][-1])
            jobs.append(job)
            tasks.append({'job': asdict(job), 'network': network, 'history': history, 'network_seed': seed,
                          'output': str(args.output), 'selection_times': config['selection_times'],
                          'future_seed': config['root_seed'], 'future_replicates': config['future_replicates'],
                          'delta': config['delta'], 'response_mcs': config['response_mcs'],
                          'excluded_network_hashes': config['pilot_network_hashes']})
    archive_source(args.output, jobs, config['workers'])
    write_json(args.output/'hierarchy.json', tasks)
    status = {'state': 'running', 'total': len(tasks), 'completed': 0, 'failed': 0, 'training_not_eligible': 0}
    write_json(args.output/'status.json', status)
    network_hashes = {}
    with ProcessPoolExecutor(config['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(run_history, task): task for task in tasks}
        for future in as_completed(pending):
            try:
                result = future.result()
                network = result['network']
                digest = result['training']['network_hash']
                if network in network_hashes and network_hashes[network] != digest:
                    raise AssertionError('Histories on the same network have different graphs')
                if any(h != network and d == digest for h, d in network_hashes.items()):
                    raise AssertionError('Different network IDs have identical graphs')
                network_hashes[network] = digest
                status['completed'] += 1
                status['training_not_eligible'] += result['status'] == 'training_not_eligible'
            except Exception as error:
                task = pending[future]
                result = {'network': task['network'], 'history': task['history'], 'status': 'failed',
                          'error': repr(error), 'traceback': traceback.format_exc()}
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
