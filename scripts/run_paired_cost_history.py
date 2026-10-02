"""E4b: paired natural cost phases starting from the same selected checkpoint."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
import shutil
from time import perf_counter
import traceback

import numpy as np

from hidden_memory.checkpoint import load_checkpoint, save_checkpoint, source_hash
from hidden_memory.events import make_tape
from hidden_memory.experiment_io import archive_source, file_hash, write_json
from hidden_memory.natural_washout import wash_and_audit
from hidden_memory.runner import run_trajectory
from hidden_memory.state import clone


def run_network(task: dict) -> dict:
    config = task['config']; network = task['network']
    output = Path(task['output'])/f'network-{network:02d}'; output.mkdir(exist_ok=False)
    reference, params, _, meta = load_checkpoint(Path(task['checkpoint']))
    if meta['checksum'] != task['checkpoint_checksum']: raise ValueError('Source checkpoint changed')
    sources = {key: clone(reference) for key in ('low', 'high')}
    phase_seed = int(np.random.SeedSequence([config['root_seed'], 402, network]).generate_state(1, dtype=np.uint64)[0])
    wash_seed = int(np.random.SeedSequence([config['root_seed'], 400, network]).generate_state(1, dtype=np.uint64)[0])
    rng = np.random.default_rng(phase_seed)
    trajectories = {key: [] for key in sources}
    for key, state in sources.items():
        save_checkpoint(output/f'initial-{key}.npz', state, params, {}, {'purpose': 'identical_cost_phase_start'})
    for start in range(0, config['cost_phase_mcs'], 1000):
        tape = make_tape(rng, reference.graph.n*min(1000, config['cost_phase_mcs']-start))
        for key, state in sources.items():
            phase_params = params if key == 'low' else replace(params, alpha=config['alpha_high'])
            trajectories[key].append(run_trajectory(state, phase_params, tape, backend='numba'))
    np.savez_compressed(output/'cost-phase.npz', **{
        f'{key}_{metric}': np.concatenate([part[metric][1:] for part in parts])
        for key, parts in trajectories.items() for metric in parts[0]})
    for key, state in sources.items():
        phase_params = params if key == 'low' else replace(params, alpha=config['alpha_high'])
        save_checkpoint(output/f'cost-end-{key}.npz', state, phase_params, {'cost_phase': rng.bit_generator.state},
                        {'network': network, 'cost_phase_mcs': config['cost_phase_mcs'], 'reference_checksum': meta['checksum']})
    washouts = wash_and_audit(reference, sources, params, output, config, network, meta, wash_seed)
    result = {'network': network, 'network_hash': meta['network_hash'], 'reference_checksum': meta['checksum'],
              'phase_seed': phase_seed, 'wash_seed': wash_seed, 'washouts': washouts}
    write_json(output/'result.json', result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('config', type=Path); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); config = json.loads(args.config.read_text())
    if source_hash() != config['source_sha256'] or file_hash(Path(__file__)) != config['launcher_sha256']:
        raise ValueError('Paired-cost implementation differs from lock')
    tasks = []
    for network in config['networks']:
        cp = Path(config['source'])/'histories'/f"network-{network:02d}-history-{config.get('history', 0)}/checkpoints/mcs-100000.npz"
        _, _, _, meta = load_checkpoint(cp)
        tasks.append({'network': network, 'checkpoint': str(cp), 'checkpoint_checksum': meta['checksum'],
                      'output': str(args.output), 'config': config})
    args.output.mkdir(exist_ok=False); archive_source(args.output, [], config['workers'])
    shutil.copyfile(args.config, args.output/'config.json'); shutil.copyfile(__file__, args.output/Path(__file__).name)
    write_json(args.output/'tasks.json', tasks)
    status = {'state': 'running', 'total': len(tasks), 'completed': 0, 'failed': 0}; start = perf_counter()
    write_json(args.output/'status.json', status)
    with ProcessPoolExecutor(config['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(run_network, task): task for task in tasks}
        for future in as_completed(pending):
            try:
                result = future.result(); status['completed'] += 1
            except Exception as error:
                result = {'network': pending[future]['network'], 'error': repr(error), 'traceback': traceback.format_exc()}; status['failed'] += 1
            with (args.output/'results.jsonl').open('a') as stream: stream.write(json.dumps(result)+'\n')
            write_json(args.output/'status.json', status); print(json.dumps(status), flush=True)
    status.update(state='failed' if status['failed'] else 'completed', elapsed_seconds=perf_counter()-start)
    write_json(args.output/'status.json', status)
    if status['failed']: raise SystemExit(1)


if __name__ == '__main__': main()
