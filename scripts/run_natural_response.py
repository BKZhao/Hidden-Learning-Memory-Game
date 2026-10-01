"""Responses to fixed natural-gap transplants and descriptive raw histories."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import shutil
from time import perf_counter
import traceback

import numpy as np

from hidden_memory.checkpoint import load_checkpoint, source_hash
from hidden_memory.events import make_tape
from hidden_memory.experiment_io import archive_source, file_hash, write_json
from hidden_memory.learner import policy
from hidden_memory.runner import run_trajectory
from hidden_memory.shock import apply_shock, sample_shock
from hidden_memory.state import clone


def run_network(task: dict) -> dict:
    config = task['config']; network = task['network']
    output = Path(task['output'])/f'network-{network:02d}'; output.mkdir(exist_ok=False)
    streams = np.random.SeedSequence([config['root_seed'], 401, network]).spawn(2)
    shock_seed, event_seed = [int(s.generate_state(1, dtype=np.uint64)[0]) for s in streams]
    tape = None; records = []; updates = 0
    for entry in task['washouts']:
        folder = output/f"wash-{entry['wash_mcs']}"; folder.mkdir()
        arrays = {}; sources = {}; record = {'wash_mcs': entry['wash_mcs'], 'transplant_eligible': entry['eligible'], 'groups': {}}
        for name, spec in entry['checkpoints'].items():
            if file_hash(Path(spec['path'])) != spec['sha256']: raise ValueError('Locked checkpoint file changed')
            state, params, _, _ = load_checkpoint(Path(spec['path']))
            sources[name] = state
            if tape is None: tape = make_tape(np.random.default_rng(event_seed), state.graph.n*config['response_mcs'])
        if entry['eligible']:
            initial = sources['transplant_sham']
            for key in ('low', 'high'):
                if not np.array_equal(policy(sources[f'transplant_{key}'].q, params.epsilon), policy(initial.q, params.epsilon)):
                    raise AssertionError('Transplant initial policies differ')
        for group, names in [('transplant', ('sham', 'low', 'high')), ('raw', ('low', 'high'))]:
            if group == 'transplant' and not entry['eligible']: continue
            details = {}
            for name in names:
                initial = sources[f'{group}_{name}']
                legal = int(np.count_nonzero(initial.strategy != 1)) >= int(np.floor(config['delta']*initial.graph.n))
                nodes = sample_shock(initial, config['delta'], np.random.default_rng(shock_seed)) if legal else None
                for z in ((0, 1) if legal else (0,)):
                    branch = clone(initial)
                    if z: apply_shock(branch, nodes)
                    trajectory = run_trajectory(branch, params, tape, backend='numba'); updates += len(tape)
                    for metric, values in trajectory.items(): arrays[f'{group}_{name}_z{z}_{metric}'] = values
                loss = float(np.mean(arrays[f'{group}_{name}_z0_q'][1:]-arrays[f'{group}_{name}_z1_q'][1:])) if legal else None
                details[name] = {'shock_legal': legal, 'shock_nodes': nodes.tolist() if legal else None, 'loss': loss}
            details['theta_low_minus_high'] = details['low']['loss']-details['high']['loss'] if all(details[k]['shock_legal'] for k in ('low', 'high')) else None
            record['groups'][group] = details
        np.savez_compressed(folder/'trajectories.npz', **arrays)
        if entry['frozen']:
            frozen = {}
            for name in ('sham', 'low', 'high'):
                initial = sources[f'transplant_{name}']
                nodes = sample_shock(initial, config['delta'], np.random.default_rng(shock_seed))
                for z in (0, 1):
                    branch = clone(initial)
                    if z: apply_shock(branch, nodes)
                    trajectory = run_trajectory(branch, params, tape, freeze=True, backend='numba'); updates += len(tape)
                    for metric, values in trajectory.items():
                        frozen[f'{name}_z{z}_{metric}'] = values
                        if not np.array_equal(values, frozen[f'sham_z{z}_{metric}']): raise AssertionError('Frozen natural transplant differs')
            np.savez_compressed(folder/'frozen.npz', **frozen)
        record['frozen_passed'] = entry['frozen']; records.append(record)
        write_json(folder/'result.json', record)
    result = {'network': network, 'event_seed': event_seed, 'shock_seed': shock_seed, 'washouts': records, 'micro_updates': updates}
    write_json(output/'result.json', result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('config', type=Path); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); config = json.loads(args.config.read_text())
    if source_hash() != config['source_sha256'] or file_hash(Path(__file__)) != config['launcher_sha256']:
        raise ValueError('Response implementation differs from lock')
    source = Path(config['histories'])
    if file_hash(source/'results.jsonl') != config['history_results_sha256']: raise ValueError('History results changed')
    if json.loads((source/'status.json').read_text())['state'] != 'completed': raise ValueError('History stage incomplete')
    tasks = []; frozen_assigned = False
    for network in config['networks']:
        washouts = []
        for wash in config['wash_mcs']:
            folder = source/f'network-{network:02d}/wash-{wash}'
            audit = json.loads((folder/'audit.json').read_text())
            paths = {f'raw_{key}': folder/f'{key}.npz' for key in ('low', 'high')}
            if audit['eligible']: paths.update({f'transplant_{key}': folder/f'common-{key}.npz' for key in ('sham', 'low', 'high')})
            freeze = audit['eligible'] and not frozen_assigned; frozen_assigned |= freeze
            washouts.append({'wash_mcs': wash, 'eligible': audit['eligible'], 'frozen': freeze,
                             'audit_sha256': file_hash(folder/'audit.json'),
                             'checkpoints': {key: {'path': str(path), 'sha256': file_hash(path)} for key, path in paths.items()}})
        tasks.append({'network': network, 'config': config, 'output': str(args.output), 'washouts': washouts})
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
