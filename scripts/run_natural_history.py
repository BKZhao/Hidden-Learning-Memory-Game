"""Fixed natural-cost histories, common washout, and outcome-blind transplant support."""

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
from hidden_memory.experiment_io import archive_source, file_hash, write_json
from hidden_memory.experiment_specs import Job
from hidden_memory.state import clone
from hidden_memory.training import train_job
from hidden_memory.natural_washout import wash_and_audit


def run_network(task: dict) -> dict:
    config = task['config']; network = task['network']
    output = Path(task['output'])/f'network-{network:02d}'
    output.mkdir(exist_ok=False)
    reference, params, _, meta = load_checkpoint(Path(task['checkpoint']))
    if meta['checksum'] != task['checkpoint_checksum']:
        raise ValueError('Locked source checkpoint changed')
    job = Job('E4', 0, network, config['root_seed'], reference.graph.n,
              len(reference.graph.edges), params.r, config['alpha_high'], params.beta,
              config['training_mcs'])
    summary = train_job(job, output/'high-training', network_seed=task['network_seed'])
    if summary['network_hash'] != meta['network_hash']:
        raise AssertionError('Natural cost histories must share the same graph')
    high, _, _, _ = load_checkpoint(output/'high-training/checkpoints'/f"mcs-{config['training_mcs']}.npz")
    wash_seed = int(np.random.SeedSequence([config['root_seed'], 400, network]).generate_state(1, dtype=np.uint64)[0])
    records = wash_and_audit(reference, {'low': clone(reference), 'high': high}, params, output, config, network, meta, wash_seed)
    result = {'network': network, 'network_hash': meta['network_hash'], 'reference_checksum': meta['checksum'],
              'high_training': summary, 'wash_seed': wash_seed, 'washouts': records}
    write_json(output/'result.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    if source_hash() != config['source_sha256'] or file_hash(Path(__file__)) != config['launcher_sha256']:
        raise ValueError('History implementation differs from locked configuration')
    source = Path(config['source'])
    hierarchy = json.loads((source/'hierarchy.json').read_text())
    tasks = []
    for network in config['networks']:
        h = next(t for t in hierarchy if t['network'] == network and t['history'] == 0)
        cp = source/'histories'/f'network-{network:02d}-history-0/checkpoints/mcs-100000.npz'
        _, _, _, meta = load_checkpoint(cp)
        tasks.append({'network': network, 'network_seed': h['network_seed'], 'checkpoint': str(cp),
                      'checkpoint_checksum': meta['checksum'], 'output': str(args.output), 'config': config})
    args.output.mkdir(exist_ok=False)
    archive_source(args.output, [], config['workers'])
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
