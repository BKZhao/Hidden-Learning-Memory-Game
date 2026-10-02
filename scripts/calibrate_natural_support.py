"""Select one prespecified cost history using input support only, never responses."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import shutil
from time import perf_counter
import traceback

from hidden_memory.checkpoint import load_checkpoint, source_hash
from hidden_memory.experiment_io import archive_source, file_hash, write_json
from run_paired_cost_history import run_network


def support_gate(record: dict, minimum_q: float) -> dict:
    public_ok = all(w['raw_history'][key]['snapshot']['q'] >= minimum_q
                    for w in record['washouts'] for key in ('low', 'high'))
    matched = all(w['eligible'] for w in record['washouts'])
    return {'network': record['network'], 'passed': public_ok and matched,
            'public_q_passed': public_ok, 'common_support_passed': matched,
            'common_rows': record['washouts'][0]['common_rows']}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('config', type=Path); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); config = json.loads(args.config.read_text())
    if source_hash() != config['source_sha256']: raise ValueError('Simulation changed after lock')
    for name, digest in config['script_sha256'].items():
        if file_hash(Path(__file__).parent/name) != digest: raise ValueError('Launcher changed after lock')
    args.output.mkdir(exist_ok=False); archive_source(args.output, [], config['workers'])
    shutil.copyfile(args.config, args.output/'config.json')
    for name in config['script_sha256']: shutil.copyfile(Path(__file__).parent/name, args.output/name)
    tasks = []; graphs = []
    for candidate in config['candidates']:
        output = args.output/candidate['id']; output.mkdir()
        for h, path in enumerate(config['checkpoints']):
            _, _, _, meta = load_checkpoint(Path(path)); graphs.append(meta['network_hash'])
            tasks.append({'candidate': candidate['id'], 'network': h, 'checkpoint': path,
                          'checkpoint_checksum': meta['checksum'], 'output': str(output),
                          'config': {**config, **candidate}})
    if len(set(graphs)) != len(config['checkpoints']): raise ValueError('Calibration graphs not distinct')
    write_json(args.output/'tasks.json', tasks)
    results = []; start = perf_counter()
    status = {'state': 'running', 'total': len(tasks), 'completed': 0, 'failed': 0}
    write_json(args.output/'status.json', status)
    with ProcessPoolExecutor(config['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(run_network, task): task for task in tasks}
        for future in as_completed(pending):
            task = pending[future]
            try:
                result = {'candidate': task['candidate'], **future.result()}; results.append(result); status['completed'] += 1
            except Exception as error:
                result = {'candidate': task['candidate'], 'network': task['network'], 'error': repr(error), 'traceback': traceback.format_exc()}; status['failed'] += 1
            with (args.output/'results.jsonl').open('a') as stream: stream.write(json.dumps(result)+'\n')
            write_json(args.output/'status.json', status); print(json.dumps(status), flush=True)
    status.update(state='failed' if status['failed'] else 'completed', elapsed_seconds=perf_counter()-start)
    write_json(args.output/'status.json', status)
    if status['failed']: raise SystemExit(1)
    candidates = []; selected = None
    for candidate in config['candidates']:
        gates = [support_gate(r, config['minimum_public_q']) for r in sorted(results, key=lambda r:r['network']) if r['candidate'] == candidate['id']]
        passed = sum(g['passed'] for g in gates) >= config['minimum_calibration_networks']
        candidates.append({'candidate': candidate, 'passed': passed, 'networks': gates})
        if passed and selected is None: selected = candidate
    write_json(args.output/'decision.json', {'selected': selected, 'candidates': candidates,
               'rule': 'First candidate in locked order passing input-only gates; no response outcomes generated',
               'network_hashes': sorted(set(graphs)), 'results_sha256': file_hash(args.output/'results.jsonl')})


if __name__ == '__main__': main()
