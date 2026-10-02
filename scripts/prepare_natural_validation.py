"""Enforce E4c input-only stage gates before creating executable configurations."""

import argparse
import json
from pathlib import Path

from hidden_memory.checkpoint import load_checkpoint, source_hash
from hidden_memory.experiment_io import file_hash, write_json
from calibrate_natural_support import support_gate


def prepare_history(calibration: Path, output: Path) -> dict:
    decision = json.loads((calibration/'decision.json').read_text())
    base = json.loads((calibration/'config.json').read_text())
    if decision['results_sha256'] != file_hash(calibration/'results.jsonl'):
        raise ValueError('Calibration evidence changed')
    if decision['selected'] is None:
        return {'status': 'stopped', 'reason': 'No candidate passed the fixed calibration gate'}
    graphs = []
    for network in range(10, 20):
        path = Path(base['source'])/'histories'/f'network-{network:02d}-history-1/checkpoints/mcs-100000.npz'
        meta = load_checkpoint(path)[3]; graphs.append(meta['network_hash'])
    if len(set(graphs)) != 10 or set(graphs)&set(decision['network_hashes']):
        raise ValueError('Calibration and validation graphs must be distinct')
    config = {**base, **decision['selected'], 'scope': 'E4c validation input audit, separated from support calibration',
              'networks': list(range(10, 20)), 'history': 1, 'root_seed': 2026100141,
              'bootstrap_seed': 2026100142, 'source_sha256': source_hash(),
              'launcher_sha256': file_hash(Path(__file__).parent/'run_paired_cost_history.py'),
              'calibration_decision': str(calibration/'decision.json'),
              'calibration_decision_sha256': file_hash(calibration/'decision.json'),
              'validation_network_hashes': graphs}
    for key in ('candidates', 'checkpoints', 'script_sha256'): config.pop(key, None)
    write_json(output, config)
    return {'status': 'ready', 'selected': decision['selected'], 'config': str(output)}


def prepare_response(history: Path, output: Path) -> dict:
    if json.loads((history/'status.json').read_text())['state'] != 'completed':
        raise ValueError('Validation histories incomplete')
    config = json.loads((history/'config.json').read_text())
    if file_hash(Path(config['calibration_decision'])) != config['calibration_decision_sha256']:
        raise ValueError('Locked calibration decision changed')
    records = [json.loads(s) for s in (history/'results.jsonl').read_text().splitlines()]
    if sorted(r['network'] for r in records) != config['networks']: raise ValueError('Missing or duplicate validation network')
    gates = [support_gate(r, config['minimum_public_q']) for r in sorted(records, key=lambda r:r['network'])]
    eligible = [g['network'] for g in gates if g['passed']]
    passed = len(eligible) >= config['minimum_validation_networks']
    gate = {'status': 'passed' if passed else 'stopped', 'networks': gates, 'approved_networks': eligible,
            'minimum_required': config['minimum_validation_networks'], 'response_outcomes_used': False}
    path = history/'response-gate.json'
    if path.exists(): raise FileExistsError(path)
    write_json(path, gate)
    if passed:
        response = {**config, 'scope': 'E4c natural-gap response after fixed independent support gate',
                    'networks': eligible, 'histories': str(history), 'history_results_sha256': file_hash(history/'results.jsonl'),
                    'source_sha256': source_hash(), 'launcher_sha256': file_hash(Path(__file__).parent/'run_natural_response.py'),
                    'response_gate': str(path), 'response_gate_sha256': file_hash(path)}
        write_json(output, response)
    return gate


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); subs = p.add_subparsers(dest='stage', required=True)
    for stage in ('history', 'response'):
        q = subs.add_parser(stage); q.add_argument('source', type=Path); q.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    if args.config.exists(): raise FileExistsError(args.config)
    result = prepare_history(args.source, args.config) if args.stage == 'history' else prepare_response(args.source, args.config)
    print(json.dumps(result, indent=2))
    if result['status'] == 'stopped': raise SystemExit(2)
