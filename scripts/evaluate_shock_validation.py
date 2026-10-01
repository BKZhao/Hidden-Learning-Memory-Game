"""Apply fixed operational hidden-state and sham-response gates to every planned run."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from hidden_memory.checkpoint import load_checkpoint
from hidden_memory.payoff import reward_bounds
from hidden_memory.training import hidden_diagnostics


def hidden_changes(before, after, params) -> dict:
    low, high = reward_bounds(before.graph.g, params)
    width = max(0., high/(1-params.gamma))-min(0., low/(1-params.gamma))
    first, last = hidden_diagnostics(before), hidden_diagnostics(after)
    gap_change = np.abs(np.array(last['gap_quantiles'][1:4])-first['gap_quantiles'][1:4]).max()/width
    greedy_change = np.abs(np.array(last['greedy_set_counts'][1:])-first['greedy_set_counts'][1:]).max()/(3*before.graph.n)
    return {'normalized_gap_change': float(gap_change), 'greedy_fraction_change': float(greedy_change),
            'visited_fraction_change': abs(last['visited_fraction']-first['visited_fraction'])}


def passes_hidden(changes: dict, gate: dict) -> bool:
    return all(changes[key] <= gate[key] for key in changes)


def evaluate(root: Path) -> dict:
    config = json.loads((root/'validation-config.json').read_text())
    expected_hash = config['script_sha256'][Path(__file__).name]
    if sha256(Path(__file__).read_bytes()).hexdigest() != expected_hash:
        raise ValueError('Evaluator differs from the fixed design')
    status = json.loads((root/'status.json').read_text())
    if status['state'] not in ('completed', 'failed'):
        raise ValueError('Wait for all planned tasks before evaluation')
    manifest = json.loads((root/'manifest.json').read_text())
    raw = (root/'results.jsonl').read_bytes()
    records = [json.loads(s) for s in raw.splitlines()]
    results = {r['job_id']: r for r in records}
    if len(results) != len(records):
        raise ValueError('Duplicate result ID')
    rows = []
    for job in manifest['jobs']:
        key = f"E1-p{job['point']:02d}-r{job['replicate']:02d}"
        result = results.get(key, {})
        row = {'job_id': key, 'point': job['point'], 'status': result.get('status', 'missing'),
               'public_stable': False, 'hidden_stable': False, 'joint_stable': False, 'moderate': False,
               'recovery_time': None, 'right_censored': None, 'collapse': None,
               'normalized_gap_change': None, 'greedy_fraction_change': None, 'visited_fraction_change': None}
        pilot = result.get('pilot')
        if result.get('status') == 'completed' and pilot and pilot['status'] == 'completed':
            path = root/'jobs'/key
            before, params, _, _ = load_checkpoint(path/'checkpoints'/f"mcs-{job['training_mcs']}.npz")
            after, _, _, _ = load_checkpoint(path/'pilot/delta-0.00-end.npz')
            changes = hidden_changes(before, after, params)
            row.update(changes)
            row['public_stable'] = bool(result['stability']['passed'] and pilot['candidate'])
            row['hidden_stable'] = passes_hidden(changes, config['hidden_gate'])
            row['joint_stable'] = row['public_stable'] and row['hidden_stable']
            shock = next(s for s in pilot['shocks'] if s['delta'] == config['primary_delta'])
            if shock['status'] == 'completed':
                for name in ('recovery_time', 'right_censored', 'collapse'):
                    row[name] = shock[name]
                row['moderate'] = bool(row['joint_stable'] and not shock['right_censored'] and not shock['collapse']
                                       and config['recovery_range'][0] <= shock['recovery_time'] <= config['recovery_range'][1])
        rows.append(row)
    points = []
    for point in config['point_priority']:
        selected = [r for r in rows if r['point'] == point]
        if len(selected) != config['replicates']:
            raise ValueError('Manifest does not match the fixed sample denominator')
        safe = all(r['right_censored'] is False and r['collapse'] is False for r in selected if r['joint_stable'])
        enough = sum(r['moderate'] for r in selected) >= config['minimum_moderate']
        points.append({'point': point, 'planned': len(selected), 'public_stable': sum(r['public_stable'] for r in selected),
                       'joint_stable': sum(r['joint_stable'] for r in selected), 'moderate': sum(r['moderate'] for r in selected),
                       'passed': bool(safe and enough and all(r['status'] == 'completed' for r in selected))})
    passed = [p['point'] for p in points if p['passed']]
    verdict = {'protocol_version': config['protocol_version'], 'points': points,
               'recommended_primary_point': passed[0] if passed else None, 'other_passed_points': passed[1:],
               'confirmatory_locked': False, 'results_sha256': sha256(raw).hexdigest(),
               'config_sha256': sha256((root/'validation-config.json').read_bytes()).hexdigest(),
               'evaluator_sha256': expected_hash}
    output = root/'validation'
    output.mkdir(exist_ok=False)
    with (output/'runs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/'verdict.json').write_text(json.dumps(verdict, indent=2)+'\n')
    return verdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    print(json.dumps(evaluate(parser.parse_args().campaign), indent=2))
