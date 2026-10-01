"""Recompute paired effects from raw files and bootstrap whole independent networks."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from hidden_memory.experiment_io import file_hash


def analyze(root: Path) -> dict:
    config = json.loads((root/'config.json').read_text())
    if file_hash(Path(__file__)) != config['script_sha256'][Path(__file__).name]:
        raise ValueError('Analysis script differs from the preregistered version')
    status = json.loads((root/'status.json').read_text())
    if status['state'] != 'completed' or status['failed']:
        raise ValueError('Scientific analysis requires an error-free complete batch')
    records = [json.loads(s) for s in (root/'results.jsonl').read_text().splitlines()]
    if len(records) != config['independent_networks']*config['training_histories']:
        raise ValueError('Missing planned history records')
    expected = {(h,b) for h in range(config['independent_networks']) for b in range(config['training_histories'])}
    identities = [(r['network'],r['history']) for r in records]
    if len(set(identities)) != len(identities) or set(identities) != expected:
        raise ValueError('Duplicate or missing network/history identities')
    grouped, hashes, history_rows, all_future_rows = {}, {}, [], []
    for record in records:
        h, b = record['network'], record['history']
        digest = record['training']['network_hash']
        if h in hashes and hashes[h] != digest:
            raise ValueError('A network has inconsistent graph hashes')
        hashes[h] = digest
        if record['status'] == 'training_not_eligible':
            history_rows.append({'network': h, 'history': b, 'selected_mcs': None, 'mean_theta': None})
            continue
        response = record['response']
        future_ids = [r['future'] for r in response['outcomes']]
        if sorted(future_ids) != list(range(config['future_replicates'])):
            raise ValueError('Duplicate or missing future identities')
        folder = root/'histories'/f'network-{h:02d}-history-{b}'/'response'/f"network-{response['replicate']:02d}"
        values = []
        for outcome in response['outcomes']:
            with np.load(folder/f"future-{outcome['future']:02d}.npz") as data:
                if any(data[f'lambda_{scale}_shock_{z}_q'].shape != (config['response_mcs']+1,)
                       for scale in (.5,1.,2.) for z in (0,1)):
                    raise ValueError('Raw trajectory does not cover the fixed horizon')
                if any(not np.isfinite(data[key]).all() for key in data.files):
                    raise ValueError('Nonfinite raw response trajectory')
                losses = {scale: float((data[f'lambda_{scale}_shock_0_q'][1:]-data[f'lambda_{scale}_shock_1_q'][1:]).mean()) for scale in (.5,1.,2.)}
            theta = losses[.5]-losses[2.]
            if not np.isclose(theta, outcome['theta'], rtol=0, atol=1e-15):
                raise ValueError('Raw paired trajectories disagree with recorded theta')
            values.append(theta)
            all_future_rows.append({'network': h, 'history': b, 'future': outcome['future'],
                                    'loss_near': losses[.5], 'loss_sham': losses[1.], 'loss_far': losses[2.], 'theta': theta})
        if len(values) != config['future_replicates']:
            raise ValueError('Missing paired future repetition')
        with np.load(folder/'frozen-control.npz') as data:
            for key in data.files:
                if data[key].shape != (config['response_mcs']+1,) or not np.isfinite(data[key]).all():
                    raise ValueError('Invalid full-horizon frozen trajectory')
                if key.startswith('lambda_1.0_'):
                    for scale in (.5,2.):
                        if not np.array_equal(data[key], data[key.replace('lambda_1.0_', f'lambda_{scale}_')]):
                            raise AssertionError('Full-horizon frozen control failed')
        mean_theta = float(np.mean(values))
        grouped.setdefault(h, []).append(mean_theta)
        history_rows.append({'network': h, 'history': b, 'selected_mcs': record['training']['selection']['selected_mcs'], 'mean_theta': mean_theta})
    if len(set(hashes.values())) != len(hashes) or len(grouped) < 2:
        raise ValueError('Insufficient distinct eligible network clusters')
    network_rows = [{'network': h, 'eligible_histories': len(grouped[h]), 'mean_theta': float(np.mean(grouped[h]))} for h in sorted(grouped)]
    values = np.array([r['mean_theta'] for r in network_rows])
    rng = np.random.default_rng(config['analysis_seed'])
    bootstrap = rng.choice(values, size=(config['bootstrap_replicates'],len(values)), replace=True).mean(axis=1)
    interval = np.quantile(bootstrap, [.025,.975]).tolist()
    complete = [r['mean_theta'] for r in network_rows if r['eligible_histories'] == config['training_histories']]
    result = {'stage': 'E2', 'protocol_version': config['protocol_version'],
              'planned_networks': config['independent_networks'], 'effective_networks': len(values),
              'eligible_histories': len(history_rows)-status['training_not_eligible'],
              'training_not_eligible': status['training_not_eligible'], 'paired_futures': len(all_future_rows),
              'theta': float(values.mean()), 'network_sd': float(values.std(ddof=1)),
              'network_bootstrap_95_ci': interval, 'minimum_effect': config['minimum_effect'],
              'ci_excludes_zero': bool(interval[0] > 0 or interval[1] < 0),
              'ci_inside_practical_bounds': bool(interval[0] > -config['minimum_effect'] and interval[1] < config['minimum_effect']),
              'complete_history_networks': len(complete), 'complete_history_mean_theta': float(np.mean(complete)) if complete else None,
              'full_horizon_frozen_controls_passed': True,
              'results_sha256': file_hash(root/'results.jsonl'), 'config_sha256': file_hash(root/'config.json'),
              'analysis_sha256': file_hash(Path(__file__))}
    output = root/'analysis'
    output.mkdir(exist_ok=False)
    for name, rows in [('histories', history_rows), ('networks', network_rows), ('paired-futures', all_future_rows)]:
        with (output/f'{name}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.save(output/'network-bootstrap.npy', bootstrap)
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    print(json.dumps(analyze(parser.parse_args().campaign), indent=2))
