"""Network-level variance planning; never changes the previously locked work point."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from hidden_memory.experiment_io import file_hash


def summarize(root: Path) -> dict:
    status = json.loads((root/'status.json').read_text())
    if status['state'] != 'completed' or status['failed']:
        raise ValueError('Require a complete error-free variance pilot')
    config = json.loads((root/'config.json').read_text())
    records = sorted([json.loads(s) for s in (root/'results.jsonl').read_text().splitlines()], key=lambda r: r['replicate'])
    if len({r['network_hash'] for r in records}) != len(records):
        raise ValueError('Networks are not independent units')
    differences = np.array([np.mean([x['theta'] for x in r['outcomes']]) for r in records])
    sd = float(differences.std(ddof=1))
    factor = (1.96+.84)/config['minimum_effect']
    estimate = max(config['minimum_confirmatory_networks'], int(np.ceil((factor*sd)**2)))
    rng = np.random.default_rng(config['bootstrap_seed'])
    samples = rng.choice(differences, size=(config['bootstrap_replicates'], len(differences)), replace=True)
    bootstrap_sd = samples.std(axis=1, ddof=1)
    plans = np.maximum(config['minimum_confirmatory_networks'], np.ceil((factor*bootstrap_sd)**2))
    if any(r['outcomes'][0]['full_horizon_frozen_theta'] != 0 for r in records):
        raise AssertionError('Frozen control failed')
    result = {'networks': len(records), 'histories_per_network': 1, 'futures_per_checkpoint': config['future_replicates'],
              'network_sd': sd, 'exploratory_mean_theta': float(differences.mean()),
              'minimum_effect': config['minimum_effect'], 'planned_confirmatory_networks': estimate,
              'within_budget': estimate <= config['maximum_confirmatory_networks'],
              'bootstrap_sd_2_5_50_97_5': np.quantile(bootstrap_sd, [.025,.5,.975]).tolist(),
              'bootstrap_planned_networks_2_5_50_97_5': np.quantile(plans, [.025,.5,.975]).tolist(),
              'all_full_horizon_frozen_controls_passed': True,
              'limitations': 'B=1/K=3 pilot cannot separate network/history variance or guarantee exact power for B=2/K=10; pilot networks excluded from E2.',
              'results_sha256': file_hash(root/'results.jsonl'), 'script_sha256': file_hash(Path(__file__))}
    output = root/'analysis'
    output.mkdir(exist_ok=False)
    with (output/'network-differences.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['replicate', 'network_hash', 'mean_theta'])
        writer.writerows((r['replicate'], r['network_hash'], d) for r, d in zip(records, differences))
    (output/'planning.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    print(json.dumps(summarize(parser.parse_args().campaign), indent=2))
