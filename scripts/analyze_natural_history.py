"""Audit complete E4 histories and report support failures without imputing effects."""

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hidden_memory.checkpoint import load_checkpoint
from hidden_memory.experiment_io import file_hash, write_json
from hidden_memory.metrics import snapshot


def analyze(history: Path, response: Path) -> dict:
    config = json.loads((history/'config.json').read_text())
    records = [json.loads(s) for s in (history/'results.jsonl').read_text().splitlines()]
    outcomes = [json.loads(s) for s in (response/'results.jsonl').read_text().splitlines()]
    response_config = json.loads((response/'config.json').read_text())
    expected = response_config['networks']
    if sorted(r['network'] for r in records) != config['networks']:
        raise ValueError('Missing or duplicate input-audit network')
    excluded = [r['network'] for r in records if r['network'] not in expected]
    if excluded and 'response_gate' not in response_config:
        raise ValueError('Response exclusions require a prior input-only gate')
    if 'response_gate' in response_config:
        gate_path = Path(response_config['response_gate'])
        if file_hash(gate_path) != response_config['response_gate_sha256']:
            raise ValueError('Response gate changed')
        gate = json.loads(gate_path.read_text())
        if gate['status'] != 'passed' or gate['approved_networks'] != expected:
            raise ValueError('Response set differs from prior gate')
    records = [r for r in records if r['network'] in expected]
    for root in (history, response):
        if json.loads((root/'status.json').read_text())['state'] != 'completed': raise ValueError('Incomplete E4 stage')
    for rows in (records, outcomes):
        if sorted(r['network'] for r in rows) != expected: raise ValueError('Missing or duplicate network')
    if len({r['network_hash'] for r in records}) != len(records): raise ValueError('Repeated graphs')
    output = response/'analysis'; output.mkdir(exist_ok=False)
    shutil.copyfile(__file__, output/Path(__file__).name)
    rows = []; strata_rows = []; curves = {}; low_losses = {}; transplant_theta = {}; raw_theta = {}; arm_means = {}; transplant_curves = {}
    for record in sorted(records, key=lambda r:r['network']):
        network = record['network']
        result = next(r for r in outcomes if r['network'] == network)
        common_ids = None
        for audit in record['washouts']:
            wash = audit['wash_mcs']; folder = history/f'network-{network:02d}/wash-{wash}'
            outcome = next(r for r in result['washouts'] if r['wash_mcs'] == wash)
            row = {'network': network, 'wash_mcs': wash, 'individual_rows': audit['support']['matched'],
                   'common_rows': audit['common_rows'], 'eligible': audit['eligible']}
            with np.load(folder/'common-mapping.npz') as mapping:
                ids = np.sort(mapping['rows'])
                if common_ids is not None and not np.array_equal(ids, common_ids): raise ValueError('Changing support across washout')
                common_ids = ids
            for key in ('low', 'high'):
                state, params, _, meta = load_checkpoint(folder/f'{key}.npz')
                actual = snapshot(state, params)
                if params.alpha != .02 or meta['network_hash'] != record['network_hash']: raise ValueError('Incorrect washout parameters or graph')
                if state.event != (100000+config.get('cost_phase_mcs', 0)+wash)*1000: raise ValueError('Incorrect history duration')
                for metric in ('q', 'fP', 'welfare'):
                    if actual[metric] != audit['raw_history'][key]['snapshot'][metric]: raise ValueError('History metric mismatch')
                    row[f'{key}_{metric}'] = actual[metric]
                row[f'{key}_shock_legal'] = outcome['groups']['raw'][key]['shock_legal']
            for s in audit['support']['strata']:
                strata_rows.append({'network': network, 'wash_mcs': wash, 'stratum': s['stratum'],
                                    **s['pool_sizes'], 'joint_legal': s['joint_legal'], 'reason': s['reason']})
            with np.load(response/f'network-{network:02d}/wash-{wash}/trajectories.npz') as data:
                for key in data.files:
                    if data[key].shape != (5001,) or not np.isfinite(data[key]).all(): raise ValueError('Invalid raw response')
                for key in data.files:
                    if key.endswith(('_q', '_fP', '_welfare')):
                        arm_means.setdefault((wash, key), []).append(float(data[key][1:].mean()))
                    if key.startswith('transplant_') and key.endswith('_q'):
                        transplant_curves.setdefault((wash, key), []).append(data[key])
                for group, details in outcome['groups'].items():
                    for name in (('low', 'high', 'sham') if group == 'transplant' else ('low', 'high')):
                        if not details[name]['shock_legal']:
                            if details[name]['loss'] is not None: raise ValueError('Illegal shock has fabricated loss')
                            continue
                        loss = float(np.mean(data[f'{group}_{name}_z0_q'][1:]-data[f'{group}_{name}_z1_q'][1:]))
                        if not np.isclose(loss, details[name]['loss'], atol=1e-14, rtol=0): raise ValueError('Raw loss mismatch')
                    if details['theta_low_minus_high'] is not None:
                        target = transplant_theta if group == 'transplant' else raw_theta
                        theta = float(np.mean(data[f'{group}_low_z0_q'][1:]-data[f'{group}_low_z1_q'][1:])
                                      - np.mean(data[f'{group}_high_z0_q'][1:]-data[f'{group}_high_z1_q'][1:]))
                        if not np.isclose(theta, details['theta_low_minus_high'], atol=1e-14, rtol=0):
                            raise ValueError('Raw theta mismatch')
                        target.setdefault(wash, []).append((network, theta))
                low_losses.setdefault(wash, []).append(outcome['groups']['raw']['low']['loss'])
                for name, z in [('low', 0), ('low', 1), ('high', 0)]:
                    key = f'raw_{name}_z{z}_q'
                    if key in data: curves.setdefault((wash, name, z), []).append(data[key])
            if outcome['frozen_passed']:
                with np.load(response/f'network-{network:02d}/wash-{wash}/frozen.npz') as frozen:
                    for key in frozen.files:
                        suffix = key.split('_z', 1)[1]
                        if not np.array_equal(frozen[key], frozen[f'sham_z{suffix}']):
                            raise ValueError('Frozen transplant mismatch')
            rows.append(row)
    for name, values in [('histories.csv', rows), ('strata.csv', strata_rows)]:
        with (output/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    rng = np.random.default_rng(config['bootstrap_seed'])
    index = rng.integers(0, len(records), size=(10000, len(records)))
    summaries = []
    for wash in config['wash_mcs']:
        selected = [r for r in rows if r['wash_mcs'] == wash]
        entry = {'wash_mcs': wash, 'networks': len(selected), 'transplant_eligible_networks': sum(r['eligible'] for r in selected),
                 'raw_high_shock_legal_networks': sum(r['high_shock_legal'] for r in selected),
                 'common_rows_min_max': [min(r['common_rows'] for r in selected), max(r['common_rows'] for r in selected)]}
        for key in ('low_q', 'high_q', 'low_fP', 'high_fP'):
            values = np.array([r[key] for r in selected])
            entry[key] = {'mean': float(values.mean()), 'range': [float(values.min()), float(values.max())],
                          'ci95_network': np.quantile(values[index].mean(axis=1), [.025, .975]).tolist()}
        for label, effects in [('transplant_theta', transplant_theta), ('raw_history_theta', raw_theta)]:
            values = effects.get(wash, [])
            entry[label] = {'status': 'not_identifiable' if not values else 'descriptive', 'networks': len(values),
                            'mean': float(np.mean([v for _, v in values])) if values else None}
            if len(values) >= 3:
                v = np.array([value for _, value in values])
                idx = np.random.default_rng(config['bootstrap_seed']).integers(0, len(v), size=(10000, len(v)))
                entry[label]['ci95_network'] = np.quantile(v[idx].mean(axis=1), [.025, .975]).tolist()
                entry[label]['network_ids'] = [network for network, _ in values]
        entry['response_arm_means'] = {key: float(np.mean(v)) for (w, key), v in arm_means.items() if w == wash}
        entry['low_history_mean_loss'] = float(np.mean(low_losses[wash]))
        summaries.append(entry)
    paired_changes = []
    for label, effects in [('transplant', transplant_theta), ('raw', raw_theta)]:
        base = dict(effects.get(0, []))
        for wash in config['wash_mcs'][1:]:
            current = dict(effects.get(wash, [])); ids = sorted(set(base)&set(current))
            values = np.array([current[h]-base[h] for h in ids])
            item = {'group': label, 'wash_mcs': wash, 'network_ids': ids, 'mean_change_from_zero': float(values.mean()) if len(values) else None}
            if len(values) >= 3:
                idx = np.random.default_rng(config['bootstrap_seed']).integers(0, len(values), size=(10000, len(values)))
                item['ci95_network'] = np.quantile(values[idx].mean(axis=1), [.025, .975]).tolist()
            paired_changes.append(item)
    report = {'excluded_by_input_gate': sorted(excluded), 'paired_wash_changes': paired_changes, 'scope': 'Exploratory natural history; nonoverlap is not a zero effect', 'washouts': summaries,
              'bootstrap_seed': config['bootstrap_seed'], 'bootstrap_draws': 10000,
              'history_seconds': json.loads((history/'status.json').read_text())['elapsed_seconds'],
              'response_seconds': json.loads((response/'status.json').read_text())['elapsed_seconds'],
              'response_micro_updates': sum(r['micro_updates'] for r in outcomes),
              'history_results_sha256': file_hash(history/'results.jsonl'), 'response_results_sha256': file_hash(response/'results.jsonl')}
    write_json(output/'summary.json', report)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    washes = config['wash_mcs']
    for metric, ax in [('q', axes[0,0]), ('fP', axes[0,1])]:
        for source, color in [('low', 'tab:blue'), ('high', 'tab:orange')]:
            values = [r[f'{source}_{metric}']['mean'] for r in summaries]
            ax.plot(washes, values, marker='o', color=color, label=f'{source.capitalize()}-cost history')
        ax.set(xscale='symlog', xlabel='Common-parameter washout (MCS)', ylabel=metric,
               title='Natural source contribution' if metric == 'q' else 'Natural source punisher fraction')
        ax.legend(fontsize=8)
    axes[0,0].set_ylim(0, 1.03)
    for wash in washes:
        axes[1,0].plot(np.arange(1, 5001), np.mean(curves[(wash, 'low', 1)], axis=0)[1:], label=f'Washout {wash}')
        axes[1,1].plot(np.arange(1, 5001), np.mean(curves[(wash, 'high', 0)], axis=0)[1:], label=f'Washout {wash}', lw=.8)
    for ax, title in [(axes[1,0], 'Low-cost history: shocked'), (axes[1,1], 'High-cost history: unshocked')]:
        ax.set(xscale='log', xlabel='Response time (MCS)', ylabel='Contribution q', title=title)
        ax.legend(fontsize=8)
    for suffix in ('png', 'pdf', 'svg'): fig.savefig(output/f'natural-history.{suffix}', dpi=180)
    plt.close(fig)
    if transplant_curves:
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), layout='constrained')
        for ax, wash in zip(axes[:3], washes):
            for source, color in [('low', 'tab:blue'), ('high', 'tab:orange')]:
                for z, style in [(0, '-'), (1, '--')]:
                    key = (wash, f'transplant_{source}_z{z}_q')
                    if key in transplant_curves:
                        ax.plot(np.arange(1, 5001), np.mean(transplant_curves[key], axis=0)[1:],
                                color=color, ls=style, label=f'{source}, shock={z}')
            ax.set(xscale='log', xlabel='Response time (MCS)', ylabel='Contribution q', title=f'Natural gap: washout {wash}')
        axes[0].legend(fontsize=8)
        for j, entry in enumerate(summaries):
            effect = entry['transplant_theta']
            if 'ci95_network' in effect:
                m = effect['mean']; lo, hi = effect['ci95_network']
                axes[3].errorbar(100*m, j, xerr=np.array([[m-lo], [hi-m]])*100, fmt='o', capsize=4)
        axes[3].axvline(0, color='gray', ls=':')
        axes[3].set(yticks=range(len(washes)), yticklabels=[str(w) for w in washes],
                    ylabel='Washout (MCS)', xlabel='L_low minus L_high (pp)', title='Network mean and 95% CI')
        for suffix in ('png', 'pdf', 'svg'): fig.savefig(output/f'transplant-effects.{suffix}', dpi=180)
        plt.close(fig)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--history', type=Path, required=True); p.add_argument('--response', type=Path, required=True)
    args = p.parse_args(); print(json.dumps(analyze(args.history, args.response), indent=2))
