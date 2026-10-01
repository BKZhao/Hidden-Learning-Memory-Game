"""Read all completed pilot candidates; expose raw response and hidden-state drift."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hidden_memory.checkpoint import load_checkpoint
from hidden_memory.intervention import intervene


def diagnose(root: Path) -> None:
    status = json.loads((root / 'status.json').read_text())
    if status['state'] != 'completed' or status['failed']:
        raise ValueError('Campaign must complete without failures')
    raw = (root / 'results.jsonl').read_bytes()
    records = sorted([json.loads(s) for s in raw.splitlines()], key=lambda r: r['job_id'])
    jobs = json.loads((root / 'manifest.json').read_text())['jobs']
    duration, nodes = jobs[0]['training_mcs'], jobs[0]['nodes']
    output = root / 'diagnostics'
    output.mkdir(exist_ok=False)
    rows, deficits, drift = [], [], []
    for result in records:
        path = root / 'jobs' / result['job_id']
        history = json.loads((path / 'diagnostics.json').read_text())
        start = next(x for x in history if x['mcs'] == (15000 if duration == 20000 else 20000))
        end = history[-1]
        # Descriptive checkpoint differences; no new stability threshold is selected.
        drift.append({'job_id': result['job_id'], 'start_mcs': start['mcs'], 'end_mcs': end['mcs'],
                      'median_gap_change': end['hidden_state']['gap_quantiles'][2]-start['hidden_state']['gap_quantiles'][2],
                      'visited_fraction_change': end['hidden_state']['visited_fraction']-start['hidden_state']['visited_fraction'],
                      'p_greedy_rows_start': sum(start['hidden_state']['greedy_set_counts'][4:]),
                      'p_greedy_rows_end': sum(end['hidden_state']['greedy_set_counts'][4:])})
        if not result.get('pilot') or not result['pilot']['candidate']:
            continue
        with np.load(path/'pilot/delta-0.00.npz') as data:
            baseline = data['q'].copy()
        with np.load(path/'pilot/delta-0.10.npz') as data:
            shocked = data['q'].copy()
        state, params, _, _ = load_checkpoint(path/'checkpoints'/f'mcs-{duration}.npz')
        _, eligible = intervene(state, params)
        drows = state.q[:, 1, :]
        greedy = drows == drows.max(axis=-1, keepdims=True)
        d_probability = greedy[:, 1]/greedy.sum(axis=-1)
        below = np.flatnonzero(shocked[1:51] < result['tail_means']['q']-.05)+1
        deficits.append(baseline[:51]-shocked[:51])
        rows.append({'job_id': result['job_id'], 'r': result['r'], 'alpha': result['alpha'], 'beta': result['beta'],
                     'eligible_p_fraction': float(eligible.mean()),
                     'd_row_greedy_d_probability': float(d_probability.mean()),
                     'last_below_reference_in_first_50': int(below[-1]) if len(below) else 0,
                     'first_50_mean_paired_deficit': float((baseline[1:51]-shocked[1:51]).mean())})
    for name, data in [('candidates', rows), ('hidden-drift', drift)]:
        if data:
            with (output/f'{name}.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    summary = {'completed': len(records), 'candidate_count': len(rows), 'nodes': nodes, 'training_mcs': duration,
               'results_sha256': sha256(raw).hexdigest(), 'script_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
               'interpretation': 'Sham diagnostic only. Eligibility is computed without evolving treated branches. Hidden differences are descriptive, not a convergence test.'}
    if rows:
        keys = ['eligible_p_fraction', 'd_row_greedy_d_probability', 'first_50_mean_paired_deficit', 'last_below_reference_in_first_50']
        summary['quantiles_0_25_50_75_100'] = {key: np.quantile([r[key] for r in rows], [0,.25,.5,.75,1]).tolist() for key in keys}
        summary['last_below_reference_counts'] = {str(k): sum(r['last_below_reference_in_first_50']==k for r in rows) for k in sorted({r['last_below_reference_in_first_50'] for r in rows})}
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), layout='constrained')
        data = np.asarray(deficits)
        t = np.arange(21)
        axes[0].plot(t, data[:, :21].T, color='#0072b2', alpha=.12, lw=.6)
        axes[0].plot(t, np.median(data[:, :21], axis=0), color='black', lw=1.8, label='Median')
        axes[0].axhline(0, color='gray', ls=':', lw=.8)
        axes[0].set(xlabel='Time after shock (MCS)', ylabel='Paired q deficit', title='(a) Raw sham response')
        axes[0].legend(frameon=False)
        axes[1].hist([r['eligible_p_fraction'] for r in rows], bins=12, color='#009e73', edgecolor='white')
        axes[1].set(xlabel='Eligible P entries / all state rows', ylabel='Candidate count', title='(b) Intervention coverage')
        x = [r['median_gap_change'] for r in drift]
        y = [(r['p_greedy_rows_end']-r['p_greedy_rows_start'])/(3*nodes) for r in drift]
        axes[2].scatter(x, y, s=10, color='#d55e00', alpha=.65)
        axes[2].axhline(0, color='gray', ls=':', lw=.8)
        axes[2].axvline(0, color='gray', ls=':', lw=.8)
        axes[2].set(xlabel='Change in median P gap', ylabel='Change in P-greedy row fraction', title='(c) Hidden drift: all 108 runs')
        fig.suptitle(f'N = {nodes}; {len(rows)} stable candidates; delta = 0.10', fontsize=11)
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(output/f'pilot-diagnostics.{ext}', dpi=220)
        plt.close(fig)
        np.savez_compressed(output/'raw-deficits.npz', mcs=np.arange(51), deficits=data, job_ids=np.array([r['job_id'] for r in rows]))
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    diagnose(parser.parse_args().campaign)
