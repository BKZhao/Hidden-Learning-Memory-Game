"""Visualize every predeclared sham-validation replicate without selecting favorable runs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hidden_memory.experiment_io import file_hash


def plot(root: Path) -> None:
    config = json.loads((root/'validation-config.json').read_text())
    verdict = json.loads((root/'validation/verdict.json').read_text())
    with (root/'validation/runs.csv').open() as f:
        rows = list(csv.DictReader(f))
    output = root/'figures'
    output.mkdir(exist_ok=False)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), layout='constrained')
    saved = {}
    for col, point in enumerate(config['point_priority']):
        subset = [r for r in rows if int(r['point']) == point]
        q_differences = []
        for row in subset:
            folder = root/'jobs'/row['job_id']/'pilot'
            with np.load(folder/'delta-0.00.npz') as data:
                baseline = data['q'].copy()
            with np.load(folder/'delta-0.30.npz') as data:
                shocked = data['q'].copy()
            q_differences.append(baseline[:1001]-shocked[:1001])
        differences = np.asarray(q_differences)
        saved[f'point_{point}_deficits'] = differences
        t = np.arange(1001)
        axes[0,col].plot(t, differences.T, color='#0072b2', lw=.7, alpha=.2)
        axes[0,col].plot(t, np.median(differences, axis=0), color='black', lw=1.2, label='Median')
        axes[0,col].axhline(0, color='gray', ls=':', lw=.6)
        axes[0,col].set(xlabel='Time after pulse (MCS)', ylabel='Paired q deficit', title=f'Point {point:02d}: delta = 0.30')
        axes[0,col].legend(frameon=False)
        for index, row in enumerate(subset):
            valid = row['joint_stable'] == 'True'
            axes[1,col].scatter(index, float(row['recovery_time']), color='#009e73' if valid else '#d55e00',
                                marker='o' if valid else 'x', s=28)
        axes[1,col].axhline(config['recovery_range'][0], color='gray', ls='--', lw=.8, label='Lower moderate bound')
        passed = next(p for p in verdict['points'] if p['point']==point)
        axes[1,col].set(xlabel='Independent replicate', ylabel='Recovery onset (MCS)',
                        title=f"Joint stable + moderate: {passed['moderate']}/10", ylim=(0,450))
    fig.suptitle('Independent sham validation; orange crosses fail hidden-state screen', fontsize=11)
    for ext in ('png','pdf','svg'):
        fig.savefig(output/f'shock-validation.{ext}', dpi=220)
    plt.close(fig)
    np.savez_compressed(output/'plotted-data.npz', mcs=np.arange(1001), **saved)
    (output/'provenance.json').write_text(json.dumps({'script_sha256':file_hash(Path(__file__)),
        'validation_sha256':file_hash(root/'validation/verdict.json'), 'plot_note':'Raw deficit, no smoothing; recovery uses the fixed 50-MCS/200-MCS definition.'},indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    plot(parser.parse_args().campaign)
