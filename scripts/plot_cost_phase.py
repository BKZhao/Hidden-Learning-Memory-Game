"""Describe the prespecified paired cost phase without selecting new experiments."""

import argparse
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hidden_memory.experiment_io import write_json


def plot(root: Path) -> None:
    output = root/'analysis'; output.mkdir(exist_ok=False)
    shutil.copyfile(__file__, output/Path(__file__).name)
    values = {}
    for source in ('low', 'high'):
        for metric in ('q', 'fP'):
            values[(source, metric)] = np.array([np.load(root/f'network-{h:02d}/cost-phase.npz')[f'{source}_{metric}'] for h in range(10)])
    high = values[('high', 'q')]
    summary = {'scope': 'Descriptive crossings of the fixed 10000 MCS cost phase; not stopping rules',
               'high_mean_curve_first_below_half_mcs': int(np.flatnonzero(high.mean(axis=0)<.5)[0])+1,
               'high_mean_curve_first_below_tenth_mcs': int(np.flatnonzero(high.mean(axis=0)<.1)[0])+1,
               'high_network_first_below_half_mcs': [int(np.flatnonzero(row<.5)[0])+1 for row in high],
               'high_tail_1000_mean_q': float(high[:,-1000:].mean()),
               'low_tail_1000_mean_q': float(values[('low', 'q')][:,-1000:].mean())}
    write_json(output/'cost-phase-summary.json', summary)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained')
    for metric, ax in zip(('q', 'fP'), axes):
        for source, color, label in [('low', 'tab:blue', 'Cost alpha=0.02'), ('high', 'tab:orange', 'Cost alpha=0.30')]:
            data = values[(source, metric)]; x = np.arange(1, data.shape[1]+1)
            ax.plot(x, data.mean(axis=0), color=color, label=label)
            ax.fill_between(x, data.min(axis=0), data.max(axis=0), color=color, alpha=.15)
        ax.set(xscale='log', xlabel='Cost phase (MCS)', ylabel='Contribution q' if metric == 'q' else 'Punisher fraction', title='Same initial state and shared events')
        ax.legend(fontsize=8)
    for suffix in ('png', 'pdf', 'svg'): fig.savefig(output/f'cost-phase.{suffix}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('root', type=Path); args = p.parse_args(); plot(args.root)
