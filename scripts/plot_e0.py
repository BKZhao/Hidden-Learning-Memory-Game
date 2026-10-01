"""Plot completed E0 trajectories and export final-window statistics."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root: Path) -> None:
    status = json.loads((root / 'status.json').read_text())
    if status['state'] != 'completed' or status['failed']:
        raise ValueError('E0 must finish successfully before final plotting')
    manifest = json.loads((root / 'manifest.json').read_text())
    output = root / 'figures'
    output.mkdir(exist_ok=False)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True, layout='constrained')
    rows, series = [], {}
    times = np.unique(np.geomspace(1, 500000, 1400).astype(int))
    for point, ax in enumerate(axes):
        jobs = [j for j in manifest['jobs'] if j['point'] == point]
        paths = [root / 'jobs' / f"E0-p{point:02d}-r{j['replicate']:02d}" for j in jobs]
        samples, tails = [], []
        for path in paths:
            columns = json.loads((path / 'trajectory_columns.json').read_text())
            data = np.load(path / 'trajectory.npy', mmap_mode='r')
            fc, fp = data[:, columns.index('fC')], data[:, columns.index('fP')]
            samples.append(np.column_stack([fc[times], 1-fc[times]-fp[times], fp[times]]))
            tails.append([fc[-5000:].mean(), (1-fc[-5000:]-fp[-5000:]).mean(), fp[-5000:].mean()])
        samples, tails = np.asarray(samples), np.asarray(tails)
        avg, sd = samples.mean(axis=0), samples.std(axis=0, ddof=1)
        series[f'point_{point}_mean'] = avg
        series[f'point_{point}_sd'] = sd
        for i, (name, color, style) in enumerate(zip('CDP', ['#d55e00', '#0072b2', '#009e73'], ['-', '--', '-.'])):
            ax.plot(times, avg[:, i], color=color, ls=style, lw=1.5, label=name)
            ax.fill_between(times, avg[:, i]-sd[:, i], avg[:, i]+sd[:, i], color=color, alpha=.14)
            rows.append({'r': jobs[0]['r'], 'strategy': name, 'n': len(jobs),
                         'mean': float(tails[:, i].mean()), 'sd': float(tails[:, i].std(ddof=1))})
        ax.set(xscale='log', ylim=(0, 1), xlabel='Time (MCS)', title=f"({chr(97+point)}) r = {jobs[0]['r']:.1f}")
        ax.tick_params(direction='in', top=True, right=True)
    axes[0].set_ylabel('Strategy frequency')
    axes[0].legend(frameon=False)
    fig.suptitle('E0: 20 independent runs per point; shading = between-run SD', fontsize=10)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'e0-timeseries.{ext}', dpi=220)
    plt.close(fig)
    np.savez_compressed(output / 'plotted-data.npz', mcs=times, **series)
    with (output / 'tail-statistics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {'script_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
                'manifest_sha256': sha256((root/'manifest.json').read_bytes()).hexdigest(),
                'results_sha256': sha256((root/'results.jsonl').read_bytes()).hexdigest(),
                'sampling': '1400 log-spaced integer MCS; no temporal smoothing',
                'bands': 'sample standard deviation across all 20 runs; not confidence intervals',
                'reference': 'Zou and Huang, Fig. 3(a-c), visual comparison only'}
    (output / 'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    plot(parser.parse_args().campaign)
