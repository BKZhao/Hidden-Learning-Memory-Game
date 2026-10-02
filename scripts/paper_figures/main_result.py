"""Figure 3: absolute paths, baseline decomposition, and network inference."""

import csv
import numpy as np
from style import ROOT, COLORS, LABELS, STYLES, panel, plt, read_json, save


def main():
    source = 'artifacts/e2-20261001/decomposition/network-curves.npz'
    q = np.load(ROOT / source)['curves'][:, :, :, 0]
    mean = q.mean(axis=0)
    summary_path = 'artifacts/e2-20261001/analysis/summary.json'
    summary = read_json(summary_path)
    times = np.unique(np.r_[np.arange(1, 201), np.geomspace(201, 5000, 650).astype(int)])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), layout='constrained')
    for z, ax in enumerate(axes[0]):
        for i in range(3):
            ax.plot(times, mean[i, z, times], color=COLORS[i], ls=STYLES[i], label=LABELS[i])
        ax.set(xscale='log', ylim=(.64, 1.005), xlabel='Time after branching (MCS)',
               ylabel='Contribution rate $q$')
        ax.text(.97, .07, ['Unshocked', r'Shocked ($\delta=0.30$)'][z],
                transform=ax.transAxes, ha='right')
    axes[0, 0].legend(loc='lower left')
    ax = axes[1, 0]
    for z, color, line, label in [(0, '#d55e00', '-', 'Unshocked'), (1, '#0072b2', '--', 'Shocked')]:
        ax.plot(times, 100*(mean[0, z, times]-mean[2, z, times]),
                color=color, ls=line, label=label)
    ax.axhline(0, color='.6', lw=.7)
    ax.set(xscale='log', xlabel='Time after branching (MCS)', ylabel='Near minus far $q$ (pp)')
    ax.legend(loc='upper right')
    ax = axes[1, 1]
    effects = (q[:, 0, 0, 1:]-q[:, 0, 1, 1:]-q[:, 2, 0, 1:]+q[:, 2, 1, 1:]).mean(axis=-1)
    assert np.isclose(effects.mean(), summary['theta'], rtol=0, atol=1e-14)
    ax.scatter(100*effects, np.arange(len(effects)), s=16, color='#0072b2')
    lo, hi = np.array(summary['network_bootstrap_95_ci'])*100
    theta = summary['theta']*100
    ax.errorbar(theta, 22, xerr=[[theta-lo], [hi-theta]], fmt='D', color='black', capsize=3)
    ax.axvline(0, color='.55', ls=':', lw=.9)
    ax.axvline(summary['minimum_effect']*100, color='#d55e00', ls='--', lw=.9)
    ax.set(xlabel=r'$\theta=L_{0.5}-L_2$ (pp)', ylabel='Independent network',
           yticks=[0, 5, 10, 15, 22], yticklabels=['0', '5', '10', '15', 'Mean'], xlim=(-.12, 2.15))
    ax.text(1.93, 3, 'Planning threshold', rotation=90, ha='right', fontsize=8)
    for ax, letter in zip(axes.flat, 'abcd'):
        panel(ax, letter)
    save(fig, 'fig03-controlled', [source, summary_path],
         'Networks equally weighted after future/history averaging; curves descriptive, CI is network bootstrap; pp=percentage points.')
    output = ROOT / 'artifacts/paper-draft/tables'
    output.mkdir(parents=True, exist_ok=True)
    with (output/'controlled-arm-means.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['lambda', 'unshocked_q', 'shocked_q', 'signed_loss'])
        for i, scale in enumerate((.5, 1, 2)):
            a, b = mean[i, :, 1:].mean(axis=-1)
            writer.writerow([scale, a, b, a-b])


if __name__ == '__main__':
    main()
