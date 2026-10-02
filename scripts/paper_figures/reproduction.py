"""Figure 2: all three completed reference-model reproduction conditions."""

import numpy as np
from style import ROOT, panel, plt, save


def main():
    source = 'artifacts/e0-20261001/figures/plotted-data.npz'
    data = np.load(ROOT / source)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=True, layout='constrained')
    for i, ax in enumerate(axes):
        mean, sd = data[f'point_{i}_mean'], data[f'point_{i}_sd']
        for j, (color, line, label) in enumerate(zip(
                ['#d55e00', '#0072b2', '#009e73'], ['-', '--', '-.'], 'CDP')):
            ax.plot(data['mcs'], mean[:, j], color=color, ls=line, label=label)
            ax.fill_between(data['mcs'], mean[:, j]-sd[:, j], mean[:, j]+sd[:, j],
                            color=color, alpha=.12, lw=0)
        ax.set(xscale='log', ylim=(0, 1), xlabel='Time (MCS)')
        ax.text(.95, .94, f'$r={1+i*.1:.1f}$', transform=ax.transAxes, ha='right')
        panel(ax, chr(97+i))
    axes[0].set_ylabel('Strategy frequency')
    axes[0].legend(loc='center left')
    save(fig, 'fig02-reproduction', [source, 'artifacts/e0-20261001/manifest.json'],
         'Existing log-spaced samples; 20 runs per panel; bands are run SD, not CI.')


if __name__ == '__main__':
    main()
