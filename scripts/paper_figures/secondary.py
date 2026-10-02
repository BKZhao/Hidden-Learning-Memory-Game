"""Appendix figure: descriptive punishment and welfare paths in all E2 arms."""

import numpy as np
from style import ROOT, COLORS, LABELS, STYLES, panel, plt, save


def main():
    source = 'artifacts/e2-20261001/decomposition/network-curves.npz'
    mean = np.load(ROOT / source)['curves'].mean(axis=0)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.9), layout='constrained')
    t = np.unique(np.r_[np.arange(1, 201), np.geomspace(201, 5000, 650).astype(int)])
    for m, label, row in [(1, 'Punisher fraction', 0), (2, 'Snapshot welfare', 1)]:
        for z in (0, 1):
            ax = axes[row, z]
            for i in range(3):
                ax.plot(t, mean[i, z, m, t], color=COLORS[i], ls=STYLES[i], label=LABELS[i])
            ax.set(xscale='log', xlabel='Time after branching (MCS)', ylabel=label)
            ax.text(.97, .05, ['Unshocked', 'Shocked'][z], transform=ax.transAxes, ha='right')
    axes[0, 0].legend(loc='lower left', fontsize=7)
    for ax, letter in zip(axes.flat, 'abcd'):
        panel(ax, letter)
    save(fig, 'figA1-secondary', [source], 'Descriptive network means, no temporal smoothing; snapshot welfare is per group membership.')


if __name__ == '__main__':
    main()
