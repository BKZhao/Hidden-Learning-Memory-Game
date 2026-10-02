"""Figure 5: failed long history support and estimable short-history contrasts."""

import numpy as np
from style import ROOT, panel, plt, read_json, save


def main():
    paths = [f'artifacts/e4b-paired-cost-history-20261001/network-{i:02d}/cost-phase.npz' for i in range(10)]
    samples = [np.load(ROOT/p) for p in paths]
    source = 'artifacts/e4c-validation-response-20261001/analysis/summary.json'
    gate = 'artifacts/e4c-validation-history-20261001/response-gate.json'
    summary, eligibility = read_json(source), read_json(gate)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), layout='constrained')
    ax = axes[0, 0]
    for key, color, line, label in [('low_q', '#0072b2', '-', r'$\alpha=0.02$'), ('high_q', '#d55e00', '--', r'$\alpha=0.30$')]:
        mean = np.array([d[key] for d in samples]).mean(axis=0)
        times = np.arange(1, len(mean)+1)
        ax.plot(times, mean, color=color, ls=line, label=label)
    ax.set(xscale='log', xlabel='Cost-phase time (MCS)', ylabel='Contribution rate $q$', ylim=(0, 1.03))
    ax.legend(loc='center left')
    ax = axes[0, 1]
    rows = eligibility['networks']
    ax.scatter([r['network'] for r in rows], [r['common_rows'] for r in rows], color='#0072b2', s=22)
    ax.axhline(600, color='.5', ls='--', lw=1, label='Input gate (600)')
    ax.set(xlabel='Validation network', ylabel='Common eligible rows', ylim=(0, 1700), xticks=[10, 13, 16, 19])
    ax.legend(loc='lower left')
    for ax, key, label in [(axes[1, 0], 'transplant_theta', 'Transplanted gaps'), (axes[1, 1], 'raw_history_theta', 'Raw source histories')]:
        for j, wash in enumerate(summary['washouts']):
            estimate = wash[key]
            v = estimate['mean']*100
            lo, hi = np.array(estimate['ci95_network'])*100
            ax.errorbar(v, j, xerr=[[v-lo], [hi-v]], color='#0072b2', fmt='o', capsize=3)
        ax.axvline(0, color='.55', lw=.8, ls=':')
        ax.set(yticks=range(3), yticklabels=[str(w['wash_mcs']) for w in summary['washouts']],
               xlabel=r'$L_{\mathrm{low}}-L_{\mathrm{high}}$ (pp)', ylabel='Washout time (MCS)',
               xlim=(-.22, .34), ylim=(2.8, -.55))
        ax.text(.96, .05, label, transform=ax.transAxes, ha='right', fontsize=8)
    for ax, letter in zip(axes.flat, 'abcd'):
        panel(ax, letter)
    save(fig, 'fig05-natural-history', paths + [source, gate],
         'Panel a: 10000-MCS cost phase on networks 0-9. Panels b-d: separate 10-MCS pulse design on networks 10-19, not the same exposure; exploratory network bootstrap CIs.')


if __name__ == '__main__':
    main()
