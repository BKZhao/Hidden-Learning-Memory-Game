"""Figure 4: actual local Q crossing, all replay onsets, matched controls."""

import numpy as np
from style import panel, plt, read_json, save


def main():
    source = 'artifacts/e3-divergence-20261001-v2/summary.json'
    matched = 'artifacts/e3-matched-20261001/analysis/summary.json'
    replay = read_json(source)
    sample = replay['results'][0]['arms']['0']['first_policy']['records']
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), layout='constrained')
    ax = axes[0]
    for key, color, label in [('0.5', '#009e73', 'Near'), ('2.0', '#d55e00', 'Far')]:
        row = sample[key]
        for action, line in [(0, '-'), (2, '--')]:
            ax.plot([0, 1], [row['q_row_before'][action], row['q_row_after'][action]],
                    color=color, ls=line, marker='o' if key == '0.5' else 'x',
                    label=label + ' $Q_' + 'CDP'[action] + '$')
    ax.axhline(sample['0.5']['q_row_before'][1], color='.6', ls=':', label='$Q_D$ (both)')
    ax.set(xticks=[0, 1], xticklabels=['Before', 'After'], ylabel='Action value', ylim=(7.92, 8.72))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=5, fontsize=7)
    ax = axes[1]
    labels = []
    for j, (result, z) in enumerate((r, z) for r in replay['results'] for z in ('0', '1')):
        branch = result['arms'][z]
        p = branch['first_policy']['response_event_zero_based']
        a = branch['first_action']['response_event_zero_based']
        ax.plot([p, a], [j, j], color='.75', lw=.8)
        ax.scatter(p, j, color='#009e73', marker='o', s=20, label='Policy' if j == 0 else None)
        ax.scatter(a, j, color='#0072b2', marker='x', s=24, label='Action' if j == 0 else None)
        labels.append(f"{result['network']} / {z}")
    ax.set(yticks=range(8), yticklabels=labels, xlabel='Event index (zero based)', ylabel='Network / shock indicator')
    ax.legend(loc='upper right')
    ax.invert_yaxis()
    ax = axes[2]
    estimates = read_json(matched)['by_control_action']['1']['estimates']
    for j, key in enumerate(('P_theta', 'control_theta', 'difference')):
        d = estimates[key]
        v = d['mean']*100
        lo, hi = np.array(d['ci95'])*100
        ax.errorbar(v, j, xerr=[[v-lo], [hi-v]], fmt='o', color=['#009e73', '#0072b2', 'black'][j], capsize=3)
    ax.axvline(0, color='.6', lw=.8, ls=':')
    ax.set(yticks=range(3), yticklabels=['P subset', 'D subset', 'P minus D'],
           xlabel='Paired contrast (pp)', ylim=(2.7, -.7))
    ax.text(.04, .03, 'C: insufficient matches', transform=ax.transAxes, fontsize=7)
    for ax, letter in zip(axes, 'abc'):
        panel(ax, letter)
    save(fig, 'fig04-mechanism', [source, matched],
         'Eight prespecified prefixes, not a population estimate; actual before/after values for both arms; matched CI is network bootstrap.')


if __name__ == '__main__':
    main()
