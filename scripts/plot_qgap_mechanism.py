"""Descriptive Q-gap threshold-crossing mechanism figure from the E3 divergence replay."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hidden_memory.experiment_io import write_json

NEAR, SHAM, FAR = '0.5', '1.0', '2.0'
STYLES = {NEAR: ('tab:orange', rf'$\lambda$=0.5 (near)'),
          SHAM: ('tab:gray', rf'$\lambda$=1 (sham)'),
          FAR: ('tab:blue', rf'$\lambda$=2 (far)')}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_policy_divergence(prefix: Path) -> dict:
    """First event where the updated row's greedy set differs between near and far."""
    with open(prefix) as handle:
        for line in handle:
            record = json.loads(line)
            near, far = record['records'][NEAR], record['records'][FAR]
            if set(near['trace']['greedy_after']) != set(far['trace']['greedy_after']):
                near_row = np.array(near['q_row_before'], dtype=float)
                far_row = np.array(far['q_row_before'], dtype=float)
                trace = near['trace']
                return {
                    'event': record['response_event_zero_based'],
                    'node': trace['node'],
                    'state': trace['old_state'],
                    'action': trace['action'],
                    'near_gap_C_minus_P': float(near_row[0] - near_row[2]),
                    'far_gap_C_minus_P': float(far_row[0] - far_row[2]),
                    'update_drop': float(trace['old_value'] - trace['new_value']),
                    'greedy_before': trace['greedy_before'],
                    'near_greedy_after': trace['greedy_after'],
                    'far_greedy_after': far['trace']['greedy_after'],
                    'near_row_before': near_row.tolist(),
                    'near_row_after': near['q_row_after'],
                    'far_row_before': far_row.tolist(),
                }
    raise RuntimeError(f'no near/far greedy-set divergence found in {prefix}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    output = args.source/'qgap-figures'
    output.mkdir(exist_ok=False)
    shutil.copyfile(__file__, output/Path(__file__).name)

    cases = []
    for network in range(4):
        for shock in (0, 1):
            case = {'network': network, 'shock': shock,
                    'crossing': first_policy_divergence(
                        args.source/f'network-{network:02d}'/f'prefix-z{shock}.jsonl')}
            replay = np.load(args.source/f'network-{network:02d}'/f'replay-z{shock}.npz')
            case['q_curves'] = {lam: replay[f'lambda_{lam}_q'] for lam in (NEAR, SHAM, FAR)}
            cases.append(case)

    fig = plt.figure(figsize=(12.5, 7.4), layout='constrained')
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.35])
    curve_axes = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(3)]
    for case, ax in zip(cases, curve_axes):
        for lam in (NEAR, SHAM, FAR):
            color, label = STYLES[lam]
            ax.plot(np.arange(len(case['q_curves'][lam])), case['q_curves'][lam],
                    color=color, lw=1.1, label=label)
        c = case['crossing']
        ax.set_title(f"net {case['network']} / shock z{case['shock']}  "
                     f"(first greedy split at event {c['event']})", fontsize=8)
        ax.set_ylim(.80, 1.005)
        ax.tick_params(labelsize=7)
        if case is cases[0]:
            ax.legend(fontsize=6.5, loc='lower left')
    for ax in (curve_axes[0], curve_axes[3]):
        ax.set_ylabel('Contribution q', fontsize=8)
    for ax in (curve_axes[3], curve_axes[4], curve_axes[5]):
        ax.set_xlabel('Replay time (MCS)', fontsize=8)

    ax_gap = fig.add_subplot(grid[0, 3])
    positions = np.arange(len(cases))
    for case, y in zip(cases, positions):
        c = case['crossing']
        ax_gap.scatter([abs(c['near_gap_C_minus_P'])], [y], marker='o', s=42,
                       color='tab:orange', zorder=3, edgecolor='k', lw=.4)
        far_marker = 's' if set(c['far_greedy_after']) == {0} else '^'
        ax_gap.scatter([abs(c['far_gap_C_minus_P'])], [y], marker=far_marker, s=46,
                       color='tab:blue', zorder=3, edgecolor='k', lw=.4)
        ax_gap.scatter([abs(c['update_drop'])], [y], marker='|', s=260,
                       color='black', zorder=4, lw=1.4)
    ax_gap.set_xscale('log')
    ax_gap.set_yticks(positions, [f"net{c['network']}-z{c['shock']}" for c in cases], fontsize=7)
    ax_gap.set_xlabel(r'gap $Q_C-Q_P$ before update (log)', fontsize=8)
    ax_gap.set_title('Single C-update crosses near gaps;\nfar row keeps C or flips to D', fontsize=9)
    ax_gap.scatter([], [], marker='o', color='tab:orange', label='near gap')
    ax_gap.scatter([], [], marker='s', color='tab:blue', label='far gap (far stays C)')
    ax_gap.scatter([], [], marker='^', color='tab:blue', label='far gap (far flips to D)')
    ax_gap.scatter([], [], marker='|', color='black', label=r'$|\Delta Q_C|$ of the update')
    ax_gap.legend(fontsize=6.5, loc='lower right')
    ax_gap.grid(axis='x', alpha=.25)

    focal = cases[0]['crossing']
    ax_focal = fig.add_subplot(grid[1, 3])
    labels = ['before', 'after']
    qc = [focal['near_row_before'][0], focal['near_row_after'][0]]
    bars = ax_focal.bar(labels, qc, width=.5, color='tab:orange', label=r'$Q_C$ (near)')
    qp = focal['near_row_before'][2]
    ax_focal.axhline(qp, color='tab:red', ls='--', lw=1.4, label=r'$Q_P$ (near)')
    ax_focal.scatter([0, 1], [focal['far_row_before'][0], focal['far_row_before'][0]],
                     marker='_', s=220, color='tab:blue', label=r'$Q_C$ (far)')
    ax_focal.scatter([0, 1], [focal['far_row_before'][2], focal['far_row_before'][2]],
                     marker='_', s=220, color='tab:blue', label=r'$Q_P$ (far)')
    for bar, value in zip(bars, qc):
        ax_focal.text(bar.get_x() + bar.get_width()/2, value, f'{value:.4f}',
                      ha='center', va='bottom', fontsize=7.5)
    ax_focal.text(1, qp, f'  {qp:.4f}', va='bottom', fontsize=7.5, color='tab:red')
    ax_focal.set_ylabel('Q value (state row)', fontsize=8)
    ax_focal.set_title(f"Focal split: net0-z0, event {focal['event']}, node {focal['node']}\n"
                       rf"near greedy {{{', '.join(map(str, focal['greedy_before']))}}}$\to$"
                       rf"{{{', '.join(map(str, focal['near_greedy_after']))}}}, "
                       rf"far stays {{{', '.join(map(str, focal['far_greedy_after']))}}}  (0=C, 1=D, 2=P)",
                       fontsize=8)
    ax_focal.legend(fontsize=7, loc='lower left')
    ax_focal.tick_params(labelsize=8)

    fig.suptitle('Threshold-crossing mechanism of the first near/far greedy-set separation\n'
                 r'Policy-preserving $Q_P$ rescaling leaves behavior identical until one C-update '
                 r'pushes $Q_C$ below the rescaled $Q_P$ in near only; $P$ itself is not updated.',
                 fontsize=10)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output/f'qgap-mechanism.{suffix}', dpi=180)
    plt.close(fig)

    summary = {
        'scope': 'Descriptive mechanism summary of the prespecified divergence replay; no new simulation',
        'definition': 'First event where the updated row greedy set differs between near and far branches',
        'source': str(args.source),
        'cases': [{k: v for k, v in case.items() if k != 'q_curves'} for case in cases],
        'crossing_update_always_action_C': all(c['crossing']['action'] == 0 for c in cases),
        'near_always_flips_to_P': all(set(c['crossing']['near_greedy_after']) == {2} for c in cases),
        'far_never_flips_to_P': all(2 not in c['crossing']['far_greedy_after'] for c in cases),
        'update_drop_exceeds_near_gap_all_cases': all(
            abs(c['crossing']['update_drop']) > abs(c['crossing']['near_gap_C_minus_P'])
            for c in cases),
        'cases_with_drop_between_near_and_far_gap': [
            f"net{c['network']}-z{c['shock']}" for c in cases
            if abs(c['crossing']['near_gap_C_minus_P']) < abs(c['crossing']['update_drop'])
            < abs(c['crossing']['far_gap_C_minus_P'])],
        'cases_where_far_flips_to_D': [
            f"net{c['network']}-z{c['shock']}" for c in cases
            if set(c['crossing']['far_greedy_after']) == {1}],
        'near_gap_is_far_gap_quarter_all_cases': all(
            abs(abs(c['crossing']['near_gap_C_minus_P']) * 4
                - abs(c['crossing']['far_gap_C_minus_P'])) < 1e-9
            for c in cases),
        'replay_script_sha256': sha256(args.source/'replay_divergence.py'),
    }
    write_json(output/'qgap-mechanism-summary.json', summary)


if __name__ == '__main__':
    main()
