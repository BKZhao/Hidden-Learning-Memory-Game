"""Figure 3: dense parameter-space maps and response eligibility."""

import csv
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from style import ROOT, panel, plt, save

R_VALUES = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
ALPHA_VALUES = (0.02, 0.05, 0.08, 0.11, 0.15, 0.22, 0.30)
BETA_VALUES = (0.4, 0.8, 1.2)


def load_points():
    source = ROOT / 'artifacts/e5-dense-phase-20261001'
    rows = list(csv.DictReader((source/'analysis/points.csv').open()))
    points = {}
    for row in rows:
        key = (float(row['r']), float(row['alpha']), float(row['beta']))
        points[key] = {
            'q': float(row['mean_q']),
            'fP': float(row['mean_fP']),
            'loss': float(row['mean_loss']) if row['mean_loss'] else np.nan,
            'response_n': int(row['response_n']),
            'split_phase': float(row['sd_q']) > .5,
        }
    return points


def matrix(points, beta, metric):
    return np.array([[points[(r, alpha, beta)][metric] for r in R_VALUES]
                     for alpha in ALPHA_VALUES])


def main():
    points = load_points()
    figure, axes = plt.subplots(3, 3, figsize=(7.2, 6.9), layout='constrained')
    row_specs = (
        ('q', 'Contribution rate $q$', plt.colormaps['viridis'], colors.Normalize(0, 1)),
        ('fP', 'Punisher fraction $f_P$', plt.colormaps['cividis'], colors.Normalize(0, .5)),
        ('loss', r'Sham loss $L_1$', plt.colormaps['magma_r'],
         colors.PowerNorm(gamma=.25, vmin=0, vmax=1)),
    )
    for row_index, (metric, label, colormap, normalization) in enumerate(row_specs):
        colormap = colormap.copy()
        colormap.set_bad('#eeeeee')
        images = []
        for column_index, beta in enumerate(BETA_VALUES):
            ax = axes[row_index, column_index]
            values = matrix(points, beta, metric)
            display_values = np.maximum(values, 0) if metric == 'loss' else values
            image = ax.imshow(display_values, origin='lower', aspect='auto', cmap=colormap,
                              norm=normalization, interpolation='nearest')
            images.append(image)
            if metric == 'loss':
                response_n = matrix(points, beta, 'response_n').astype(int)
                for cell_row, cell_column in np.argwhere(response_n == 0):
                    ax.add_patch(Rectangle((cell_column-.5, cell_row-.5), 1, 1,
                                           facecolor='none', edgecolor='.60', hatch='///',
                                           linewidth=0))
                for cell_row, cell_column in np.argwhere(response_n == 1):
                    ax.add_patch(Rectangle((cell_column-.5, cell_row-.5), 1, 1,
                                           facecolor='none', edgecolor='white', hatch='...',
                                           linewidth=0))
            if metric == 'q':
                split_phase = matrix(points, beta, 'split_phase').astype(bool)
                split_rows, split_columns = np.nonzero(split_phase)
                ax.scatter(split_columns, split_rows, marker='x', s=27, color='white',
                           linewidth=.9, zorder=3)
            tick_columns = range(0, len(R_VALUES), 2)
            ax.set_xticks(tick_columns, [f'{R_VALUES[index]:g}' for index in tick_columns])
            ax.set_yticks(range(len(ALPHA_VALUES)), [f'{value:g}' for value in ALPHA_VALUES])
            ax.set_xlabel('$r$')
            if column_index == 0:
                ax.set_ylabel('$\\alpha$')
            else:
                ax.set_yticklabels([])
            if row_index == 0:
                ax.text(.5, 1.08, rf'$\beta={beta:g}$', transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=9)
            if beta == .4:
                ax.scatter(0, 0, marker='*', s=90, facecolor='none', edgecolor='white', linewidth=1.2)
                ax.scatter(0, 0, marker='*', s=68, facecolor='none', edgecolor='black', linewidth=.6)
            panel(ax, chr(97 + 3*row_index + column_index))
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(.5)
                spine.set_color('.45')
        colorbar = figure.colorbar(images[-1], ax=axes[row_index].tolist(), shrink=.82, pad=.015)
        colorbar.set_label(label)
        if metric == 'loss':
            colorbar.set_ticks([0, 1e-4, 1e-3, 1e-2, 1e-1, 1])
            colorbar.set_ticklabels(['0', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '1'])
    figure.legend(handles=[
        Line2D([], [], marker='x', linestyle='none', color='.25', markersize=5,
               label='Split high/low repeats'),
        Patch(facecolor='#eeeeee', edgecolor='.60', hatch='///', label='No eligible response'),
        Patch(facecolor='.45', edgecolor='white', hatch='...', label='$n=1$ response'),
    ], loc='outside lower center', ncol=3, frameon=False, fontsize=8)
    save(figure, 'fig03-phase-map', [
        'artifacts/e5-dense-phase-20261001/analysis/points.csv',
        'artifacts/e5-dense-phase-20261001/analysis/summary.json',
        'artifacts/e5-dense-phase-20261001/manifest.json',
        'artifacts/e5-dense-phase-response-20261001/manifest.json',
    ], 'Dense phase grid: 189 parameter points, two independent training replicates each. '
       'q and fP are final-5000-MCS means. Shock loss uses only stable replicates; hatching marks '
       'zero eligible responses and stippling marks n=1. Crosses mark one high and one low q replicate. '
       'Negative loss is displayed at the zero floor. '
       'The star marks the locked operating point selected before this dense grid.')
    output = ROOT/'artifacts/paper-draft/tables'
    output.mkdir(parents=True, exist_ok=True)
    with (output/'phase-map-values.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['r', 'alpha', 'beta', 'mean_q', 'mean_fP', 'delta_0.30_sham_loss',
                         'eligible_response_replicates'])
        for key in sorted(points):
            item = points[key]
            writer.writerow([*key, item['q'], item['fP'], item['loss'], item['response_n']])


if __name__ == '__main__':
    main()
