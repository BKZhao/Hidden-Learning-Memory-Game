"""Descriptive phase-plane projection of the E1 scale-validation screening grid."""

import argparse
import csv
import hashlib
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hidden_memory.experiment_io import write_json

R_VALUES = (0.55, 0.70, 0.85, 0.95)
ALPHA_VALUES = (0.02, 0.10, 0.30)
BETA_VALUES = (0.4, 0.8, 1.2)
FP_COL = 3
TAIL_MCS = 5000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_grid(source: Path) -> dict[tuple[float, float, float], dict]:
    points: dict[tuple[float, float, float], dict] = {}
    with open(source/'analysis'/'points.csv', newline='') as handle:
        for row in csv.DictReader(handle):
            key = (float(row['r']), float(row['alpha']), float(row['beta']))
            points[key] = {
                'q': float(row['mean_q']),
                'loss30': float(row['delta_0.30_mean_loss']) if row['delta_0.30_mean_loss'] else np.nan,
                'fP': np.nan,
            }
    fp_by_point: dict[int, list[float]] = {}
    for job_dir in sorted((source/'jobs').iterdir()):
        parts = job_dir.name.split('-')
        point = int(parts[1][1:])
        trajectory = np.load(job_dir/'trajectory.npy', mmap_mode='r')
        fp_by_point.setdefault(point, []).append(float(np.mean(trajectory[-TAIL_MCS:, FP_COL])))
    with open(source/'analysis'/'points.csv', newline='') as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            key = (float(row['r']), float(row['alpha']), float(row['beta']))
            points[key]['fP'] = float(np.mean(fp_by_point[index]))
    return points


def project(points: dict, quantity: str, x_values, y_values, z_values, key_of) -> np.ndarray:
    grid = np.full((len(y_values), len(x_values)), np.nan)
    for iy, yv in enumerate(y_values):
        for ix, xv in enumerate(x_values):
            cells = [points[key_of(xv, yv, zv)][quantity] for zv in z_values]
            grid[iy, ix] = np.mean(cells)
    return grid


def annotate(ax, grid, fmt: str) -> None:
    for iy in range(grid.shape[0]):
        for ix in range(grid.shape[1]):
            value = grid[iy, ix]
            if np.isfinite(value):
                dark = value <= np.nanmax(grid) * .55
                ax.text(ix, iy, fmt.format(value), ha='center', va='center',
                        fontsize=7, color='white' if dark else 'black')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    output = args.source/'phase-figures'
    output.mkdir(exist_ok=False)
    shutil.copyfile(__file__, output/Path(__file__).name)

    points = load_grid(args.source)
    planes = [
        (r'$\alpha$–$r$ plane', ALPHA_VALUES, R_VALUES, BETA_VALUES,
         lambda x, y, z: (x, y, z), r'$r$', r'$\alpha$'),
        (r'$\beta$–$r$ plane', BETA_VALUES, R_VALUES, ALPHA_VALUES,
         lambda x, y, z: (x, z, y), r'$r$', r'$\beta$'),
        (r'$\beta$–$\alpha$ plane', BETA_VALUES, ALPHA_VALUES, R_VALUES,
         lambda x, y, z: (z, x, y), r'$\alpha$', r'$\beta$'),
    ]
    rows = [
        ('q', 'Contribution rate q (final 5000 MCS)', 'viridis', '{:.2f}'),
        ('fP', 'Punisher fraction fP (final 5000 MCS)', 'cividis', '{:.2f}'),
        ('loss30', r'Paired loss L, $\delta$=0.30 shock', 'magma', '{:.3f}'),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 9.2), layout='constrained')
    for row_index, (quantity, title, cmap, fmt) in enumerate(rows):
        images = []
        for col_index, (plane, y_values, x_values, z_values, key_of, x_label, y_label) in enumerate(planes):
            ax = axes[row_index, col_index]
            grid = project(points, quantity, x_values, y_values, z_values, key_of)
            image = ax.imshow(grid, cmap=cmap, origin='lower', aspect='auto',
                              vmin=np.nanmin(grid), vmax=np.nanmax(grid))
            images.append(image)
            ax.set_xticks(range(len(x_values)), [f'{v:g}' for v in x_values])
            ax.set_yticks(range(len(y_values)), [f'{v:g}' for v in y_values])
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if row_index == 0:
                ax.set_title(plane)
            annotate(ax, grid, fmt)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.colorbar(images[2], ax=axes[row_index].tolist(), shrink=.85, pad=.02)
    fig.suptitle('E1 scale-validation screening grid projected to parameter planes\n'
                 r'N=1000, 100000 MCS training, 36 points × 3 replicates; cells average over the third parameter.'
                 '\nExploratory projection of screening data; not work-point selection.',
                 fontsize=10)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output/f'phase-projection.{suffix}', dpi=180)
    plt.close(fig)

    summary = {
        'scope': 'Descriptive projection of the completed E1 scale-validation grid; no new simulation, no selection rule',
        'source': str(args.source),
        'points_csv_sha256': sha256(args.source/'analysis'/'points.csv'),
        'grid': {'r': R_VALUES, 'alpha': ALPHA_VALUES, 'beta': BETA_VALUES,
                 'replicates': 3, 'tail_mcs': TAIL_MCS},
        'high_q_points': sum(1 for v in points.values() if v['q'] >= .9),
        'max_fP_point': max(points, key=lambda k: points[k]['fP']),
        'max_loss30_point': max(points, key=lambda k: points[k]['loss30']),
        'min_loss30_point': min(points, key=lambda k: points[k]['loss30']),
    }
    write_json(output/'phase-projection-summary.json', summary)


if __name__ == '__main__':
    main()
