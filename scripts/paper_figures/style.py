"""Shared style and provenance for manuscript figures; no simulation execution."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'artifacts/paper-draft/figures'
COLORS = ['#009e73', '#666666', '#d55e00']
LABELS = [r'Near ($\lambda=0.5$)', r'Sham ($\lambda=1$)', r'Far ($\lambda=2$)']
STYLES = ['-', '--', '-.']
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'legend.frameon': False, 'axes.spines.top': False,
    'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'lines.linewidth': 1.3, 'savefig.dpi': 300,
})


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def panel(ax, letter):
    ax.text(-0.13, 1.035, f'({letter})', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom')


def save(fig, name, sources, note):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png'):
        fig.savefig(OUTPUT / f'{name}.{extension}', bbox_inches='tight')
    paths = [ROOT / p for p in sources]
    paths.extend(sorted(Path(__file__).parent.glob('*.py')))
    provenance = {'note': note, 'inputs_and_scripts_sha256': {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths}}
    (OUTPUT / f'{name}.json').write_text(json.dumps(provenance, indent=2) + '\n')
    plt.close(fig)
