"""Build the Markdown-owned manuscript using the archived official Elsevier class."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/paper-draft'
SECTIONS = ('abstract', 'introduction', 'model', 'results', 'conclusion', 'appendix')


def run(command, cwd, log):
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'Build failed: {command[0]}; inspect {log}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures', action='store_true', help='Regenerate data figures before compilation')
    parser.add_argument('--tex-bin', type=Path, default=Path('/usr/bin'),
                        help='TeX binary directory; avoids incomplete Conda TeX installation')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/'sections').mkdir(exist_ok=True)
    template = ROOT/'tmp/elsarticle'
    template.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ROOT/'artifacts/literature/elsarticle-official.zip') as archive:
        for name in ('elsarticle.dtx', 'elsarticle.ins', 'elsarticle-harv.bst', 'README'):
            (template/name).write_bytes(archive.read('elsarticle/'+name))
    run([str(args.tex_bin/'pdflatex'), '-interaction=nonstopmode', 'elsarticle.ins'],
        template, OUTPUT/'template-build.log')
    if args.figures:
        run([str(ROOT/'.venv-analysis/bin/python'), str(ROOT/'scripts/paper_figures/design.py'),
             '--tex-bin', str(args.tex_bin)], ROOT, OUTPUT/'plot-design.log')
        for name in ('reproduction', 'phase_map', 'main_result', 'mechanism', 'natural_history', 'secondary'):
            run([str(ROOT/'.venv-analysis/bin/python'), str(ROOT/f'scripts/paper_figures/{name}.py')],
                ROOT, OUTPUT/f'plot-{name}.log')
    pandoc = shutil.which('pandoc')
    if pandoc is None:
        raise FileNotFoundError('pandoc is required for the manuscript build')
    for name in SECTIONS:
        run([pandoc, str(ROOT/f'paper/draft/{name}.md'), '-f', 'markdown+raw_tex',
             '-t', 'latex', '--natbib', '--wrap=none', '-o', str(OUTPUT/f'sections/{name}.tex')],
            ROOT, OUTPUT/f'convert-{name}.log')
    os.environ['TEXINPUTS'] = str(template) + os.pathsep + os.environ.get('TEXINPUTS', '')
    os.environ['BSTINPUTS'] = str(template) + os.pathsep + os.environ.get('BSTINPUTS', '')
    os.environ['PATH'] = str(args.tex_bin) + os.pathsep + os.environ['PATH']
    command = [str(args.tex_bin/'latexmk'), '-pdf', '-interaction=nonstopmode', '-halt-on-error',
               '-outdir='+str(OUTPUT), 'main.tex']
    run(command, ROOT/'paper', OUTPUT/'build.log')
    build_log = (OUTPUT/'main.log').read_text()
    if any(message in build_log for message in ('Overfull', 'undefined', 'Missing character:')):
        raise RuntimeError('PDF compiled with layout or reference problems; inspect main.log')
    paths = sorted((ROOT/'paper').rglob('*.md')) + sorted((ROOT/'paper').rglob('*.tex'))
    paths += [ROOT/'paper/references.bib', Path(__file__)]
    (OUTPUT/'manuscript-provenance.json').write_text(json.dumps({
        'sources_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'pdf_sha256': hashlib.sha256((OUTPUT/'main.pdf').read_bytes()).hexdigest(),
        'source_of_truth': 'paper/draft/*.md; generated LaTeX is in artifacts/paper-draft/sections',
    }, indent=2)+'\n')
    print(OUTPUT/'main.pdf')


if __name__ == '__main__':
    main()
