"""Export the manuscript-owned TikZ schematic as a standalone vector figure."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tex-bin', type=Path, default=Path('/usr/bin'))
    args = parser.parse_args()
    source = ROOT/'paper/figures/design.tex'
    text = source.read_text()
    diagram = text[text.index(r'\begin{tikzpicture}'):text.index(r'\end{tikzpicture}')+len(r'\end{tikzpicture}')]
    work = ROOT/'tmp/paper-design'
    work.mkdir(parents=True, exist_ok=True)
    output = ROOT/'artifacts/paper-draft/figures'
    output.mkdir(parents=True, exist_ok=True)
    document = (r'\documentclass[tikz,border=4pt]{standalone}' '\n'
                r'\usepackage{lmodern,amsmath}' '\n'
                r'\usetikzlibrary{arrows.meta,positioning}' '\n'
                r'\begin{document}' '\n'+diagram+'\n'+r'\end{document}'+'\n')
    (work/'design.tex').write_text(document)
    result = subprocess.run([str(args.tex_bin/'pdflatex'), '-halt-on-error',
                             '-interaction=nonstopmode', 'design.tex'], cwd=work,
                            text=True, capture_output=True, check=False)
    (work/'build.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(f'Schematic build failed; inspect {work}/build.log')
    shutil.copyfile(work/'design.pdf', output/'fig01-design.pdf')
    subprocess.run(['pdftoppm', '-r', '300', '-singlefile', '-png', str(work/'design.pdf'),
                    str(output/'fig01-design')], check=True, capture_output=True)
    (output/'fig01-design.json').write_text(json.dumps({
        'scope': 'Protocol schematic, no invented or empirical trajectory data',
        'source': str(source.relative_to(ROOT)),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2)+'\n')


if __name__ == '__main__':
    main()
