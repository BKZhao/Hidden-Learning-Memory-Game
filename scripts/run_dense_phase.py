"""Run the fixed dense E5 parameter grid without intervention branches."""

import argparse
from dataclasses import replace
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import shutil

from hidden_memory.campaign import execute_jobs
from hidden_memory.experiment_specs import Job


def dense_jobs(config: dict) -> list[Job]:
    if config['stage'] != 'E5_dense_phase' or config['replicates'] != 2:
        raise ValueError('Require the fixed two-replicate dense-phase design')
    points = list(product(config['r_values'], config['alpha_values'], config['beta_values']))
    expected = len(config['r_values'])*len(config['alpha_values'])*len(config['beta_values'])
    if len(points) != expected or len(set(points)) != expected:
        raise ValueError('Dense-phase grid contains duplicate parameter points')
    template = Job('E5', 0, 0, config['root_seed'], config['nodes'], config['edges'],
                   points[0][0], points[0][1], points[0][2], config['training_mcs'],
                   group_size=config['group_size'], window_mcs=config['window_mcs'],
                   continuation_mcs=config['response_mcs'], response_mcs=config['response_mcs'],
                   chunk_mcs=config['chunk_mcs'])
    return [replace(template, point=point, replicate=replicate, r=r, alpha=alpha, beta=beta)
            for point, (r, alpha, beta) in enumerate(points)
            for replicate in range(config['replicates'])]


def verify_scripts(config: dict) -> None:
    scripts = Path(__file__).parent
    for name, expected in config['script_sha256'].items():
        actual = sha256((scripts/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Fixed script hash mismatch: {name}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    verify_scripts(config)
    jobs = dense_jobs(config)
    if len(jobs) != 378:
        raise ValueError('Fixed dense-phase design requires 378 training jobs')
    status = execute_jobs(jobs, args.output, config['workers'])
    for name in config['script_sha256']:
        shutil.copyfile(Path(__file__).parent/name, args.output/name)
    shutil.copyfile(args.config, args.output/'config.json')
    if status['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
