"""Run the fixed v1.1 candidate-point sham validation, with archived launch evidence."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil

from hidden_memory.campaign import execute_jobs
from hidden_memory.experiment_specs import build_jobs, Job


def validation_jobs(config: dict) -> list[Job]:
    if config['stage'] != 'E1_shock_validation' or config['primary_delta'] != 0.3:
        raise ValueError('Require the v1.1 sham validation configuration')
    templates = {j.point: j for j in build_jobs('E1', config['root_seed']) if j.replicate == 0}
    return [replace(templates[point], replicate=rep, nodes=config['nodes'], edges=config['edges'],
                    training_mcs=config['training_mcs'])
            for point in config['point_priority'] for rep in range(config['replicates'])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    scripts = Path(__file__).parent
    for name, digest in config['script_sha256'].items():
        if sha256((scripts/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Fixed script hash mismatch: {name}')
    jobs = validation_jobs(config)
    status = execute_jobs(jobs, args.output, config['workers'])
    for name in config['script_sha256']:
        shutil.copyfile(scripts/name, args.output/name)
    shutil.copyfile(args.config, args.output/'validation-config.json')
    if status['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
