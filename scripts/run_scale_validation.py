"""Launch a preregistered full-grid E1 scale validation using the unchanged simulation."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from hidden_memory.campaign import execute_jobs
from hidden_memory.experiment_specs import build_jobs


def validation_jobs(config: dict) -> list:
    if config['stage'] != 'E1_scale_validation' or config['work_point_locked']:
        raise ValueError('This launcher only supports unlocked E1 scale validation')
    jobs = build_jobs('E1', config['root_seed'])
    return [replace(job, nodes=config['nodes'], edges=config['edges'],
                    training_mcs=config['training_mcs']) for job in jobs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    actual_hash = sha256(Path(__file__).read_bytes()).hexdigest()
    if actual_hash != config['launcher_sha256']:
        raise ValueError('Launcher differs from the locked validation configuration')
    status = execute_jobs(validation_jobs(config), args.output, config['workers'])
    if status['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
