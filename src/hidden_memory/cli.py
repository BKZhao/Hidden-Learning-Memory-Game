"""Command line entry points for engineering audits and throughput measurement."""

import argparse
import json
from pathlib import Path

from .audit import benchmark, run_audit
from .config import validate_confirmatory
from .campaign import execute_jobs
from .experiment_specs import build_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description='Hidden memory protocol: engineering validation')
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit', help='Run small-scale invariant checks and save evidence')
    audit.add_argument('--output', type=Path, required=True)
    audit.add_argument('--nodes', type=int, default=20)
    audit.add_argument('--training-mcs', type=int, default=200)
    audit.add_argument('--response-mcs', type=int, default=100)
    audit.add_argument('--seed', type=int, default=20260930)
    bench = sub.add_parser('benchmark', help='Measure N=1000 reference throughput')
    bench.add_argument('--output', type=Path, required=True)
    bench.add_argument('--events', type=int, default=1000000)
    bench.add_argument('--backend', choices=('reference', 'numba'), default='reference')
    bench.add_argument('--nodes', type=int, default=1000)
    gate = sub.add_parser('validate-confirmatory', help='Reject configurations without a pilot lock')
    gate.add_argument('config', type=Path)
    campaign = sub.add_parser('campaign', help='Run protocol E0 or sham-only E1 independent jobs')
    campaign.add_argument('stage', choices=('E0', 'E1'))
    campaign.add_argument('--output', type=Path, required=True)
    campaign.add_argument('--workers', type=int, default=2)
    campaign.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'audit':
        report = run_audit(args.output, args.nodes, args.training_mcs, args.response_mcs, args.seed)
        print(json.dumps({key: report[key] for key in ('status', 'purpose', 'checkpoint_id', 'frozen_theta')}, indent=2))
    elif args.command == 'benchmark':
        print(json.dumps(benchmark(args.output, args.events, backend=args.backend, nodes=args.nodes), indent=2))
    elif args.command == 'campaign':
        status = execute_jobs(build_jobs(args.stage, args.seed), args.output, args.workers)
        if status['failed']:
            raise SystemExit(1)
    else:
        validate_confirmatory(json.loads(args.config.read_text()))
        print('Configuration fields validated; no experiment was started.')
