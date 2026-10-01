"""Schedule independent training jobs and retain every success/failure record."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import traceback

import numpy as np

from .audit import invariant_events
from .checkpoint import load_checkpoint
from .events import make_tape
from .experiment_io import archive_source, write_json
from .experiment_specs import Job, job_seeds
from .payoff import projected_reward, reward
from .pilot import screen_pilot
from .training import train_job


def run_job(job: Job, campaign: Path) -> dict:
    output = campaign / 'jobs' / job.job_id
    try:
        summary = train_job(job, output)
        state, params, _, _ = load_checkpoint(output / 'checkpoints' / f'mcs-{job.training_mcs}.npz')
        control_nodes = np.flatnonzero(state.strategy != 1)[:1]
        tape = make_tape(np.random.default_rng(job_seeds(job)['audit']), 32)
        controls = invariant_events(state, params, tape, control_nodes)
        maximum_error = max(abs(reward(state, node, action, params) - projected_reward(state, node, action, params))
                            for node in range(min(10, job.nodes)) for action in range(3))
        if maximum_error >= 1e-10:
            raise AssertionError('T06: reward/projection discrepancy in trained checkpoint')
        controls['projection_maximum_error'] = maximum_error
        controls['control_shock_nodes'] = control_nodes.tolist()
        write_json(output / 'controls.json', controls)
        pilot = screen_pilot(job, output, summary) if job.stage == 'E1' else None
        result = {**summary, 'status': 'completed', 'controls': controls, 'pilot': pilot,
                  'source_figure_comparison': 'pending' if job.stage == 'E0' else 'not_applicable'}
        write_json(output / 'result.json', result)
        write_json(output / 'progress.json', {'job_id': job.job_id, 'phase': 'completed',
                   'completed_mcs': job.training_mcs, 'target_mcs': job.training_mcs})
        return result
    except Exception as exc:
        # Worker boundary preserves the complete failure; the campaign exits nonzero.
        output.mkdir(parents=True, exist_ok=True)
        failure = {'job_id': job.job_id, 'status': 'failed', 'error_type': type(exc).__name__,
                   'message': str(exc), 'traceback': traceback.format_exc(), 'job': asdict(job)}
        write_json(output / 'failure.json', failure)
        raise


def execute_jobs(jobs: list[Job], output: Path, workers: int) -> dict:
    if not jobs or workers < 1 or len({j.job_id for j in jobs}) != len(jobs):
        raise ValueError('Require nonempty unique jobs and positive worker count')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'jobs').mkdir()
    archive_source(output, jobs, workers)
    status = {'state': 'running', 'pid': os.getpid(), 'total': len(jobs), 'completed': 0, 'failed': 0,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'workers': workers}
    write_json(output / 'status.json', status)
    # spawn avoids inheriting live JIT runtime state; each independent trajectory stays sequential.
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(run_job, job, output): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                status['completed'] += 1
            except Exception as exc:
                status['failed'] += 1
                result = {'job_id': job.job_id, 'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc)}
            with (output / 'results.jsonl').open('a') as stream:
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
            status['last_job'] = job.job_id
            status['updated_utc'] = datetime.now(timezone.utc).isoformat()
            write_json(output / 'status.json', status)
            print(json.dumps({'job_id': job.job_id, 'status': result['status'],
                              'completed': status['completed'], 'failed': status['failed']}, ensure_ascii=False), flush=True)
    status['state'] = 'completed' if status['failed'] == 0 else 'failed'
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(output / 'status.json', status)
    return status
