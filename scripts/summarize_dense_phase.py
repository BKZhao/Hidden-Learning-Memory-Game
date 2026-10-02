"""Summarize fixed dense-phase training and paired-response results by parameter point."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, stdev


def aggregate(training_root: Path, response_root: Path, output: Path) -> dict:
    training_status = json.loads((training_root/'status.json').read_text())
    response_status = json.loads((response_root/'status.json').read_text())
    if training_status['state'] != 'completed' or response_status['state'] != 'completed':
        raise ValueError('Dense campaigns must complete before summary')
    manifest = json.loads((training_root/'manifest.json').read_text())
    training = {row['job_id']: row for row in map(json.loads, (training_root/'results.jsonl').read_text().splitlines())}
    responses = {(row['point'], row['replicate']): row
                 for row in map(json.loads, (response_root/'results.jsonl').read_text().splitlines())}
    rows = []
    for point in sorted({job['point'] for job in manifest['jobs']}):
        jobs = [job for job in manifest['jobs'] if job['point'] == point]
        records = [training[f"E5-p{point:02d}-r{job['replicate']:02d}"] for job in jobs]
        eligible = [responses[(point, job['replicate'])] for job in jobs
                    if (point, job['replicate']) in responses and responses[(point, job['replicate'])]['eligible']]
        q_values = [record['tail_means']['q'] for record in records]
        fp_values = [record['tail_means']['fP'] for record in records]
        loss_values = [record['loss'] for record in eligible]
        rows.append({'point': point, 'r': jobs[0]['r'], 'alpha': jobs[0]['alpha'], 'beta': jobs[0]['beta'],
                     'training_n': len(records), 'training_stable_n': sum(r['stability']['passed'] for r in records),
                     'mean_q': mean(q_values), 'sd_q': stdev(q_values),
                     'mean_fP': mean(fp_values), 'sd_fP': stdev(fp_values),
                     'response_n': len(eligible), 'mean_loss': mean(loss_values) if loss_values else None,
                     'sd_loss': stdev(loss_values) if len(loss_values) > 1 else None})
    output.mkdir(exist_ok=False)
    with (output/'points.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'points': len(rows), 'training_jobs': len(training), 'response_jobs': len(responses),
               'response_eligible': sum(row['response_n'] for row in rows),
               'full_training_points': sum(row['training_n'] == 2 for row in rows),
               'full_response_points': sum(row['response_n'] == 2 for row in rows),
               'training_results_sha256': sha256((training_root/'results.jsonl').read_bytes()).hexdigest(),
               'response_results_sha256': sha256((response_root/'results.jsonl').read_bytes()).hexdigest(),
               'script_sha256': sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--response', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.training, args.response, args.output), indent=2))


if __name__ == '__main__':
    main()
