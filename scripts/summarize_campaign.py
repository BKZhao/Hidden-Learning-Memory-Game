"""Read completed campaign records and summarize every protocol point without selecting one."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, stdev


def summarize(root: Path) -> None:
    status = json.loads((root / 'status.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    raw = (root / 'results.jsonl').read_bytes()
    records = [json.loads(line) for line in raw.decode().splitlines()]
    by_id = {row['job_id']: row for row in records}
    if len(by_id) != len(records):
        raise ValueError('Duplicate job results')
    points = sorted({job['point'] for job in manifest['jobs']})
    rows = []
    for point in points:
        jobs = [job for job in manifest['jobs'] if job['point'] == point]
        ids = [f"{job['stage']}-p{point:02d}-r{job['replicate']:02d}" for job in jobs]
        finished = [by_id[key] for key in ids if key in by_id and by_id[key]['status'] == 'completed']
        qs = [row['tail_means']['q'] for row in finished]
        row = {'point': point, 'r': jobs[0]['r'], 'alpha': jobs[0]['alpha'], 'beta': jobs[0]['beta'],
               'planned': len(jobs), 'completed': len(finished),
               'failed': sum(key in by_id and by_id[key]['status'] == 'failed' for key in ids),
               'mean_q': mean(qs) if qs else None, 'sd_q': stdev(qs) if len(qs) > 1 else None,
               'training_stable': sum(x['stability']['passed'] for x in finished),
               'continuation_stable': sum(bool(x.get('pilot') and x['pilot']['candidate']) for x in finished)}
        for delta in (0.05, 0.1, 0.2, 0.3):
            prefix = f'delta_{delta:.2f}'
            responses = [shock for result in finished if result.get('pilot')
                         for shock in result['pilot']['shocks']
                         if shock['delta'] == delta and shock['status'] == 'completed']
            row[prefix + '_n'] = len(responses)
            row[prefix + '_mean_loss'] = mean(x['signed_loss'] for x in responses) if responses else None
            row[prefix + '_collapse_count'] = sum(x['collapse'] for x in responses)
            row[prefix + '_recovered_count'] = sum(not x['right_censored'] for x in responses)
            row[prefix + '_mean_restricted_unrecovered_mcs'] = mean(x['restricted_unrecovered_time'] for x in responses) if responses else None
        rows.append(row)
    output = root / 'analysis'
    output.mkdir(exist_ok=False)
    with (output / 'points.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'campaign_state': status['state'], 'completed': status['completed'], 'failed': status['failed'],
               'point_count': len(rows), 'training_stable': sum(row['training_stable'] for row in rows),
               'continuation_stable': sum(row['continuation_stable'] for row in rows),
               'all_replicates_stable_points': [row['point'] for row in rows if row['continuation_stable'] == row['planned']],
               'work_point_locked': False, 'results_sha256': sha256(raw).hexdigest(),
               'analysis_script_sha256': sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    summarize(parser.parse_args().campaign)
