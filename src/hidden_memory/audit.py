"""Small-scale invariant audit and event-level diagnostic evidence."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from .checkpoint import array_hash, json_text, load_checkpoint, provenance, save_checkpoint
from .events import make_tape
from .hypergraph import sample_connected
from .intervention import intervene
from .learner import policy, step
from .parameters import Parameters
from .payoff import projected_reward, reward
from .runner import run_paired, run_trajectory
from .shock import apply_shock, sample_shock
from .state import State, clone, initialize


def invariant_events(state: State, params: Parameters, tape: NDArray[np.float64],
                     nodes: NDArray[np.int64]) -> dict[str, object]:
    for z in (0, 1):
        branches, _ = intervene(state, params)
        for branch in branches.values():
            if z:
                apply_shock(branch, nodes)
        originals = {scale: branch.q.copy() for scale, branch in branches.items()}
        for event in tape:
            traces = [step(branch, params, event, freeze=True) for branch in branches.values()]
            if len({(t.node, t.action, t.reward) for t in traces}) != 1:
                raise AssertionError('T08: frozen event differs')
            for branch in branches.values():
                if not np.array_equal(branch.strategy, branches[1.0].strategy):
                    raise AssertionError('T08: frozen strategy configuration differs')
        for scale, branch in branches.items():
            if not np.array_equal(branch.q, originals[scale]):
                raise AssertionError('T08: frozen Q changed')
        a, b = clone(state), intervene(state, params)[0][1.0]
        if z:
            apply_shock(a, nodes)
            apply_shock(b, nodes)
        for event in tape:
            if step(a, params, event) != step(b, params, event):
                raise AssertionError('T09: sham event differs')
            if not np.array_equal(a.q, b.q) or not np.array_equal(a.strategy, b.strategy):
                raise AssertionError('T09: sham state differs')
    return {'T08_frozen_events_per_shock_arm': len(tape),
            'T09_sham_events_per_shock_arm': len(tape), 'shock_arms_checked': [0, 1]}


def mechanism_log(path: Path, state: State, params: Parameters,
                  tape: NDArray[np.float64], nodes: NDArray[np.int64]) -> dict[str, object]:
    edited, _ = intervene(state, params)
    branches = {scale: edited[scale] for scale in (0.5, 2.0)}
    for branch in branches.values():
        apply_shock(branch, nodes)
    first_policy = None
    first_action = None
    with path.open('x') as stream:
        for offset, event in enumerate(tape):
            current = {scale: step(branch, params, event) for scale, branch in branches.items()}
            if first_action is None and current[0.5].action != current[2.0].action:
                first_action = {'response_event': offset, 'traces': {str(k): asdict(v) for k, v in current.items()}}
            if first_policy is None and not np.array_equal(policy(branches[0.5].q, params.epsilon), policy(branches[2.0].q, params.epsilon)):
                first_policy = {'after_response_event': offset, 'traces': {str(k): asdict(v) for k, v in current.items()}}
            for scale, trace in current.items():
                stream.write(json_text({'lambda': scale, 'shock': 1, **asdict(trace)}) + '\n')
    return {'first_all_state_policy_divergence': first_policy, 'first_actual_action_divergence': first_action}


def run_audit(output: Path, nodes: int = 20, training_mcs: int = 200,
              response_mcs: int = 100, seed: int = 20260930) -> dict[str, object]:
    if nodes < 5 or training_mcs < 1 or response_mcs < 1:
        raise ValueError('Audit requires N>=5 and positive MCS durations')
    output.mkdir(parents=True, exist_ok=False)
    params = Parameters(r=0.7, alpha=0.1, beta=0.8)
    seeds = np.random.SeedSequence(seed).spawn(4)
    streams = {name: np.random.default_rng(child) for name, child in zip(('network', 'training', 'future', 'shock'), seeds)}
    graph = sample_connected(nodes, 5, round(nodes * 8 / 5), streams['network'])
    state = initialize(graph, seed)
    training = run_trajectory(state, params, make_tape(streams['training'], nodes * training_mcs))
    checkpoint_meta = save_checkpoint(output / 'checkpoint.npz', state, params,
                                     {key: rng.bit_generator.state for key, rng in streams.items()},
                                     {'training_id': f'audit-seed-{seed}', 'stable': None,
                                      'purpose': 'engineering_validation_only'})
    restored, _, _, _ = load_checkpoint(output / 'checkpoint.npz')
    tape = make_tape(streams['future'], nodes * response_mcs)
    shocked = sample_shock(state, 0.1, streams['shock'])
    projection_error = max(abs(reward(state, i, a, params) - projected_reward(state, i, a, params))
                           for i in range(nodes) for a in range(3))
    if projection_error >= 1e-10:
        raise AssertionError('T06: projection reward mismatch')
    if not np.array_equal(run_trajectory(clone(state), params, tape)['q'], run_trajectory(restored, params, tape)['q']):
        raise AssertionError('T12: reloaded continuation differs')
    audit = invariant_events(state, params, tape, shocked)
    result = run_paired(state, params, tape, shocked)
    frozen = run_paired(state, params, tape, shocked, freeze=True)
    mechanism = mechanism_log(output / 'events.jsonl', state, params, tape, shocked)
    config = {'purpose': 'engineering_validation_only', 'protocol_version': 'memory_resilience_v1.0',
              'seed': seed, 'n': nodes, 'group_size': 5, 'mean_hyperdegree': 8,
              'training_mcs': training_mcs, 'response_mcs': response_mcs, 'parameters': asdict(params),
              'lambdas': [0.5, 1, 2], 'shock_fraction_of_total_population': 0.1,
              'metric_interval_mcs': 1, 'reference_window_requirement_met': False}
    (output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    np.savez_compressed(output / 'training.npz', **training)
    np.savez_compressed(output / 'events.npz', tape=tape, shock_nodes=shocked)
    for name, paired in [('paired', result), ('frozen', frozen)]:
        arrays = {f'lambda_{scale}_z_{z}_{metric}': values
                  for (scale, z), trajectory in paired.trajectories.items() for metric, values in trajectory.items()}
        np.savez_compressed(output / f'{name}_trajectories.npz', **arrays)
    gaps = state.q[..., :2].max(axis=-1) - state.q[..., 2]
    report = {'status': 'passed', 'purpose': 'engineering_validation_only',
              'checkpoint_id': checkpoint_meta['checkpoint_id'], 'checkpoint_checksum': checkpoint_meta['checksum'],
              'provenance': provenance(), 'branch_count': 6, 'training_micro_updates': nodes * training_mcs,
              'future_micro_updates_per_branch': len(tape), 'network_rejections': graph.rejected_graphs,
              'maximum_projection_reward_error': projection_error, 'eligible_fraction': result.eligible_fraction,
              'low_intervention_coverage': result.eligible_fraction < 0.2,
              'natural_support_reference': None, 'gap_quantiles_all_rows': np.quantile(gaps, [0.01, 0.25, 0.5, 0.75, 0.99]).tolist(),
              'shock_nodes': shocked.tolist(), 'shock_nodes_hash': array_hash(shocked),
              'event_tape_hash': array_hash(tape), 'frozen_theta': frozen.theta,
              'engineering_sample_theta': result.theta, 'signed_losses': result.losses,
              'invariant_checks': audit, 'mechanism': mechanism}
    report['artifact_sha256'] = {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}
    (output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def benchmark(output: Path, events: int, seed: int = 123,
              backend: str = 'reference', nodes: int = 1000) -> dict[str, object]:
    if nodes < 10 or events < nodes or events % nodes:
        raise ValueError('Benchmark requires N>=10 and a positive whole number of MCS')
    output.mkdir(parents=True, exist_ok=False)
    graph = sample_connected(nodes, 5, round(nodes * 8 / 5), np.random.default_rng(seed))
    state = initialize(graph, seed)
    params = Parameters(r=0.7)
    rng = np.random.default_rng(seed + 1)
    warmup_start = perf_counter()
    run_trajectory(clone(state), params, make_tape(np.random.default_rng(seed + 2), nodes), backend=backend)
    warmup_seconds = perf_counter() - warmup_start
    start = perf_counter()
    # Bound tape memory; fixed four-slot events give identical input under chunking.
    chunk = nodes * 10
    for offset in range(0, events, chunk):
        run_trajectory(state, params, make_tape(rng, min(chunk, events - offset)), backend=backend)
    elapsed = perf_counter() - start
    io_start = perf_counter()
    save_checkpoint(output / 'checkpoint.npz', state, params, {'events': rng.bit_generator.state},
                    {'purpose': 'benchmark', 'training_id': f'benchmark-{seed}', 'stable': None})
    report = {'purpose': 'implementation_benchmark', 'backend': backend, 'n': nodes, 'g': 5,
              'micro_updates': events, 'elapsed_seconds': elapsed, 'seconds_per_million_updates': elapsed * 1e6 / events,
              'warmup_seconds_excluded': warmup_seconds,
              'checkpoint_write_seconds': perf_counter() - io_start,
              'checkpoint_bytes': (output / 'checkpoint.npz').stat().st_size,
              'event_tape_max_bytes': chunk * 4 * 8,
              'E1_serial_hours_extrapolated': elapsed / events * (36 * 3 * 300 * 20000) / 3600,
              'E2_serial_days_extrapolated': elapsed / events * 3.2e10 / 86400,
              'provenance': provenance()}
    (output / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
