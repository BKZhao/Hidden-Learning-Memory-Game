"""Shared continuous washout and input-only support audit for natural histories."""

from pathlib import Path

import numpy as np

from .checkpoint import load_checkpoint, save_checkpoint
from .events import make_tape
from .experiment_io import write_json
from .learner import policy
from .metrics import snapshot
from .natural_intervention import transplant
from .parameters import Parameters
from .runner import run_trajectory
from .state import State, clone
from .training import hidden_diagnostics


def wash_and_audit(reference: State, sources: dict[str, State], params: Parameters,
                   output: Path, config: dict, network: int, meta: dict, wash_seed: int) -> list[dict]:
    rng = np.random.default_rng(wash_seed)
    records = []; previous = 0
    for wash in config['wash_mcs']:
        folder = output/f'wash-{wash}'; folder.mkdir()
        if wash > previous:
            trajectories = {key: [] for key in sources}
            for offset in range(previous, wash, 1000):
                tape = make_tape(rng, reference.graph.n*min(1000, wash-offset))
                for key, state in sources.items():
                    trajectories[key].append(run_trajectory(state, params, tape, backend='numba'))
            np.savez_compressed(folder/'wash-trajectories.npz', **{
                f'{key}_{metric}': np.concatenate([part[metric][1:] for part in parts])
                for key, parts in trajectories.items() for metric in parts[0]})
        public = {}; cp_checksums = {}
        for key, state in sources.items():
            diagnostic = {'network': network, 'history_type': key, 'wash_mcs': wash,
                          'alpha_pre': params.alpha if key == 'low' else config['alpha_high'],
                          'snapshot': snapshot(state, params), 'hidden': hidden_diagnostics(state)}
            cp = save_checkpoint(folder/f'{key}.npz', state, params, {'wash': rng.bit_generator.state}, diagnostic)
            cp_checksums[key] = cp['checksum']; public[key] = diagnostic
        branches, mapping, support = transplant(reference, sources['low'], sources['high'], params, config['minimum_pool'])
        np.savez_compressed(folder/'mapping.npz', **mapping)
        for key, state in branches.items():
            save_checkpoint(folder/f'transplant-{key}.npz', state, params, {},
                            {'purpose': 'natural_value_calibrated_intervention', 'network': network,
                             'wash_mcs': wash, 'source_checksums': cp_checksums,
                             'reference_checksum': meta['checksum'], 'history_type': key})
        public['strategy_agreement'] = float(np.mean(sources['low'].strategy == sources['high'].strategy))
        public['policy_row_agreement'] = float(np.mean(np.all(
            policy(sources['low'].q, params.epsilon) == policy(sources['high'].q, params.epsilon), axis=-1)))
        eligible = support['matched'] >= config['minimum_rows'] and support['fraction_all_rows'] >= config['minimum_fraction']
        result = {'network': network, 'wash_mcs': wash, 'individual_eligible': eligible,
                  'reason': 'passed_input_support' if eligible else 'insufficient_common_support',
                  'support': support, 'raw_history': public}
        write_json(folder/'audit.json', result); records.append(result); previous = wash
    common = None
    for wash in config['wash_mcs']:
        with np.load(output/f'wash-{wash}/mapping.npz') as mapping:
            common = mapping['rows'].copy() if common is None else np.intersect1d(common, mapping['rows'])
    common_eligible = len(common) >= config['minimum_rows'] and len(common)/reference.q[..., 2].size >= config['minimum_fraction']
    for record in records:
        folder = output/f"wash-{record['wash_mcs']}"
        with np.load(folder/'mapping.npz') as mapping:
            chosen = np.isin(mapping['rows'], common)
            np.savez_compressed(folder/'common-mapping.npz', **{key: mapping[key][chosen] for key in mapping.files})
        dose = {}
        for key in ('sham', 'low', 'high'):
            individual, _, _, _ = load_checkpoint(folder/f'transplant-{key}.npz')
            branch = clone(reference)
            branch.q.reshape(-1, 3)[common, 2] = individual.q.reshape(-1, 3)[common, 2]
            delta = branch.q-reference.q
            dose[key] = {'changed_rows': int(np.count_nonzero(delta)), 'l1': float(np.abs(delta).sum()), 'l2': float(np.linalg.norm(delta))}
            if not np.array_equal(policy(branch.q, params.epsilon), policy(reference.q, params.epsilon)):
                raise AssertionError('Common-support transplant changed policy')
            save_checkpoint(folder/f'common-{key}.npz', branch, params, {},
                            {'network': network, 'wash_mcs': record['wash_mcs'], 'history_type': key,
                             'reference_checksum': meta['checksum'], 'common_support_rows': len(common)})
        record.update(common_rows=len(common), common_fraction=len(common)/reference.q[..., 2].size,
                      eligible=common_eligible, common_dose=dose)
        write_json(folder/'audit.json', record)
    return records
