"""Map empirical natural P gaps to a fixed recipient without changing its policy."""

import numpy as np
from numpy.typing import NDArray

from .learner import policy
from .matched_intervention import strata
from .parameters import Parameters
from .payoff import reward_bounds
from .state import State, clone


def transplant(reference: State, low: State, high: State, params: Parameters,
               minimum_pool: int = 20) -> tuple[dict[str, State], dict[str, NDArray], dict]:
    """Use common empirical quantiles in exact public-state/row/greedy strata.

    Recipient ranks depend only on its original gap. Donor gaps are actual observed
    values, with replacement if needed. Both donor values must be legal together.
    This constructs natural-value calibrated states, not natural joint histories.
    """
    if minimum_pool < 1:
        raise ValueError('minimum_pool must be positive')
    states = {'reference': reference, 'low': low, 'high': high}
    labels = {key: strata(state).ravel() for key, state in states.items()}
    gaps = {key: (state.q[..., :2].max(axis=-1)-state.q[..., 2]).ravel()
            for key, state in states.items()}
    eligible = {key: value > 1e-10 for key, value in gaps.items()}
    maximum = reference.q[..., :2].max(axis=-1).ravel()
    original = reference.q[..., 2].ravel()
    lower, upper = reward_bounds(reference.graph.g, params)
    lower, upper = min(0., lower/(1-params.gamma)), max(0., upper/(1-params.gamma))
    rows, donor_low, donor_high, strata_audit = [], [], [], []
    for label in sorted(np.unique(labels['reference'][eligible['reference']])):
        pools = {key: np.flatnonzero((value == label) & eligible[key])
                 for key, value in labels.items()}
        for key, pool in pools.items():
            pools[key] = pool[np.argsort(gaps[key][pool], kind='stable')]
        sizes = {key: len(pool) for key, pool in pools.items()}
        audit = {'stratum': int(label), 'pool_sizes': sizes, 'joint_legal': 0}
        if min(sizes['low'], sizes['high']) < minimum_pool:
            audit['reason'] = 'insufficient_donor_pool'
            strata_audit.append(audit)
            continue
        recipient = pools['reference']
        quantiles = (np.arange(len(recipient))+.5)/len(recipient)
        donor = {key: pools[key][np.minimum((quantiles*sizes[key]).astype(int), sizes[key]-1)]
                 for key in ('low', 'high')}
        legal = np.ones(len(recipient), dtype=bool)
        for key in ('low', 'high'):
            values = maximum[recipient]-gaps[key][donor[key]]
            legal &= (values >= lower) & (values <= upper) & (values < maximum[recipient])
        rows.extend(recipient[legal].tolist())
        donor_low.extend(donor['low'][legal].tolist())
        donor_high.extend(donor['high'][legal].tolist())
        audit['joint_legal'] = int(legal.sum())
        audit['reason'] = 'matched' if legal.any() else 'no_joint_legal_values'
        strata_audit.append(audit)
    arrays = {'rows': np.asarray(rows, dtype=np.int64),
              'low_donor_rows': np.asarray(donor_low, dtype=np.int64),
              'high_donor_rows': np.asarray(donor_high, dtype=np.int64)}
    rows = arrays['rows']
    branches = {'sham': clone(reference)}
    report = {'recipient_candidates': int(eligible['reference'].sum()), 'matched': len(rows),
              'fraction_all_rows': len(rows)/original.size, 'strata': strata_audit,
              'minimum_pool': minimum_pool, 'sources': {}}
    for key in ('low', 'high'):
        branch = clone(reference)
        donor_rows = arrays[f'{key}_donor_rows']
        mapped = gaps[key][donor_rows]
        values = maximum[rows]-mapped
        # Self-mapping must be a bitwise sham, without subtract/add roundoff.
        values = np.where(mapped == gaps['reference'][rows], original[rows], values)
        branch.q.reshape(-1, 3)[rows, 2] = values
        if not np.array_equal(policy(branch.q, params.epsilon), policy(reference.q, params.epsilon)):
            raise AssertionError('Natural-gap transplant changed initial policy')
        branches[key] = branch
        arrays[f'{key}_gaps'] = mapped
        report['sources'][key] = {
            'unique_donor_rows': len(np.unique(donor_rows)),
            'changed_rows': int(np.count_nonzero(values != original[rows])),
            'l1_change': float(np.abs(values-original[rows]).sum()),
            'l2_change': float(np.linalg.norm(values-original[rows])),
            'mapped_gap_quantiles': np.quantile(mapped, [.1, .25, .5, .75, .9]).tolist() if len(rows) else []}
    return branches, arrays, report
