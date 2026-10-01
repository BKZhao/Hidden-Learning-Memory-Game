"""Fail-closed gate for the later confirmatory experiment."""


def validate_confirmatory(config: dict) -> None:
    if config.get('status') != 'locked':
        raise ValueError('Confirmatory configuration must be locked using independent pilot evidence')
    required = ('r', 'alpha', 'beta', 'independent_networks', 'training_histories',
                'future_replicates', 'maximum_micro_updates', 'pilot_evidence', 'code_version')
    if any(config.get(key) is None for key in required):
        raise ValueError('Confirmatory configuration contains unlocked fields')
    if not 1 / config['group_size'] < config['r'] < 1:
        raise ValueError('Confirmatory work point must lie in the social dilemma region')
    for key in ('independent_networks', 'training_histories', 'future_replicates', 'maximum_micro_updates'):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
