"""Immutable checkpoint files with content checksum and provenance."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from . import __version__
from .hypergraph import from_edges
from .parameters import Parameters
from .state import State, validate_state

ARRAY_NAMES = ('strategy', 'q', 'counts', 'visit_count', 'last_update_event')


def json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def array_hash(array: NDArray) -> str:
    header = json_text({'shape': array.shape, 'dtype': str(array.dtype)}).encode()
    return sha256(header + array.tobytes(order='C')).hexdigest()


def source_hash() -> str:
    digest = sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def provenance() -> dict[str, Any]:
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent,
                            capture_output=True, text=True, check=False)
    return {'code_commit': result.stdout.strip() if result.returncode == 0 else None,
            'source_sha256': source_hash(), 'package_version': __version__,
            'python': platform.python_version(), 'numpy': np.__version__,
            'dtype': 'float64', 'actions': ['C', 'D', 'P'], 'tie_rule': 'exact_uniform'}


def save_checkpoint(path: Path, state: State, params: Parameters,
                    rng_states: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    arrays = {name: getattr(state, name) for name in ARRAY_NAMES}
    arrays['edges'] = state.graph.edges
    metadata = {'schema': 1, 'checkpoint_id': str(uuid4()), 'n': state.graph.n,
                'event': state.event, 'rejected_graphs': state.graph.rejected_graphs,
                'parameters': asdict(params), 'rng_states': rng_states,
                'diagnostics': diagnostics, 'provenance': provenance(),
                'network_hash': array_hash(state.graph.edges),
                'parameter_hash': sha256(json_text(asdict(params)).encode()).hexdigest(),
                'array_hashes': {name: array_hash(a) for name, a in arrays.items()}}
    metadata['checksum'] = sha256(json_text(metadata).encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents overwriting a checkpoint or following an existing symlink.
    with path.open('xb') as stream:
        np.savez_compressed(stream, metadata=np.array(json_text(metadata)), **arrays)
    return metadata


def load_checkpoint(path: Path) -> tuple[State, Parameters, dict[str, Any], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        checksum = metadata.pop('checksum')
        if sha256(json_text(metadata).encode()).hexdigest() != checksum:
            raise ValueError('Checkpoint metadata checksum mismatch')
        if metadata['schema'] != 1:
            raise ValueError('Unsupported checkpoint schema')
        arrays = {name: data[name].copy() for name in (*ARRAY_NAMES, 'edges')}
        for name, array in arrays.items():
            if array_hash(array) != metadata['array_hashes'][name]:
                raise ValueError(f'Checkpoint checksum mismatch: {name}')
    graph = from_edges(metadata['n'], arrays.pop('edges'), metadata['rejected_graphs'])
    state = State(graph=graph, event=metadata['event'], **arrays)
    validate_state(state)
    metadata['checksum'] = checksum
    return state, Parameters(**metadata['parameters']), metadata['rng_states'], metadata
