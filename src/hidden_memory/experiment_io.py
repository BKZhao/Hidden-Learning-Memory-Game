"""Atomic progress records, file inventories and local source archives."""

from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import tarfile
from typing import Any

from .checkpoint import provenance
from .experiment_specs import Job


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def archive_source(output: Path, jobs: list[Job], workers: int) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    archive = output / 'source.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for directory in ('src/hidden_memory', 'tests', 'configs', 'docs'):
            for path in sorted((root / directory).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    tar.add(path, arcname=str(path.relative_to(root)))
        for name in ('pyproject.toml', 'uv.lock', 'AGENTS.md', 'hidden_learning_memory_protocol_v1.docx'):
            tar.add(root / name, arcname=name)
    manifest = {'schema': 1, 'jobs': [asdict(job) for job in jobs], 'workers': workers,
                'provenance': provenance(), 'source_archive_sha256': file_hash(archive),
                'numba': version('numba'), 'llvmlite': version('llvmlite'),
                'protocol_sha256': file_hash(root / 'hidden_learning_memory_protocol_v1.docx')}
    write_json(output / 'manifest.json', manifest)
    return manifest
