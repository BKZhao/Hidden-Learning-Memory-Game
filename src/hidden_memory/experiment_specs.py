"""Protocol-owned E0/E1 grids and deterministic independent random streams."""

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class Job:
    stage: str
    point: int
    replicate: int
    root_seed: int
    nodes: int
    edges: int
    r: float
    alpha: float
    beta: float
    training_mcs: int
    group_size: int = 5
    window_mcs: int = 5000
    continuation_mcs: int = 5000
    response_mcs: int = 5000
    chunk_mcs: int = 1000

    @property
    def job_id(self) -> str:
        return f'{self.stage}-p{self.point:02d}-r{self.replicate:02d}'


def build_jobs(stage: str, root_seed: int) -> list[Job]:
    if stage == 'E0':
        points = list(product((1.0, 1.1, 1.2), (0.8,), (0.6,)))
        nodes, edges, duration, repeats = 1000, 1382, 500000, 20
    elif stage == 'E1':
        points = list(product((0.55, 0.70, 0.85, 0.95), (0.02, 0.10, 0.30), (0.4, 0.8, 1.2)))
        nodes, edges, duration, repeats = 300, 480, 20000, 3
    else:
        raise ValueError('Only E0 and E1 are authorized by this stage-B runner')
    return [Job(stage, point, rep, root_seed, nodes, edges, r, alpha, beta, duration)
            for point, (r, alpha, beta) in enumerate(points) for rep in range(repeats)]


def job_seeds(job: Job) -> dict[str, int]:
    sequence = np.random.SeedSequence([job.root_seed, int(job.stage[1:]), job.point, job.replicate])
    names = ('network', 'initialization', 'training', 'future', 'shock', 'audit')
    return {name: int(child.generate_state(1, dtype=np.uint64)[0])
            for name, child in zip(names, sequence.spawn(len(names)))}
