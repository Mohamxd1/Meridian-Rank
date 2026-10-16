"""Personalized PageRank using weighted teleportation seeds."""

import numpy as np
from scipy.sparse import csr_matrix

from .pagerank import _iterate


def personalized_pagerank(
    P: csr_matrix,
    seeds: dict[int, float],
    d: float = 0.85,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> np.ndarray:
    """Solve r = d(P r + dangling_mass*s) + (1-d)*s for normalized seeds s."""
    s = np.zeros(P.shape[0], dtype=np.float64)
    for i, weight in seeds.items():
        if not isinstance(i, (int, np.integer)) or not 0 <= i < len(s):
            raise ValueError("seed index outside graph")
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("seed weights must be finite and nonnegative")
        s[i] = weight
    if not np.isfinite(s.sum()) or s.sum() <= 0:
        raise ValueError("at least one positive seed is required")
    return _iterate(P, s / s.sum(), d, tol, max_iter)[0]
