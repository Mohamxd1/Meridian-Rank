"""Probability-preserving power iteration."""

import numpy as np
from scipy.sparse import csr_matrix


def _iterate(
    P: csr_matrix, teleport: np.ndarray, d: float, tol: float, max_iter: int
) -> tuple[np.ndarray, list[float]]:
    P = csr_matrix(P, dtype=np.float64)
    if P.shape[0] != P.shape[1] or not np.isfinite(P.data).all() or (P.data < 0).any():
        raise ValueError("transition must be square, finite, and nonnegative")
    if not 0 <= d < 1 or not np.isfinite(tol) or tol <= 0 or max_iter < 1:
        raise ValueError("require 0 <= damping < 1, positive tolerance and iterations")
    sums = np.asarray(P.sum(axis=0)).ravel()
    dangling = sums == 0
    if not np.allclose(sums[~dangling], 1, atol=1e-12, rtol=1e-12):
        raise ValueError("non-dangling transition columns must sum to one")
    if not P.shape[0]:
        return np.array([], dtype=np.float64), []
    rank = np.full(P.shape[0], 1 / P.shape[0], dtype=np.float64)
    history = []
    for _ in range(max_iter):
        # Personalized teleportation also supplies the dangling distribution.
        updated = d * (P @ rank + rank[dangling].sum() * teleport) + (1 - d) * teleport
        error = float(np.abs(updated - rank).sum())
        history.append(error)
        rank = updated
        if error < tol:
            return rank / rank.sum(), history
    raise RuntimeError(f"PageRank did not converge after {max_iter} iterations (L1={error:g})")


def pagerank(
    P: csr_matrix,
    d: float = 0.85,
    tol: float = 1e-10,
    max_iter: int = 200,
    return_history: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[float]]:
    """Solve r = d(P r + dangling_mass/n) + (1-d)/n by power iteration."""
    n = P.shape[0]
    teleport = np.full(n, 1 / n) if n else np.array([], dtype=np.float64)
    rank, history = _iterate(P, teleport, d, tol, max_iter)
    return (rank, history) if return_history else rank
