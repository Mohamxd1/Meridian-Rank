"""Truncated SVD and orthogonal-projection residuals."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


@dataclass
class LSAModel:
    """Factorization X approximately U diag(S) Vt of rank k."""

    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    k: int


def fit_lsa(X: csr_matrix, k: int | None = None, energy: float = 0.90, seed: int = 0) -> LSAModel:
    """Fit descending singular triplets, selecting k by squared spectral energy."""
    X = csr_matrix(X, dtype=np.float64)
    minimum = min(X.shape)
    if minimum < 1 or not np.isfinite(X.data).all():
        raise ValueError("X must be a nonempty finite matrix")
    if not 0 < energy <= 1:
        raise ValueError("energy must lie in (0, 1]")
    if k is not None and (not isinstance(k, int) or not 1 <= k <= minimum):
        raise ValueError("k must lie in [1, min(X.shape)]")
    count = k if k is not None else min(100, max(1, minimum - 1))
    if minimum <= 2 or count == minimum or X.nnz == 0:
        if X.shape[0] * X.shape[1] > 4_000_000:
            raise ValueError(
                "full-rank dense SVD is limited to 4 million entries; choose smaller k"
            )
        U, S, Vt = np.linalg.svd(X.toarray(), full_matrices=False)
        U, S, Vt = U[:, :count], S[:count], Vt[:count]
    else:
        U, S, Vt = svds(X, k=count, random_state=seed)
        order = np.argsort(S)[::-1]
        U, S, Vt = U[:, order], S[order], Vt[order]
    # Canonicalize the otherwise arbitrary singular-vector signs.
    signs = np.sign(Vt[np.arange(len(S)), np.abs(Vt).argmax(axis=1)])
    signs[signs == 0] = 1
    U, Vt = U * signs, Vt * signs[:, None]
    if k is None:
        total = float(X.multiply(X).sum())
        count = min(count, int(np.searchsorted(np.cumsum(S**2), energy * total)) + 1)
    return LSAModel(U[:, :count], S[:count], Vt[:count], count)


def doc_vectors(model: LSAModel) -> np.ndarray:
    """Return document embeddings U_k diag(S_k)."""
    return model.U * model.S


def similar(model: LSAModel, doc_idx: int, limit: int = 10) -> list[tuple[int, float]]:
    """Rank other documents by cosine similarity in the latent space."""
    vectors = doc_vectors(model)
    if not 0 <= doc_idx < len(vectors) or limit < 0:
        raise ValueError("invalid document index or limit")
    norms = np.linalg.norm(vectors, axis=1)
    unit = np.divide(vectors, norms[:, None], out=np.zeros_like(vectors), where=norms[:, None] > 0)
    scores = np.clip(unit @ unit[doc_idx], -1, 1)
    order = sorted((i for i in range(len(scores)) if i != doc_idx), key=lambda i: (-scores[i], i))
    return [(i, float(scores[i])) for i in order[:limit]]


def reconstruction_error(model: LSAModel, X: csr_matrix) -> np.ndarray:
    """Return ||x - x V V^T|| per row without materializing a dense reconstruction."""
    X = csr_matrix(X, dtype=np.float64)
    if X.shape[1] != model.Vt.shape[1] or not np.isfinite(X.data).all():
        raise ValueError("X must have finite values and the fitted feature count")
    norm2 = np.asarray(X.multiply(X).sum(axis=1)).ravel()
    projection = np.asarray(X @ model.Vt.T)
    residual = np.maximum(norm2 - np.sum(projection**2, axis=1), 0)
    residual[residual <= 1e-12 * np.maximum(norm2, 1)] = 0
    return np.sqrt(residual)


def anomaly_threshold(errors: np.ndarray, rule: str = "p99_or_3sigma") -> float:
    """Return min(p99, mean + 3*std) over finite nonnegative calibration errors."""
    errors = np.asarray(errors, dtype=np.float64)
    if rule != "p99_or_3sigma" or errors.ndim != 1 or not len(errors):
        raise ValueError("require nonempty 1-D errors and rule p99_or_3sigma")
    if not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("errors must be finite and nonnegative")
    return float(min(np.percentile(errors, 99), errors.mean() + 3 * errors.std()))


def anomaly_flags(errors: np.ndarray, rule: str = "p99_or_3sigma") -> np.ndarray:
    """Flag errors strictly above min(p99, mean + 3*std)."""
    return np.asarray(errors) > anomaly_threshold(errors, rule)
