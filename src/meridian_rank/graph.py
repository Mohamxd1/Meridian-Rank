"""Column-oriented sparse graph construction."""

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix, diags


@dataclass
class NodeIndex:
    """Bijection between external IDs and integers in [0, n)."""

    ids: list[str]
    _lookup: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.ids = list(self.ids)
        if any(not isinstance(node, str) or not node for node in self.ids):
            raise ValueError("node IDs must be nonempty strings")
        self._lookup = {node: i for i, node in enumerate(self.ids)}
        if len(self._lookup) != len(self.ids):
            raise ValueError("node IDs must be unique")

    def to_int(self, node_id: str) -> int:
        """Return the unique integer corresponding to node_id."""
        return self._lookup[node_id]

    def to_id(self, i: int) -> str:
        """Return the ID at index i, requiring 0 <= i < n."""
        if not 0 <= i < len(self.ids):
            raise IndexError(i)
        return self.ids[i]


def build_adjacency(
    edges: list[tuple[str, str, float]], index: NodeIndex | None = None
) -> tuple[csr_matrix, NodeIndex]:
    """Build A[dst, src] = sum(weights) while preserving isolated indexed nodes."""
    if index is None:
        index = NodeIndex(list(dict.fromkeys(node for src, dst, _ in edges for node in (src, dst))))
    rows, cols, weights = [], [], []
    for src, dst, weight in edges:
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("edge weights must be finite and nonnegative")
        rows.append(index.to_int(dst))
        cols.append(index.to_int(src))
        weights.append(weight)
    n = len(index.ids)
    matrix = csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=np.float64)
    matrix.eliminate_zeros()
    return matrix, index


def to_transition(A: csr_matrix) -> csr_matrix:
    """Return P = A D^-1; dangling columns remain zero."""
    A = csr_matrix(A, dtype=np.float64)
    if A.shape[0] != A.shape[1] or not np.isfinite(A.data).all() or (A.data < 0).any():
        raise ValueError("adjacency must be square, finite, and nonnegative")
    sums = np.asarray(A.sum(axis=0)).ravel()
    inverse = np.divide(1.0, sums, out=np.zeros_like(sums), where=sums > 0)
    return (A @ diags(inverse)).tocsr()
