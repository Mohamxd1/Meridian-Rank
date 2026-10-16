"""Spectral and weighted shortest-path centrality, without NetworkX."""

import heapq

import numpy as np

from ..graph import to_transition
from ..pagerank import pagerank
from .routing import _neighbors
from .topology import Topology


def eigenvector_centrality(
    topo: Topology, tol: float = 1e-10, max_iter: int = 500
) -> dict[str, float]:
    """Compute the unit-L2 Perron vector of unit adjacency using shifted power iteration."""
    if tol <= 0 or not np.isfinite(tol) or max_iter < 1:
        raise ValueError("positive tolerance and iteration count required")
    n = len(topo.nodes)
    if not n:
        return {}
    matrix, _ = topo.adjacency("unit")
    vector = np.full(n, 1 / np.sqrt(n))
    for _ in range(max_iter):
        updated = vector + matrix @ vector  # A + I avoids oscillation on bipartite graphs.
        updated /= np.linalg.norm(updated)
        if np.abs(updated - vector).sum() < tol:
            return dict(zip(topo.nodes, map(float, updated)))
        vector = updated
    raise RuntimeError("eigenvector centrality did not converge")


def betweenness_centrality(topo: Topology, normalized: bool = True) -> dict[str, float]:
    """Accumulate weighted Brandes dependencies sigma_st(v)/sigma_st."""
    neighbors = _neighbors(topo)
    scores = dict.fromkeys(topo.nodes, 0.0)
    for source in topo.nodes:
        stack = []
        predecessors = {node: [] for node in topo.nodes}
        sigma = dict.fromkeys(topo.nodes, 0.0)
        sigma[source] = 1.0
        distance = dict.fromkeys(topo.nodes, np.inf)
        distance[source] = 0.0
        queue = [(0.0, source)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != distance[node]:
                continue
            stack.append(node)
            for target, weight in neighbors[node]:
                candidate = cost + weight
                if candidate < distance[target]:
                    distance[target] = candidate
                    heapq.heappush(queue, (candidate, target))
                    predecessors[target] = [node]
                    sigma[target] = sigma[node]
                elif candidate == distance[target]:
                    predecessors[target].append(node)
                    sigma[target] += sigma[node]
        dependency = dict.fromkeys(topo.nodes, 0.0)
        for node in reversed(stack):
            for parent in predecessors[node]:
                dependency[parent] += sigma[parent] / sigma[node] * (1 + dependency[node])
            if node != source:
                scores[node] += dependency[node]
    n = len(topo.nodes)
    scale = 1 / ((n - 1) * (n - 2)) if normalized and n > 2 else 1.0
    if not normalized and not topo.directed:
        scale = 0.5
    return {node: score * scale for node, score in scores.items()}


def pagerank_centrality(topo: Topology, d: float = 0.85) -> dict[str, float]:
    """Rank nodes by the damped stationary distribution over unit edges."""
    matrix, index = topo.adjacency("unit")
    return dict(zip(index.ids, map(float, pagerank(to_transition(matrix), d=d))))
