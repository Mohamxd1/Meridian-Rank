"""Shortest-path demand routing and low-rank traffic residuals."""

import math

import numpy as np
from scipy.sparse import csr_matrix

from ..lsa import fit_lsa, reconstruction_error
from .centrality import pagerank_centrality
from .routing import _dijkstra, _neighbors
from .topology import Topology


def _key(topo: Topology, src: str, dst: str) -> tuple[str, str]:
    return (src, dst) if topo.directed else tuple(sorted((src, dst)))


def link_load(
    topo: Topology, demands: dict[tuple[str, str], float]
) -> dict[tuple[str, str], float]:
    """Sum Mbps over deterministic minimum-cost paths; reject unroutable positive demand."""
    load = {_key(topo, link.src, link.dst): 0.0 for link in topo.links}
    neighbors = _neighbors(topo)
    routes = {}
    for (src, dst), demand in demands.items():
        if not np.isfinite(demand) or demand < 0:
            raise ValueError("demand must be finite and nonnegative")
        if src not in neighbors or dst not in neighbors:
            raise ValueError("demand endpoint not in topology")
        if demand == 0 or src == dst:
            continue
        if src not in routes:
            routes[src] = _dijkstra(neighbors, src)
        distance, previous = routes[src]
        if math.isinf(distance[dst]):
            raise ValueError(f"unroutable demand: {src} -> {dst}")
        current = dst
        while current != src:
            parent = previous[current]
            load[_key(topo, parent, current)] += demand
            current = parent
    return load


def utilization(topo: Topology, load: dict) -> dict[tuple[str, str], float]:
    """Divide aggregate physical-link Mbps by the link's bandwidth capacity."""
    capacities = {_key(topo, link.src, link.dst): link.bandwidth_mbps for link in topo.links}
    if any(
        key not in capacities or not np.isfinite(value) or value < 0 for key, value in load.items()
    ):
        raise ValueError("loads must be finite, nonnegative, and reference known links")
    return {key: float(load.get(key, 0) / capacity) for key, capacity in capacities.items()}


def markov_flow(topo: Topology, d: float = 0.85) -> dict[str, float]:
    """Return the normalized damped random-walk stationary distribution."""
    return pagerank_centrality(topo, d)


def traffic_anomaly(series: np.ndarray, k: int | None = None) -> np.ndarray:
    """Return per-timestep orthogonal residuals of a rank-k traffic matrix."""
    series = np.asarray(series, dtype=np.float64)
    if series.ndim != 2:
        raise ValueError("series must have shape (timesteps, flows)")
    matrix = csr_matrix(series)
    return reconstruction_error(fit_lsa(matrix, k=k), matrix)
