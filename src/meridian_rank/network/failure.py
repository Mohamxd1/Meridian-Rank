"""Immutable failure simulation and reachability damage."""

import math
from dataclasses import dataclass, replace
from itertools import combinations, permutations

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

from .centrality import betweenness_centrality, pagerank_centrality
from .routing import _dijkstra, _neighbors
from .topology import Topology
from .traffic import link_load, utilization


@dataclass
class Impact:
    """Connectivity loss, surviving-path stretch, load change, and L1 rank shift."""

    components: int
    unreachable_pairs: int
    avg_path_stretch: float
    max_utilization_change: float
    pagerank_shift: float


def fail_node(topo: Topology, node: str) -> Topology:
    """Remove a node and all incident links without changing the input graph."""
    if node not in topo.nodes:
        raise ValueError("node not in topology")
    return Topology(
        [n for n in topo.nodes if n != node],
        [replace(link) for link in topo.links if node not in (link.src, link.dst)],
        topo.directed,
    )


def fail_link(topo: Topology, src: str, dst: str) -> Topology:
    """Remove one physical link, respecting direction for directed topologies."""
    kept = [
        link
        for link in topo.links
        if not (
            (link.src, link.dst) == (src, dst)
            or (not topo.directed and (link.dst, link.src) == (src, dst))
        )
    ]
    if len(kept) == len(topo.links):
        raise ValueError("link not in topology")
    return Topology(list(topo.nodes), [replace(link) for link in kept], topo.directed)


def _distances(topo: Topology) -> dict[str, dict[str, float]]:
    neighbors = _neighbors(topo)
    return {node: _dijkstra(neighbors, node)[0] for node in topo.nodes}


def _impact(topo: Topology, failed: Topology, demands: dict | None, baseline: tuple) -> Impact:
    before, rank_before = baseline
    after = _distances(failed)
    pairs = permutations(topo.nodes, 2) if topo.directed else combinations(topo.nodes, 2)
    lost, stretches = 0, []
    for src, dst in pairs:
        old = before[src][dst]
        new = after.get(src, {}).get(dst, math.inf)
        if math.isfinite(old):
            if not math.isfinite(new):
                lost += 1
            else:
                stretches.append(new / old)
    matrix, _ = failed.adjacency("unit")
    components = int(connected_components(matrix, directed=failed.directed, connection="weak")[0])
    rank_after = pagerank_centrality(failed)
    shift = sum(abs(rank_before[node] - rank_after.get(node, 0.0)) for node in topo.nodes)
    change = 0.0
    if demands:
        previous = utilization(topo, link_load(topo, demands))
        surviving = {
            pair: value
            for pair, value in demands.items()
            if math.isfinite(after.get(pair[0], {}).get(pair[1], math.inf))
        }
        current = utilization(failed, link_load(failed, surviving))
        change = max(
            (
                abs(current.get(key, 0) - previous.get(key, 0))
                for key in set(previous) | set(current)
            ),
            default=0.0,
        )
    return Impact(components, lost, float(np.mean(stretches)) if stretches else 1.0, change, shift)


def impact(topo: Topology, failed: Topology, demands: dict | None = None) -> Impact:
    """Measure newly unreachable pairs and mean distance ratios after a failure."""
    if failed.directed != topo.directed or not set(failed.nodes) <= set(topo.nodes):
        raise ValueError("failed topology must retain the original direction and a subset of nodes")
    return _impact(topo, failed, demands, (_distances(topo), pagerank_centrality(topo)))


def criticality_ranking(topo: Topology, demands: dict | None = None) -> list[tuple[str, Impact]]:
    """Rank single-node failures by newly unreachable pairs, then L1 PageRank shift."""
    baseline = (_distances(topo), pagerank_centrality(topo))
    result = [
        (node, _impact(topo, fail_node(topo, node), demands, baseline)) for node in topo.nodes
    ]
    return sorted(
        result, key=lambda pair: (-pair[1].unreachable_pairs, -pair[1].pagerank_shift, pair[0])
    )


def centrality_vs_impact(topo: Topology) -> float:
    """Return Spearman rho between weighted betweenness and unreachable-pair damage."""
    centrality = betweenness_centrality(topo)
    ranking = criticality_ranking(topo)
    x = [centrality[node] for node, _ in ranking]
    y = [damage.unreachable_pairs for _, damage in ranking]
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return 0.0  # Defined convention for a constant ranking.
    return float(spearmanr(x, y).statistic)
