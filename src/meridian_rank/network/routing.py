"""Heap-based shortest paths and synchronous distance-vector convergence."""

import heapq
import math

from .topology import Topology


def _neighbors(topo: Topology) -> dict[str, list[tuple[str, float]]]:
    neighbors = {node: [] for node in topo.nodes}
    for link in topo.links:
        neighbors[link.src].append((link.dst, float(link.cost)))
        if not topo.directed and link.src != link.dst:
            neighbors[link.dst].append((link.src, float(link.cost)))
    return {node: sorted(edges) for node, edges in neighbors.items()}


def _dijkstra(neighbors: dict, source: str) -> tuple[dict[str, float], dict[str, str | None]]:
    if source not in neighbors:
        raise ValueError("source is not in topology")
    distance = dict.fromkeys(neighbors, math.inf)
    previous = dict.fromkeys(neighbors, None)
    distance[source] = 0.0
    queue = [(0.0, source)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distance[node]:
            continue
        for neighbor, weight in neighbors[node]:
            candidate = cost + weight
            if candidate < distance[neighbor]:
                distance[neighbor], previous[neighbor] = candidate, node
                heapq.heappush(queue, (candidate, neighbor))
    return distance, previous


def dijkstra(topo: Topology, source: str) -> tuple[dict[str, float], dict[str, str | None]]:
    """Minimize sum of positive link costs from source using heapq."""
    return _dijkstra(_neighbors(topo), source)


def shortest_path(topo: Topology, src: str, dst: str) -> tuple[list[str], float] | None:
    """Return a minimum-cost path, or None for an unreachable destination."""
    if dst not in topo.nodes:
        raise ValueError("destination is not in topology")
    distance, previous = dijkstra(topo, src)
    if math.isinf(distance[dst]):
        return None
    path, current = [], dst
    while current is not None:
        path.append(current)
        current = previous[current]
    return list(reversed(path)), distance[dst]


def bellman_ford_rounds(topo: Topology) -> int:
    """Count synchronous neighbor-table exchanges, including the final unchanged round."""
    n = len(topo.nodes)
    if n < 2:
        return 0
    neighbors = _neighbors(topo)
    table = {a: {b: (0.0 if a == b else math.inf) for b in topo.nodes} for a in topo.nodes}
    for a in topo.nodes:
        for b, cost in neighbors[a]:
            table[a][b] = min(table[a][b], cost)
    for iteration in range(1, n + 1):
        updated = {a: row.copy() for a, row in table.items()}
        for a in topo.nodes:
            for b, cost in neighbors[a]:
                for dst in topo.nodes:
                    updated[a][dst] = min(updated[a][dst], cost + table[b][dst])
        if updated == table:
            return iteration
        table = updated
    raise RuntimeError("distance-vector routing failed to converge")


def next_hop_table(topo: Topology) -> dict[str, dict[str, str]]:
    """Map each reachable nonlocal destination to its first shortest-path hop."""
    result = {}
    neighbors = _neighbors(topo)
    for source in topo.nodes:
        distance, previous = _dijkstra(neighbors, source)
        result[source] = {}
        for dest in topo.nodes:
            if dest == source or math.isinf(distance[dest]):
                continue
            hop = dest
            while previous[hop] != source:
                hop = previous[hop]
            result[source][dest] = hop
    return result
