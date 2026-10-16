"""Validated network topologies and deterministic synthetic generators."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from ..graph import NodeIndex, build_adjacency


@dataclass
class Link:
    """A positive-cost link with cost=max(1, 100000/bandwidth) by default."""

    src: str
    dst: str
    bandwidth_mbps: float = 1000.0
    latency_ms: float = 1.0
    cost: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.bandwidth_mbps) or self.bandwidth_mbps <= 0:
            raise ValueError("bandwidth must be finite and positive")
        if not np.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency must be finite and nonnegative")
        if self.cost is None:
            self.cost = max(1.0, 100_000 / self.bandwidth_mbps)
        if not np.isfinite(self.cost) or self.cost <= 0:
            raise ValueError("cost must be finite and positive")


@dataclass
class Topology:
    """Simple directed or bidirectional graph with explicitly indexed nodes."""

    nodes: list[str]
    links: list[Link]
    directed: bool = False

    def __post_init__(self) -> None:
        NodeIndex(self.nodes)
        known = set(self.nodes)
        seen = set()
        for link in self.links:
            if link.src not in known or link.dst not in known:
                raise ValueError("link endpoints must exist in nodes")
            key = (link.src, link.dst) if self.directed else tuple(sorted((link.src, link.dst)))
            if key in seen:
                raise ValueError("parallel links are not supported")
            seen.add(key)

    def adjacency(self, weight: str = "cost") -> tuple[csr_matrix, NodeIndex]:
        """Build A[dst, src] with cost, bandwidth_mbps, latency_ms, or unit weights."""
        if weight not in {"cost", "bandwidth_mbps", "latency_ms", "unit"}:
            raise ValueError("unknown weight")
        edges = []
        for link in self.links:
            value = 1.0 if weight == "unit" else float(getattr(link, weight))
            edges.append((link.src, link.dst, value))
            if not self.directed and link.src != link.dst:
                edges.append((link.dst, link.src, value))
        return build_adjacency(edges, NodeIndex(self.nodes))


def make_ring(n: int, **link_kwargs) -> Topology:
    """Create a cycle with n >= 3 nodes and n undirected links."""
    if n < 3:
        raise ValueError("a simple ring requires at least three nodes")
    return Topology(
        [str(i) for i in range(n)],
        [Link(str(i), str((i + 1) % n), **link_kwargs) for i in range(n)],
    )


def make_grid(rows: int, cols: int, **link_kwargs) -> Topology:
    """Create rows*cols nodes joined to their horizontal and vertical neighbors."""
    if rows < 1 or cols < 1:
        raise ValueError("grid dimensions must be positive")
    nodes = [f"{r},{c}" for r in range(rows) for c in range(cols)]
    links = []
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                links.append(Link(f"{r},{c}", f"{r + 1},{c}", **link_kwargs))
            if c + 1 < cols:
                links.append(Link(f"{r},{c}", f"{r},{c + 1}", **link_kwargs))
    return Topology(nodes, links)


def make_scale_free(n: int, m: int = 2, seed: int = 0, **link_kwargs) -> Topology:
    """Preferential attachment from an (m+1)-node star, adding m links per node."""
    if not 1 <= m < n:
        raise ValueError("require 1 <= m < n")
    rng = np.random.default_rng(seed)
    edges = [(0, i) for i in range(1, m + 1)]
    degrees = np.zeros(n)
    degrees[0], degrees[1 : m + 1] = m, 1
    for node in range(m + 1, n):
        targets = rng.choice(node, size=m, replace=False, p=degrees[:node] / degrees[:node].sum())
        for target in targets:
            edges.append((node, int(target)))
            degrees[target] += 1
        degrees[node] = m
    return Topology(
        [str(i) for i in range(n)], [Link(str(a), str(b), **link_kwargs) for a, b in edges]
    )


def make_leaf_spine(spines: int, leaves: int, hosts_per_leaf: int = 4, **link_kwargs) -> Topology:
    """Join each leaf to every spine and to its hosts."""
    if spines < 1 or leaves < 1 or hosts_per_leaf < 0:
        raise ValueError("positive spine/leaf counts and nonnegative host count required")
    nodes = [f"spine{i}" for i in range(spines)] + [f"leaf{i}" for i in range(leaves)]
    links = [
        Link(f"spine{s}", f"leaf{leaf}", **link_kwargs)
        for s in range(spines)
        for leaf in range(leaves)
    ]
    for leaf in range(leaves):
        for host in range(hosts_per_leaf):
            name = f"host{leaf}-{host}"
            nodes.append(name)
            links.append(Link(f"leaf{leaf}", name, **link_kwargs))
    return Topology(nodes, links)


def load_topology_yaml(path: str) -> Topology:
    """Load a safe YAML mapping of nodes and positive-cost links via the I/O layer."""
    from ..store import read_topology_yaml

    return read_topology_yaml(path)
