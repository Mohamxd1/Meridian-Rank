"""Compare network algorithms with NetworkX on deterministic topologies."""

from time import perf_counter

import networkx as nx

from meridian_rank.network.centrality import betweenness_centrality
from meridian_rank.network.failure import criticality_ranking
from meridian_rank.network.routing import dijkstra
from meridian_rank.network.topology import make_grid, make_scale_free

from .common import write_results


def main() -> None:
    rows = []
    for n in [25, 50, 100]:
        topo = make_scale_free(n, seed=42, cost=1)
        graph = nx.Graph()
        graph.add_nodes_from(topo.nodes)
        graph.add_weighted_edges_from((link.src, link.dst, link.cost) for link in topo.links)
        algorithms = {
            "dijkstra": lambda: dijkstra(topo, "0"),
            "networkx_dijkstra": lambda: nx.single_source_dijkstra_path_length(graph, "0"),
            "betweenness": lambda: betweenness_centrality(topo),
            "networkx_betweenness": lambda: nx.betweenness_centrality(graph, weight="weight"),
            "criticality": lambda: criticality_ranking(topo),
        }
        for name, operation in algorithms.items():
            start = perf_counter()
            operation()
            rows.append(
                {
                    "topology": "scale_free",
                    "nodes": n,
                    "algorithm": name,
                    "seconds": perf_counter() - start,
                }
            )
    topo = make_grid(100, 100, cost=1)
    start = perf_counter()
    distance, _ = dijkstra(topo, "0,0")
    seconds = perf_counter() - start
    assert distance["99,99"] == 198
    rows.append(
        {"topology": "100x100_grid", "nodes": 10000, "algorithm": "dijkstra", "seconds": seconds}
    )
    write_results("network", rows, {"seed": 42, "repetitions": 1})


if __name__ == "__main__":
    main()
