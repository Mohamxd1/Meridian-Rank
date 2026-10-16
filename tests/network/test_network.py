import math
from dataclasses import asdict

import networkx as nx
import numpy as np
import pytest

from meridian_rank.network.centrality import (
    betweenness_centrality,
    eigenvector_centrality,
    pagerank_centrality,
)
from meridian_rank.network.failure import (
    centrality_vs_impact,
    criticality_ranking,
    fail_link,
    fail_node,
    impact,
)
from meridian_rank.network.routing import (
    bellman_ford_rounds,
    dijkstra,
    next_hop_table,
    shortest_path,
)
from meridian_rank.network.topology import (
    Link,
    Topology,
    make_grid,
    make_leaf_spine,
    make_ring,
    make_scale_free,
)
from meridian_rank.network.traffic import link_load, markov_flow, traffic_anomaly, utilization


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("directed", [False, True])
def test_dijkstra_and_brandes_against_networkx(seed, directed):
    graph = nx.gnp_random_graph(15, 0.17, seed=seed, directed=directed)
    rng = np.random.default_rng(seed)
    graph = nx.relabel_nodes(graph, str)
    links = []
    for a, b in graph.edges:
        cost = int(rng.integers(1, 6))
        graph[a][b]["weight"] = cost
        links.append(Link(a, b, cost=cost))
    topo = Topology(list(graph.nodes), links, directed)
    distances, _ = dijkstra(topo, "0")
    oracle = nx.single_source_dijkstra_path_length(graph, "0")
    for node in topo.nodes:
        assert distances[node] == oracle.get(node, math.inf)
    for normalized in [True, False]:
        centrality = nx.betweenness_centrality(graph, weight="weight", normalized=normalized)
        assert betweenness_centrality(topo, normalized) == pytest.approx(centrality, abs=1e-8)
    assert pagerank_centrality(topo) == pytest.approx(
        nx.pagerank(graph, weight=None, tol=1e-13, max_iter=1000), abs=1e-8
    )


@pytest.mark.parametrize("seed", range(5))
def test_eigenvector_matches_networkx(seed):
    topo = make_scale_free(25, seed=seed)
    graph = nx.Graph()
    graph.add_nodes_from(topo.nodes)
    graph.add_edges_from((link.src, link.dst) for link in topo.links)
    assert eigenvector_centrality(topo) == pytest.approx(
        nx.eigenvector_centrality(graph, tol=1e-12), abs=1e-8
    )


def test_generator_counts_determinism_and_ring_rounds():
    assert len(make_ring(8).links) == 8
    assert len(make_grid(3, 4).links) == 17
    fabric = make_leaf_spine(2, 4, 3)
    assert len(fabric.nodes) == 18
    assert len(fabric.links) == 20
    assert asdict(make_scale_free(20, seed=4)) == asdict(make_scale_free(20, seed=4))
    assert bellman_ford_rounds(make_ring(10)) == 5
    assert bellman_ford_rounds(fabric) == 4
    with pytest.raises(ValueError):
        make_ring(2)
    with pytest.raises(ValueError):
        make_scale_free(2, m=2)
    with pytest.raises(ValueError):
        Topology(["a"], [Link("a", "b")])


def test_path_next_hop_disconnection_and_traffic_conservation():
    topo = Topology(["a", "b", "c", "x"], [Link("a", "b", cost=2), Link("b", "c", cost=3)])
    assert shortest_path(topo, "a", "c") == (["a", "b", "c"], 5)
    assert shortest_path(topo, "a", "x") is None
    assert next_hop_table(topo)["a"] == {"b": "b", "c": "b"}
    load = link_load(topo, {("a", "c"): 10, ("c", "a"): 5})
    assert sum(load.values()) == 30
    assert utilization(topo, load) == {("a", "b"): 0.015, ("b", "c"): 0.015}
    assert sum(markov_flow(topo).values()) == pytest.approx(1)
    with pytest.raises(ValueError):
        link_load(topo, {("a", "x"): 1})


def test_star_failure_includes_removed_endpoints_and_no_mutation():
    topo = Topology(["center", "a", "b", "c"], [Link("center", leaf) for leaf in ["a", "b", "c"]])
    original = asdict(topo)
    center = impact(topo, fail_node(topo, "center"))
    leaf = impact(topo, fail_node(topo, "a"))
    assert center.components == 3
    assert center.unreachable_pairs == 6
    assert leaf.components == 1
    assert leaf.unreachable_pairs == 3
    assert leaf.avg_path_stretch == 1
    assert criticality_ranking(topo)[0][0] == "center"
    assert centrality_vs_impact(topo) == pytest.approx(1)
    assert asdict(topo) == original
    assert impact(topo, fail_link(topo, "a", "center")).components == 2
    assert impact(topo, topo).pagerank_shift == 0
    failed = fail_node(topo, "a")
    failed.links[0].cost = 999
    assert asdict(topo) == original


def test_directed_failure_and_load_change():
    topo = Topology(
        ["a", "b", "c"], [Link("a", "b", cost=1), Link("b", "c", cost=1)], directed=True
    )
    assert shortest_path(topo, "c", "a") is None
    result = impact(topo, fail_link(topo, "b", "c"), {("a", "c"): 100})
    assert result.unreachable_pairs == 2
    assert result.max_utilization_change == pytest.approx(0.1)
    assert centrality_vs_impact(Topology([], [])) == 0


def test_spikes_have_top_five_traffic_residuals():
    rng = np.random.default_rng(42)
    series = rng.uniform(10, 20, (200, 1)) @ np.ones((1, 8))
    series += rng.normal(0, 0.01, series.shape)
    spikes = [20, 80, 150]
    for i, row in enumerate(spikes):
        series[row, i] += 20
    errors = traffic_anomaly(series, k=1)
    assert set(spikes) <= set(np.argsort(errors)[-5:])


def test_self_loops_and_empty_topology():
    topo = Topology(["a", "b"], [Link("a", "a"), Link("a", "b")])
    assert betweenness_centrality(topo) == {"a": 0, "b": 0}
    assert sum(pagerank_centrality(topo).values()) == pytest.approx(1)
    assert impact(Topology([], []), Topology([], [])).components == 0
