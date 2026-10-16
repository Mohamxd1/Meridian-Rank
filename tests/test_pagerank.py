import networkx as nx
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from meridian_rank.graph import NodeIndex, build_adjacency, to_transition
from meridian_rank.pagerank import pagerank
from meridian_rank.ppr import personalized_pagerank


@pytest.mark.parametrize("seed", range(20))
def test_pagerank_matches_networkx_with_dangling_and_disconnected_nodes(seed):
    graph = nx.gnp_random_graph(20, 0.08, seed=seed, directed=True)
    graph.add_nodes_from([20, 21])
    graph.add_edge(0, 0)
    index = NodeIndex([str(i) for i in graph.nodes])
    matrix, _ = build_adjacency([(str(a), str(b), 1) for a, b in graph.edges], index)
    transition = to_transition(matrix)
    actual, history = pagerank(transition, return_history=True)
    expected = nx.pagerank(graph, tol=1e-13, max_iter=1000)
    np.testing.assert_allclose(actual, list(expected.values()), atol=1e-8)
    assert actual.sum() == pytest.approx(1, abs=1e-12)
    assert history[-1] < 1e-10
    uniform = personalized_pagerank(transition, dict.fromkeys(range(22), 1))
    np.testing.assert_allclose(uniform, actual, atol=1e-12)
    seeded = personalized_pagerank(transition, {0: 1, 1: 2})
    oracle = nx.pagerank(graph, personalization={0: 1, 1: 2}, tol=1e-13, max_iter=1000)
    np.testing.assert_allclose(seeded, list(oracle.values()), atol=1e-8)


def test_empty_isolates_nonconvergence_and_bad_inputs():
    assert pagerank(csr_matrix((0, 0))).size == 0
    np.testing.assert_allclose(pagerank(csr_matrix((3, 3))), [1 / 3] * 3)
    np.testing.assert_allclose(personalized_pagerank(csr_matrix((3, 3)), {0: 1}), [1, 0, 0])
    with pytest.raises(ValueError):
        pagerank(csr_matrix([[2.0]]))
    with pytest.raises(ValueError):
        pagerank(csr_matrix((1, 1)), d=1)
    with pytest.raises(ValueError):
        personalized_pagerank(csr_matrix((3, 3)), {})
    with pytest.raises(ValueError):
        personalized_pagerank(csr_matrix((3, 3)), {3: 1})
    with pytest.raises(RuntimeError):
        pagerank(to_transition(csr_matrix([[0, 1], [0, 0]])), max_iter=1)


def test_four_node_stationary_eigenvector():
    transition = np.array([[0, 0, 1, 0], [0.5, 0, 0, 0], [0.5, 1, 0, 1], [0, 0, 0, 0]])
    google = 0.85 * transition + 0.15 / 4
    values, vectors = np.linalg.eig(google)
    vector = vectors[:, np.argmin(abs(values - 1))].real
    vector /= vector.sum()
    np.testing.assert_allclose(pagerank(csr_matrix(transition)), vector, atol=1e-8)
