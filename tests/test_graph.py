import numpy as np
import pytest
from scipy.sparse import csr_matrix

from meridian_rank.graph import NodeIndex, build_adjacency, to_transition


def test_orientation_duplicates_isolates_and_roundtrip():
    index = NodeIndex(["a", "b", "isolated"])
    matrix, result = build_adjacency([("a", "b", 2), ("a", "b", 3), ("b", "b", 4)], index)
    assert matrix[1, 0] == 5
    assert matrix[0, 1] == 0
    assert result.to_id(result.to_int("isolated")) == "isolated"
    np.testing.assert_allclose(to_transition(matrix).sum(axis=0), [[1, 1, 0]])
    with pytest.raises(IndexError):
        index.to_id(-1)


@pytest.mark.parametrize("weight", [-1, np.inf, np.nan])
def test_reject_invalid_weights(weight):
    with pytest.raises(ValueError):
        build_adjacency([("a", "b", weight)])


def test_empty_and_invalid_graphs():
    assert build_adjacency([])[0].shape == (0, 0)
    with pytest.raises(ValueError):
        NodeIndex(["a", "a"])
    with pytest.raises(ValueError):
        to_transition(csr_matrix(np.ones((2, 3))))
