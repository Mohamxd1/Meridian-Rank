"""Reproduce the numerical examples and figures used in the math notes."""

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

from meridian_rank.graph import to_transition
from meridian_rank.network.failure import criticality_ranking
from meridian_rank.network.routing import bellman_ford_rounds
from meridian_rank.network.topology import load_topology_yaml, make_leaf_spine, make_ring
from meridian_rank.pagerank import pagerank


def main() -> None:
    destination = Path("notebooks/assets")
    destination.mkdir(parents=True, exist_ok=True)
    transition = csr_matrix([[0, 0, 1, 0], [0.5, 0, 0, 0], [0.5, 1, 0, 1], [0, 0, 0, 0]])
    google = 0.85 * transition.toarray() + 0.15 / 4
    eigenvalues, vectors = np.linalg.eig(google)
    eigenvector = vectors[:, np.argmin(abs(eigenvalues - 1))].real
    eigenvector /= eigenvector.sum()
    rank = pagerank(transition)
    assert np.allclose(rank, eigenvector, atol=1e-8)
    results = {
        "four_node_pagerank": rank.tolist(),
        "eigenvector_max_error": float(np.max(abs(rank - eigenvector))),
    }
    rng = np.random.default_rng(42)
    adjacency = csr_matrix(
        (rng.uniform(0.1, 2, 400), (rng.integers(0, 100, 400), rng.integers(0, 100, 400))),
        shape=(100, 100),
    )
    fig, axis = plt.subplots(figsize=(6, 4))
    for damping in [0.5, 0.7, 0.85, 0.95]:
        _, history = pagerank(
            to_transition(adjacency), d=damping, max_iter=2000, return_history=True
        )
        results[f"iterations_d{damping}"] = len(history)
        axis.semilogy(range(1, len(history) + 1), history, label=f"d={damping}")
    axis.set(xlabel="Iteration", ylabel="L1 change")
    axis.legend()
    fig.tight_layout()
    fig.savefig(destination / "pagerank_convergence.png", dpi=150)
    plt.close(fig)
    results["convergence_rounds"] = {
        "ring10": bellman_ford_rounds(make_ring(10)),
        "ring20": bellman_ford_rounds(make_ring(20)),
        "leaf_spine_2_4_4": bellman_ford_rounds(make_leaf_spine(2, 4, 4)),
        "leaf_spine_4_8_4": bellman_ford_rounds(make_leaf_spine(4, 8, 4)),
    }
    topo = load_topology_yaml("examples/meridian_infra.yaml")
    results["illustrative_infrastructure"] = [
        {"node": node, **asdict(damage)} for node, damage in criticality_ranking(topo)
    ]
    (destination / "measurements.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
