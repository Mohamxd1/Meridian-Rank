"""Measure sparse PageRank at 1K–1M sampled edges (duplicates are summed)."""

import argparse
import tracemalloc
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

from meridian_rank.graph import to_transition
from meridian_rank.pagerank import pagerank

from .common import write_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=int, nargs="+", default=[1000, 10000, 100000, 1000000])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = []
    for edges in args.edges:
        rng = np.random.default_rng(args.seed)
        nodes = max(10, edges // 10)
        src, dst = rng.integers(0, nodes, size=(2, edges))
        tracemalloc.start()
        start = perf_counter()
        adjacency = csr_matrix((np.ones(edges), (dst, src)), shape=(nodes, nodes))
        transition = to_transition(adjacency)
        preparation = perf_counter() - start
        start = perf_counter()
        rank, history = pagerank(transition, return_history=True)
        seconds = perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1] / 1024**2
        tracemalloc.stop()
        assert abs(rank.sum() - 1) < 1e-10
        rows.append(
            {
                "sampled_edges": edges,
                "unique_edges": adjacency.nnz,
                "nodes": nodes,
                "prepare_seconds": preparation,
                "pagerank_seconds": seconds,
                "traced_peak_mib": peak,
                "iterations": len(history),
            }
        )
    target = write_results(
        "pagerank_scale",
        rows,
        {
            "seed": args.seed,
            "memory": (
                "tracemalloc peak during sparse construction and iteration; "
                "excludes initial edge arrays; not total RSS"
            ),
            "repetitions": 1,
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].loglog([r["sampled_edges"] for r in rows], [r["pagerank_seconds"] for r in rows], "o-")
    axes[0].set(xlabel="Sampled edges", ylabel="PageRank seconds")
    axes[1].loglog([r["sampled_edges"] for r in rows], [r["traced_peak_mib"] for r in rows], "o-")
    axes[1].set(xlabel="Sampled edges", ylabel="Traced peak MiB")
    fig.tight_layout()
    fig.savefig(target.with_suffix(".png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
