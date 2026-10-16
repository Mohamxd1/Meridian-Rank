> Archived original design, retained for traceability. Current behavior and validation are documented in README.md and implementation.md. The PageRank snippet below is historical; the implementation corrects its dangling-mass formula.

# Meridian-Rank
A graph and linear algebra engine for [Meridian Data](https://github.com/Mohamxd1/MeridianData), a multi-tenant AI document processing platform.

It has two parts that share one math core:

- **Part A: Document Intelligence.** PageRank, personalized PageRank, and truncated SVD rank documents, power search, find similar documents, and flag anomalous ones for human review.
- **Part B: Network Engineering Module.** The same graph and matrix machinery applied to network topologies: shortest-path routing, convergence, centrality, traffic modeling, failure simulation, and traffic anomaly detection. It also models Meridian's own infrastructure as a topology to find its critical components.

> Status: in development. Benchmark figures are `TBD` until measured. Never publish numbers you have not run.

---

## Table of Contents

1. [Why this exists](#1-why-this-exists)
2. [How the two parts connect](#2-how-the-two-parts-connect)
3. [Conventions (read first)](#3-conventions-read-first)
4. [Default decisions](#4-default-decisions)
5. [Tech stack](#5-tech-stack)
6. [Project structure](#6-project-structure)
7. [Getting started](#7-getting-started)
8. [Module specifications](#8-module-specifications)
9. [Data schemas](#9-data-schemas)
10. [Build guide (step by step, with acceptance criteria)](#10-build-guide-step-by-step-with-acceptance-criteria)
11. [API reference](#11-api-reference)
12. [Integration with Meridian Data](#12-integration-with-meridian-data)
13. [Testing](#13-testing)
14. [Benchmarks and evaluation](#14-benchmarks-and-evaluation)
15. [Roadmap and stretch goals](#15-roadmap-and-stretch-goals)
16. [License](#16-license)

---

## 1. Why this exists

Meridian extracts structured data from documents such as SEC filings. Extraction alone does not answer "which documents matter most?", "which results look unusual?", or "which parts of our infrastructure can we least afford to lose?". Meridian Rank answers all three with linear algebra:

| Question | Technique | Part |
|---|---|---|
| Which documents and entities matter most? | PageRank | A |
| Which documents are relevant to this query? | Personalized PageRank | A |
| Which extraction looks unlike this tenant's normal documents? | Truncated SVD reconstruction error | A |
| What is the best route between two nodes, and how fast do routes converge? | Dijkstra, Bellman-Ford rounds | B |
| Which nodes and links are critical? | Centrality plus failure simulation | B |
| Where does traffic concentrate? | Markov chain flow, link load | B |
| Is this traffic pattern abnormal? | SVD on traffic matrices | B |

## 2. How the two parts connect

```
                +---------------------------+
                |        core math          |
                |  graph.py  pagerank.py    |
                |  ppr.py    lsa.py         |
                +------+-------------+------+
                       |             |
          +------------v--+       +--v----------------+
          |   Part A      |       |   Part B          |
          |  documents    |       |  network/         |
          |  (tenants)    |       |  topologies       |
          +-------+-------+       +---------+---------+
                  |                         |
                  +-----------+-------------+
                              |
                       FastAPI router
                      (mounted in Meridian)
```

Shared pieces: the sparse matrix builder, `pagerank()`, and the SVD routine (`lsa.reconstruction_error`) are reused directly by Part B. Part B adds nothing to the core; it imports it.

## 3. Conventions (read first)

Every module must follow these so the pieces fit together.

- **Node indexing.** Nodes are integers `0..n-1`. A `NodeIndex` maps external IDs (strings) to ints and back.
- **Adjacency matrix.** `A` is `scipy.sparse.csr_matrix` of shape `(n, n)` with `A[dst, src] = weight` for an edge `src -> dst`. Column `j` therefore holds node `j`'s outgoing weights.
- **Transition matrix.** `P = A @ D^-1`, where `D` is the diagonal of column sums. `P` is column-stochastic: each non-dangling column sums to 1.
- **Dangling nodes.** Columns that sum to 0 get their mass spread uniformly during iteration (see the pagerank snippet).
- **Rank vectors.** 1-D `numpy.ndarray`, float64, summing to 1.
- **Determinism.** Every random function takes `seed: int`.
- **Tenant scoping.** Part A functions receive a `tenant_id` and never query without it.
- **Type hints and docstrings** on every public function; each docstring states the math in one line.
- **No hidden globals.** Functions are pure where possible; I/O lives in `store.py` and `api.py` only.

## 4. Default decisions

These were left open earlier. Use these defaults unless changed here.

| Decision | Default |
|---|---|
| Text representation for SVD | TF-IDF (`sklearn.feature_extraction.text.TfidfVectorizer`, `max_features=20000`, English stop words). Embeddings are a stretch goal. |
| SVD rank `k` | Smallest `k` retaining 90% of squared singular value energy, capped at `min(100, min(X.shape) - 1)` |
| Anomaly threshold | Reconstruction error above the tenant's 99th percentile, or mean + 3 standard deviations, whichever is lower. Configurable per tenant. |
| Damping factor | 0.85 |
| PageRank tolerance | L1 change below 1e-10, max 200 iterations |
| PPR seed count | Top 5 text matches, weights proportional to match score |
| Edge types and weights | `filing_company`=1.0, `company_subsidiary`=1.0, `filing_citation`=0.7, `filing_amendment`=1.0, `entity_cooccurrence`=0.3 |
| Cache TTL | 300 seconds |
| OSPF reference bandwidth | 100,000 Mbps (link cost = ref_bw / link_bw, minimum 1) |
| Recompute trigger | More than 5% of a tenant's documents changed since last computation |
| Python version | 3.11+ |

## 5. Tech stack

- **Language:** Python 3.11+
- **Numerics:** NumPy, SciPy (`scipy.sparse`, `scipy.sparse.linalg`)
- **ML utilities:** scikit-learn (TF-IDF)
- **Verification only:** NetworkX
- **Service:** FastAPI, Uvicorn
- **Storage and cache:** PostgreSQL, Redis
- **Testing:** pytest
- **Plots:** Matplotlib
- **Config files:** PyYAML (for infrastructure topologies)

## 6. Project structure

```
meridian-rank/
  src/meridian_rank/
    __init__.py
    graph.py            # NodeIndex, adjacency and transition matrices
    pagerank.py         # power iteration
    ppr.py              # personalized PageRank
    lsa.py              # truncated SVD, similarity, reconstruction error
    store.py            # PostgreSQL loaders, Redis cache
    api.py              # FastAPI router (Parts A and B)
    network/
      __init__.py
      topology.py       # Topology dataclass and generators
      routing.py        # Dijkstra, Bellman-Ford convergence
      centrality.py     # eigenvector centrality, betweenness (Brandes)
      traffic.py        # demand matrices, link load, Markov flow, anomaly
      failure.py        # failure simulation and criticality ranking
  examples/
    meridian_infra.yaml # Meridian's own services as a topology
  tests/
    test_graph.py
    test_pagerank.py
    test_ppr.py
    test_lsa.py
    test_tenant_isolation.py
    network/
      test_topology.py
      test_routing.py
      test_centrality.py
      test_traffic.py
      test_failure.py
  notebooks/
    01_markov_chains.md
    02_pagerank_convergence.md
    03_svd_anomaly.md
    04_network_graphs.md
  benchmarks/
    bench_pagerank_scale.py
    bench_search_quality.py
    bench_anomaly.py
    bench_network.py
  pyproject.toml
  README.md
```

## 7. Getting started

```bash
git clone https://github.com/<your-username>/meridian-rank.git
cd meridian-rank
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
uvicorn meridian_rank.api:app --reload
```

`pyproject.toml` dependencies: `numpy`, `scipy`, `scikit-learn`, `fastapi`, `uvicorn`, `redis`, `psycopg[binary]`, `pyyaml`. Dev extras: `networkx`, `pytest`, `matplotlib`, `httpx`.

## 8. Module specifications

Signatures are the contract. Implement exactly these so tests and the API line up.

### 8.1 Core

**`graph.py`**

```python
@dataclass
class NodeIndex:
    ids: list[str]

    def to_int(self, node_id: str) -> int: ...
    def to_id(self, i: int) -> str: ...


def build_adjacency(
    edges: list[tuple[str, str, float]], index: NodeIndex | None = None
) -> tuple[csr_matrix, NodeIndex]:
    """A[dst, src] = weight. Duplicate edges are summed."""


def to_transition(A: csr_matrix) -> csr_matrix:
    """P = A @ D^-1. Dangling columns stay all zero (handled in pagerank)."""
```

**`pagerank.py`**

```python
def pagerank(
    P: csr_matrix,
    d: float = 0.85,
    tol: float = 1e-10,
    max_iter: int = 200,
    return_history: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[float]]:
    """r_{t+1} = d * P r_t + (1 - d)/n, with dangling mass redistributed uniformly."""
```

Reference implementation:

```python
def pagerank(P, d=0.85, tol=1e-10, max_iter=200, return_history=False):
    n = P.shape[0]
    r = np.full(n, 1 / n)
    history = []
    for _ in range(max_iter):
        r_new = d * (P @ r) + (1 - d) / n
        r_new += d * (1 - r_new.sum()) / n  # dangling-mass correction
        err = np.abs(r_new - r).sum()
        history.append(err)
        r = r_new
        if err < tol:
            break
    return (r, history) if return_history else r
```

**`ppr.py`**

```python
def personalized_pagerank(
    P: csr_matrix, seeds: dict[int, float], d: float = 0.85, tol: float = 1e-10, max_iter: int = 200
) -> np.ndarray:
    """r = d * P r + (1 - d) * s, where s is the normalized seed vector."""
```

Invariant: with a uniform seed vector the result equals `pagerank()`.

**`lsa.py`**

```python
@dataclass
class LSAModel:
    U: np.ndarray; S: np.ndarray; Vt: np.ndarray; k: int

def fit_lsa(X: csr_matrix, k: int | None = None, energy: float = 0.90, seed: int = 0) -> LSAModel
def doc_vectors(model: LSAModel) -> np.ndarray              # U_k * S_k
def similar(model: LSAModel, doc_idx: int, limit: int = 10) -> list[tuple[int, float]]   # cosine
def reconstruction_error(model: LSAModel, X: csr_matrix) -> np.ndarray   # ||x - x V V^T|| per row
def anomaly_flags(errors: np.ndarray, rule: str = "p99_or_3sigma") -> np.ndarray   # bool
```

### 8.2 Network module (Part B)

**`network/topology.py`**

```python
@dataclass
class Link:
    src: str; dst: str
    bandwidth_mbps: float = 1000.0
    latency_ms: float = 1.0
    cost: float | None = None        # defaults to max(1, ref_bw / bandwidth_mbps)

@dataclass
class Topology:
    nodes: list[str]
    links: list[Link]                # links are bidirectional unless directed=True
    directed: bool = False
    def adjacency(self, weight: str = "cost") -> tuple[csr_matrix, NodeIndex]: ...

def make_ring(n: int, **link_kwargs) -> Topology
def make_grid(rows: int, cols: int, **link_kwargs) -> Topology
def make_scale_free(n: int, m: int = 2, seed: int = 0, **link_kwargs) -> Topology   # Barabasi-Albert
def make_leaf_spine(spines: int, leaves: int, hosts_per_leaf: int = 4, **link_kwargs) -> Topology
def load_topology_yaml(path: str) -> Topology
```

**`network/routing.py`**

```python
def dijkstra(topo: Topology, source: str) -> tuple[dict[str, float], dict[str, str | None]]:
    """From scratch with heapq. Returns (distance, predecessor)."""

def shortest_path(topo: Topology, src: str, dst: str) -> tuple[list[str], float] | None

def bellman_ford_rounds(topo: Topology) -> int:
    """Simulate distance-vector routing: each round every node merges neighbors' tables.
    Returns the number of rounds until no table changes (convergence time)."""

def next_hop_table(topo: Topology) -> dict[str, dict[str, str]]
```

**`network/centrality.py`**

```python
def eigenvector_centrality(topo: Topology, tol: float = 1e-10, max_iter: int = 500) -> dict[str, float]
    """Power iteration on the symmetric adjacency matrix."""

def betweenness_centrality(topo: Topology, normalized: bool = True) -> dict[str, float]
    """Brandes' algorithm, from scratch."""

def pagerank_centrality(topo: Topology, d: float = 0.85) -> dict[str, float]
    """Reuses core pagerank()."""
```

**`network/traffic.py`**

```python
def link_load(topo: Topology, demands: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    """Route each demand (Mbps) along its shortest path; sum per link."""

def utilization(topo: Topology, load: dict) -> dict[tuple[str, str], float]   # load / bandwidth

def markov_flow(topo: Topology, d: float = 0.85) -> dict[str, float]:
    """Stationary distribution of a random walk: where random traffic accumulates."""

def traffic_anomaly(series: np.ndarray, k: int | None = None) -> np.ndarray:
    """series has shape (timesteps, flows). Fit LSA on the series and return the
    per-timestep reconstruction error. Reuses lsa.fit_lsa and lsa.reconstruction_error."""
```

**`network/failure.py`**

```python
@dataclass
class Impact:
    components: int
    unreachable_pairs: int
    avg_path_stretch: float          # mean(new_dist / old_dist) over still-reachable pairs
    max_utilization_change: float
    pagerank_shift: float            # L1 distance between PageRank before and after

def fail_node(topo: Topology, node: str) -> Topology
def fail_link(topo: Topology, src: str, dst: str) -> Topology
def impact(topo: Topology, failed: Topology, demands: dict | None = None) -> Impact
def criticality_ranking(topo: Topology, demands: dict | None = None) -> list[tuple[str, Impact]]
    """Fail each node in turn and sort by damage (unreachable_pairs, then pagerank_shift)."""
def centrality_vs_impact(topo: Topology) -> float
    """Spearman correlation between centrality rank and failure-impact rank."""
```

## 9. Data schemas

```sql
-- Part A: document graph (per tenant)
CREATE TABLE rank_nodes (
    tenant_id   UUID NOT NULL,
    node_id     TEXT NOT NULL,
    node_type   TEXT NOT NULL,           -- filing | company | entity
    PRIMARY KEY (tenant_id, node_id)
);

CREATE TABLE rank_edges (
    tenant_id   UUID NOT NULL,
    src         TEXT NOT NULL,
    dst         TEXT NOT NULL,
    weight      DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    edge_type   TEXT NOT NULL,
    PRIMARY KEY (tenant_id, src, dst, edge_type)
);
CREATE INDEX rank_edges_tenant ON rank_edges (tenant_id);

CREATE TABLE rank_scores (
    tenant_id   UUID NOT NULL,
    node_id     TEXT NOT NULL,
    pagerank    DOUBLE PRECISION,
    anomaly     DOUBLE PRECISION,
    computed_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (tenant_id, node_id)
);

-- Part B: topologies (stored as JSON, small)
CREATE TABLE network_topologies (
    id          UUID PRIMARY KEY,
    tenant_id   UUID NOT NULL,
    name        TEXT NOT NULL,
    spec        JSONB NOT NULL,          -- nodes, links, directed
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Example `examples/meridian_infra.yaml` (Meridian's own services as a topology):

```yaml
name: meridian-infra
directed: false
nodes: [client, cloudflare, api, worker, postgres, redis, r2, openai, anthropic, stripe, resend, sentry]
links:
  - {src: client,     dst: cloudflare, bandwidth_mbps: 1000, latency_ms: 20}
  - {src: cloudflare, dst: api,        bandwidth_mbps: 1000, latency_ms: 5}
  - {src: api,        dst: postgres,   bandwidth_mbps: 10000, latency_ms: 1}
  - {src: api,        dst: redis,      bandwidth_mbps: 10000, latency_ms: 1}
  - {src: api,        dst: worker,     bandwidth_mbps: 10000, latency_ms: 1}
  - {src: worker,     dst: postgres,   bandwidth_mbps: 10000, latency_ms: 1}
  - {src: worker,     dst: r2,         bandwidth_mbps: 1000, latency_ms: 10}
  - {src: worker,     dst: openai,     bandwidth_mbps: 1000, latency_ms: 80}
  - {src: worker,     dst: anthropic,  bandwidth_mbps: 1000, latency_ms: 80}
  - {src: api,        dst: stripe,     bandwidth_mbps: 1000, latency_ms: 60}
  - {src: api,        dst: resend,     bandwidth_mbps: 1000, latency_ms: 60}
  - {src: api,        dst: sentry,     bandwidth_mbps: 1000, latency_ms: 60}
```

Adjust this to match Meridian's real architecture before using it in results.

## 10. Build guide (step by step, with acceptance criteria)

Estimated time: 6 to 8 weeks at a few hours per week. Each step lists a done-when condition. Do the steps in order; later steps depend on earlier ones.

### Phase 0: Setup

- [ ] **1.** Create the repo with the layout from section 6 and a `pyproject.toml` (installable with `pip install -e ".[dev]"`).
  *Done when:* `pytest` runs and reports zero tests collected without errors.
- [ ] **2.** Add CI (GitHub Actions) that runs `pytest` on push.
  *Done when:* the badge is green on an empty test.

### Phase 1: Learn the math

- [ ] **3.** Write `notebooks/01_markov_chains.md` covering stochastic matrices, stationary distribution as the eigenvalue-1 eigenvector, ergodicity and the damping factor, and why convergence speed depends on the second eigenvalue.
  *Done when:* it includes a hand-worked 4-node example.
- [ ] **4.** Reproduce that 4-node example in NumPy with a dense matrix and confirm it matches `numpy.linalg.eig`.
  *Done when:* the eigenvector and your power-iteration result agree to 1e-8.

### Phase 2: Core engine

- [ ] **5.** Implement `graph.py` (`NodeIndex`, `build_adjacency`, `to_transition`).
  *Done when:* `test_graph.py` passes: duplicates are summed, non-dangling columns sum to 1, index round-trips.
- [ ] **6.** Implement `pagerank.py` per section 8.1.
  *Done when:* results match `networkx.pagerank` within 1e-6 on 20 random graphs, including graphs with dangling nodes and disconnected components.
- [ ] **7.** Plot convergence error vs. iteration for `d` in {0.5, 0.7, 0.85, 0.95} and save to `notebooks/02_pagerank_convergence.md`.
  *Done when:* the plot shows slower convergence as `d` approaches 1.
- [ ] **8.** Implement `ppr.py`.
  *Done when:* a uniform seed reproduces `pagerank()` and a single-node seed ranks that node highest.
- [ ] **9.** Implement `lsa.py`.
  *Done when:* `test_lsa.py` passes: reconstruction error is near zero at full rank, decreases as `k` grows, and similar documents score higher than unrelated ones on a toy corpus.

### Phase 3: Part A on real data

- [ ] **10.** Create the schema from section 9 and write `store.py` loaders that fetch one tenant's edges and return `(A, NodeIndex)`.
  *Done when:* loading tenant A never returns tenant B's rows (`test_tenant_isolation.py`).
- [ ] **11.** Build an edge-extraction job from Meridian's processed documents using the edge types and weights in section 4.
  *Done when:* a sample of SEC filings produces a connected graph you can inspect.
- [ ] **12.** Run PageRank on a real tenant graph and review the top 20.
  *Done when:* you can explain in a sentence why each of the top 5 ranks highly.
- [ ] **13.** Add PPR search: take the top-5 text matches as seeds and return the PPR ranking.
  *Done when:* for 10 hand-picked queries, PPR returns at least one relevant document that plain keyword search missed.
- [ ] **14.** Build the TF-IDF matrix per tenant, fit LSA, and compute similarity and anomaly scores.
  *Done when:* injecting 10 unrelated documents into a tenant makes at least 8 of them exceed the anomaly threshold.
- [ ] **15.** Add Redis caching (tenant-scoped keys, 300 second TTL) and a recompute job triggered at 5% document change.
  *Done when:* a repeated query is served from cache and a data change invalidates it.

### Phase 4: Part B, network module

- [ ] **16.** Implement `network/topology.py`: the dataclasses, the four generators, and `load_topology_yaml`.
  *Done when:* generators produce the expected node and link counts (ring of n has n links, grid has `rows*(cols-1) + cols*(rows-1)`, leaf-spine is fully meshed between layers), and the Meridian YAML loads.
- [ ] **17.** Implement `network/routing.py`: `dijkstra`, `shortest_path`, `next_hop_table`.
  *Done when:* distances match `networkx.single_source_dijkstra` on 20 random weighted topologies.
- [ ] **18.** Implement `bellman_ford_rounds`.
  *Done when:* convergence rounds on a ring of n nodes is about n/2, and on a leaf-spine fabric is small and constant. Record both in `notebooks/04_network_graphs.md`.
- [ ] **19.** Implement `network/centrality.py`: eigenvector centrality (power iteration), betweenness (Brandes), and the PageRank wrapper.
  *Done when:* eigenvector and betweenness match NetworkX within 1e-6.
- [ ] **20.** Implement `network/traffic.py`: `link_load`, `utilization`, `markov_flow`.
  *Done when:* total load on a path topology equals the sum of demands times path length, and `markov_flow` sums to 1.
- [ ] **21.** Implement `traffic_anomaly` using the core LSA code on a `(timesteps, flows)` matrix.
  *Done when:* on synthetic traffic with 3 injected spikes, all 3 timesteps rank in the top 5 by reconstruction error.
- [ ] **22.** Implement `network/failure.py`: `fail_node`, `fail_link`, `impact`, `criticality_ranking`, `centrality_vs_impact`.
  *Done when:* failing the center of a star topology yields maximal unreachable pairs and the top rank; failing a leaf does not partition the network.
- [ ] **23.** Model Meridian's own infrastructure from `examples/meridian_infra.yaml` and run the criticality ranking.
  *Done when:* you have a ranked list of Meridian's most critical components and can explain it (for example, why `api` or `postgres` ranks where it does). Add this to the README as a real result.

### Phase 5: Service and integration

- [ ] **24.** Build `api.py` with the routes in section 11 and tenant context on every route.
  *Done when:* each route has an `httpx` test and returns a 4xx without tenant context.
- [ ] **25.** Route documents above the anomaly threshold into Meridian's review queue instead of auto-approval.
  *Done when:* an injected outlier lands in the queue in an end-to-end test.
- [ ] **26.** Mount the router inside the Meridian FastAPI app.
  *Done when:* the routes appear in Meridian's OpenAPI docs and respect its auth.

### Phase 6: Benchmarks and evaluation

- [ ] **27.** `bench_pagerank_scale.py`: runtime and memory from 1K to 1M edges.
  *Done when:* results are saved as a CSV and a plot.
- [ ] **28.** `bench_search_quality.py`: compare PPR against a keyword baseline using precision@10 on at least 20 hand-labeled queries.
  *Done when:* the table in section 14 is filled with measured values.
- [ ] **29.** `bench_anomaly.py`: precision and recall at your threshold on a labeled normal/odd set.
  *Done when:* the table is filled.
- [ ] **30.** `bench_network.py`: Dijkstra, betweenness, and failure-ranking runtime vs. topology size, plus your own implementations vs. NetworkX.
  *Done when:* the table is filled.

### Phase 7: Write-up and release

- [ ] **31.** Fill every `TBD` in this README with measured numbers and add the charts.
  *Done when:* no `TBD` remains.
- [ ] **32.** Add the resume bullets using only real figures, for example: "Built a graph ranking engine (PageRank, personalized PageRank, truncated SVD) for a multi-tenant document platform, plus a network module (Dijkstra, Brandes betweenness, failure simulation) that identified critical components in the platform's infrastructure. Scaled to N edges in X seconds; flagged Y% of injected outliers."
- [ ] **33.** Tag `v0.1.0` and write release notes.

## 11. API reference

All routes require tenant context (for example, a tenant ID resolved from auth) and return JSON.

### Part A

| Method | Route | Description |
|---|---|---|
| `GET` | `/rank/top?limit=20` | Top nodes by global PageRank |
| `POST` | `/rank/related` | Body `{"query": "...", "limit": 10}`. Personalized PageRank results |
| `GET` | `/docs/{id}/similar?limit=10` | Nearest documents in SVD space |
| `GET` | `/docs/{id}/anomaly` | Reconstruction-error score and a review flag |
| `POST` | `/admin/recompute` | Trigger a graph and SVD refresh |

### Part B

| Method | Route | Description |
|---|---|---|
| `POST` | `/network/topologies` | Body: a generator spec (`{"kind": "leaf_spine", "spines": 2, "leaves": 4}`) or a full topology. Returns an `id` |
| `GET` | `/network/{id}/path?src=a&dst=b` | Shortest path and total cost |
| `GET` | `/network/{id}/convergence` | Distance-vector convergence rounds |
| `GET` | `/network/{id}/centrality?kind=betweenness` | Centrality scores (`eigenvector`, `betweenness`, `pagerank`) |
| `POST` | `/network/{id}/fail` | Body `{"node": "api"}` or `{"link": ["a","b"]}`. Returns an `Impact` |
| `GET` | `/network/{id}/critical?limit=5` | Criticality ranking |
| `POST` | `/network/{id}/traffic/load` | Body: demands. Returns link load and utilization |
| `POST` | `/network/{id}/traffic/anomaly` | Body: a `(timesteps, flows)` series. Returns per-timestep error and flags |

Example:

```bash
curl -X POST http://localhost:8000/rank/related \
  -H "Content-Type: application/json" \
  -d '{"query": "supply chain risk factors", "limit": 10}'
```

## 12. Integration with Meridian Data

- Meridian writes entity links and citations to `rank_edges` as documents are processed.
- Meridian Rank reads per-tenant data and exposes results through its router, mounted in the Meridian FastAPI app.
- The anomaly flag feeds the existing review queue.
- Redis is shared, with tenant-scoped keys.
- Part B can be pointed at Meridian's infrastructure definition to produce a living view of its critical components.

## 13. Testing

```bash
pytest                         # everything
pytest tests/network -v        # network module only
```

Required coverage:

- PageRank, eigenvector centrality, betweenness, and Dijkstra all match NetworkX within tolerance
- Dangling nodes, disconnected graphs, and self-loops behave correctly
- PPR with a uniform seed equals standard PageRank
- SVD reconstruction error is near zero in-distribution and high for injected outliers
- Failure simulation: star center is critical, leaf is not
- Tenant isolation holds across every route
- Determinism: the same seed gives identical output

## 14. Benchmarks and evaluation

| Metric | Result |
|---|---|
| PageRank runtime at 1M edges | TBD |
| Memory at 1M edges | TBD |
| Precision@10, PPR vs. keyword baseline | TBD |
| Anomaly precision / recall | TBD |
| Dijkstra runtime, 10K-node topology | TBD |
| Betweenness runtime vs. NetworkX | TBD |
| Meridian infra: top 3 critical components | TBD |

## 15. Roadmap and stretch goals

- Compare against Google Cloud options (Vertex AI embeddings, BigQuery) and document the trade-offs
- Embeddings instead of TF-IDF for the SVD input
- Randomized SVD or sparse GPU acceleration for larger graphs
- HITS (hubs and authorities) as another eigenvector-based ranking
- Incremental PageRank updates instead of full recomputation
- ECMP (equal-cost multipath) load splitting in `link_load`
- Link-state simulation (flooding and SPF recomputation) alongside distance-vector

## 16. License

MIT (update to your preference).
