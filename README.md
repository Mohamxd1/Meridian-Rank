# Meridian Rank

A Python engine for tenant-scoped document ranking and network analysis, sharing
one sparse graph and linear algebra core. Version 0.1.0 is implemented locally;
production host deployment and real-tenant evaluation remain external work.

- **Documents:** PageRank, personalized search, TF-IDF/SVD similarity, residual
  anomaly scores, weighted link extraction, and a review outbox.
- **Networks:** topology generators, Dijkstra, distance-vector convergence,
  eigenvector/Brandes/PageRank centrality, demand routing, and failure simulation.
- **Service:** FastAPI, PostgreSQL persistence, Redis caching, explicit tenant
  ownership, and a mounting adapter for Meridian Data's client authentication.

## Quick start

Python 3.11 or later is required. Run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
RANK_DEV_MODE=true python -m uvicorn meridian_rank.api:app --host 127.0.0.1 --port 8000
```

For this cloud workspace, `bash scripts/cloud-install.sh` recreates the tested
environment using [Python 3.12 constraints](constraints/python312-cloud.txt).
Those pins record the validated Linux environment; other Python versions resolve
the compatible dependency ranges in `pyproject.toml`.

This starts an ephemeral in-memory development server. `RANK_DEV_MODE=true`
explicitly enables a UUID `X-Tenant-ID` header and development admin access.
Without it, the standalone server rejects unverified tenant headers. Production
must supply authenticated tenant and admin dependencies; see
[host integration](docs/integration.md).

In another shell with the virtual environment activated:

```bash
python scripts/smoke.py
curl -X PUT http://127.0.0.1:8000/admin/documents \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: 00000000-0000-0000-0000-000000000001' \
  --data-binary @examples/documents.json
curl -X POST http://127.0.0.1:8000/rank/related \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: 00000000-0000-0000-0000-000000000001' \
  -d '{"query":"semiconductor", "limit":10}'
```

`PUT /admin/documents` replaces the entire authenticated tenant's snapshot.
It accepts normalized processed documents, not raw PDF uploads. See
[the example](examples/documents.json) for the complete JSON shape.

## PostgreSQL and Redis

Docker Compose supplies local development services on loopback ports 55432 and
56379. Its PostgreSQL trust authentication is only suitable for local development.

```bash
docker compose up -d --wait
export RANK_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/meridian_rank
export RANK_REDIS_URL=redis://127.0.0.1:56379/0
python -c 'import os; from meridian_rank.store import PostgresStore; PostgresStore(os.environ["RANK_DATABASE_URL"]).initialize()'
RANK_DEV_MODE=true python -m uvicorn meridian_rank.api:app --host 127.0.0.1
```

The schema is packaged in `src/meridian_rank/schema.sql`. Initialization is
explicit and idempotent. Database credentials are read from environment
configuration; no production credentials are included.

Ingestion serializes updates per tenant and stores document/graph revisions
transactionally. Redis keys contain tenant, revision, operation, and parameters,
with a 300-second TTL. More than 5% changes since the last analysis trigger eager
recomputation; smaller changes are analyzed on an uncached read. Results from an
old revision cannot be served as a new one. An unavailable Redis falls back to
computation; an unavailable database returns an error.

## API

Every business route requires tenant context. Admin routes additionally require
verified admin authorization outside explicit local development mode. The API
schema is available at `/openapi.json` and the interactive docs at `/docs`.

| Method | Route | Result |
|---|---|---|
| GET | `/health` | Process liveness and storage adapter name; not a database readiness probe |
| GET | `/rank/top?limit=20` | Global node ranks |
| POST | `/rank/related` | `{"query":"...","limit":10}` → personalized document ranks |
| GET | `/docs/{id}/similar?limit=10` | Cosine neighbors in latent space |
| GET | `/docs/{id}/anomaly` | Score, threshold, tenant, and review decision |
| PUT | `/admin/documents` | Replace one tenant's normalized corpus |
| POST | `/admin/recompute` | Recompute graph/SVD and review events |
| PATCH | `/admin/settings` | `{"anomaly_threshold":0.5}`; null restores the batch rule |
| GET | `/admin/reviews` | Pending anomaly review outbox |
| POST | `/network/topologies` | Generator specification or explicit nodes/links → UUID |
| GET | `/network/{id}/path?src=a&dst=b` | Shortest path and total cost |
| GET | `/network/{id}/convergence` | Synchronous distance-vector rounds |
| GET | `/network/{id}/centrality?kind=betweenness` | Eigenvector, betweenness, or PageRank scores |
| POST | `/network/{id}/fail` | `{"node":"a"}` or `{"link":["a","b"]}` → impact |
| GET | `/network/{id}/critical?limit=5` | Ranked single-node failures |
| POST | `/network/{id}/traffic/load` | `{"demands":[{"src":"a","dst":"b","mbps":10}]}` |
| POST | `/network/{id}/traffic/anomaly` | `{"series":[[1,2],[2,4],[8,1]],"k":1}` |

Generators accept `ring`, `grid`, `scale_free`, and `leaf_spine`. For example:
`{"kind":"leaf_spine","spines":2,"leaves":4,"hosts_per_leaf":4}`.
A custom topology uses `{"nodes":["a","b"],"links":[{"src":"a","dst":"b"}],"directed":false}`.
Ring nodes are `"0"`, `"1"`, …; grid nodes are `"row,col"`; fabric nodes are
`spine0`, `leaf0`, and `host0-0`.

The API bounds requests to 500 topology nodes and 5,000 links; failure simulation
and distance-vector analysis allow 200 nodes, exhaustive criticality 100.
Larger analyses can use the Python library offline. Missing and foreign-owned
resources both return 404. Invalid inputs return 422; no tenant context returns
401; insufficient privileges return 403.

## Mathematical contracts

- Adjacency is `A[dst,src]`; duplicate graph edges sum. Transition columns sum to
  one, with zero dangling columns.
- PageRank redistributes dangling mass before each update. Rank vectors are
  float64 and normalized. PPR uses normalized seed weights for both teleportation
  and dangling redistribution. Uniform seeds reproduce global PageRank.
- Default damping is 0.85, L1 tolerance 1e-10, and iteration limit 200.
  Nonconvergence raises an explicit error instead of returning an inaccurate result.
- Automatic SVD rank targets 90% squared singular energy, capped at 100 and
  `min(shape)-1`, with rank one for singleton dimensions. Small full-rank fits are
  supported explicitly. Cosine similarity excludes the query document.
- Document nodes are namespaced as `filing:`, `company:`, and `entity:` to avoid
  ID collisions. Extraction uses the five edge types and weights in the design.
- Routing and Brandes use positive link costs. Spectral centrality uses unit
  edges. Parallel physical links are rejected; duplicate document edges aggregate.
- Undirected loads aggregate both directions of a physical link. Unroutable
  positive demands raise an error. Failure counts include removed endpoints;
  directed pairs are ordered and undirected pairs unordered.

See the four [math notes](notebooks/01_markov_chains.md), the
[implementation decisions](docs/implementation.md), and the
[archived original design](docs/design.md).

## Validation

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m build
```

Real storage checks require running PostgreSQL and Redis:

```bash
RANK_TEST_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/meridian_rank \
RANK_TEST_REDIS_URL=redis://127.0.0.1:56379/0 \
python -m pytest -q
```

Without those variables, the three integration tests are reported as skipped.
Coverage includes 20 random directed PageRank graphs, 40 weighted directed and
undirected routing/Brandes comparisons, spectral centrality, dangling and isolated
nodes, SVD reconstruction, deterministic generators, traffic spikes, every API
route, cross-tenant denial, real SQL/Redis behavior, and anomaly review delivery.
GitHub Actions defines Python 3.11–3.13 jobs with PostgreSQL/Redis; remote execution
is not implied by a local pass.

## Measured examples

| Metric | Measured result |
|---|---:|
| PageRank, 1M sampled edges / 100K nodes | 0.0581 s |
| Traced peak allocation, sparse construction and PageRank | 28.33 MiB; not total RSS |
| Dijkstra, 10K-node grid | 0.0881 s |
| Weighted Brandes, 100 nodes / NetworkX | 0.0155 s / 0.0338 s |
| Synthetic topic search precision@10, PPR / TF-IDF | 0.50 / 0.10 |
| Held-out synthetic anomaly precision / recall | 0.7692 / 1.0000 |
| Illustrative infrastructure's top three | api, worker, cloudflare |

These are single local runs. The 1M edge sample contains 999,957 unique edges.
The synthetic topic and anomaly results are not real-tenant quality estimates.
Raw CSVs, platform metadata, plots, and reproduction commands are in
[benchmark documentation](docs/benchmarks.md).

## Integration status and limitations

The host adapter was checked against MeridianData's actual API-key guard and
export gate at commit `482abd2e614e38157f7152659e4bb3e8fe167037`.
[Integration instructions](docs/integration.md) explain how to mount the routes,
map verified clients to tenants, normalize processed records, and deliver reviews.
The deployed host has not been changed. Its approval/export pipeline still needs
to call the provided gate and consume the outbox.

The default anomaly threshold is estimated within the corpus. It cannot guarantee
detection of eight of ten injected outliers: a p99 threshold recomputed on the
contaminated batch usually flags about 1%, and unseen vocabulary can be invisible.
Use normal calibration data and independent labels before relying on it for
production approval. Real SEC tenant rankings, real hand-labeled query evaluation,
and a verified deployment topology are not available in this checkout.

The implementation follows research → plan → implement → verify → review practices
from [claude-code-best-practice](https://github.com/shanraisshan/claude-code-best-practice),
with concise [agent instructions](AGENTS.md), Claude configuration, and repeatable
validation. See [release notes](CHANGELOG.md). No remote release has been published.

MIT licensed. Stretch goals include embeddings, incremental PageRank, ECMP,
link-state simulation, HITS, and GPU acceleration.
