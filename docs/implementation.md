# Implementation plan and evidence

Reference reviewed: shanraisshan/claude-code-best-practice at
`aad783325c24a2602d93da873fee43a94566fbb4`, especially the README planning tips,
`best-practice/claude-memory.md`, and the research/plan/implement workflow.
Applied practices: explicit contracts, phased validation, short reusable project
instructions, repeatable commands, and review against independent numerical oracles.

## Work plan

1. Implement sparse graph, normalized PageRank/PPR, and truncated SVD.
2. Implement topology, routing, centrality, traffic, and failure simulation.
3. Implement document extraction, tenant-scoped storage/cache, and API integration.
4. Exercise mathematical oracles, API isolation, database/cache integration,
   packaging, and a real HTTP request. Add CI and runnable examples.
5. Run reproducible benchmarks and document measured evidence and limitations.

## Decisions that refine the design

- The README's dangling correction snippet loses probability mass. Redistribute
  `d * sum(rank[dangling])` before testing convergence.
- A single-node PPR seed need not have the highest stationary score on every
  graph. Test seed bias and the defining stationary equation instead.
- Use sparse SVD with a bounded dense fallback for tiny/full-rank matrices;
  automatic rank retains up to 90% energy subject to the documented cap.
- Routing cost is positive; betweenness is weighted by cost. Spectral centrality
  uses unweighted edges so bandwidth cost is not misread as relationship strength.
- Physical undirected link loads aggregate both directions. Failed endpoints
  count as unreachable; directed reachability uses ordered pairs.
- Database changes are transactional; cached responses are keyed by tenant data
  revision so old responses cannot be used after a successful ingestion.

## External acceptance criteria

Real tenant rankings, hand-labeled search relevance, production review-queue and
auth wiring, and published CI/release status require the host application and
data. Synthetic evaluation is labeled as such and does not establish those results.

## Delivered implementation

All five repository work stages above are implemented. The original design is
archived in `docs/design.md`; README.md describes the executable application.

| Area | Implemented capability |
|---|---|
| Core | Sparse indexing, column transitions, normalized PageRank/PPR, SVD, cosine similarity, residual thresholds |
| Documents | Five weighted edge types, TF-IDF search seeds, tenant analysis, configurable thresholds, review gate |
| Networks | Four generators, YAML, Dijkstra, distance-vector rounds, Brandes/spectral centrality, loads, failures |
| Persistence | Transactional PostgreSQL graph/documents/scores, revisions, review outbox, tenant-owned topologies |
| Cache | Tenant/revision-scoped JSON responses, TTL 300, bounded in-memory fallback |
| API | Every specified route, ingestion/settings/reviews, auth injection, admin guards, request bounds |
| Integration | Explicit verified client→tenant mapping, mount helper, pre-export review decision |
| Project | Packaging, MIT license, CI matrix, local services, agent guidance, reproducible benchmarks and math notes |

## Executed validation

- **129 tests passed, zero skipped**, with the local PostgreSQL 16 and Redis 7
  containers enabled. This includes the 20 random PageRank comparisons and 40
  weighted directed/undirected Dijkstra and Brandes comparisons.
- Ruff lint and formatting checks passed. `pip check` found no broken requirements.
- Wheel and source archive built. The wheel was installed in a separate clean
  virtual environment and exercised from `/tmp`, checking packaged SQL, document
  analysis, routing, and OpenAPI without relying on the editable checkout.
- A real Uvicorn process passed the HTTP smoke script, including successful
  operations, missing-auth rejection, and cross-tenant denial. Restarting it
  and repeating the smoke check also passed.
- The compatibility script imported MeridianData's real `verify_api_key` and
  `export_record`, checked mounted routes and client isolation, and confirmed an
  anomalous candidate cannot be exported as approved. The host checkout was not changed.
- Benchmarks and math artifact scripts ran; measured CSVs, metadata, and plots
  are stored in the repository. See `docs/benchmarks.md` for their limited scope.
- `scripts/cloud-install.sh` ran successfully on the existing environment using
  the recorded Python 3.12 constraints. The cloud draft's `install_script` and
  `start_skill` fields were updated and confirmed saved; publication is separate.

The installed Starlette version emits an upstream deprecation notice for httpx
TestClient compatibility. All checks pass; no warning or assertion was suppressed.
The host's model also emits its existing Pydantic field-name warning during the
compatibility check. Local validation used Python 3.12; the 3.11/3.13 CI jobs have
been configured but not executed on GitHub here.

## Work requiring production context

Deploying the mount helper requires the host's persisted tenant mapping and admin
authorization policy. Its processing pipeline must call the review gate before
export and consume the outbox idempotently. Real tenant data, approved relevance
labels, and an actual infrastructure inventory are required for the original
real-data acceptance criteria. The synthetic fixtures do not substitute for
those inputs. Release publication and environment publication were not performed.
