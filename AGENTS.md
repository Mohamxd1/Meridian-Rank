# Working on Meridian Rank

Use the existing checkout; cloud tasks are already isolated. Do not create a
worktree unless requested. Read README.md for the mathematical and API contracts.

## Commands

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m build
.venv/bin/python -m uvicorn meridian_rank.api:app
```

## Invariants

- Sparse adjacency uses A[dst, src]; transition columns sum to one or zero.
- Redis keys, database queries, documents, and topologies are tenant scoped.
- NetworkX is a test/benchmark oracle, never a runtime algorithm dependency.
- Random algorithms accept an explicit seed. Never invent benchmark results.
- Keep I/O in store.py and api.py; math functions do not access services.
- Authentication is injected by the host. Never trust a caller-supplied tenant
  header in production. See docs/integration.md.

## Workflow

Research the affected contract, plan a bounded change, implement, then run the
relevant mathematical and API checks. Inspect failures before changing tests.
Record externally blocked acceptance criteria in docs/implementation.md.
Use the full suite before delivery. Keep this file short and factual.
