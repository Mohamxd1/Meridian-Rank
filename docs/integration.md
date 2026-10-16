# Integrating with Meridian Data

The host repository was inspected at commit
`482abd2e614e38157f7152659e4bb3e8fe167037`. It currently exposes
`/clients/{client_id}/...` with `dataforge.auth.verify_api_key`, storing records
with `client_id`, `id`, `raw_text`, and JSON `extracted_fields`. Client IDs are
strings; Meridian Rank's tenant keys are UUIDs. The adapter requires an explicit,
stable mapping and rejects duplicate tenant bindings.

## Mounting the routes

Install this package in the host's environment. Supply the host's real
administrator dependency (the current client API key alone carries no admin
role), a persisted client→tenant mapping, and configured PostgreSQL/Redis:

```python
from dataforge.auth import verify_api_key
from meridian_rank.api import mount_client_routes
from meridian_rank.store import PostgresStore, RedisCache, RankingService

# application, rank_dsn, redis_url, persisted_tenant_ids, and require_rank_admin
# are provided by the host's configuration and authentication layer.
service = RankingService(PostgresStore(rank_dsn), RedisCache(redis_url))
mount_client_routes(
    application,
    service,
    verify_api_key,
    persisted_tenant_ids,
    require_rank_admin,
)
```

This adds `/clients/{client_id}/rank/top`, network routes, and admin routes to the
host OpenAPI schema. Authentication runs before the client ID is mapped. A
caller-provided `X-Tenant-ID` is ignored by this adapter. The mount helper also
registers domain exception handlers; review those handlers if the host already
has custom handlers for `KeyError`, `ValueError`, or `RuntimeError`.

Schema initialization is explicit: call `PostgresStore.initialize()` during a
controlled deployment migration, not on each request. Credentials come from
host configuration. The local Compose stack uses trust authentication on a
loopback-published port and is for development only.

## Normalizing processed documents

Read records through the host's tenant-filtered loaders. Convert each owned
record into `Document(id=row['id'], text=row['raw_text'], ...)`. The optional
`company`, `subsidiaries`, `citations`, `amends`, and `entities` fields accept
already-extracted stable IDs. This package does not guess those IDs from arbitrary
text or invoke an LLM. Pass a complete tenant snapshot to
`service.ingest(tenant_uuid, documents)` or the admin `PUT /admin/documents`
endpoint. Missing documents are deleted from that tenant snapshot.

Ingestion atomically rebuilds typed graph nodes/edges, increments the revision,
and invalidates old responses by revision. Changes exceeding 5% since the last
fit trigger eager recomputation; smaller changes are recomputed on the next
uncached read. Empty corpora and stop-word-only text still support graph ranking.
No count-only heuristic is used: changes to text and metadata are detected.

## Review queue and export gate

`rank_review_queue` is an outbox for the host's existing review queue. Read
`store.reviews(tenant_uuid)`, deliver each item to the host with idempotency key
`(tenant_uuid, document_id, revision)`, and only then call
`store.acknowledge_review(tenant_uuid, document_id, revision)`. Do not acknowledge
failed deliveries. Recomputing a revision can regenerate an event, so host
delivery must be idempotent. Changed data supersedes prior queued evaluations.

For automatic approval, gate the decision **before export**:

```python
from meridian_rank.documents import review_gate

decision = service.query(tenant_uuid, "anomaly", document_id=record_id)
record = review_gate(tenant_uuid, record_id, record, decision)
```

The gate preserves rejection and sets flagged candidates to `pending_review`.
Persist the returned status and matching validation result through the host's
existing save/audit mechanism before calling its exporter. The inspected
`dataforge.pipeline.export.export_record` refuses records whose status is not
`approved`; the compatibility script exercises that actual function.
If evaluation is unavailable, the host must keep the record pending rather than
silently auto-approving it. Production wiring still requires a host change at
the intake/save/export boundary; this repository cannot install that change into
the deployed service by itself.

The default p99 threshold is a within-corpus exploratory score. Calibrate on
normal labeled data and set a tenant threshold through `PATCH /admin/settings`
before relying on it. A threshold change creates a new revision. The library
also exposes low-level LSA projection to evaluate held-out matrices.

## Reproduce compatibility checks

```bash
python scripts/check_meridian_adapter.py /path/to/MeridianData
```

This imports the host's actual API-key guard and export function, uses ephemeral
in-process fixture credentials, checks wrong-key and cross-client denial,
verifies mounted OpenAPI routes, and confirms a flagged record is refused by the
host exporter. It does not modify the host checkout or call external services.
