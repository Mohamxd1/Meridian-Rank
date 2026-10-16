"""I/O boundaries for tenant data, PostgreSQL, Redis, and analysis orchestration."""

import json
import math
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Protocol
from uuid import UUID, uuid4

import psycopg
import redis
import yaml
from psycopg.types.json import Jsonb
from scipy.sparse import csr_matrix

from .documents import Document, DocumentAnalysis, analyze_documents, extract_graph
from .graph import NodeIndex, build_adjacency
from .network.topology import Link, Topology


def _tenant(value: str) -> str:
    return str(UUID(str(value)))


def _changed_fraction(previous: list[dict], current: list[dict]) -> float:
    old = {doc["id"]: doc for doc in previous}
    new = {doc["id"]: doc for doc in current}
    changed = sum(old.get(key) != new.get(key) for key in old.keys() | new.keys())
    return changed / max(len(old), 1)


def read_topology_yaml(path: str) -> Topology:
    """Read a YAML graph with safe_load; adjacency uses column-source orientation."""
    with Path(path).open(encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    return topology_from_dict(spec)


def topology_from_dict(spec: dict) -> Topology:
    """Construct a validated topology from its nodes, links, and direction flag."""
    if not isinstance(spec, dict) or not isinstance(spec.get("directed", False), bool):
        raise ValueError("topology must be a mapping with a boolean directed flag")
    return Topology(
        spec["nodes"], [Link(**link) for link in spec["links"]], spec.get("directed", False)
    )


@dataclass
class Snapshot:
    """One tenant's coherent revision and processed documents."""

    revision: int
    documents: list[Document]
    threshold: float | None = None


class Store(Protocol):
    """Tenant-keyed persistence contract shared by memory and PostgreSQL adapters."""

    def snapshot(self, tenant_id: str) -> Snapshot: ...
    def replace_documents(self, tenant_id: str, documents: list[Document]) -> tuple[int, float]: ...
    def save_analysis(self, tenant_id: str, revision: int, analysis: DocumentAnalysis) -> bool: ...
    def save_topology(self, tenant_id: str, topology: Topology, name: str = "") -> str: ...
    def get_topology(self, tenant_id: str, topology_id: str) -> Topology: ...
    def reviews(self, tenant_id: str) -> list[dict]: ...
    def set_threshold(self, tenant_id: str, threshold: float | None) -> None: ...


class MemoryStore:
    """Thread-safe ephemeral store with isolated tenant snapshots."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._data: dict[str, Snapshot] = {}
        self._computed: dict[str, list[dict]] = {}
        self._reviews: dict[str, list[dict]] = {}
        self._topologies: dict[tuple[str, str], Topology] = {}

    def snapshot(self, tenant_id: str) -> Snapshot:
        """Read a coherent copy of a tenant's document revision."""
        with self._lock:
            return deepcopy(self._data.get(_tenant(tenant_id), Snapshot(0, [])))

    def set_threshold(self, tenant_id: str, threshold: float | None) -> None:
        """Change the tenant's calibrated threshold and invalidate the prior revision."""
        if threshold is not None and (not math.isfinite(threshold) or threshold < 0):
            raise ValueError("threshold must be finite and nonnegative")
        tenant_id = _tenant(tenant_id)
        with self._lock:
            snapshot = self.snapshot(tenant_id)
            if snapshot.threshold != threshold:
                snapshot.revision += 1
                snapshot.threshold = threshold
                self._data[tenant_id] = snapshot
                self._reviews.pop(tenant_id, None)

    def replace_documents(self, tenant_id: str, documents: list[Document]) -> tuple[int, float]:
        """Atomically replace one tenant's corpus and increment revision on change."""
        tenant_id = _tenant(tenant_id)
        extract_graph(tenant_id, documents)
        documents = sorted(documents, key=lambda doc: doc.id)
        with self._lock:
            previous = self._data.get(tenant_id, Snapshot(0, []))
            revision = previous.revision + (previous.documents != documents)
            self._data[tenant_id] = Snapshot(revision, deepcopy(documents), previous.threshold)
            if previous.revision != revision:
                self._reviews.pop(tenant_id, None)
            fraction = _changed_fraction(
                self._computed.get(tenant_id, []), list(map(asdict, documents))
            )
            return revision, fraction

    def save_analysis(self, tenant_id: str, revision: int, analysis: DocumentAnalysis) -> bool:
        """Compare-and-swap scores and review outbox entries at a coherent revision."""
        tenant_id = _tenant(tenant_id)
        if analysis.tenant_id != tenant_id:
            raise ValueError("analysis tenant does not match destination")
        with self._lock:
            if self.snapshot(tenant_id).revision != revision:
                return False
            self._computed[tenant_id] = list(map(asdict, analysis.documents))
            self._reviews[tenant_id] = [
                dict(analysis.anomaly(doc.id), revision=revision)
                for doc in analysis.documents
                if analysis.anomaly(doc.id)["review"]
            ]
            return True

    def save_topology(self, tenant_id: str, topology: Topology, name: str = "") -> str:
        """Persist an isolated copy under a new UUID for this tenant."""
        identity = str(uuid4())
        with self._lock:
            self._topologies[(_tenant(tenant_id), identity)] = deepcopy(topology)
        return identity

    def get_topology(self, tenant_id: str, topology_id: str) -> Topology:
        """Read a topology only when both tenant and topology IDs match."""
        with self._lock:
            return deepcopy(self._topologies[(_tenant(tenant_id), str(UUID(topology_id)))])

    def reviews(self, tenant_id: str) -> list[dict]:
        """Return this tenant's pending anomaly review entries."""
        with self._lock:
            return deepcopy(self._reviews.get(_tenant(tenant_id), []))

    def acknowledge_review(self, tenant_id: str, document_id: str, revision: int) -> None:
        """Remove an outbox entry only if its revision was successfully delivered."""
        with self._lock:
            tenant_id = _tenant(tenant_id)
            self._reviews[tenant_id] = [
                row
                for row in self._reviews.get(tenant_id, [])
                if (row["document_id"], row["revision"]) != (document_id, revision)
            ]


class PostgresStore:
    """Parameterized, tenant-filtered PostgreSQL storage with atomic revision checks."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def initialize(self) -> None:
        """Create the idempotent rank schema on an explicitly configured database."""
        with psycopg.connect(self.dsn) as connection:
            connection.execute(files("meridian_rank").joinpath("schema.sql").read_text())

    def set_threshold(self, tenant_id: str, threshold: float | None) -> None:
        """Atomically change one tenant's calibration and bump its cache revision."""
        if threshold is not None and (not math.isfinite(threshold) or threshold < 0):
            raise ValueError("threshold must be finite and nonnegative")
        tenant_id = _tenant(tenant_id)
        with psycopg.connect(self.dsn) as connection:
            connection.execute(
                "INSERT INTO rank_tenants(tenant_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (tenant_id,),
            )
            changed = connection.execute(
                "UPDATE rank_tenants SET anomaly_threshold=%s, revision=revision+1, "
                "computed_revision=NULL WHERE tenant_id=%s "
                "AND anomaly_threshold IS DISTINCT FROM %s RETURNING revision",
                (threshold, tenant_id, threshold),
            ).fetchone()
            if changed:
                connection.execute("DELETE FROM rank_scores WHERE tenant_id=%s", (tenant_id,))
                connection.execute("DELETE FROM rank_review_queue WHERE tenant_id=%s", (tenant_id,))

    def snapshot(self, tenant_id: str) -> Snapshot:
        """Read documents and revision in a repeatable-read transaction."""
        tenant_id = _tenant(tenant_id)
        with psycopg.connect(self.dsn) as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            row = connection.execute(
                "SELECT revision, anomaly_threshold FROM rank_tenants WHERE tenant_id = %s",
                (tenant_id,),
            ).fetchone()
            if row is None:
                return Snapshot(0, [])
            documents = connection.execute(
                "SELECT spec FROM rank_documents WHERE tenant_id = %s ORDER BY document_id",
                (tenant_id,),
            ).fetchall()
            return Snapshot(row[0], [Document(**doc[0]) for doc in documents], row[1])

    def replace_documents(self, tenant_id: str, documents: list[Document]) -> tuple[int, float]:
        """Serialize ingestion per tenant and atomically replace its graph and text."""
        tenant_id = _tenant(tenant_id)
        documents = sorted(documents, key=lambda doc: doc.id)
        specs = list(map(asdict, documents))
        nodes, edges = extract_graph(tenant_id, documents)
        with psycopg.connect(self.dsn) as connection:
            connection.execute(
                "INSERT INTO rank_tenants(tenant_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (tenant_id,),
            )
            revision, baseline = connection.execute(
                "SELECT revision, computed_documents FROM rank_tenants "
                "WHERE tenant_id = %s FOR UPDATE",
                (tenant_id,),
            ).fetchone()
            current = connection.execute(
                "SELECT spec FROM rank_documents WHERE tenant_id = %s ORDER BY document_id",
                (tenant_id,),
            ).fetchall()
            fraction = _changed_fraction(baseline, specs)
            if [row[0] for row in current] == specs:
                return revision, fraction
            connection.execute("DELETE FROM rank_edges WHERE tenant_id = %s", (tenant_id,))
            connection.execute("DELETE FROM rank_nodes WHERE tenant_id = %s", (tenant_id,))
            connection.execute("DELETE FROM rank_documents WHERE tenant_id = %s", (tenant_id,))
            connection.execute("DELETE FROM rank_scores WHERE tenant_id = %s", (tenant_id,))
            connection.execute("DELETE FROM rank_review_queue WHERE tenant_id = %s", (tenant_id,))
            with connection.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO rank_documents(tenant_id, document_id, spec) VALUES (%s,%s,%s)",
                    [(tenant_id, doc["id"], Jsonb(doc)) for doc in specs],
                )
                cursor.executemany(
                    "INSERT INTO rank_nodes(tenant_id, node_id, node_type) VALUES (%s,%s,%s)",
                    [(tenant_id, node, kind) for node, kind in nodes],
                )
                cursor.executemany(
                    "INSERT INTO rank_edges(tenant_id, src, dst, weight, edge_type) "
                    "VALUES (%s,%s,%s,%s,%s)",
                    [(tenant_id, src, dst, weight, kind) for src, dst, weight, kind in edges],
                )
            revision += 1
            connection.execute(
                "UPDATE rank_tenants SET revision = %s, computed_revision = NULL "
                "WHERE tenant_id = %s",
                (revision, tenant_id),
            )
            return revision, fraction

    def load_graph(self, tenant_id: str) -> tuple[csr_matrix, NodeIndex]:
        """Load only this tenant's A[dst, src], preserving isolated nodes."""
        tenant_id = _tenant(tenant_id)
        with psycopg.connect(self.dsn) as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            nodes = connection.execute(
                "SELECT node_id FROM rank_nodes WHERE tenant_id = %s ORDER BY node_id", (tenant_id,)
            ).fetchall()
            edges = connection.execute(
                "SELECT src, dst, weight FROM rank_edges WHERE tenant_id = %s", (tenant_id,)
            ).fetchall()
        return build_adjacency(edges, NodeIndex([row[0] for row in nodes]))

    def save_analysis(self, tenant_id: str, revision: int, analysis: DocumentAnalysis) -> bool:
        """Persist scores and an anomaly outbox only if the data revision is still current."""
        tenant_id = _tenant(tenant_id)
        if analysis.tenant_id != tenant_id:
            raise ValueError("analysis tenant does not match destination")
        errors = {
            f"filing:{doc.id}": float(value)
            for doc, value in zip(analysis.documents, analysis.errors)
        }
        with psycopg.connect(self.dsn) as connection:
            connection.execute(
                "INSERT INTO rank_tenants(tenant_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (tenant_id,),
            )
            current = connection.execute(
                "SELECT revision FROM rank_tenants WHERE tenant_id = %s FOR UPDATE", (tenant_id,)
            ).fetchone()[0]
            if current != revision:
                return False
            connection.execute("DELETE FROM rank_scores WHERE tenant_id = %s", (tenant_id,))
            connection.execute("DELETE FROM rank_review_queue WHERE tenant_id = %s", (tenant_id,))
            with connection.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO rank_scores(tenant_id, node_id, pagerank, anomaly) "
                    "VALUES (%s,%s,%s,%s)",
                    [
                        (tenant_id, node, float(score), errors.get(node))
                        for node, score in zip(analysis.index.ids, analysis.ranks)
                    ],
                )
                cursor.executemany(
                    "INSERT INTO rank_review_queue"
                    "(tenant_id, document_id, score, threshold, revision) "
                    "VALUES (%s,%s,%s,%s,%s)",
                    [
                        (tenant_id, doc.id, float(score), analysis.threshold, revision)
                        for doc, score in zip(analysis.documents, analysis.errors)
                        if score > analysis.threshold
                    ],
                )
            connection.execute(
                """UPDATE rank_tenants SET computed_revision=%s, computed_documents=%s
                   WHERE tenant_id=%s""",
                (revision, Jsonb(list(map(asdict, analysis.documents))), tenant_id),
            )
            return True

    def save_topology(self, tenant_id: str, topology: Topology, name: str = "") -> str:
        """Insert one topology JSON document under the authenticated tenant's UUID."""
        identity = str(uuid4())
        with psycopg.connect(self.dsn) as connection:
            connection.execute(
                "INSERT INTO network_topologies(id, tenant_id, name, spec) VALUES (%s,%s,%s,%s)",
                (identity, _tenant(tenant_id), name, Jsonb(asdict(topology))),
            )
        return identity

    def get_topology(self, tenant_id: str, topology_id: str) -> Topology:
        """Require tenant ownership as part of the SQL topology lookup."""
        with psycopg.connect(self.dsn) as connection:
            row = connection.execute(
                "SELECT spec FROM network_topologies WHERE tenant_id = %s AND id = %s",
                (_tenant(tenant_id), UUID(topology_id)),
            ).fetchone()
        if row is None:
            raise KeyError(topology_id)
        return topology_from_dict(row[0])

    def reviews(self, tenant_id: str) -> list[dict]:
        """Read pending review events for exactly one tenant."""
        with psycopg.connect(self.dsn) as connection:
            rows = connection.execute(
                """SELECT document_id, score, threshold, revision FROM rank_review_queue
                   WHERE tenant_id=%s ORDER BY document_id""",
                (_tenant(tenant_id),),
            ).fetchall()
        return [
            dict(zip(("document_id", "score", "threshold", "revision"), row), review=True)
            for row in rows
        ]

    def acknowledge_review(self, tenant_id: str, document_id: str, revision: int) -> None:
        """Acknowledge only the tenant/document/revision successfully delivered to the host."""
        with psycopg.connect(self.dsn) as connection:
            connection.execute(
                "DELETE FROM rank_review_queue "
                "WHERE tenant_id=%s AND document_id=%s AND revision=%s",
                (_tenant(tenant_id), document_id, revision),
            )


class MemoryCache:
    """Bounded TTL response cache for local development."""

    def __init__(self, ttl: int = 300, capacity: int = 1024) -> None:
        self.ttl, self.capacity = ttl, capacity
        self._values: OrderedDict = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> object | None:
        """Return a copy only while now is below the stored expiry."""
        with self._lock:
            entry = self._values.get(key)
            if entry and entry[0] > monotonic():
                self._values.move_to_end(key)
                return deepcopy(entry[1])
            self._values.pop(key, None)
            return None

    def set(self, key: str, value: object) -> None:
        """Retain one response for ttl seconds, evicting the least-recent entry."""
        with self._lock:
            self._values[key] = (monotonic() + self.ttl, deepcopy(value))
            self._values.move_to_end(key)
            while len(self._values) > self.capacity:
                self._values.popitem(last=False)


class RedisCache:
    """JSON-only Redis response cache with a default 300-second TTL."""

    def __init__(self, url: str, ttl: int = 300) -> None:
        self.client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=3)
        self.ttl = ttl

    def get(self, key: str) -> object | None:
        """Read a revision-scoped response, degrading to a miss when Redis is unavailable."""
        try:
            value = self.client.get(key)
            return json.loads(value) if value is not None else None
        except (redis.RedisError, ValueError):
            return None

    def set(self, key: str, value: object) -> None:
        """Store a JSON response with TTL; cache failure does not lose database writes."""
        try:
            self.client.set(key, json.dumps(value, allow_nan=False), ex=self.ttl)
        except redis.RedisError:
            pass


class RankingService:
    """Coordinate tenant snapshots, lazy analysis, eager 5% refresh, and response caching."""

    def __init__(self, store: Store, cache: MemoryCache | RedisCache | None = None) -> None:
        self.store = store
        self.cache = cache if cache is not None else MemoryCache()
        self._analyses: OrderedDict = OrderedDict()
        self._lock = RLock()

    def ingest(self, tenant_id: str, documents: list[Document]) -> dict:
        """Replace a corpus and eagerly recompute above 5% changes since the last fit."""
        revision, fraction = self.store.replace_documents(tenant_id, documents)
        if fraction > 0.05:
            self.recompute(tenant_id)
        return {"revision": revision, "changed_fraction": fraction, "recomputed": fraction > 0.05}

    def _analysis(
        self, tenant_id: str, snapshot: Snapshot, force: bool = False
    ) -> DocumentAnalysis:
        key = (_tenant(tenant_id), snapshot.revision)
        with self._lock:
            entry = self._analyses.get(key)
            if not force and entry and entry[0] > monotonic():
                self._analyses.move_to_end(key)
                return entry[1]
        analysis = analyze_documents(key[0], snapshot.documents, snapshot.threshold)
        if not self.store.save_analysis(key[0], snapshot.revision, analysis):
            raise RuntimeError("tenant data changed during analysis; retry the request")
        with self._lock:
            self._analyses[key] = (monotonic() + 300, analysis)
            self._analyses.move_to_end(key)
            while len(self._analyses) > 32:
                self._analyses.popitem(last=False)
        return analysis

    def recompute(self, tenant_id: str) -> dict:
        """Refresh one tenant's graph/SVD and atomically enqueue its anomaly reviews."""
        snapshot = self.store.snapshot(tenant_id)
        analysis = self._analysis(tenant_id, snapshot, force=True)
        return {
            "revision": snapshot.revision,
            "nodes": len(analysis.index.ids),
            "documents": len(analysis.documents),
            "review_count": int((analysis.errors > analysis.threshold).sum()),
        }

    def query(self, tenant_id: str, operation: str, **parameters) -> object:
        """Cache tenant/revision/operation/parameter-specific results, preventing stale reuse."""
        tenant_id = _tenant(tenant_id)
        snapshot = self.store.snapshot(tenant_id)
        if operation not in {"top", "related", "similar", "anomaly"}:
            raise ValueError("unknown ranking operation")
        key = f"meridian-rank:v1:{tenant_id}:{snapshot.revision}:{operation}:" + json.dumps(
            parameters, sort_keys=True
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        analysis = self._analysis(tenant_id, snapshot)
        result = getattr(analysis, operation)(**parameters)
        self.cache.set(key, result)
        return result
