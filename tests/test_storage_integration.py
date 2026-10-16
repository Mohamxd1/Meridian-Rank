import os
from dataclasses import asdict
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient

from meridian_rank.api import create_app
from meridian_rank.documents import Document, analyze_documents
from meridian_rank.network.topology import make_ring
from meridian_rank.store import PostgresStore, RankingService, RedisCache

pytestmark = pytest.mark.integration


@pytest.fixture
def storage():
    dsn, url = os.getenv("RANK_TEST_DATABASE_URL"), os.getenv("RANK_TEST_REDIS_URL")
    if not dsn or not url:
        pytest.skip(
            "set RANK_TEST_DATABASE_URL and RANK_TEST_REDIS_URL for real service integration"
        )
    store, cache = PostgresStore(dsn), RedisCache(url)
    store.initialize()
    store.initialize()  # Schema creation is repeatable.
    assert cache.client.ping()
    tenants = [str(uuid4()), str(uuid4())]
    yield store, cache, tenants
    with psycopg.connect(dsn) as connection:
        for tenant in tenants:
            for table in [
                "rank_review_queue",
                "rank_scores",
                "rank_edges",
                "rank_nodes",
                "rank_documents",
                "rank_tenants",
                "network_topologies",
            ]:
                connection.execute(
                    psycopg.sql.SQL("DELETE FROM {} WHERE tenant_id=%s").format(
                        psycopg.sql.Identifier(table)
                    ),
                    (tenant,),
                )
            for key in cache.client.scan_iter(f"meridian-rank:v1:{tenant}:*"):
                cache.client.delete(key)


def test_real_postgres_graph_isolation_atomicity_and_persistence(storage, documents):
    store, cache, (a, b) = storage
    service = RankingService(store, cache)
    service.ingest(a, documents)
    service.ingest(b, [Document("other", "private tenant material")])
    graph, index = store.load_graph(a)
    assert "filing:other" not in index.ids
    assert graph[index.to_int("filing:b"), index.to_int("filing:a")] == pytest.approx(0.7)
    assert len(store.snapshot(b).documents) == 1
    topology = make_ring(5)
    identity = store.save_topology(a, topology)
    assert asdict(PostgresStore(store.dsn).get_topology(a, identity)) == asdict(topology)
    with pytest.raises(KeyError):
        store.get_topology(b, identity)
    old = store.snapshot(a)
    service.ingest(a, [Document("new", "replacement corpus")])
    assert not store.save_analysis(a, old.revision, analyze_documents(a, old.documents))
    with pytest.raises(ValueError):
        service.ingest(a, [Document("duplicate", "one"), Document("duplicate", "two")])
    assert store.snapshot(a).documents[0].id == "new"


def test_real_redis_cache_hit_expiry_metadata_and_revision_invalidation(storage, documents):
    store, cache, (a, b) = storage
    service = RankingService(store, cache)
    service.ingest(a, documents)
    expected = service.query(a, "top")
    keys = list(cache.client.scan_iter(f"meridian-rank:v1:{a}:*"))
    assert len(keys) == 1
    assert 0 < cache.client.ttl(keys[0]) <= 300
    assert cache.get(keys[0]) == expected
    # A fresh service instance must reuse Redis without fitting the analysis.
    second = RankingService(store, cache)
    second._analysis = lambda *args, **kwargs: pytest.fail("expected a Redis cache hit")
    assert second.query(a, "top") == expected
    service.ingest(a, [Document("new", "replacement corpus")])
    assert service.query(a, "top") == [{"node_id": "filing:new", "score": 1}]
    assert service.query(b, "top") == []


def test_database_backed_http_ingestion_to_review_queue(storage):
    store, cache, (tenant, other) = storage
    client = TestClient(create_app(store, cache, allow_dev_auth=True))
    client.headers["X-Tenant-ID"] = tenant
    docs = [{"id": str(i), "text": "company revenue earnings business growth"} for i in range(100)]
    docs.append({"id": "outlier", "text": "volcano magma geology lava eruption"})
    response = client.put("/admin/documents", json={"documents": docs})
    assert response.status_code == 200, response.text
    revision = response.json()["revision"]
    assert client.get("/docs/outlier/anomaly").json()["review"] is True
    assert client.get("/admin/reviews").json()[0]["document_id"] == "outlier"
    client.headers["X-Tenant-ID"] = other
    assert client.get("/admin/reviews").json() == []
    assert client.get("/docs/outlier/anomaly").status_code == 404
    store.acknowledge_review(tenant, "outlier", revision)
    assert store.reviews(tenant) == []
    client.headers["X-Tenant-ID"] = tenant
    response = client.patch("/admin/settings", json={"anomaly_threshold": 2.0})
    assert response.status_code == 200, response.text
    assert response.json()["revision"] == revision + 1
    assert client.get("/docs/outlier/anomaly").json()["review"] is False
    assert store.snapshot(other).threshold is None
    assert client.patch("/admin/settings", json={"anomaly_threshold": -1}).status_code == 422
