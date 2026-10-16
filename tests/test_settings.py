from uuid import uuid4

import pytest

from meridian_rank.documents import Document, review_gate
from meridian_rank.store import MemoryCache, MemoryStore, RankingService


def test_threshold_updates_invalidate_cached_anomaly_decisions(tenant):
    store = MemoryStore()
    service = RankingService(store)
    docs = [Document(str(i), "company revenue earnings business growth") for i in range(100)]
    docs.append(Document("outlier", "volcano magma geology lava eruption"))
    service.ingest(tenant, docs)
    first = service.query(tenant, "anomaly", document_id="outlier")
    assert first["review"]
    before = store.snapshot(tenant).revision
    store.set_threshold(tenant, 2.0)
    assert store.snapshot(tenant).revision == before + 1
    assert service.query(tenant, "anomaly", document_id="outlier")["review"] is False
    assert service.query(tenant, "anomaly", document_id="outlier")["threshold"] == 2
    assert store.reviews(tenant) == []
    assert store.snapshot(str(uuid4())).threshold is None
    with pytest.raises(ValueError):
        review_gate(str(uuid4()), "outlier", {"status": "approved"}, first)
    with pytest.raises(ValueError):
        store.set_threshold(tenant, float("nan"))


def test_memory_cache_honors_expiration_and_capacity(monkeypatch):
    import meridian_rank.store as storage

    clock = [0]
    monkeypatch.setattr(storage, "monotonic", lambda: clock[0])
    cache = MemoryCache(ttl=10, capacity=1)
    cache.set("a", [1])
    returned = cache.get("a")
    returned.append(2)
    assert cache.get("a") == [1]
    clock[0] = 10
    assert cache.get("a") is None
    cache.set("a", [2])
    cache.set("b", [3])
    assert cache.get("a") is None
    assert cache.get("b") == [3]
