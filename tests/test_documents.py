from dataclasses import asdict
from uuid import uuid4

import pytest

from meridian_rank.documents import Document, analyze_documents, extract_graph
from meridian_rank.store import MemoryCache, MemoryStore, RankingService


def test_extraction_weights_and_namespace_collisions(tenant):
    docs = [
        Document(
            "same",
            "text",
            company="same",
            subsidiaries=["child"],
            citations=["old"],
            amends="original",
            entities=["one", "two", "one"],
        )
    ]
    nodes, edges = extract_graph(tenant, docs)
    assert ("filing:same", "filing") in nodes
    assert ("company:same", "company") in nodes
    assert ("filing:same", "company:same", 1, "filing_company") in edges
    assert ("filing:same", "filing:old", 0.7, "filing_citation") in edges
    assert ("filing:same", "filing:original", 1, "filing_amendment") in edges
    assert ("company:same", "company:child", 1, "company_subsidiary") in edges
    assert ("entity:one", "entity:two", 0.3, "entity_cooccurrence") in edges
    with pytest.raises(ValueError):
        extract_graph(tenant, docs * 2)


def test_document_rank_search_similarity_and_unknown(tenant, documents):
    analysis = analyze_documents(tenant, documents)
    assert sum(item["score"] for item in analysis.top(100)) == pytest.approx(1)
    assert analysis.related("semiconductor")[0]["document_id"] in {"a", "c"}
    assert analysis.related("wordnotincorpus") == []
    assert len(analysis.similar("a")) == 3
    assert analysis.anomaly("a")["score"] >= 0
    with pytest.raises(KeyError):
        analysis.anomaly("missing")


def test_empty_corpus_and_empty_vocabulary(tenant):
    assert analyze_documents(tenant, []).top() == []
    analysis = analyze_documents(tenant, [Document("a", "the and or"), Document("b", "")])
    assert analysis.related("anything") == []
    assert analysis.anomaly("a")["score"] == 0


def test_cache_hit_revision_invalidation_and_tenant_isolation(tenant, documents):
    class CountingCache(MemoryCache):
        hits = 0

        def get(self, key):
            result = super().get(key)
            self.hits += result is not None
            return result

    store, cache = MemoryStore(), CountingCache()
    service = RankingService(store, cache)
    service.ingest(tenant, documents)
    first = service.query(tenant, "top")
    assert service.query(tenant, "top") == first
    assert cache.hits == 1
    assert service.query(str(uuid4()), "top") == []
    service.ingest(tenant, [Document("replacement", "different material")])
    assert service.query(tenant, "top") == [{"node_id": "filing:replacement", "score": 1}]
    assert cache.hits == 1


def test_cumulative_recompute_threshold_and_stale_write(tenant):
    documents = [Document(str(i), "normal revenue company") for i in range(100)]
    store, cache = MemoryStore(), MemoryCache()
    service = RankingService(store, cache)
    service.ingest(tenant, documents)
    old = store.snapshot(tenant)
    for i in range(5):
        documents[i] = Document(str(i), "updated revenue company")
    assert service.ingest(tenant, documents)["recomputed"] is False
    documents[5] = Document("5", "updated revenue company")
    assert service.ingest(tenant, documents)["recomputed"] is True
    assert not store.save_analysis(tenant, old.revision, analyze_documents(tenant, old.documents))
    assert service.ingest(tenant, documents)["changed_fraction"] == 0


def test_outlier_enters_review_queue_and_ack_is_revision_guarded(tenant):
    documents = [Document(str(i), "company revenue earnings business growth") for i in range(100)]
    documents.append(Document("outlier", "volcano magma geology lava eruption"))
    store = MemoryStore()
    result = RankingService(store).ingest(tenant, documents)
    assert store.reviews(tenant)[0]["document_id"] == "outlier"
    assert store.reviews(str(uuid4())) == []
    store.acknowledge_review(tenant, "outlier", result["revision"] - 1)
    assert len(store.reviews(tenant)) == 1
    store.acknowledge_review(tenant, "outlier", result["revision"])
    assert store.reviews(tenant) == []


def test_memory_store_defends_against_mutation(tenant, documents):
    store = MemoryStore()
    store.replace_documents(tenant, documents)
    expected = asdict(store.snapshot(tenant))
    documents[0].text = "mutated"
    snapshot = store.snapshot(tenant)
    snapshot.documents.clear()
    assert asdict(store.snapshot(tenant)) == expected
