from uuid import uuid4

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from meridian_rank.api import mount_client_routes
from meridian_rank.documents import Document, review_gate
from meridian_rank.store import MemoryStore, RankingService


def test_mounted_client_auth_context_and_openapi():
    application = FastAPI()
    tenants = {"alice": str(uuid4()), "bob": str(uuid4())}

    def verify_client(client_id: str, x_api_key: str | None = Header(default=None)):
        if x_api_key != f"test-key-for-{client_id}":
            raise HTTPException(403)

    def admin(x_rank_admin: str | None = Header(default=None)):
        if x_rank_admin != "test-admin":
            raise HTTPException(403)

    store = MemoryStore()
    service = RankingService(store)
    service.ingest(tenants["alice"], [Document("private", "semiconductor supply chain")])
    mount_client_routes(application, service, verify_client, tenants, admin)
    client = TestClient(application)
    assert client.get("/clients/alice/rank/top").status_code == 403
    headers = {"X-API-Key": "test-key-for-alice", "X-Tenant-ID": tenants["bob"]}
    assert (
        client.get("/clients/alice/rank/top", headers=headers).json()[0]["node_id"]
        == "filing:private"
    )
    assert client.get("/clients/bob/rank/top", headers=headers).status_code == 403
    assert (
        client.get("/clients/bob/rank/top", headers={"X-API-Key": "test-key-for-bob"}).json() == []
    )
    assert client.post("/clients/alice/admin/recompute", headers=headers).status_code == 403
    headers["X-Rank-Admin"] = "test-admin"
    assert client.post("/clients/alice/admin/recompute", headers=headers).status_code == 200
    assert "/clients/{client_id}/rank/top" in client.get("/openapi.json").json()["paths"]


def test_outlier_review_gate_prevents_export_approval(tenant):
    service = RankingService(MemoryStore())
    docs = [Document(str(i), "company revenue earnings business growth") for i in range(100)]
    docs.append(Document("outlier", "volcano magma geology lava eruption"))
    service.ingest(tenant, docs)
    decision = service.query(tenant, "anomaly", document_id="outlier")
    record = {"status": "approved", "company": "Acme"}
    gated = review_gate(tenant, "outlier", record, decision)
    assert gated["status"] == "pending_review"
    assert record["status"] == "approved"
    assert review_gate(tenant, "outlier", {"status": "rejected"}, decision)["status"] == "rejected"
    with pytest.raises(ValueError):
        review_gate(tenant, "another", record, decision)
