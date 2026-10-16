"""Check the adapter against an existing MeridianData checkout without modifying it."""

import argparse
import os
import sys
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from meridian_rank.api import mount_client_routes
from meridian_rank.documents import Document, review_gate
from meridian_rank.store import MemoryStore, RankingService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host_checkout", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.host_checkout.resolve()))
    from dataforge.auth import verify_api_key
    from dataforge.pipeline.export import export_record

    # These are ephemeral fixture keys in this process, never production credentials.
    client_id = "rank_adapter_test"
    variable = "API_KEY_RANK_ADAPTER_TEST"
    previous = os.environ.get(variable)
    key = uuid4().hex
    os.environ[variable] = key
    try:
        tenant = str(uuid4())
        service = RankingService(MemoryStore())
        documents = [
            Document(str(i), "company revenue earnings business growth") for i in range(100)
        ]
        documents.append(Document("outlier", "volcano magma geology lava eruption"))
        service.ingest(tenant, documents)
        application = FastAPI()

        def deny_admin():
            raise HTTPException(403, "Admin role intentionally absent in this fixture")

        mount_client_routes(application, service, verify_api_key, {client_id: tenant}, deny_admin)
        with TestClient(application) as client:
            path = f"/clients/{client_id}/rank/top"
            assert client.get(path).status_code == 403
            assert (
                client.get(path, headers={"X-API-Key": "incorrect-fixture-key"}).status_code == 403
            )
            assert client.get(path, headers={"X-API-Key": key}).status_code == 200
            assert (
                client.get("/clients/other/rank/top", headers={"X-API-Key": key}).status_code == 403
            )
            assert "/clients/{client_id}/rank/top" in client.get("/openapi.json").json()["paths"]
        decision = service.query(tenant, "anomaly", document_id="outlier")
        record = review_gate(tenant, "outlier", {"status": "approved"}, decision)
        result = export_record(record, {"type": "csv", "config": {}})
        assert result["exported"] is False
        print("MeridianData auth adapter, OpenAPI mount, and existing export gate passed.")
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


if __name__ == "__main__":
    main()
