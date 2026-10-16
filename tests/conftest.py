from dataclasses import asdict
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from meridian_rank.api import create_app
from meridian_rank.documents import Document
from meridian_rank.store import MemoryStore


@pytest.fixture
def tenant():
    return str(uuid4())


@pytest.fixture
def documents():
    return [
        Document("a", "supply chain semiconductor chip shortage", company="Acme", citations=["b"]),
        Document(
            "b", "manufacturing production supplier disruption", company="Acme", citations=["c"]
        ),
        Document("c", "chip semiconductor supply disruption", company="Acme"),
        Document("d", "cooking pasta tomato kitchen recipe", company="Kitchen"),
    ]


@pytest.fixture
def api(tenant, documents):
    store = MemoryStore()
    application = create_app(store, allow_dev_auth=True)
    with TestClient(application) as client:
        client.headers["X-Tenant-ID"] = tenant
        response = client.put("/admin/documents", json={"documents": list(map(asdict, documents))})
        assert response.status_code == 200, response.text
        yield client, store
