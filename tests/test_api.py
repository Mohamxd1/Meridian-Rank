from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from meridian_rank.api import create_app

NETWORK_CALLS = [
    ("get", "/path?src=0&dst=2", None),
    ("get", "/convergence", None),
    ("get", "/centrality?kind=betweenness", None),
    ("get", "/centrality?kind=eigenvector", None),
    ("get", "/centrality?kind=pagerank", None),
    ("post", "/fail", {"node": "0"}),
    ("post", "/fail", {"link": ["0", "1"]}),
    ("get", "/critical", None),
    ("post", "/traffic/load", {"demands": [{"src": "0", "dst": "2", "mbps": 10}]}),
    ("post", "/traffic/anomaly", {"series": [[1, 2], [2, 4], [3, 6], [20, 1]], "k": 1}),
]


@pytest.mark.parametrize("method,suffix,body", NETWORK_CALLS)
def test_network_routes_and_cross_tenant_denial(api, method, suffix, body):
    client, _ = api
    response = client.post("/network/topologies", json={"kind": "ring", "n": 5})
    assert response.status_code == 201
    identity = response.json()["id"]
    kwargs = {} if body is None else {"json": body}
    assert client.request(method, f"/network/{identity}{suffix}", **kwargs).status_code == 200
    client.headers["X-Tenant-ID"] = str(uuid4())
    assert client.request(method, f"/network/{identity}{suffix}", **kwargs).status_code == 404
    client.headers.pop("X-Tenant-ID")
    assert 400 <= client.request(method, f"/network/{identity}{suffix}", **kwargs).status_code < 500


@pytest.mark.parametrize(
    "method,url,body",
    [
        ("get", "/rank/top", None),
        ("post", "/rank/related", {"query": "semiconductor"}),
        ("get", "/docs/a/similar", None),
        ("get", "/docs/a/anomaly", None),
        ("post", "/admin/recompute", None),
        ("get", "/admin/reviews", None),
        ("put", "/admin/documents", {"documents": []}),
        ("patch", "/admin/settings", {"anomaly_threshold": 0.5}),
        ("post", "/network/topologies", {"kind": "ring", "n": 4}),
    ],
)
def test_remaining_routes_require_tenant(api, method, url, body):
    client, _ = api
    kwargs = {} if body is None else {"json": body}
    assert client.request(method, url, **kwargs).status_code in {200, 201}
    client.headers.pop("X-Tenant-ID")
    assert 400 <= client.request(method, url, **kwargs).status_code < 500


def test_documents_are_isolated_on_all_reads(api):
    client, _ = api
    client.headers["X-Tenant-ID"] = str(uuid4())
    assert client.get("/rank/top").json() == []
    assert client.post("/rank/related", json={"query": "semiconductor"}).json() == []
    assert client.get("/docs/a/similar").status_code == 404
    assert client.get("/docs/a/anomaly").status_code == 404
    assert client.get("/admin/reviews").json() == []
    assert client.post("/admin/recompute").json()["documents"] == 0


def test_production_denies_spoofed_header_and_requires_admin(tenant):
    application = create_app()
    client = TestClient(application)
    assert client.get("/rank/top", headers={"X-Tenant-ID": tenant}).status_code == 401

    application = create_app()

    @application.middleware("http")
    async def verified_tenant(request, call_next):
        request.state.rank_tenant_id = tenant
        return await call_next(request)

    client = TestClient(application)
    assert client.get("/rank/top").status_code == 200
    assert client.post("/admin/recompute").status_code == 403


@pytest.mark.parametrize(
    "kind,parameters",
    [
        ("grid", {"rows": 3, "cols": 4}),
        ("scale_free", {"n": 10, "m": 2}),
        ("leaf_spine", {"spines": 2, "leaves": 3}),
    ],
)
def test_generators(api, kind, parameters):
    client, _ = api
    assert client.post("/network/topologies", json={"kind": kind, **parameters}).status_code == 201


@pytest.mark.parametrize(
    "body",
    [
        {"kind": "grid", "rows": 500, "cols": 500},
        {"kind": "ring", "n": 2},
        {"nodes": ["a"], "links": [{"src": "a", "dst": "b"}]},
        {"nodes": ["a", "a"], "links": []},
        {"kind": "ring", "cost": -1},
        {"kind": "unknown"},
    ],
)
def test_invalid_topology_requests(api, body):
    assert api[0].post("/network/topologies", json=body).status_code == 422


def test_unknown_resource_bad_payload_and_openapi(api):
    client, _ = api
    assert client.get(f"/network/{uuid4()}/convergence").status_code == 404
    assert client.get("/network/not-a-uuid/convergence").status_code == 422
    identity = client.post("/network/topologies", json={"kind": "ring", "n": 4}).json()["id"]
    assert client.post(f"/network/{identity}/fail", json={}).status_code == 422
    assert (
        client.post(
            f"/network/{identity}/traffic/anomaly", json={"series": [[1, 2], [3]]}
        ).status_code
        == 422
    )
    assert "/rank/top" in client.get("/openapi.json").json()["paths"]
