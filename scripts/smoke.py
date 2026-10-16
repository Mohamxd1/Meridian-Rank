"""Exercise a running local development server with a fresh synthetic tenant."""

import argparse
from uuid import uuid4

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    with httpx.Client(base_url=args.base_url, timeout=30, trust_env=False) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/rank/top").status_code == 401
        client.headers["X-Tenant-ID"] = str(uuid4())
        docs = [
            {"id": str(i), "text": "company revenue earnings business growth"} for i in range(100)
        ]
        docs.append({"id": "outlier", "text": "volcano magma geology lava eruption"})
        response = client.put("/admin/documents", json={"documents": docs})
        response.raise_for_status()
        assert client.get("/docs/outlier/anomaly").json()["review"] is True
        assert client.get("/admin/reviews").json()[0]["document_id"] == "outlier"
        response = client.post("/network/topologies", json={"kind": "ring", "n": 5, "cost": 1})
        response.raise_for_status()
        identity = response.json()["id"]
        assert (
            client.get(f"/network/{identity}/path", params={"src": "0", "dst": "2"}).json()["cost"]
            == 2
        )
        assert client.get(f"/network/{identity}/convergence").json()["rounds"] == 2
        client.headers["X-Tenant-ID"] = str(uuid4())
        assert client.get(f"/network/{identity}/critical").status_code == 404
        assert client.get("/docs/outlier/anomaly").status_code == 404
    print(
        "Live HTTP checks passed: auth, ingestion, anomaly review, routing, and tenant isolation."
    )


if __name__ == "__main__":
    main()
