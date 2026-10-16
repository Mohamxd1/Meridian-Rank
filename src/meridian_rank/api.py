"""FastAPI routes with injected authentication and bounded analysis requests."""

import os
from dataclasses import asdict
from typing import Annotated, Callable, Literal
from uuid import UUID

import numpy as np
import psycopg
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .documents import Document
from .lsa import anomaly_flags
from .network.centrality import betweenness_centrality, eigenvector_centrality, pagerank_centrality
from .network.failure import criticality_ranking, fail_link, fail_node, impact
from .network.routing import bellman_ford_rounds, shortest_path
from .network.topology import Link, Topology, make_grid, make_leaf_spine, make_ring, make_scale_free
from .network.traffic import link_load, traffic_anomaly, utilization
from .store import MemoryCache, MemoryStore, PostgresStore, RankingService, RedisCache, Store


class StrictModel(BaseModel):
    """Reject extra fields and nonfinite JSON numbers at the request boundary."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class LinkInput(StrictModel):
    src: str = Field(min_length=1, max_length=200)
    dst: str = Field(min_length=1, max_length=200)
    bandwidth_mbps: float = Field(default=1000, gt=0)
    latency_ms: float = Field(default=1, ge=0)
    cost: float | None = Field(default=None, gt=0)


class TopologyInput(StrictModel):
    kind: Literal["custom", "ring", "grid", "scale_free", "leaf_spine"] = "custom"
    name: str = Field(default="", max_length=200)
    nodes: list[str] | None = Field(default=None, max_length=500)
    links: list[LinkInput] | None = Field(default=None, max_length=5000)
    directed: bool = False
    n: int = Field(default=10, ge=1, le=500)
    m: int = Field(default=2, ge=1, le=499)
    seed: int = 0
    rows: int = Field(default=3, ge=1, le=500)
    cols: int = Field(default=3, ge=1, le=500)
    spines: int = Field(default=2, ge=1, le=500)
    leaves: int = Field(default=4, ge=1, le=500)
    hosts_per_leaf: int = Field(default=4, ge=0, le=500)
    bandwidth_mbps: float = Field(default=1000, gt=0)
    latency_ms: float = Field(default=1, ge=0)
    cost: float | None = Field(default=None, gt=0)

    def topology(self) -> Topology:
        """Construct at most 500 nodes and 5000 edges after validating generator size."""
        if self.kind == "custom":
            if self.nodes is None or self.links is None:
                raise ValueError("custom topology requires nodes and links")
            return Topology(
                self.nodes, [Link(**link.model_dump()) for link in self.links], self.directed
            )
        if self.nodes is not None or self.links is not None or self.directed:
            raise ValueError(
                "generators use undirected graphs and cannot accept custom nodes or links"
            )
        kwargs = {
            "bandwidth_mbps": self.bandwidth_mbps,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
        }
        if self.kind == "ring":
            result = make_ring(self.n, **kwargs)
        elif self.kind == "grid":
            if self.rows * self.cols > 500:
                raise ValueError("generated topology exceeds 500 nodes")
            result = make_grid(self.rows, self.cols, **kwargs)
        elif self.kind == "scale_free":
            if self.n * self.m > 5000:
                raise ValueError("generated topology exceeds 5000 links")
            result = make_scale_free(self.n, self.m, self.seed, **kwargs)
        else:
            if self.spines + self.leaves * (1 + self.hosts_per_leaf) > 500:
                raise ValueError("generated topology exceeds 500 nodes")
            if self.leaves * (self.spines + self.hosts_per_leaf) > 5000:
                raise ValueError("generated topology exceeds 5000 links")
            result = make_leaf_spine(self.spines, self.leaves, self.hosts_per_leaf, **kwargs)
        return result


class RelatedInput(StrictModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=10, ge=1, le=100)


class DocumentInput(StrictModel):
    id: str = Field(min_length=1, max_length=200)
    text: str = Field(max_length=200_000)
    company: str | None = Field(default=None, min_length=1, max_length=200)
    subsidiaries: list[str] = Field(default_factory=list, max_length=100)
    citations: list[str] = Field(default_factory=list, max_length=100)
    amends: str | None = Field(default=None, min_length=1, max_length=200)
    entities: list[str] = Field(default_factory=list, max_length=100)


class IngestInput(StrictModel):
    documents: list[DocumentInput] = Field(max_length=1000)


class SettingsInput(StrictModel):
    anomaly_threshold: float | None = Field(ge=0)


class FailureInput(StrictModel):
    node: str | None = None
    link: tuple[str, str] | None = None

    @model_validator(mode="after")
    def one_target(self) -> "FailureInput":
        """Require exactly one of node or physical link."""
        if (self.node is None) == (self.link is None):
            raise ValueError("provide exactly one of node or link")
        return self


class DemandInput(StrictModel):
    src: str
    dst: str
    mbps: float = Field(ge=0)


class TrafficInput(StrictModel):
    demands: list[DemandInput] = Field(max_length=5000)


class AnomalyInput(StrictModel):
    series: list[list[float]] = Field(min_length=2, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=100)

    @model_validator(mode="after")
    def rectangular(self) -> "AnomalyInput":
        """Require a finite rectangular matrix with at most 500 flow columns."""
        width = len(self.series[0])
        if not 1 <= width <= 500 or any(len(row) != width for row in self.series):
            raise ValueError("series must be rectangular with 1 to 500 flows")
        return self


def get_tenant(request: Request) -> str:
    """Resolve a UUID from authenticated request state; headers require explicit dev mode."""
    value = getattr(request.state, "rank_tenant_id", None)
    if value is None and getattr(request.app.state, "rank_allow_dev_auth", False):
        value = request.headers.get("X-Tenant-ID")
    if value is None:
        raise HTTPException(401, "Authenticated tenant context is required")
    try:
        return str(UUID(str(value)))
    except ValueError as error:
        raise HTTPException(400, "Tenant context must be a UUID") from error


def require_admin(request: Request) -> None:
    """Require the host's verified admin privilege before replacing data or recomputing."""
    if getattr(request.state, "rank_is_admin", False):
        return
    if getattr(request.app.state, "rank_allow_dev_auth", False):
        return
    raise HTTPException(403, "Administrator privilege is required")


def build_router(
    service: RankingService,
    tenant_dependency: Callable = get_tenant,
    admin_dependency: Callable = require_admin,
) -> APIRouter:
    """Build tenant-scoped routes suitable for inclusion in a host FastAPI application."""
    router = APIRouter()
    Tenant = Annotated[str, Depends(tenant_dependency)]
    Limit = Annotated[int, Query(ge=1, le=100)]

    def topology(tenant: str, identity: UUID, maximum: int = 500) -> Topology:
        result = service.store.get_topology(tenant, str(identity))
        if len(result.nodes) > maximum:
            raise HTTPException(
                422, f"This analysis is limited to {maximum} nodes; use the library offline"
            )
        return result

    @router.get("/rank/top")
    def top(tenant: Tenant, limit: Limit = 20):
        return service.query(tenant, "top", limit=limit)

    @router.post("/rank/related")
    def related(body: RelatedInput, tenant: Tenant):
        return service.query(tenant, "related", **body.model_dump())

    @router.get("/docs/{id}/similar")
    def similar(id: str, tenant: Tenant, limit: Limit = 10):
        return service.query(tenant, "similar", document_id=id, limit=limit)

    @router.get("/docs/{id}/anomaly")
    def anomaly(id: str, tenant: Tenant):
        return service.query(tenant, "anomaly", document_id=id)

    @router.post("/admin/recompute", dependencies=[Depends(admin_dependency)])
    def recompute(tenant: Tenant):
        return service.recompute(tenant)

    @router.put("/admin/documents", dependencies=[Depends(admin_dependency)])
    def ingest(body: IngestInput, tenant: Tenant):
        return service.ingest(tenant, [Document(**doc.model_dump()) for doc in body.documents])

    @router.get("/admin/reviews", dependencies=[Depends(admin_dependency)])
    def reviews(tenant: Tenant):
        return service.store.reviews(tenant)

    @router.patch("/admin/settings", dependencies=[Depends(admin_dependency)])
    def settings(body: SettingsInput, tenant: Tenant):
        service.store.set_threshold(tenant, body.anomaly_threshold)
        return service.recompute(tenant)

    @router.post("/network/topologies", status_code=201)
    def create_topology(body: TopologyInput, tenant: Tenant):
        return {"id": service.store.save_topology(tenant, body.topology(), body.name)}

    @router.get("/network/{id}/path")
    def path(id: UUID, src: str, dst: str, tenant: Tenant):
        result = shortest_path(topology(tenant, id), src, dst)
        if result is None:
            raise HTTPException(404, "No route between the requested nodes")
        return {"path": result[0], "cost": result[1]}

    @router.get("/network/{id}/convergence")
    def convergence(id: UUID, tenant: Tenant):
        return {"rounds": bellman_ford_rounds(topology(tenant, id, 200))}

    @router.get("/network/{id}/centrality")
    def centrality(
        id: UUID,
        tenant: Tenant,
        kind: Literal["eigenvector", "betweenness", "pagerank"] = "betweenness",
    ):
        functions = {
            "eigenvector": eigenvector_centrality,
            "betweenness": betweenness_centrality,
            "pagerank": pagerank_centrality,
        }
        return functions[kind](topology(tenant, id))

    @router.post("/network/{id}/fail")
    def failure(id: UUID, body: FailureInput, tenant: Tenant):
        original = topology(tenant, id, 200)
        failed = (
            fail_node(original, body.node)
            if body.node is not None
            else fail_link(original, *body.link)
        )
        return asdict(impact(original, failed))

    @router.get("/network/{id}/critical")
    def critical(id: UUID, tenant: Tenant, limit: Limit = 5):
        return [
            {"node": node, "impact": asdict(damage)}
            for node, damage in criticality_ranking(topology(tenant, id, 100))[:limit]
        ]

    @router.post("/network/{id}/traffic/load")
    def traffic(id: UUID, body: TrafficInput, tenant: Tenant):
        topo = topology(tenant, id)
        demands = {}
        for row in body.demands:
            pair = (row.src, row.dst)
            demands[pair] = demands.get(pair, 0.0) + row.mbps
        load = link_load(topo, demands)
        ratios = utilization(topo, load)
        return [
            {"src": src, "dst": dst, "load_mbps": value, "utilization": ratios[(src, dst)]}
            for (src, dst), value in sorted(load.items())
        ]

    @router.post("/network/{id}/traffic/anomaly")
    def traffic_errors(id: UUID, body: AnomalyInput, tenant: Tenant):
        topology(tenant, id)  # Ownership is required even though the matrix analysis is pure.
        errors = traffic_anomaly(np.asarray(body.series), body.k)
        return {"errors": errors.tolist(), "flags": anomaly_flags(errors).tolist()}

    return router


def install_error_handlers(application: FastAPI) -> None:
    """Map domain and storage failures to JSON without exposing connection strings."""

    @application.exception_handler(KeyError)
    async def missing(_request: Request, _error: KeyError):
        return JSONResponse(status_code=404, content={"detail": "Resource not found"})

    @application.exception_handler(ValueError)
    async def invalid(_request: Request, error: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @application.exception_handler(RuntimeError)
    async def computation(_request: Request, error: RuntimeError):
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @application.exception_handler(psycopg.OperationalError)
    async def unavailable(_request: Request, _error: psycopg.OperationalError):
        return JSONResponse(status_code=503, content={"detail": "Storage is unavailable"})


def create_app(
    store: Store | None = None,
    cache: MemoryCache | RedisCache | None = None,
    *,
    allow_dev_auth: bool = False,
    tenant_dependency: Callable = get_tenant,
    admin_dependency: Callable = require_admin,
) -> FastAPI:
    """Create an isolated application; default auth rejects unverified tenant headers."""
    application = FastAPI(title="Meridian Rank", version="0.1.0")
    application.state.rank_allow_dev_auth = allow_dev_auth
    service = RankingService(store if store is not None else MemoryStore(), cache)
    application.state.rank_service = service
    application.include_router(build_router(service, tenant_dependency, admin_dependency))
    install_error_handlers(application)

    @application.get("/health")
    def health():
        return {"status": "ok", "storage": type(service.store).__name__}

    return application


def app_from_environment() -> FastAPI:
    """Read optional database/cache configuration without creating or migrating a database."""
    dsn, redis_url = os.getenv("RANK_DATABASE_URL"), os.getenv("RANK_REDIS_URL")
    return create_app(
        PostgresStore(dsn) if dsn else MemoryStore(),
        RedisCache(redis_url) if redis_url else MemoryCache(),
        allow_dev_auth=os.getenv("RANK_DEV_MODE", "false").lower() == "true",
    )


def mount_client_routes(
    application: FastAPI,
    service: RankingService,
    verify_client: Callable,
    tenant_ids: dict[str, str],
    authorize_admin: Callable,
) -> None:
    """Mount under /clients/{client_id}, deriving tenant UUID only after host authentication."""
    tenants = {client: str(UUID(value)) for client, value in tenant_ids.items()}
    if len(set(tenants.values())) != len(tenants):
        raise ValueError("each client must map to a distinct tenant UUID")

    def verified_tenant(client_id: str, _verified=Depends(verify_client)) -> str:
        if client_id not in tenants:
            raise HTTPException(403, "Client has no rank tenant binding")
        return tenants[client_id]

    application.include_router(
        build_router(service, verified_tenant, authorize_admin), prefix="/clients/{client_id}"
    )
    install_error_handlers(application)


app = app_from_environment()
