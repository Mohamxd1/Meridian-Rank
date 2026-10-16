CREATE TABLE IF NOT EXISTS rank_tenants (
    tenant_id UUID PRIMARY KEY,
    revision BIGINT NOT NULL DEFAULT 0,
    computed_revision BIGINT,
    computed_documents JSONB NOT NULL DEFAULT '[]',
    anomaly_threshold DOUBLE PRECISION CHECK (anomaly_threshold >= 0)
);
CREATE TABLE IF NOT EXISTS rank_documents (
    tenant_id UUID NOT NULL REFERENCES rank_tenants(tenant_id),
    document_id TEXT NOT NULL,
    spec JSONB NOT NULL,
    PRIMARY KEY (tenant_id, document_id)
);
CREATE TABLE IF NOT EXISTS rank_nodes (
    tenant_id UUID NOT NULL REFERENCES rank_tenants(tenant_id),
    node_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    PRIMARY KEY (tenant_id, node_id)
);
CREATE TABLE IF NOT EXISTS rank_edges (
    tenant_id UUID NOT NULL REFERENCES rank_tenants(tenant_id),
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    weight DOUBLE PRECISION NOT NULL DEFAULT 1.0 CHECK (weight >= 0),
    edge_type TEXT NOT NULL,
    PRIMARY KEY (tenant_id, src, dst, edge_type),
    FOREIGN KEY (tenant_id, src) REFERENCES rank_nodes(tenant_id, node_id),
    FOREIGN KEY (tenant_id, dst) REFERENCES rank_nodes(tenant_id, node_id)
);
CREATE INDEX IF NOT EXISTS rank_edges_tenant ON rank_edges(tenant_id);
CREATE TABLE IF NOT EXISTS rank_scores (
    tenant_id UUID NOT NULL REFERENCES rank_tenants(tenant_id),
    node_id TEXT NOT NULL,
    pagerank DOUBLE PRECISION,
    anomaly DOUBLE PRECISION,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, node_id)
);
CREATE TABLE IF NOT EXISTS network_topologies (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    name TEXT NOT NULL,
    spec JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS network_topologies_tenant ON network_topologies(tenant_id);
-- This outbox is the integration boundary for the host's existing review queue.
CREATE TABLE IF NOT EXISTS rank_review_queue (
    tenant_id UUID NOT NULL REFERENCES rank_tenants(tenant_id),
    document_id TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    revision BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, document_id)
);
