"""Pure tenant document extraction, TF-IDF, ranking, and anomaly computation."""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .graph import NodeIndex, build_adjacency, to_transition
from .lsa import LSAModel, anomaly_threshold, fit_lsa, reconstruction_error, similar
from .pagerank import pagerank
from .ppr import personalized_pagerank


@dataclass
class Document:
    """Normalized processed filing; IDs are external IDs within a tenant."""

    id: str
    text: str
    company: str | None = None
    subsidiaries: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    amends: str | None = None
    entities: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str) or not isinstance(self.text, str):
            raise ValueError("document requires a nonempty string ID and string text")
        if any(
            not isinstance(value, str) or not value
            for value in [
                *self.subsidiaries,
                *self.citations,
                *self.entities,
            ]
        ):
            raise ValueError("reference IDs must be nonempty strings")


def extract_graph(
    tenant_id: str, documents: list[Document]
) -> tuple[list[tuple[str, str]], list[tuple[str, str, float, str]]]:
    """Create typed nodes and the five documented weighted edge types for one tenant."""
    if not tenant_id:
        raise ValueError("tenant_id is required")
    if len({doc.id for doc in documents}) != len(documents):
        raise ValueError("document IDs must be unique within a tenant")
    nodes = {}
    edges = {}

    def node(kind: str, identifier: str) -> str:
        key = f"{kind}:{identifier}"
        nodes[key] = kind
        return key

    def edge(src: str, dst: str, weight: float, kind: str) -> None:
        edges[(src, dst, kind)] = edges.get((src, dst, kind), 0.0) + weight

    for doc in documents:
        filing = node("filing", doc.id)
        if doc.company:
            company = node("company", doc.company)
            edge(filing, company, 1.0, "filing_company")
            for subsidiary in sorted(set(doc.subsidiaries)):
                edge(company, node("company", subsidiary), 1.0, "company_subsidiary")
        for cited in sorted(set(doc.citations)):
            edge(filing, node("filing", cited), 0.7, "filing_citation")
        if doc.amends:
            edge(filing, node("filing", doc.amends), 1.0, "filing_amendment")
        entities = [node("entity", item) for item in sorted(set(doc.entities))]
        for left, right in combinations(entities, 2):
            edge(left, right, 0.3, "entity_cooccurrence")
            edge(right, left, 0.3, "entity_cooccurrence")
    return sorted(nodes.items()), [
        (src, dst, weight, kind) for (src, dst, kind), weight in sorted(edges.items())
    ]


@dataclass
class DocumentAnalysis:
    """A tenant's immutable mathematical snapshot at one data revision."""

    tenant_id: str
    documents: list[Document]
    index: NodeIndex
    transition: csr_matrix
    ranks: np.ndarray
    vectorizer: TfidfVectorizer | None
    matrix: csr_matrix
    model: LSAModel | None
    errors: np.ndarray
    threshold: float

    def top(self, limit: int = 20) -> list[dict]:
        """Return the largest global PageRank probabilities."""
        order = np.argsort(-self.ranks, kind="stable")[:limit]
        return [{"node_id": self.index.to_id(int(i)), "score": float(self.ranks[i])} for i in order]

    def related(self, query: str, limit: int = 10) -> list[dict]:
        """Seed PPR from the top five positive TF-IDF cosine matches and rank documents."""
        if not self.vectorizer or not query.strip():
            return []
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        seeds = {
            self.index.to_int(f"filing:{self.documents[i].id}"): float(scores[i])
            for i in np.argsort(-scores, kind="stable")[:5]
            if scores[i] > 0
        }
        if not seeds:
            return []
        ranks = personalized_pagerank(self.transition, seeds)
        rows = [
            {"document_id": doc.id, "score": float(ranks[self.index.to_int(f"filing:{doc.id}")])}
            for doc in self.documents
            if ranks[self.index.to_int(f"filing:{doc.id}")] > 0
        ]
        return sorted(rows, key=lambda item: (-item["score"], item["document_id"]))[:limit]

    def similar(self, document_id: str, limit: int = 10) -> list[dict]:
        """Return cosine neighbors of an existing document in the fitted latent space."""
        idx = self._document_index(document_id)
        if self.model is None:
            return []
        return [
            {"document_id": self.documents[i].id, "score": score}
            for i, score in similar(self.model, idx, limit)
        ]

    def anomaly(self, document_id: str) -> dict:
        """Return the orthogonal residual and whether it exceeds the calibrated threshold."""
        idx = self._document_index(document_id)
        score = float(self.errors[idx])
        return {
            "tenant_id": self.tenant_id,
            "document_id": document_id,
            "score": score,
            "threshold": self.threshold,
            "review": score > self.threshold,
        }

    def _document_index(self, document_id: str) -> int:
        for idx, doc in enumerate(self.documents):
            if doc.id == document_id:
                return idx
        raise KeyError(document_id)


def analyze_documents(
    tenant_id: str, documents: list[Document], threshold: float | None = None
) -> DocumentAnalysis:
    """Fit a tenant graph and TF-IDF/SVD snapshot, using only that tenant's documents."""
    nodes, edges = extract_graph(tenant_id, documents)
    matrix, index = build_adjacency(
        [(a, b, w) for a, b, w, _ in edges], NodeIndex([n for n, _ in nodes])
    )
    transition = to_transition(matrix)
    ranks = pagerank(transition)
    vectorizer, model = None, None
    text_matrix = csr_matrix((len(documents), 0), dtype=float)
    errors = np.zeros(len(documents))
    if documents:
        candidate = TfidfVectorizer(max_features=20_000, stop_words="english")
        # Empty/stop-word-only corpora have no latent geometry, but still have graph ranks.
        analyzer = candidate.build_analyzer()
        if any(analyzer(doc.text) for doc in documents):
            text_matrix = candidate.fit_transform([doc.text for doc in documents])
            vectorizer = candidate
            model = fit_lsa(text_matrix)
            errors = reconstruction_error(model, text_matrix)
    if threshold is None:
        threshold = anomaly_threshold(errors) if len(errors) else 0.0
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    return DocumentAnalysis(
        tenant_id,
        documents,
        index,
        transition,
        ranks,
        vectorizer,
        text_matrix,
        model,
        errors,
        float(threshold),
    )


def review_gate(tenant_id: str, document_id: str, record: dict, decision: dict) -> dict:
    """Keep an anomalous candidate out of auto-approval; never promote a record's status."""
    if (
        not tenant_id
        or decision.get("tenant_id") != tenant_id
        or decision.get("document_id") != document_id
    ):
        raise ValueError("a matching tenant-scoped anomaly decision is required")
    result = dict(record)
    if decision["review"] and result.get("status") != "rejected":
        result["status"] = "pending_review"
    return result
