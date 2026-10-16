"""Precision@10 from an explicitly labeled corpus JSON file; never invent real labels."""

import argparse
import json
from pathlib import Path
from uuid import UUID

import numpy as np

from meridian_rank.documents import Document, analyze_documents

from .common import write_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", type=Path, help="JSON with documents and >=20 {query,relevant} rows"
    )
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text())
    if len(data["queries"]) < 20:
        raise ValueError("at least 20 labeled queries are required")
    analysis = analyze_documents(str(UUID(int=1)), [Document(**doc) for doc in data["documents"]])
    if analysis.vectorizer is None:
        raise ValueError("dataset has no searchable text")
    known = {doc.id for doc in analysis.documents}
    rows = []
    for item in data["queries"]:
        relevant = set(item["relevant"])
        if not relevant or not relevant <= known:
            raise ValueError("each query needs nonempty relevance labels referencing corpus IDs")
        scores = (
            (analysis.matrix @ analysis.vectorizer.transform([item["query"]]).T).toarray().ravel()
        )
        keyword = [
            analysis.documents[i].id
            for i in np.argsort(-scores, kind="stable")[:10]
            if scores[i] > 0
        ]
        ppr = [
            row["document_id"] for row in analysis.related(item["query"], 10) if row["score"] > 0
        ]
        rows.append(
            {
                "query": item["query"],
                "keyword_precision_at_10": len(set(keyword) & relevant) / 10,
                "ppr_precision_at_10": len(set(ppr) & relevant) / 10,
            }
        )
    write_results(
        "search_quality",
        rows,
        {
            "dataset": str(args.dataset),
            "dataset_description": data.get("description", "user-supplied labels"),
            "queries": len(rows),
            "keyword_mean": float(np.mean([row["keyword_precision_at_10"] for row in rows])),
            "ppr_mean": float(np.mean([row["ppr_precision_at_10"] for row in rows])),
        },
    )


if __name__ == "__main__":
    main()
