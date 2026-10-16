"""CSV output and machine context for benchmark runs."""

import csv
import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


def write_results(name: str, rows: list[dict], context: dict | None = None) -> Path:
    """Save measured rows and dependency/platform metadata beside them."""
    root = Path("benchmarks/results")
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{name}.csv"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": {
            name: version(name) for name in ["numpy", "scipy", "networkx", "scikit-learn"]
        },
        "context": context or {},
    }
    target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(target.read_text())
    return target
