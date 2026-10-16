"""Held-out synthetic normal/odd residual evaluation, with separate calibration."""

import numpy as np
from scipy.sparse import csr_matrix

from meridian_rank.lsa import anomaly_threshold, fit_lsa, reconstruction_error

from .common import write_results


def main() -> None:
    rng = np.random.default_rng(42)
    basis = rng.normal(size=(3, 20))
    train = rng.normal(size=(300, 3)) @ basis + rng.normal(0, 0.01, (300, 20))
    calibration = rng.normal(size=(200, 3)) @ basis + rng.normal(0, 0.01, (200, 20))
    normal = rng.normal(size=(200, 3)) @ basis + rng.normal(0, 0.01, (200, 20))
    odd = rng.normal(size=(10, 20)) * 4
    model = fit_lsa(csr_matrix(train), k=3)
    threshold = anomaly_threshold(reconstruction_error(model, csr_matrix(calibration)))
    errors = reconstruction_error(model, csr_matrix(np.vstack([normal, odd])))
    labels = np.r_[np.zeros(len(normal), dtype=bool), np.ones(len(odd), dtype=bool)]
    flagged = errors > threshold
    true_positive = int((flagged & labels).sum())
    false_positive = int((flagged & ~labels).sum())
    write_results(
        "anomaly",
        [
            {
                "dataset": "synthetic_heldout_rank3",
                "normal": 200,
                "odd": 10,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "precision": true_positive / max(int(flagged.sum()), 1),
                "recall": true_positive / int(labels.sum()),
                "threshold": threshold,
            }
        ],
        {"seed": 42, "calibration_samples": 200, "training_samples": 300},
    )


if __name__ == "__main__":
    main()
