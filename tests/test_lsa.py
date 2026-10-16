import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from meridian_rank.lsa import anomaly_flags, fit_lsa, reconstruction_error, similar


def test_rank_residual_monotonic_and_full_rank_exact():
    X = csr_matrix(np.random.default_rng(3).normal(size=(12, 7)))
    previous = np.full(12, np.inf)
    for k in range(1, 8):
        error = reconstruction_error(fit_lsa(X, k=k), X)
        assert np.all(error <= previous + 1e-7)
        previous = error
    np.testing.assert_allclose(previous, 0, atol=1e-7)


def test_similarity_and_repeatability():
    X = TfidfVectorizer().fit_transform(
        [
            "apple orange fruit juice",
            "apple orange fruit smoothie",
            "server network routing packets",
            "routing network server traffic",
            "orange apple fruit",
        ]
    )
    model = fit_lsa(X, k=2, seed=42)
    assert similar(model, 0, 1)[0][0] in {1, 4}
    assert similar(model, 0, 4)[0][1] > similar(model, 0, 4)[-1][1]
    np.testing.assert_allclose(model.Vt, fit_lsa(X, k=2, seed=42).Vt)


def test_auto_energy_and_heldout_outliers():
    X = csr_matrix(np.diag([10.0, 2.0, 1.0]))
    assert fit_lsa(X).k == 1
    model = fit_lsa(csr_matrix(np.tile([1.0, 0, 0], (100, 1))), k=1)
    heldout = csr_matrix(np.vstack([np.tile([1, 0, 0], (100, 1)), np.tile([0, 1, 0], (10, 1))]))
    errors = reconstruction_error(model, heldout)
    # Calibrate on normal data; p99 on a contaminated batch would mask anomalies.
    assert np.all(errors[-10:] > 0.5)
    assert np.all(errors[:100] == 0)
    assert anomaly_flags(np.r_[np.zeros(100), 5]).sum() == 1


@pytest.mark.parametrize("X", [csr_matrix((0, 2)), csr_matrix([[np.nan]])])
def test_invalid_inputs(X):
    with pytest.raises(ValueError):
        fit_lsa(X)


def test_tiny_and_zero_matrices():
    for X in [csr_matrix([[1.0]]), csr_matrix((3, 4)), csr_matrix([[1, 2, 3]])]:
        model = fit_lsa(X)
        np.testing.assert_allclose(reconstruction_error(model, X), 0, atol=1e-7)
