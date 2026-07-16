import pickle

import numpy as np

from .hypnogram_validation import (
    align_hypnograms,
    compute_hypnogram_metrics,
    load_hypnogram_array,
)


def test_load_hypnogram_array_from_list_payload(tmp_path):
    payload = [np.array([0, 1, 2, 2, 1], dtype=int), {"sampling_rate": 250}]
    path = tmp_path / "ref_hypno.pickle"
    with path.open("wb") as fh:
        pickle.dump(payload, fh)

    loaded = load_hypnogram_array(path)
    np.testing.assert_array_equal(loaded, payload[0])


def test_align_hypnograms_reports_overlap_metadata():
    reference = np.array([0, 1, 2, 0, 1, 2, 2, 1], dtype=int)
    automatic = np.array([1, 2, 0, 1, 2], dtype=int)

    aligned = align_hypnograms(reference, automatic, reference_offset_epochs=2, automatic_offset_epochs=0)
    np.testing.assert_array_equal(aligned.reference, np.array([2, 0, 1, 2, 2], dtype=int))
    np.testing.assert_array_equal(aligned.automatic, automatic)
    assert aligned.metadata["aligned_length"] == 5
    assert aligned.metadata["reference_start"] == 2


def test_compute_hypnogram_metrics_shapes_and_keys():
    y_true = np.array([0, 0, 1, 1, 2, 2, 2, 1, 0, 2], dtype=int)
    y_pred = np.array([0, 1, 1, 1, 2, 0, 2, 1, 0, 2], dtype=int)

    metrics = compute_hypnogram_metrics(y_true, y_pred)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "cohen_kappa" in metrics
    assert "cohen_kappa_quadratic" in metrics
    assert "macro_f1" in metrics
    assert len(metrics["confusion_matrix"]) == 3
    assert set(metrics["per_class"].keys()) == {"0", "1", "2"}
    assert metrics["rem_minutes_absolute_error"] >= 0.0
