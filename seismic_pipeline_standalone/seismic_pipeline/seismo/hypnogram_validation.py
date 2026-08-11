"""Validation helpers for automatic vs reference hypnograms."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class HypnogramAlignment:
    """Aligned vectors and explicit overlap metadata."""

    reference: np.ndarray
    automatic: np.ndarray
    metadata: Dict[str, int]


def load_hypnogram_array(path: str | Path) -> np.ndarray:
    """
    Load hypnogram from pickle path.

    Supported payloads:
    - ndarray
    - [ndarray, metadata_dict]
    - (ndarray, metadata_dict)
    """
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, np.ndarray):
        return np.asarray(payload).astype(int)
    if isinstance(payload, (list, tuple)) and len(payload) > 0:
        first = payload[0]
        if isinstance(first, np.ndarray):
            return np.asarray(first).astype(int)
    raise ValueError(f"Unsupported hypnogram payload format in {path}")


def align_hypnograms(
    reference: Sequence[int],
    automatic: Sequence[int],
    *,
    reference_offset_epochs: int = 0,
    automatic_offset_epochs: int = 0,
) -> HypnogramAlignment:
    """
    Align two hypnograms over the shared epoch range.

    Offsets are explicit (in epochs) and reflected in returned metadata.
    """
    ref = np.asarray(reference).astype(int)
    auto = np.asarray(automatic).astype(int)
    ref_start = max(0, int(reference_offset_epochs))
    auto_start = max(0, int(automatic_offset_epochs))
    overlap = min(ref.size - ref_start, auto.size - auto_start)
    if overlap <= 0:
        raise ValueError(
            "No overlapping epochs after offsets "
            f"(ref={ref.size}, auto={auto.size}, ref_offset={ref_start}, auto_offset={auto_start})"
        )
    ref_end = ref_start + overlap
    auto_end = auto_start + overlap
    return HypnogramAlignment(
        reference=ref[ref_start:ref_end],
        automatic=auto[auto_start:auto_end],
        metadata={
            "reference_length": int(ref.size),
            "automatic_length": int(auto.size),
            "reference_start": int(ref_start),
            "automatic_start": int(auto_start),
            "reference_end": int(ref_end),
            "automatic_end": int(auto_end),
            "aligned_length": int(overlap),
        },
    )


def rem_fraction_profile(
    hypnogram: Sequence[int],
    *,
    stage: int = 2,
    epoch_length_sec: int = 5,
    window_hours: int = 6,
    step_hours: int = 1,
) -> np.ndarray:
    """Compute Stage4-like sliding-window stage fraction profile."""
    arr = np.asarray(hypnogram).astype(int)
    per_hour = int(3600 / epoch_length_sec)
    window = max(1, int(window_hours * per_hour))
    step = max(1, int(step_hours * per_hour))
    if arr.size == 0:
        return np.array([], dtype=float)
    if arr.size < window:
        return np.array([float(np.mean(arr == stage))], dtype=float)
    vals = []
    for start in range(0, arr.size - window + 1, step):
        chunk = arr[start : start + window]
        vals.append(float(np.mean(chunk == stage)))
    return np.asarray(vals, dtype=float)


def compute_hypnogram_metrics(
    reference: Sequence[int],
    automatic: Sequence[int],
    *,
    labels: Iterable[int] = (0, 1, 2),
    epoch_length_sec: int = 5,
    rem_window_hours: int = 6,
    rem_step_hours: int = 1,
) -> Dict[str, object]:
    """Compute agreement metrics for aligned stage sequences."""
    y_true = np.asarray(reference).astype(int)
    y_pred = np.asarray(automatic).astype(int)
    labels_tuple: Tuple[int, ...] = tuple(int(v) for v in labels)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Expected aligned arrays of equal length, got {y_true.shape} and {y_pred.shape}"
        )
    acc = float(accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=list(labels_tuple)))
    kappa_q = float(
        cohen_kappa_score(y_true, y_pred, labels=list(labels_tuple), weights="quadratic")
    )
    p, r, f, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(labels_tuple),
        average=None,
        zero_division=0,
    )
    macro_f1 = float(np.mean(f)) if len(f) else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=list(labels_tuple))
    ref_rem_minutes = float(np.sum(y_true == 2) * epoch_length_sec / 60.0)
    auto_rem_minutes = float(np.sum(y_pred == 2) * epoch_length_sec / 60.0)
    rem_profile_ref = rem_fraction_profile(
        y_true,
        stage=2,
        epoch_length_sec=epoch_length_sec,
        window_hours=rem_window_hours,
        step_hours=rem_step_hours,
    )
    rem_profile_auto = rem_fraction_profile(
        y_pred,
        stage=2,
        epoch_length_sec=epoch_length_sec,
        window_hours=rem_window_hours,
        step_hours=rem_step_hours,
    )
    n_profile = min(rem_profile_ref.size, rem_profile_auto.size)
    rem_profile_mae = (
        float(np.mean(np.abs(rem_profile_ref[:n_profile] - rem_profile_auto[:n_profile])))
        if n_profile > 0
        else float("nan")
    )
    per_class = {}
    for idx, label in enumerate(labels_tuple):
        per_class[str(label)] = {
            "precision": float(p[idx]),
            "recall": float(r[idx]),
            "f1": float(f[idx]),
            "support": int(s[idx]),
        }
    return {
        "accuracy": acc,
        "cohen_kappa": kappa,
        "cohen_kappa_quadratic": kappa_q,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": list(labels_tuple),
        "rem_minutes_reference": ref_rem_minutes,
        "rem_minutes_automatic": auto_rem_minutes,
        "rem_minutes_absolute_error": abs(auto_rem_minutes - ref_rem_minutes),
        "rem_fraction_profile_mae": rem_profile_mae,
    }
