"""Canonical day-mask for 8-day REM windows (artifacts ∪ missing hypnograms).

Primary stratum: mask ON, window eligible iff n_valid_days >= MIN_VALID_DAYS (K=6).
Sensitivity mask OFF: same cohort, but artifact days return to the likelihood;
physically missing days stay NaN.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# Corrected artifact map (rev. 3). Days are 0-based within the 8-day window.
ARTIFACT_DAYS_BY_KEY: dict[tuple[str, str, str], set[int]] = {
    ("R2", "2022-11-07", "before"): {0},
    ("R2", "2023-05-03", "before"): {2, 3},
    ("R3", "2023-05-03", "before"): {2},
    # Corrected: was {3}; d1 and d2 are the broken days (d3 is high, keep).
    ("R2", "2023-04-21", "after_reversed"): {1, 2},
}

MIN_VALID_DAYS: int = 6
DEFAULT_WINDOW_DAYS: int = 8


def event_key(rat_id: str, event_date: str, direction: str) -> tuple[str, str, str]:
    """Normalize identifiers to the ARTIFACT_DAYS_BY_KEY key form."""
    date = str(event_date).strip().replace("_", "-")
    if len(date) >= 10:
        date = date[:10]
    return (str(rat_id).strip(), date, str(direction).strip())


def parse_missing_day_indices(
    window_dates: str | Iterable[str],
    missing_dates: str | Iterable[str] | float | None,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> set[int]:
    """Map missing calendar dates onto 0-based day indices in the window."""
    if isinstance(window_dates, str):
        dates = [d.strip() for d in window_dates.split(";") if d.strip()]
    else:
        dates = [str(d).strip() for d in window_dates if str(d).strip()]
    if not dates:
        return set()

    if missing_dates is None or (isinstance(missing_dates, float) and np.isnan(missing_dates)):
        missing_list: list[str] = []
    elif isinstance(missing_dates, str):
        missing_list = [d.strip() for d in missing_dates.split(";") if d.strip()]
    else:
        missing_list = [str(d).strip() for d in missing_dates if str(d).strip()]

    def _norm(d: str) -> str:
        return d.replace("_", "-")[:10]

    missing_set = {_norm(d) for d in missing_list}
    out: set[int] = set()
    for i, d in enumerate(dates[: int(window_days)]):
        if _norm(d) in missing_set:
            out.add(i)
    return out


def artifact_day_indices(rat_id: str, event_date: str, direction: str) -> set[int]:
    """Return artifact day indices for one window key (may be empty)."""
    return set(ARTIFACT_DAYS_BY_KEY.get(event_key(rat_id, event_date, direction), set()))


def masked_day_indices(
    rat_id: str,
    event_date: str,
    direction: str,
    *,
    window_dates: str | Iterable[str] = "",
    missing_dates: str | Iterable[str] | float | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    apply_artifacts: bool = True,
    apply_missing: bool = True,
) -> set[int]:
    """Union of artifact and/or missing day indices for one window."""
    masked: set[int] = set()
    if apply_artifacts:
        masked |= artifact_day_indices(rat_id, event_date, direction)
    if apply_missing:
        masked |= parse_missing_day_indices(
            window_dates,
            missing_dates,
            window_days=window_days,
        )
    return {d for d in masked if 0 <= int(d) < int(window_days)}


def day_valid_mask(
    rat_id: str,
    event_date: str,
    direction: str,
    *,
    window_dates: str | Iterable[str] = "",
    missing_dates: str | Iterable[str] | float | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    apply_artifacts: bool = True,
    apply_missing: bool = True,
    profile_day_all_nan: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask of shape (window_days,) — True where the day is valid."""
    masked = masked_day_indices(
        rat_id,
        event_date,
        direction,
        window_dates=window_dates,
        missing_dates=missing_dates,
        window_days=window_days,
        apply_artifacts=apply_artifacts,
        apply_missing=apply_missing,
    )
    valid = np.ones(int(window_days), dtype=bool)
    for d in masked:
        valid[int(d)] = False
    if profile_day_all_nan is not None:
        pad = np.asarray(profile_day_all_nan, dtype=bool).reshape(-1)
        n = min(int(window_days), pad.size)
        # True in profile_day_all_nan means the day slot is empty → invalid.
        valid[:n] &= ~pad[:n]
    return valid


def n_valid_days(valid: np.ndarray) -> int:
    return int(np.asarray(valid, dtype=bool).sum())


def apply_day_mask_to_profiles(
    profiles: np.ndarray,
    day_masks: list[set[int]] | np.ndarray,
    *,
    n_points_per_day: int,
    n_days: int = DEFAULT_WINDOW_DAYS,
) -> np.ndarray:
    """Copy profiles and set masked day slots to NaN (fixed-N layout)."""
    out = np.array(profiles, dtype=float, copy=True)
    total = int(n_days) * int(n_points_per_day)
    if out.ndim != 2 or out.shape[1] < total:
        raise ValueError(
            f"profiles shape {out.shape} incompatible with "
            f"n_days={n_days}, n_points_per_day={n_points_per_day}"
        )
    n_events = int(out.shape[0])
    if isinstance(day_masks, np.ndarray) and day_masks.dtype == bool:
        if day_masks.shape != (n_events, n_days):
            raise ValueError(
                f"boolean day_masks shape {day_masks.shape} != {(n_events, n_days)}"
            )
        for i in range(n_events):
            for d in range(n_days):
                if not bool(day_masks[i, d]):
                    lo = d * n_points_per_day
                    hi = lo + n_points_per_day
                    out[i, lo:hi] = np.nan
        return out

    if len(day_masks) != n_events:
        raise ValueError(f"len(day_masks)={len(day_masks)} != n_events={n_events}")
    for i, masked in enumerate(day_masks):
        for d in masked:
            di = int(d)
            if 0 <= di < n_days:
                lo = di * n_points_per_day
                hi = lo + n_points_per_day
                out[i, lo:hi] = np.nan
    return out


def profile_empty_day_flags(
    profiles: np.ndarray,
    *,
    n_points_per_day: int,
    n_days: int = DEFAULT_WINDOW_DAYS,
) -> np.ndarray:
    """True where a day slot is entirely non-finite (missing / empty export)."""
    total = int(n_days) * int(n_points_per_day)
    slice_ = np.asarray(profiles, dtype=float)[:, :total]
    shaped = slice_.reshape(slice_.shape[0], n_days, n_points_per_day)
    return ~np.any(np.isfinite(shaped), axis=2)


def build_valid_day_matrix_from_metadata(
    meta: pd.DataFrame,
    profiles: np.ndarray | None = None,
    *,
    n_points_per_day: int | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    apply_artifacts: bool = True,
    apply_missing: bool = True,
) -> np.ndarray:
    """Build (n_rows, window_days) validity matrix aligned with exported metadata rows."""
    n = len(meta)
    empty_flags = None
    if profiles is not None and n_points_per_day is not None:
        empty_flags = profile_empty_day_flags(
            profiles,
            n_points_per_day=int(n_points_per_day),
            n_days=window_days,
        )
        if empty_flags.shape[0] != n:
            raise ValueError(
                f"profiles rows ({empty_flags.shape[0]}) != metadata rows ({n})"
            )

    valid = np.ones((n, int(window_days)), dtype=bool)
    for i, row in meta.reset_index(drop=True).iterrows():
        pad = empty_flags[int(i)] if empty_flags is not None else None
        valid[int(i)] = day_valid_mask(
            str(row["rat_id"]),
            str(row["event_date"]),
            str(row.get("window_direction", "before")),
            window_dates=str(row.get("window_dates", "")),
            missing_dates=row.get("missing_dates", ""),
            window_days=window_days,
            apply_artifacts=apply_artifacts,
            apply_missing=apply_missing,
            profile_day_all_nan=pad,
        )
    return valid


def ineligible_indices(
    valid_matrix: np.ndarray,
    *,
    min_valid_days: int = MIN_VALID_DAYS,
) -> list[int]:
    """Row indices with fewer than ``min_valid_days`` valid days."""
    counts = np.asarray(valid_matrix, dtype=bool).sum(axis=1)
    return [int(i) for i, c in enumerate(counts) if int(c) < int(min_valid_days)]


def primary_cohort_bad_indices(
    meta: pd.DataFrame,
    profiles: np.ndarray | None = None,
    *,
    n_points_per_day: int | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_valid_days: int = MIN_VALID_DAYS,
) -> list[int]:
    """Bad indices for primary ranking: mask ON (artifacts ∪ missing), K threshold."""
    valid = build_valid_day_matrix_from_metadata(
        meta,
        profiles,
        n_points_per_day=n_points_per_day,
        window_days=window_days,
        apply_artifacts=True,
        apply_missing=True,
    )
    return ineligible_indices(valid, min_valid_days=min_valid_days)
