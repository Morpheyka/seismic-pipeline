"""Day-to-day REM profile shape-shift features."""
from __future__ import annotations

import numpy as np


def _split_row_into_daily_profiles(row: np.ndarray, n_days: int) -> np.ndarray:
    """Split a concatenated (optionally NaN-padded) row into ``(n_days, n_points)``."""
    if n_days <= 0:
        raise ValueError(f"n_days must be > 0, got {n_days}")
    row = np.asarray(row, dtype=float).reshape(-1)
    finite = row[np.isfinite(row)]
    if finite.size == 0:
        raise ValueError("Cannot split an all-NaN REM profile row.")
    if finite.size < n_days:
        raise ValueError(
            f"Finite profile length {finite.size} < n_days={n_days}"
        )
    n_points = int(finite.size // n_days)
    if n_points <= 0:
        raise ValueError(
            f"Cannot form daily profiles: finite={finite.size}, n_days={n_days}"
        )
    usable = finite[: n_days * n_points]
    return usable.reshape(n_days, n_points)


class REMShapeShiftCalculator:
    """L1 day-to-day shape shift: sum_p |profile[d+1,p] - profile[d,p]|."""

    def __init__(self, fill_first: bool = False):
        self.fill_first = bool(fill_first)

    def compute(self, daily: np.ndarray) -> np.ndarray:
        daily = np.asarray(daily, dtype=float)
        if daily.ndim != 2:
            raise ValueError(f"daily must be 2D (n_days, n_points), got shape {daily.shape}")
        n_days = int(daily.shape[0])
        if n_days <= 1:
            out = np.zeros(0 if not self.fill_first else n_days, dtype=float)
            return out

        shifts = np.sum(np.abs(np.diff(daily, axis=0)), axis=1)
        if self.fill_first:
            return np.concatenate([np.array([0.0], dtype=float), shifts])
        return shifts


def compute_shape_shift_map(
    profiles: np.ndarray,
    *,
    n_days: int,
    fill_first: bool = False,
) -> np.ndarray:
    """Batch shape-shift for rows of concatenated daily profiles."""
    profiles = np.asarray(profiles, dtype=float)
    if profiles.ndim != 2:
        raise ValueError(f"profiles must be 2D, got shape {profiles.shape}")
    calc = REMShapeShiftCalculator(fill_first=fill_first)
    n_out = n_days if fill_first else max(n_days - 1, 0)
    out = np.full((profiles.shape[0], n_out), np.nan, dtype=float)
    for i in range(profiles.shape[0]):
        daily = _split_row_into_daily_profiles(profiles[i], n_days)
        out[i] = calc.compute(daily)
    return out
