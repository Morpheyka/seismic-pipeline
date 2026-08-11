"""Changepoint model defaults and presets."""
from __future__ import annotations

import os
import warnings
from typing import Dict, List

from .paths import local_data_root


DEFAULT_EVENTS_10D: List[Dict[str, str]] = [
    {"rat_id": "R2", "date": "2022-11-07"},
    {"rat_id": "R2", "date": "2022-11-18"},
    {"rat_id": "R2", "date": "2023-04-03"},
    {"rat_id": "R2", "date": "2023-05-03"},
    {"rat_id": "R2", "date": "2024-09-30"},
    {"rat_id": "R2", "date": "2024-10-29"},
    {"rat_id": "R3", "date": "2025-01-23"},
    {"rat_id": "R3", "date": "2025-03-14"},
    {"rat_id": "R1", "date": "2025-07-02"},
    {"rat_id": "R2", "date": "2025-07-02"},
    {"rat_id": "R3", "date": "2025-07-02"},
    {"rat_id": "R4", "date": "2025-07-02"},
    {"rat_id": "R1", "date": "2025-07-20"},
    {"rat_id": "R2", "date": "2025-07-20"},
    {"rat_id": "R3", "date": "2025-07-20"},
    {"rat_id": "R4", "date": "2025-07-20"},
]

FULL_EXHAUSTIVE_EVENTS_8DAY: List[Dict[str, str]] = [
    {"rat_id": "R2", "date": "2022-11-07", "direction": "before"},
    {"rat_id": "R2", "date": "2022-11-18", "direction": "before"},
    {"rat_id": "R2", "date": "2023-04-03", "direction": "before"},
    {"rat_id": "R2", "date": "2023-04-18", "direction": "before"},
    {"rat_id": "R2", "date": "2023-05-03", "direction": "before"},
    {"rat_id": "R3", "date": "2023-05-03", "direction": "before"},
    {"rat_id": "R3", "date": "2024-09-30", "direction": "before"},
    {"rat_id": "R2", "date": "2024-10-29", "direction": "before"},
    {"rat_id": "R3", "date": "2025-01-23", "direction": "before"},
    {"rat_id": "R3", "date": "2025-03-14", "direction": "before"},
    {"rat_id": "R1", "date": "2025-07-02", "direction": "before"},
    {"rat_id": "R2", "date": "2025-07-02", "direction": "before"},
    {"rat_id": "R3", "date": "2025-07-02", "direction": "before"},
    {"rat_id": "R4", "date": "2025-07-02", "direction": "before"},
    {"rat_id": "R1", "date": "2025-07-20", "direction": "before"},
    {"rat_id": "R2", "date": "2025-07-20", "direction": "before"},
    {"rat_id": "R3", "date": "2025-07-20", "direction": "before"},
    {"rat_id": "R4", "date": "2025-07-20", "direction": "before"},
    {"rat_id": "R2", "date": "2022-11-07", "direction": "after_reversed"},
    {"rat_id": "R2", "date": "2022-11-18", "direction": "after_reversed"},
    {"rat_id": "R2", "date": "2023-04-03", "direction": "after_reversed"},
    {"rat_id": "R2", "date": "2023-04-11", "direction": "after_reversed"},
    {"rat_id": "R2", "date": "2023-04-21", "direction": "after_reversed"},
    {"rat_id": "R2", "date": "2024-09-30", "direction": "after_reversed"},
    {"rat_id": "R3", "date": "2024-09-30", "direction": "after_reversed"},
    {"rat_id": "R2", "date": "2024-10-29", "direction": "after_reversed"},
    {"rat_id": "R3", "date": "2025-01-23", "direction": "after_reversed"},
    {"rat_id": "R3", "date": "2025-03-14", "direction": "after_reversed"},
    {"rat_id": "R1", "date": "2025-07-02", "direction": "after_reversed"},
    {"rat_id": "R2", "date": "2025-07-02", "direction": "after_reversed"},
    {"rat_id": "R3", "date": "2025-07-02", "direction": "after_reversed"},
    {"rat_id": "R4", "date": "2025-07-02", "direction": "after_reversed"},
    {"rat_id": "R1", "date": "2025-07-20", "direction": "after_reversed"},
    {"rat_id": "R2", "date": "2025-07-20", "direction": "after_reversed"},
    {"rat_id": "R3", "date": "2025-07-20", "direction": "after_reversed"},
    {"rat_id": "R4", "date": "2025-07-20", "direction": "after_reversed"},
]


def s3_config_from_env() -> Dict[str, str]:
    """Build S3 client config from environment variables."""
    return {
        "service_name": "s3",
        "endpoint_url": os.environ.get("S3_ENDPOINT", "http://10.132.230.2:7770"),
        "aws_access_key_id": os.environ.get("S3_ACCESS_KEY_ID", "quantum"),
        "aws_secret_access_key": os.environ.get("S3_SECRET_ACCESS_KEY", "s3password"),
    }


DEFAULT_S3_CONFIG = s3_config_from_env()

FEATURE_SELECTION_PRESETS: dict[str, dict[str, list[str]]] = {
    "concat_mean_odd_range": {"concat": ["mean"], "odd": ["range"]},
    "concat_range_odd_mean": {"concat": ["range"], "odd": ["mean"]},
    "concat_mean_even_range": {"concat": ["mean"], "even": ["range"]},
    "odd_mean_even_range": {"odd": ["mean"], "even": ["range"]},
    "concat_mean_range": {"concat": ["mean", "range"]},
    "odd_even_mean_range": {"odd": ["mean", "range"], "even": ["mean", "range"]},
    "concat_shape_shift": {"concat": ["shape_shift"]},
    "odd_shape_shift": {"odd": ["shape_shift"]},
    "even_shape_shift": {"even": ["shape_shift"]},
    "odd_even_shape_shift": {"odd": ["shape_shift"], "even": ["shape_shift"]},
    "mean_and_shape_shift": {"concat": ["mean", "shape_shift"]},
    "all_metrics": {
        "concat": ["mean", "range", "shape_shift"],
        "even": ["mean", "shape_shift"],
        "odd": ["mean", "shape_shift"],
    },
}

LIKELIHOOD_CHOICES_BY_METRIC: dict[str, list[str]] = {
    "mean": ["student_t", "skew_normal"],
    "range": [
        "beta",
        "beta_constrained",
        "lognormal",
        "interval_inflated_beta",
        "zero_inflated_beta",
    ],
    "std": ["student_t", "lognormal", "gamma"],
    "shape_shift": ["lognormal", "gamma"],
}

PARAMETER_SELECTION_PRESETS: dict[str, dict[str, dict]] = {
    "range_zoib": {
        "range": {
            "likelihood": "zero_inflated_beta",
            "support_upper": 2.0,
            "pi_prior": {"dist": "beta", "alpha": 1.0, "beta": 10.0},
            "alpha_prior": {"dist": "gamma", "mu": 3.0, "sigma": 1.0},
            "beta_prior": {"dist": "gamma", "mu": 3.0, "sigma": 1.0},
            "eps": 1e-6,
        }
    },
    "range_iib": {
        "range": {
            "likelihood": "interval_inflated_beta",
            "threshold": 0.9,
            "support_upper": 2.0,
            "pi_prior": {"dist": "beta", "alpha": 1.0, "beta": 10.0},
            "alpha_prior": {"dist": "gamma", "mu": 3.0, "sigma": 1.0},
            "beta_prior": {"dist": "gamma", "mu": 3.0, "sigma": 1.0},
            "eps": 1e-6,
        }
    },
    "range_beta_constrained": {
        "range": {
            "likelihood": "beta_constrained",
            "support_upper": 2.0,
            "alpha_prior": {
                "dist": "gamma_offset",
                "mu": 2.0,
                "sigma": 1.0,
                "offset": 1.0,
            },
            "beta_prior": {
                "dist": "gamma_offset",
                "mu": 2.0,
                "sigma": 1.0,
                "offset": 1.0,
            },
            "eps": 1e-4,
        }
    },
}

REM_PROFILE_CHOICES: list[dict[str, float | int]] = [
    {"n_points_per_day": 12, "overlap": 0.0, "rem_stage": 2},
    {"n_points_per_day": 12, "overlap": 0.25, "rem_stage": 2},
    {"n_points_per_day": 12, "overlap": 0.5, "rem_stage": 2},
    {"n_points_per_day": 24, "overlap": 0.0, "rem_stage": 2},
    {"n_points_per_day": 24, "overlap": 0.25, "rem_stage": 2},
    {"n_points_per_day": 24, "overlap": 0.5, "rem_stage": 2},
    {"n_points_per_day": 48, "overlap": 0.0, "rem_stage": 2},
    {"n_points_per_day": 48, "overlap": 0.25, "rem_stage": 2},
    {"n_points_per_day": 48, "overlap": 0.5, "rem_stage": 2},
]


def validate_rem_profile_params(params: dict) -> dict:
    """Validate and normalize REM profile parameters."""
    if "n_points_per_day" in params:
        n = int(params["n_points_per_day"])
        overlap = float(params.get("overlap", 0.0))
        rem_stage = int(params.get("rem_stage", 2))

        if n < 4 or n > 96:
            raise ValueError(f"n_points_per_day must be between 4 and 96, got {n}")
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap must be in [0.0, 1.0), got {overlap}")

        return {
            "n_points_per_day": n,
            "overlap": overlap,
            "rem_stage": rem_stage,
        }
    if "window_size_hours" in params:
        w = float(params["window_size_hours"])
        s = float(params["step_size_hours"])
        rem_stage = int(params.get("rem_stage", 2))

        n_points = int(24.0 / s)
        overlap = 1.0 - (s / w) if w > s else 0.0

        warnings.warn(
            f"Deprecated REM profile params (window={w}h, step={s}h). "
            f"Converted to n_points={n_points}, overlap={overlap:.2f}",
            DeprecationWarning,
            stacklevel=2,
        )

        return {
            "n_points_per_day": n_points,
            "overlap": max(0.0, min(0.99, overlap)),
            "rem_stage": rem_stage,
        }
    raise ValueError(f"Invalid REM profile params: {params}")


def default_export_base_cfg(*, output_dir: str = ".") -> dict:
    """Default export configuration for parallel exhaustive search."""
    return {
        "events": list(DEFAULT_EVENTS_10D),
        "window_days": 8,
        "rem_stage": 2,
        "drop_incomplete_events": True,
        "local_data_root": local_data_root(),
        "s3_config": dict(DEFAULT_S3_CONFIG),
        "output_dir": str(output_dir),
    }


DEFAULT_EXPORT_BASE_CFG = default_export_base_cfg()
