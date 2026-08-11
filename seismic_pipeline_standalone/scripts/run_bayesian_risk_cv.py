#!/usr/bin/env python3
"""Fit Bayesian risk models with grouped CV and temporal holdout."""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import pymc as pm
except Exception as exc:  # pragma: no cover
    pm = None
    _PYMC_IMPORT_ERROR = exc
else:
    _PYMC_IMPORT_ERROR = None


@dataclass
class FoldResult:
    split: str
    model_name: str
    n_train: int
    n_test: int
    log_score: float
    brier: float
    roc_auc: float
    pr_auc: float
    ece_10bin: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bayesian risk model with grouped CV.")
    parser.add_argument(
        "--dataset-csv",
        default="./run_output_risk_model/risk_daily_dataset.csv",
        help="Input dataset produced by build_seismic_risk_dataset.py",
    )
    parser.add_argument(
        "--out-dir",
        default="./run_output_risk_model",
        help="Directory to write CV metrics and report JSON.",
    )
    parser.add_argument("--n-folds", type=int, default=5, help="Grouped outer CV folds.")
    parser.add_argument("--draws", type=int, default=600, help="Posterior draws per chain.")
    parser.add_argument("--tune", type=int, default=600, help="Tuning steps per chain.")
    parser.add_argument("--chains", type=int, default=2, help="MCMC chains.")
    parser.add_argument("--target-accept", type=float, default=0.9, help="NUTS target_accept.")
    return parser.parse_args()


def _roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1, dtype=float)
    rank_sum_pos = float(ranks[pos].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_prob)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.trapz(precision, recall))


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += abs(conf - acc) * (float(mask.sum()) / len(y_true))
    return ece


def _score_probs(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    eps = 1e-9
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    log_score = float(np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))
    brier = float(np.mean((y_true - y_prob) ** 2))
    return {
        "log_score": log_score,
        "brier": brier,
        "roc_auc": _roc_auc(y_true, y_prob),
        "pr_auc": _pr_auc(y_true, y_prob),
        "ece_10bin": _expected_calibration_error(y_true, y_prob, n_bins=10),
    }


def _group_kfold(groups: np.ndarray, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    uniq = np.array(sorted(set(groups.tolist())))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_id in range(min(n_folds, len(uniq))):
        test_groups = uniq[fold_id::n_folds]
        test_mask = np.isin(groups, test_groups)
        train_idx = np.flatnonzero(~test_mask)
        test_idx = np.flatnonzero(test_mask)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        folds.append((train_idx, test_idx))
    return folds


def _prepare_design(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train = df.iloc[train_idx][feature_cols].copy()
    test = df.iloc[test_idx][feature_cols].copy()
    med = train.median(axis=0, numeric_only=True)
    train = train.fillna(med)
    test = test.fillna(med)
    mu = train.mean(axis=0)
    sd = train.std(axis=0).replace(0.0, 1.0)
    x_train = ((train - mu) / sd).to_numpy(dtype=float)
    x_test = ((test - mu) / sd).to_numpy(dtype=float)
    return x_train, x_test


def _fit_predict_bayes(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    rat_train: np.ndarray,
    rat_test: np.ndarray,
    month_train: np.ndarray,
    month_test: np.ndarray,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    use_features: bool,
    use_calendar: bool,
) -> np.ndarray:
    if pm is None:
        raise RuntimeError(f"pymc is not available: {_PYMC_IMPORT_ERROR}")

    # Map categories.
    rat_levels = {v: i for i, v in enumerate(sorted(set(rat_train.tolist())))}
    month_levels = {v: i for i, v in enumerate(sorted(set(month_train.tolist())))}
    rat_idx_train = np.array([rat_levels[v] for v in rat_train], dtype=int)
    rat_idx_test = np.array([rat_levels.get(v, 0) for v in rat_test], dtype=int)
    month_idx_train = np.array([month_levels[v] for v in month_train], dtype=int)
    month_idx_test = np.array([month_levels.get(v, 0) for v in month_test], dtype=int)

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.5)
        eta_train = alpha

        if use_calendar:
            sigma_rat = pm.HalfNormal("sigma_rat", sigma=1.0)
            sigma_month = pm.HalfNormal("sigma_month", sigma=1.0)
            rat_offset = pm.Normal("rat_offset", mu=0.0, sigma=1.0, shape=len(rat_levels))
            month_offset = pm.Normal("month_offset", mu=0.0, sigma=1.0, shape=len(month_levels))
            rat_eff = pm.Deterministic("rat_eff", sigma_rat * rat_offset)
            month_eff = pm.Deterministic("month_eff", sigma_month * month_offset)
            eta_train = eta_train + rat_eff[rat_idx_train] + month_eff[month_idx_train]
        else:
            rat_eff = None
            month_eff = None

        if use_features:
            beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=x_train.shape[1])
            eta_train = eta_train + pm.math.dot(x_train, beta)
        else:
            beta = None

        pm.Bernoulli("y", logit_p=eta_train, observed=y_train)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            progressbar=False,
            return_inferencedata=True,
        )

    posterior = idata.posterior
    alpha_s = posterior["alpha"].values.reshape(-1, 1)
    eta_test = np.repeat(alpha_s, repeats=x_test.shape[0], axis=1)

    if use_calendar and rat_eff is not None and month_eff is not None:
        rat_s = posterior["rat_eff"].values.reshape(-1, len(rat_levels))
        month_s = posterior["month_eff"].values.reshape(-1, len(month_levels))
        eta_test = eta_test + rat_s[:, rat_idx_test] + month_s[:, month_idx_test]

    if use_features and beta is not None:
        beta_s = posterior["beta"].values.reshape(-1, x_train.shape[1])
        eta_test = eta_test + beta_s @ x_test.T

    probs = 1.0 / (1.0 + np.exp(-eta_test))
    return probs.mean(axis=0)


def _evaluate_split(
    *,
    split_name: str,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_cols: list[str],
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
) -> list[FoldResult]:
    x_train, x_test = _prepare_design(df, feature_cols, train_idx, test_idx)
    y_train = df.iloc[train_idx]["label_next24h"].to_numpy(dtype=int)
    y_test = df.iloc[test_idx]["label_next24h"].to_numpy(dtype=int)
    rat_train = df.iloc[train_idx]["rat_id"].astype(str).to_numpy()
    rat_test = df.iloc[test_idx]["rat_id"].astype(str).to_numpy()
    month_train = pd.to_datetime(df.iloc[train_idx]["sample_date"]).dt.month.to_numpy(dtype=int)
    month_test = pd.to_datetime(df.iloc[test_idx]["sample_date"]).dt.month.to_numpy(dtype=int)

    variants = [
        ("null", False, False, []),
        ("calendar", False, True, []),
        ("full", True, True, feature_cols),
        ("no_shape", True, True, [c for c in feature_cols if "shape_" not in c and "anomaly_" not in c and "pairwise_" not in c]),
    ]

    out: list[FoldResult] = []
    for model_name, use_feat, use_cal, model_cols in variants:
        if use_feat:
            x_train_m, x_test_m = _prepare_design(df, model_cols, train_idx, test_idx)
        else:
            x_train_m = np.zeros((len(train_idx), 0), dtype=float)
            x_test_m = np.zeros((len(test_idx), 0), dtype=float)
        probs = _fit_predict_bayes(
            x_train=x_train_m,
            y_train=y_train,
            x_test=x_test_m,
            rat_train=rat_train,
            rat_test=rat_test,
            month_train=month_train,
            month_test=month_test,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            use_features=use_feat,
            use_calendar=use_cal,
        )
        metrics = _score_probs(y_test.astype(float), probs.astype(float))
        out.append(
            FoldResult(
                split=split_name,
                model_name=model_name,
                n_train=len(train_idx),
                n_test=len(test_idx),
                log_score=metrics["log_score"],
                brier=metrics["brier"],
                roc_auc=metrics["roc_auc"],
                pr_auc=metrics["pr_auc"],
                ece_10bin=metrics["ece_10bin"],
            )
        )
    return out


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(args.dataset_csv).resolve())
    if "label_next24h" not in df.columns:
        raise ValueError("Dataset must contain label_next24h column.")

    feature_cols = [
        c
        for c in df.columns
        if c
        not in {
            "rat_id",
            "sample_date",
            "source_event_date",
            "group_event_date",
            "day_idx",
            "label_next24h",
            "is_quiet_control",
        }
    ]

    # Grouped outer CV by source event date.
    groups = df["group_event_date"].astype(str).to_numpy()
    cv_splits = _group_kfold(groups, n_folds=int(args.n_folds))

    results: list[FoldResult] = []
    for i, (train_idx, test_idx) in enumerate(cv_splits, start=1):
        split_name = f"grouped_cv_fold_{i}"
        results.extend(
            _evaluate_split(
                split_name=split_name,
                df=df,
                train_idx=train_idx,
                test_idx=test_idx,
                feature_cols=feature_cols,
                draws=int(args.draws),
                tune=int(args.tune),
                chains=int(args.chains),
                target_accept=float(args.target_accept),
            )
        )

    # Temporal holdout.
    sample_dates = pd.to_datetime(df["sample_date"]).dt.date
    cutoff = sorted(sample_dates.unique())[max(1, int(0.8 * len(sample_dates.unique()))) - 1]
    train_idx = np.flatnonzero(sample_dates <= cutoff)
    test_idx = np.flatnonzero(sample_dates > cutoff)
    if len(train_idx) > 0 and len(test_idx) > 0:
        results.extend(
            _evaluate_split(
                split_name="temporal_holdout",
                df=df,
                train_idx=train_idx,
                test_idx=test_idx,
                feature_cols=feature_cols,
                draws=int(args.draws),
                tune=int(args.tune),
                chains=int(args.chains),
                target_accept=float(args.target_accept),
            )
        )

    out_csv = out_dir / "risk_model_cv_metrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "model_name",
                "n_train",
                "n_test",
                "log_score",
                "brier",
                "roc_auc",
                "pr_auc",
                "ece_10bin",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "split": r.split,
                    "model_name": r.model_name,
                    "n_train": r.n_train,
                    "n_test": r.n_test,
                    "log_score": r.log_score,
                    "brier": r.brier,
                    "roc_auc": r.roc_auc,
                    "pr_auc": r.pr_auc,
                    "ece_10bin": r.ece_10bin,
                }
            )

    summary = {
        "dataset_csv": str(Path(args.dataset_csv).resolve()),
        "n_rows": int(len(df)),
        "class_balance": {
            "positive": int((df["label_next24h"] == 1).sum()),
            "negative": int((df["label_next24h"] == 0).sum()),
        },
        "n_grouped_cv_folds": len(cv_splits),
        "features_used": feature_cols,
        "metrics_csv": str(out_csv),
        "pymc_available": pm is not None,
    }
    (out_dir / "risk_model_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
