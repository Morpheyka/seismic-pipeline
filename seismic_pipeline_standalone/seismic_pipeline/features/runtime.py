"""Runtime state for changepoint model runs (export/prepare configs, data_norm)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChangepointRunContext:
    """Explicit run context for changepoint workflows."""

    export_cfg: dict | None = None
    prepare_cfg: dict | None = None
    data_norm: np.ndarray | None = None
    data_raw: np.ndarray | None = None


_DEFAULT_CONTEXT = ChangepointRunContext()

# Legacy module-level names (imported by split submodules)
_RUNTIME_DATA_NORM: np.ndarray | None = None
_RUNTIME_DATA_RAW: np.ndarray | None = None
_RUNTIME_LAST_EXPORT_CFG: dict | None = None
_RUNTIME_LAST_PREPARE_CFG: dict | None = None


def _sync_from_context() -> None:
    global _RUNTIME_DATA_NORM, _RUNTIME_DATA_RAW, _RUNTIME_LAST_EXPORT_CFG, _RUNTIME_LAST_PREPARE_CFG
    _RUNTIME_DATA_NORM = _DEFAULT_CONTEXT.data_norm
    _RUNTIME_DATA_RAW = _DEFAULT_CONTEXT.data_raw
    _RUNTIME_LAST_EXPORT_CFG = _DEFAULT_CONTEXT.export_cfg
    _RUNTIME_LAST_PREPARE_CFG = _DEFAULT_CONTEXT.prepare_cfg


def get_context() -> ChangepointRunContext:
    return _DEFAULT_CONTEXT


def set_context(ctx: ChangepointRunContext) -> None:
    """Replace the active run context (preferred over individual setters)."""
    global _DEFAULT_CONTEXT
    _DEFAULT_CONTEXT = ctx
    _sync_from_context()


def set_runtime_data_norm(data_norm: np.ndarray) -> None:
    _DEFAULT_CONTEXT.data_norm = data_norm
    _sync_from_context()


def get_runtime_data_norm() -> np.ndarray | None:
    return _DEFAULT_CONTEXT.data_norm


def set_runtime_data_raw(data_raw: np.ndarray) -> None:
    _DEFAULT_CONTEXT.data_raw = data_raw
    _sync_from_context()


def get_runtime_data_raw() -> np.ndarray | None:
    return _DEFAULT_CONTEXT.data_raw


def set_runtime_export_cfg(cfg: dict) -> None:
    _DEFAULT_CONTEXT.export_cfg = dict(cfg)
    _sync_from_context()


def get_runtime_export_cfg() -> dict | None:
    return _DEFAULT_CONTEXT.export_cfg


def set_runtime_prepare_cfg(cfg: dict) -> None:
    _DEFAULT_CONTEXT.prepare_cfg = dict(cfg)
    _sync_from_context()


def get_runtime_prepare_cfg() -> dict | None:
    return _DEFAULT_CONTEXT.prepare_cfg
