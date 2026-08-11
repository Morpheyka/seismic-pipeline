"""Path and environment-based defaults for seismic pipeline."""
from __future__ import annotations

import os


def local_data_root() -> str:
    """Default local data root from SEISMIC_LOCAL_DATA_ROOT or fallback."""
    return os.environ.get("SEISMIC_LOCAL_DATA_ROOT", "/mnt/wd/rat")
