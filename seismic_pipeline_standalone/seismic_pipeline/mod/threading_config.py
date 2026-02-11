"""
Threading configuration for the seismic pipeline.

This module must be imported BEFORE numpy or any library that uses OMP/MKL/OPENBLAS
at import time. It has no heavy dependencies.
"""

import os


def configure_threading(cores: int = 60, threads_per_job: int = 1) -> None:
    """
    Set OMP/MKL/OPENBLAS/etc. env vars to limit threading.

    Call this BEFORE importing numpy or any library that uses these env vars
    at import time.

    Parameters
    ----------
    cores : int, default=60
        Max parallel jobs (e.g. for GridSearchCV).
    threads_per_job : int, default=1
        Threads per job to prevent oversubscription.
    """
    val = str(threads_per_job)
    os.environ['OMP_NUM_THREADS'] = val
    os.environ['MKL_NUM_THREADS'] = val
    os.environ['NUMEXPR_NUM_THREADS'] = val
    os.environ['OPENBLAS_NUM_THREADS'] = val
    os.environ['VECLIB_MAXIMUM_THREADS'] = val
    os.environ['BLIS_NUM_THREADS'] = val
    os.environ['TBB_NUM_THREADS'] = val
