"""
Shared date parsing utilities for seismic pipeline.

Normalizes handling of YYYY_MM_DD and YYYY-MM-DD formats across modules.
"""

from datetime import datetime


def parse_event_date(date_str: str, date_format: str = "%Y_%m_%d") -> datetime:
    """
    Parse event date string to datetime.

    Handles both underscore (YYYY_MM_DD) and dash (YYYY-MM-DD) formats.

    Parameters
    ----------
    date_str : str
        Date string in YYYY_MM_DD or YYYY-MM-DD format.
    date_format : str, default="%Y_%m_%d"
        Primary format to try.

    Returns
    -------
    datetime
        Parsed datetime object.

    Raises
    ------
    ValueError
        If date string cannot be parsed.
    """
    for fmt in (date_format, "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Could not parse date: {date_str!r}")


def normalize_date_to_yyyymmdd(date_str: str) -> str:
    """
    Normalize date string to YYYY_MM_DD format.

    Parameters
    ----------
    date_str : str
        Date string in various formats.

    Returns
    -------
    str
        Date in YYYY_MM_DD format.
    """
    try:
        if '_' in date_str:
            return date_str
        if '-' in date_str:
            return date_str.replace('-', '_')
        if len(date_str) >= 8:
            return f"{date_str[:4]}_{date_str[4:6]}_{date_str[6:8]}"
        return date_str
    except Exception:
        return date_str
