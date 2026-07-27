"""Data ingestion: every series enters the library already content-addressed.

See :mod:`mmar.io.loaders`. The invariant this subpackage exists to enforce is
that a :class:`~mmar.io.loaders.Series` cannot be constructed without a
:class:`~mmar.core.provenance.Provenance` binding the digest of the bytes it
came from, so no downstream root is ever a statement about anonymous data.
"""

from __future__ import annotations

from mmar.io.loaders import (
    Series,
    align_series,
    datetime64_to_days,
    days_to_datetime64,
    file_digest,
    load_array,
    load_csv,
    load_lockheed,
    resample_business_days,
)

__all__ = [
    "Series",
    "align_series",
    "datetime64_to_days",
    "days_to_datetime64",
    "file_digest",
    "load_array",
    "load_csv",
    "load_lockheed",
    "resample_business_days",
]
