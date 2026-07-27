r"""Data ingestion with provenance from the first byte.

Traceability that begins *after* the data is loaded is worthless: the two
questions that actually get asked six months later are "which file was this?"
and "what did you do to the missing values?", and neither is answerable from an
array. Every loader here therefore returns a :class:`Series` whose
:class:`~mmar.core.provenance.Provenance` binds

* the **SHA-256 of the file bytes** (:func:`file_digest`) --- not the path, not
  the modification time, the bytes;
* the row count and the column names actually present in the file;
* the NaN policy and how many non-finite values it had to handle.

The content address of every downstream result therefore changes if the input
file changes by one byte, and does *not* change if the same bytes are moved to a
different directory. That asymmetry is deliberate and is what makes a root a
statement about a computation rather than about a filesystem.

Conventions
-----------
``Series.values`` is the raw observable as it appears in the file (a *price*,
for the datasets this library ships with). The transformations a multifractal
estimator actually consumes are derived views:

.. math::
    \text{log\_price}(t) = \ln P(t) - \ln P(0), \qquad
    \text{log\_returns}(t) = \ln P(t+1) - \ln P(t),

with :math:`\text{log\_price}(0) = 0` exactly, by construction. Times are
carried as **float days since the UNIX epoch** (1970-01-01), which is dense,
sortable, differencing-friendly and exact to well under a microsecond over any
range of historical market data; :func:`days_to_datetime64` inverts it.

Failure modes this module refuses to hide
-----------------------------------------
*Interpolation manufactures regularity.* Linear interpolation across a gap
inserts a locally :math:`C^1` segment whose Hölder exponent is 1, biasing every
estimated exponent *upward* and narrowing :math:`f(\alpha)`. It is available,
never default, and always counted in the provenance diagnostics.

*Forward-filling a holiday manufactures a zero return.* See
:func:`resample_business_days`.

*A non-overlapping increment count is a floor, and the floor is not free.*
:meth:`Series.increments` with ``overlapping=False`` returns exactly
:math:`\lfloor (n-1)/\delta \rfloor` increments --- which is *the same count*
strided slicing ``x[d::d] - x[:-d:d]`` gives, since both slices always have
length :math:`\lfloor (n-1)/\delta \rfloor`. Nothing here repairs legacy defect
D6: that defect is the floor itself. On this repository's dataset
(:math:`n = 7049`) over the legacy scale grid ``arange(2, 1000)``, the count
:math:`Z(0,\delta) = \lfloor (n-1)/\delta \rfloor` has log-log slope
:math:`-1.0179` rather than :math:`-1`, and no indexing convention can change
that. The cure is on the estimator side (restrict to divisors of :math:`n-1`,
or correct for the floor); see ``tests/unit/test_structure_function.py``. This
module only guarantees that the count is the honest floor and never an
accidental off-by-one.

References
----------
Mandelbrot, B., Fisher, A., Calvet, L. (1997). A multifractal model of asset
returns. *Cowles Foundation Discussion Paper* 1164. (Source of the Lockheed-type
daily-close design used as this library's regression fixture.)
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np
from numpy.typing import NDArray

from mmar.core.provenance import InputRef, Provenance, new_provenance
from mmar.core.types import FloatArray
from mmar.core.validate import DataQualityError, NanPolicy

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

_VERSION: Final = "1"
_DAY_NS: Final = 86_400_000_000_000
"""Nanoseconds per day; the conversion factor between ``datetime64[ns]`` and days."""

_SEPARATOR_CANDIDATES: Final = (",", "\t", ";", "|")
_SNIFF_BYTES: Final = 1 << 16
_SNIFF_LINES: Final = 20

LOCKHEED_FILENAME: Final = "lockheed_1995_2023.tsv"
LOCKHEED_ROWS: Final = 7049
LOCKHEED_FIRST_DATE: Final = "1995-03-17"
LOCKHEED_LAST_DATE: Final = "2023-03-17"

FillMethod = Literal["ffill", "nan"]


# --------------------------------------------------------------------------- #
# Digests and time conversion
# --------------------------------------------------------------------------- #


def file_digest(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of the raw bytes of a file, streamed.

    The digest is over the file *as stored*, before any parsing: it is invariant
    to where the file lives and sensitive to every byte in it, including the
    columns a given load ignores. That is exactly the identity a provenance
    record needs, since a change to an unread column is still a change to the
    dataset a reader would reproduce from.

    Parameters
    ----------
    path
        File to hash. Read in ``chunk_size`` blocks, so arbitrarily large files
        are fine.

    Returns
    -------
    str
        64 lower-case hex characters.
    """
    h = hashlib.sha256()
    p = Path(path)
    with p.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def datetime64_to_days(t: NDArray[np.datetime64]) -> FloatArray:
    """Convert ``datetime64`` timestamps to float days since 1970-01-01.

    Raises
    ------
    ValueError
        If any entry is ``NaT``. A missing timestamp cannot be silently mapped to
        a number: it would place an observation at an arbitrary point on the axis
        and corrupt every scale-dependent statistic computed from it.
    """
    ns = np.asarray(t, dtype="datetime64[ns]")
    if np.isnat(ns).any():
        bad = int(np.isnat(ns).sum())
        raise ValueError(
            f"{bad} unparseable timestamp(s) (NaT) in the time column; fix or drop those "
            "rows before loading, rather than letting them land at an arbitrary time"
        )
    return np.asarray(ns.astype(np.int64) / _DAY_NS, dtype=np.float64)


def days_to_datetime64(days: FloatArray) -> NDArray[np.datetime64]:
    """Inverse of :func:`datetime64_to_days`, to the nearest nanosecond.

    Float days lose resolution below roughly a microsecond for contemporary
    dates (``float64`` has 53 bits of mantissa and the epoch offset is
    :math:`\\sim 2\\times10^{4}` days), which is irrelevant for daily and
    intraday-minute data and would not be for tick data.
    """
    d = np.asarray(days, dtype=np.float64)
    return np.asarray(np.rint(d * _DAY_NS).astype(np.int64).astype("datetime64[ns]"))


# --------------------------------------------------------------------------- #
# The Series carrier
# --------------------------------------------------------------------------- #


def _frozen(arr: NDArray[Any]) -> FloatArray:
    """Return an owned, C-contiguous, write-protected ``float64`` copy.

    The copy is what makes the lock airtight: freezing a *view* leaves its base
    writable and reachable through ``.base``, so the guarantee would be
    cosmetic. One copy per :class:`Series` construction is the price of a
    content address that cannot be invalidated after the fact.
    """
    out = np.array(np.asarray(arr, dtype=np.float64), dtype=np.float64, order="C", copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class Series:
    r"""An observed univariate series and the audit record of where it came from.

    Parameters
    ----------
    values
        The observable as it appears in the source, coerced to contiguous
        ``float64``. For the datasets this library ships with, a *price*.
    t
        Observation times in float days since the UNIX epoch, strictly
        increasing, or ``None`` for an unindexed sample (in which case the
        implied index is :math:`0,1,\dots,n-1`).
    name
        Series label, e.g. ``"LMT.Close"``.
    source
        Human-readable origin, e.g. a path or ``"in-memory"``. Descriptive only:
        the *identity* of the data is the digest inside ``provenance``.
    provenance
        Content-addressed record. ``provenance.root`` is a function of the file
        digest (or the array digest), the parameters of the load and the
        environment, and of nothing else.

    Notes
    -----
    Equality on this dataclass compares numpy arrays elementwise and therefore
    raises on ambiguity, exactly as for the other array-carrying dataclasses in
    :mod:`mmar.core.types`. Compare ``provenance.root`` instead: two series are
    the same series iff they are the same computation.

    ``values`` and ``t`` are **owned, C-contiguous and write-protected**.
    ``frozen=True`` only stops rebinding the attribute; without the write lock
    ``s.values[0] = 1e9`` would change the data while ``provenance.root`` --- a
    digest taken at construction --- kept certifying the old bytes. A content
    address that can be falsified in place is worse than none, so the arrays are
    copied once on entry and frozen. Callers who need to modify a series build a
    new one through :func:`load_array`, which re-digests.
    """

    values: FloatArray
    t: FloatArray | None
    name: str
    source: str
    provenance: Provenance

    def __post_init__(self) -> None:
        v = _frozen(np.asarray(self.values, dtype=np.float64).ravel())
        object.__setattr__(self, "values", v)
        if v.size == 0:
            raise DataQualityError(f"series {self.name!r} is empty")
        if self.t is None:
            return
        ta = _frozen(np.asarray(self.t, dtype=np.float64).ravel())
        if ta.shape != v.shape:
            raise ValueError(f"t{ta.shape} and values{v.shape} must have the same length")
        if not np.all(np.isfinite(ta)):
            raise ValueError(f"series {self.name!r} has non-finite observation times")
        if ta.size > 1 and np.any(np.diff(ta) <= 0.0):
            n_bad = int(np.sum(np.diff(ta) <= 0.0))
            raise ValueError(
                f"series {self.name!r} has {n_bad} non-increasing time step(s); sort and "
                "de-duplicate the index before loading. Out-of-order timestamps make every "
                "increment at every scale meaningless."
            )
        object.__setattr__(self, "t", ta)

    # ---------------------------------------------------------------- views --

    @property
    def n(self) -> int:
        return int(self.values.size)

    @property
    def span(self) -> float:
        """Length of the observation window in days, or in samples when ``t`` is absent."""
        if self.t is None:
            return float(self.n - 1)
        return float(self.t[-1] - self.t[0])

    @property
    def log_price(self) -> FloatArray:
        r"""Normalised log price :math:`\ln P(t) - \ln P(0)`, starting at exactly ``0.0``.

        Subtracting the first log price is a change of numeraire: it removes an
        additive constant that no increment-based statistic can see, and it makes
        the first element exactly zero (``x - x`` in IEEE-754), which is a cheap
        invariant to assert on.

        Raises
        ------
        DataQualityError
            If any value is non-positive. A logarithm of a non-positive price is
            not a missing value to be patched around: it means the column is not
            a price (a return series, a spread, an adjusted level that went
            negative), and every downstream Hölder exponent would be a statement
            about a different object.
        """
        v = self.values
        bad = ~(v > 0.0)
        if bad.any():
            idx = np.nonzero(bad)[0]
            raise DataQualityError(
                f"series {self.name!r} has {int(bad.sum())} non-positive or non-finite "
                f"value(s) at indices {idx[:10].tolist()}"
                f"{'...' if idx.size > 10 else ''} (first offending value "
                f"{v[idx[0]]!r}); log_price is undefined there. If this column is not a "
                "price, use `values` or `increments` directly."
            )
        return np.asarray(np.log(v) - math.log(float(v[0])), dtype=np.float64)

    @property
    def log_returns(self) -> FloatArray:
        r""":math:`r_i = \ln P(t_{i+1}) - \ln P(t_i)`, of length :math:`n-1`.

        The first difference of :func:`log_price`; the normalising constant
        cancels. Irregular spacing is *not* corrected for --- these are returns
        per observation, not per unit time. Use :func:`resample_business_days`
        first if the sampling grid has gaps and the estimator assumes uniformity.
        """
        return np.asarray(np.diff(self.log_price), dtype=np.float64)

    def increments(self, delta: int = 1, *, overlapping: bool = True) -> FloatArray:
        r"""Increments :math:`\Delta_\delta X(t) = X(t+\delta) - X(t)` of ``values``.

        Parameters
        ----------
        delta
            Lag in samples, :math:`1 \le \delta \le n-1`.
        overlapping
            ``True`` returns all :math:`n-\delta` sliding increments: maximal
            sample size, but the increments overlap, so they are correlated and
            the effective number of independent observations is only
            :math:`n/\delta`. Confidence intervals computed as if they were
            independent are too narrow.
            ``False`` returns the :math:`\lfloor (n-1)/\delta \rfloor`
            non-overlapping increments, indexed explicitly: the trailing partial
            block is discarded. This is bit-for-bit what ``x[d::d] - x[:-d:d]``
            returns --- both strided slices have length
            :math:`\lfloor (n-1)/\delta \rfloor` for every :math:`(n,\delta)`,
            so explicit indexing buys clarity, not a different answer.

            **It does not fix legacy defect D6.** :math:`Z(0,\delta)` is the
            floored count, so on a scale grid containing non-divisors of
            :math:`n-1` its log-log slope is not :math:`-1`
            (:math:`-1.0179` at :math:`n = 7049` over ``arange(2, 1000)``). Any
            estimator that reads :math:`\hat\tau(0)` as a self-consistency check
            must handle the floor itself.

        Returns
        -------
        FloatArray
            The increments of the raw ``values``. For a price series, compose
            with :attr:`log_price` first if log-increments are wanted.
        """
        d = int(delta)
        if d < 1 or d > self.n - 1:
            raise ValueError(f"delta must satisfy 1 <= delta <= n-1 = {self.n - 1}, got {delta!r}")
        v = self.values
        if overlapping:
            return np.asarray(v[d:] - v[:-d], dtype=np.float64)
        m = (self.n - 1) // d
        idx = np.arange(m + 1, dtype=np.int64) * d
        return np.asarray(v[idx[1:]] - v[idx[:-1]], dtype=np.float64)

    def times_as_datetime64(self) -> NDArray[np.datetime64]:
        """Observation times as ``datetime64[ns]``; raises when the series is unindexed."""
        if self.t is None:
            raise ValueError(f"series {self.name!r} carries no time index")
        return days_to_datetime64(self.t)

    # ------------------------------------------------------------- summary --

    def summary(self) -> dict[str, Any]:
        r"""Scalar description of the series, suitable for a run manifest.

        Returns
        -------
        dict
            ``n``, ``span`` (days, or samples when unindexed), ``returns_basis``,
            ``mean_return``, ``std_return``, ``kurtosis`` (excess, i.e. Fisher:
            :math:`m_4/m_2^2 - 3`, zero for a Gaussian), ``n_nonfinite`` and the
            provenance ``root``.

        Notes
        -----
        ``returns_basis`` is ``"log"`` when every value is strictly positive and
        ``"difference"`` otherwise. Reporting the basis matters: the excess
        kurtosis of *log* returns is the number quoted in the multifractal
        literature (:math:`\approx 8.8` for this repository's Lockheed close),
        and it is not the same statistic as that of raw price differences.
        """
        v = self.values
        finite = np.isfinite(v)
        if bool(np.all(finite)) and bool(np.all(v > 0.0)):
            r = self.log_returns
            basis = "log"
        else:
            r = np.asarray(np.diff(v), dtype=np.float64)
            basis = "difference"
        r = r[np.isfinite(r)]
        mean = float(np.mean(r)) if r.size else float("nan")
        std = float(np.std(r, ddof=1)) if r.size > 1 else float("nan")
        m2 = float(np.mean((r - mean) ** 2)) if r.size else float("nan")
        m4 = float(np.mean((r - mean) ** 4)) if r.size else float("nan")
        kurt = (m4 / m2**2 - 3.0) if (r.size > 3 and m2 > 0.0) else float("nan")
        return {
            "name": self.name,
            "n": self.n,
            "span": self.span,
            "returns_basis": basis,
            "mean_return": mean,
            "std_return": std,
            "kurtosis": kurt,
            "n_nonfinite": int((~finite).sum()),
            "root": self.provenance.root,
        }

    def __str__(self) -> str:
        return f"Series({self.name!r}, n={self.n}, root={self.provenance.short_root})"


# --------------------------------------------------------------------------- #
# NaN handling
# --------------------------------------------------------------------------- #


def _apply_nan_policy(
    values: FloatArray, policy: str, *, name: str
) -> tuple[FloatArray, NDArray[np.bool_], int]:
    """Apply a NaN policy and report what it did.

    Returns ``(values, keep_mask, n_nonfinite)``. ``keep_mask`` indexes the
    *original* rows that survive, so any parallel column (a time index) can be
    filtered identically.

    ``policy`` is typed ``str`` rather than :data:`~mmar.core.validate.NanPolicy`
    so that the unknown-policy branch stays reachable for callers who reach this
    library from untyped code.
    """
    if policy not in ("raise", "drop", "interpolate"):
        raise ValueError(f"unknown nan_policy {policy!r}; use 'raise', 'drop' or 'interpolate'")
    bad = ~np.isfinite(values)
    n_bad = int(bad.sum())
    keep = np.ones(values.size, dtype=bool)
    if n_bad == 0:
        return values, keep, 0

    idx = np.nonzero(bad)[0]
    if policy == "raise":
        raise DataQualityError(
            f"{name} contains {n_bad} non-finite value(s) at row(s) "
            f"{idx[:10].tolist()}{'...' if idx.size > 10 else ''}. Pass "
            "nan_policy='drop' (removes the rows, leaving a gapped time index) or "
            "nan_policy='interpolate' (fills linearly, which manufactures local "
            "regularity and biases Hoelder exponents upward) to state explicitly "
            "how they should be handled."
        )
    if policy == "drop":
        keep = ~bad
        return np.ascontiguousarray(values[keep]), keep, n_bad
    if bad[0] or bad[-1]:
        raise DataQualityError(
            f"{name} has a non-finite value at the first or last row; linear "
            "interpolation would extrapolate a constant there, fabricating data "
            "outside the observed support. Trim the ends or use nan_policy='drop'."
        )
    pos = np.arange(values.size, dtype=np.float64)
    filled = np.interp(pos, pos[~bad], values[~bad])
    return np.ascontiguousarray(filled, dtype=np.float64), keep, n_bad


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def load_array(
    values: NDArray[Any] | list[float] | tuple[float, ...],
    *,
    name: str,
    t: NDArray[Any] | list[float] | tuple[float, ...] | None = None,
    source: str = "in-memory",
    nan_policy: NanPolicy = "raise",
) -> Series:
    """Wrap in-memory data in a :class:`Series` with an array-digest provenance.

    The identity of an in-memory series is the content address of its bytes
    (:func:`~mmar.core.provenance.array_digest`), so a simulated path and a
    reloaded copy of it share a root, and any modification breaks it.

    Parameters
    ----------
    values, t
        Coerced to ``float64``. ``t`` is in float days since the epoch when it is
        meant to be calendar time, but nothing here enforces that: it only has to
        be strictly increasing.
    nan_policy
        Applied before the digest is taken, so the recorded digest is the digest
        of the data as returned.
    """
    v = np.ascontiguousarray(np.asarray(values, dtype=np.float64).ravel())
    cleaned, keep, n_bad = _apply_nan_policy(v, nan_policy, name=name)
    ta: FloatArray | None = None
    if t is not None:
        ta = np.ascontiguousarray(np.asarray(t, dtype=np.float64).ravel())
        if ta.shape != v.shape:
            raise ValueError(f"t{ta.shape} and values{v.shape} must have the same length")
        ta = np.ascontiguousarray(ta[keep])

    inputs = [InputRef.from_array("values", cleaned)]
    if ta is not None:
        inputs.append(InputRef.from_array("t", ta))
    prov = new_provenance(
        "io.load_array",
        estimator_version=_VERSION,
        params={
            "name": name,
            "n": int(cleaned.size),
            "has_time": ta is not None,
            "nan_policy": nan_policy,
        },
        inputs=inputs,
        seed=None,
    ).with_diagnostics(n_nonfinite=n_bad, source=source)
    return Series(values=cleaned, t=ta, name=name, source=source, provenance=prov)


def _sniff_separator(sample: str) -> str:
    """Pick the delimiter that splits the sample into a consistent, wide table.

    Deterministic by construction: a candidate scores the number of fields it
    produces, but only if *every* inspected line yields the same count (a real
    delimiter partitions a rectangular file; a character that merely occurs in
    the text does not). Ties break in the order comma, tab, semicolon, pipe.
    """
    lines = [ln for ln in sample.splitlines()[:_SNIFF_LINES] if ln.strip()]
    if not lines:
        raise DataQualityError("the file is empty; nothing to sniff")
    best: tuple[int, str] | None = None
    for cand in _SEPARATOR_CANDIDATES:
        counts = {ln.count(cand) for ln in lines}
        if len(counts) != 1:
            continue
        n_sep = counts.pop()
        if n_sep < 1:
            continue
        if best is None or n_sep > best[0]:
            best = (n_sep, cand)
    if best is None:
        raise DataQualityError(
            "could not infer a separator: none of "
            f"{[repr(c) for c in _SEPARATOR_CANDIDATES]} splits the first "
            f"{len(lines)} lines into a consistent number of fields. Pass sep= "
            "explicitly (whitespace-delimited files need sep=r'\\s+')."
        )
    return best[1]


def load_csv(
    path: str | Path,
    *,
    value_column: str,
    time_column: str | None = None,
    sep: str | None = None,
    nan_policy: NanPolicy = "raise",
    name: str | None = None,
    source: str | None = None,
    notes: tuple[str, ...] = (),
) -> Series:
    r"""Load one column of a delimited text file into a :class:`Series`.

    Parameters
    ----------
    path
        Delimited file **with a header row**; ``value_column`` and
        ``time_column`` are column names, not positions.
    value_column
        Column to load as the observable. Non-numeric entries are coerced to
        ``NaN`` and then handled by ``nan_policy``; the number that failed to
        parse is reported separately from the number that were already empty, in
        the provenance diagnostics, because the two mean different things.
    time_column
        Optional; parsed with :func:`pandas.to_datetime` and stored as float days
        since 1970-01-01. The result must be strictly increasing (see
        :class:`Series`).
    sep
        Field delimiter. ``None`` sniffs it from the first
        :data:`_SNIFF_LINES` non-blank lines (see :func:`_sniff_separator`).
    nan_policy
        ``"raise"`` (default), ``"drop"`` or ``"interpolate"``.

    Returns
    -------
    Series
        With a provenance whose ``root`` binds the **file digest**, the row
        count, the file's column names, the chosen separator, the NaN policy and
        the series label. Nothing about the *filesystem* is identifying: neither
        the directory nor the basename appears in ``params`` (both are in the
        non-identifying diagnostics, as ``path`` and ``filename``). The same
        bytes therefore produce the same root wherever they live and whatever
        they are called, and different bytes never do.

        Note the one path-shaped dependency that remains, and is deliberate:
        when ``name`` is not given the label defaults to
        ``f"{path.stem}.{value_column}"``, and the label *is* identifying.
        Renaming a file therefore moves the root unless ``name=`` is passed
        explicitly --- which is what :func:`load_lockheed` does.

    Raises
    ------
    ImportError
        If pandas is not installed. It is an optional dependency
        (``pip install 'mmar[data]'``); :func:`load_array` needs nothing beyond
        numpy.
    DataQualityError
        On a missing column, a non-finite value under ``nan_policy="raise"``, or
        an unsniffable separator.
    """
    try:
        import pandas as pd  # noqa: PLC0415 - optional dependency, probed lazily
    except ImportError as exc:  # pragma: no cover - exercised only without pandas
        raise ImportError(
            "mmar.io.load_csv requires pandas, which is an optional dependency of mmar. "
            "Install it with `pip install 'mmar[data]'` or `uv sync --extra data`. "
            "For data already in memory use mmar.io.load_array, which needs only numpy."
        ) from exc

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such data file: {p}")

    digest = file_digest(p)
    if sep is None:
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            sep = _sniff_separator(fh.read(_SNIFF_BYTES))

    frame = pd.read_csv(p, sep=sep, engine="python")
    columns = tuple(str(c) for c in frame.columns)
    n_rows = int(frame.shape[0])
    for col in (value_column, time_column):
        if col is not None and col not in columns:
            raise DataQualityError(f"column {col!r} not in {p.name}; available columns: {columns}")

    raw = frame[value_column]
    n_missing_in_file = int(raw.isna().sum())
    numeric = pd.to_numeric(raw, errors="coerce")
    v = np.ascontiguousarray(np.asarray(numeric.to_numpy(), dtype=np.float64).ravel())
    # Entries that were present in the file but did not parse as a number: NaN after
    # coercion minus those already empty. Distinguishing the two matters — an empty
    # cell is a gap, an unparseable cell is a schema error.
    n_unparsed = int(np.sum(np.isnan(v))) - n_missing_in_file
    # Infinities parse fine and are neither a gap nor a schema error, but they are
    # non-finite, so without this term n_nonfinite would not reconcile with its parts.
    n_infinite = int(np.sum(np.isinf(v)))

    cleaned, keep, n_bad = _apply_nan_policy(v, nan_policy, name=f"{p.name}:{value_column}")

    ta: FloatArray | None = None
    if time_column is not None:
        stamps = np.asarray(
            pd.to_datetime(frame[time_column]).to_numpy(dtype="datetime64[ns]"),
            dtype="datetime64[ns]",
        )
        ta = np.ascontiguousarray(datetime64_to_days(stamps)[keep])

    label = name if name is not None else f"{p.stem}.{value_column}"
    inputs = [
        # The ref is named "file", not f"file:{p.name}": an InputRef's *name* is part
        # of its canonical form, so putting the basename there would smuggle the
        # filesystem back into the identity that params was just cleaned of.
        InputRef(name="file", digest=digest, shape=(), dtype="text/sha256"),
        InputRef.from_array("values", cleaned),
    ]
    if ta is not None:
        inputs.append(InputRef.from_array("t", ta))

    prov = new_provenance(
        "io.load_csv",
        estimator_version=_VERSION,
        params={
            "name": label,
            "value_column": value_column,
            "time_column": time_column,
            "sep": sep,
            "nan_policy": nan_policy,
            "n_rows": n_rows,
            "columns": list(columns),
        },
        inputs=inputs,
        seed=None,
        notes=notes,
    ).with_diagnostics(
        path=str(p.resolve()),
        filename=p.name,
        file_sha256=digest,
        n_rows=n_rows,
        n_values=int(cleaned.size),
        n_nonfinite=n_bad,
        n_missing_in_file=n_missing_in_file,
        n_unparsed=max(n_unparsed, 0),
        n_infinite_in_file=n_infinite,
        nan_policy=nan_policy,
        columns=list(columns),
    )
    return Series(
        values=cleaned,
        t=ta,
        name=label,
        source=source if source is not None else str(p),
        provenance=prov,
    )


def _lockheed_path() -> Path:
    """Locate the repository's own dataset.

    Resolved relative to this source file (``<repo>/src/mmar/io/loaders.py`` →
    ``<repo>/data/``), with the ``MMAR_DATA_DIR`` environment variable taking
    precedence. The dataset is part of the repository, not of the wheel, so an
    installed copy of the library will not find it and must be given a path.
    """
    override = os.environ.get("MMAR_DATA_DIR")
    if override:
        return Path(override) / LOCKHEED_FILENAME
    return Path(__file__).resolve().parents[3] / "data" / LOCKHEED_FILENAME


def load_lockheed(
    *,
    column: str = "Close",
    path: str | Path | None = None,
    nan_policy: NanPolicy = "raise",
) -> Series:
    """Load the repository's canonical regression fixture: Lockheed Martin daily bars.

    Tab-separated, columns ``Date, Close, Open, High, Low, Volume``,
    7049 rows spanning 1995-03-17 to 2023-03-17. This is the series the original
    thesis analysed, so it is the fixture against which v2 results are compared
    to v1's; its excess kurtosis of log returns is :math:`\\approx 8.8`, which is
    the specific number the legacy simulator failed to reproduce (it returned
    0.064 — see ``docs/LEGACY_AUDIT.md``, defect D1).

    Parameters
    ----------
    column
        Which bar field to take; ``"Close"`` by default.
    path
        Override for the file location. Needed when ``mmar`` is installed from a
        wheel, which does not ship the data directory.

    Raises
    ------
    FileNotFoundError
        With the resolved path, when the dataset is not on disk.
    """
    p = Path(path) if path is not None else _lockheed_path()
    if not p.is_file():
        raise FileNotFoundError(
            f"the Lockheed fixture is not at {p}. It ships with the repository, not with "
            "the wheel: clone the repository, pass path=..., or set MMAR_DATA_DIR."
        )
    return load_csv(
        p,
        value_column=column,
        time_column="Date",
        sep="\t",
        nan_policy=nan_policy,
        name=f"LMT.{column}",
        source=f"repository fixture {LOCKHEED_FILENAME}",
        notes=(
            f"Lockheed Martin daily {column}, {LOCKHEED_FIRST_DATE} to {LOCKHEED_LAST_DATE}",
            "canonical regression fixture of the v1 thesis",
        ),
    )


# --------------------------------------------------------------------------- #
# Resampling and alignment
# --------------------------------------------------------------------------- #


def resample_business_days(series: Series, *, method: FillMethod = "ffill") -> Series:
    """Place a daily series on the complete Monday-to-Friday grid it spans.

    Parameters
    ----------
    series
        Must carry a time index. Each timestamp is floored to its calendar day,
        so this is a *daily* operation; two observations on the same day collapse
        to the last one, counted in the ``n_observations_collapsed`` diagnostic.
        Observations whose day is not a business day are dropped entirely and
        counted in ``n_observations_off_grid``.
    method
        ``"ffill"`` carries the last observation forward across gaps;
        ``"nan"`` leaves them missing, which is the honest choice when the gaps
        will be handled downstream.

    Returns
    -------
    Series
        On a uniform one-business-day grid, with a provenance chained to the
        input's.

    Warnings
    --------
    **There is no holiday calendar here.** Every exchange holiday is a weekday
    with no observation, so ``"ffill"`` inserts a repeated price and therefore an
    *exact zero* log return. Roughly nine such days a year on a US equity
    calendar is ~3.5% of the sample; those zeros deflate the small-scale
    fluctuation function, biasing :math:`\\hat h(q)` upward and narrowing
    :math:`f(\\alpha)` — the same mechanism as interpolation, at a different
    scale. Prefer ``method="nan"`` plus an estimator that tolerates gaps, or
    supply a calendar-aware grid yourself. This function exists to make an
    irregular grid *uniform*, not to make missing data *appear*.
    """
    if series.t is None:
        raise ValueError(
            f"series {series.name!r} carries no time index; resampling needs one. "
            "Load with time_column=... or pass t= to load_array."
        )
    requested: str = method
    if requested not in ("ffill", "nan"):
        raise ValueError(f"unknown method {method!r}; use 'ffill' or 'nan'")

    obs_day = np.floor(series.t).astype(np.int64).astype("datetime64[D]")
    grid = np.arange(obs_day[0], obs_day[-1] + np.timedelta64(1, "D"), dtype="datetime64[D]")
    grid = grid[np.is_busday(grid)]
    if grid.size == 0:
        raise DataQualityError(
            f"series {series.name!r} spans no business day between {obs_day[0]} and {obs_day[-1]}"
        )

    pos = np.searchsorted(grid, obs_day)
    on_grid = (pos < grid.size) & (grid[np.clip(pos, 0, grid.size - 1)] == obs_day)
    n_off_grid = int((~on_grid).sum())
    # Intraday observations sharing a calendar day collapse to the last one. That is a
    # deletion of data, so it is counted: the caller must be able to see from the
    # manifest alone that n_in - n_out is not entirely explained by the grid.
    n_collapsed = int(on_grid.sum()) - int(np.unique(pos[on_grid]).size)

    out = np.full(grid.size, np.nan, dtype=np.float64)
    out[pos[on_grid]] = series.values[on_grid]
    present = np.zeros(grid.size, dtype=bool)
    present[pos[on_grid]] = True
    n_gaps = int((~present).sum())

    if requested == "ffill":
        # Index of the most recent observed grid point, vectorised: the running maximum
        # of (index where observed, else 0) is exactly the last-observed index. Grid
        # points preceding the first observation map to index 0, which is itself NaN
        # unless it was observed, so no value is ever carried *backward*.
        last = np.maximum.accumulate(np.where(present, np.arange(grid.size), 0))
        out = out[last]

    t_days = np.asarray(grid.astype("datetime64[ns]").astype(np.int64) / _DAY_NS, dtype=np.float64)
    prov = series.provenance.derive(
        "io.resample_business_days",
        estimator_version=_VERSION,
        params={"method": requested, "n_out": int(grid.size)},
        extra_inputs=(InputRef.from_array("values", out),),
    ).with_diagnostics(
        n_in=series.n,
        n_out=int(grid.size),
        n_gaps_filled=n_gaps if requested == "ffill" else 0,
        n_gaps_left_missing=0 if requested == "ffill" else n_gaps,
        n_observations_off_grid=n_off_grid,
        n_observations_collapsed=n_collapsed,
    )
    return Series(
        values=out,
        t=t_days,
        name=series.name,
        source=f"{series.source} [business-day {requested}]",
        provenance=prov,
    )


def align_series(*series: Series) -> tuple[Series, ...]:
    """Restrict several series to the timestamps they all share.

    Parameters
    ----------
    *series
        Two or more series, each with a time index on the **same convention**
        (float days since the epoch, as produced by every loader here).

    Returns
    -------
    tuple of Series
        In the input order, each on the common index, each with a provenance
        chained to its own parent and carrying the roots of the others as inputs
        — so a joint result cannot be confused with one computed on a different
        alignment.

    Notes
    -----
    Matching is by **exact float equality** of timestamps. That is the right
    default (a silent tolerance would align an 09:30 print with a 16:00 close)
    and it is why the loaders keep integral day counts exactly representable.
    If two sources disagree on the grid, put them on a common one first with
    :func:`resample_business_days`; do not add a tolerance here.

    No forward-filling and no reindexing: the intersection can be much smaller
    than either input, and that is information, not a nuisance — the number kept
    is recorded in the diagnostics of every returned series.
    """
    if len(series) < 2:
        raise ValueError(f"align_series needs at least two series, got {len(series)}")
    for s in series:
        if s.t is None:
            raise ValueError(f"series {s.name!r} carries no time index; cannot align")

    grids = [cast(FloatArray, s.t) for s in series]
    common: FloatArray = grids[0]
    for g in grids[1:]:
        common = np.asarray(np.intersect1d(common, g, assume_unique=True), dtype=np.float64)
    if common.size == 0:
        spans = [(float(g[0]), float(g[-1])) for g in grids]
        raise DataQualityError(
            "the series share no timestamp; check that they use the same time convention "
            f"(spans, in days since the epoch: {spans})"
        )

    out: list[Series] = []
    for s, t_s in zip(series, grids, strict=True):
        idx = np.searchsorted(t_s, common)
        v = np.ascontiguousarray(s.values[idx])
        prov = s.provenance.derive(
            "io.align_series",
            estimator_version=_VERSION,
            params={"n_common": int(common.size), "names": [o.name for o in series]},
            extra_inputs=tuple(
                InputRef.from_provenance(f"aligned_with:{o.name}", o.provenance)
                for o in series
                if o is not s
            ),
        ).with_diagnostics(n_in=s.n, n_out=int(common.size))
        out.append(
            Series(
                values=v,
                t=np.ascontiguousarray(common),
                name=s.name,
                source=f"{s.source} [aligned]",
                provenance=prov,
            )
        )
    return tuple(out)
