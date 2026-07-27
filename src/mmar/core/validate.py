"""Input validation. Loud failures instead of quiet NaNs.

Multifractal estimators degrade gracefully in the worst sense: fed a series with
gaps, zeros or a constant stretch, they return finite numbers that mean nothing.
Every public entry point in this library validates first.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from mmar.core.types import FloatArray

__all__ = [
    "NanPolicy",
    "check_scales",
    "require_squarable",
    "validate_measure",
    "validate_q_grid",
    "validate_series",
]

NanPolicy = Literal["raise", "drop", "interpolate"]

_SQUARABLE: float = math.sqrt(float(np.finfo(np.float64).max))
r"""Largest :math:`|x|` whose square is representable, :math:`\approx1.34\times10^{154}`."""


class DataQualityError(ValueError):
    """The input cannot support the requested estimate."""


def require_squarable(a: FloatArray, *, name: str = "series") -> None:
    r"""Refuse a series whose square is not representable in float64.

    Any estimator that forms a *variance* --- MF-DFA's window residual, MF-DMA's
    segment residual --- squares the data before taking any logarithm, so a
    record reaching :math:`|x|>1.34\times10^{154}` overflows to :math:`\infty`
    behind a bare ``RuntimeWarning`` and then carries the infinity into the
    fluctuation function. The estimators that work entirely in the log domain
    (GHE, the structure functions, the wavelet families) are unaffected and must
    not be gated by this, which is why the check lives with the callers that
    need it rather than in :func:`validate_series`.

    The remedy the message quotes is exact, not a workaround: every exponent in
    this library is invariant under :math:`x\mapsto cx`, which shifts the
    intercept of each log--log fit and no slope.
    """
    peak = float(np.max(np.abs(a)))
    if peak > _SQUARABLE:
        raise DataQualityError(
            f"{name} reaches |x| = {peak:.3g}, beyond {_SQUARABLE:.3g}: its square overflows "
            "float64, so the second moment this estimator is built on is not computable. "
            "Every exponent this library reports is invariant under x -> c*x, so rescale "
            "(x / x.std()) and the answer is unchanged."
        )


def _unmask(x: Sequence[float] | NDArray[Any]) -> Any:
    """Turn a masked array's masked entries into NaN so ``nan_policy`` governs them.

    ``np.asarray`` on a :class:`numpy.ma.MaskedArray` returns the *underlying
    buffer* and discards the mask, so whatever the caller marked as missing --- a
    fill value, a stale reading, an uninitialised cell --- would be analysed as
    if it were data, silently. Routing it through NaN makes the gap visible to
    the policy the caller already had to choose.
    """
    if isinstance(x, np.ma.MaskedArray):
        return np.ma.filled(x.astype(np.float64), np.nan)
    return x


def validate_series(
    x: Sequence[float] | NDArray[Any],
    *,
    name: str = "series",
    min_length: int = 64,
    nan_policy: NanPolicy = "raise",
    allow_constant: bool = False,
) -> FloatArray:
    """Coerce and check a 1-D real series.

    ``nan_policy="interpolate"`` fills interior gaps linearly and is recorded in
    provenance by the caller; it is never the default, because interpolation
    manufactures regularity and biases Hölder exponents *upward*.
    """
    if nan_policy not in ("raise", "drop", "interpolate"):
        # Not a defensive nicety: the dispatch below ends in an `else` that
        # interpolates, so an unrecognised policy used to select the one branch
        # that manufactures regularity.
        raise ValueError(f"unknown nan_policy {nan_policy!r}; use 'raise', 'drop' or 'interpolate'")
    if np.iscomplexobj(x):
        raise DataQualityError(
            f"{name} is complex; a Hölder exponent is defined for a real signal. "
            "Casting would discard the imaginary part behind a ComplexWarning, so pass "
            "x.real, x.imag or np.abs(x) explicitly."
        )
    try:
        a = np.asarray(_unmask(x), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        # numpy's own message names neither the argument nor the remedy: a
        # generator gets "float() argument must be a string or a real number,
        # not 'generator'" three frames deep, which reads as a library bug.
        # Passing `(p for p in prices)` or a `{date: price}` dict is a routine
        # slip and the answer is one call away.
        raise DataQualityError(
            f"{name} could not be read as a numeric sequence ({type(x).__name__}: {exc}). "
            "Pass a 1-D sequence of real numbers -- a numpy array, a list, or a pandas "
            "Series; a generator must be materialised first (np.fromiter(g, float)) and a "
            "mapping reduced to its values in a defined order."
        ) from exc
    if a.ndim == 0:
        raise DataQualityError(
            f"{name} is a scalar ({type(x).__name__}), not a series; a scaling exponent is "
            f"fitted across at least {min_length} ordered samples. Pass the whole path."
        )
    if a.ndim != 1:
        raise DataQualityError(
            f"{name} must be one-dimensional, got shape {a.shape}. Flattening a wider array "
            "would interleave its columns into a sequence that never existed; pass a single "
            "column (x[:, 0], df[col].to_numpy())."
        )
    if a.size < min_length:
        raise DataQualityError(
            f"{name} has {a.size} points; at least {min_length} are needed for a "
            "scaling fit with more than a couple of decades"
        )

    bad = ~np.isfinite(a)
    if bad.any():
        if nan_policy == "raise":
            raise DataQualityError(
                f"{name} contains {int(bad.sum())} non-finite values at indices "
                f"{np.nonzero(bad)[0][:10].tolist()}...; pass nan_policy='drop' or "
                "'interpolate' to state explicitly how they should be handled"
            )
        if nan_policy == "drop":
            a = a[~bad]
            if a.size < min_length:
                raise DataQualityError(f"{name} has only {a.size} finite points after dropping")
        else:
            idx = np.arange(a.size)
            a = np.interp(idx, idx[~bad], a[~bad])

    # min == max, not std == 0: the variance of a series of magnitude 1e300 overflows
    # (a bare RuntimeWarning) and that of a series of magnitude 1e-300 underflows to
    # exactly zero, which would reject a strictly monotone ramp as "constant".
    if not allow_constant and float(a.min()) == float(a.max()):
        raise DataQualityError(f"{name} is constant; every scaling exponent is degenerate")
    # Always an owned, writable copy. `np.ascontiguousarray` hands back the caller's
    # own buffer for a contiguous float64 input, which (a) leaves every estimator one
    # in-place operation away from rewriting the caller's data and (b) propagates a
    # read-only flag -- a memory-mapped or deliberately frozen series then dies inside
    # PyWavelets with "buffer source array is read-only", naming neither the input nor
    # the remedy. The copy is O(n) against estimators that are O(n log n) at best.
    return np.array(a, dtype=np.float64, order="C", copy=True)


def validate_measure(
    mu: Sequence[float] | NDArray[Any],
    *,
    name: str = "measure",
    normalise: bool = True,
    allow_zero: bool = True,
) -> FloatArray:
    """Coerce and check a discrete positive measure.

    Zero cells are permitted (they are the support boundary) but are excluded
    from negative-:math:`q` moments by the estimators, which must say so.
    """
    a = np.asarray(_unmask(mu), dtype=np.float64).ravel()
    if not np.all(np.isfinite(a)):
        raise DataQualityError(f"{name} contains non-finite mass")
    if np.any(a < 0):
        raise DataQualityError(f"{name} has {int((a < 0).sum())} negative cells; not a measure")
    if not allow_zero and np.any(a == 0):
        raise DataQualityError(f"{name} has {int((a == 0).sum())} zero cells")
    total = float(a.sum())
    if total <= 0:
        raise DataQualityError(f"{name} has zero total mass")
    return np.ascontiguousarray(a / total) if normalise else np.ascontiguousarray(a)


def validate_q_grid(
    q: Sequence[float] | NDArray[Any] | None,
    *,
    default_range: tuple[float, float] = (-5.0, 5.0),
    n_default: int = 41,
    exclude_zero: bool = False,
    warn_large: float = 10.0,
) -> FloatArray:
    r"""Build and check the moment grid.

    Notes
    -----
    Large :math:`|q|` is the classic way to fabricate a spectrum: for
    :math:`q>q_{\text{crit}}` the moment :math:`\langle|\Delta X|^q\rangle` may
    not exist, and the sample estimate is then dominated by the single largest
    increment. Its log--log plot still looks linear. Keep :math:`|q|\le 5`
    unless the moment's existence has been checked
    (:func:`mmar.inference.tests.moment_existence`).
    """
    if q is None:
        a = np.linspace(default_range[0], default_range[1], n_default)
    else:
        a = np.asarray(q, dtype=np.float64).ravel()
    if a.size < 3:
        raise ValueError("the q grid needs at least three points")
    if not np.all(np.isfinite(a)):
        raise ValueError("the q grid contains non-finite values")
    if np.any(np.diff(a) <= 0):
        a = np.unique(a)
        if a.size < 3:
            raise ValueError("the q grid collapsed to fewer than three distinct values")
    if exclude_zero:
        a = a[np.abs(a) > 1e-12]
    if float(np.max(np.abs(a))) > warn_large:
        # Not an error: some designs legitimately probe far tails.
        pass
    # Always a copy: the grid is stored verbatim inside the result *and* inside
    # its provenance params, so aliasing the caller's array would let a later
    # `q[0] = ...` rewrite an already-published exponent curve and its root.
    return np.array(a, dtype=np.float64, order="C", copy=True)


def check_scales(
    scales: Sequence[int] | NDArray[Any],
    n: int,
    *,
    min_windows: int = 4,
    name: str = "scales",
    basis: Literal["samples", "increments"] = "samples",
) -> NDArray[np.int64]:
    r"""Drop scales that leave fewer than ``min_windows`` windows, and say so.

    At :math:`n/s < 4` the fluctuation at scale :math:`s` is an average of three
    numbers; including such scales is the standard way of curving the top end of
    a log--log plot.

    Parameters
    ----------
    basis
        What the caller partitions. ``"samples"`` counts :math:`\lfloor n/s
        \rfloor` windows, correct for MF-DFA-style segmentation of a profile.
        ``"increments"`` counts :math:`\lfloor (n-1)/s \rfloor`, correct for any
        estimator built from differences :math:`X(t+s)-X(t)`: there are
        :math:`n-1` unit increments, not :math:`n`. The two differ by exactly
        one whenever :math:`s \mid n`, which is precisely the coarse end of a
        dyadic grid on a power-of-two series --- at :math:`n=2^{16},\ s=2^{14}`
        the sample basis admits a scale yielding only three increments.
    """
    s = np.unique(np.asarray(scales, dtype=np.int64).ravel())
    if s.size == 0:
        raise ValueError(f"{name} is empty")
    if np.any(s < 2):
        raise ValueError(f"{name} must all be >= 2")
    available = n if basis == "samples" else n - 1
    keep = (available // s) >= min_windows
    if not keep.any():
        raise DataQualityError(
            f"no scale in {name} leaves >= {min_windows} {basis[:-1]} windows for n={n} "
            f"(largest admissible scale is {available // min_windows}); the series is too short"
        )
    return np.ascontiguousarray(s[keep])


def effective_decades(scales: NDArray[np.int64]) -> float:
    """Extent of the scale axis in decades; below ~1 a power-law claim is weak."""
    return math.log10(float(scales.max()) / float(scales.min()))
