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
    "validate_measure",
    "validate_q_grid",
    "validate_series",
]

NanPolicy = Literal["raise", "drop", "interpolate"]


class DataQualityError(ValueError):
    """The input cannot support the requested estimate."""


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
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.ndim != 1:  # pragma: no cover - ravel guarantees 1-D
        raise DataQualityError(f"{name} must be one-dimensional")
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

    if not allow_constant and float(np.std(a)) == 0.0:
        raise DataQualityError(f"{name} is constant; every scaling exponent is degenerate")
    return np.ascontiguousarray(a)


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
    a = np.asarray(mu, dtype=np.float64).ravel()
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
    return np.ascontiguousarray(a)


def check_scales(
    scales: Sequence[int] | NDArray[Any],
    n: int,
    *,
    min_windows: int = 4,
    name: str = "scales",
) -> NDArray[np.int64]:
    """Drop scales that leave fewer than ``min_windows`` windows, and say so.

    At :math:`n/s < 4` the fluctuation at scale :math:`s` is an average of three
    numbers; including such scales is the standard way of curving the top end of
    a log--log plot.
    """
    s = np.unique(np.asarray(scales, dtype=np.int64).ravel())
    if s.size == 0:
        raise ValueError(f"{name} is empty")
    if np.any(s < 2):
        raise ValueError(f"{name} must all be >= 2")
    keep = (n // s) >= min_windows
    if not keep.any():
        raise DataQualityError(
            f"no scale in {name} leaves >= {min_windows} windows for n={n}; the series is too short"
        )
    return np.ascontiguousarray(s[keep])


def effective_decades(scales: NDArray[np.int64]) -> float:
    """Extent of the scale axis in decades; below ~1 a power-law claim is weak."""
    return math.log10(float(scales.max()) / float(scales.min()))
