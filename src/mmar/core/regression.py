r"""Log--log regression and automatic scaling-range selection.

A multifractal exponent is a slope. Everything downstream --- :math:`\tau(q)`,
:math:`f(\alpha)`, the Hurst exponent --- inherits the quality of these fits, so
this module is deliberately fussy about three things the literature routinely
gets wrong:

1. **Standard errors in log space.** The leverage term must be computed from
   :math:`\log\delta`, not :math:`\delta`. Using raw scales inflates or deflates
   the confidence interval by orders of magnitude.
2. **Weighting.** Large scales carry far fewer independent windows than small
   ones. With ``FitMethod.WLS`` the fit is weighted by :math:`\sqrt{n_j}`, which
   stops a handful of noisy coarse scales from dominating the slope.
3. **The scaling range is an estimate, not a given.** A power law holds over a
   window; :func:`select_scaling_range` finds the widest window consistent with
   linearity instead of quietly fitting through a crossover.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mmar.core.types import FitMethod, FloatArray, LogLogFit, ScalingRange

__all__ = [
    "SelectionResult",
    "loglog_fit",
    "select_scaling_range",
    "weighted_by_counts",
]


def weighted_by_counts(counts: NDArray[np.int64] | Sequence[int]) -> FloatArray:
    r"""Regression weights :math:`w_j = n_j` for ``n_j`` independent windows.

    The variance of a log-moment estimated from :math:`n_j` windows scales like
    :math:`1/n_j`, so inverse-variance weighting is weighting by the count.
    """
    n = np.asarray(counts, dtype=np.float64)
    if np.any(n <= 0):
        raise ValueError("window counts must be strictly positive")
    return n


def _huber_irls(
    x: FloatArray, y: FloatArray, w: FloatArray, *, delta: float = 1.345, iters: int = 30
) -> tuple[float, float]:
    """Huber M-estimate of (slope, intercept) by iteratively reweighted least squares."""
    slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(w))
    for _ in range(iters):
        r = y - (slope * x + intercept)
        scale = 1.4826 * np.median(np.abs(r - np.median(r)))
        if scale <= 0 or not np.isfinite(scale):
            break
        u = np.abs(r) / scale
        hw = np.where(u <= delta, 1.0, delta / np.maximum(u, 1e-12))
        new_slope, new_intercept = np.polyfit(x, y, 1, w=np.sqrt(w * hw))
        if abs(new_slope - slope) < 1e-12 and abs(new_intercept - intercept) < 1e-12:
            slope, intercept = new_slope, new_intercept
            break
        slope, intercept = new_slope, new_intercept
    return float(slope), float(intercept)


def loglog_fit(
    x: FloatArray | Sequence[float],
    y: FloatArray | Sequence[float],
    *,
    method: FitMethod = FitMethod.OLS,
    weights: FloatArray | Sequence[float] | None = None,
    confidence: float = 0.95,
    log_y: bool = True,
    scaling_range: ScalingRange | None = None,
) -> LogLogFit:
    r"""Fit :math:`\log y = \beta \log x + c` and report honest uncertainty.

    Parameters
    ----------
    x, y
        Scales and the corresponding statistic. ``y`` must be strictly positive
        unless ``log_y=False``, in which case ``y`` is taken to be already
        logged (the usual case for wavelet-leader log-moments, which may be
        negative).
    method
        ``OLS``, ``WLS`` (requires ``weights``), ``THEILSEN`` (breakdown 29%, for
        scaling ranges contaminated by a crossover) or ``HUBER``.
    confidence
        Two-sided level for the slope interval, from a
        :math:`t_{n-2}` distribution.

    Raises
    ------
    ValueError
        If fewer than three usable scales survive, or if ``y`` contains
        non-positive values while ``log_y`` is true. Silently dropping those
        points is how spurious power laws get published.
    """
    if not 0.0 < confidence < 1.0:
        # t.ppf(0.5 + confidence/2) is silently NaN for confidence > 1 (the
        # percent-for-fraction slip), infinite at 1 and negative below 0, which
        # returns an inverted interval rather than raising.
        raise ValueError(
            f"confidence must be a probability in (0, 1), got {confidence!r}; "
            "the 95% level is 0.95, not 95"
        )
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.shape != ya.shape:
        raise ValueError(f"x{xa.shape} and y{ya.shape} must have the same shape")
    if np.any(xa <= 0):
        raise ValueError("scales must be strictly positive to take logarithms")

    lx = np.log(xa)
    if log_y:
        if np.any(ya <= 0) or not np.all(np.isfinite(ya)):
            bad = int(np.sum((ya <= 0) | ~np.isfinite(ya)))
            raise ValueError(
                f"{bad}/{ya.size} non-positive or non-finite moments; a power-law fit "
                "is undefined there. Restrict q, widen the scale range, or pass log_y=False."
            )
        ly = np.log(ya)
    else:
        if not np.all(np.isfinite(ya)):
            raise ValueError("non-finite values in the (already logged) response")
        ly = ya

    n = lx.size
    if n < 3:
        raise ValueError(f"need at least 3 scales for a fit with error bars, got {n}")

    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != lx.shape:
            raise ValueError(f"weights{w.shape} must match x{lx.shape}")
        if np.any(w <= 0):
            raise ValueError("weights must be strictly positive")
    if method is FitMethod.OLS:
        w = np.ones(n, dtype=np.float64)

    if method is FitMethod.THEILSEN:
        slope, intercept, lo_ts, hi_ts = stats.theilslopes(ly, lx, alpha=confidence)
        slope, intercept = float(slope), float(intercept)
    elif method is FitMethod.HUBER:
        slope, intercept = _huber_irls(lx, ly, w)
        lo_ts = hi_ts = float("nan")
    else:
        sw = w / w.sum()
        xbar = float(np.sum(sw * lx))
        ybar = float(np.sum(sw * ly))
        sxx = float(np.sum(w * (lx - xbar) ** 2))
        if sxx <= 0:
            raise ValueError("all scales identical after log transform; cannot fit a slope")
        slope = float(np.sum(w * (lx - xbar) * (ly - ybar)) / sxx)
        intercept = float(ybar - slope * xbar)
        lo_ts = hi_ts = float("nan")

    resid = ly - (slope * lx + intercept)
    dof = n - 2
    sw = w / w.sum()
    xbar_w = float(np.sum(sw * lx))
    sxx_w = float(np.sum(w * (lx - xbar_w) ** 2))
    s2 = float(np.sum(w * resid**2) / dof) if dof > 0 else float("nan")
    slope_se = math.sqrt(s2 / sxx_w) if dof > 0 and sxx_w > 0 else float("nan")
    has_spread = dof > 0 and sxx_w > 0
    leverage = 1.0 / w.sum() + xbar_w**2 / sxx_w if has_spread else float("nan")
    intercept_se = math.sqrt(s2 * leverage) if has_spread else float("nan")

    if method is FitMethod.THEILSEN:
        ci = (float(lo_ts), float(hi_ts))
    elif math.isfinite(slope_se):
        t_crit = float(stats.t.ppf(0.5 + confidence / 2.0, dof))
        ci = (slope - t_crit * slope_se, slope + t_crit * slope_se)
    else:  # pragma: no cover - dof <= 0 is rejected above
        ci = (float("nan"), float("nan"))

    ybar_w = float(np.sum(sw * ly))
    ss_tot = float(np.sum(w * (ly - ybar_w) ** 2))
    ss_res = float(np.sum(w * resid**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return LogLogFit(
        slope=slope,
        intercept=intercept,
        slope_stderr=slope_se,
        intercept_stderr=intercept_se,
        slope_ci=ci,
        r_squared=r2,
        n_points=n,
        method=method,
        residuals=resid,
        log_x=lx,
        log_y=ly,
        weights=w if method is FitMethod.WLS else None,
        scaling_range=scaling_range
        or ScalingRange(
            s_min=float(xa.min()),
            s_max=float(xa.max()),
            n_scales=n,
            selection="manual",
            r_squared=r2,
        ),
    )


class SelectionResult:
    """The chosen scale window plus the fit computed on it."""

    __slots__ = ("fit", "mask", "scaling_range")

    def __init__(
        self, mask: NDArray[np.bool_], fit: LogLogFit, scaling_range: ScalingRange
    ) -> None:
        self.mask = mask
        self.fit = fit
        self.scaling_range = scaling_range


def select_scaling_range(
    x: FloatArray | Sequence[float],
    y: FloatArray | Sequence[float],
    *,
    min_points: int = 5,
    criterion: str = "r2",
    r2_threshold: float = 0.995,
    weights: FloatArray | Sequence[float] | None = None,
    method: FitMethod = FitMethod.OLS,
    log_y: bool = True,
) -> SelectionResult:
    r"""Find the widest contiguous scale window over which the power law holds.

    Two criteria:

    ``"r2"``
        Longest contiguous window whose fit reaches ``r2_threshold``. Falls back
        to the best-:math:`R^2` window of length ``min_points`` if none does.
    ``"bic"``
        Minimises :math:`n\log(\mathrm{RSS}/n) + \log(n)\cdot k` with a penalty
        that rewards using more scales. Prefer this when the crossover is soft.

    Reporting the selected window is mandatory: a slope quoted without the
    decades it was fitted over is not a scaling exponent, it is a number.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    n = xa.size
    if n < min_points:
        raise ValueError(f"need at least min_points={min_points} scales, got {n}")
    wa = None if weights is None else np.asarray(weights, dtype=np.float64)

    best: tuple[float, int, int, LogLogFit] | None = None
    for lo in range(0, n - min_points + 1):
        for hi in range(lo + min_points, n + 1):
            sl = slice(lo, hi)
            try:
                fit = loglog_fit(
                    xa[sl],
                    ya[sl],
                    method=method,
                    weights=None if wa is None else wa[sl],
                    log_y=log_y,
                )
            except ValueError:
                continue
            width = hi - lo
            if criterion == "r2":
                reach = float(width) if fit.r_squared >= r2_threshold else 0.0
                key = reach * 1e6 + fit.r_squared
            elif criterion == "bic":
                rss = float(np.sum(fit.residuals**2))
                bic = width * math.log(max(rss, 1e-300) / width) + math.log(width) * 2
                key = -bic
            else:
                raise ValueError(f"unknown criterion {criterion!r}; use 'r2' or 'bic'")
            if best is None or key > best[0]:
                best = (key, lo, hi, fit)

    if best is None:  # pragma: no cover - only if every sub-window fails to fit
        raise ValueError("no admissible scaling range found")

    _, lo, hi, fit = best
    mask = np.zeros(n, dtype=bool)
    mask[lo:hi] = True
    sr = ScalingRange(
        s_min=float(xa[lo]),
        s_max=float(xa[hi - 1]),
        n_scales=hi - lo,
        selection=f"auto-{criterion}",
        r_squared=fit.r_squared,
        mask=tuple(bool(m) for m in mask),
    )
    return SelectionResult(mask, fit, sr)
