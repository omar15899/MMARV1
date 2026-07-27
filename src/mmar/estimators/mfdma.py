r"""Multifractal detrending moving average (MFDMA).

MFDMA (Gu & Zhou 2010) is MFDFA's sibling: the local trend is removed with a
*moving average* filter instead of a per-window polynomial fit. Both are
:math:`O(N)` per scale, but MFDMA's constant is a single running sum against
MFDFA's per-window normal equations; it has no polynomial order to choose; and
its finite-size behaviour is genuinely different, so running both on the same
data is a cross-check rather than a repetition. What it buys in simplicity it
pays for in trend handling — see bias 1 below, which is not a caveat but the
defining property of the filter.

The algorithm
-------------
Let :math:`Y(1..N)` be the *profile* (see ``integrate`` below). For a window
length :math:`n` and a **position parameter** :math:`\theta\in[0,1]` define the
moving average

.. math::
    \tilde Y(i) \;=\; \frac1n \sum_{k=-\lfloor (n-1)\theta\rfloor}
                              ^{\lceil (n-1)(1-\theta)\rceil} Y(i-k) ,

so with :math:`L=\lfloor (n-1)\theta\rfloor` the window covers
:math:`[\,i-(n-1-L),\; i+L\,]`: :math:`L` points ahead of :math:`i` and
:math:`n-1-L` behind it (the two limits always sum to :math:`n-1`, since
:math:`\lfloor a\rfloor+\lceil (n-1)-a\rceil = n-1` for every real :math:`a`).
:math:`\theta=0` is the *backward* (causal) filter, :math:`\theta=1` the
*forward* one, :math:`\theta=1/2` the *centred* one.

The residual is defined only where the whole window exists,

.. math::
    \varepsilon(i) = Y(i)-\tilde Y(i), \qquad
    n-L \;\le\; i \;\le\; N-L \quad (\text{1-indexed}),

i.e. exactly :math:`N-n+1` points. This index range is where MFDMA
implementations go wrong; it is derived line by line in
:func:`mfdma_residual` and pinned by a test.

The residual is cut into :math:`N_n=\lfloor (N-n+1)/n\rfloor` disjoint segments
of length :math:`n`, each contributing

.. math::
    F_v^2(n) \;=\; \frac1n\sum_{i\in v}\varepsilon(i)^2 ,

and the :math:`q`-th order fluctuation is the power mean

.. math::
    F_q(n) = \Big[\frac1{N_n}\sum_{v} F_v(n)^{q}\Big]^{1/q}\ (q\neq0),
    \qquad
    F_0(n) = \exp\Big[\frac1{N_n}\sum_v \ln F_v(n)\Big],

the :math:`q=0` case being the continuous limit :math:`q\to0` (the logarithmic
branch), not a special convention.

Convention (this library's law, see :mod:`mmar.core.types`)
-----------------------------------------------------------
MFDMA natively produces a **fluctuation exponent** :math:`h(q)`, defined by
:math:`F_q(n)\sim n^{h(q)}`. The conversion to the carrier convention is

.. math::
    \zeta(q) = q\,h(q), \qquad \tau(q) = \zeta(q)-d = q\,h(q)-d ,

because :math:`F_q(n)^q = \frac1{N_n}\sum_v F_v(n)^q` is a :math:`q`-th moment
of a fluctuation over a window of size :math:`n`, which is what
:math:`S(q,\delta)=\langle|\Delta_\delta X|^q\rangle\sim\delta^{\zeta(q)}`
measures. Two immediate consequences, both tested:

* :math:`\zeta(0)=0` and hence :math:`\tau(0)=-d` **identically**. For a
  fluctuation estimator this anchor is an algebraic identity, not a
  measurement; the measured content at :math:`q=0` is :math:`h(0)`, which must
  agree with :math:`\lim_{q\to0}h(q)`.
* fBm with Hurst :math:`H` gives :math:`h(q)\equiv H` and :math:`\zeta(q)=qH`;
  the cumulative binomial cascade (trading time) gives
  :math:`\zeta(q)=1-\log_2(m_0^q+m_1^q)`.

Known biases, and their causes
------------------------------
1. **DMA does not remove a linear trend unless it is centred.** For
   :math:`Y(i)=a+bi` the window mean is :math:`a+b(i-(n-1)/2+L)`, so

   .. math::
       \varepsilon(i) \;=\; b\Big[\tfrac{n-1}{2}-\big\lfloor (n-1)\theta
       \big\rfloor\Big],

   a *constant* that vanishes only when :math:`L=(n-1)/2`, i.e. for
   :math:`\theta=1/2` with odd :math:`n`. For any other :math:`\theta` a drift
   :math:`b` injects a deterministic term growing like :math:`n` into
   :math:`F_q(n)`, which pulls :math:`h(q)` towards 1. Unlike DFA, DMA is
   *not* a detrending method in the polynomial sense — it is a high-pass
   filter whose phase is set by :math:`\theta`. De-trend the profile first if
   the series carries a deterministic drift.
2. **Small scales.** For :math:`n\lesssim8` the filter has too few taps and
   :math:`F_q(n)` departs from the power law; the log--log plot curves upward.
   The default scale grid starts at :math:`n=8`.
3. **Large scales.** :math:`N_n\approx N/n-1` falls with :math:`n`, so the
   coarse moments are averages of a handful of numbers. This is exactly why
   the fit is weighted by :math:`N_n` (``FitMethod.WLS``) and why scales
   leaving fewer than ``min_segments`` segments are refused.
4. **Negative :math:`q`.** :math:`F_q` with :math:`q<0` is dominated by the
   segments where the residual nearly vanishes, so :math:`h(q)` for
   :math:`q\ll0` has much larger variance than for :math:`q>0` and is biased
   upward on short samples; the right branch of :math:`f(\alpha)` inherits
   that error.

   A *degenerate* segment — one whose residual is zero, e.g. a constant
   stretch of the profile, or a perfectly linear one under the centred filter
   — would contribute :math:`F_v^{q}=\infty`. Testing for exact zero is not
   enough: the running sum leaves round-off of relative size
   :math:`\sim N\epsilon`, so a genuinely flat stretch produces
   :math:`F_v\sim10^{-16}F_{\max}` rather than 0, and at :math:`q=-5` that
   single segment carries a weight of :math:`10^{80}`. Segments with
   :math:`F_v \le \texttt{zero\_rtol}\cdot\max_v F_v` at the same scale are
   therefore **dropped**, with ``zero_rtol`` defaulting to :math:`10^{-12}` —
   twelve orders of magnitude below the loudest segment is far below any
   dynamic range a measured fluctuation can carry, and four above the
   round-off floor. The number dropped per scale is recorded in
   ``provenance.diagnostics['n_degenerate_segments_by_scale']``. Nothing is
   dropped silently, and ``zero_rtol`` is part of the provenance root.
5. **The reported intervals are too narrow.** ``zeta_stderr`` and ``zeta_ci``
   are log--log regression scatter and assume independent scales, which nested
   window statistics are not. Measured coverage of the nominal 95% interval:
   14/40 on exact fBm paths, 1/41 on the deterministic cascade. See the
   warning on :func:`mfdma_exponents_from_structure` and
   ``test_interval_coverage_is_measured_not_assumed``.
6. **Which :math:`\theta` is least biased is an empirical question.** Xu et
   al. (2005) found the centred DMA best for monofractal fBm; Gu & Zhou (2010)
   report the backward MFDMA (:math:`\theta=0`) performing best on
   multifractal binomial measures. The measurement for this implementation, on
   exact fBm, is reported with numbers in ``tests/unit/test_mfdma.py``.

References
----------
Gu, G.-F. and Zhou, W.-X. (2010). Detrending moving average algorithm for
multifractals. *Physical Review E* **82**, 011136.

Alessio, E., Carbone, A., Castelli, G. and Frappietro, V. (2002). Second-order
moving average and scaling of stochastic time series. *European Physical
Journal B* **27**, 197--200.

Xu, L., Ivanov, P.Ch., Hu, K., Chen, Z., Carbone, A. and Stanley, H.E. (2005).
Quantifying signals with power-law correlations: a comparative study of
detrended fluctuation analysis and detrended moving average techniques.
*Physical Review E* **71**, 051101.

Kantelhardt, J.W., Zschiegner, S.A., Koscielny-Bunde, E., Havlin, S., Bunde, A.
and Stanley, H.E. (2002). Multifractal detrended fluctuation analysis of
nonstationary time series. *Physica A* **316**, 87--114.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from mmar.core.provenance import InputRef, Provenance, new_provenance
from mmar.core.regression import loglog_fit, select_scaling_range, weighted_by_counts
from mmar.core.rng import SeedLike, seed_info
from mmar.core.types import (
    FitMethod,
    FloatArray,
    LogLogFit,
    ScalingExponents,
    ScalingRange,
    StructureFunctions,
    log_spaced_scales,
)
from mmar.core.validate import check_scales, validate_q_grid, validate_series

__all__ = [
    "ESTIMATOR_ID",
    "mfdma",
    "mfdma_exponents_from_structure",
    "mfdma_profile",
    "mfdma_residual",
    "mfdma_structure",
]

ESTIMATOR_ID = "estimators.mfdma"
_VERSION = "1"

_Q_ZERO_TOL = 1e-6
r"""Below this :math:`|q|` the logarithmic branch replaces :math:`(\cdot)^{1/q}`.

The power mean is analytic through :math:`q=0` with
:math:`\ln F_q = \overline{\ln F_v} + \tfrac{q}{2}\operatorname{Var}(\ln F_v)
+ O(q^2)`, so substituting the limit costs :math:`O(10^{-6})` in
:math:`h`, whereas evaluating :math:`(\mathrm{logsumexp}-\ln N_n)/q` at
:math:`q\sim10^{-12}` loses every significant digit to cancellation.
"""


# --------------------------------------------------------------------------- #
# Filter geometry
# --------------------------------------------------------------------------- #


def _check_theta(theta: float) -> float:
    t = float(theta)
    if not math.isfinite(t) or not 0.0 <= t <= 1.0:
        raise ValueError(
            f"the MFDMA position parameter theta must lie in [0, 1]; got {theta!r} "
            "(0 = backward/causal, 0.5 = centred, 1 = forward)"
        )
    return t


def _future_span(n: int, theta: float) -> int:
    r"""Number of points *ahead* of :math:`i` in the window: :math:`\lfloor(n-1)\theta\rfloor`."""
    return math.floor((n - 1) * theta)


def mfdma_profile(
    x: Sequence[float] | NDArray[Any], *, integrate: bool = False, min_length: int = 64
) -> FloatArray:
    r"""The profile :math:`Y` that the moving average is applied to.

    Parameters
    ----------
    integrate
        ``False`` (default): the input **is** the profile. Use this when ``x``
        is the self-affine signal :math:`X` whose increments define
        :math:`\zeta` — an fBm path, a log-price, a cumulative measure. This is
        the setting in which the recovered :math:`h(q)` equals the
        structure-function :math:`\zeta(q)/q` of ``x`` itself, which is the
        convention :mod:`mmar.core.types` fixes.

        ``True``: form the classical MFDFA/MFDMA profile
        :math:`Y(i)=\sum_{k\le i}\big(x_k-\bar x\big)`. Use this when ``x`` is
        a *noise* series (returns, increments); the recovered :math:`h(q)` then
        describes the cumulated process, i.e. :math:`h(2)=H` for fGn input.

    Getting this switch wrong shifts every exponent by exactly one
    (:math:`h\mapsto h+1`), which is the most common way DFA-family code is
    misread.
    """
    a = validate_series(x, name="series", min_length=min_length)
    if not integrate:
        return a
    return np.ascontiguousarray(np.cumsum(a - float(a.mean())), dtype=np.float64)


def mfdma_residual(y: FloatArray, n: int, theta: float) -> FloatArray:
    r"""Detrended residual :math:`\varepsilon = Y-\tilde Y`, on the valid range only.

    Index derivation (the part implementations get wrong)
    -----------------------------------------------------
    With :math:`L=\lfloor(n-1)\theta\rfloor`, the window attached to the
    1-indexed point :math:`i` is :math:`[\,i-(n-1-L),\ i+L\,]`. It lies inside
    :math:`[1,N]` iff

    .. math::
        i-(n-1-L) \ge 1 \iff i \ge n-L, \qquad i+L \le N \iff i \le N-L ,

    which is Gu & Zhou's range :math:`n-L\le i\le N-L`, of length
    :math:`N-n+1`. In 0-indexed form the evaluation points are
    :math:`i_0,\dots,i_0+(N-n)` with :math:`i_0=n-1-L`, and the window attached
    to the :math:`t`-th of them is the *fixed* block
    :math:`[t,\;t+n-1]` — the set of windows does not depend on
    :math:`\theta`; only which sample each window is compared against does.
    That observation is what turns the moving average into a single
    :math:`O(N)` running sum.

    Notes
    -----
    The running sum is formed on the mean-removed profile. Subtracting a
    constant leaves :math:`\varepsilon` algebraically unchanged (it cancels
    between :math:`Y` and :math:`\tilde Y`) but keeps the partial sums small,
    which is what makes the exact-annihilation test hold at
    :math:`10^{-12}` rather than at the round-off of a large accumulator.
    """
    if n < 2:
        raise ValueError(f"the moving-average window must be at least 2 samples; got n={n}")
    size = int(y.size)
    if n > size:
        raise ValueError(f"window n={n} exceeds the series length N={size}")
    t = _check_theta(theta)
    lead = _future_span(n, t)

    centred = y - float(y.mean())
    csum = np.empty(size + 1, dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(centred, out=csum[1:])
    ma = (csum[n:] - csum[:-n]) / float(n)

    i0 = n - 1 - lead
    return np.ascontiguousarray(centred[i0 : i0 + ma.size] - ma, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Fluctuation ladder
# --------------------------------------------------------------------------- #


def _segment_log_fluctuations(eps: FloatArray, n: int, zero_rtol: float) -> tuple[FloatArray, int]:
    r"""Return :math:`(\ln F_v(n))_v` over usable segments, and how many were dropped.

    :math:`F_v(n)=\big[\frac1n\sum_{i\in v}\varepsilon_i^2\big]^{1/2}`. A
    segment with :math:`F_v \le \texttt{zero\_rtol}\cdot\max_v F_v` carries no
    scale information and would dominate every :math:`q<0` moment by many
    orders of magnitude; it is removed here and counted by the caller. The
    comparison is made on :math:`F^2`, hence the squared tolerance.
    """
    n_seg = eps.size // n
    block = eps[: n_seg * n].reshape(n_seg, n)
    f2 = np.asarray(np.mean(block * block, axis=1), dtype=np.float64)
    floor = zero_rtol * zero_rtol * float(f2.max())
    good = f2 > floor
    n_dropped = int(n_seg - int(good.sum()))
    return 0.5 * np.log(f2[good]), n_dropped


def _power_mean_log(log_f: FloatArray, q: FloatArray) -> FloatArray:
    r""":math:`\ln F_q` for every :math:`q`, from the segment log-fluctuations.

    .. math::
        \ln F_q = \frac1q\Big[\ln\!\sum_v e^{q\ln F_v} - \ln N_n\Big],
        \qquad \ln F_0 = \overline{\ln F_v}.

    Evaluated through :func:`scipy.special.logsumexp` so that
    :math:`q=-5` on a segment with :math:`F_v\sim10^{-80}` neither overflows
    nor loses precision.
    """
    n_seg = log_f.size
    lse = np.asarray(logsumexp(q[:, None] * log_f[None, :], axis=1), dtype=np.float64)
    log_mean = lse - math.log(n_seg)
    small = np.abs(q) < _Q_ZERO_TOL
    safe_q = np.where(small, 1.0, q)
    return np.asarray(np.where(small, float(np.mean(log_f)), log_mean / safe_q), dtype=np.float64)


def _resolve_scales(
    n_samples: int,
    scales: Sequence[int] | NDArray[Any] | None,
    min_segments: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Admissible scales and the segment count each of them yields.

    ``check_scales`` bounds :math:`\\lfloor N/n\\rfloor`, but MFDMA spends
    :math:`n-1` points on the filter boundary, so only :math:`N-n+1` residual
    points survive and the realisable segment count is
    :math:`\\lfloor (N-n+1)/n\\rfloor`, one lower whenever :math:`n \\nmid N`.
    Filtering on the realisable count is the honest test.
    """
    if min_segments < 3:
        raise ValueError(
            f"min_segments must be at least 3 for a fit with error bars; got {min_segments}"
        )
    raw = log_spaced_scales(n_samples) if scales is None else np.asarray(scales, dtype=np.int64)
    admissible = check_scales(raw, n_samples, min_windows=min_segments, name="scales")
    counts = (n_samples - admissible + 1) // admissible
    keep = counts >= min_segments
    if int(keep.sum()) < 3:
        raise ValueError(
            f"only {int(keep.sum())} scale(s) leave >= {min_segments} MFDMA segments for "
            f"N={n_samples}; at least 3 are needed for a log-log fit with error bars. "
            "Shorten the scale grid, lower min_segments, or use a longer series."
        )
    return (
        np.ascontiguousarray(admissible[keep], dtype=np.int64),
        np.ascontiguousarray(counts[keep], dtype=np.int64),
    )


# --------------------------------------------------------------------------- #
# Public surface
# --------------------------------------------------------------------------- #


def mfdma_structure(
    x: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    theta: float = 0.0,
    integrate: bool = False,
    min_segments: int = 4,
    zero_rtol: float = 1e-12,
    seed: SeedLike = None,
) -> StructureFunctions:
    r"""The raw MFDMA scaling surface :math:`F_q(n)`, with no exponent fitted.

    Separating this from :func:`mfdma` is what makes a refit over a different
    scaling range free: the :math:`O(N\,|\text{scales}|)` filtering work is done
    here once, and :func:`mfdma_exponents_from_structure` consumes the result.

    Returns
    -------
    StructureFunctions
        ``values[i, j]`` is :math:`F_{q_i}(n_j)`, the **fluctuation function**,
        whose log--log slope is :math:`h(q)`. It is deliberately *not*
        :math:`F_q^q`: the latter is identically 1 at :math:`q=0` and carries no
        information there, whereas :math:`F_q` is finite and informative for
        every :math:`q`. The structure-function moment in the sense of
        :mod:`mmar.core.types` is :math:`S(q,n)=F_q(n)^q`, with slope
        :math:`\zeta(q)=q\,h(q)`; the conversion is applied in
        :func:`mfdma_exponents_from_structure`.
        ``counts[j]`` is the number of segments actually averaged at scale
        :math:`n_j`, after zero-fluctuation segments were dropped.
    """
    y = mfdma_profile(x, integrate=integrate)
    qa = validate_q_grid(q)
    t = _check_theta(theta)
    if not 0.0 <= zero_rtol < 1.0:
        raise ValueError(f"zero_rtol must lie in [0, 1); got {zero_rtol!r}")
    admissible, _ = _resolve_scales(y.size, scales, min_segments)

    values = np.empty((qa.size, admissible.size), dtype=np.float64)
    used = np.empty(admissible.size, dtype=np.int64)
    dropped = np.empty(admissible.size, dtype=np.int64)

    for j, n in enumerate(admissible):
        eps = mfdma_residual(y, int(n), t)
        log_f, n_dropped = _segment_log_fluctuations(eps, int(n), float(zero_rtol))
        # min_segments, not 2: `_resolve_scales` already refuses a scale whose
        # *nominal* segment count is below min_segments, and the survivors after the
        # degenerate-segment drop must clear the same bar. A spike on a flat
        # background leaves exactly two live segments at every scale and used to be
        # fitted through them, returning h(2) = -0.49 with no diagnostic.
        if log_f.size < min_segments:
            raise ValueError(
                f"scale n={int(n)} left only {log_f.size} non-degenerate segment(s) "
                f"({n_dropped} had a vanishing residual), fewer than min_segments="
                f"{min_segments}; the fluctuation at that scale is not estimable. "
                "Drop the scale or check the input for a piecewise-constant stretch."
            )
        values[:, j] = np.exp(_power_mean_log(log_f, qa))
        used[j] = log_f.size
        dropped[j] = n_dropped

    prov = new_provenance(
        f"{ESTIMATOR_ID}.structure",
        estimator_version=_VERSION,
        params={
            "theta": t,
            "integrate": bool(integrate),
            "min_segments": int(min_segments),
            "zero_rtol": float(zero_rtol),
            "n_samples": int(y.size),
            "q_zero_tol": _Q_ZERO_TOL,
        },
        inputs=(
            InputRef.from_array("x", np.asarray(x, dtype=np.float64).ravel()),
            InputRef.from_array("q", qa),
            InputRef.from_array("scales", admissible),
        ),
        seed=seed_info(seed) if seed is not None else None,
        notes=(
            "values[i, j] = F_{q_i}(n_j), the MFDMA fluctuation function; "
            "zeta(q) = q * h(q) with F_q(n) ~ n^{h(q)}",
        ),
    ).with_diagnostics(
        n_degenerate_segments_by_scale=[int(v) for v in dropped],
        n_degenerate_segments_total=int(dropped.sum()),
        segments_by_scale=[int(v) for v in used],
    )

    return StructureFunctions(
        q=qa,
        scales=np.asarray(admissible, dtype=np.float64),
        values=values,
        counts=used,
        provenance=prov,
    )


def _fit_all_q(
    structure: StructureFunctions,
    *,
    fit_method: FitMethod,
    select_range: bool,
) -> tuple[tuple[LogLogFit, ...], ScalingRange]:
    r"""One log--log fit of :math:`\ln F_q` against :math:`\ln n` per :math:`q`.

    When ``select_range`` is on, the window is chosen **once**, on the row
    closest to :math:`q=2`, and imposed on every other :math:`q`. Selecting per
    :math:`q` would make :math:`\hat\zeta(q)` a fit over a :math:`q`-dependent
    range of scales, so its curvature — the entire multifractal signal — would
    partly reflect the selector rather than the data.
    """
    scales = structure.scales
    counts = structure.counts
    weights = weighted_by_counts(counts) if counts is not None else None
    values = structure.values
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        bad = int(np.sum(~np.isfinite(values) | (values <= 0.0)))
        raise ValueError(
            f"{bad}/{values.size} fluctuation values are non-positive or non-finite; "
            "a power-law fit is undefined there. This means whole segments had a "
            "vanishing residual at some scale -- inspect "
            "provenance.diagnostics['n_degenerate_segments_by_scale']."
        )
    log_values = np.log(values)

    mask = np.ones(scales.size, dtype=bool)
    selection = "manual"
    if select_range:
        ref = int(np.argmin(np.abs(structure.q - 2.0)))
        min_points = max(3, min(5, scales.size))
        result = select_scaling_range(
            scales,
            log_values[ref],
            min_points=min_points,
            weights=weights,
            method=fit_method,
            log_y=False,
        )
        mask = np.asarray(result.mask, dtype=bool)
        selection = f"auto-r2@q={structure.q[ref]:g}"

    xs = scales[mask]
    ws = None if weights is None else weights[mask]
    fits = tuple(
        loglog_fit(xs, log_values[i][mask], method=fit_method, weights=ws, log_y=False)
        for i in range(structure.q.size)
    )
    r2 = float(np.median([f.r_squared for f in fits]))
    scaling_range = ScalingRange(
        s_min=float(xs.min()),
        s_max=float(xs.max()),
        n_scales=int(xs.size),
        selection=selection,
        r_squared=r2,
        mask=tuple(bool(m) for m in mask),
    )
    return fits, scaling_range


def mfdma_exponents_from_structure(
    structure: StructureFunctions,
    *,
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
    d_support: float = 1.0,
) -> ScalingExponents:
    r"""Fit :math:`h(q)` on a precomputed surface and convert to :math:`\zeta=q\,h(q)`.

    The uncertainty transforms with the same map: :math:`\operatorname{se}
    (\hat\zeta) = |q|\operatorname{se}(\hat h)`, and the confidence interval
    endpoints **swap** for :math:`q<0` because multiplication by a negative
    number reverses the order. At :math:`q=0` the interval collapses to
    :math:`\{0\}`, correctly: :math:`\zeta(0)=0` is an identity of the
    convention, not an estimate.

    The returned ``fits[i]`` is the native fit, so ``fits[i].slope`` is
    :math:`h(q_i)`, not :math:`\zeta(q_i)`.

    .. warning::
        **The interval measures scatter about the fitted line, not accuracy.**
        ``zeta_stderr`` and ``zeta_ci`` come from the log--log regression, which
        treats the :math:`|\text{scales}|` points as independent. They are not:
        the residual at scale :math:`n` and at scale :math:`2n` are computed
        from the same samples, so the regression sees far less information than
        its degrees-of-freedom count claims, and the interval is too narrow
        *even when the input really is stochastic*. Neither does it know about
        the estimator's finite-scale bias, which dominates on a
        near-deterministic input.

        Measured (:math:`q` grid :math:`[-5,5]`, default scales):

        * exact fBm, :math:`H=0.7`, :math:`N=8192`, :math:`\theta=1/2`, 40
          paths: across-path :math:`\operatorname{sd}(\hat h(2))=0.0132`
          against a median reported :math:`\operatorname{se}=0.0050` — narrow
          by :math:`2.7\times`; the nominal 95% interval covers :math:`2H` in
          **14 of 40** paths.
        * deterministic binomial cascade :math:`(0.6,0.4)`, :math:`k_{\max}=14`:
          median :math:`\operatorname{se}(\hat\zeta)=0.0137` against a median
          :math:`|\hat\zeta-\zeta|=0.0948` from the closed-form oracle, and the
          interval covers the truth at **1 of 41** :math:`q` values.

        Use these intervals to compare fits, never to test a hypothesis about
        an exponent. For that, use an oracle comparison or a surrogate
        ensemble (:mod:`mmar.inference`).
    """
    fits, scaling_range = _fit_all_q(structure, fit_method=fit_method, select_range=select_range)
    qa = structure.q
    h = np.asarray([f.slope for f in fits], dtype=np.float64)
    h_se = np.asarray([f.slope_stderr for f in fits], dtype=np.float64)
    lo = np.asarray([f.slope_ci[0] for f in fits], dtype=np.float64)
    hi = np.asarray([f.slope_ci[1] for f in fits], dtype=np.float64)

    zeta = qa * h
    zeta_se = np.abs(qa) * h_se
    zeta_lo = np.where(qa >= 0.0, qa * lo, qa * hi)
    zeta_hi = np.where(qa >= 0.0, qa * hi, qa * lo)

    prov: Provenance | None = None
    if structure.provenance is not None:
        prov = structure.provenance.derive(
            ESTIMATOR_ID,
            estimator_version=_VERSION,
            params={
                "fit_method": str(fit_method),
                "select_range": bool(select_range),
                "d_support": float(d_support),
            },
            notes=(
                "native exponent is h(q) with F_q(n) ~ n^{h(q)}; stored zeta(q) = q*h(q), "
                "tau(q) = zeta(q) - d",
            ),
        )
        if structure.provenance.diagnostics:
            prov = prov.with_diagnostics(**dict(structure.provenance.diagnostics))

    return ScalingExponents(
        q=qa,
        zeta=np.ascontiguousarray(zeta, dtype=np.float64),
        d_support=float(d_support),
        zeta_stderr=np.ascontiguousarray(zeta_se, dtype=np.float64),
        zeta_ci=(
            np.ascontiguousarray(zeta_lo, dtype=np.float64),
            np.ascontiguousarray(zeta_hi, dtype=np.float64),
        ),
        fits=fits,
        scaling_range=scaling_range,
        structure=structure,
        estimator=ESTIMATOR_ID,
        provenance=prov,
    )


def mfdma(
    x: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    theta: float = 0.0,
    integrate: bool = False,
    min_segments: int = 4,
    zero_rtol: float = 1e-12,
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
    d_support: float = 1.0,
    seed: SeedLike = None,
) -> ScalingExponents:
    r"""Multifractal detrending moving average exponents (Gu & Zhou 2010).

    Parameters
    ----------
    x
        The series. With ``integrate=False`` (default) it is treated as the
        profile itself — the self-affine signal whose increments define
        :math:`\zeta`. See :func:`mfdma_profile`.
    q
        Moment grid; defaults to 41 points on :math:`[-5,5]`. **Negative
        :math:`q` are required**: the right branch :math:`\alpha>\alpha_0` of
        :math:`f(\alpha)` is generated by :math:`q<0` and is unreachable in
        principle from a non-negative grid.
    scales
        Moving-average window lengths :math:`n`. Defaults to
        :func:`~mmar.core.types.log_spaced_scales`, i.e. geometric spacing from
        8 to :math:`N/4`; a linear grid piles almost every point into the top
        half-decade and turns the fit into a coarse-scale fit.
    theta
        Position parameter in :math:`[0,1]`: 0 backward (causal), 0.5 centred,
        1 forward.
    integrate
        Cumulate ``x`` into an MFDFA-style profile first. See
        :func:`mfdma_profile`; this switch moves every exponent by one.
    min_segments
        Refuse scales leaving fewer than this many disjoint residual segments.
    zero_rtol
        Relative floor below which a segment's fluctuation is treated as
        degenerate and dropped; see the module docstring, bias 4. Set it to 0
        to keep every segment and let a flat stretch blow up the negative
        moments, which is occasionally the diagnostic one wants.
    fit_method
        ``WLS`` by default, weighted by the segment count at each scale: the
        variance of a log-moment estimated from :math:`N_n` segments is
        :math:`O(1/N_n)`, and :math:`N_n` falls like :math:`1/n`.
    select_range
        Choose the scaling window automatically (on the row nearest
        :math:`q=2`) instead of using every admissible scale.
    d_support
        Topological dimension of the support; :math:`\tau=\zeta-d`.
    seed
        Accepted for signature uniformity. **MFDMA is deterministic**: no
        random number is drawn, and the seed is recorded in provenance only
        when it is not ``None``, so that repeated default calls share a root.

    Returns
    -------
    ScalingExponents
        Carrying :math:`\zeta(q)=q\,h(q)`, its standard errors and confidence
        intervals, the per-:math:`q` fits (whose slopes are :math:`h(q)`), the
        scaling range, the full :math:`F_q(n)` surface and provenance.

    Notes
    -----
    Cost is :math:`O(N)` per scale for the filtering plus
    :math:`O(|q|\,N_n)` for the moments, with no Python loop over samples.

    See the module docstring for the estimator's biases; the two that bite
    hardest in practice are that :math:`\theta\neq1/2` does **not** annihilate
    a linear trend, and that :math:`h(q)` for :math:`q\ll0` is high-variance
    because it is set by the quietest segments.
    """
    structure = mfdma_structure(
        x,
        q=q,
        scales=scales,
        theta=theta,
        integrate=integrate,
        min_segments=min_segments,
        zero_rtol=zero_rtol,
        seed=seed,
    )
    return mfdma_exponents_from_structure(
        structure,
        fit_method=fit_method,
        select_range=select_range,
        d_support=d_support,
    )
