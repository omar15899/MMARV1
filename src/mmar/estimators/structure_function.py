r"""Method of moments on increments: the Mandelbrot--Fisher--Calvet partition function.

This is the oldest and most direct multifractal estimator, and the one the
legacy version of this library implemented incorrectly. It measures the
:math:`q`-th absolute moment of the increments of a signal at a range of lags
and reads the scaling exponent off a log--log regression.

The two forms and their exact relation
--------------------------------------
For a signal :math:`X` sampled at :math:`n` points and a lag :math:`\delta`,
write the non-overlapping increments
:math:`\Delta_j = X((j+1)\delta) - X(j\delta)`, :math:`j = 0,\dots,N(\delta)-1`
with :math:`N(\delta) = \lfloor (n-1)/\delta \rfloor`. Then

.. math::
    S(q,\delta) \;=\; \frac{1}{N(\delta)}\sum_{j} |\Delta_j|^{q}
    \;\sim\; \delta^{\zeta(q)}
    \qquad\text{(mean form, SIGNAL view)},

.. math::
    Z(q,\delta) \;=\; \sum_{j} |\Delta_j|^{q} \;\sim\; \delta^{\tau(q)}
    \qquad\text{(sum form, MEASURE view)}.

**They differ by exactly one power of the scale.** Since the increments tile
the sample, :math:`N(\delta) = (n-1)/\delta` whenever :math:`\delta \mid (n-1)`,
so :math:`Z = S \cdot N = S\cdot (n-1)\,\delta^{-1}` and

.. math::
    \log Z(q,\delta) = \log S(q,\delta) - \log\delta + \log(n-1),
    \qquad\Longrightarrow\qquad
    \frac{d\log Z}{d\log\delta} = \frac{d\log S}{d\log\delta} - 1,

that is :math:`\tau(q) = \zeta(q) - 1`, which is the :math:`d=1` case of the
library-wide convention :math:`\zeta = \tau + d`. The legacy code fitted the sum
and called the slope :math:`\tau`, which is correct, but never stated it; anyone
switching to the mean form silently shifted every exponent by one
(:doc:`LEGACY_AUDIT` A6).

This module **natively produces** :math:`\zeta`, because :func:`mfc_exponents`
fits the *mean* form. :func:`partition_function` exposes the sum form for
callers who want the measure view, and is labelled as such.

Why :math:`\tau(0) = -1` is a free self-consistency check
----------------------------------------------------------
:math:`S(0,\delta) = 1` identically, so :math:`\zeta(0)=0` and
:math:`\tau(0)=-1` to machine precision, *unconditionally* --- there is no
sampling error in a mean of ones. In the sum form
:math:`Z(0,\delta) = N(\delta)`, and the slope of
:math:`\log \lfloor (n-1)/\delta\rfloor` against :math:`\log\delta` equals
:math:`-1` exactly only when every scale divides :math:`n-1`. The floor
introduces a systematically negative correction
:math:`\log\!\big(N(\delta)\delta/(n-1)\big) \in (\log(1-\delta/(n-1)),\,0]`
that grows with :math:`\delta`, biasing :math:`\hat\tau(0)` below :math:`-1`.
That is the mechanism behind the legacy :math:`\hat\tau(0) = -1.0174`
(defect D6), and :func:`increment_counts` exposes the counts so it can be
checked at any time.

Known bias
----------
1. **Linearisation effect** (Lashermes, Abry and Chainais 2004). On a sample of
   :math:`n` points the empirical :math:`\hat\zeta(q)` cannot curve beyond a
   critical order :math:`q^\star_n`: past it, the sample moment is carried by
   the single largest (or, for :math:`q<0`, smallest) increment and
   :math:`\hat\zeta` becomes *affine* in :math:`q`. A bilinear
   :math:`\hat\zeta` with a kink is therefore a finite-size artefact, not a
   phase transition, and the Legendre transform of it is a spurious linear
   :math:`f(\alpha)` branch. Use :func:`critical_moment_order` before widening
   :math:`q`.
2. **Negative :math:`q` on a signal.** :math:`\langle|\Delta X|^q\rangle` with
   :math:`q<0` is dominated by increments near zero. For a Gaussian-increment
   process the population moment diverges for :math:`q\le-1`, so the negative
   branch of :math:`\hat\zeta` on fBm is not an estimate of anything: it is a
   function of the sample minimum. On a *measure* (a monotone trading time,
   cell masses bounded away from zero) the negative branch is well posed and
   is exactly where the right half of :math:`f(\alpha)` comes from. This is why
   wavelet leaders exist; see :mod:`mmar.estimators.wavelet_leaders`.
3. **No regularisation of the small-scale end.** Unlike MFDFA, no trend is
   removed: a non-stationary mean contaminates :math:`S(q,\delta)` at large
   :math:`\delta` and flattens :math:`\hat\zeta`.
4. **``zeta_stderr`` is a regression error, not a sampling error, and it is
   several times too small.** Every scale is computed from the *same* series,
   so the residuals of the log--log fit are strongly dependent; the textbook
   WLS formula in :func:`~mmar.core.regression.loglog_fit` treats them as
   independent. Measured on exact fBm (:math:`H=0.7`, default grid, WLS): the
   Monte-Carlo standard deviation of :math:`\hat\zeta(q)` over independent
   paths exceeds the mean reported ``zeta_stderr`` by a factor of
   :math:`4.6`--:math:`5.7` at :math:`n=2^{14}` and :math:`n=2^{16}`, and the
   nominal 95 % ``zeta_ci`` has actual coverage :math:`0.25`--:math:`0.32`.
   Treat the interval as a *goodness-of-line* diagnostic; for inference use a
   block bootstrap or a surrogate ensemble over whole paths. The same caveat
   applies to :mod:`mmar.estimators.ghe`, which states it for the overlapping
   design; it holds for the disjoint design too, and for the same reason.
5. **Finite-sample bias in** :math:`\hat h(2)` **is small and** :math:`O(1/N)`.
   :math:`\log` of a sample mean is downward biased (Jensen) by
   :math:`O(1/N(\delta))`, worst at the coarse end, which tilts the log--log
   slope down. Measured over independent paths (not hand-picked seeds), mean
   :math:`\hat h(2)-H` at :math:`H=0.7`: :math:`-0.008` at :math:`n=2^{13}`,
   :math:`-0.0026\pm0.0010` at :math:`n=2^{15}`, :math:`-0.0016\pm0.0009`
   at :math:`n=2^{16}` (270 replicas). Trimming the coarse end confirms the
   mechanism: at :math:`n=2^{15}`, 300 replicas, the mean bias moves
   :math:`-0.0026\to-0.0012\to-0.0003\to+0.0001` as :math:`\delta_{\max}` falls
   :math:`n/4\to1024\to256\to64`. The effect is real but roughly ten times
   smaller than the per-path scatter (:math:`\mathrm{sd}\approx0.013` at
   :math:`n=2^{16}`), so a single path says nothing about it, and it does *not*
   grow appreciably with :math:`H`.

References
----------
Kolmogorov, A.N. (1941). The local structure of turbulence in incompressible
viscous fluid for very large Reynolds numbers. *Dokl. Akad. Nauk SSSR* 30, 301.

Mandelbrot, B., Fisher, A. and Calvet, L. (1997). A multifractal model of asset
returns. *Cowles Foundation Discussion Paper* 1164.

Calvet, L. and Fisher, A. (2002). Multifractality in asset returns: theory and
evidence. *Rev. Econ. Stat.* 84, 381--406.

Lashermes, B., Abry, P. and Chainais, P. (2004). New insights into the
estimation of scaling exponents. *Int. J. Wavelets Multiresolut. Inf. Process.*
2(4), 497--523.

Hill, B.M. (1975). A simple general approach to inference about the tail of a
distribution. *Ann. Statist.* 3(5), 1163--1174.

Ossiander, M. and Waymire, E.C. (2000). Statistical estimation for
multiplicative cascades. *Ann. Statist.* 28(6), 1533--1560.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import logsumexp

from mmar.core.provenance import InputRef, Provenance, SeedInfo, new_provenance
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
    "critical_moment_order",
    "exponents_from_structure",
    "hurst_from_tau_root",
    "increment_counts",
    "mfc_exponents",
    "mfc_exponents_structure",
    "partition_function",
    "structure_functions",
]

MODULE_VERSION = "1"

ESTIMATOR_ID = "structure_function.mfc"
"""Stable identifier written into every :class:`ScalingExponents` this module returns."""

_LOG_OVERFLOW = 700.0
"""``exp`` of anything past this overflows float64; used to fail loudly instead."""


def _seed_record(seed: SeedLike) -> SeedInfo | None:
    """Seed record for provenance, or ``None`` when no seed was supplied.

    This estimator draws no randomness. ``seed_info(None)`` would mint a fresh
    :class:`numpy.random.SeedSequence` with operating-system entropy and write
    it into the identifying part of the record, so two identical deterministic
    calls would carry different content addresses --- a reproducibility claim
    that is false. An absent seed is recorded as absent.
    """
    return None if seed is None else seed_info(seed)


# --------------------------------------------------------------------------- #
# Increment geometry
# --------------------------------------------------------------------------- #


def increment_counts(
    n: int, scales: Sequence[int] | NDArray[Any], *, overlapping: bool = False
) -> NDArray[np.int64]:
    r"""Number of increments available at each lag. Exposed so it can be audited.

    For a series of :math:`n` samples,

    .. math::
        N_{\text{disjoint}}(\delta) = \Big\lfloor \frac{n-1}{\delta} \Big\rfloor,
        \qquad
        N_{\text{overlap}}(\delta) = n - \delta .

    The disjoint count is the one that carries the whole content of the
    :math:`\tau(0)=-1` identity: it equals :math:`(n-1)/\delta` *exactly* iff
    :math:`\delta \mid (n-1)`, and otherwise the floor makes
    :math:`\log N` fall below :math:`\log(n-1)-\log\delta` by an amount that
    grows with :math:`\delta`, tilting the :math:`q=0` partition-function slope
    below :math:`-1`.

    Parameters
    ----------
    n
        Length of the series (number of samples, not of increments).
    scales
        Integer lags :math:`\delta \ge 1`.
    overlapping
        Stride-one increments instead of stride-:math:`\delta` ones.

    Returns
    -------
    NDArray[np.int64]
        Same length as ``scales``, in the order given.
    """
    s = np.asarray(scales, dtype=np.int64).ravel()
    if s.size == 0:
        raise ValueError("scales is empty")
    if np.any(s < 1):
        raise ValueError(f"every scale must be >= 1, got min={int(s.min())}")
    if n < 2:
        raise ValueError(f"n must be at least 2 to form an increment, got {n}")
    if np.any(s > n - 1):
        raise ValueError(f"scale {int(s.max())} exceeds n-1 = {n - 1}; no increment exists")
    counts = (n - s) if overlapping else ((n - 1) // s)
    return np.asarray(counts, dtype=np.int64)


def _effective_counts(n: int, scales: NDArray[np.int64], *, overlapping: bool) -> NDArray[np.int64]:
    r"""Number of *independent* windows, which is what the fit must be weighted by.

    Overlapping increments at lag :math:`\delta` number :math:`n-\delta`, but
    consecutive ones share :math:`\delta-1` samples, so the variance of
    :math:`\log \hat S(q,\delta)` does not fall like :math:`1/(n-\delta)`; it
    saturates at the disjoint-block rate :math:`\delta/n`. Weighting the
    log--log fit by the raw overlapping count would therefore claim a precision
    at coarse scales that the data does not have --- a subtler version of the
    legacy defect D5, where seven-sample moments entered the fit with the same
    weight as the full-sample one.
    """
    disjoint = np.asarray((n - 1) // scales, dtype=np.int64)
    return disjoint if overlapping else np.asarray(increment_counts(n, scales), dtype=np.int64)


def _resolve_scales(
    n: int,
    scales: Sequence[int] | NDArray[Any] | None,
    *,
    min_windows: int,
    overlapping: bool,
) -> NDArray[np.int64]:
    r"""Default grid, spine check, and the exact-count check the spine cannot do.

    :func:`~mmar.core.validate.check_scales` filters on :math:`\lfloor n/\delta
    \rfloor`, but the number of increments is :math:`\lfloor (n-1)/\delta
    \rfloor`, which is one smaller whenever :math:`\delta \mid n`. Fitting a
    moment estimated from three windows because the spine counted four is
    exactly the failure mode ``min_windows`` exists to prevent, so the count is
    re-checked here against the geometry this estimator actually uses.
    """
    raw = log_spaced_scales(n) if scales is None else np.asarray(scales, dtype=np.int64)
    sc = check_scales(raw, n, min_windows=min_windows, name="scales")
    keep = _effective_counts(n, sc, overlapping=overlapping) >= min_windows
    if not keep.any():
        raise ValueError(
            f"no scale leaves >= {min_windows} independent increments for n={n}; "
            f"the coarsest admissible lag is {(n - 1) // min_windows}"
        )
    return np.ascontiguousarray(sc[keep])


def _increments(x: FloatArray, s: int, *, overlapping: bool) -> FloatArray:
    """Increments at lag ``s``: stride 1 if ``overlapping``, stride ``s`` otherwise."""
    if overlapping:
        return np.asarray(x[s:] - x[:-s], dtype=np.float64)
    m = (x.size - 1) // s
    j = np.arange(m, dtype=np.intp) * s
    return np.asarray(x[j + s] - x[j], dtype=np.float64)


# --------------------------------------------------------------------------- #
# The raw scaling surface
# --------------------------------------------------------------------------- #


def _moment_row(
    a: FloatArray, q: FloatArray, *, absolute: bool, signed: FloatArray | None
) -> FloatArray:
    r"""One column of the scaling surface: :math:`S(q,\delta)` for all :math:`q`.

    The absolute moments are formed in the log domain,
    :math:`\log S = \operatorname{logsumexp}(q\log|\Delta|) - \log N`, rather
    than as ``(a**q).mean()``. At :math:`q=-5` on cascade cell masses of order
    :math:`2^{-16}` the naive form reaches :math:`10^{24}` per term and at
    :math:`q=\pm 20` it overflows outright, while the log-domain form is exact
    to a few ulp. It also makes :math:`S(0,\delta)=\exp(\log N - \log N) = 1`
    bit-exactly, which is what turns :math:`\tau(0)=-1` into a machine-precision
    identity rather than an approximate one.
    """
    n = a.size
    if not absolute:
        if signed is None:  # pragma: no cover - guarded by the caller
            raise ValueError("signed increments required when absolute=False")
        # A signed moment of order q <= 0 over an increment that is exactly zero
        # is infinite. Nothing is dropped on this path (the sub-sample would
        # differ from the absolute one and the two would not be comparable), so
        # the infinity is produced deliberately and reported by the caller.
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray([float(np.mean(signed**qi)) for qi in q], dtype=np.float64)
    with np.errstate(divide="ignore"):
        log_a = np.log(a)
    log_n = math.log(n)
    out = np.empty(q.size, dtype=np.float64)
    for i, qi in enumerate(q):
        out[i] = float(logsumexp(qi * log_a)) - log_n
    if np.any(np.abs(out) > _LOG_OVERFLOW):
        bad = int(np.sum(np.abs(out) > _LOG_OVERFLOW))
        raise ValueError(
            f"{bad}/{q.size} moments overflow float64 at this scale "
            f"(|log S| > {_LOG_OVERFLOW:g}). Narrow the q grid, or rescale the series: "
            "x -> c*x multiplies S by c^q and shifts only the intercept, never the slope."
        )
    return np.asarray(np.exp(out), dtype=np.float64)


def structure_functions(
    x: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    overlapping: bool = False,
    absolute: bool = True,
    min_windows: int = 4,
    provenance: Provenance | None = None,
) -> StructureFunctions:
    r"""The raw scaling surface :math:`S(q,\delta)=\langle|\Delta_\delta X|^q\rangle`.

    .. math::
        S(q,\delta) \;=\; \frac{1}{N(\delta)}
        \sum_{j=0}^{N(\delta)-1} \big| X(t_j+\delta) - X(t_j) \big|^{q}
        \;\sim\; \delta^{\zeta(q)} .

    Parameters
    ----------
    x
        The signal (a *path*, not its increments).
    q
        Moment orders; ``None`` gives the library default :math:`[-5,5]` on 41
        points. The negative branch is what generates :math:`\alpha>\alpha_0`.
    scales
        Integer lags; ``None`` gives :func:`~mmar.core.types.log_spaced_scales`.
        Scales leaving fewer than ``min_windows`` disjoint windows are dropped
        by :func:`~mmar.core.validate.check_scales`.
    overlapping
        ``False`` (default) uses stride-:math:`\delta` increments: the
        Mandelbrot--Fisher--Calvet choice, in which the increments are disjoint
        and hence, for a process with weakly dependent increments, close to
        independent. ``True`` uses stride-one increments, giving
        :math:`n-\delta` of them instead of :math:`(n-1)/\delta`. That is a
        large variance reduction at coarse scales but the windows are strongly
        correlated, so the *effective* sample size stays :math:`O(n/\delta)`;
        :func:`mfc_exponents` weights the regression accordingly and never by
        the raw overlapping count.
    absolute
        ``True`` computes :math:`\langle|\Delta X|^q\rangle`, the object the
        multifractal formalism is stated for. ``False`` computes the *signed*
        moment :math:`\langle(\Delta X)^q\rangle`, which requires integer
        :math:`q`, may be negative or zero, and therefore admits no log--log
        fit; it is exposed for skewness diagnostics (the analogue of
        Kolmogorov's :math:`4/5` law), not for exponent estimation.

    Returns
    -------
    StructureFunctions
        ``values[i, j] = S(q_i, delta_j)``; ``counts[j]`` is the number of
        increments actually averaged at scale :math:`j`.

    Notes
    -----
    **Exact zeros.** An increment of exactly zero makes every moment with
    :math:`q\le 0` infinite. When the grid contains any :math:`q\le0`, zero
    increments are dropped *at every* :math:`q` --- not only the negative ones
    --- so that all moments at a given scale are computed from the same
    sub-sample and the Legendre transform stays coherent; the number dropped
    per scale is recorded in ``provenance.diagnostics["zeros_dropped"]``. When
    every :math:`q>0` they are kept, because :math:`0^q=0` is then the correct
    contribution. Nothing is dropped silently.
    """
    xa = validate_series(x, name="x", min_length=16)
    qa = validate_q_grid(q)
    n = xa.size
    sc = _resolve_scales(n, scales, min_windows=min_windows, overlapping=overlapping)

    if not absolute and np.any(np.abs(qa - np.round(qa)) > 1e-12):
        raise ValueError(
            "absolute=False computes the signed moment <(dX)^q>, which is real only "
            "for integer q; the grid contains non-integer orders. Use absolute=True "
            "for the multifractal formalism."
        )

    drop_zeros = bool(np.any(qa <= 0.0)) and absolute
    values = np.empty((qa.size, sc.size), dtype=np.float64)
    counts = np.empty(sc.size, dtype=np.int64)
    zeros_dropped = np.zeros(sc.size, dtype=np.int64)

    for j, s in enumerate(sc):
        d = _increments(xa, int(s), overlapping=overlapping)
        a = np.abs(d)
        if drop_zeros:
            keep = a > 0.0
            zeros_dropped[j] = int(a.size - int(keep.sum()))
            if zeros_dropped[j]:
                d, a = d[keep], a[keep]
            if a.size < min_windows:
                raise ValueError(
                    f"scale {int(s)} retains only {a.size} non-zero increments after "
                    f"dropping {zeros_dropped[j]} exact zeros; the q<=0 moments there are "
                    "not estimable. Restrict q to positive orders or drop this scale."
                )
        counts[j] = a.size
        values[:, j] = _moment_row(a, qa, absolute=absolute, signed=None if absolute else d)

    prov = provenance or new_provenance(
        f"{ESTIMATOR_ID}.structure",
        estimator_version=MODULE_VERSION,
        params={
            "overlapping": bool(overlapping),
            "absolute": bool(absolute),
            "min_windows": int(min_windows),
            "q": qa,
            "scales": sc,
        },
        inputs=(InputRef.from_array("x", xa),),
        seed=None,
        notes=("mean form: S(q,delta) = <|dX|^q> ~ delta^zeta(q); zeta = tau + d",),
    )
    n_nonfinite = int((~np.isfinite(values)).sum())
    if drop_zeros:
        policy = "q<=0 present: exact zeros dropped at every q"
    elif absolute:
        policy = "not needed: every q > 0, so 0^q = 0 is the correct contribution"
    else:
        policy = "not applicable: signed moments are never sub-sampled"
    if n_nonfinite:
        warnings.warn(
            f"{n_nonfinite}/{values.size} entries of the scaling surface are non-finite "
            f"(absolute={absolute}); with absolute=False an exact zero increment makes "
            "every moment of order q <= 0 infinite. No log-log fit is possible on those "
            "rows. See provenance.diagnostics['n_nonfinite_values'].",
            UserWarning,
            stacklevel=2,
        )
    prov = prov.with_diagnostics(
        n_samples=int(n),
        zeros_dropped=[int(z) for z in zeros_dropped],
        drop_zeros_policy=policy,
        n_nonfinite_values=n_nonfinite,
    )

    return StructureFunctions(
        q=qa,
        scales=np.asarray(sc, dtype=np.float64),
        values=values,
        counts=counts,
        provenance=prov,
    )


def partition_function(
    x: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    min_windows: int = 4,
) -> StructureFunctions:
    r"""The SUM form :math:`Z(q,\delta)=\sum_j|\Delta_j|^q \sim \delta^{\tau(q)}`.

    **This returns the MEASURE convention.** Its log--log slope is
    :math:`\tau(q)`, not :math:`\zeta(q)`; add :math:`d=1` to compare with
    anything this library stores in a :class:`ScalingExponents`.

    Proposition
    -----------
    Let :math:`N(\delta)` be the number of disjoint increments. Then
    :math:`Z(q,\delta) = N(\delta)\,S(q,\delta)` by definition of the mean, and
    whenever :math:`\delta \mid (n-1)`, :math:`N(\delta)=(n-1)/\delta` exactly,
    so

    .. math::
        \log Z(q,\delta) = \log S(q,\delta) + \log(n-1) - \log\delta .

    Differentiating in :math:`\log\delta`,

    .. math::
        \boxed{\;
        \text{slope}_{Z}(q) \;=\; \text{slope}_{S}(q) - 1
        \;=\; \zeta(q) - 1 \;=\; \tau(q)
        \;}

    for :math:`d=1`. :math:`\square` This is the identity the legacy code relied
    on without ever writing down (LEGACY_AUDIT A6), and it is why its
    :math:`f=q\alpha-\tau` was nevertheless self-consistent.

    Corollary (the :math:`q=0` check). :math:`Z(0,\delta)=N(\delta)`, so
    :math:`\hat\tau(0)` is the slope of :math:`\log N` against
    :math:`\log\delta` and equals :math:`-1` *exactly* iff every scale divides
    :math:`n-1`. Otherwise :math:`\lfloor\cdot\rfloor` biases it below
    :math:`-1`. See :func:`increment_counts` and the module docstring.

    Notes
    -----
    Only disjoint increments are offered here. Overlapping windows would count
    the same mass :math:`\delta` times and the sum would no longer be a
    partition function of a measure.
    """
    sf = structure_functions(
        x,
        q=q,
        scales=scales,
        overlapping=False,
        absolute=True,
        min_windows=min_windows,
    )
    if sf.counts is None:  # pragma: no cover - structure_functions always sets counts
        raise RuntimeError("structure surface is missing its window counts")
    z = sf.values * sf.counts[None, :].astype(np.float64)
    prov = sf.provenance
    if prov is not None:
        prov = prov.with_diagnostics(
            convention="measure: slope of log Z vs log delta is tau(q) = zeta(q) - d"
        )
    return StructureFunctions(
        q=sf.q,
        scales=sf.scales,
        values=np.asarray(z, dtype=np.float64),
        counts=sf.counts,
        provenance=prov,
    )


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #


def mfc_exponents_structure(
    x: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    overlapping: bool = False,
    absolute: bool = True,
    min_windows: int = 4,
    d_support: float = 1.0,
    seed: SeedLike = None,
) -> StructureFunctions:
    """The scaling surface :func:`mfc_exponents` fits, computed on its own.

    Exposed so that exponents can be refitted over a different scaling range,
    with a different weighting or a different regression method, without
    recomputing the moments --- which is the expensive half. Feed the result to
    :func:`exponents_from_structure`.
    """
    xa = validate_series(x, name="x", min_length=64)
    qa = validate_q_grid(q)
    sc = _resolve_scales(xa.size, scales, min_windows=min_windows, overlapping=overlapping)
    prov = new_provenance(
        f"{ESTIMATOR_ID}.structure",
        estimator_version=MODULE_VERSION,
        params={
            "overlapping": bool(overlapping),
            "absolute": bool(absolute),
            "min_windows": int(min_windows),
            "d_support": float(d_support),
            "q": qa,
            "scales": sc,
        },
        inputs=(InputRef.from_array("x", xa),),
        seed=_seed_record(seed),
        notes=("mean form: S(q,delta) = <|dX|^q> ~ delta^zeta(q)",),
    )
    return structure_functions(
        xa,
        q=qa,
        scales=sc,
        overlapping=overlapping,
        absolute=absolute,
        min_windows=min_windows,
        provenance=prov,
    )


def _fit_weights(structure: StructureFunctions) -> FloatArray | None:
    """Inverse-variance weights: the *independent*-window count at each scale.

    For a disjoint surface that is the increment count itself. For an
    overlapping one it is the disjoint-equivalent count, recovered from the
    provenance record (``params['overlapping']`` and
    ``diagnostics['n_samples']``); a surface carrying no provenance falls back
    to its stored counts, which for an overlapping design over-weights the
    coarse scales.
    """
    if structure.counts is None:  # pragma: no cover - always populated here
        return None
    prov = structure.provenance
    if prov is not None and bool(prov.params.get("overlapping", False)):
        n = prov.diagnostics.get("n_samples")
        if isinstance(n, int):
            sc = np.asarray(np.rint(structure.scales), dtype=np.int64)
            return weighted_by_counts(_effective_counts(n, sc, overlapping=True))
    return weighted_by_counts(structure.counts)


def exponents_from_structure(
    structure: StructureFunctions,
    *,
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
    d_support: float = 1.0,
    weights: FloatArray | None = None,
    reference_q: float = 2.0,
    estimator: str = ESTIMATOR_ID,
    provenance: Provenance | None = None,
) -> ScalingExponents:
    r"""Fit :math:`\zeta(q)` from a precomputed scaling surface.

    One log--log regression per :math:`q`, all of them over the *same* scale
    window. Selecting a different range for each :math:`q` would make
    :math:`\hat\zeta` a mixture of exponents measured over different physical
    ranges, and its Legendre transform would not be the spectrum of anything;
    when ``select_range=True`` the window is therefore chosen once, on the row
    nearest ``reference_q``, and applied to every moment.

    Provenance. When ``provenance`` is not supplied and the surface carries one,
    a child record is *derived* from it (a Merkle edge naming the refit
    parameters), matching :func:`~mmar.estimators.mfdfa.mfdfa_from_structure`.
    Returning an unprovenanced result from a documented entry point would make
    the refit path the one hole in the audit chain.

    See :func:`mfc_exponents` for why ``zeta_stderr`` and ``zeta_ci`` are
    regression uncertainties, not sampling uncertainties.
    """
    q = structure.q
    scales = structure.scales
    w = weights
    if w is not None and np.asarray(w).shape != scales.shape:
        raise ValueError(
            f"weights{np.asarray(w).shape} must match the surface's "
            f"scales{scales.shape}; one weight per scale, not per q"
        )
    if w is None and fit_method is FitMethod.WLS:
        w = _fit_weights(structure)

    mask = np.ones(scales.size, dtype=bool)
    common_range: ScalingRange | None = None
    if select_range:
        ref = int(np.argmin(np.abs(q - reference_q)))
        sel = select_scaling_range(
            scales,
            structure.values[ref],
            weights=w,
            method=fit_method,
            log_y=True,
        )
        mask = np.asarray(sel.mask, dtype=bool)
        common_range = sel.scaling_range

    xs = scales[mask]
    ws = None if w is None else np.asarray(w[mask], dtype=np.float64)

    fits: list[LogLogFit] = []
    zeta = np.empty(q.size, dtype=np.float64)
    stderr = np.empty(q.size, dtype=np.float64)
    lo = np.empty(q.size, dtype=np.float64)
    hi = np.empty(q.size, dtype=np.float64)
    for i in range(q.size):
        fit = loglog_fit(
            xs,
            structure.values[i][mask],
            method=fit_method,
            weights=ws,
            scaling_range=common_range,
        )
        fits.append(fit)
        zeta[i] = fit.slope
        stderr[i] = fit.slope_stderr
        lo[i], hi[i] = fit.slope_ci

    if common_range is None:
        common_range = ScalingRange(
            s_min=float(xs.min()),
            s_max=float(xs.max()),
            n_scales=int(xs.size),
            selection="manual",
            # nanmedian, not median: S(0, delta) = 1 bit-exactly, so the q = 0 row
            # has zero total sum of squares and its R^2 is NaN by construction. A
            # plain median would poison the fit-quality summary of every grid that
            # contains q = 0 -- which the library default linspace(-5, 5, 41) does.
            r_squared=float(np.nanmedian([f.r_squared for f in fits])),
            mask=tuple(bool(m) for m in mask),
        )

    prov = provenance
    if prov is None and structure.provenance is not None:
        prov = structure.provenance.derive(
            estimator,
            estimator_version=MODULE_VERSION,
            params={
                "fit_method": str(fit_method),
                "select_range": bool(select_range),
                "d_support": float(d_support),
                "reference_q": float(reference_q),
                "weights_supplied": weights is not None,
            },
            notes=("refit of a precomputed scaling surface; no moment recomputed",),
        )
        # `derive` starts a fresh diagnostics map, so without this a refitted
        # result no longer records how many exact zeros were dropped or how many
        # moment cells overflowed -- the two facts that decide whether its
        # negative-q branch means anything. Diagnostics never enter `root`.
        prov = prov.with_diagnostics(
            **{
                **dict(structure.provenance.diagnostics),
                "scaling_range": common_range.to_canonical(),
            }
        )

    return ScalingExponents(
        q=q,
        zeta=zeta,
        d_support=float(d_support),
        zeta_stderr=stderr,
        zeta_ci=(lo, hi),
        fits=tuple(fits),
        scaling_range=common_range,
        structure=structure,
        estimator=estimator,
        provenance=prov,
    )


def mfc_exponents(
    x: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    overlapping: bool = False,
    absolute: bool = True,
    min_windows: int = 4,
    reference_q: float = 2.0,
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
    d_support: float = 1.0,
    seed: SeedLike = None,
) -> ScalingExponents:
    r"""Mandelbrot--Fisher--Calvet method of moments: :math:`\hat\zeta(q)`.

    Fits :math:`\log S(q,\delta) = \zeta(q)\log\delta + c_q` by weighted least
    squares over a geometric grid of lags, one regression per :math:`q`.

    **Native convention: SIGNAL.** The mean form is fitted, so the slope *is*
    :math:`\zeta(q)`; the returned object derives
    :math:`\tau = \zeta - d` and :math:`h(q)=\zeta(q)/q`. Use
    :func:`partition_function` if you want the sum form, whose slope is
    :math:`\tau` directly (the two differ by exactly one, proved there).

    Parameters
    ----------
    x
        The signal. For a *measure*, pass its cumulative distribution (the
        trading time): the increments of the CDF over a box are that box's
        mass, so :math:`S(q,\delta)=Z(q,\delta)/N(\delta)` exactly and the
        cascade oracle :math:`\zeta=\tau+1` is recovered to machine precision
        on dyadic-aligned scales.
    q
        Moment grid; ``None`` gives :math:`[-5,5]` on 41 points. **Negative
        orders are not optional**: the branch :math:`\alpha>\alpha_0` of
        :math:`f(\alpha)` is generated entirely by :math:`q<0`, and a
        :math:`q\ge0` grid recovers a left fragment of the spectrum (legacy
        defect D2). See the module docstring for when the negative branch is
        trustworthy.
    scales
        Integer lags; ``None`` gives a geometric grid capped at :math:`n/4`.
    overlapping, absolute, min_windows
        As in :func:`structure_functions`.
    reference_q
        Row on which the scaling range is selected when ``select_range=True``.
    fit_method
        ``WLS`` by default, weighted by the number of *independent* windows at
        each scale, which falls like :math:`1/\delta`. Unweighted OLS over a
        grid whose coarse end holds a handful of windows is how the legacy code
        let seven-sample moments set the slope (defect D5).
    select_range
        Choose the widest linear window automatically instead of using every
        scale.
    d_support
        Topological dimension of the support; 1 for a time series.
    seed
        Unused: this estimator is deterministic. Accepted for signature
        uniformity across the library and recorded in provenance, so that a
        manifest produced here is comparable with one from a bootstrap-based
        estimator.

    Returns
    -------
    ScalingExponents
        With per-:math:`q` fits, standard errors, confidence intervals, the
        scaling range, the full :class:`StructureFunctions` surface and a
        content-addressed provenance record.

        ``zeta_stderr`` and ``zeta_ci`` describe how well a straight line fits
        the :math:`n_{\text{scales}}` points, **not** how much
        :math:`\hat\zeta` would move on another realisation of the process. All
        the scales come from one series, so the residuals are dependent and the
        interval is roughly five times too narrow: on exact fBm the nominal
        95 % interval covers about 30 % (module docstring, bias 4). Do not
        propagate it as a sampling error.

    Notes
    -----
    Bias. See the module docstring: the linearisation effect bounds the
    :math:`|q|` at which :math:`\hat\zeta` can still curve, on a signal with
    Gaussian increments the :math:`q<0` branch estimates a divergent population
    moment, the reported uncertainty is a goodness-of-line diagnostic, and
    :math:`\hat h(2)` carries a small :math:`O(1/N)` Jensen bias. All are
    properties of the *method*, not of the implementation, and are the reason
    this estimator is the baseline rather than the recommendation.
    """
    xa = validate_series(x, name="x", min_length=64)
    qa = validate_q_grid(q)
    sc = _resolve_scales(xa.size, scales, min_windows=min_windows, overlapping=overlapping)

    prov = new_provenance(
        ESTIMATOR_ID,
        estimator_version=MODULE_VERSION,
        params={
            "overlapping": bool(overlapping),
            "absolute": bool(absolute),
            "min_windows": int(min_windows),
            "reference_q": float(reference_q),
            "fit_method": str(fit_method),
            "select_range": bool(select_range),
            "d_support": float(d_support),
            "q": qa,
            "scales": sc,
        },
        inputs=(InputRef.from_array("x", xa),),
        seed=_seed_record(seed),
        notes=(
            "signal view: slope of log <|dX|^q> vs log delta is zeta(q); tau = zeta - d",
            "seed is recorded but unused: the estimator is deterministic",
        ),
    )

    structure = structure_functions(
        xa,
        q=qa,
        scales=sc,
        overlapping=overlapping,
        absolute=absolute,
        min_windows=min_windows,
        provenance=prov,
    )
    return exponents_from_structure(
        structure,
        fit_method=fit_method,
        select_range=select_range,
        d_support=d_support,
        weights=_fit_weights(structure) if fit_method is FitMethod.WLS else None,
        reference_q=reference_q,
        estimator=ESTIMATOR_ID,
        provenance=structure.provenance,
    )


# --------------------------------------------------------------------------- #
# The MFC Hurst identity
# --------------------------------------------------------------------------- #


def hurst_from_tau_root(exponents: ScalingExponents, *, q_min: float = 1e-6) -> float:
    r"""MMAR Hurst exponent from :math:`\tau(q^\*)=0`, :math:`H = 1/q^\*`.

    **Identity (Mandelbrot, Fisher and Calvet 1997).** For
    :math:`X(t)=B_H(\theta(t))` with a conservative trading time,
    :math:`\tau_X(q)=\tau_\theta(qH)` and :math:`\tau_\theta(1)=0`, hence

    .. math::
        \tau_X(1/H) = \tau_\theta(1) = 0 .

    The positive root is unique: :math:`\tau_X` is concave with
    :math:`\tau_X(0)=-1<0`, so it crosses zero at most twice, and the crossing
    on :math:`q>0` is :math:`q^\*=1/H`.

    Method
    ------
    A shape-preserving monotone (PCHIP) interpolant is built through the
    estimated :math:`\tau` restricted to :math:`q>q_{\min}`, the first sign
    change is bracketed, and :func:`scipy.optimize.brentq` refines it. PCHIP
    rather than a smoothing spline because a cubic smoothing spline can
    overshoot between knots and manufacture a root that the data does not have;
    ``brentq`` rather than ``UnivariateSpline.roots`` because the latter
    returns *all* roots of the spline's polynomial pieces, in no meaningful
    order, and silently returns an empty array when the fit did not cross zero
    --- at which point the caller's ``[0]`` raises, or worse, their fallback
    quietly substitutes another estimator.

    Returns
    -------
    float
        :math:`H=1/q^\*`, or NaN if :math:`\hat\tau` does not change sign on
        the positive branch. NaN is accompanied by a :class:`UserWarning`
        stating the observed range of :math:`\tau`, so a failed identification
        cannot propagate unnoticed into a calibration.

    Notes
    -----
    Bias. Since :math:`H<1` the root sits at :math:`q^\*>1`, in the region where
    high-order moments are least reliable, so :math:`\hat H` inherits the full
    curvature error of :math:`\hat\tau` there and is the *least* robust summary
    this library computes. When :math:`\hat\tau` is noisy prefer
    :math:`h(2)=\zeta(2)/2` or a joint calibration.
    """
    order = np.argsort(exponents.q)
    qs = np.asarray(exponents.q[order], dtype=np.float64)
    ts = np.asarray(exponents.tau[order], dtype=np.float64)
    keep = np.isfinite(ts) & (qs > q_min)
    qs, ts = qs[keep], ts[keep]
    if qs.size < 2:
        warnings.warn(
            f"only {qs.size} finite tau values with q > {q_min}; the MFC identity "
            "tau(1/H) = 0 cannot be evaluated. Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    sign_change = np.nonzero(np.diff(np.sign(ts)) != 0)[0]
    if sign_change.size == 0:
        warnings.warn(
            f"tau has no sign change on q in [{qs[0]:.4g}, {qs[-1]:.4g}] "
            f"(tau ranges over [{ts.min():.4g}, {ts.max():.4g}]); H is not identified "
            "by tau(1/H) = 0 on this grid. Widen the positive q window. Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    i = int(sign_change[0])
    interp = PchipInterpolator(qs, ts)
    q_star = float(brentq(lambda z: float(interp(z)), qs[i], qs[i + 1], xtol=1e-14, rtol=8.9e-16))
    if q_star <= 0.0:  # pragma: no cover - excluded by the q > q_min filter
        warnings.warn(f"non-positive root q* = {q_star!r}; H undefined.", UserWarning, stacklevel=2)
        return float("nan")
    return 1.0 / q_star


# --------------------------------------------------------------------------- #
# Moment existence
# --------------------------------------------------------------------------- #


def _hill(a: FloatArray, k: int) -> float:
    r"""Hill (1975) tail index from the top :math:`k` order statistics.

    .. math::
        \hat\gamma_k = \frac1k\sum_{i=1}^{k}
        \ln\frac{a_{(i)}}{a_{(k+1)}},\qquad \hat\xi_k = 1/\hat\gamma_k ,

    with :math:`a_{(1)}\ge a_{(2)}\ge\dots`. :math:`\hat\xi` estimates the
    Pareto index of :math:`\Pr(|a|>u)\sim u^{-\xi}`.
    """
    top = np.sort(a)[::-1][: k + 1]
    gamma = float(np.mean(np.log(top[:k])) - math.log(top[k]))
    return 1.0 / gamma if gamma > 0.0 else float("inf")


def critical_moment_order(
    x: Sequence[float] | NDArray[Any],
    *,
    lag: int = 1,
    fractions: Sequence[float] = (0.005, 0.01, 0.02, 0.05, 0.10),
) -> float:
    r"""Estimated :math:`q_{\mathrm{crit}}`: the order past which the moment diverges.

    If the increment magnitudes satisfy
    :math:`\Pr(|\Delta X| > u) \sim L(u)\,u^{-\xi}` with slowly varying
    :math:`L`, then :math:`\mathbb E|\Delta X|^q < \infty` iff :math:`q<\xi`.
    The Hill estimator of :math:`\xi` therefore *is* an estimate of
    :math:`q_{\mathrm{crit}}`, and fitting :math:`\hat\zeta(q)` beyond it
    produces a beautifully straight log--log line that estimates nothing: past
    :math:`q_{\mathrm{crit}}` the sample moment is carried by the single
    largest increment, :math:`\hat S(q,\delta)\approx\max_j|\Delta_j|^q/N`, so
    :math:`\hat\zeta(q)` becomes affine and the Legendre transform grows a
    spurious linear branch (Lashermes, Abry and Chainais 2004). For an
    :math:`\alpha`-stable motion this is exactly the mechanism by which a
    monofractal process is reported as strongly multifractal
    (Mandelbrot 1997).

    Method
    ------
    :math:`\hat\xi` is computed at each tail fraction in ``fractions`` and the
    **median** is returned. Reporting a single :math:`k` invites the classic
    Hill-plot cherry-pick; the median across a decade of :math:`k` is a crude
    but honest plateau estimate.

    Known bias
    ----------
    The Hill estimator is a *maximum-domain-of-attraction* statistic: it is
    consistent only inside the Fréchet domain. A Gaussian or lognormal sample
    (all moments finite, :math:`\xi=+\infty`) still returns a finite number,
    which drifts upward roughly like :math:`\sqrt{\log n}` --- so a *large*
    value should be read as "no evidence of a finite critical order", never as
    a Pareto exponent. It is also biased when :math:`L` is far from constant,
    and it ignores dependence entirely, which for a cascade (whose increments
    are strongly dependent) inflates the variance without a closed-form
    correction.

    Returns
    -------
    float
        The median Hill index over ``fractions``; NaN if fewer than 50 strictly
        positive increments are available.
    """
    xa = validate_series(x, name="x", min_length=16)
    if lag < 1:
        raise ValueError(f"lag must be a positive integer, got {lag}")
    a = np.abs(xa[lag:] - xa[:-lag])
    a = a[np.isfinite(a) & (a > 0.0)]
    m = a.size
    if m < 50:
        return float("nan")
    fr = np.asarray(fractions, dtype=np.float64)
    if fr.size == 0 or np.any(fr <= 0.0) or np.any(fr >= 1.0):
        raise ValueError("every tail fraction must lie strictly inside (0, 1)")
    ks = np.unique(np.clip(np.ceil(fr * m).astype(np.int64), 2, m - 1))
    return float(np.median([_hill(a, int(k)) for k in ks]))
