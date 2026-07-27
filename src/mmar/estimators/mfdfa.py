r"""Multifractal detrended fluctuation analysis.

MFDFA is the workhorse estimator for series that are *non-stationary in the
mean*: it replaces the raw increment moment of the structure-function method by
the residual variance of a local polynomial fit, so a deterministic trend of
degree :math:`\le m` is annihilated instead of being read as a scaling exponent.

Algorithm (Kantelhardt et al. 2002, §2)
---------------------------------------
Given a series :math:`x_1,\dots,x_N`:

1. **Profile.** :math:`Y(i)=\sum_{k\le i}\big(x_k-\bar x\big)`. This is a
   cumulative sum, so the input is expected to be *noise-like* (returns,
   fGn, a measure's cell masses) and the profile is the *motion*.

2. **Partition, from both ends.** For a scale :math:`s`, let
   :math:`N_s=\lfloor N/s\rfloor`. Windows :math:`v=1..N_s` tile
   :math:`Y(1..N_s s)` and windows :math:`v=N_s+1..2N_s` tile
   :math:`Y(N-N_s s+1..N)`. The second sweep exists purely so that the
   :math:`N-N_s s` samples at the tail are not thrown away, which for
   :math:`s\approx N/4` is up to a quarter of the record.

3. **Detrend.** In each window fit a least-squares polynomial
   :math:`p_v^{(m)}` of order :math:`m` and keep the residual.

4. **Window variance.**
   :math:`F^2(v,s)=\frac1s\sum_{i=1}^{s}\big[Y((v-1)s+i)-p_v^{(m)}(i)\big]^2`.

5. **Generalised fluctuation.**

   .. math::
       F_q(s)=\Big\{\frac{1}{2N_s}\sum_{v=1}^{2N_s}
                \big[F^2(v,s)\big]^{q/2}\Big\}^{1/q}, \qquad q\neq 0,

   and at :math:`q=0` the value is the **logarithmic limit**, not the naive
   substitution :math:`(\cdot)^{1/0}`:

   .. math::
       F_0(s)=\exp\Big\{\frac{1}{4N_s}\sum_{v=1}^{2N_s}
                \ln F^2(v,s)\Big\}.

   (Take :math:`\ln F_q(s) = q^{-1}\ln\langle e^{(q/2)\ln F^2}\rangle`, expand
   the cumulant generating function about :math:`q=0`: the limit is
   :math:`\tfrac12\langle \ln F^2\rangle`, i.e. the geometric mean of
   :math:`F(v,s)`. Setting :math:`q=0` in the power formula gives
   :math:`\langle 1\rangle^{\infty}=1`, which is the classic MFDFA bug.)

6. **Exponent.** :math:`F_q(s)\sim s^{h(q)}`, fitted by weighted log--log
   regression.

Convention
----------
The quantity this estimator produces **natively** is the generalised Hurst
function :math:`h(q)` --- a *fluctuation* exponent of the profile, not a box
sum. The library's carrier is :math:`\zeta`, so the conversion applied here is

.. math::
    \zeta(q) = q\,h(q), \qquad \tau(q) = \zeta(q) - d = q\,h(q) - d ,

with :math:`d=1` for a time series; the second identity is Kantelhardt's
eq. (14). Two consequences are used as self-checks below:
:math:`\tau(0)=-1` holds *identically* (it is :math:`0\cdot h(0)-1`, an
algebraic identity, not an estimate), and for a conservative measure fed in as
``x`` one must find :math:`h(1)\approx 1`, i.e. :math:`\tau(1)\approx 0`.

Anchors: fBm of Hurst :math:`H` gives a flat :math:`h(q)\equiv H` and
:math:`\zeta(q)=qH`; the binomial cascade :math:`(m_0,1-m_0)` gives
:math:`\zeta(q)=1-\log_2(m_0^q+m_1^q)`.

.. warning::
    **The integration trap.** ``integrate=True`` (the default) applies step 1.
    If the input is *already* a motion --- a log-price, a cumulated cascade
    (trading time), any integrated series --- the profile integrates it a
    second time and every exponent comes back shifted by exactly one:
    :math:`\hat h = h + 1`, hence :math:`\hat\zeta(q)=\zeta(q)+q`. This is by
    far the most common MFDFA user error and it is silent, because
    :math:`h=1.3` on a log-price still looks like a publishable number. Pass
    ``integrate=False`` when ``x`` is the profile.

Bias
----
*Small scales.* Detrending with :math:`m+1` free coefficients removes a
fraction :math:`\approx (m+1)/s` of the window's degrees of freedom, so
:math:`F^2(v,s)` is deflated by roughly :math:`1-(m+1)/s`. The deflation is
strongest at small :math:`s` and dies off as :math:`1/s`, which tilts the
log--log plot upward and **overestimates** :math:`h`. Kantelhardt et al. (2001,
2002) give the exact white-noise correction and recommend discarding
:math:`s\lesssim 10(m+1)`; that is the default ``s_min`` here
(:math:`20` for DFA1), enforced silently rather than warned about.

*Large scales.* At :math:`s>N/4` fewer than four windows enter the average and
the moment is noise. ``scales`` therefore defaults to a geometric grid capped
at :math:`N/4`, and :func:`~mmar.core.validate.check_scales` drops anything
coarser.

*Negative* :math:`q`. :math:`F_q` with :math:`q<0` is a generalised mean tilted
towards the *smallest* window fluctuations, and those are exactly the windows
the polynomial happened to fit well. On a short record the small-:math:`F^2`
tail is undersampled and its deficit is scale-dependent, which inflates
:math:`\hat h(q)` for :math:`q<0` --- a spurious right branch of
:math:`f(\alpha)` even for monofractal input. The size of the effect is
measured, not asserted, in ``tests/unit/test_mfdfa.py``: on exact fBm with
:math:`N=16384` the mean bias over :math:`q\in[-5,-1]` is stated there as a
number. A window that the polynomial reproduces exactly --- a constant stretch,
a gap filled by interpolation, :math:`s\le m+1` --- has an algebraically zero
residual, so :math:`F^2` is pure round-off and every negative moment diverges.
Such windows are detected against an explicit round-off floor (see
:func:`_window_variances`), dropped, and counted in
``provenance.diagnostics["degenerate_windows"]``. Nothing is silently absorbed:
if fewer than four windows survive at some scale the estimator raises.

References
----------
Kantelhardt, J.W., Zschiegner, S.A., Koscielny-Bunde, E., Havlin, S., Bunde, A.
and Stanley, H.E. (2002). Multifractal detrended fluctuation analysis of
nonstationary time series. *Physica A* **316**, 87--114.

Kantelhardt, J.W., Koscielny-Bunde, E., Rego, H.H.A., Havlin, S. and Bunde, A.
(2001). Detecting long-range correlations with detrended fluctuation analysis.
*Physica A* **295**, 441--454.

Peng, C.-K., Buldyrev, S.V., Havlin, S., Simons, M., Stanley, H.E. and
Goldberger, A.L. (1994). Mosaic organization of DNA nucleotides.
*Phys. Rev. E* **49**, 1685--1689.

Ludescher, J., Bogachev, M.I., Kantelhardt, J.W., Schumann, A.Y. and Bunde, A.
(2011). On spurious and corrupted multifractality: the effects of additive
noise, short-term memory and periodic trends. *Physica A* **390**, 2480--2490.
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
from mmar.core.validate import (
    DataQualityError,
    check_scales,
    require_squarable,
    validate_q_grid,
    validate_series,
)

__all__ = [
    "dfa",
    "mfdfa",
    "mfdfa_from_structure",
    "mfdfa_structure",
]

ESTIMATOR = "mfdfa"
_VERSION = "1"

_Q_ZERO_TOL = 1e-6
r"""Below this :math:`|q|` the logarithmic branch is substituted.

The two branches differ by :math:`\tfrac{q}{8}\operatorname{Var}(\ln F^2) +
O(q^2)`, so at :math:`|q|<10^{-6}` the substitution is far below the estimation
error, while the power form has already lost most of its significant digits to
the :math:`1/q` exponent.
"""

_MIN_LENGTH = 256
"""Refuse shorter records. Below ~2000 points the admissible scale range
:math:`[10(m+1),\\,N/4]` spans under one decade and the fitted slope is not a
scaling exponent."""


# --------------------------------------------------------------------------- #
# Detrending
# --------------------------------------------------------------------------- #


def _detrend_basis(s: int, order: int) -> FloatArray:
    r"""Orthonormal basis :math:`Q\in\mathbb R^{s\times(m+1)}` of the polynomials.

    The least-squares residual operator is :math:`I-QQ^{\mathsf T}`, so
    detrending every window at a given scale is one matrix product rather than
    :math:`2N_s` calls to :func:`numpy.polyfit`. The Vandermonde matrix is built
    on :math:`t\in[-1,1]` and orthonormalised by QR: on :math:`t=1..s` the
    monomial basis has condition number :math:`\sim s^{m}` and DFA3 on
    :math:`s=1000` would lose eight digits before the QR ever sees it.
    """
    t = np.linspace(-1.0, 1.0, s)
    vander = np.vander(t, order + 1, increasing=True)
    q_basis, _ = np.linalg.qr(vander)
    return np.ascontiguousarray(q_basis, dtype=np.float64)


def _window_variances(y: FloatArray, s: int, order: int) -> tuple[FloatArray, FloatArray]:
    r"""The :math:`F^2(v,s)` of step 4 and its round-off floor, at one scale.

    Both sweeps of step 2 are taken. When :math:`s\mid N` the forward and
    backward tilings coincide *cell for cell*; the literature still speaks of
    :math:`2N_s` segments, but counting each window twice would double the
    regression weight of those scales for no gain in information, so the
    duplicate sweep is dropped and the honest count returned.

    The second return value is the level below which :math:`F^2` is not a
    fluctuation but the round-off of the projection itself. A window that the
    polynomial reproduces exactly (a constant stretch, a linear ramp under
    DFA1) has an algebraically zero residual, realised in float64 as
    :math:`\|r\|\sim s\,\varepsilon\,\|w\|`; the floor is the square of that.
    It sits some twenty orders of magnitude below a genuine
    :math:`F^2/\langle w^2\rangle`, so the separation is not delicate --- but
    without it a single flat stretch sends every :math:`q<0` moment to
    :math:`+\infty` and the right branch of :math:`f(\alpha)` with it.
    """
    n = y.size
    ns = n // s
    head = y[: ns * s].reshape(ns, s)
    windows = (
        head if ns * s == n else np.concatenate((head, y[n - ns * s :].reshape(ns, s)), axis=0)
    )
    basis = _detrend_basis(s, order)
    resid = windows - (windows @ basis) @ basis.T
    f2 = np.asarray(np.mean(resid * resid, axis=1), dtype=np.float64)
    energy = np.asarray(np.mean(windows * windows, axis=1), dtype=np.float64)
    floor = (float(s) * float(np.finfo(np.float64).eps)) ** 2 * energy
    return f2, np.asarray(floor, dtype=np.float64)


def _generalised_fluctuation(log_f2: FloatArray, q: FloatArray) -> FloatArray:
    r"""Step 5, evaluated in the log domain.

    For :math:`q\neq0`,
    :math:`\ln F_q = q^{-1}\big[\operatorname{logsumexp}(\tfrac q2\ln F^2)
    - \ln(2N_s)\big]`; the naive :math:`(\text{mean of } (F^2)^{q/2})^{1/q}`
    overflows at :math:`q=-5` whenever :math:`F^2\lesssim10^{-60}`, which a
    cascade measure at :math:`k_{\max}=14` comfortably reaches.
    """
    m = float(log_f2.size)
    out = np.empty(q.size, dtype=np.float64)
    for i, qi in enumerate(q):
        if abs(float(qi)) < _Q_ZERO_TOL:
            out[i] = math.exp(0.5 * float(np.mean(log_f2)))
        else:
            lse = float(logsumexp(0.5 * float(qi) * log_f2))
            out[i] = math.exp((lse - math.log(m)) / float(qi))
    return out


# --------------------------------------------------------------------------- #
# Scaling surface
# --------------------------------------------------------------------------- #


def _profile(x: FloatArray, integrate: bool) -> FloatArray:
    return np.asarray(np.cumsum(x - float(np.mean(x))), dtype=np.float64) if integrate else x


def _default_scales(n: int, order: int) -> NDArray[np.int64]:
    return log_spaced_scales(n, s_min=10 * (order + 1), n_scales=24, max_fraction=0.25)


def _resolve_scales(
    n: int, order: int, scales: Sequence[int] | NDArray[Any] | None
) -> NDArray[np.int64]:
    """Geometric default, then the two hard admissibility constraints."""
    if scales is None:
        # The default grid starts at s_min = 10(m+1), so `order` alone can push every
        # default scale past n/4. check_scales then reports "no scale in scales ...;
        # the series is too short", which names an argument the caller never passed
        # and prescribes a longer series when a lower order would do: at n=256 the
        # same series fits at orders 1-5 and only order >= 6 fails.
        raw = _default_scales(n, order)
        try:
            kept = check_scales(raw, n, min_windows=4, name="scales")
        except DataQualityError as exc:
            raise DataQualityError(
                f"the default MFDFA scale grid for order={order} starts at "
                f"s_min = 10(order+1) = {10 * (order + 1)}, and no scale that large leaves "
                f"4 windows in n={n} (the largest admissible scale is {n // 4}). No `scales` "
                f"were supplied, so this is a constraint on `order`, not on the data: lower "
                f"the order (any order <= {max(n // 40 - 1, 0)} fits this series), pass "
                f"`scales=` explicitly, or lengthen the series."
            ) from exc
    else:
        raw = np.asarray(scales, dtype=np.int64)
        kept = check_scales(raw, n, min_windows=4, name="scales")
    # s = m+1 makes the polynomial interpolate the window and F^2 vanish identically.
    admissible = kept[kept >= order + 2]
    if admissible.size < 3:
        raise DataQualityError(
            f"only {admissible.size} scales survive s >= order+2 = {order + 2} and "
            f"N/s >= 4 for N={n}; MFDFA cannot fit a slope. Lower the order or "
            "lengthen the series."
        )
    return np.ascontiguousarray(admissible)


def mfdfa_structure(
    x: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    order: int = 1,
    integrate: bool = True,
    seed: SeedLike = None,
) -> StructureFunctions:
    r"""The raw MFDFA surface :math:`F_q(s)`, with no regression performed.

    Separating the surface from the fit is what makes a scaling range
    revisable: :func:`mfdfa_from_structure` refits over any sub-window of
    ``scales`` without touching the :math:`O(N n_s)` detrending work again.

    Parameters
    ----------
    x
        Noise-like input (see the module warning about ``integrate``).
    q
        Moment grid; defaults to 41 points on :math:`[-5,5]`, negative
        :math:`q` included --- the right branch of :math:`f(\alpha)` exists
        only there.
    scales
        Window lengths :math:`s`. Defaults to 24 geometrically spaced integers
        on :math:`[10(m+1),\,N/4]`. Scales with fewer than four windows, or
        with :math:`s<m+2`, are dropped.
    order
        Polynomial order :math:`m` of the local detrending. ``1`` is DFA1 and
        removes a linear trend in the profile, i.e. a constant offset in
        ``x``.
    integrate
        Apply the cumulative-sum profile. See the module-level warning.
    seed
        Accepted for signature uniformity and recorded in provenance. MFDFA is
        deterministic and draws no randomness; a seed changes nothing but the
        audit record.

    Returns
    -------
    StructureFunctions
        ``values[i, j]`` :math:`= F_{q_i}(s_j)`, ``counts[j]`` the number of
        windows that entered scale :math:`s_j` *after* zero-variance windows
        were removed.
    """
    xa = validate_series(x, name="x", min_length=_MIN_LENGTH)
    qa = validate_q_grid(q)
    # int() truncates toward zero, so order=-0.5 used to slip past the guard below
    # as DFA0 and order=2.99 to run as DFA2 -- a silently different detrending than
    # the one asked for, with no trace anywhere but the provenance record.
    if isinstance(order, bool) or order != int(order):
        raise ValueError(
            f"order must be an integer polynomial degree, got {order!r}; "
            "int() would truncate it toward zero and detrend at a different order"
        )
    m = int(order)
    if m < 0:
        raise ValueError(f"order must be non-negative, got {order!r}")
    sa = _resolve_scales(xa.size, m, scales)

    y = _profile(xa, bool(integrate))
    # The window variance below squares the profile; a record beyond 1.3e154
    # overflows there behind a bare RuntimeWarning and carries inf into F_q.
    require_squarable(y, name="MFDFA profile")

    values = np.empty((qa.size, sa.size), dtype=np.float64)
    counts = np.empty(sa.size, dtype=np.int64)
    dropped = 0
    for j, s in enumerate(sa):
        f2, floor = _window_variances(y, int(s), m)
        good = f2 > floor
        n_bad = int(f2.size - good.sum())
        if int(good.sum()) < 4:
            raise DataQualityError(
                f"scale s={int(s)} leaves only {int(good.sum())} windows with a detrended "
                f"variance above the round-off floor ({n_bad} vanished). Negative moments "
                "are undefined there; the series has flat or exactly polynomial stretches."
            )
        dropped += n_bad
        values[:, j] = _generalised_fluctuation(np.log(f2[good]), qa)
        counts[j] = int(good.sum())

    prov = new_provenance(
        f"{ESTIMATOR}.structure",
        estimator_version=_VERSION,
        params={
            "order": m,
            "integrate": bool(integrate),
            "scales": [int(s) for s in sa],
        },
        inputs=(InputRef.from_array("x", xa), InputRef.from_array("q", qa)),
        seed=seed_info(seed) if seed is not None else None,
        notes=(
            "native exponent is h(q) (fluctuation of the profile); "
            "zeta(q) = q*h(q) and tau(q) = q*h(q) - d",
        ),
    ).with_diagnostics(degenerate_windows=dropped, n_samples=int(xa.size))

    return StructureFunctions(
        q=qa, scales=sa.astype(np.float64), values=values, counts=counts, provenance=prov
    )


# --------------------------------------------------------------------------- #
# Exponents
# --------------------------------------------------------------------------- #


def _reference_index(q: FloatArray) -> int:
    """Index of the :math:`q` closest to 2: the classical DFA moment."""
    return int(np.argmin(np.abs(q - 2.0)))


def _fit_all(
    structure: StructureFunctions,
    *,
    fit_method: FitMethod,
    select_range: bool,
) -> tuple[tuple[LogLogFit, ...], ScalingRange]:
    r"""One weighted log--log fit per :math:`q`, over a single common window.

    The scaling range is selected **once**, on :math:`q\approx2`, and imposed
    on every other moment. Selecting per :math:`q` would let each exponent come
    from a different band of scales, and :math:`\zeta` assembled from
    inconsistent windows has no Legendre transform worth taking.
    """
    s = np.asarray(structure.scales, dtype=np.float64)
    counts = structure.counts
    if counts is None:  # pragma: no cover - mfdfa_structure always sets counts
        raise ValueError("structure.counts is required for weighted MFDFA fits")
    w = weighted_by_counts(counts)
    ref = _reference_index(structure.q)

    mask = np.ones(s.size, dtype=bool)
    scaling_range: ScalingRange
    if select_range:
        sel = select_scaling_range(
            s, structure.values[ref], weights=w, method=fit_method, min_points=5
        )
        mask = sel.mask
        scaling_range = sel.scaling_range
    else:
        ref_fit = loglog_fit(s, structure.values[ref], method=fit_method, weights=w)
        scaling_range = ScalingRange(
            s_min=float(s.min()),
            s_max=float(s.max()),
            n_scales=int(s.size),
            selection="manual",
            r_squared=ref_fit.r_squared,
            mask=tuple(bool(b) for b in mask),
        )

    fits = tuple(
        loglog_fit(
            s[mask],
            structure.values[i][mask],
            method=fit_method,
            weights=w[mask],
            scaling_range=scaling_range,
        )
        for i in range(structure.q.size)
    )
    return fits, scaling_range


def mfdfa_from_structure(
    structure: StructureFunctions,
    *,
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
    d_support: float = 1.0,
) -> ScalingExponents:
    r"""Fit :math:`\zeta(q)=q\,h(q)` from a precomputed MFDFA surface.

    The uncertainty is propagated exactly, since the map is linear in the
    fitted slope: :math:`\operatorname{se}(\zeta)=|q|\operatorname{se}(h)` and
    the interval endpoints swap for :math:`q<0`. At :math:`q=0` the value
    :math:`\zeta(0)=0` is an algebraic identity, so its standard error is
    reported as exactly zero rather than as the (finite, irrelevant) error on
    :math:`h(0)`.

    .. warning::
        **The interval measures scatter about the fitted line, not accuracy.**
        ``zeta_stderr`` and ``zeta_ci`` come from the log--log regression and
        know nothing about the estimator's finite-scale bias, which is the
        dominant error term whenever the input is not noisy. Measured on the
        deterministic binomial cascade :math:`(0.6,0.4)`, :math:`k_{\max}=14`,
        dyadic scales :math:`2^5..2^{12}`: the median
        :math:`\operatorname{se}(\zeta)` is 0.0063 while the median
        :math:`|\hat\zeta-\zeta|` against the closed-form oracle is 0.0352, and
        the nominal 95% interval covers the true value at **1 of 41**
        :math:`q` values. On exact fGn, where the residual really is sampling
        noise, coverage is 35/41. Do not use these intervals to test a
        hypothesis about a near-deterministic measure; use the oracle
        comparison or a surrogate ensemble.
    """
    fits, scaling_range = _fit_all(structure, fit_method=fit_method, select_range=select_range)
    q = structure.q
    h = np.array([f.slope for f in fits], dtype=np.float64)
    h_se = np.array([f.slope_stderr for f in fits], dtype=np.float64)
    lo = np.array([f.slope_ci[0] for f in fits], dtype=np.float64)
    hi = np.array([f.slope_ci[1] for f in fits], dtype=np.float64)

    zeta = q * h
    zeta_se = np.abs(q) * h_se
    a, b = q * lo, q * hi
    zeta_ci = (np.minimum(a, b), np.maximum(a, b))

    prov: Provenance | None = None
    if structure.provenance is not None:
        prov = structure.provenance.derive(
            ESTIMATOR,
            estimator_version=_VERSION,
            params={
                "fit_method": str(fit_method),
                "select_range": bool(select_range),
                "d_support": float(d_support),
            },
            notes=("zeta(q) = q*h(q); tau(q) = zeta(q) - d_support",),
        )
        # `derive` starts a fresh diagnostics map, so without this the refit path
        # returns exponents whose audit record has lost the degenerate-window
        # counts the direct path reports -- and the reader of a refitted result
        # has no way to see that a scale was fitted through dead windows.
        # Diagnostics are non-identifying, so this cannot move `root`.
        prov = prov.with_diagnostics(
            **{
                **dict(structure.provenance.diagnostics),
                "scaling_range": scaling_range.to_canonical(),
            }
        )

    return ScalingExponents(
        q=q,
        zeta=zeta,
        d_support=float(d_support),
        zeta_stderr=zeta_se,
        zeta_ci=zeta_ci,
        fits=fits,
        scaling_range=scaling_range,
        structure=structure,
        estimator=ESTIMATOR,
        provenance=prov,
    )


def mfdfa(
    x: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    order: int = 1,
    integrate: bool = True,
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
    d_support: float = 1.0,
    seed: SeedLike = None,
) -> ScalingExponents:
    r"""Multifractal detrended fluctuation analysis.

    Computes the surface :math:`F_q(s)` (:func:`mfdfa_structure`), fits
    :math:`F_q(s)\sim s^{h(q)}` by count-weighted log--log regression, and
    returns the exponents in the library's carrier convention

    .. math::
        \zeta(q) = q\,h(q), \qquad \tau(q) = q\,h(q) - d .

    Parameters
    ----------
    x
        Noise-like series. **Read the module warning on ``integrate``**: a
        log-price is *not* noise-like and passing one with the default
        ``integrate=True`` returns :math:`h+1`.
    q, scales, order, integrate, seed
        As in :func:`mfdfa_structure`.
    fit_method
        Regression used for every :math:`\log F_q` vs :math:`\log s` fit.
        ``WLS`` by default, weighted by the number of windows: at
        :math:`s=N/4` a moment rests on four windows and at :math:`s=20` on
        :math:`N/10`, and an unweighted fit hands those two the same authority.
    select_range
        Search for the widest sub-window of ``scales`` over which the power
        law holds (on :math:`q\approx2`, then imposed on all :math:`q`).
        Off by default: the geometric grid is already restricted to the
        admissible band, and automatic selection on a short record tends to
        find whichever five scales are locally straightest.
    d_support
        Topological dimension of the support; 1 for a time series. Enters only
        through :math:`\tau=\zeta-d`.

    Returns
    -------
    ScalingExponents
        With ``zeta``, ``zeta_stderr``, ``zeta_ci``, one ``LogLogFit`` per
        :math:`q`, the ``scaling_range`` actually fitted, the full
        ``structure`` surface and a content-addressed ``provenance``.

    Notes
    -----
    Self-checks worth running on the result. :math:`\tau(0)=-d` holds by
    construction and is *not* evidence the estimate is sound. What is
    informative is :math:`h(0)`, which must sit inside the range of the other
    :math:`h(q)`; :math:`h(2)`, the classical DFA exponent (:func:`dfa`); and,
    when ``x`` is a conservative measure, :math:`h(1)\approx1`, equivalently
    :math:`\tau(1)\approx0`.

    Examples
    --------
    >>> import numpy as np
    >>> from mmar.models.fbm import fgn
    >>> from mmar.estimators.mfdfa import mfdfa
    >>> e = mfdfa(fgn(8192, 0.7, seed=0), q=np.linspace(-3, 3, 13))
    >>> bool(abs(e.h2 - 0.7) < 0.05)
    True
    """
    structure = mfdfa_structure(x, q=q, scales=scales, order=order, integrate=integrate, seed=seed)
    exponents = mfdfa_from_structure(
        structure, fit_method=fit_method, select_range=select_range, d_support=d_support
    )
    diagnostics: dict[str, Any] = (
        dict(structure.provenance.diagnostics) if structure.provenance is not None else {}
    )
    if exponents.scaling_range is not None:
        diagnostics["scaling_range"] = exponents.scaling_range.to_canonical()
    # A root derived from the surface would not bind the fit-stage parameters to
    # the input array directly; rebuild it so every argument of this call is in it.
    prov = new_provenance(
        ESTIMATOR,
        estimator_version=_VERSION,
        params={
            "order": int(order),
            "integrate": bool(integrate),
            "fit_method": str(fit_method),
            "select_range": bool(select_range),
            "d_support": float(d_support),
            "scales": None if scales is None else [int(s) for s in np.asarray(scales).ravel()],
            "scales_used": [int(s) for s in structure.scales],
        },
        inputs=(
            InputRef.from_array("x", np.asarray(x, dtype=np.float64).ravel()),
            InputRef.from_array("q", structure.q),
        ),
        seed=seed_info(seed) if seed is not None else None,
        notes=(
            "MFDFA (Kantelhardt et al. 2002); native exponent h(q), stored as zeta(q) = q*h(q)",
        ),
    ).with_diagnostics(**diagnostics)
    return ScalingExponents(
        q=exponents.q,
        zeta=exponents.zeta,
        d_support=exponents.d_support,
        zeta_stderr=exponents.zeta_stderr,
        zeta_ci=exponents.zeta_ci,
        fits=exponents.fits,
        scaling_range=exponents.scaling_range,
        structure=structure,
        estimator=ESTIMATOR,
        provenance=prov,
    )


def dfa(
    x: Sequence[float] | NDArray[Any],
    *,
    order: int = 1,
    scales: Sequence[int] | NDArray[Any] | None = None,
    integrate: bool = True,
    fit_method: FitMethod = FitMethod.WLS,
) -> LogLogFit:
    r"""Classical detrended fluctuation analysis: the :math:`q=2` special case.

    .. math::
        F_2(s)=\Big\{\frac{1}{2N_s}\sum_{v=1}^{2N_s}F^2(v,s)\Big\}^{1/2}
        \sim s^{\,h(2)} ,

    the root-mean-square detrended fluctuation of Peng et al. (1994). For a
    monofractal :math:`h(2)` *is* the Hurst exponent, and on exact fBm this
    function returns :math:`H`; for a multifractal it is only one point of
    :math:`h(q)` and says nothing about the width of the spectrum.

    Returns
    -------
    LogLogFit
        ``slope`` is :math:`h(2)`, with its standard error, :math:`t`-interval
        and :math:`R^2`. Multiply by 2 for :math:`\zeta(2)`.
    """
    grid = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    structure = mfdfa_structure(x, q=grid, scales=scales, order=order, integrate=integrate)
    counts = structure.counts
    if counts is None:  # pragma: no cover - mfdfa_structure always sets counts
        raise ValueError("structure.counts is required for a weighted DFA fit")
    return loglog_fit(
        np.asarray(structure.scales, dtype=np.float64),
        structure.values[1],
        method=fit_method,
        weights=weighted_by_counts(counts),
    )
