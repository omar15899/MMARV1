r"""Log-cumulant estimation: the best-conditioned summary of a multifractal.

The scaling function of a multifractal is analytic at the origin under mild
conditions, so it admits the *cumulant expansion*

.. math::
    \zeta(q) \;=\; \sum_{p\ge 1} \frac{c_p}{p!}\, q^{p},
    \qquad c_p = \frac{d^{p}\zeta}{dq^{p}}\Big|_{q=0},

which is the object this module estimates. The coefficients are not a
reparameterisation of convenience: they are the cumulants of the logarithm of
the increment, in the following exact sense. Write
:math:`L_\delta = \ln|\Delta_\delta X|` and let :math:`C_p(\delta)` be its
:math:`p`-th cumulant. Since
:math:`S(q,\delta)=\mathbb E\,e^{qL_\delta}` is precisely the moment generating
function of :math:`L_\delta`,

.. math::
    \ln S(q,\delta) \;=\; \sum_{p\ge1} \frac{C_p(\delta)}{p!}\,q^{p},

and comparing with :math:`\ln S(q,\delta) = \zeta(q)\ln\delta + \ln K_q` term by
term in :math:`q` gives the defining equation of the second route implemented
here,

.. math::
    \boxed{\;C_p(\delta) \;=\; c_p \ln\delta \;+\; \text{const}\;}

Interpretation, in the order in which the coefficients matter:

``c1``
    The most probable Hölder exponent, :math:`c_1 = \alpha_0`. For fBm it is
    :math:`H`; for a lognormal cascade it is :math:`\lambda`.
``c2``
    :math:`c_2 \le 0` by concavity of :math:`\zeta`, and :math:`c_2 = 0`
    *exactly* for a monofractal, because a monofractal has :math:`\zeta(q)=qH`
    affine. It sets the curvature of the spectrum at its apex,
    :math:`f(\alpha)\approx d + (\alpha-c_1)^2/(2c_2)`, so
    :math:`-c_2` is a single-number multifractality diagnostic
    (:func:`multifractality_index`). It is far more reliable than the width
    :math:`\Delta\alpha` of a Legendre-transformed curve, which is a
    *support* estimate: it depends on the :math:`q` window probed, is dominated
    by the noisiest tails of :math:`\hat\zeta`, and is bounded away from zero on
    any finite sample even under the monofractal null.
``c3``
    Asymmetry. A lognormal cascade has :math:`c_p = 0` for all :math:`p\ge3`, so
    a :math:`c_3` significantly different from zero is evidence *against* the
    lognormal model (its spectrum is exactly symmetric); log-Poisson cascades
    have :math:`c_3\neq0`.

Two independent routes are implemented, and they are meant to be run together.

1. :func:`cumulants_from_exponents` — polynomial regression of an *already
   estimated* :math:`\hat\zeta(q)` on :math:`q`. Estimator-agnostic: it consumes
   a :class:`~mmar.core.types.ScalingExponents` from anything.
2. :func:`cumulants_from_increments` (and :func:`cumulants_from_moments`, its
   structure-function-surface twin) — the direct route, which never forms
   :math:`\hat\zeta` at all: it estimates :math:`C_p(\delta)` at each scale and
   regresses on :math:`\ln\delta`.

They fail differently, which is the point. Route 1 inherits every pathology of
the moment estimates at large :math:`|q|` — divergent moments, saturation on
heavy tails — and the truncation bias of fitting a degree-:math:`P` polynomial
over a wide :math:`q` window to a function that is only asymptotically
polynomial near the origin. Route 2 touches no moment beyond the low-order
cumulants of :math:`\ln|\Delta X|`, so it is immune to those, but it is exposed
to small increments: :math:`\ln|\Delta X|` is unbounded below, so a single
near-zero increment (a stale quote, a rounded price) dominates :math:`C_2` and
:math:`C_3`. A discrepancy between the two routes larger than their combined
standard errors (:func:`compare_cumulants`) means at least one of those failure
modes is active, and no reported number should be trusted until it is resolved.

References
----------
Delour, J., Muzy, J.F. and Arneodo, A. (2001). Intermittency of 1D velocity
spatial profiles in turbulence: a magnitude cumulant analysis.
*Eur. Phys. J. B* **23**, 243--248.

Bacry, E., Delour, J. and Muzy, J.F. (2001). Multifractal random walk.
*Phys. Rev. E* **64**, 026103.

Muzy, J.F., Bacry, E. and Kozhemyak, A. (2006). Extreme values and fat tails of
multifractal fluctuations. *Phys. Rev. E* **73**, 066114.

Wendt, H., Abry, P. and Jaffard, S. (2007). Bootstrap for empirical multifractal
analysis. *IEEE Signal Processing Magazine* **24**(4), 38--48.

Kendall, M.G. and Stuart, A. (1969). *The Advanced Theory of Statistics*, Vol. 1,
ch. 12 (k-statistics).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray

from mmar.core.provenance import InputRef, new_provenance
from mmar.core.regression import loglog_fit
from mmar.core.types import (
    FitMethod,
    FloatArray,
    HolderSpectrum,
    LogCumulants,
    ScalingExponents,
    StructureFunctions,
)
from mmar.core.validate import check_scales, validate_series

__all__ = [
    "MAX_ORDER",
    "ZeroPolicy",
    "compare_cumulants",
    "cumulant_spectrum",
    "cumulants_from_exponents",
    "cumulants_from_increments",
    "cumulants_from_moments",
    "cumulants_from_series",
    "log_cumulants_by_scale",
    "multifractality_index",
]

MODULE_VERSION: Final = "1"

MAX_ORDER: Final = 4
"""Highest cumulant order supported.

Not an implementation limit but a statistical one: :math:`\\hat C_4` is a
fourth-order sample cumulant, its variance scales like :math:`\\kappa_8/n`, and
on the sample sizes multifractal analysis operates at it is pure noise. Orders
above three are exposed for diagnostics, never for reporting.
"""

ZeroPolicy = Literal["raise", "drop"]

_ZERO_POLICIES: Final = ("raise", "drop")


def _check_zero_policy(policy: str) -> ZeroPolicy:
    """Reject an unrecognised policy instead of defaulting to the lossy branch.

    The type annotation only binds callers that are type-checked; a policy read
    from a configuration file or a CLI flag is an arbitrary string. Without this
    check any value other than ``"raise"`` falls through to the dropping branch,
    which silently discards the most singular increments in the sample — the one
    behaviour this module documents as never being neutral.
    """
    if policy not in _ZERO_POLICIES:
        raise ValueError(f"zero_policy must be one of {_ZERO_POLICIES}, got {policy!r}")
    return "raise" if policy == "raise" else "drop"


_Q_ZERO_TOL: Final = 1e-12
"""Below this a grid point counts as :math:`q=0` and carries no information."""


# --------------------------------------------------------------------------- #
# Polynomial design and weighted least squares through the origin
# --------------------------------------------------------------------------- #


def _check_order(order: int) -> int:
    p = int(order)
    if p < 1 or p > MAX_ORDER:
        raise ValueError(f"order must lie in 1..{MAX_ORDER}, got {order!r}")
    return p


def _design(q: FloatArray, order: int) -> FloatArray:
    r"""Design matrix of the expansion, column :math:`p-1` being :math:`q^p/p!`.

    The :math:`1/p!` is carried in the design, not applied afterwards to the
    coefficients: putting it here means the fitted parameters *are* the
    :math:`c_p`, and the reported covariance is the covariance of the
    :math:`c_p` rather than of the raw polynomial coefficients. Getting this
    wrong rescales :math:`c_3` by a factor of six.
    """
    powers = np.arange(1, order + 1, dtype=np.float64)
    fact = np.asarray([math.factorial(p) for p in range(1, order + 1)], dtype=np.float64)
    return np.asarray(q[:, None] ** powers[None, :] / fact[None, :], dtype=np.float64)


def _wls_through_origin(
    x: FloatArray,
    y: FloatArray,
    order: int,
    weights: FloatArray,
    *,
    scale_covariance: bool,
) -> tuple[FloatArray, FloatArray, float, int]:
    r"""Weighted least squares of ``y`` on the cumulant design, no intercept.

    Returns ``(coef, cov, chi2, dof)``. The intercept is omitted because
    :math:`\zeta(0)=0` is an identity, not an estimate: in the signal convention
    :math:`S(0,\delta)=\langle|\Delta_\delta X|^0\rangle = 1` for every
    :math:`\delta`, so the scaling law forces :math:`\zeta(0)=0` exactly. Fitting
    a free intercept spends a degree of freedom estimating a known zero, and —
    worse — lets a constant offset in :math:`\hat\zeta` (the classic symptom of
    a mis-set convention, :math:`\tau` passed where :math:`\zeta` was expected)
    be absorbed silently instead of showing up as a bad fit.
    """
    n = x.size
    design = _design(x, order)
    root_w = np.sqrt(weights)
    a = design * root_w[:, None]
    b = y * root_w
    coef, _res, rank, _sv = np.linalg.lstsq(a, b, rcond=None)
    if rank < order:
        raise ValueError(
            f"the q grid does not identify {order} cumulants: the design matrix has rank "
            f"{rank} over {n} usable points. Use a wider or finer q grid, or a lower order."
        )
    resid = y - design @ coef
    chi2 = float(np.sum(weights * resid**2))
    dof = n - order
    try:
        cov_unit = np.linalg.inv(a.T @ a)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - rank check catches this first
        raise ValueError(f"singular normal equations for order {order}: {exc}") from exc
    if scale_covariance:
        cov = cov_unit * (chi2 / dof) if dof > 0 else np.full_like(cov_unit, np.nan)
    else:
        cov = cov_unit
    return (
        np.asarray(coef, dtype=np.float64),
        np.asarray(cov, dtype=np.float64),
        chi2,
        dof,
    )


# --------------------------------------------------------------------------- #
# Route 1: from estimated scaling exponents
# --------------------------------------------------------------------------- #


def cumulants_from_exponents(
    exponents: ScalingExponents,
    *,
    order: int = 3,
    q_range: tuple[float, float] | None = None,
    use_stderr: bool = True,
    scale_covariance: bool = True,
) -> LogCumulants:
    r"""Log-cumulants by polynomial regression of :math:`\hat\zeta(q)` on :math:`q`.

    Fits

    .. math::
        \hat\zeta(q_i) \;=\; \sum_{p=1}^{P} \frac{c_p}{p!} q_i^{\,p} + \varepsilon_i,
        \qquad
        \operatorname{Var}\varepsilon_i \propto \hat\sigma_i^{2},

    by weighted least squares **without an intercept**, the weights being
    :math:`w_i = 1/\hat\sigma_i^2` when ``exponents.zeta_stderr`` is available and
    uniform otherwise.

    Why no intercept
    ----------------
    :math:`\zeta(0)=0` identically in the signal convention (see
    :func:`_wls_through_origin`). The constraint is imposed by construction
    rather than fitted, which removes one parameter and makes a convention error
    visible: if the caller passes :math:`\tau=\zeta-d` by mistake, the residual
    at small :math:`|q|` is :math:`-d` and the fit is conspicuously bad instead
    of quietly returning cumulants that are wrong by an offset. Grid points at
    :math:`q=0` carry no information under this parameterisation (their design
    row is identically zero) and are dropped rather than allowed to inflate the
    degrees of freedom.

    Parameters
    ----------
    exponents
        Any estimator's output, or a closed-form oracle from
        :mod:`mmar.models.theory`. Only ``q``, ``zeta``, ``zeta_stderr`` and
        ``d_support`` are read.
    order
        Truncation order :math:`P` of the expansion, at most :attr:`MAX_ORDER`.
    q_range
        Restrict the fit to :math:`q\in[q_{\min},q_{\max}]`. **This is the main
        bias/variance knob.** The expansion is a Taylor series about
        :math:`q=0`: fitting a cubic over :math:`|q|\le 5` to a
        :math:`\zeta` that is not a cubic aliases the neglected orders into
        :math:`c_2` and :math:`c_3`, while a narrow window inflates their
        variance. A window of :math:`|q|\lesssim 2`--:math:`3` is the usual
        compromise for financial data.
    use_stderr
        Use ``zeta_stderr`` for the weights. Set ``False`` to fall back to OLS
        when the reported standard errors are known to be unreliable.
    scale_covariance
        Multiply the covariance by the reduced :math:`\chi^2`. This is the right
        default whenever the weights are only correct up to a common factor
        (they usually are, since :math:`\hat\zeta` errors are correlated across
        :math:`q` and ``zeta_stderr`` ignores that correlation). Set ``False``
        to take the supplied variances at face value.

    Returns
    -------
    LogCumulants
        With ``stderr`` the square roots of the diagonal of the regression
        covariance, and provenance recording the fitted window, the reduced
        :math:`\chi^2` and the weighting mode.

    Failure modes
    -------------
    * *Correlated residuals.* :math:`\hat\zeta(q_i)` at neighbouring :math:`q`
      come from the same sample and are strongly correlated, so the reported
      standard errors are optimistic even after the :math:`\chi^2` rescaling.
      For inference use :mod:`mmar.inference.bootstrap`, which resamples the
      data rather than the exponents.
    * *Saturating* :math:`\hat\zeta`. On heavy-tailed data
      :math:`\hat\zeta(q)\to\text{const}` for :math:`q` beyond the tail index;
      the resulting spurious concavity is reported here as a large :math:`-c_2`.
      This is the single most common false positive in the literature, and
      restricting ``q_range`` is not a fix — check the moments exist.

    Raises
    ------
    ValueError
        If fewer than ``order`` non-zero :math:`q` points survive, if the
        exponents contain non-finite values, or if ``use_stderr`` is requested
        with non-positive or non-finite standard errors (silently dropping them
        would turn a broken fit into a plausible number).
    """
    p = _check_order(order)
    q = np.asarray(exponents.q, dtype=np.float64)
    zeta = np.asarray(exponents.zeta, dtype=np.float64)
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(zeta)):
        raise ValueError("the exponents contain non-finite values; nothing can be fitted to them")

    keep = np.abs(q) > _Q_ZERO_TOL
    if q_range is not None:
        lo, hi = float(q_range[0]), float(q_range[1])
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            raise ValueError(f"q_range must be a finite increasing interval, got {q_range!r}")
        keep &= (q >= lo) & (q <= hi)
    n_used = int(keep.sum())
    if n_used < p:
        raise ValueError(
            f"only {n_used} usable q values (non-zero, inside q_range={q_range!r}) for an "
            f"order-{p} fit; at least {p} are needed and {p + 2} are advisable"
        )

    qu, zu = q[keep], zeta[keep]
    se = exponents.zeta_stderr
    weighted = use_stderr and se is not None
    if weighted:
        sigma = np.asarray(se, dtype=np.float64)[keep]
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError(
                f"{int(np.sum(~np.isfinite(sigma)) + np.sum(sigma <= 0.0))} of {n_used} "
                "zeta_stderr entries are non-positive or non-finite, so 1/sigma^2 weights are "
                "undefined. Pass use_stderr=False to fit unweighted, explicitly."
            )
        w = 1.0 / sigma**2
    else:
        w = np.ones(n_used, dtype=np.float64)

    coef, cov, chi2, dof = _wls_through_origin(qu, zu, p, w, scale_covariance=scale_covariance)
    stderr = np.sqrt(np.diag(cov))

    parent = exponents.provenance
    inputs = (
        InputRef.from_array("q", qu),
        InputRef.from_array("zeta", zu),
    )
    params: dict[str, Any] = {
        "order": p,
        "q_range": None if q_range is None else [float(q_range[0]), float(q_range[1])],
        "weighting": "inverse_variance" if weighted else "uniform",
        "scale_covariance": bool(scale_covariance),
        "estimator_source": exponents.estimator,
        "d_support": float(exponents.d_support),
    }
    if parent is not None:
        prov = parent.derive(
            "inference.cumulants_from_exponents",
            estimator_version=MODULE_VERSION,
            params=params,
            extra_inputs=inputs,
        )
    else:
        prov = new_provenance(
            "inference.cumulants_from_exponents",
            estimator_version=MODULE_VERSION,
            params=params,
            inputs=inputs,
            seed=None,
            notes=("zeta(0) = 0 imposed: fitted without an intercept",),
        )
    prov = prov.with_diagnostics(
        n_points=n_used,
        dof=dof,
        chi2=chi2,
        chi2_reduced=chi2 / dof if dof > 0 else float("nan"),
        q_min=float(qu.min()),
        q_max=float(qu.max()),
        route="exponents",
    )
    return LogCumulants(
        c=coef,
        stderr=np.asarray(stderr, dtype=np.float64),
        order=p,
        d_support=float(exponents.d_support),
        provenance=prov,
    )


def cumulants_from_series(
    x: Sequence[float] | NDArray[Any],
    estimator: Callable[[FloatArray], ScalingExponents],
    *,
    order: int = 3,
    q_range: tuple[float, float] | None = None,
) -> LogCumulants:
    r"""Convenience composition: estimate :math:`\hat\zeta` then its cumulants.

    ``estimator`` is any callable mapping a series to
    :class:`~mmar.core.types.ScalingExponents` — MFDFA, WTMM, wavelet leaders, a
    partial application of the structure-function estimator, or a closed-form
    oracle. Taking it as a parameter rather than importing one keeps this module
    independent of which estimator is in use, and makes the choice visible in
    the caller's code and in the provenance chain.
    """
    xa = validate_series(x, name="series")
    return cumulants_from_exponents(estimator(xa), order=order, q_range=q_range)


# --------------------------------------------------------------------------- #
# Route 2: directly from the log-increments
# --------------------------------------------------------------------------- #


def _sample_cumulants(v: FloatArray, order: int) -> FloatArray:
    r"""Unbiased sample cumulants (k-statistics) :math:`k_1,\dots,k_{\text{order}}`.

    With :math:`m_r = n^{-1}\sum_i (v_i-\bar v)^r` the plug-in central moments,

    .. math::
        k_1 = \bar v, \quad
        k_2 = \frac{n}{n-1} m_2, \quad
        k_3 = \frac{n^2}{(n-1)(n-2)} m_3,

    .. math::
        k_4 = \frac{n^2\big[(n+1)m_4 - 3(n-1)m_2^2\big]}{(n-1)(n-2)(n-3)} .

    **Why unbiasedness matters here specifically.** The plug-in estimators are
    biased by factors :math:`1-O(1/n)`. The number of increments at scale
    :math:`\delta` is :math:`n_\delta \propto 1/\delta`, so that bias is a
    *systematic function of the scale*: it adds a spurious contribution to the
    regression of :math:`C_p(\delta)` on :math:`\ln\delta`, i.e. directly to
    :math:`\hat c_p`, and it does so in the direction that manufactures
    multifractality (:math:`m_2` is deflated more at coarse scales, tilting
    :math:`\hat c_2` negative). The k-statistics remove it exactly under
    independence. Under dependence — which is the realistic case — a residual
    bias survives, and it is the dominant part of the detection floor measured
    in the test suite (Kendall & Stuart 1969, ch. 12).
    """
    n = float(v.size)
    mean = float(v.mean())
    dev = v - mean
    out = np.empty(order, dtype=np.float64)
    out[0] = mean
    if order >= 2:
        m2 = float(np.mean(dev**2))
        out[1] = n / (n - 1.0) * m2
    if order >= 3:
        m3 = float(np.mean(dev**3))
        out[2] = n * n / ((n - 1.0) * (n - 2.0)) * m3
    if order >= 4:
        m2 = float(np.mean(dev**2))
        m4 = float(np.mean(dev**4))
        out[3] = (
            n
            * n
            * ((n + 1.0) * m4 - 3.0 * (n - 1.0) * m2 * m2)
            / ((n - 1.0) * (n - 2.0) * (n - 3.0))
        )
    return out


def _increments(x: FloatArray, s: int, *, overlapping: bool) -> FloatArray:
    r""":math:`\Delta_s X(t) = X(t+s) - X(t)`, overlapping or disjoint.

    The disjoint branch indexes explicitly rather than slicing with a stride on
    both operands: ``x[s::s] - x[:-s:s]`` silently truncates to the shorter
    operand when :math:`s \nmid n`, so the increment count is not
    :math:`\lfloor (n-1)/s \rfloor` and varies non-monotonically with :math:`s`
    — a defect that puts a scale-dependent wobble into every log--log fit
    (LEGACY_AUDIT D6).
    """
    if overlapping:
        return np.asarray(x[s:] - x[:-s], dtype=np.float64)
    k = (x.size - 1) // s
    i = np.arange(k, dtype=np.int64) * s
    return np.asarray(x[i + s] - x[i], dtype=np.float64)


def log_cumulants_by_scale(
    x: Sequence[float] | NDArray[Any],
    *,
    scales: Sequence[int] | NDArray[Any],
    order: int = 3,
    overlapping: bool = True,
    zero_policy: ZeroPolicy = "raise",
    min_windows: int = 4,
) -> tuple[NDArray[np.int64], FloatArray, NDArray[np.int64], int]:
    r"""The scale-resolved cumulants :math:`C_p(\delta)` of :math:`\ln|\Delta_\delta X|`.

    This is the raw material of :func:`cumulants_from_increments`, exposed
    separately because the *shape* of :math:`C_p(\delta)` against
    :math:`\ln\delta` is the diagnostic: the model says it is a straight line,
    and a visible bend means the scaling range is wrong, not that the cumulant
    is interesting.

    Returns
    -------
    (scales_used, C, counts, n_zero_dropped)
        ``C`` has shape ``(order, len(scales_used))`` with ``C[p-1, j]`` the
        :math:`p`-th cumulant at scale ``scales_used[j]``; ``counts[j]`` is the
        number of *disjoint* windows at that scale, which is the effective
        sample size for weighting even when overlapping increments are used;
        ``n_zero_dropped`` is the total number of vanishing increments removed
        under ``zero_policy="drop"`` (always zero otherwise), returned so that
        the caller can record it — a silent drop is not auditable.

    Parameters
    ----------
    overlapping
        ``True`` (default) uses all :math:`n-s` increments, which reduces the
        variance of :math:`\hat C_p` but leaves them correlated; the weights are
        computed from the disjoint count regardless, so the regression is not
        told it has more information than it does.
    zero_policy
        What to do with exactly-zero increments, for which
        :math:`\ln|\Delta X| = -\infty`. ``"raise"`` (default) refuses;
        ``"drop"`` removes them and records how many. Dropping is not neutral:
        zero increments are the most singular events in the sample (a flat
        stretch is :math:`\alpha=+\infty`), so removing them truncates the right
        tail of the log-increment distribution and biases :math:`\hat c_1` down
        and :math:`\hat c_2` up (towards monofractality). Rounded prices and
        stale quotes produce them in quantity.
    """
    p = _check_order(order)
    policy = _check_zero_policy(zero_policy)
    xa = validate_series(x, name="series")
    sc = check_scales(scales, xa.size, min_windows=max(min_windows, p + 2))
    if sc.size < 3:
        raise ValueError(f"need at least three scales for a regression, got {sc.size}")

    kept: list[int] = []
    counts: list[int] = []
    columns: list[FloatArray] = []
    dropped_zeros = 0
    for s in sc.tolist():
        d = np.abs(_increments(xa, int(s), overlapping=overlapping))
        n_zero = int(np.count_nonzero(d == 0.0))
        if n_zero:
            if policy == "raise":
                raise ValueError(
                    f"{n_zero}/{d.size} increments vanish exactly at scale {s}, so "
                    "log|dX| = -inf there. Pass zero_policy='drop' to remove them and accept "
                    "the documented bias, or de-duplicate the series first."
                )
            dropped_zeros += n_zero
            d = d[d > 0.0]
        if d.size < p + 2:
            continue
        columns.append(_sample_cumulants(np.log(d), p))
        kept.append(int(s))
        counts.append(max((xa.size - 1) // int(s), 1))

    if len(kept) < 3:
        raise ValueError(
            f"only {len(kept)} scales survived (need 3); the series is too short or too "
            "degenerate for a scale-resolved cumulant regression"
        )
    c = np.asarray(columns, dtype=np.float64).T
    return (
        np.asarray(kept, dtype=np.int64),
        np.ascontiguousarray(c),
        np.asarray(counts, dtype=np.int64),
        dropped_zeros,
    )


def cumulants_from_increments(
    x: Sequence[float] | NDArray[Any],
    *,
    scales: Sequence[int] | NDArray[Any],
    order: int = 3,
    overlapping: bool = True,
    zero_policy: ZeroPolicy = "raise",
    method: FitMethod = FitMethod.WLS,
    d_support: float = 1.0,
) -> LogCumulants:
    r"""Log-cumulants by the direct route: regress :math:`C_p(\delta)` on :math:`\ln\delta`.

    Implements the defining equation of the module,

    .. math::
        C_p(\delta) \;=\; c_p \ln\delta \;+\; \text{const},

    one ordinary regression per order :math:`p`, weighted by the number of
    disjoint windows at each scale (the variance of a sample cumulant is
    :math:`O(1/n_\delta)`, so inverse-variance weighting is weighting by the
    count).

    This never forms :math:`\hat\zeta(q)`, never raises anything to a power
    :math:`q`, and therefore never touches a moment that might not exist. On
    heavy-tailed data — where the structure-function route reports the saturation
    :math:`\hat\zeta(q)=\min(q/\alpha,1)` as strong multifractality — this route
    is the one that stays honest, because :math:`\mathbb E|\ln|X||^p` is finite
    for an :math:`\alpha`-stable :math:`X` at every :math:`p`.

    Parameters
    ----------
    x
        The *signal*, not its increments. Pass a log-price path, or the
        cumulative measure (trading time) of a cascade; the increments are
        formed here.
    scales
        Integer lags in samples. Physical units are irrelevant to the slope:
        rescaling :math:`\delta\mapsto a\delta` shifts every :math:`C_p` by
        :math:`c_p\ln a`, i.e. moves the intercept and leaves :math:`c_p` fixed.
    method
        ``WLS`` (default, weighted by window counts), ``OLS``, or the robust
        ``THEILSEN`` / ``HUBER`` when one scale is contaminated.
    d_support
        Topological dimension of the support, carried into the returned object
        for the spectrum reconstruction; it does not enter the fit.

    Returns
    -------
    LogCumulants
        ``stderr`` are the slope standard errors of the per-order regressions,
        computed in log space. Per-order :math:`R^2` is attached to the
        provenance diagnostics: a low :math:`R^2` for :math:`p=2` while
        :math:`p=1` is clean means the scaling range supports a Hurst exponent
        but not a multifractality claim.

    Failure modes
    -------------
    * *Small increments dominate.* :math:`\ln|\Delta X|` has no lower bound;
      one increment :math:`10^{-12}` times the typical size shifts
      :math:`C_1` and inflates :math:`C_2`. Check the left tail before
      believing a large :math:`-c_2`.
    * *Correlated increments.* Overlapping windows and long memory both bias
      the higher sample cumulants in a scale-dependent way; that residual is
      the empirical detection floor of :math:`c_2`, measured against exact fBm
      in the test suite rather than assumed to be zero.
    * *Too few coarse-scale windows.* ``check_scales`` enforces a floor of four
      disjoint windows, which is generous; with fewer than ~30 the sample
      cumulants of order 3 are noise.
    """
    p = _check_order(order)
    policy = _check_zero_policy(zero_policy)
    xa = validate_series(x, name="series")
    sc, c_scale, counts, n_zero_dropped = log_cumulants_by_scale(
        xa,
        scales=scales,
        order=p,
        overlapping=overlapping,
        zero_policy=policy,
    )

    scales_f = sc.astype(np.float64)
    weights = counts.astype(np.float64)
    coef = np.empty(p, dtype=np.float64)
    stderr = np.empty(p, dtype=np.float64)
    r2 = np.empty(p, dtype=np.float64)
    for i in range(p):
        fit = loglog_fit(
            scales_f,
            c_scale[i],
            method=method,
            weights=weights if method is FitMethod.WLS else None,
            log_y=False,
        )
        coef[i] = fit.slope
        stderr[i] = fit.slope_stderr
        r2[i] = fit.r_squared

    prov = new_provenance(
        "inference.cumulants_from_increments",
        estimator_version=MODULE_VERSION,
        params={
            "order": p,
            "overlapping": bool(overlapping),
            "zero_policy": policy,
            "method": str(method),
            "scales": [int(s) for s in sc],
            "d_support": float(d_support),
        },
        inputs=(InputRef.from_array("x", xa), InputRef.from_array("scales", sc)),
        seed=None,
        notes=("C_p(delta) = c_p * ln(delta) + const; cumulants of ln|increment|",),
    ).with_diagnostics(
        n_samples=int(xa.size),
        n_scales=int(sc.size),
        window_counts=[int(n) for n in counts],
        zero_increments_dropped=n_zero_dropped,
        fit_r_squared=[float(v) for v in r2],
        route="increments",
    )
    return LogCumulants(
        c=coef,
        stderr=stderr,
        order=p,
        d_support=float(d_support),
        provenance=prov,
    )


def cumulants_from_moments(
    structure: StructureFunctions,
    *,
    order: int = 3,
    q_range: tuple[float, float] | None = None,
    method: FitMethod = FitMethod.WLS,
    d_support: float = 1.0,
) -> LogCumulants:
    r"""Log-cumulants from a structure-function *surface*, without fitting :math:`\zeta`.

    Uses the identity that makes :math:`\ln S(\cdot,\delta)` the cumulant
    generating function of :math:`\ln|\Delta_\delta X|`:

    .. math::
        \ln S(q,\delta) \;=\; \sum_{p\ge1}\frac{C_p(\delta)}{p!}q^{p},

    so a polynomial fit in :math:`q` **at each fixed scale** recovers
    :math:`C_p(\delta)`, and the slopes of those against :math:`\ln\delta` are
    the :math:`c_p`. This is the same estimator as
    :func:`cumulants_from_increments` seen through the moment surface: it is
    available when only :math:`S(q,\delta)` was retained, and it agrees with the
    direct route in the limit of a fine :math:`q` grid narrow about the origin.

    The fit at each scale is through the origin, because
    :math:`\ln S(0,\delta)=\ln 1 = 0`. When the grid contains :math:`q=0` that
    column is subtracted first, which also normalises away any common
    :math:`q`-independent factor in the surface.

    Parameters
    ----------
    structure
        A **moment** surface, :math:`S(q,\delta)=\langle|\Delta_\delta X|^q\rangle`.
    q_range
        As in :func:`cumulants_from_exponents`, and more important here: the
        polynomial in :math:`q` is fitted at each scale, so a wide grid aliases
        high-order cumulants into low-order ones scale by scale.

    Raises
    ------
    ValueError
        If the surface is not a moment surface. The check is
        :math:`S(0,\delta)=1`, which holds by definition for
        :math:`\langle|\Delta X|^q\rangle` and fails for the fluctuation
        surfaces of MFDFA/MFDMA, whose log--log slope is :math:`h(q)=\zeta(q)/q`
        rather than :math:`\zeta(q)`. Applying this routine to one of those
        silently returns the cumulants of :math:`h`, which are not the
        :math:`c_p` and differ from them by a factor that depends on the order.
        Use :func:`cumulants_from_exponents` on the estimator's
        :class:`~mmar.core.types.ScalingExponents` instead.
    """
    p = _check_order(order)
    if structure.log_base != math.e:
        raise ValueError(
            f"structure.log_base = {structure.log_base!r}; this routine expects `values` to hold "
            "the raw moments S(q, delta), not their logarithm in another base"
        )
    q = np.asarray(structure.q, dtype=np.float64)
    scales = np.asarray(structure.scales, dtype=np.float64)
    values = np.asarray(structure.values, dtype=np.float64)
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        bad = int(np.sum((values <= 0.0) | ~np.isfinite(values)))
        raise ValueError(f"{bad} non-positive or non-finite entries in the moment surface")

    log_s = np.log(values)
    zero = np.abs(q) <= _Q_ZERO_TOL
    if zero.any():
        base = log_s[int(np.argmax(zero))]
        if float(np.max(np.abs(base))) > 1e-6:
            raise ValueError(
                "S(0, delta) is not 1 (max |log S(0, delta)| = "
                f"{float(np.max(np.abs(base))):.3e}), so this is not a moment surface "
                "<|dX|^q>. MFDFA/MFDMA fluctuation surfaces scale as delta^h(q), not "
                "delta^zeta(q); use cumulants_from_exponents on their ScalingExponents."
            )
        log_s = log_s - base[None, :]

    keep = ~zero
    if q_range is not None:
        lo, hi = float(q_range[0]), float(q_range[1])
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            raise ValueError(f"q_range must be a finite increasing interval, got {q_range!r}")
        keep &= (q >= lo) & (q <= hi)
    if int(keep.sum()) < p + 1:
        raise ValueError(
            f"only {int(keep.sum())} usable q values for an order-{p} fit at each scale"
        )

    # One least-squares solve for the whole surface: the design depends on q
    # only, so every scale shares it and the (n_q, n_scales) response is solved
    # in a single call rather than scale by scale.
    design = _design(q[keep], p)
    c_scale, _res, rank, _sv = np.linalg.lstsq(design, log_s[keep, :], rcond=None)
    if rank < p:
        raise ValueError(f"the q grid does not identify {p} cumulants (design rank {rank})")

    weights = None
    if structure.counts is not None and method is FitMethod.WLS:
        weights = np.asarray(structure.counts, dtype=np.float64)

    coef = np.empty(p, dtype=np.float64)
    stderr = np.empty(p, dtype=np.float64)
    r2 = np.empty(p, dtype=np.float64)
    for i in range(p):
        fit = loglog_fit(scales, c_scale[i], method=method, weights=weights, log_y=False)
        coef[i] = fit.slope
        stderr[i] = fit.slope_stderr
        r2[i] = fit.r_squared

    params: dict[str, Any] = {
        "order": p,
        "q_range": None if q_range is None else [float(q_range[0]), float(q_range[1])],
        "method": str(method),
        "d_support": float(d_support),
    }
    inputs = (InputRef.from_array("q", q[keep]), InputRef.from_array("scales", scales))
    parent = structure.provenance
    prov = (
        parent.derive(
            "inference.cumulants_from_moments",
            estimator_version=MODULE_VERSION,
            params=params,
            extra_inputs=inputs,
        )
        if parent is not None
        else new_provenance(
            "inference.cumulants_from_moments",
            estimator_version=MODULE_VERSION,
            params=params,
            inputs=inputs,
            seed=None,
            notes=("log S(q, delta) is the CGF of log|increment| at scale delta",),
        )
    )
    prov = prov.with_diagnostics(
        n_q=int(keep.sum()),
        n_scales=int(scales.size),
        fit_r_squared=[float(v) for v in r2],
        route="moments",
    )
    return LogCumulants(
        c=np.asarray(coef, dtype=np.float64),
        stderr=stderr,
        order=p,
        d_support=float(d_support),
        provenance=prov,
    )


# --------------------------------------------------------------------------- #
# Reconstruction and summaries
# --------------------------------------------------------------------------- #


def cumulant_spectrum(
    cumulants: LogCumulants,
    alpha: FloatArray | Sequence[float] | None = None,
    *,
    order: int | None = None,
    n_alpha: int = 201,
    span: float = 1.5,
) -> HolderSpectrum:
    r"""The singularity spectrum implied by the cumulant expansion.

    Legendre-transforming the truncated expansion and inverting
    :math:`\alpha=\zeta'(q)` perturbatively in :math:`u=\alpha-c_1` gives, to
    third order,

    .. math::
        \boxed{\;f(\alpha) \;=\; d \;+\; \frac{(\alpha-c_1)^2}{2c_2}
               \;-\; \frac{c_3\,(\alpha-c_1)^3}{6\,c_2^{3}}
               \;+\; O\big((\alpha-c_1)^4\big)\;}

    *Derivation.* With :math:`\zeta(q)=c_1q+\tfrac12c_2q^2+\tfrac16c_3q^3` one
    has :math:`\alpha(q)=c_1+c_2q+\tfrac12c_3q^2`, so
    :math:`q = u/c_2 - c_3u^2/(2c_2^3)+O(u^3)`. Substituting into
    :math:`f = q\alpha-\zeta(q)+d = qu-\tfrac12c_2q^2-\tfrac16c_3q^3+d`, the
    :math:`u^3` terms from :math:`qu` and from :math:`-\tfrac12 c_2q^2` cancel
    exactly, leaving :math:`-c_3u^3/(6c_2^3)`. At order 2 this reduces to the
    parabola :math:`d+(\alpha-c_1)^2/(2c_2)`, i.e.
    :meth:`~mmar.core.types.LogCumulants.parabolic_spectrum`, which is the exact
    spectrum of a lognormal cascade.

    The curve is *local*: it is an expansion about the apex, and its accuracy
    degrades as :math:`|\alpha-c_1|` grows. Overlay it on the Legendre spectrum
    to see where the two part company — that is where the quadratic (or cubic)
    approximation stops describing the data, and it is usually well inside the
    nominal support.

    Parameters
    ----------
    alpha
        Grid to evaluate on. ``None`` spans
        :math:`c_1 \pm \text{span}\sqrt{-2c_2}`, i.e. ``span`` times the
        half-width at which the second-order spectrum reaches zero.
    order
        Expansion order used for the reconstruction, at most 3 and at most
        ``cumulants.order``. ``None`` takes the largest admissible value.
    span
        Half-width of the default grid, in units of :math:`\sqrt{-2c_2}`.

    Returns
    -------
    HolderSpectrum
        With ``method`` recording the order actually used. If
        :math:`c_2 \ge 0` — an estimate inconsistent with a concave
        :math:`\zeta` — every value is NaN rather than a flipped parabola,
        matching :meth:`~mmar.core.types.LogCumulants.parabolic_spectrum`.
    """
    p = cumulants.order if order is None else int(order)
    p = min(p, 3, int(cumulants.c.size))
    if p < 2:
        raise ValueError(
            "a spectrum needs at least the second cumulant: with c2 absent the expansion "
            "is affine and its Legendre transform is the degenerate monofractal point"
        )
    c1, c2 = cumulants.c1, cumulants.c2
    c3 = cumulants.c3 if p >= 3 else 0.0
    d = float(cumulants.d_support)

    if alpha is None:
        if c2 >= 0.0 or not math.isfinite(c2):
            raise ValueError(
                f"c2 = {c2!r} is non-negative, so no natural alpha grid exists (the implied "
                "spectrum is not concave). Pass an explicit alpha grid to inspect it anyway."
            )
        half = span * math.sqrt(-2.0 * c2)
        a = np.linspace(c1 - half, c1 + half, int(n_alpha))
    else:
        a = np.asarray(alpha, dtype=np.float64).ravel()
        if a.size == 0:
            raise ValueError("the alpha grid is empty")

    if c2 >= 0.0 or not math.isfinite(c2):
        f = np.full_like(a, np.nan)
    else:
        u = a - c1
        f = d + u**2 / (2.0 * c2)
        if p >= 3 and c3 != 0.0:
            f = f - c3 * u**3 / (6.0 * c2**3)

    parent = cumulants.provenance
    params = {"order": p, "n_alpha": int(a.size), "span": float(span)}
    prov = (
        parent.derive(
            "inference.cumulant_spectrum",
            estimator_version=MODULE_VERSION,
            params=params,
            extra_inputs=(InputRef.from_array("alpha", a),),
        )
        if parent is not None
        else new_provenance(
            "inference.cumulant_spectrum",
            estimator_version=MODULE_VERSION,
            params=params,
            inputs=(InputRef.from_array("alpha", a),),
            seed=None,
        )
    )
    return HolderSpectrum(
        alpha=a,
        f_alpha=np.asarray(f, dtype=np.float64),
        d_support=d,
        method=f"cumulant-order{p}",
        provenance=prov,
    )


def multifractality_index(cumulants: LogCumulants) -> float:
    r"""The intermittency coefficient :math:`-c_2 \ge 0`.

    A single number for "how multifractal", with a genuine null: it is
    **exactly zero** for a monofractal, because :math:`\zeta(q)=qH` is affine.
    Contrast the spectrum width :math:`\Delta\alpha`, which is positive on any
    finite sample of anything.

    Scale of the answer (all in the natural-logarithm convention used
    throughout, where the lognormal cascade has :math:`-c_2=\sigma^2\ln b`):

    ==================================  ===================
    Object                              :math:`-c_2`
    ==================================  ===================
    Monofractal (fBm, Brownian motion)  :math:`0` exactly
    Fully developed turbulence          :math:`\approx 0.03`
    Equity index / FX returns           :math:`\approx 0.03`--:math:`0.08`
    Strongly intermittent cascade       :math:`\gtrsim 0.2`
    ==================================  ===================

    The turbulence value is the classical intermittency exponent
    :math:`\mu\approx0.025`--:math:`0.03` (Kolmogorov 1962; Delour, Muzy &
    Arneodo 2001). The financial range is the multifractal-random-walk
    intermittency :math:`\lambda^2` of Bacry, Delour & Muzy (2001), who report
    :math:`\lambda^2\approx0.03` for the S&P 500 and comparable values across
    equity indices and exchange rates. Anything above :math:`0.1` from market
    data should be treated as a diagnosis of the estimator or of the tails, not
    of the market: heavy tails alone produce a spurious :math:`-c_2` of that
    order through the saturation of :math:`\hat\zeta`.

    A **negative** return value (i.e. :math:`c_2>0`) is not a small
    multifractality; it is an inconsistent fit, since it implies a convex
    :math:`\zeta` and a spectrum that opens upwards. Report it as a failure.
    """
    return -cumulants.c2


def compare_cumulants(
    a: LogCumulants,
    b: LogCumulants,
    *,
    labels: tuple[str, str] = ("a", "b"),
) -> dict[str, Any]:
    r"""Cross-check two cumulant estimates on the same data.

    For each order :math:`p` present in both, reports the difference and the
    standardised discrepancy

    .. math::
        z_p = \frac{c_p^{(a)} - c_p^{(b)}}
                   {\sqrt{\hat\sigma_{p,a}^2 + \hat\sigma_{p,b}^2}} ,

    which assumes independence of the two errors. That assumption is *false* —
    both routes read the same sample — and it makes :math:`|z_p|` conservative
    (the true standard error of the difference is smaller when the errors are
    positively correlated, as they are). So :math:`|z_p|\lesssim 2` is weak
    evidence of agreement, while :math:`|z_p|\gg 2` is strong evidence that one
    route has failed: on the same data the two estimate the same population
    quantity, and only a violated assumption can separate them.

    When an estimate carries no standard error (``stderr is None``, or a zero
    entry) the denominator vanishes and :math:`z_p` is :math:`\pm\infty` for any
    non-zero difference. That infinity is *reported*, and it propagates into
    ``max_abs_z``: masking it out would turn the strongest possible disagreement
    into the largest of the remaining, finite ones — a comparison that cannot
    fail on the order that actually disagrees.

    Returns
    -------
    dict
        ``{"order": p, "labels": ..., "c_a": [...], "c_b": [...],
        "difference": [...], "z": [...], "max_abs_z": float}``.
    """
    p = min(int(a.c.size), int(b.c.size))
    ca = np.asarray(a.c[:p], dtype=np.float64)
    cb = np.asarray(b.c[:p], dtype=np.float64)
    sa = np.zeros(p) if a.stderr is None else np.asarray(a.stderr[:p], dtype=np.float64)
    sb = np.zeros(p) if b.stderr is None else np.asarray(b.stderr[:p], dtype=np.float64)
    diff = ca - cb
    denom = np.sqrt(sa**2 + sb**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        # The degenerate branch keeps the sign of the difference: an unsigned +inf
        # would say "they disagree" without saying which way round. np.sign(0)*inf
        # is NaN, so the exact-agreement case is spelled out separately.
        degenerate = np.where(diff == 0.0, 0.0, np.copysign(np.inf, diff))
        z = np.where(denom > 0.0, diff / denom, degenerate)
    usable = ~np.isnan(z)
    return {
        "order": p,
        "labels": list(labels),
        "c_a": [float(v) for v in ca],
        "c_b": [float(v) for v in cb],
        "stderr_a": [float(v) for v in sa],
        "stderr_b": [float(v) for v in sb],
        "difference": [float(v) for v in diff],
        "z": [float(v) for v in z],
        "max_abs_z": float(np.max(np.abs(z[usable]))) if usable.any() else float("nan"),
    }
