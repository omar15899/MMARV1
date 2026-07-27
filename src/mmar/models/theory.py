r"""Closed-form multifractal oracles.

This module is the *ground truth* of the library. Every estimator elsewhere in
:mod:`mmar` is validated by pointing it at a process whose scaling function and
singularity spectrum are known analytically and asking how far off it lands.
Nothing here is estimated, fitted, sampled or approximated: every quantity is a
formula, and the formulas are derived in the docstrings.

Conventions (see :mod:`mmar.core.types`, which is law)
------------------------------------------------------
*Measure view.* :math:`Z(q,\delta)=\sum_i \mu_i^q \sim \delta^{\tau(q)}`.

*Signal view.* :math:`S(q,\delta)=\langle|\Delta_\delta X|^q\rangle \sim
\delta^{\zeta(q)}`, and :math:`\zeta(q)=\tau(q)+d` with :math:`d` the
topological dimension of the support (:math:`d=1` for a time series).

*Spectrum.* :math:`f(\alpha)=\inf_q\{q\alpha-\tau(q)\}`, always the measure-view
spectrum, so :math:`\max_\alpha f = d` for a space-filling support.

Functions named ``*_tau`` return :math:`\tau`; functions named ``*_zeta`` return
:math:`\zeta`. There is no third option and no silent offset anywhere in this
file.

Contents
--------
====================================  ==============================================
Object                                :math:`\tau(q)`
====================================  ==============================================
Deterministic cascade, base :math:`b`  :math:`-\log_b \sum_i m_i^q`
Lognormal cascade, base :math:`b`      :math:`q\lambda - \tfrac12 q^2\sigma^2\ln b - 1`
Monofractal / fBm, Hurst :math:`H`     :math:`qH - d`
MMAR :math:`B_H(\theta(t))`            :math:`\tau_\theta(qH)`
====================================  ==============================================

References
----------
Mandelbrot, B. (1974). Intermittent turbulence in self-similar cascades.
*J. Fluid Mech.* **62**, 331--358.

Halsey, T., Jensen, M., Kadanoff, L., Procaccia, I., Shraiman, B. (1986).
Fractal measures and their singularities. *Phys. Rev. A* **33**, 1141.

Mandelbrot, B., Fisher, A., Calvet, L. (1997). A multifractal model of asset
returns. *Cowles Foundation Discussion Paper* 1164.

Calvet, L., Fisher, A. (2002). Multifractality in asset returns.
*Rev. Econ. Stat.* **84**, 381--406.

This module is entirely deterministic: it contains no randomness, takes no
``seed``, and every provenance record it emits therefore carries ``seed=None``.
Sampling from these models lives in the cascade/MMAR simulators, not here.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import logsumexp

from mmar.core.provenance import InputRef, Provenance, new_provenance
from mmar.core.types import FloatArray, HolderSpectrum, ScalingExponents
from mmar.core.validate import validate_measure, validate_q_grid

__all__ = [
    "cascade_alpha",
    "cascade_alpha_range",
    "cascade_dq",
    "cascade_exponents",
    "cascade_falpha",
    "cascade_falpha_parametric",
    "cascade_spectrum",
    "cascade_tau",
    "cascade_zeta",
    "cumulants_from_tau_lognormal",
    "fbm_exponents",
    "fbm_falpha",
    "fbm_tau",
    "fbm_zeta",
    "lognormal_alpha_range",
    "lognormal_conservation_sigma2",
    "lognormal_exponents",
    "lognormal_falpha",
    "lognormal_lambda_from_sigma2",
    "lognormal_spectrum",
    "lognormal_tau",
    "mmar_alpha0",
    "mmar_exponents_lognormal",
    "mmar_falpha_lognormal",
    "mmar_hurst_from_tau",
    "mmar_lambda_from_alpha0",
    "mmar_spectrum_lognormal",
    "mmar_tau",
    "mmar_tau_lognormal",
]

_VERSION = "1"


# --------------------------------------------------------------------------- #
# Coercion and validation helpers
# --------------------------------------------------------------------------- #


def _as_array(x: ArrayLike, name: str) -> FloatArray:
    """Coerce to a finite 1-D float array (scalars become one-element arrays)."""
    a = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(a)


def _cascade_weights(m: ArrayLike, b: float | int | None) -> tuple[FloatArray, float]:
    r"""Validate the multiplier vector and resolve the base.

    The weights must be *strictly positive* and sum to one. Positivity is not
    fussiness: a zero weight makes :math:`\sum_i m_i^q` diverge for every
    :math:`q<0`, so :math:`\tau` would not exist on the negative branch and
    :math:`\alpha_{\max}` would be infinite. Conservation
    (:math:`\sum_i m_i = 1`) is what makes :math:`\tau(1)=0`.

    ``b`` defaults to ``len(m)``. Allowing ``b > len(m)`` describes a cascade
    that keeps only ``len(m)`` of the ``b`` subintervals at each generation, so
    the support is a Cantor-type set of box dimension
    :math:`-\tau(0)=\log_b(\#m) < 1`.
    """
    w = validate_measure(
        np.asarray(m, dtype=np.float64).ravel(),
        name="cascade weights m",
        normalise=False,
        allow_zero=False,
    )
    if w.size < 2:
        raise ValueError(f"a cascade needs at least two multipliers, got {w.size}")
    total = float(w.sum())
    if abs(total - 1.0) > 1e-12:
        raise ValueError(
            f"cascade weights must be conservative (sum to 1), got sum(m)={total!r}; "
            "a non-conservative deterministic cascade has tau(1) != 0 and is not "
            "the object this oracle describes"
        )
    if np.any(w >= 1.0):
        raise ValueError("every multiplier must satisfy 0 < m_i < 1 for a non-degenerate cascade")
    base = float(w.size if b is None else b)
    if base <= 1.0:
        raise ValueError(f"the cascade base must exceed 1, got b={base!r}")
    if base < w.size:
        raise ValueError(f"base b={base!r} cannot be smaller than the number of branches {w.size}")
    return w, base


def _q_grid_input(q: ArrayLike | None) -> FloatArray | None:
    """Narrow ``ArrayLike`` to what :func:`~mmar.core.validate.validate_q_grid` accepts."""
    return None if q is None else np.asarray(q, dtype=np.float64).ravel()


def _check_positive(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive, got {value!r}")
    return v


def _lognormal_curvature(sigma2: float, b: float) -> float:
    r"""Return :math:`\sigma^2 \ln b`, the single curvature scale of the parabola.

    Every lognormal formula below depends on :math:`\sigma^2` and :math:`b` only
    through this product, which is also :math:`-c_2` (minus the second log-cumulant).
    """
    s2 = float(sigma2)
    if not math.isfinite(s2) or s2 <= 0.0:
        raise ValueError(
            f"sigma2 must be finite and strictly positive, got {sigma2!r}; "
            "sigma2=0 is the degenerate (Lebesgue/monofractal) case, use fbm_falpha"
        )
    return s2 * math.log(_check_positive(b, "b"))


# --------------------------------------------------------------------------- #
# Deterministic multiplicative cascade
# --------------------------------------------------------------------------- #


def cascade_tau(q: ArrayLike, m: ArrayLike, b: float | int | None = None) -> FloatArray:
    r"""Scaling function of a deterministic multiplicative cascade (MEASURE view).

    .. math::
        \tau(q) = -\log_b \Big( \sum_{i=0}^{N-1} m_i^{\,q} \Big).

    Derivation
    ----------
    Split :math:`[0,1]` into :math:`b` equal subintervals and give the
    :math:`i`-th one a fraction :math:`m_i` of the mass, recursively. At
    generation :math:`k` there are :math:`N^k` cells of side
    :math:`\delta=b^{-k}` (with :math:`N=\#\{m_i\}`), and the cell indexed by
    the digit string :math:`(i_1,\dots,i_k)` carries mass
    :math:`\mu = \prod_{j=1}^k m_{i_j}`. Hence the partition function factorises,

    .. math::
        Z(q,\delta) = \sum_{i_1,\dots,i_k} \prod_{j} m_{i_j}^{q}
                    = \Big(\sum_i m_i^q\Big)^{k}
                    = b^{-k\,\big[\log_b \sum_i m_i^q\big]}
                    = \delta^{\,\log_b \sum_i m_i^q},

    and comparing with :math:`Z\sim\delta^{\tau(q)}` gives the formula. The
    scaling is *exact* at every dyadic-type scale :math:`\delta=b^{-k}`, not
    merely asymptotic --- which is precisely what makes this an oracle.

    Anchors
    -------
    :math:`\tau(0) = -\log_b N`, equal to :math:`-1` when :math:`N=b` (full
    support), and :math:`\tau(1) = -\log_b 1 = 0` by conservation.

    Notes
    -----
    Computed through :func:`scipy.special.logsumexp`, so :math:`|q|` of several
    hundred is safe; the naive :math:`\sum m_i^q` overflows around
    :math:`q\approx -300` for :math:`m_i\approx 0.1`.

    Returns
    -------
    FloatArray
        :math:`\tau(q)` in the measure convention. Add :math:`d` (see
        :func:`cascade_zeta`) for the signal convention.
    """
    qa = _as_array(q, "q")
    w, base = _cascade_weights(m, b)
    ls = np.asarray(logsumexp(qa[:, None] * np.log(w)[None, :], axis=1), dtype=np.float64)
    return np.asarray(-ls / math.log(base), dtype=np.float64)


def cascade_zeta(
    q: ArrayLike, m: ArrayLike, b: float | int | None = None, d: float = 1.0
) -> FloatArray:
    r"""Cascade scaling function in the SIGNAL view: :math:`\zeta(q)=\tau(q)+d`.

    See :func:`cascade_tau`. The offset is the number of boxes,
    :math:`N(\delta)=\delta^{-d}`, and nothing else.
    """
    return np.asarray(cascade_tau(q, m, b) + float(d), dtype=np.float64)


def cascade_alpha(q: ArrayLike, m: ArrayLike, b: float | int | None = None) -> FloatArray:
    r"""Hölder exponent :math:`\alpha(q)=\tau'(q)`, in closed form.

    .. math::
        \alpha(q) \;=\; -\,\frac{\sum_i m_i^{\,q}\ln m_i}
                              {\ln b \;\sum_i m_i^{\,q}} \;>\; 0 .

    Derivation
    ----------
    Write :math:`S(q)=\sum_i m_i^q`, so :math:`\tau=-\ln S/\ln b` and

    .. math::
        \frac{d\tau}{dq} = -\frac{1}{\ln b}\,\frac{S'(q)}{S(q)},
        \qquad S'(q)=\sum_i m_i^q \ln m_i .

    **The minus sign is the whole point.** Since :math:`0<m_i<1` we have
    :math:`\ln m_i<0`, so :math:`S'<0` and the leading minus makes
    :math:`\alpha>0`, as a Hölder exponent must be. Dropping it flips the
    spectrum about the origin and is the classic sign bug in cascade code.

    Interpretation
    --------------
    Writing :math:`w_i(q)=m_i^q/S(q)` (a softmax over :math:`q\ln m_i`),

    .. math::
        \alpha(q) = \sum_i w_i(q)\,\big(-\log_b m_i\big),

    i.e. :math:`\alpha(q)` is the :math:`q`-tilted *average* of the per-branch
    exponents :math:`-\log_b m_i`. It is therefore automatically confined to
    :math:`[\alpha_{\min},\alpha_{\max}]` and monotonically non-increasing in
    :math:`q` (the tilt concentrates on the largest :math:`m_i` as
    :math:`q\to+\infty`), which is :math:`\tau''\le 0`, i.e. concavity.

    Returns
    -------
    FloatArray
        :math:`\alpha(q)=d\tau/dq = d\zeta/dq` (the two agree: they differ by a
        :math:`q`-independent constant).
    """
    qa = _as_array(q, "q")
    w, base = _cascade_weights(m, b)
    lm = np.log(w)
    x = qa[:, None] * lm[None, :]
    ls = np.asarray(logsumexp(x, axis=1), dtype=np.float64)
    tilt = np.exp(x - ls[:, None])
    return np.asarray(-(tilt @ lm) / math.log(base), dtype=np.float64)


def cascade_falpha_parametric(
    q: ArrayLike, m: ArrayLike, b: float | int | None = None
) -> tuple[FloatArray, FloatArray]:
    r"""Parametric spectrum :math:`\big(\alpha(q),\,f(\alpha(q))\big)`.

    .. math::
        f(\alpha(q)) = q\,\alpha(q) - \tau(q).

    This is the Legendre *tangent* construction. For a deterministic cascade
    :math:`\tau` is real-analytic and strictly concave (unless all
    :math:`m_i` are equal), so the tangent construction and the
    Legendre--Fenchel infimum coincide exactly; the distinction only bites for
    estimated, non-concave :math:`\hat\tau` (see :mod:`mmar.core.legendre`).

    At :math:`q=0` this returns the apex: :math:`f=-\tau(0)=\log_b N`, which is
    the box dimension of the support.

    Returns
    -------
    (alpha, f)
        Both of shape ``q.shape``. ``f`` is the measure-view spectrum.
    """
    qa = _as_array(q, "q")
    alpha = cascade_alpha(qa, m, b)
    tau = cascade_tau(qa, m, b)
    return alpha, np.asarray(qa * alpha - tau, dtype=np.float64)


def cascade_alpha_range(m: ArrayLike, b: float | int | None = None) -> tuple[float, float]:
    r"""Support of the spectrum: :math:`(\alpha_{\min},\alpha_{\max})`.

    .. math::
        \alpha_{\min} = -\log_b \max_i m_i, \qquad
        \alpha_{\max} = -\log_b \min_i m_i .

    These are the limits of :math:`\alpha(q)` as :math:`q\to+\infty` and
    :math:`q\to-\infty` respectively: the :math:`q`-tilt in
    :func:`cascade_alpha` concentrates all weight on the largest multiplier for
    :math:`q\to+\infty` (the densest, least singular cells, smallest
    :math:`\alpha`) and on the smallest for :math:`q\to-\infty`.
    """
    w, base = _cascade_weights(m, b)
    lb = math.log(base)
    return (float(-math.log(float(w.max())) / lb), float(-math.log(float(w.min())) / lb))


def cascade_falpha(alpha: ArrayLike, m: ArrayLike, b: float | int | None = None) -> FloatArray:
    r"""Closed-form :math:`f(\alpha)` for a TWO-BRANCH cascade of base :math:`b`.

    With :math:`\alpha_{\min}=-\log_b \max(m_0,m_1)`,
    :math:`\alpha_{\max}=-\log_b \min(m_0,m_1)` and

    .. math::
        t = \frac{\alpha_{\max}-\alpha}{\alpha_{\max}-\alpha_{\min}} \in [0,1],

    .. math::
        f(\alpha) = \frac{H_2(t)}{\log_2 b},
        \qquad H_2(t) = -\big[\, t\log_2 t + (1-t)\log_2(1-t) \,\big],

    the binary entropy of :math:`t` measured in base-:math:`b` digits. For the
    binomial cascade (:math:`b=2`, the default) the prefactor is 1 and this is
    the classical :math:`f(\alpha)=H_2(t)`.

    Derivation
    ----------
    At generation :math:`k` a surviving cell is labelled by a binary string; if
    a fraction :math:`t` of its digits select the *larger* multiplier its mass
    is :math:`\mu = m_{\max}^{tk} m_{\min}^{(1-t)k}`, so with
    :math:`\delta=b^{-k}` its coarse Hölder exponent is

    .. math::
        \alpha = \frac{\ln\mu}{\ln\delta} = t\,\alpha_{\min} + (1-t)\,\alpha_{\max},

    which inverts to the :math:`t` above. The number of such cells is
    :math:`\binom{k}{tk} = 2^{k H_2(t) + O(\log k)}` by Stirling, and the
    box-counting definition
    :math:`N_\delta(\alpha)\sim\delta^{-f(\alpha)} = b^{k f(\alpha)}` gives
    :math:`b^{kf}=2^{kH_2(t)}`, i.e. :math:`f(\alpha)=H_2(t)\log_b 2`. This is
    the Cramér/large-deviation rate function of the digit process, which is why
    it is an entropy.

    **The base matters.** Counting is binary but boxes shrink by :math:`b`, so
    dropping the :math:`\log_2 b` would report the apex at :math:`f=1` for a
    reduced-support cascade whose support has dimension :math:`\log_b 2 < 1`.
    Cross-checked against :func:`cascade_falpha_parametric`, which knows
    :math:`b`.

    Checks (verified numerically in ``tests/theory/test_theory_oracles.py``)
    -----------------------------------------------------------------------
    :math:`t=\tfrac12 \Rightarrow f=\log_b 2` at
    :math:`\alpha_0=\tfrac12(\alpha_{\min}+\alpha_{\max})`, which equals
    :math:`-\tau(0)=D_0`; :math:`t\in\{0,1\} \Rightarrow f=0` at the two
    endpoints, using :math:`0\log 0 := 0` by continuity.

    Parameters
    ----------
    b
        Cascade base; defaults to 2 (the classical binomial). ``b > 2`` keeps
        only two of the ``b`` subintervals, so the support is a Cantor set of
        dimension :math:`\log_b 2` and the whole spectrum scales down by that
        factor.

    Returns
    -------
    FloatArray
        :math:`f(\alpha)` inside :math:`[\alpha_{\min},\alpha_{\max}]` and
        :math:`-\infty` outside. Outside the support no cell has that exponent
        at *any* generation, so :math:`-\infty` is the exact value of the
        Legendre--Fenchel transform there, not a missing-data marker.

    Raises
    ------
    ValueError
        If ``m`` is not a two-element vector, or if :math:`m_0=m_1`
        (equal multipliers: the spectrum degenerates to the single point
        :math:`(\log_b 2, \log_b 2)` and :math:`t` is undefined).
    """
    a = _as_array(alpha, "alpha")
    w, base = _cascade_weights(m, b)
    if w.size != 2:
        raise ValueError(
            f"cascade_falpha is the closed form for the binomial (two-multiplier) "
            f"cascade; got {w.size} multipliers. Use cascade_falpha_parametric "
            "for an arbitrary multiplier vector."
        )
    a_min, a_max = cascade_alpha_range(w, base)
    span = a_max - a_min
    if span <= 0.0:
        raise ValueError(
            "m0 = m1 = 1/2 with b = 2 is Lebesgue measure: the spectrum is the degenerate "
            "point {(1, 1)} and f(alpha) has no entropy form. Use fbm_falpha instead."
        )

    # The support test is taken with a tolerance, not as an exact inequality.
    # cascade_falpha_parametric saturates at alpha_max for q below about -40 and
    # lands one ULP outside; an exact test would then return -inf where the true
    # value is 0, and -inf is infectious in any residual. The tolerance is far
    # below the resolution of any estimator, so it cannot mask a real overshoot.
    atol = 1e-12 * max(1.0, abs(a_min), abs(a_max))
    inside = (a >= a_min - atol) & (a <= a_max + atol)
    t = np.clip((a_max - a) / span, 0.0, 1.0)
    # log2 is evaluated on a clamped copy so the out-of-support branch never
    # produces a warning; 0*log2(0) is set to 0 by continuity of x -> x log x.
    ts = np.where(inside, t, 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        left = np.where(ts > 0.0, ts * np.log2(np.where(ts > 0.0, ts, 1.0)), 0.0)
        right = np.where(ts < 1.0, (1.0 - ts) * np.log2(np.where(ts < 1.0, 1.0 - ts, 1.0)), 0.0)
    f = -(left + right) / math.log2(base)
    return np.asarray(np.where(inside, f, -np.inf), dtype=np.float64)


def cascade_dq(q: ArrayLike, m: ArrayLike, b: float | int | None = None) -> FloatArray:
    r"""Generalised (Rényi) dimensions :math:`D_q = \tau(q)/(q-1)`.

    The :math:`q=1` value is the removable singularity

    .. math::
        D_1 = \lim_{q\to 1}\frac{\tau(q)}{q-1} = \tau'(1)
            = -\sum_i m_i \log_b m_i ,

    the **information dimension**, obtained from L'Hôpital using
    :math:`\tau(1)=0` (conservation) and :math:`\alpha(1)=\tau'(1)`, which is
    :func:`cascade_alpha` evaluated at :math:`q=1` where the tilt weights
    :math:`w_i(1)=m_i` are the multipliers themselves --- i.e. :math:`D_1` is
    the Shannon entropy of :math:`(m_i)` in base :math:`b`.

    Landmarks: :math:`D_0=-\tau(0)=\log_b N` (box dimension of the support,
    :math:`=1` for :math:`N=b`), :math:`D_2` the correlation dimension.
    :math:`q\mapsto D_q` is non-increasing --- a consequence of concavity of
    :math:`\tau` together with :math:`\tau(1)=0`, since :math:`D_q` is the slope
    of the chord from :math:`(1,0)` to :math:`(q,\tau(q))` and a concave function
    has non-increasing chord slopes.

    Notes
    -----
    Within :math:`|q-1|<10^{-6}` the analytic limit is substituted: the direct
    quotient loses about eight significant digits there to cancellation.
    """
    qa = _as_array(q, "q")
    w, base = _cascade_weights(m, b)
    d1 = float(-np.sum(w * np.log(w)) / math.log(base))
    tau = cascade_tau(qa, w, base)
    near_one = np.abs(qa - 1.0) < 1e-6
    denom = np.where(near_one, 1.0, qa - 1.0)
    return np.asarray(np.where(near_one, d1, tau / denom), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Lognormal (canonical Mandelbrot) cascade
# --------------------------------------------------------------------------- #


def lognormal_conservation_sigma2(lam: float, b: float = 2.0) -> float:
    r"""The :math:`\sigma^2` that makes the lognormal cascade conservative.

    .. math::
        \sigma^2 = \frac{2(\lambda-1)}{\ln b}.

    Derivation
    ----------
    The multiplier is :math:`M=b^{-V}` with :math:`V\sim\mathcal N(\lambda,\sigma^2)`.
    Using the Gaussian MGF, :math:`\mathbb E[e^{-tV}]=e^{-t\lambda+t^2\sigma^2/2}`
    with :math:`t=\ln b`,

    .. math::
        \mathbb E[M] = b^{-\lambda + \sigma^2 \ln b / 2}.

    Mass conservation in expectation over :math:`b` children requires
    :math:`b\,\mathbb E[M]=1`, i.e. :math:`\mathbb E[M]=1/b`, hence
    :math:`\lambda - \tfrac12\sigma^2\ln b = 1`. Solving for :math:`\sigma^2`
    gives the statement; this is the constraint that pins :math:`\tau(1)=0`.

    Raises
    ------
    ValueError
        If :math:`\lambda \le 1`. :math:`\lambda=1` forces :math:`\sigma^2=0`
        (Lebesgue measure) and :math:`\lambda<1` would need a negative variance:
        a conservative lognormal cascade *must* have
        :math:`\alpha_0=\lambda>1=D_0`, the intuition being that the typical cell
        is less singular than the average because the mass concentrates on rare
        cells.
    """
    lam_f = float(lam)
    if not math.isfinite(lam_f) or lam_f <= 1.0:
        raise ValueError(
            f"a conservative lognormal cascade needs lam > 1, got {lam!r}; "
            "lam = 1 is the degenerate sigma2 = 0 (Lebesgue) case"
        )
    return 2.0 * (lam_f - 1.0) / math.log(_check_positive(b, "b"))


def lognormal_lambda_from_sigma2(sigma2: float, b: float = 2.0) -> float:
    r"""Inverse of :func:`lognormal_conservation_sigma2`:
    :math:`\lambda = 1 + \tfrac12\sigma^2\ln b`.

    Because :math:`\alpha_0=\lambda` and :math:`c_2=-\sigma^2\ln b`, this reads
    :math:`\alpha_0 = 1 - c_2/2`: the apex of the spectrum sits above the
    support dimension by exactly half the intermittency.
    """
    s2 = float(sigma2)
    if not math.isfinite(s2) or s2 < 0.0:
        raise ValueError(f"sigma2 must be finite and non-negative, got {sigma2!r}")
    return 1.0 + 0.5 * s2 * math.log(_check_positive(b, "b"))


def lognormal_tau(q: ArrayLike, lam: float, sigma2: float, b: float = 2.0) -> FloatArray:
    r"""Scaling function of the lognormal cascade (MEASURE view).

    **Theorem.** For the canonical cascade of base :math:`b` with multipliers
    :math:`M=b^{-V}`, :math:`V\sim\mathcal N(\lambda,\sigma^2)` i.i.d. across
    cells and generations,

    .. math::
        \boxed{\;\tau(q) \;=\; q\lambda \;-\; \tfrac12 q^2\sigma^2\ln b \;-\; 1\;}

    *Proof.* First, the :math:`q`-th moment of one multiplier. With
    :math:`t=q\ln b` in the Gaussian MGF,

    .. math::
        \mathbb E[M^q] = \mathbb E\!\left[e^{-q V\ln b}\right]
        = \exp\!\Big(-q\lambda\ln b + \tfrac12 q^2\sigma^2\ln^2 b\Big)
        = b^{\,-q\lambda + q^2\sigma^2\ln b/2}.

    At generation :math:`k` the grid has :math:`b^k` cells of side
    :math:`\delta=b^{-k}`, and each cell's mass is the product of the :math:`k`
    independent multipliers along its branch, so :math:`\mathbb E[\mu^q] =
    \mathbb E[M^q]^k`. Therefore

    .. math::
        \mathbb E[Z(q,\delta)] = b^k\,\mathbb E[M^q]^k
        = b^{\,k\left(1 - q\lambda + q^2\sigma^2\ln b/2\right)}
        = \delta^{\,-\left(1 - q\lambda + q^2\sigma^2\ln b/2\right)} .

    Matching :math:`Z\sim\delta^{\tau(q)}` gives
    :math:`\tau(q) = q\lambda - \tfrac12 q^2\sigma^2\ln b - 1`. :math:`\square`

    Anchors: :math:`\tau(0)=-1` unconditionally, and under conservation
    (:math:`\lambda-\tfrac12\sigma^2\ln b=1`, see
    :func:`lognormal_conservation_sigma2`) also :math:`\tau(1)=0`.

    Caveat (Mandelbrot 1974; Molchan 1996)
    --------------------------------------
    This is the exponent of :math:`\mathbb E[Z]`, i.e. the *annealed* scaling
    function. It coincides with the almost-sure exponent only for
    :math:`q<q_{\mathrm{crit}}`, where :math:`q_{\mathrm{crit}}` solves
    :math:`\tau(q)=q\tau'(q)` --- equivalently
    :math:`q_{\mathrm{crit}}=\sqrt{2/(\sigma^2\ln b)}`, the point where the
    tangent from the origin touches the parabola and where
    :math:`f(\alpha(q))=0`. Above it a single sample's partition function is
    dominated by cells that a.s. do not occur, the empirical :math:`\tau`
    linearises, and the estimated spectrum is *linear*, not parabolic. This is
    the main reason estimators tested against this oracle must be given a
    bounded :math:`q` window.
    """
    qa = _as_array(q, "q")
    lam_f = float(lam)
    if not math.isfinite(lam_f):
        raise ValueError(f"lam must be finite, got {lam!r}")
    kappa = _lognormal_curvature(sigma2, b)
    return np.asarray(qa * lam_f - 0.5 * qa**2 * kappa - 1.0, dtype=np.float64)


def lognormal_falpha(alpha: ArrayLike, lam: float, sigma2: float, b: float = 2.0) -> FloatArray:
    r"""Exact singularity spectrum of the lognormal cascade: a parabola.

    .. math::
        \boxed{\;f(\alpha) \;=\; 1 \;-\; \frac{(\alpha-\lambda)^2}{2\,\sigma^2\ln b}\;}

    Derivation (the algebra, in full, because a factor of 2 here is fatal)
    ---------------------------------------------------------------------
    Write :math:`\kappa=\sigma^2\ln b`, so
    :math:`\tau(q)=q\lambda-\tfrac12 q^2\kappa-1`. Then

    .. math::
        \alpha(q) = \tau'(q) = \lambda - q\kappa
        \quad\Longleftrightarrow\quad
        q = \frac{\lambda-\alpha}{\kappa} = -\frac{\alpha-\lambda}{\kappa}.

    Since :math:`\tau''=-\kappa<0`, :math:`\tau` is strictly concave and the
    stationary point is the infimum, so :math:`f(\alpha)=q\alpha-\tau(q)` at
    that :math:`q`. Substituting,

    .. math::
        f = q\alpha - q\lambda + \tfrac12 q^2\kappa + 1
          = q(\alpha-\lambda) + \tfrac12 q^2 \kappa + 1 .

    Term by term, with :math:`u=\alpha-\lambda` and :math:`q=-u/\kappa`:

    .. math::
        q(\alpha-\lambda) = -\frac{u^2}{\kappa},
        \qquad
        \tfrac12 q^2\kappa = \tfrac12\,\frac{u^2}{\kappa^2}\,\kappa
                           = \frac{u^2}{2\kappa} .

    Their sum is :math:`-u^2/\kappa + u^2/(2\kappa) = -u^2/(2\kappa)`, giving
    :math:`f = 1 - u^2/(2\kappa)`. :math:`\square`

    Immediate consequences: :math:`\alpha_0=\lambda` is the apex,
    :math:`f(\alpha_0)=1=D_0`, the curvature is
    :math:`f''=-1/\kappa=1/c_2` (matching
    :meth:`~mmar.core.types.LogCumulants.parabolic_spectrum`), and the spectrum
    is exactly symmetric --- lognormal cascades cannot produce a skewed
    :math:`f(\alpha)`, so measured asymmetry is evidence *against* this model.

    Returns
    -------
    FloatArray
        The parabola evaluated everywhere, **unclipped**. Values
        :math:`f(\alpha)<0` outside :func:`lognormal_alpha_range` are not an
        error: :math:`\tau` is a smooth strictly concave quadratic on all of
        :math:`\mathbb R`, so its Legendre--Fenchel transform is the full
        parabola, and negative :math:`f` is the large-deviation statement that
        such exponents occur in a vanishing fraction of samples. Clip only for
        the box-counting reading, and say so.
    """
    a = _as_array(alpha, "alpha")
    lam_f = float(lam)
    if not math.isfinite(lam_f):
        raise ValueError(f"lam must be finite, got {lam!r}")
    kappa = _lognormal_curvature(sigma2, b)
    return np.asarray(1.0 - (a - lam_f) ** 2 / (2.0 * kappa), dtype=np.float64)


def lognormal_alpha_range(lam: float, sigma2: float, b: float = 2.0) -> tuple[float, float]:
    r"""Where the lognormal spectrum vanishes:
    :math:`\lambda \pm \sqrt{2\sigma^2\ln b}`.

    Setting :math:`f(\alpha)=0` in :func:`lognormal_falpha` gives
    :math:`(\alpha-\lambda)^2 = 2\sigma^2\ln b`. The width is therefore
    :math:`\Delta\alpha = 2\sqrt{2\sigma^2\ln b} = 2\sqrt{-2c_2}`, the standard
    scalar measure of multifractality for this family.

    Outside this interval :math:`f<0`: those exponents exist in the
    large-deviation sense but occupy sets of negative dimension, i.e. they are
    absent from a typical finite sample. This is the interval an estimator can
    honestly be asked to reproduce.
    """
    lam_f = float(lam)
    if not math.isfinite(lam_f):
        raise ValueError(f"lam must be finite, got {lam!r}")
    half = math.sqrt(2.0 * _lognormal_curvature(sigma2, b))
    return (lam_f - half, lam_f + half)


def cumulants_from_tau_lognormal(lam: float, sigma2: float, b: float = 2.0) -> tuple[float, float]:
    r"""Log-cumulants :math:`(c_1,c_2)` of the lognormal cascade.

    In the expansion :math:`\zeta(q)=\sum_{p\ge1} c_p q^p/p!` with
    :math:`\zeta=\tau+1`,

    .. math::
        \zeta(q) = q\lambda - \tfrac12 q^2 \sigma^2\ln b
        \;\Longrightarrow\;
        c_1 = \lambda, \qquad \frac{c_2}{2!}q^2 = -\tfrac12 q^2\sigma^2\ln b
        \;\Longrightarrow\; c_2 = -\sigma^2\ln b .

    All :math:`c_p=0` for :math:`p\ge3`: the lognormal cascade is *exactly*
    second order, which is why it is the reference model for the log-cumulant
    estimator. :math:`c_1=\alpha_0` and the intermittency coefficient is
    :math:`-c_2=\sigma^2\ln b\ge 0`, vanishing iff :math:`\sigma^2=0`
    (monofractal).

    Returns
    -------
    (c1, c2)
        With ``c2 <= 0`` always.
    """
    lam_f = float(lam)
    if not math.isfinite(lam_f):
        raise ValueError(f"lam must be finite, got {lam!r}")
    return lam_f, -_lognormal_curvature(sigma2, b)


# --------------------------------------------------------------------------- #
# Monofractal / fractional Brownian motion
# --------------------------------------------------------------------------- #


def fbm_zeta(q: ArrayLike, H: float) -> FloatArray:
    r"""fBm structure-function exponent (SIGNAL view): :math:`\zeta(q)=qH`.

    Self-similarity of fractional Brownian motion,
    :math:`B_H(ct)\stackrel{d}{=}c^H B_H(t)`, gives
    :math:`\mathbb E|\Delta_\delta B_H|^q = \delta^{qH}\,\mathbb E|B_H(1)|^q`,
    exactly linear in :math:`q` for every :math:`q>-1` (below which the moment
    of a Gaussian absolute value diverges).

    Linearity of :math:`\zeta` is the *definition* of monofractality: it is the
    null hypothesis every multifractality claim must beat.
    """
    qa = _as_array(q, "q")
    h = float(H)
    if not math.isfinite(h) or not 0.0 < h < 1.0:
        raise ValueError(f"the Hurst exponent of fBm must lie in (0, 1), got {H!r}")
    return np.asarray(qa * h, dtype=np.float64)


def fbm_tau(q: ArrayLike, H: float, d: float = 1.0) -> FloatArray:
    r"""fBm exponent in the MEASURE view: :math:`\tau(q)=qH-d`.

    Note :math:`\tau(1/H)=1-d=0` for :math:`d=1`: the MMAR identity
    :math:`\tau(1/H)=0` degenerates to the monofractal case, as it must, since
    fBm is the MMAR with a trivial (Lebesgue) trading time.
    """
    return np.asarray(fbm_zeta(q, H) - float(d), dtype=np.float64)


def fbm_falpha(alpha: ArrayLike, H: float, d: float = 1.0) -> FloatArray:
    r"""Degenerate spectrum of fBm: :math:`f(H)=d`, :math:`f(\alpha)=-\infty` else.

    :math:`\tau(q)=qH-d` is affine, so its Legendre--Fenchel transform

    .. math::
        f(\alpha)=\inf_q\{q\alpha-qH+d\}
        = \begin{cases} d, & \alpha = H,\\ -\infty, & \alpha\neq H,\end{cases}

    because for :math:`\alpha\neq H` the linear function :math:`q(\alpha-H)` is
    unbounded below. The whole sample path has the single exponent :math:`H`
    almost everywhere.

    Numerically the test :math:`\alpha=H` is exact equality. A finite estimator
    will never reproduce a spike; what it produces is a narrow, noise-driven
    parabola whose width measures the estimator's variance, not
    multifractality. That is exactly what makes fBm the calibration null.
    """
    a = _as_array(alpha, "alpha")
    h = float(H)
    if not math.isfinite(h) or not 0.0 < h < 1.0:
        raise ValueError(f"the Hurst exponent of fBm must lie in (0, 1), got {H!r}")
    return np.asarray(np.where(a == h, float(d), -np.inf), dtype=np.float64)


# --------------------------------------------------------------------------- #
# MMAR: Mandelbrot-Fisher-Calvet (1997)
# --------------------------------------------------------------------------- #


def mmar_tau(
    q: ArrayLike, H: float, cascade_tau_fn: Callable[[FloatArray], FloatArray]
) -> FloatArray:
    r"""MMAR scaling function by time deformation: :math:`\tau_X(q)=\tau_\theta(qH)`.

    **Setup.** :math:`X(t)=B_H(\theta(t))` where :math:`\theta` is the c.d.f. of
    a multifractal measure (*trading time*), independent of the fBm
    :math:`B_H`.

    **Theorem (MFC 1997).** If :math:`\theta` has scaling function
    :math:`\tau_\theta` then :math:`X` has :math:`\tau_X(q)=\tau_\theta(qH)`.

    *Sketch.* By self-similarity of :math:`B_H` and independence,

    .. math::
        \mathbb E|\Delta_\delta X|^q
        = \mathbb E\big[\,\mathbb E\big[|B_H(\Delta_\delta\theta)|^q
          \,\big|\,\theta\big]\,\big]
        = \mathbb E|B_H(1)|^q \;\mathbb E\big[(\Delta_\delta\theta)^{qH}\big].

    The trading-time increment over a box of side :math:`\delta` is precisely
    that box's measure, so :math:`\mathbb E[(\Delta_\delta\theta)^{p}] \sim
    \delta^{\tau_\theta(p)+1}` (the extra :math:`+1` converts the partition sum
    :math:`\sum_i\mu_i^p\sim\delta^{\tau_\theta(p)}` into an average over the
    :math:`\delta^{-1}` boxes). Hence
    :math:`\mathbb E|\Delta_\delta X|^q \sim \delta^{\tau_\theta(qH)+1}
    = \delta^{\zeta_X(q)}`, i.e. :math:`\zeta_X(q)=\tau_\theta(qH)+1` and
    :math:`\tau_X(q)=\zeta_X(q)-1=\tau_\theta(qH)`. :math:`\square`

    Consequences, all of which this module exposes separately:
    :math:`\tau_X(0)=\tau_\theta(0)=-1`;
    :math:`\tau_X(1/H)=\tau_\theta(1)=0` (:func:`mmar_hurst_from_tau`);
    :math:`f_X(\alpha)=f_\theta(\alpha/H)`;
    :math:`\alpha_0^X = H\,\alpha_0^\theta` (:func:`mmar_alpha0`).

    Parameters
    ----------
    cascade_tau_fn
        Any :math:`\tau_\theta` in the *measure* convention, e.g.
        ``lambda p: cascade_tau(p, m)``. It is evaluated at :math:`qH`, so it
        must accept the rescaled grid --- note the argument range needed is
        :math:`H` times narrower than the :math:`q` range asked for.
    """
    qa = _as_array(q, "q")
    h = float(H)
    if not math.isfinite(h) or not 0.0 < h < 1.0:
        raise ValueError(f"the MMAR Hurst exponent must lie in (0, 1), got {H!r}")
    out = np.asarray(cascade_tau_fn(np.asarray(qa * h, dtype=np.float64)), dtype=np.float64)
    if out.shape != qa.shape:
        raise ValueError(f"cascade_tau_fn returned shape {out.shape}, expected {qa.shape}")
    return out


def mmar_tau_lognormal(
    q: ArrayLike, H: float, lam: float, sigma2: float, b: float = 2.0
) -> FloatArray:
    r"""MMAR with lognormal trading time:
    :math:`\tau_X(q)=qH\lambda-\tfrac12 q^2H^2\sigma^2\ln b-1`.

    Direct substitution of :math:`qH` into :func:`lognormal_tau`. The MMAR
    inherits the parabola with :math:`\lambda\mapsto H\lambda` and
    :math:`\sigma^2\mapsto H^2\sigma^2`, so an MMAR of Hurst :math:`H` over a
    cascade :math:`(\lambda,\sigma^2)` is observationally identical --- at the
    level of :math:`\tau` alone --- to a *measure* with those rescaled
    parameters. Identification of :math:`H` separately requires the extra
    conservation constraint :math:`\tau_\theta(1)=0`; see
    :func:`mmar_hurst_from_tau`.
    """
    h = float(H)
    if not math.isfinite(h) or not 0.0 < h < 1.0:
        raise ValueError(f"the MMAR Hurst exponent must lie in (0, 1), got {H!r}")
    return lognormal_tau(_as_array(q, "q") * h, lam, sigma2, b)


def mmar_falpha_lognormal(
    alpha: ArrayLike, H: float, lam: float, sigma2: float, b: float = 2.0
) -> FloatArray:
    r"""MMAR spectrum with lognormal trading time:
    :math:`f_X(\alpha)=f_\theta(\alpha/H)`.

    .. math::
        f_X(\alpha) = 1 - \frac{(\alpha/H-\lambda)^2}{2\sigma^2\ln b}
                    = 1 - \frac{(\alpha-H\lambda)^2}{2 H^2\sigma^2\ln b} .

    *Why the argument is rescaled.* From :math:`\tau_X(q)=\tau_\theta(qH)`,

    .. math::
        f_X(\alpha)=\inf_q\{q\alpha-\tau_\theta(qH)\}
        \overset{p=qH}{=}\inf_p\Big\{p\,\frac{\alpha}{H}-\tau_\theta(p)\Big\}
        = f_\theta(\alpha/H).

    So the MMAR spectrum is the trading-time spectrum *stretched* horizontally
    by :math:`H` about the origin, at unchanged height: :math:`\max f_X = 1`
    still, apex at :math:`\alpha_0^X=H\lambda`, width scaled by :math:`H`. The
    price path is more singular than the trading time exactly by the factor
    :math:`H`.

    Unclipped, as in :func:`lognormal_falpha`.
    """
    h = float(H)
    if not math.isfinite(h) or not 0.0 < h < 1.0:
        raise ValueError(f"the MMAR Hurst exponent must lie in (0, 1), got {H!r}")
    a = _as_array(alpha, "alpha")
    return lognormal_falpha(a / h, lam, sigma2, b)


def mmar_hurst_from_tau(q: ArrayLike, tau: ArrayLike, *, q_min: float = 1e-6) -> float:
    r"""Recover :math:`H` from :math:`\tau_X` via the MFC identity
    :math:`\tau_X(1/H)=0`.

    **Proposition.** If the trading time is conservative
    (:math:`\tau_\theta(1)=0`) then

    .. math::
        \tau_X(1/H) = \tau_\theta\!\big(\tfrac1H \cdot H\big)
                    = \tau_\theta(1) = 0 .

    *Proof.* Immediate from :math:`\tau_X(q)=\tau_\theta(qH)`
    (:func:`mmar_tau`) evaluated at :math:`q=1/H`. :math:`\square`

    **The positive root is not unique, and picking the wrong one is silent.**
    :math:`\tau_X` is concave with :math:`\tau_X(0)=-1<0`, so it has at most two
    roots, but *both* can be positive: for a lognormal trading time
    :math:`\tau_\theta(p)=p\lambda-\tfrac12 p^2\kappa-1` vanishes at :math:`p=1`
    *and* at :math:`p=(2\lambda-\kappa)/\kappa` --- with
    :math:`\lambda=1.05,\kappa=0.1` that second root is :math:`p=20`, i.e.
    :math:`q=20/H` in the :math:`X` variable. Only the **up-crossing** (the one
    where :math:`\tau_X` passes from negative to positive) is :math:`1/H`; it is
    the up-crossing because :math:`\tau_\theta'(1)=D_1>0`. This function
    therefore selects the first up-crossing and ignores down-crossings; a
    :math:`q` window that contains only the down-crossing raises rather than
    returning :math:`H=1/36.4`.

    Even the right root is only correct in principle: on a finite sample it
    inherits the full curvature error of :math:`\hat\tau`, and since
    :math:`H<1` it sits at :math:`q^*>1`, in the region where high moments are
    least reliable. Prefer a joint fit when :math:`\hat\tau` is noisy.

    Parameters
    ----------
    q, tau
        A monotone :math:`q` grid and :math:`\tau_X` on it, in the **measure**
        convention. Pass ``exponents.tau``, never ``exponents.zeta``.
    q_min
        Lower end of the bracket; the trivial neighbourhood of :math:`q=0`,
        where :math:`\tau(0)=-1`, is excluded automatically.

    Returns
    -------
    float
        :math:`H = 1/q^*`.

    Raises
    ------
    ValueError
        If :math:`\tau` has no up-crossing on :math:`q>0`. Returning NaN here
        would let a failed identification propagate silently into a calibration,
        and returning a down-crossing would be worse still: a confident, wrong
        :math:`H`.
    """
    qa = _as_array(q, "q")
    ta = _as_array(tau, "tau")
    if qa.shape != ta.shape:
        raise ValueError(f"q{qa.shape} and tau{ta.shape} must match")
    order = np.argsort(qa)
    qs, ts = qa[order], ta[order]
    pos = qs > q_min
    if int(pos.sum()) < 2:
        raise ValueError(f"need at least two grid points with q > {q_min}, got {int(pos.sum())}")
    qs, ts = qs[pos], ts[pos]
    # Only an up-crossing identifies H: tau_theta'(1) = D_1 > 0, so tau_X passes
    # from negative to positive at q = 1/H. The second (down) crossing of the
    # concave tau_X is the trading time's upper root and has nothing to do with H.
    upcross = np.nonzero((ts[:-1] <= 0.0) & (ts[1:] > 0.0))[0]
    if upcross.size == 0:
        raise ValueError(
            f"tau has no sign change upward on q in [{qs[0]:.3g}, {qs[-1]:.3g}] "
            f"(tau ranges over [{ts.min():.3g}, {ts.max():.3g}]); the MMAR identity "
            "tau_X(1/H) = 0 cannot identify H from this grid. Widen the q window "
            "downward: the root that identifies H is the one where tau crosses zero "
            "from below."
        )
    i = int(upcross[0])
    interp = PchipInterpolator(qs, ts)
    q_star = float(brentq(lambda z: float(interp(z)), qs[i], qs[i + 1], xtol=1e-14, rtol=8.9e-16))
    if q_star <= 0.0:  # pragma: no cover - excluded by the q > q_min filter
        raise ValueError(f"non-positive root q* = {q_star!r}; H is undefined")
    return 1.0 / q_star


def mmar_alpha0(H: float, lam: float) -> float:
    r"""MMAR calibration identity: :math:`\alpha_0^X = H\,\alpha_0^\theta = H\lambda`.

    Since :math:`f_X(\alpha)=f_\theta(\alpha/H)`, the maximiser rescales:
    :math:`\alpha_0^X = H\alpha_0^\theta`, and for the lognormal cascade
    :math:`\alpha_0^\theta=\lambda`. Inverting,

    .. math::
        \lambda = \frac{\alpha_0}{H},

    which together with the conservation constraint
    :math:`\sigma^2 = 2(\lambda-1)/\ln b` fully determines the MMAR from the two
    measured quantities :math:`(H,\alpha_0)` --- this is the standard MFC
    calibration and the reason the estimation pipeline reports
    :math:`\alpha_0` at all.
    """
    h = float(H)
    if not math.isfinite(h) or not 0.0 < h < 1.0:
        raise ValueError(f"the MMAR Hurst exponent must lie in (0, 1), got {H!r}")
    lam_f = float(lam)
    if not math.isfinite(lam_f):
        raise ValueError(f"lam must be finite, got {lam!r}")
    return h * lam_f


def mmar_lambda_from_alpha0(alpha_0: float, H: float) -> float:
    r"""Inverse calibration identity :math:`\lambda=\alpha_0/H`; see :func:`mmar_alpha0`."""
    h = float(H)
    if not math.isfinite(h) or not 0.0 < h < 1.0:
        raise ValueError(f"the MMAR Hurst exponent must lie in (0, 1), got {H!r}")
    a0 = float(alpha_0)
    if not math.isfinite(a0):
        raise ValueError(f"alpha_0 must be finite, got {alpha_0!r}")
    return a0 / h


# --------------------------------------------------------------------------- #
# Result objects with provenance
# --------------------------------------------------------------------------- #


def _provenance(
    estimator: str, params: dict[str, Any], arrays: dict[str, FloatArray]
) -> Provenance:
    return new_provenance(
        estimator=estimator,
        estimator_version=_VERSION,
        params=params,
        inputs=tuple(InputRef.from_array(k, v) for k, v in sorted(arrays.items())),
        seed=None,
        notes=("closed-form oracle; exact, not estimated",),
    )


def cascade_exponents(
    q: ArrayLike | None = None,
    m: ArrayLike = (0.6, 0.4),
    *,
    b: float | int | None = None,
    d_support: float = 1.0,
) -> ScalingExponents:
    r"""Exact :class:`~mmar.core.types.ScalingExponents` of a deterministic cascade.

    The returned object carries :math:`\zeta=\tau+d` and derives :math:`\tau`,
    so it can be fed to anything that consumes estimated exponents --- which is
    the point: an estimator's output and this oracle are the same type and are
    compared directly.
    """
    qa = validate_q_grid(_q_grid_input(q))
    w, base = _cascade_weights(m, b)
    zeta = cascade_tau(qa, w, base) + float(d_support)
    return ScalingExponents(
        q=qa,
        zeta=zeta,
        d_support=float(d_support),
        estimator="theory.cascade",
        provenance=_provenance(
            "theory.cascade_exponents",
            {"b": base, "d_support": float(d_support)},
            {"q": qa, "m": w},
        ),
    )


def cascade_spectrum(
    q: ArrayLike | None = None,
    m: ArrayLike = (0.6, 0.4),
    *,
    b: float | int | None = None,
    d_support: float = 1.0,
) -> HolderSpectrum:
    r"""Exact :math:`f(\alpha)` of a deterministic cascade, on the tangent grid.

    Uses the parametric construction, which for an analytic strictly concave
    :math:`\tau` is exact (see :func:`cascade_falpha_parametric`). Sampling is
    therefore in :math:`q`, not in :math:`\alpha`: the tails of the spectrum are
    sparsely covered unless the :math:`q` grid is widened.
    """
    qa = validate_q_grid(_q_grid_input(q))
    w, base = _cascade_weights(m, b)
    alpha, f = cascade_falpha_parametric(qa, w, base)
    return HolderSpectrum(
        alpha=alpha,
        f_alpha=f,
        q=qa,
        d_support=float(d_support),
        method="theory.cascade",
        provenance=_provenance(
            "theory.cascade_spectrum",
            {"b": base, "d_support": float(d_support)},
            {"q": qa, "m": w},
        ),
    )


def lognormal_exponents(
    q: ArrayLike | None = None,
    *,
    lam: float = 1.05,
    sigma2: float | None = None,
    b: float = 2.0,
    d_support: float = 1.0,
) -> ScalingExponents:
    r"""Exact exponents of the lognormal cascade.

    ``sigma2=None`` (the default) imposes conservation via
    :func:`lognormal_conservation_sigma2`, which is the only case in which
    :math:`\tau(1)=0` and the MMAR identities hold.
    """
    qa = validate_q_grid(_q_grid_input(q))
    s2 = lognormal_conservation_sigma2(lam, b) if sigma2 is None else float(sigma2)
    zeta = lognormal_tau(qa, lam, s2, b) + float(d_support)
    return ScalingExponents(
        q=qa,
        zeta=zeta,
        d_support=float(d_support),
        estimator="theory.lognormal",
        provenance=_provenance(
            "theory.lognormal_exponents",
            {"lam": float(lam), "sigma2": s2, "b": float(b), "d_support": float(d_support)},
            {"q": qa},
        ),
    )


def lognormal_spectrum(
    alpha: ArrayLike | None = None,
    *,
    lam: float = 1.05,
    sigma2: float | None = None,
    b: float = 2.0,
    n_alpha: int = 201,
    d_support: float = 1.0,
) -> HolderSpectrum:
    r"""Exact parabolic spectrum of the lognormal cascade on an :math:`\alpha` grid.

    ``alpha=None`` spans :func:`lognormal_alpha_range`, i.e. exactly the support
    on which :math:`f\ge 0`.
    """
    s2 = lognormal_conservation_sigma2(lam, b) if sigma2 is None else float(sigma2)
    if alpha is None:
        lo, hi = lognormal_alpha_range(lam, s2, b)
        a = np.linspace(lo, hi, int(n_alpha))
    else:
        a = _as_array(alpha, "alpha")
    f = lognormal_falpha(a, lam, s2, b)
    return HolderSpectrum(
        alpha=a,
        f_alpha=f,
        d_support=float(d_support),
        method="theory.lognormal",
        provenance=_provenance(
            "theory.lognormal_spectrum",
            {"lam": float(lam), "sigma2": s2, "b": float(b), "d_support": float(d_support)},
            {"alpha": a},
        ),
    )


def fbm_exponents(
    q: ArrayLike | None = None, *, H: float = 0.5, d_support: float = 1.0
) -> ScalingExponents:
    r"""Exact monofractal exponents :math:`\zeta(q)=qH`; the null model."""
    qa = validate_q_grid(_q_grid_input(q))
    return ScalingExponents(
        q=qa,
        zeta=fbm_zeta(qa, H),
        d_support=float(d_support),
        estimator="theory.fbm",
        provenance=_provenance(
            "theory.fbm_exponents",
            {"H": float(H), "d_support": float(d_support)},
            {"q": qa},
        ),
    )


def mmar_exponents_lognormal(
    q: ArrayLike | None = None,
    *,
    H: float = 0.55,
    lam: float | None = None,
    sigma2: float | None = None,
    b: float = 2.0,
    d_support: float = 1.0,
) -> ScalingExponents:
    r"""Exact MMAR exponents with lognormal trading time.

    ``lam=None`` uses the mildly intermittent default :math:`\lambda=1.05`
    typical of the MFC equity calibrations; ``sigma2=None`` then imposes
    conservation, which is what makes :math:`\tau_X(1/H)=0` hold.

    Do not push :math:`\lambda` far above 1 casually: the trading-time critical
    moment is :math:`q_{\mathrm{crit}}^\theta=\sqrt{2/(\sigma^2\ln b)}`, so in
    the :math:`X` variable it sits at :math:`q_{\mathrm{crit}}^\theta/H`, and
    once that falls below :math:`1/H` the identifying root of :math:`\tau_X`
    lies past the point where the annealed exponent stops describing a sample.
    """
    qa = validate_q_grid(_q_grid_input(q))
    lam_f = 1.05 if lam is None else float(lam)
    s2 = lognormal_conservation_sigma2(lam_f, b) if sigma2 is None else float(sigma2)
    zeta = mmar_tau_lognormal(qa, H, lam_f, s2, b) + float(d_support)
    return ScalingExponents(
        q=qa,
        zeta=zeta,
        d_support=float(d_support),
        estimator="theory.mmar_lognormal",
        provenance=_provenance(
            "theory.mmar_exponents_lognormal",
            {
                "H": float(H),
                "lam": lam_f,
                "sigma2": s2,
                "b": float(b),
                "d_support": float(d_support),
            },
            {"q": qa},
        ),
    )


def mmar_spectrum_lognormal(
    alpha: ArrayLike | None = None,
    *,
    H: float = 0.55,
    lam: float | None = None,
    sigma2: float | None = None,
    b: float = 2.0,
    n_alpha: int = 201,
    d_support: float = 1.0,
) -> HolderSpectrum:
    r"""Exact MMAR spectrum with lognormal trading time; support scaled by :math:`H`.

    Defaults match :func:`mmar_exponents_lognormal`; the apex sits at
    :math:`\alpha_0^X = H\lambda` and the support is
    :math:`H\cdot` :func:`lognormal_alpha_range`.
    """
    h = float(H)
    lam_f = 1.05 if lam is None else float(lam)
    s2 = lognormal_conservation_sigma2(lam_f, b) if sigma2 is None else float(sigma2)
    if alpha is None:
        lo, hi = lognormal_alpha_range(lam_f, s2, b)
        a = np.linspace(h * lo, h * hi, int(n_alpha))
    else:
        a = _as_array(alpha, "alpha")
    f = mmar_falpha_lognormal(a, h, lam_f, s2, b)
    return HolderSpectrum(
        alpha=a,
        f_alpha=f,
        d_support=float(d_support),
        method="theory.mmar_lognormal",
        provenance=_provenance(
            "theory.mmar_spectrum_lognormal",
            {
                "H": h,
                "lam": lam_f,
                "sigma2": s2,
                "b": float(b),
                "d_support": float(d_support),
            },
            {"alpha": a},
        ),
    )
