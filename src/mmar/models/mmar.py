r"""The Mandelbrot--Fisher--Calvet Multifractal Model of Asset Returns.

The model (Mandelbrot, Fisher and Calvet 1997, *A Multifractal Model of Asset
Returns*, Cowles Foundation DP 1164) is a **subordination**: the log-price is a
fractional Brownian motion evaluated along a random, multifractal clock,

.. math::
    X(t) = B_H\big(\theta(t)\big), \qquad P(t) = P(0)\,e^{X(t)},

where

* :math:`\theta` --- *trading time* --- is the cumulative distribution function
  of a multifractal measure :math:`\mu` on :math:`[0,T]`, a non-decreasing
  singular function with :math:`\theta(0)=0`, :math:`\theta(T)=T`;
* :math:`B_H` is a fractional Brownian motion of Hurst index :math:`H`,
  **independent of** :math:`\theta`.

The independence is a modelling assumption, not a derived fact. It is what
makes the exponents composable, and it also means the model cannot generate a
leverage effect: sign and volatility are independent by construction. Anything
resembling an asymmetric smile must come from elsewhere.

Scaling
-------
Self-similarity of fBm gives :math:`|B_H(s+\Delta s)-B_H(s)|
\stackrel{d}{=}|\Delta s|^{H}|B_H(1)|`; conditioning on the clock,

.. math::
    S_X(q,\Delta t) = \mathbb{E}\big|X(t+\Delta t)-X(t)\big|^q
      = \mathbb{E}\big|B_H(1)\big|^q \;
        \mathbb{E}\big[(\Delta\theta)^{qH}\big].

If the trading time obeys :math:`\mathbb{E}[(\Delta\theta)^{p}] \sim
(\Delta t)^{\tau_\theta(p)+1}` then

.. math::
    \zeta_X(q) = \tau_\theta(qH) + 1, \qquad
    \tau_X(q) = \zeta_X(q) - 1 = \tau_\theta(qH),

which is the MFC "trading-time" theorem. Setting :math:`q=1/H` and using
:math:`\tau_\theta(1)=0` (mass conservation) recovers the identity
:math:`\tau_X(1/H)=0` used by
:attr:`~mmar.core.types.ScalingExponents.hurst_tau_root`.

Lognormal closure
-----------------
For a base-:math:`b` lognormal cascade with multiplier :math:`M=b^{-V}`,
:math:`V\sim\mathcal N(\lambda,\sigma^2)`, one has
:math:`\tau_\theta(p) = -\log_b \mathbb{E}[M^p] - 1
= p\lambda - \tfrac{\sigma^2\ln b}{2}p^2 - 1`, hence

.. math::
    \tau_X(q) = H\lambda q - \tfrac{\sigma^2 \ln b}{2}H^2q^2 - 1 ,
    \qquad
    \zeta_X(q) = H\lambda q - \tfrac{\sigma^2 \ln b}{2}H^2q^2 .

Conservation :math:`\tau_\theta(1)=0` closes the parameters:
:math:`\sigma^2 = 2(\lambda-1)/\ln b`, whereupon
:math:`\tfrac{\sigma^2\ln b}{2} = \lambda-1` and the modal Hölder exponent of
the price is :math:`\alpha_0 = \tau_X'(0) = H\lambda`. Those two identities are
the whole of :func:`calibrate_mmar`.

Admissibility (all checked by :meth:`MMARSpec.validate`):

======================  ==========================================
:math:`\lambda>1`       :math:`\sigma^2>0`; otherwise no cascade.
:math:`\lambda<2`       Kahane--Peyrière non-degeneracy,
                        :math:`\tau_\theta'(1)=2-\lambda>0`.
:math:`H\lambda<1`      :math:`\alpha_0<1`: the log-price is
                        nowhere differentiable, as a price should be.
:math:`q<1/(\lambda-1)` Critical moment of the cascade; moments of
                        :math:`\Delta\theta` diverge beyond it, so
                        :math:`\zeta_X(q)` is only meaningful for
                        :math:`qH < 1/(\lambda-1)`.
======================  ==========================================

The composition is the model
---------------------------
Simulating :math:`\theta` and :math:`B_H` and then *not* composing them --- for
instance plotting :math:`P(0)e^{B_H(t)}` against a trading-time abscissa ---
produces a plain fBm with a relabelled axis. It is exactly monofractal:
:math:`h(q)\equiv H`, Gaussian increments, zero spectrum width. The entire
multifractality of the MMAR lives in :math:`B_H\circ\theta`, and
:func:`simulate_mmar` therefore evaluates the fBm *at the trading-time
abscissae* by interpolation on a fine uniform grid in trading-time units.
``tests/theory/test_mmar_sim.py::test_composition_creates_multifractality``
is the regression guard.

Discretisation bias
-------------------
The fBm is sampled on a uniform grid of spacing :math:`\Delta u = T/m` in
trading time and read off at :math:`\theta(t_i)` by linear interpolation.
Whenever :math:`\Delta\theta_i \ll \Delta u` --- and a cascade produces
arbitrarily small cells --- the increment is a *chord* of the fBm rather than
an fBm increment, so it scales like :math:`\Delta\theta_i^{1}` instead of
:math:`\Delta\theta_i^{H}`. The effect biases the smooth (large-:math:`\alpha`)
side of the spectrum and attenuates :math:`|c_2|`. ``oversample`` raises
:math:`m` and shrinks the affected range; it cannot remove it, because the
exact model has no finest scale.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mmar.core.provenance import InputRef, Provenance, new_provenance
from mmar.core.rng import SeedLike, seed_info, spawn_rngs
from mmar.core.types import FloatArray
from mmar.core.validate import validate_measure, validate_series
from mmar.models import cascade as _cascade_mod
from mmar.models import theory as _theory

# Imported from the submodule, not the package: mmar/models/__init__.py exports
# the `fbm` FUNCTION under that name, which shadows the `mmar.models.fbm`
# submodule attribute. `from mmar.models import fbm` therefore yields a callable,
# and `_fbm_mod.fbm` raises AttributeError.
from mmar.models.fbm import fbm as _fbm

__all__ = [
    "MMARForecast",
    "MMARPath",
    "MMARSpec",
    "calibrate_mmar",
    "default_oversample",
    "mmar_forecast_distribution",
    "mmar_zeta_lognormal",
    "simulate_ensemble",
    "simulate_mmar",
]

_VERSION = "1"

T_MAX: float = 1.0
"""Physical time is normalised to :math:`[0,1]`; a horizon is a fraction of it."""

DEFAULT_OVERSAMPLE = 8
"""fBm grid points per physical sample. See *Discretisation bias* above."""

DEFAULT_QUANTILES: tuple[float, ...] = (
    0.001,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    0.999,
)


# --------------------------------------------------------------------------- #
# Closed-form exponents
# --------------------------------------------------------------------------- #


def _tau_of_spec(
    q: FloatArray | Sequence[float] | float, *, H: float, lam: float, sigma2: float, b: int
) -> FloatArray:
    r"""Measure-view :math:`\tau_X(q)`, delegating to the closed-form oracle.

    :math:`\tau_X(q)=\tau_\theta(Hq)=H\lambda q-\tfrac{\sigma^2\ln b}{2}H^2q^2-1`
    via :func:`mmar.models.theory.mmar_tau_lognormal`, with the degenerate
    :math:`\sigma^2=0` branch routed to :func:`mmar.models.theory.fbm_tau`
    (the oracle rightly refuses a zero-variance "lognormal" cascade: at
    :math:`\sigma^2=0` the trading time is Lebesgue and the model *is* fBm).
    """
    if sigma2 == 0.0:
        return np.asarray(_theory.fbm_tau(q, H), dtype=np.float64)
    return np.asarray(_theory.mmar_tau_lognormal(q, H, lam, sigma2, float(b)), dtype=np.float64)


def mmar_zeta_lognormal(
    q: FloatArray | Sequence[float] | float,
    *,
    H: float,
    lam: float,
    sigma2: float,
    b: int = 2,
) -> FloatArray:
    r"""Signal-view exponent :math:`\zeta_X(q)=\tau_X(q)+d` with :math:`d=1`.

    :math:`S_q(\Delta t)=\langle|\Delta_{\Delta t}X|^q\rangle\sim
    \Delta t^{\zeta_X(q)}`, so :math:`h(q)=\zeta_X(q)/q = H\lambda -
    \tfrac{\sigma^2\ln b}{2}H^2 q` is affine and *decreasing* in :math:`q` --- a
    non-constant :math:`h(q)` is the operational signature of the model, and its
    slope :math:`-\tfrac{\sigma^2\ln b}{2}H^2` is what the composition
    :math:`B_H\circ\theta` creates and an uncomposed fBm does not have.

    This returns :math:`\zeta`, not :math:`\tau`; the two differ by
    :math:`d=1`.
    """
    return np.asarray(_tau_of_spec(q, H=H, lam=lam, sigma2=sigma2, b=b) + 1.0, dtype=np.float64)


def calibrate_mmar(alpha_0: float, H: float, *, b: int = 2) -> tuple[float, float]:
    r"""Invert the MFC identities: :math:`(\alpha_0, H) \mapsto (\lambda,\sigma^2)`.

    .. math::
        \lambda = \frac{\alpha_0}{H}, \qquad
        \sigma^2 = \frac{2(\lambda-1)}{\ln b}.

    The first is :math:`\alpha_0=\tau_X'(0)=H\lambda`, the modal Hölder exponent
    of the log-price; the second is mass conservation
    :math:`\tau_\theta(1)=0`.

    Parameters
    ----------
    alpha_0
        Modal Hölder exponent of the log-price, i.e. :math:`\arg\max_\alpha
        f(\alpha)` from an estimated spectrum.
    H
        Hurst index of the subordinand fBm, i.e. :math:`1/q^*` with
        :math:`\hat\tau(q^*)=0` (see
        :attr:`~mmar.core.types.ScalingExponents.hurst_tau_root`).

    Returns
    -------
    (lam, sigma2)
        Cascade parameters for :func:`mmar.models.cascade.lognormal_cascade`.

    Raises
    ------
    ValueError
        If :math:`\alpha_0 \le H`, i.e. :math:`\lambda\le1` and
        :math:`\sigma^2\le0`. This is a *diagnosis*, not an arithmetic
        accident: subordination can only push the modal exponent up
        (:math:`\alpha_0=H\lambda` with :math:`\lambda>1`), so an estimated
        :math:`\alpha_0` at or below :math:`H` says the two estimates are
        mutually inconsistent with any lognormal MMAR.
    """
    if not math.isfinite(alpha_0) or alpha_0 <= 0.0:
        raise ValueError(f"alpha_0 must be a positive finite number, got {alpha_0!r}")
    if not (0.0 < H < 1.0):
        raise ValueError(f"H must lie strictly in (0, 1) for an fBm subordinand, got {H!r}")
    if b < 2:
        raise ValueError(f"cascade base b must be at least 2, got {b}")

    lam = float(_theory.mmar_lambda_from_alpha0(alpha_0, H))
    sigma2 = 2.0 * (lam - 1.0) / math.log(b)

    if lam <= 1.0:
        raise ValueError(
            "inconsistent MMAR calibration: alpha_0 <= H.\n"
            f"  alpha_0 = {alpha_0:.6g}  (modal Holder exponent of the log-price)\n"
            f"  H       = {H:.6g}        (Hurst index of the subordinand fBm)\n"
            f"  lambda  = alpha_0 / H = {lam:.6g}   (needs lambda > 1)\n"
            f"  sigma^2 = 2(lambda-1)/ln b = {sigma2:.6g}   (needs sigma^2 > 0)\n"
            "In the MMAR the trading time inflates the modal exponent, alpha_0 = H*lambda\n"
            "with lambda > 1, so alpha_0 must exceed H. Getting alpha_0 <= H means one of:\n"
            "  (a) H was over-estimated -- tau(q) has a shallow root, so 1/q* is unstable;\n"
            "      re-estimate H from the q-range where tau actually changes sign;\n"
            "  (b) alpha_0 was under-estimated -- a truncated positive-q grid clips the\n"
            "      apex of f(alpha), pulling alpha_0 down;\n"
            "  (c) the series is monofractal (lambda = 1 exactly) or anti-multifractal,\n"
            "      in which case a lognormal MMAR is the wrong model, not a mis-fitted one.\n"
            "Refusing to return a negative cascade variance."
        )
    sigma2 = float(_theory.lognormal_conservation_sigma2(lam, float(b)))
    if lam >= 2.0:
        warnings.warn(
            f"lambda = {lam:.4g} >= 2: the limit lognormal cascade is degenerate "
            f"(Kahane-Peyriere: non-degeneracy needs tau_theta'(1) = 2 - lambda > 0). "
            f"A finite-depth simulation still returns numbers, but they describe the "
            f"pre-limit measure at that depth, not the MMAR. Implied sigma^2 = {sigma2:.4g}.",
            UserWarning,
            stacklevel=2,
        )
    return lam, sigma2


# --------------------------------------------------------------------------- #
# Specification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MMARSpec:
    r"""Parameters of a lognormal MMAR on the physical interval :math:`[0,1]`.

    Attributes
    ----------
    kmax
        Cascade depth; the grid has :math:`n=b^{k_{\max}}` cells.
    H
        Hurst index of the subordinand fBm, :math:`0<H<1`.
    lam
        Cascade location parameter :math:`\lambda` of
        :math:`V\sim\mathcal N(\lambda,\sigma^2)`, :math:`M=b^{-V}`. Fixes the
        modal Hölder exponent through :math:`\alpha_0=H\lambda`.
    sigma2
        Cascade variance. ``None`` (the default) applies the conservation
        closure :math:`\sigma^2=2(\lambda-1)/\ln b`, which is the only choice
        with :math:`\tau_\theta(1)=0` and hence the only one for which the MFC
        identity :math:`\tau_X(1/H)=0` holds.
    b
        Branching number of the cascade.
    price0
        :math:`P(0)>0`.
    conservative
        Passed through to the cascade: microcanonical (exactly mass-conserving)
        rather than canonical construction. **This invalidates the closed forms
        below.** The sibling rescaling caps the largest cell masses, so the
        realised :math:`\tau_\theta(p)` bends above the parabola for
        :math:`p>1` while :attr:`theoretical_tau` keeps returning the canonical
        one; :meth:`validate` warns, with numbers.
    """

    kmax: int
    H: float
    lam: float
    sigma2: float | None = None
    b: int = 2
    price0: float = 1.0
    conservative: bool = False

    # ---- derived quantities ------------------------------------------------ #

    @property
    def n(self) -> int:
        """Number of finest-level cascade cells, :math:`b^{k_{\\max}}`."""
        return int(self.b**self.kmax)

    @property
    def sigma2_effective(self) -> float:
        r""":math:`\sigma^2`, resolved through the conservation closure if unset."""
        if self.sigma2 is not None:
            return float(self.sigma2)
        return 2.0 * (self.lam - 1.0) / math.log(self.b)

    @property
    def alpha_0(self) -> float:
        r"""Modal Hölder exponent of the log-price, :math:`\alpha_0=H\lambda`.

        Exact only under the conservation closure; with an explicit
        ``sigma2`` the apex still sits at :math:`\tau_X'(0)=H\lambda` but
        :math:`f(\alpha_0)\neq d` because the measure is not normalised in
        expectation.
        """
        return float(self.H * self.lam)

    @property
    def critical_moment(self) -> float:
        r"""Order beyond which :math:`\mathbb E|\Delta X|^q` diverges.

        Kahane--Peyrière: the limit cascade mass has finite :math:`p`-th moment
        iff :math:`\tau_\theta(p)>0`, i.e. :math:`p<p_{\rm crit}=1/(\lambda-1)`
        (the second root of :math:`\tau_\theta`, the first being :math:`p=1`).
        Since :math:`\Delta\theta` enters at order :math:`p=qH`,
        :math:`q_{\rm crit}=1/(H(\lambda-1))`. Infinite for :math:`\lambda\le1`.
        """
        gap = self.lam - 1.0
        if gap <= 0.0:
            return float("inf")
        return float(1.0 / (self.H * gap))

    @property
    def linearisation_moment(self) -> float:
        r"""Order at which a *single sample*'s :math:`\hat\zeta_X` stops being parabolic.

        The binding constraint in practice, and much smaller than
        :attr:`critical_moment`. Above the :math:`q` where
        :math:`f_X(\alpha(q))=0` the partition function is dominated by cells
        that almost surely do not occur in one realisation, so the empirical
        :math:`\hat\tau` follows the tangent from the origin (it *linearises*)
        rather than the parabola (Mandelbrot 1974; Molchan 1996). With
        :math:`f_X(\alpha(q))=1-\tfrac{\sigma^2\ln b}{2}H^2q^2`,

        .. math::
            q_{\rm lin} = \frac{1}{H}\sqrt{\frac{2}{\sigma^2\ln b}} .

        Fitting :math:`\hat\zeta` beyond it does not fail loudly; it just
        returns a straight line and a triangular spectrum.
        """
        curvature = 0.5 * self.sigma2_effective * math.log(self.b)
        if curvature <= 0.0:
            return float("inf")
        return float(1.0 / (self.H * math.sqrt(curvature)))

    @property
    def theoretical_tau(self) -> Callable[[FloatArray | Sequence[float] | float], FloatArray]:
        r"""Closed-form :math:`q\mapsto\tau_X(q)` (measure view, :math:`\tau=\zeta-d`).

        Thin binding of :func:`mmar.models.theory.mmar_tau_lognormal` to this
        specification's parameters. Anchors: :math:`\tau_X(0)=-1` and, under the
        conservation closure, :math:`\tau_X(1/H)=0`.

        Describes the **canonical** cascade. With ``conservative=True`` the
        simulated measure is microcanonical and this function no longer
        describes it; see the ``conservative`` attribute.
        """
        H, lam, sigma2, b = self.H, self.lam, self.sigma2_effective, self.b

        def tau(q: FloatArray | Sequence[float] | float) -> FloatArray:
            return _tau_of_spec(q, H=H, lam=lam, sigma2=sigma2, b=b)

        return tau

    @property
    def theoretical_zeta(self) -> Callable[[FloatArray | Sequence[float] | float], FloatArray]:
        r"""Closed-form :math:`q\mapsto\zeta_X(q)` (signal view, :math:`S_q\sim\delta^{\zeta}`)."""
        H, lam, sigma2, b = self.H, self.lam, self.sigma2_effective, self.b

        def zeta(q: FloatArray | Sequence[float] | float) -> FloatArray:
            return np.asarray(_tau_of_spec(q, H=H, lam=lam, sigma2=sigma2, b=b) + 1.0)

        return zeta

    # ---- validation -------------------------------------------------------- #

    def validate(self) -> None:
        """Reject inadmissible parameters; warn on admissible-but-pathological ones.

        Raises
        ------
        ValueError
            Non-integer or out-of-range structural parameters, :math:`H\\notin(0,1)`,
            non-positive ``price0``, or a derived :math:`\\sigma^2\\le0`.
        """
        if int(self.kmax) != self.kmax or self.kmax < 1:
            raise ValueError(f"kmax must be a positive integer, got {self.kmax!r}")
        if int(self.b) != self.b or self.b < 2:
            raise ValueError(f"cascade base b must be an integer >= 2, got {self.b!r}")
        if not (0.0 < self.H < 1.0):
            raise ValueError(
                f"H must lie strictly in (0, 1): H={self.H!r}. H=1/2 is the Brownian case; "
                "H outside (0,1) is not a fractional Brownian motion."
            )
        if not math.isfinite(self.price0) or self.price0 <= 0.0:
            raise ValueError(f"price0 must be strictly positive, got {self.price0!r}")
        if not math.isfinite(self.lam):
            raise ValueError(f"lam must be finite, got {self.lam!r}")

        if self.sigma2 is None:
            if self.lam <= 1.0:
                raise ValueError(
                    "the conservation closure sigma^2 = 2(lambda-1)/ln b requires lambda > 1.\n"
                    f"  lambda  = {self.lam:.6g}\n"
                    f"  sigma^2 = {2.0 * (self.lam - 1.0) / math.log(self.b):.6g} (must be > 0)\n"
                    f"  implied alpha_0 = H*lambda = {self.H * self.lam:.6g} "
                    f"(must exceed H = {self.H:.6g})\n"
                    "lambda == 1 is the monofractal boundary: the cascade is deterministic and "
                    "X reduces to the fBm itself. Pass sigma2=0.0 explicitly if that degenerate "
                    "null is what you want; otherwise raise lambda (equivalently, alpha_0)."
                )
        else:
            if not math.isfinite(self.sigma2) or self.sigma2 < 0.0:
                raise ValueError(f"sigma2 must be finite and non-negative, got {self.sigma2!r}")
            closure = 2.0 * (self.lam - 1.0) / math.log(self.b)
            if abs(self.sigma2 - closure) > 1e-9 * max(1.0, abs(closure)):
                warnings.warn(
                    f"sigma2={self.sigma2:.6g} departs from the conservation closure "
                    f"2(lambda-1)/ln b = {closure:.6g}, so tau_theta(1) = "
                    f"{self.lam - 0.5 * self.sigma2 * math.log(self.b) - 1.0:+.4g} != 0: the "
                    "cascade does not conserve mass in expectation and the MFC identity "
                    "tau_X(1/H) = 0 no longer holds. Only do this deliberately.",
                    UserWarning,
                    stacklevel=2,
                )

        if self.lam >= 2.0:
            warnings.warn(
                f"lambda = {self.lam:.4g} >= 2 gives tau_theta'(1) = {2.0 - self.lam:.4g} <= 0: "
                "by Kahane-Peyriere the limit measure degenerates to zero. A depth-kmax "
                "simulation returns the pre-limit measure, whose exponents drift with kmax.",
                UserWarning,
                stacklevel=2,
            )
        if self.alpha_0 >= 1.0:
            warnings.warn(
                f"alpha_0 = H*lambda = {self.alpha_0:.4g} >= 1: the log-price is Holder-1 or "
                "smoother almost everywhere, i.e. locally differentiable. Financially this is "
                "the wrong regime -- empirical alpha_0 sits near 0.5.",
                UserWarning,
                stacklevel=2,
            )
        if self.conservative:
            warnings.warn(
                "conservative=True builds the MICROCANONICAL cascade (siblings rescaled to "
                "the parent's mass), whose tau_theta is not the canonical closed form: "
                "capping the sibling masses truncates the right tail, so the realised "
                "tau_theta(p) bends ABOVE the parabola for p > 1 and theoretical_tau / "
                "theoretical_zeta / alpha_0 / critical_moment describe a different measure "
                "from the one simulate_mmar will draw. Measured at kmax=12, lambda=1.18: "
                "tau_theta(2) is 0.64 canonical, 0.86 microcanonical. Use conservative=False "
                "whenever the closed forms are the point.",
                UserWarning,
                stacklevel=2,
            )
        if self.n > 2**24:
            warnings.warn(
                f"n = b^kmax = {self.n:.3g} cells; each simulated path allocates several arrays "
                "of that length (times the fBm oversampling factor).",
                UserWarning,
                stacklevel=2,
            )

    def to_canonical(self) -> dict[str, Any]:
        """Digest-stable parameter record for :mod:`mmar.core.provenance`."""
        return {
            "kmax": int(self.kmax),
            "H": repr(float(self.H)),
            "lam": repr(float(self.lam)),
            "sigma2": None if self.sigma2 is None else repr(float(self.sigma2)),
            "b": int(self.b),
            "price0": repr(float(self.price0)),
            "conservative": bool(self.conservative),
        }


# --------------------------------------------------------------------------- #
# Simulation products
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MMARPath:
    r"""One realisation of :math:`X=B_H\circ\theta` on a uniform physical grid.

    All four arrays share the same length :math:`N=n+1` and index the same
    physical instants ``t``; ``theta[i]`` is the trading time elapsed by
    ``t[i]``, and ``x[i] = B_H(theta[i])`` with :math:`B_H(0)=0`.
    """

    t: FloatArray
    theta: FloatArray
    x: FloatArray
    price: FloatArray
    spec: MMARSpec
    provenance: Provenance

    def __post_init__(self) -> None:
        shapes = {a.shape for a in (self.t, self.theta, self.x, self.price)}
        if len(shapes) != 1:
            raise ValueError(f"t, theta, x, price must share one shape, got {shapes}")

    @property
    def n_points(self) -> int:
        return int(self.t.size)

    @property
    def log_returns(self) -> FloatArray:
        r""":math:`\Delta X` on the uniform physical grid: the MMAR's returns."""
        return np.asarray(np.diff(self.x), dtype=np.float64)

    @property
    def trading_time_increments(self) -> FloatArray:
        r""":math:`\Delta\theta`: the realised cascade measure of each cell."""
        return np.asarray(np.diff(self.theta), dtype=np.float64)

    def at(self, time: float | FloatArray) -> FloatArray:
        """Price interpolated at arbitrary physical time(s) inside ``[0, T]``."""
        ta = np.asarray(time, dtype=np.float64)
        if np.any(ta < self.t[0]) or np.any(ta > self.t[-1]):
            raise ValueError(
                f"time outside the simulated span [{self.t[0]:g}, {self.t[-1]:g}]; "
                "extrapolating a subordinated path is not defined"
            )
        return np.asarray(np.interp(ta, self.t, self.price), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class MMARForecast:
    r"""Monte Carlo terminal-price law at a physical horizon.

    Replaces ad-hoc "probability analyser" code: every number here is a
    property of an ensemble whose parent seed is recorded in ``provenance``.
    """

    horizon: float
    n_paths: int
    price0: float
    terminal_price: FloatArray
    terminal_log_return: FloatArray
    quantile_levels: FloatArray
    quantiles: FloatArray
    mean: float
    std: float
    tail_index_upper: float
    tail_index_lower: float
    hill_fraction: float
    spec: MMARSpec
    provenance: Provenance

    @property
    def median(self) -> float:
        return float(np.median(self.terminal_price))

    def quantile(self, level: float) -> float:
        """Empirical quantile of the terminal price at an arbitrary level."""
        if not (0.0 < level < 1.0):
            raise ValueError(f"quantile level must lie in (0, 1), got {level!r}")
        return float(np.quantile(self.terminal_price, level))

    def probability_below(self, price: float) -> float:
        """:math:`\\widehat{\\Pr}[P(\\text{horizon}) \\le \\text{price}]`."""
        return float(np.mean(self.terminal_price <= price))

    def summary(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "n_paths": self.n_paths,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "tail_index_upper": self.tail_index_upper,
            "tail_index_lower": self.tail_index_lower,
            "root": self.provenance.root,
        }


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


def _cumulative_trading_time(mu: FloatArray, *, t_max: float) -> FloatArray:
    r"""Trading time :math:`\theta` from the cascade measure, anchored at 0.

    Delegates to :func:`mmar.models.cascade.trading_time` and re-asserts the
    invariants the subordination depends on: ``theta[0] == 0``,
    ``theta[-1] == t_max``, non-decreasing, length ``mu.size + 1``. Also accepts
    the ``n``-point convention (CDF without its leading zero) so that a change
    of convention upstream is absorbed here rather than silently shifting the
    clock by one cell.
    """
    theta = np.asarray(_cascade_mod.trading_time(mu, t_max=t_max), dtype=np.float64).ravel()
    if theta.size == mu.size:
        theta = np.concatenate((np.zeros(1, dtype=np.float64), theta))
    if theta.size != mu.size + 1:
        raise ValueError(
            f"trading_time returned {theta.size} points for a measure of {mu.size} cells; "
            "expected n or n+1 (the CDF with or without its leading zero)"
        )
    if np.any(np.diff(theta) < -1e-12 * t_max):
        raise ValueError(
            "trading time is not non-decreasing; the cascade measure has negative mass"
        )
    if not math.isclose(float(theta[0]), 0.0, abs_tol=1e-12 * max(t_max, 1.0)):
        raise ValueError(f"trading time must start at 0, got theta[0]={float(theta[0]):g}")
    if float(theta[-1]) <= 0.0:
        raise ValueError("trading time has zero total; the cascade measure collapsed")
    return theta


def _fbm_on_grid(
    n_steps: int, H: float, *, rng: np.random.Generator, span: float
) -> tuple[FloatArray, FloatArray]:
    r"""Sample :math:`B_H` on a uniform grid of :math:`[0,\text{span}]`; returns ``(u, B)``.

    :func:`mmar.models.fbm.fbm` returns the ``n_steps`` values
    :math:`B_H(i\,\text{span}/n)` for :math:`i=1,\dots,n` and deliberately omits
    :math:`B_H(0)=0`. The origin must be restored rather than inferred: dropping
    it would shift the whole grid by one step, so :math:`\theta(t)=0` would read
    off :math:`B_H(\text{span}/n)` and the path would not start at its own
    initial price.
    """
    inner = np.asarray(_fbm(n_steps, H, seed=rng, t_max=span), dtype=np.float64).ravel()
    if inner.size != n_steps:
        raise ValueError(f"fbm returned {inner.size} points, expected {n_steps}")
    path = np.concatenate((np.zeros(1, dtype=np.float64), inner))
    u = np.linspace(0.0, span, path.size, dtype=np.float64)
    return u, path


def _simulate_path(
    spec: MMARSpec,
    *,
    rng_cascade: np.random.Generator,
    rng_fbm: np.random.Generator,
    oversample: int,
    seed_record: Any,
    extra_params: dict[str, Any],
) -> MMARPath:
    """Draw one path. The two generators must come from independent streams."""
    mu_raw = _cascade_mod.lognormal_cascade(
        spec.kmax,
        spec.lam,
        spec.sigma2,
        b=spec.b,
        seed=rng_cascade,
        conservative=spec.conservative,
    )
    mu = validate_measure(mu_raw, name="cascade measure", normalise=True)
    theta = _cumulative_trading_time(mu, t_max=T_MAX)

    n_points = theta.size
    n_steps = oversample * (n_points - 1)
    u, b_path = _fbm_on_grid(n_steps, spec.H, rng=rng_fbm, span=float(theta[-1]))

    # The subordination itself: X(t_i) = B_H(theta(t_i)). Clipping only absorbs
    # float64 round-off at the right endpoint; theta lies in [0, theta[-1]] by
    # construction. THIS LINE IS THE MODEL -- without it the output is an fBm.
    x = np.asarray(np.interp(np.clip(theta, u[0], u[-1]), u, b_path), dtype=np.float64)
    price = np.asarray(spec.price0 * np.exp(x), dtype=np.float64)
    t = np.linspace(0.0, T_MAX, n_points, dtype=np.float64)

    validate_series(x, name="simulated log-price", min_length=16, allow_constant=False)

    d_theta = np.diff(theta)
    grid_spacing = float(u[1] - u[0])
    median_cell = float(np.median(d_theta))
    if grid_spacing > median_cell:
        warnings.warn(
            f"fBm grid spacing {grid_spacing:.3g} exceeds the median cascade cell mass "
            f"{median_cell:.3g}: more than half the increments are linear chords of the "
            f"fBm rather than fBm increments, so they scale like d(theta)^1 instead of "
            f"d(theta)^{spec.H:g}. This biases the large-alpha side of the spectrum and "
            f"shrinks |c_2|. Raise oversample (currently {oversample}) to at least "
            f"{max(1, math.ceil(oversample * grid_spacing / median_cell))}, or lower kmax.",
            UserWarning,
            stacklevel=3,
        )

    prov = new_provenance(
        "mmar.simulate",
        estimator_version=_VERSION,
        params={
            "spec": spec.to_canonical(),
            "oversample": int(oversample),
            "t_max": repr(T_MAX),
            "n_points": int(n_points),
            "fbm_grid_points": int(u.size),
            **extra_params,
        },
        inputs=(
            InputRef.from_array("trading_time", theta),
            InputRef.from_array("fbm_grid", b_path),
        ),
        seed=seed_record,
        notes=(
            "X = B_H(theta(t)); B_H and theta drawn from independent spawned streams.",
            "fBm read at theta(t_i) by linear interpolation on a uniform trading-time grid.",
        ),
    ).with_diagnostics(
        trading_time_total=float(theta[-1]),
        min_cell_mass=float(np.min(d_theta)),
        median_cell_mass=median_cell,
        max_cell_mass=float(np.max(d_theta)),
        fbm_grid_spacing=grid_spacing,
        chord_fraction=float(np.mean(d_theta < grid_spacing)),
        alpha_0=spec.alpha_0,
        critical_moment=spec.critical_moment,
        linearisation_moment=spec.linearisation_moment,
    )

    return MMARPath(t=t, theta=theta, x=x, price=price, spec=spec, provenance=prov)


def default_oversample(spec: MMARSpec, *, safety: float = 2.5, cap: int = 64) -> int:
    r"""fBm grid refinement that keeps the chord-interpolation bias out of the path.

    :math:`B_H` is sampled on a uniform grid in trading time and read off at
    :math:`\theta(t_i)`. Wherever a cascade cell is smaller than the grid
    spacing, the increment returned is a linear *chord* of the fBm rather than
    an fBm increment, so it scales like :math:`\Delta\theta^{1}` instead of
    :math:`\Delta\theta^{H}` --- biasing the large-:math:`\alpha` side of the
    spectrum and shrinking :math:`|c_2|`.

    The median cell mass of a conservative lognormal cascade at depth :math:`k`
    is :math:`\prod_{j\le k}\operatorname{med}(M) = b^{-\lambda k}`, while the
    grid spacing is :math:`b^{-k}/\text{oversample}`. Requiring the spacing to
    sit below the median cell gives

    .. math::  \text{oversample} \;>\; b^{\,k(\lambda-1)} ,

    and ``safety`` covers the half of the cells below the median. The result is
    capped: memory is linear in it, and at large ``kmax`` no affordable grid
    resolves the smallest cells. When the cap binds the runtime warning still
    fires, which is the honest signal that the path is chord-contaminated.
    """
    exponent = spec.kmax * max(spec.lam - 1.0, 0.0)
    needed = safety * float(spec.b) ** exponent
    return int(min(cap, max(DEFAULT_OVERSAMPLE, math.ceil(needed))))


def simulate_mmar(spec: MMARSpec, *, seed: SeedLike, oversample: int | None = None) -> MMARPath:
    r"""Simulate one MMAR path :math:`X=B_H\circ\theta` on :math:`[0,1]`.

    The cascade and the fBm are drawn from two children of ``seed`` obtained
    with :func:`~mmar.core.rng.spawn_rngs`. Their **independence is a modelling
    assumption of the MMAR**, not a convenience: it is what allows
    :math:`\mathbb{E}|\Delta X|^q = \mathbb{E}|B_H(1)|^q\,
    \mathbb{E}(\Delta\theta)^{qH}` and hence
    :math:`\tau_X(q)=\tau_\theta(qH)`. Spawning (rather than incrementing an
    integer seed) is what makes the two streams provably non-overlapping.

    Parameters
    ----------
    spec
        Validated on entry; see :meth:`MMARSpec.validate`.
    seed
        Parent seed. Recorded in the provenance, so the path is reproducible
        bit-for-bit from the manifest alone.
    oversample
        fBm grid points per physical sample. ``None`` (the default) derives it
        from the spec via :func:`default_oversample`, which is what keeps the
        chord-interpolation bias out of the path; a fixed 8 warned on the
        library's own documented settings at every depth, and a warning that
        always fires is a warning nobody reads. Larger values cost linear
        memory.

    Returns
    -------
    MMARPath
        With ``t``, ``theta``, ``x`` (log-price) and ``price``, all of length
        :math:`b^{k_{\max}}+1`.
    """
    spec.validate()
    oversample = default_oversample(spec) if oversample is None else oversample
    if int(oversample) != oversample or oversample < 1:
        raise ValueError(f"oversample must be a positive integer, got {oversample!r}")
    rng_cascade, rng_fbm = spawn_rngs(seed, 2)
    return _simulate_path(
        spec,
        rng_cascade=rng_cascade,
        rng_fbm=rng_fbm,
        oversample=int(oversample),
        seed_record=seed_info(seed),
        extra_params={},
    )


def simulate_ensemble(
    spec: MMARSpec,
    n_paths: int,
    *,
    seed: SeedLike,
    oversample: int | None = None,
) -> list[MMARPath]:
    r"""``n_paths`` independent MMAR paths from a single parent seed.

    Each path draws its own (cascade, fBm) pair from a distinct spawned
    sub-stream, so the paths are mutually independent *and* the whole ensemble
    is reproducible from ``seed``. Each path's provenance records the parent
    seed together with its ``path_index``, which is sufficient to regenerate
    that path alone.
    """
    spec.validate()
    if n_paths < 1:
        raise ValueError(f"n_paths must be at least 1, got {n_paths}")
    oversample = default_oversample(spec) if oversample is None else oversample
    if int(oversample) != oversample or oversample < 1:
        raise ValueError(f"oversample must be a positive integer, got {oversample!r}")

    record = seed_info(seed)
    parents = spawn_rngs(seed, n_paths)
    paths: list[MMARPath] = []
    for i, parent in enumerate(parents):
        rng_cascade, rng_fbm = spawn_rngs(parent, 2)
        paths.append(
            _simulate_path(
                spec,
                rng_cascade=rng_cascade,
                rng_fbm=rng_fbm,
                oversample=int(oversample),
                seed_record=record,
                extra_params={"path_index": i, "n_paths": int(n_paths)},
            )
        )
    return paths


# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #


_HILL_MIN_SAMPLE = 20
"""Below this many usable observations the Hill estimate is not computed at all."""

_HILL_MIN_ORDER_STATS = 2
"""Fewest order statistics the estimator will average over before it is noise."""


def _hill_tail_index(sample: FloatArray, fraction: float) -> tuple[float, int]:
    r"""Hill (1975) estimate of the right-tail index of a positive sample.

    For order statistics :math:`y_{(1)}\ge\dots\ge y_{(m)}` and
    :math:`k=\lceil fm\rceil`,

    .. math::
        \hat\gamma_k = \frac1k\sum_{i=1}^{k}\ln\frac{y_{(i)}}{y_{(k+1)}},
        \qquad \hat\xi = 1/\hat\gamma_k .

    Biased and only weakly consistent: it is a *maximum-domain-of-attraction*
    estimator, so a lognormal sample (which has all moments) still returns a
    finite number that drifts upward with :math:`m`. For the lognormal MMAR the
    terminal law is in the Gumbel domain and the true tail index is
    :math:`+\infty`; read the output as a local tail heaviness at the chosen
    ``fraction``, never as a Pareto exponent.

    Returns
    -------
    (xi, k)
        The estimate and the number of order statistics it actually averaged
        over. ``xi`` is NaN and ``k`` is ``0`` when fewer than
        :data:`_HILL_MIN_SAMPLE` usable observations are available; the caller
        is expected to say so out loud rather than publish the NaN silently.
    """
    y = np.asarray(sample, dtype=np.float64)
    y = y[np.isfinite(y) & (y > 0.0)]
    m = y.size
    if m < _HILL_MIN_SAMPLE:
        return float("nan"), 0
    k = min(max(math.ceil(fraction * m), _HILL_MIN_ORDER_STATS), m - 1)
    top = np.sort(y)[::-1][: k + 1]
    gamma = float(np.mean(np.log(top[:k])) - math.log(top[k]))
    return (float(1.0 / gamma) if gamma > 0.0 else float("inf")), k


def _warn_if_hill_is_unusable(n_paths: int, fraction: float, k_effective: int) -> None:
    """Say out loud when the reported tail indices are NaN or built on ~nothing.

    A NaN or a two-point Hill average sitting unannounced in a returned
    dataclass is precisely the quiet-garbage failure mode this library refuses
    elsewhere (see :mod:`mmar.core.validate`), so the two degenerate regimes are
    named here rather than left for the reader to discover.
    """
    if k_effective == 0:
        warnings.warn(
            f"tail_index_upper/lower are NaN: the Hill estimator needs at least "
            f"{_HILL_MIN_SAMPLE} finite positive terminal values and this ensemble has "
            f"{n_paths}. Every other field of the forecast is valid; raise n_paths to at "
            f"least {math.ceil(_HILL_MIN_ORDER_STATS / fraction)} for a tail index that "
            "averages more than a couple of order statistics.",
            UserWarning,
            stacklevel=3,
        )
        return
    requested = math.ceil(fraction * n_paths)
    if requested < _HILL_MIN_ORDER_STATS:
        warnings.warn(
            f"hill_fraction={fraction:g} of {n_paths} paths is {requested} order "
            f"statistic(s), below the floor of {_HILL_MIN_ORDER_STATS}; the reported tail "
            f"indices average {k_effective} points and are noise, not an estimate. Raise "
            f"n_paths to at least {math.ceil(_HILL_MIN_ORDER_STATS / fraction)}.",
            UserWarning,
            stacklevel=3,
        )


def mmar_forecast_distribution(
    spec: MMARSpec,
    horizon: float,
    n_paths: int,
    *,
    seed: SeedLike,
    oversample: int | None = None,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
    hill_fraction: float = 0.05,
) -> MMARForecast:
    r"""Monte Carlo law of :math:`P(\text{horizon})` under the MMAR.

    Physical time is normalised to :math:`[0,1]`, so ``horizon`` is a fraction
    of the simulated span; each path is interpolated at that instant.

    Because :math:`X(t)=B_H(\theta(t))` with :math:`\theta` independent of
    :math:`B_H`, the terminal log-return is a *variance mixture*
    :math:`X(h)\mid\theta \sim \mathcal N(0, \theta(h)^{2H})` (exactly, since
    fBm is Gaussian). Its unconditional law is therefore leptokurtic with
    tails much heavier than the lognormal benchmark, which is the whole point
    of the model, and it is **driftless**: the MMAR as specified has no
    :math:`\mu t` term, so :math:`\mathbb{E}[P(h)] = P(0)\,
    \mathbb{E}[e^{X(h)}] > P(0)` purely from Jensen convexity. Do not read the
    mean as an expected return.

    .. warning::
       The excess kurtosis vanishes as :math:`h \to T`. Trading time is
       normalised so that :math:`\theta(T)=T` on *every* path, hence
       :math:`X(T)=B_H(T)` is exactly Gaussian and :math:`P(T)` exactly
       lognormal. The mixing variance :math:`\theta(h)` is random only for
       :math:`h<T`, so the interesting forecast horizons are the interior
       ones. (Normalising away the cascade's random total mass
       :math:`\Omega` is what pins the endpoint; the alternative --- letting
       :math:`\theta(T)` be random --- makes the clock's scale unidentified
       from a single path.)

    Parameters
    ----------
    horizon
        Physical time in :math:`(0, 1]`.
    n_paths
        Ensemble size. The tail quantiles need far more paths than the centre:
        the 0.1% quantile is set by :math:`0.001\,n_{\rm paths}` observations.
    hill_fraction
        Fraction of order statistics entering the Hill estimator. The estimate
        needs :data:`_HILL_MIN_SAMPLE` usable terminal values to be computed at
        all and ``ceil(hill_fraction * n_paths) >=`` :data:`_HILL_MIN_ORDER_STATS`
        to mean anything; both shortfalls emit a :class:`UserWarning` rather
        than returning a NaN or a two-point average unannounced. The number of
        order statistics actually used is recorded as the
        ``hill_order_statistics`` diagnostic.

    Returns
    -------
    MMARForecast
        Quantiles, moments and Hill tail indices, with provenance.
    """
    spec.validate()
    if not (0.0 < horizon <= T_MAX):
        raise ValueError(
            f"horizon must lie in (0, {T_MAX:g}]: physical time is normalised to the "
            f"simulated span, got {horizon!r}. Rescale the horizon, or raise kmax and "
            "reinterpret the unit interval as a longer calendar window."
        )
    if n_paths < 2:
        raise ValueError(f"need at least 2 paths for a distribution, got {n_paths}")
    if not (0.0 < hill_fraction < 0.5):
        raise ValueError(f"hill_fraction must lie in (0, 0.5), got {hill_fraction!r}")
    levels = np.asarray(quantile_levels, dtype=np.float64)
    if levels.size == 0 or np.any(levels <= 0.0) or np.any(levels >= 1.0):
        raise ValueError("quantile_levels must be a non-empty set of levels strictly in (0, 1)")

    oversample = default_oversample(spec) if oversample is None else oversample
    paths = simulate_ensemble(spec, n_paths, seed=seed, oversample=oversample)
    terminal = np.array([float(p.at(horizon)) for p in paths], dtype=np.float64)
    log_ret = np.asarray(np.log(terminal / spec.price0), dtype=np.float64)

    quantiles = np.asarray(np.quantile(terminal, levels), dtype=np.float64)
    upper, k_upper = _hill_tail_index(terminal, hill_fraction)
    # The left tail of P is the right tail of 1/P; the reciprocal is exact here
    # because prices are strictly positive by construction.
    lower, _ = _hill_tail_index(np.asarray(1.0 / terminal, dtype=np.float64), hill_fraction)
    _warn_if_hill_is_unusable(n_paths, hill_fraction, k_upper)

    prov = new_provenance(
        "mmar.forecast",
        estimator_version=_VERSION,
        params={
            "spec": spec.to_canonical(),
            "horizon": repr(float(horizon)),
            "n_paths": int(n_paths),
            "oversample": int(oversample),
            "quantile_levels": [repr(float(x)) for x in levels],
            "hill_fraction": repr(float(hill_fraction)),
        },
        inputs=(InputRef.from_array("terminal_price", terminal),),
        seed=seed_info(seed),
        notes=(
            "Terminal law of the subordinated process; interpolated at the horizon in "
            "physical time, not trading time.",
            "Hill indices are descriptive: the lognormal MMAR has no Pareto tail.",
        ),
    ).with_diagnostics(
        ensemble_roots=tuple(p.provenance.short_root for p in paths[: min(8, len(paths))]),
        hill_order_statistics=int(k_upper),
    )

    return MMARForecast(
        horizon=float(horizon),
        n_paths=int(n_paths),
        price0=float(spec.price0),
        terminal_price=terminal,
        terminal_log_return=log_ret,
        quantile_levels=levels,
        quantiles=quantiles,
        mean=float(np.mean(terminal)),
        std=float(np.std(terminal, ddof=1)),
        tail_index_upper=upper,
        tail_index_lower=lower,
        hill_fraction=float(hill_fraction),
        spec=spec,
        provenance=prov,
    )
