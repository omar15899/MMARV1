r"""Multiplicative cascades: the canonical generators of multifractal measures.

A :math:`b`-adic multiplicative cascade builds a measure on :math:`[0,1]` by
recursive subdivision. Level :math:`k` partitions the interval into
:math:`b^{k}` cells of width :math:`\delta = b^{-k}`; each cell inherits its
parent's mass multiplied by an independent copy of a random *multiplier*
:math:`M \ge 0`,

.. math::
    \mu_{i_1 \ldots i_k} \;=\; \prod_{j=1}^{k} M_{i_1 \ldots i_j} .

Mandelbrot (1974) showed that when :math:`\mathbb{E}[bM] = 1` the sequence of
level measures is a positive martingale and converges weakly to a limit measure
:math:`\mu_\infty`. Two constructions must be kept apart:

*Canonical cascade.* :math:`\mathbb{E}\big[\sum_{\text{children}} M\big] = 1`,
so mass is conserved only *in expectation*. The total mass of the limit,
:math:`\Omega = \mu_\infty([0,1])`, is a genuinely random variable solving the
fixed point :math:`\Omega \stackrel{d}{=} \sum_{i<b} M_i \Omega_i`. It has a
heavy right tail and, above the Kahane--Peyrière critical exponent, infinite
moments. This is the construction the multifractal formalism is stated for.

*Microcanonical (conservative) cascade.* The :math:`b` children of every parent
are rescaled to sum to the parent's mass, so mass is conserved pathwise and
:math:`\Omega \equiv 1`. The rescaling makes siblings dependent, which
truncates the right tail of the mass distribution and therefore *changes the
spectrum away from the closed form of the canonical model*, most visibly at
large :math:`q`.

Conventions
-----------
Everything here is in the **measure view**, so the closed forms quoted in the
docstrings are :math:`\tau(q)`, defined by
:math:`Z(q,\delta)=\sum_i \mu_i^q \sim \delta^{\tau(q)}`. The signal-view
exponent of :mod:`mmar.core.types` is :math:`\zeta(q)=\tau(q)+d` with
:math:`d=1` here, and the spectrum is
:math:`f(\alpha)=\inf_q\{q\alpha-\tau(q)\}`. Mass conservation is exactly the
statement :math:`\tau(1)=0`, and :math:`\tau(0)=-d=-1` always.

What this module deliberately does *not* do
-------------------------------------------
The predecessor of this library normalised the finest level by multiplying it
cellwise by ``np.random.normal(2**(lam*kmax), sigma, 2**kmax)``: one independent
Gaussian *per cell*, with an astronomically large mean. That is wrong three
times over --- it destroys the multiplicative structure the cascade was built
to have, it adds an independent noise field whose own scaling is trivial, and a
Gaussian is not supported on :math:`[0,\infty)` so cells go negative and the
object stops being a measure. The correct canonical normalisation multiplies
the *whole* measure by a single positive random total mass; see
:func:`normalise_cascade`.

References
----------
Mandelbrot, B. (1974). Intermittent turbulence in self-similar cascades.
*J. Fluid Mech.* 62, 331--358.

Kahane, J.-P. and Peyrière, J. (1976). Sur certaines martingales de
Benoit Mandelbrot. *Adv. Math.* 22, 131--145.

Schertzer, D. and Lovejoy, S. (1987). Physical modeling and analysis of rain
and clouds by anisotropic scaling multiplicative processes.
*J. Geophys. Res.* 92, 9693--9714.

She, Z.-S. and Leveque, E. (1994). Universal scaling laws in fully developed
turbulence. *Phys. Rev. Lett.* 72, 336--339.

Calvet, L. and Fisher, A. (2002). Multifractality in asset returns: theory and
evidence. *Rev. Econ. Stat.* 84, 381--406.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Literal

import numpy as np
from scipy.special import logsumexp

from mmar.core.provenance import InputRef, Provenance, array_digest, new_provenance
from mmar.core.rng import SeedLike, make_rng, seed_info, spawn_rngs
from mmar.core.types import FloatArray
from mmar.core.validate import validate_measure, validate_q_grid

__all__ = [
    "CascadeFamily",
    "CascadeResult",
    "CascadeSpec",
    "MultiplierSampler",
    "NormalisationMode",
    "cascade_measure",
    "cascade_moments",
    "deterministic_cascade",
    "generate",
    "log_gamma_cascade",
    "log_gamma_multipliers",
    "log_poisson_cascade",
    "log_poisson_multipliers",
    "log_stable_cascade",
    "log_stable_multipliers",
    "lognormal_cascade",
    "lognormal_multipliers",
    "normalise_cascade",
    "random_binomial_cascade",
    "trading_time",
    "uniform_binomial_multipliers",
]

MODULE_VERSION: Final = "1"

_MAX_CELLS: Final = 1 << 24
"""Refuse to materialise more cells than this; 2**24 float64 is already 134 MB."""

MultiplierSampler = Callable[[int, np.random.Generator], FloatArray]
r"""Draws the multipliers for one cascade level.

Called as ``sampler(k, rng)`` where ``k`` is the zero-based index of the level
being *created*, and must return a strictly-shaped ``(b**(k+1),)`` array of
non-negative multipliers, ordered so that entry :math:`ib+j` is child
:math:`j` of parent :math:`i`.
"""

NormalisationMode = Literal["unit_mass", "omega", "none"]


class CascadeFamily(StrEnum):
    """Which closed-form multiplier law a :class:`CascadeSpec` refers to."""

    DETERMINISTIC = "deterministic"
    LOGNORMAL = "lognormal"
    LOG_POISSON = "log_poisson"
    LOG_GAMMA = "log_gamma"
    LOG_STABLE = "log_stable"
    RANDOM_BINOMIAL = "random_binomial"


# --------------------------------------------------------------------------- #
# Shape and parameter guards
# --------------------------------------------------------------------------- #


def _check_geometry(kmax: int, b: int) -> None:
    if int(b) != b or b < 2:
        raise ValueError(f"branching number b must be an integer >= 2, got {b!r}")
    if int(kmax) != kmax or kmax < 0:
        raise ValueError(f"kmax must be a non-negative integer, got {kmax!r}")
    if b**kmax > _MAX_CELLS:
        raise ValueError(
            f"b**kmax = {b}**{kmax} = {b**kmax} cells exceeds the {_MAX_CELLS} cell guard; "
            "reduce kmax or coarse-grain the analysis instead"
        )


def _finite_positive(m: FloatArray, *, where: str) -> FloatArray:
    if not np.all(np.isfinite(m)):
        raise ValueError(
            f"{where} produced {int((~np.isfinite(m)).sum())} non-finite multipliers; "
            "the parameters put the log-multiplier outside float64 range"
        )
    if np.any(m < 0.0):
        raise ValueError(
            f"{where} produced negative multipliers; a cascade measure requires M >= 0"
        )
    return m


# --------------------------------------------------------------------------- #
# Core engine
# --------------------------------------------------------------------------- #


def cascade_measure(
    kmax: int,
    multipliers: MultiplierSampler,
    *,
    b: int = 2,
    seed: SeedLike = None,
    normalise: bool = True,
) -> FloatArray:
    r"""Build the level-``kmax`` measure of a :math:`b`-adic multiplicative cascade.

    Parameters
    ----------
    kmax
        Number of subdivision levels. The result has ``b**kmax`` cells, each of
        width :math:`\delta=b^{-k_{\max}}`.
    multipliers
        A :data:`MultiplierSampler`. Level :math:`k` receives ``(k, rng)`` and
        must return ``b**(k+1)`` non-negative multipliers in child-major order.
    b
        Branching number.
    seed
        Anything :func:`mmar.core.rng.make_rng` accepts. No global RNG state is
        touched.
    normalise
        Divide by the total mass at the end. Note this *erases* the canonical
        cascade's random total mass :math:`\Omega`, which for some questions
        (tail of the mass distribution, degenerate cascades) is the object of
        interest. The concrete family wrappers in this module therefore default
        to ``normalise=False``; only this low-level engine defaults to ``True``.

    Returns
    -------
    FloatArray
        Shape ``(b**kmax,)``, non-negative.

    Notes
    -----
    The recursion is :math:`\mu^{(k+1)} = \operatorname{repeat}(\mu^{(k)}, b)
    \odot M^{(k)}`, evaluated as two numpy operations per level. Total work is
    :math:`O(b^{k_{\max}} \cdot b/(b-1))`, i.e. linear in the output size; there
    is no per-cell Python loop and no nested list structure.
    """
    _check_geometry(kmax, b)
    rng = make_rng(seed)
    mu = np.ones(1, dtype=np.float64)
    for k in range(kmax):
        m = np.asarray(multipliers(k, rng), dtype=np.float64).ravel()
        expected = b ** (k + 1)
        if m.size != expected:
            raise ValueError(
                f"multiplier sampler returned {m.size} values at level k={k}; "
                f"expected b**(k+1) = {expected}"
            )
        _finite_positive(m, where=f"multiplier sampler at level k={k}")
        mu = np.repeat(mu, b) * m
    if not normalise:
        return np.ascontiguousarray(mu)
    total = float(mu.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(
            f"cascade degenerated to total mass {total}; the multiplier law is not "
            "mass-conserving in expectation, or kmax is past the martingale's "
            "degeneracy threshold"
        )
    return np.ascontiguousarray(mu / total)


# --------------------------------------------------------------------------- #
# Deterministic (binomial / multinomial) cascade
# --------------------------------------------------------------------------- #


def deterministic_cascade(
    kmax: int, m: Sequence[float] | FloatArray, *, b: int | None = None
) -> FloatArray:
    r"""The multinomial measure with fixed weights ``m``.

    Every parent splits its mass in the same fixed proportions
    :math:`m_0,\dots,m_{b-1}`, so the level-:math:`k` cell indexed by digits
    :math:`(i_1,\dots,i_k)` carries :math:`\prod_j m_{i_j}`. For :math:`b=2` this
    is Mandelbrot's binomial measure, the standard textbook multifractal.

    Closed form (measure view):

    .. math::
        \tau(q) \;=\; -\log_b\!\Big(\sum_{i<b} m_i^{\,q}\Big),

    since :math:`Z(q, b^{-k}) = \big(\sum_i m_i^q\big)^{k}` *exactly* --- there
    is no ensemble average and no finite-size correction. Consequently
    :math:`\tau(0)=-1`, :math:`\tau(1)=0`, and the spectrum is the Legendre
    transform of a strictly concave function, supported on
    :math:`[\min_i -\log_b m_i,\ \max_i -\log_b m_i]`.

    Parameters
    ----------
    kmax
        Number of levels.
    m
        Weight vector. Must be non-negative and sum to one: the cascade
        conserves mass pathwise, :math:`\sum_i m_i = 1 \iff \tau(1)=0`, and
        any other normalisation makes the total mass drift like
        :math:`(\sum_i m_i)^{k}` and the fitted :math:`\tau` acquire a spurious
        additive constant.
    b
        Branching number; defaults to ``len(m)`` and must equal it if given.

    Returns
    -------
    FloatArray
        Shape ``(b**kmax,)``. Bit-for-bit reproducible: no randomness is used.
    """
    w = np.asarray(m, dtype=np.float64).ravel()
    if w.size < 2:
        raise ValueError(f"the weight vector needs at least two entries, got {w.size}")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("the weight vector must be finite and non-negative")
    total = float(w.sum())
    if abs(total - 1.0) > 1e-12:
        raise ValueError(
            f"weights sum to {total!r}, not 1. A multiplicative cascade conserves mass "
            "pathwise only when sum(m) == 1; otherwise the total mass drifts as "
            f"{total!r}**kmax and tau(1) != 0. Pass m / m.sum() if that is what you meant."
        )
    bb = w.size if b is None else int(b)
    if bb != w.size:
        raise ValueError(f"b={bb} contradicts len(m)={w.size}")
    _check_geometry(kmax, bb)

    mu: FloatArray = np.ones(1, dtype=np.float64)
    for _ in range(kmax):
        # kron(mu, w)[i*b + j] = mu[i] * w[j]: the whole level in one call.
        mu = np.asarray(np.kron(mu, w), dtype=np.float64)
    return np.ascontiguousarray(mu)


# --------------------------------------------------------------------------- #
# Random multiplier laws
# --------------------------------------------------------------------------- #


def _conserve(m: FloatArray, b: int) -> FloatArray:
    """Rescale each parent's ``b`` children to sum to one (microcanonical step)."""
    blocks = m.reshape(-1, b)
    s = blocks.sum(axis=1, keepdims=True)
    if np.any(s <= 0.0):
        raise ValueError("a sibling group summed to zero; cannot renormalise conservatively")
    return np.asarray((blocks / s).ravel(), dtype=np.float64)


def lognormal_multipliers(
    lam: float,
    sigma2: float | None = None,
    *,
    b: int = 2,
    conservative: bool = False,
) -> MultiplierSampler:
    r"""Sampler for the lognormal multiplier :math:`M=b^{-V}`, :math:`V\sim N(\lambda,\sigma^2)`.

    See :func:`lognormal_cascade` for the mathematics; this returns the level
    sampler so it can be composed with :func:`cascade_measure` directly.
    """
    if b < 2:
        raise ValueError(f"b must be >= 2, got {b}")
    ln_b = math.log(b)
    if sigma2 is None:
        if lam <= 1.0:
            raise ValueError(
                f"lam={lam} <= 1 leaves no room for the conservation constraint "
                "sigma2 = 2*(lam - 1)/ln(b), which would be non-positive. Either raise "
                "lam above 1 or pass sigma2 explicitly and accept a non-conservative cascade."
            )
        s2 = 2.0 * (lam - 1.0) / ln_b
    else:
        if sigma2 < 0.0:
            raise ValueError(f"sigma2 must be non-negative, got {sigma2}")
        s2 = float(sigma2)
    sd = math.sqrt(s2)

    def sample(k: int, rng: np.random.Generator) -> FloatArray:
        n = b ** (k + 1)
        v = rng.normal(lam, sd, size=n)
        with np.errstate(over="ignore"):
            m = np.asarray(np.exp(-v * ln_b), dtype=np.float64)
        _finite_positive(m, where="lognormal multiplier")
        return _conserve(m, b) if conservative else m

    return sample


def lognormal_cascade(
    kmax: int,
    lam: float,
    sigma2: float | None = None,
    *,
    b: int = 2,
    seed: SeedLike = None,
    conservative: bool = False,
    normalise: bool = False,
) -> FloatArray:
    r"""Lognormal cascade: :math:`M = b^{-V}` with :math:`V \sim N(\lambda,\sigma^2)`.

    The multiplier is lognormal, so the log-measure is a sum of Gaussians and the
    limit is the classical Kolmogorov--Obukhov (1962) lognormal model. With the
    canonical construction,

    .. math::
        \mathbb{E}[M^q] = b^{-q\lambda}\exp\!\Big(\tfrac{q^2\sigma^2\ln^2 b}{2}\Big),
        \qquad
        \tau(q) \;=\; q\lambda - 1 - \tfrac{\sigma^2 \ln b}{2}\, q^2 .

    Mass conservation :math:`\mathbb{E}[bM]=1` is equivalent to
    :math:`\tau(1)=0`, i.e.

    .. math::
        \sigma^2 \;=\; \frac{2(\lambda-1)}{\ln b},

    which is what ``sigma2=None`` substitutes. Under that constraint
    :math:`\tau(q) = q - 1 - \tfrac{\sigma^2\ln b}{2}(q^2-q)`, a downward
    parabola through :math:`(0,-1)` and :math:`(1,0)`, whose Legendre transform
    is the parabolic spectrum
    :math:`f(\alpha)=1-(\alpha-\alpha_0)^2/(4(\alpha_0-1))` with
    :math:`\alpha_0=\lambda`.

    Parameters
    ----------
    lam
        Mean of :math:`V`; equals :math:`\alpha_0`, the most probable Hölder
        exponent, under the conservation constraint. Must exceed 1 when
        ``sigma2`` is derived.
    sigma2
        Variance of :math:`V`. If given explicitly and inconsistent with the
        constraint above, the cascade is *not* mass conserving even in
        expectation: the total mass drifts multiplicatively by
        :math:`b^{1-\lambda}e^{\sigma^2\ln^2 b/2}` per level. That is a legitimate
        object (it merely shifts :math:`\tau` by a constant), but it is never
        applied silently --- pass ``normalise=True`` if you want unit mass.
    conservative
        ``False`` (default) is the *canonical* cascade: mass is conserved only in
        expectation and the total mass :math:`\Omega` is random with a heavy
        right tail. ``True`` renormalises every sibling group to sum to the
        parent's mass, giving the *microcanonical* cascade with
        :math:`\Omega\equiv 1`. The two agree on :math:`\tau(q)` for moderate
        :math:`q` but not in the tails: the microcanonical rescaling caps the
        largest cell masses, so the empirical :math:`\tau(q)` for large positive
        :math:`q` bends above the canonical closed form, and the left branch of
        :math:`f(\alpha)` is truncated.

    Notes
    -----
    The canonical cascade degenerates (:math:`\Omega \to 0` a.s.) once
    :math:`\tau'(1) \le 0`. Since :math:`\tau'(q) = \lambda - q\sigma^2\ln b`
    that is :math:`\lambda \le \sigma^2 \ln b`, and under the conservation
    constraint :math:`\sigma^2\ln b = 2(\lambda-1)` it reads
    :math:`\lambda \ge 2`. Equivalently, by Kahane--Peyrière the martingale
    :math:`\Omega_k` is uniformly integrable iff
    :math:`\mathbb{E}[W\ln W] < \ln b` with :math:`W = bM = b^{1-V}`; writing
    :math:`Y=(1-V)\ln b` and using :math:`\mathbb{E}[e^{Y}]=1`,

    .. math::
        \mathbb{E}[W \ln W] = \mathbb{E}[e^{Y}Y]
            = (1-\lambda)\ln b + \sigma^2\ln^2 b ,

    so the condition is exactly :math:`\sigma^2\ln b < \lambda`, i.e.
    :math:`\lambda < 2`. Nothing here forbids the degenerate range, but the
    resulting measure is not a useful multifractal.
    """
    sampler = lognormal_multipliers(lam, sigma2, b=b, conservative=conservative)
    return cascade_measure(kmax, sampler, b=b, seed=seed, normalise=normalise)


def log_poisson_multipliers(
    gamma: float, delta: float, *, b: int = 2, conservative: bool = False
) -> MultiplierSampler:
    r"""Sampler for the She--Leveque multiplier :math:`M=\delta^{N}b^{-\gamma}`.

    See :func:`log_poisson_cascade`.
    """
    if b < 2:
        raise ValueError(f"b must be >= 2, got {b}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0, 1) for a contracting cascade, got {delta}")
    if not 0.0 <= gamma < 1.0:
        raise ValueError(
            f"gamma must lie in [0, 1): it is the minimal Hoelder exponent alpha_min, "
            f"and gamma >= 1 makes the Poisson rate non-positive. Got {gamma}"
        )
    ln_b = math.log(b)
    rate = (1.0 - gamma) * ln_b / (1.0 - delta)
    ln_delta = math.log(delta)

    def sample(k: int, rng: np.random.Generator) -> FloatArray:
        n = b ** (k + 1)
        counts = rng.poisson(rate, size=n).astype(np.float64)
        m = np.asarray(np.exp(counts * ln_delta - gamma * ln_b), dtype=np.float64)
        _finite_positive(m, where="log-Poisson multiplier")
        return _conserve(m, b) if conservative else m

    return sample


def log_poisson_cascade(
    kmax: int,
    gamma: float,
    delta: float,
    *,
    b: int = 2,
    seed: SeedLike = None,
    conservative: bool = False,
    normalise: bool = False,
) -> FloatArray:
    r"""Log-Poisson (She--Leveque) cascade: :math:`M = \delta^{N} b^{-\gamma}`.

    :math:`N` is Poisson with the rate fixed by mass conservation. Writing
    :math:`\theta` for that rate,

    .. math::
        \mathbb{E}[M] = e^{\theta(\delta-1)} b^{-\gamma} \stackrel{!}{=} b^{-1}
        \quad\Longrightarrow\quad
        \theta \;=\; \frac{(1-\gamma)\ln b}{1-\delta} .

    The moments are :math:`\mathbb{E}[M^q]=b^{-\gamma q}e^{\theta(\delta^q-1)}`,
    hence the closed form (measure view)

    .. math::
        \boxed{\;\tau(q) \;=\; \gamma q - 1 + (1-\gamma)\,\frac{1-\delta^{q}}{1-\delta}\;}

    which satisfies :math:`\tau(0)=-1` and :math:`\tau(1)=0` identically.

    Unlike the lognormal model this :math:`\tau` is *not* a parabola: it is
    asymptotically linear, :math:`\tau(q)\to\gamma q - 1 +
    (1-\gamma)/(1-\delta)` as :math:`q\to\infty`. The spectrum therefore has a
    hard lower edge at :math:`\alpha_{\min}=\gamma` carrying finite
    :math:`f(\gamma)=1-(1-\gamma)/(1-\delta)`, i.e. a *left-truncated*
    spectrum --- the signature of a cascade driven by discrete, sparse
    singular events rather than by a diffuse Gaussian generator. She and
    Leveque's turbulence values correspond to :math:`\gamma=1/9`,
    :math:`\delta=2/3` (co-dimension 2 filaments).

    Parameters
    ----------
    gamma
        Minimal Hölder exponent :math:`\alpha_{\min}\in[0,1)`.
    delta
        Multiplier contraction per Poisson point, in :math:`(0,1)`.
    conservative
        As in :func:`lognormal_cascade`.
    """
    sampler = log_poisson_multipliers(gamma, delta, b=b, conservative=conservative)
    return cascade_measure(kmax, sampler, b=b, seed=seed, normalise=normalise)


def log_gamma_multipliers(
    shape: float, scale: float, *, b: int = 2, conservative: bool = False
) -> MultiplierSampler:
    r"""Sampler for :math:`M \propto b^{-V}` with :math:`V\sim\Gamma(\text{shape},\text{scale})`.

    See :func:`log_gamma_cascade`.
    """
    if b < 2:
        raise ValueError(f"b must be >= 2, got {b}")
    if shape <= 0.0 or scale <= 0.0:
        raise ValueError(f"shape and scale must be strictly positive, got {shape}, {scale}")
    ln_b = math.log(b)
    # E[b^{-V}] = (1 + scale*ln b)^{-shape}; dividing by b*E[b^{-V}] forces E[M] = 1/b.
    log_norm = shape * math.log1p(scale * ln_b) - ln_b

    def sample(k: int, rng: np.random.Generator) -> FloatArray:
        n = b ** (k + 1)
        v = rng.gamma(shape, scale, size=n)
        m = np.asarray(np.exp(-v * ln_b + log_norm), dtype=np.float64)
        _finite_positive(m, where="log-gamma multiplier")
        return _conserve(m, b) if conservative else m

    return sample


def log_gamma_cascade(
    kmax: int,
    shape: float,
    scale: float,
    *,
    b: int = 2,
    seed: SeedLike = None,
    conservative: bool = False,
    normalise: bool = False,
) -> FloatArray:
    r"""Log-gamma cascade: :math:`V \sim \Gamma(a, \theta)` and :math:`M \propto b^{-V}`.

    Because a gamma law has two free parameters, the conservation constraint
    cannot be met by both; the multiplier is therefore defined with the mean
    divided out,

    .. math::
        M \;=\; \frac{b^{-V}}{b\,\mathbb{E}[b^{-V}]},
        \qquad \mathbb{E}[b^{-V}] = (1+\theta\ln b)^{-a},

    so :math:`\mathbb{E}[bM]=1` holds exactly for every :math:`(a,\theta)`.
    Then :math:`\mathbb{E}[M^q] = b^{-q}(1+\theta\ln b)^{qa}(1+q\theta\ln b)^{-a}`
    and

    .. math::
        \boxed{\;\tau(q) \;=\; q - 1 - qa\log_b(1+\theta\ln b)
                              + a\log_b(1+q\theta\ln b)\;}

    with :math:`\tau(0)=-1`, :math:`\tau(1)=0`.

    The Laplace transform of a gamma law is finite only for
    :math:`1+q\theta\ln b>0`, so **all moments with**
    :math:`q \le -1/(\theta\ln b)` **diverge**: unlike the lognormal cascade the
    log-gamma model has a genuine left-side critical moment, and any estimator
    run at more negative :math:`q` than that is estimating a divergent
    quantity. As :math:`a\to\infty` with :math:`a\theta` fixed the log-gamma
    converges to the lognormal cascade.

    Since :math:`V\ge 0` a.s., :math:`M` is bounded above by
    :math:`(1+\theta\ln b)^{a}/b`; the spectrum is right-truncated at
    :math:`\alpha_{\max}=\infty` but left-truncated at
    :math:`\alpha_{\min}=1-a\log_b(1+\theta\ln b)`.
    """
    sampler = log_gamma_multipliers(shape, scale, b=b, conservative=conservative)
    return cascade_measure(kmax, sampler, b=b, seed=seed, normalise=normalise)


def _stable_rvs(rng: np.random.Generator, alpha: float, beta: float, size: int) -> FloatArray:
    r"""Chambers--Mallows--Stuck draw from :math:`S_\alpha(1,\beta,0)`, :math:`\alpha\neq1`.

    Uses the Samorodnitsky--Taqqu (equivalently Nolan's ``S1``) parameterisation,
    the one in which the Laplace transform of a :math:`\beta=1` variable is
    :math:`\mathbb{E}e^{-\gamma X}=\exp\{-\gamma^\alpha/\cos(\pi\alpha/2)\}`.
    """
    if alpha == 1.0:
        raise ValueError("alpha = 1 is a separate (Cauchy) case with a log-corrected transform")
    u = rng.uniform(-math.pi / 2.0, math.pi / 2.0, size=size)
    w = rng.exponential(1.0, size=size)
    zeta = beta * math.tan(math.pi * alpha / 2.0)
    b_shift = math.atan(zeta) / alpha
    s = (1.0 + zeta * zeta) ** (1.0 / (2.0 * alpha))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        num = np.sin(alpha * (u + b_shift))
        den = np.cos(u) ** (1.0 / alpha)
        tail = (np.cos(u - alpha * (u + b_shift)) / w) ** ((1.0 - alpha) / alpha)
        x = s * (num / den) * tail
    out = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(
            f"{int((~np.isfinite(out)).sum())} non-finite stable draws; alpha={alpha} is "
            "numerically too close to 1 for the Chambers-Mallows-Stuck formula"
        )
    return out


def log_stable_multipliers(
    alpha: float, sigma: float, *, b: int = 2, conservative: bool = False
) -> MultiplierSampler:
    r"""Sampler for the log-Lévy multiplier of :func:`log_stable_cascade`."""
    if b < 2:
        raise ValueError(f"b must be >= 2, got {b}")
    if not 0.0 < alpha <= 2.0 or abs(alpha - 1.0) < 1e-6:
        raise ValueError(
            f"alpha must lie in (0, 2] and stay away from 1 (the Cauchy case needs a "
            f"different Laplace transform); got {alpha}"
        )
    if sigma <= 0.0:
        raise ValueError(f"sigma must be strictly positive, got {sigma}")
    ln_b = math.log(b)
    # E[b^{-V}] = exp(-A) with A = sigma^alpha (ln b)^alpha / cos(pi*alpha/2).
    a_coef = sigma**alpha * ln_b**alpha / math.cos(math.pi * alpha / 2.0)
    log_norm = a_coef - ln_b

    def sample(k: int, rng: np.random.Generator) -> FloatArray:
        n = b ** (k + 1)
        v = sigma * _stable_rvs(rng, alpha, 1.0, n)
        with np.errstate(over="ignore"):
            m = np.asarray(np.exp(-v * ln_b + log_norm), dtype=np.float64)
        _finite_positive(m, where="log-stable multiplier")
        return _conserve(m, b) if conservative else m

    return sample


def log_stable_cascade(
    kmax: int,
    alpha: float,
    sigma: float,
    *,
    b: int = 2,
    seed: SeedLike = None,
    conservative: bool = False,
    normalise: bool = False,
) -> FloatArray:
    r"""Log-Lévy cascade: the Schertzer--Lovejoy universal multifractals.

    The log-multiplier is a maximally skewed (:math:`\beta=1`) :math:`\alpha`-stable
    variable :math:`V \sim S_\alpha(\sigma,1,0)`, and
    :math:`M = b^{-V}/(b\,\mathbb{E}[b^{-V}])`. Skewness :math:`+1` is not a
    stylistic choice: it is the only skewness for which
    :math:`\mathbb{E}[b^{-qV}]` is finite for :math:`q>0`, so it is the only one
    for which the cascade has positive moments at all. With

    .. math::
        A \;=\; \frac{\sigma^{\alpha}(\ln b)^{\alpha}}{\cos(\pi\alpha/2)},
        \qquad \mathbb{E}[b^{-qV}] = e^{-A q^{\alpha}},

    the closed form is

    .. math::
        \boxed{\;\tau(q) \;=\; q - 1 + \frac{A}{\ln b}\,\big(q^{\alpha}-q\big)
              \;=\; q - 1 - \frac{C_1}{\alpha-1}\big(q^{\alpha}-q\big)\;}

    which is exactly the universal-multifractal form with Lévy index
    :math:`\alpha` and mean co-dimension

    .. math::
        C_1 \;=\; -\,\frac{(\alpha-1)\,\sigma^{\alpha}(\ln b)^{\alpha-1}}
                          {\cos(\pi\alpha/2)} \;>\; 0 .

    Limits and pathologies:

    * :math:`\alpha=2` gives :math:`\cos(\pi)=-1`, hence
      :math:`\tau(q)=q-1-\sigma^2\ln b\,(q^2-q)`, **identical to the
      conservative lognormal cascade with** :math:`\sigma^2_{\log}=2\sigma^2`
      (the factor two is the Samorodnitsky--Taqqu convention
      :math:`S_2(\sigma,\cdot,0)=N(0,2\sigma^2)`). The Gaussian case is the only
      one with a parabolic :math:`\tau` and a spectrum unbounded on both sides.
    * :math:`\alpha<1` puts :math:`V\ge0` a.s., so :math:`M` is bounded and the
      spectrum is left-truncated at a finite :math:`\alpha_{\min}`.
    * :math:`1<\alpha<2` gives an unbounded :math:`M` and, because
      :math:`q^\alpha` grows sublinearly in the relevant sense, a divergence of
      moments for :math:`q<0` --- negative-:math:`q` estimates are meaningless
      here.
    * :math:`\alpha=1` is excluded: the Laplace transform picks up a
      :math:`q\ln q` term and the sampler formula degenerates.

    Parameters
    ----------
    alpha
        Lévy index in :math:`(0,2]`, excluding a neighbourhood of 1.
    sigma
        Stable scale of the log-multiplier.
    """
    sampler = log_stable_multipliers(alpha, sigma, b=b, conservative=conservative)
    return cascade_measure(kmax, sampler, b=b, seed=seed, normalise=normalise)


def uniform_binomial_multipliers(*, b: int = 2) -> MultiplierSampler:
    r"""Legacy sampler: children share the parent mass in Dirichlet(1,...,1) proportions.

    See :func:`random_binomial_cascade`.
    """
    if b < 2:
        raise ValueError(f"b must be >= 2, got {b}")

    def sample(k: int, rng: np.random.Generator) -> FloatArray:
        parents = b**k
        if b == 2:
            # Exactly the legacy construction: m ~ U(0,1), siblings (m, 1-m).
            u = rng.uniform(0.0, 1.0, size=parents)
            m = np.empty(2 * parents, dtype=np.float64)
            m[0::2] = u
            m[1::2] = 1.0 - u
            return m
        return np.asarray(rng.dirichlet(np.ones(b), size=parents).ravel(), dtype=np.float64)

    return sample


def random_binomial_cascade(kmax: int, *, b: int = 2, seed: SeedLike = None) -> FloatArray:
    r"""Legacy microcanonical cascade: :math:`(M, 1-M)` with :math:`M\sim U(0,1)`.

    Kept for backwards compatibility with the first version of this library,
    which used exactly this construction. It is *microcanonical* by definition
    (siblings sum to the parent mass pathwise), so the total mass is exactly 1
    and no normalisation is applied or needed.

    The marginal multiplier is uniform, so :math:`\mathbb{E}[M^q]=1/(1+q)` for
    :math:`q>-1` and, treating the level moments as a product (exact in
    expectation for :math:`b=2` because siblings enter :math:`Z` additively),

    .. math::
        \tau(q) \;=\; \log_2(1+q) - 1 , \qquad q > -1 ,

    with :math:`\tau(0)=-1`, :math:`\tau(1)=0` and
    :math:`\alpha_0 = \tau'(1) = 1/(2\ln 2) \approx 0.721`. All moments with
    :math:`q\le-1` diverge, because :math:`M` has positive density at the
    origin. For ``b > 2`` the uniform pair generalises to a flat Dirichlet, for
    which :math:`\mathbb{E}[M^q]=B(1+q,b-1)/B(1,b-1)`.

    This family is *not* a good default: it has no free parameter, so its
    multifractality cannot be tuned, and its microcanonical nature makes it a
    poor test bed for canonical-cascade theory. Prefer
    :func:`lognormal_cascade`.
    """
    return cascade_measure(kmax, uniform_binomial_multipliers(b=b), b=b, seed=seed, normalise=False)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def normalise_cascade(
    mu: FloatArray,
    *,
    mode: NormalisationMode,
    sigma_omega: float = 1.0,
    seed: SeedLike = None,
) -> FloatArray:
    r"""Rescale a cascade measure's total mass.

    Parameters
    ----------
    mode
        ``"unit_mass"``
            Divide by :math:`\sum_i \mu_i`. Deterministic, and the right choice
            whenever the measure is about to be used as a time deformation
            (:func:`trading_time`) where only the shape matters.
        ``"omega"``
            Multiply the whole measure by a *single* strictly positive random
            variable :math:`\Omega`, drawn lognormal with
            :math:`\mathbb{E}[\Omega]=1`:
            :math:`\Omega=\exp(N(-s^2/2, s^2))`. This is the correct surrogate
            for the canonical cascade's random total mass (Mandelbrot 1974;
            Calvet--Fisher 2002 use exactly this device to give the MMAR a
            random overall level). Being a single common factor it multiplies
            every :math:`\mu_i^q` by :math:`\Omega^q` and therefore shifts
            :math:`\log Z(q,\delta)` by a constant --- it changes the
            *intercept* of every log--log fit and leaves :math:`\tau(q)`
            untouched. That invariance is the whole point.
        ``"none"``
            Return a copy unchanged.
    sigma_omega
        Log-scale :math:`s` of :math:`\Omega`; ignored unless ``mode="omega"``.

    Notes
    -----
    **Why the legacy normalisation was wrong.** The predecessor multiplied the
    finest level cellwise by ``np.random.normal(2**(lam*kmax), sigma, 2**kmax)``,
    i.e. by an i.i.d. Gaussian field :math:`G_i`. Three separate failures:

    1. *It is not positive.* A Gaussian with mean :math:`m` and sd :math:`s`
       returns a negative value with probability :math:`\Phi(-m/s)`, which for
       modest ``lam*kmax`` is far from negligible over :math:`2^{k_{\max}}`
       independent draws. A single negative cell makes :math:`\mu` not a
       measure, and every :math:`\mu_i^q` with non-integer :math:`q` becomes NaN.
    2. *It injects white noise into the scaling.* An i.i.d. factor is not a
       common factor: :math:`\sum_i (\mu_i G_i)^q = \sum_i \mu_i^q G_i^q`
       concentrates on :math:`\mathbb{E}[G^q]\sum_i\mu_i^q`, so it multiplies
       :math:`Z(q,\delta)` by a *q-dependent* constant and, worse, adds a
       fluctuation of relative size :math:`\delta^{1/2}` that biases the
       small-scale end of every log--log fit.
    3. *The mean is astronomically large.* :math:`2^{\lambda k_{\max}}` with
       :math:`\lambda\approx1.05,\ k_{\max}=20` is :math:`\sim 4\times10^6`,
       so the "normalised" measure has nothing to do with unit mass.

    The correct construction is the one implemented here: one scalar
    :math:`\Omega`, positive by construction.
    """
    m = validate_measure(mu, name="cascade measure", normalise=False)
    if mode == "none":
        return np.ascontiguousarray(m.copy())
    if mode == "unit_mass":
        return np.ascontiguousarray(m / float(m.sum()))
    if mode == "omega":
        if sigma_omega <= 0.0:
            raise ValueError(f"sigma_omega must be strictly positive, got {sigma_omega}")
        rng = make_rng(seed)
        omega = float(np.exp(rng.normal(-0.5 * sigma_omega**2, sigma_omega)))
        if not math.isfinite(omega) or omega <= 0.0:  # pragma: no cover - exp of a finite normal
            raise ValueError(f"drew a non-positive total mass Omega={omega}")
        return np.ascontiguousarray(m * omega)
    raise ValueError(f"unknown mode {mode!r}; use 'unit_mass', 'omega' or 'none'")


# --------------------------------------------------------------------------- #
# Trading time
# --------------------------------------------------------------------------- #


def trading_time(mu: FloatArray, *, t_max: float = 1.0) -> FloatArray:
    r"""The cumulative measure :math:`\theta(t)=\mu([0,t])`, rescaled to :math:`[0,t_{\max}]`.

    This is the MMAR's *trading time*: Mandelbrot, Fisher and Calvet (1997)
    compound a self-affine motion :math:`B_H` with a multifractal time
    deformation, :math:`X(t)=B_H(\theta(t))`, and it is :math:`\theta` that
    carries the multifractality. :math:`\theta` is a continuous, non-decreasing,
    singular function --- the multifractal analogue of the Cantor devil's
    staircase --- whose increments over :math:`n` equal cells are exactly the
    cascade cell masses.

    Returns
    -------
    FloatArray
        Shape ``(len(mu) + 1,)``: the CDF evaluated on the ``len(mu)+1`` cell
        boundaries, so ``theta[0] == 0`` and ``theta[-1] == t_max``. Returning
        ``n+1`` points rather than ``n`` is deliberate: :math:`\theta` is a
        function of the *grid*, and ``np.diff(theta)`` must reproduce the
        measure.

    Raises
    ------
    ValueError
        If the input is not a valid measure, or if floating-point accumulation
        breaks monotonicity (it cannot, given the construction below, but the
        invariant is checked rather than assumed).

    Notes
    -----
    The normalising total is the *cumulative sum's own* endpoint, not
    ``m.sum()``. The two differ: :func:`numpy.sum` uses pairwise summation while
    :func:`numpy.cumsum` is strictly sequential, so for a heterogeneous measure
    with many cells they disagree by up to :math:`O(n\epsilon)` --- at
    ``kmax = 20`` with weights ``(0.9, 0.1)`` the gap is :math:`2\times10^{-11}`
    relative. Dividing the sequential cumulative sum by the pairwise total lets
    the last few points overshoot ``t_max``, and pinning ``theta[-1] = t_max``
    on top of that manufactures a *negative* final increment out of a perfectly
    valid measure. Normalising by the cumulative sum's own endpoint removes the
    inconsistency at the source; the clip is the remaining one-ulp guard.
    """
    m = validate_measure(mu, name="cascade measure", normalise=False)
    if not math.isfinite(t_max) or t_max <= 0.0:
        raise ValueError(f"t_max must be finite and strictly positive, got {t_max}")
    theta = np.empty(m.size + 1, dtype=np.float64)
    theta[0] = 0.0
    np.cumsum(m, out=theta[1:])
    total = float(theta[-1])
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"the measure accumulated to total mass {total}; cannot rescale")
    theta *= t_max / total
    # Scaling by a positive constant is monotone under round-to-nearest, so the
    # array is still non-decreasing; only the endpoint can land one ulp above
    # t_max, and clipping an ordered array from above preserves the order.
    np.minimum(theta, t_max, out=theta)
    theta[0] = 0.0
    theta[-1] = t_max
    if np.any(np.diff(theta) < 0.0):
        raise ValueError("trading time is not non-decreasing; the input was not a measure")
    if theta[0] != 0.0:  # pragma: no cover - assigned literally above
        raise ValueError(f"trading time must start at 0, got {theta[0]!r}")
    return theta


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def cascade_moments(mu: FloatArray, q: Sequence[float] | FloatArray, b: int = 2) -> FloatArray:
    r"""Empirical partition function :math:`\log_b Z(q,\delta)` at every :math:`b`-adic level.

    The measure is coarse-grained by successive aggregation of :math:`b`
    neighbouring cells, giving the exact box partition at every resolution
    :math:`\delta = b^{-l}` for :math:`l = 0,1,\dots,k_{\max}` with
    :math:`k_{\max}=\log_b(\text{len}(\mu))`. At each level the partition
    function :math:`Z(q,\delta)=\sum_i \mu_i^q` is returned in :math:`\log_b`.

    Since :math:`Z(q,\delta)\sim\delta^{\tau(q)}` and
    :math:`\log_b\delta = -l`, the exponent is recovered as

    .. math::
        \tau(q) \;=\; -\,\frac{\partial \log_b Z(q,\delta)}{\partial l} ,

    i.e. **minus** the slope of the returned row against the level index. For a
    deterministic cascade the relation is exact at every :math:`l`, not merely
    asymptotic, so this function doubles as a machine-precision check on
    :math:`\tau`.

    Parameters
    ----------
    mu
        Measure of length ``b**kmax``.
    q
        Moment orders, **strictly increasing**. At least three values, as
        elsewhere in the library.
    b
        Branching number used to aggregate.

    Returns
    -------
    FloatArray
        Shape ``(len(q), kmax + 1)``; row ``i`` corresponds to ``q[i]`` in the
        order given. Column ``l`` is :math:`\log_b Z(q,b^{-l})`; column ``0`` is
        the single-box level, where :math:`Z = (\text{total mass})^q`.

    Raises
    ------
    ValueError
        If ``q`` is not strictly increasing. The row-to-``q`` correspondence is
        positional and there is no returned ``q`` axis to check it against, so a
        grid that the shared validator would silently sort or de-duplicate is
        refused rather than answered: a descending grid would otherwise return
        rows in reverse order, and comparing them term-by-term against
        :func:`mmar.models.theory.cascade_tau` --- which preserves the caller's
        order --- gives a wrong answer with no diagnostic.

    Notes
    -----
    The sums are formed with :func:`scipy.special.logsumexp` over
    :math:`q\log\mu_i`, not as ``(mu**q).sum()``. The naive form is unusable at
    the :math:`|q|` this library allows: on a ``kmax=20`` lognormal cascade the
    smallest normalised cell is :math:`\sim4\times10^{-9}`, so ``(mu**q).sum()``
    reaches :math:`10^{253}` by :math:`q=-30` and flushes to exactly zero for
    :math:`q\gtrsim80`, while the log-domain form stays exact. This is a
    diagnostic, not an estimator --- it does no regression, no scaling-range
    selection and no error bars. Use :mod:`mmar.estimators` for inference.

    Empty cells are handled explicitly rather than by convention: at
    :math:`q=0` the sum is the *count of non-empty boxes* (the box-counting
    quantity whose exponent is :math:`-D_0`), and any :math:`q<0` with an empty
    cell present raises, because :math:`Z(q)` genuinely diverges there.
    """
    m = validate_measure(mu, name="cascade measure", normalise=False)
    q_in = np.asarray(q, dtype=np.float64).ravel()
    qa = validate_q_grid(q_in)
    if qa.shape != q_in.shape or not np.array_equal(qa, q_in):
        raise ValueError(
            "q must be strictly increasing: the rows of the returned array are matched to q "
            "positionally, and the shared validator sorts and de-duplicates, so a grid like "
            f"{q_in.tolist()} would silently be answered as {qa.tolist()}. "
            "Pass np.sort(np.unique(q))."
        )
    if b < 2:
        raise ValueError(f"b must be >= 2, got {b}")
    n = m.size
    kmax = round(math.log(n, b))
    if b**kmax != n:
        raise ValueError(f"measure length {n} is not a power of b={b}")
    empty = int((m == 0.0).sum())
    if empty and np.any(qa < 0.0):
        raise ValueError(
            f"the measure has {empty} empty cells, so Z(q) diverges for every q < 0. "
            "Restrict the q grid to q >= 0, or drop the empty cells explicitly."
        )

    levels: list[FloatArray] = []
    cur = m
    while True:
        levels.append(cur)
        if cur.size == 1:
            break
        cur = np.asarray(cur.reshape(-1, b).sum(axis=1), dtype=np.float64)
    levels.reverse()

    out = np.empty((qa.size, kmax + 1), dtype=np.float64)
    ln_b = math.log(b)
    for lvl, cells in enumerate(levels):
        with np.errstate(divide="ignore"):
            log_cells = np.log(cells[cells > 0.0]) if empty else np.log(cells)
        # q = 0 is box counting: Z(0) is the number of *non-empty* boxes, so the
        # 0 * (-inf) that the log-domain form would produce must not be taken.
        n_boxes = float(log_cells.size)
        for i, qi in enumerate(qa):
            out[i, lvl] = (
                math.log(n_boxes) / ln_b if qi == 0.0 else float(logsumexp(qi * log_cells)) / ln_b
            )
    return out


# --------------------------------------------------------------------------- #
# Specified, provenance-carrying generation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CascadeSpec:
    """A fully reproducible description of one cascade realisation.

    The seed is restricted to ``int | None`` on purpose: a live
    :class:`numpy.random.Generator` carries no recoverable seed, so a spec built
    around one could not be replayed and its provenance root would be a lie.
    """

    family: CascadeFamily
    kmax: int
    b: int = 2
    params: Mapping[str, float] = field(default_factory=dict)
    weights: tuple[float, ...] | None = None
    """Fixed weight vector; required by, and only used by, the deterministic family."""
    conservative: bool = False
    seed: int | None = None
    normalisation: NormalisationMode = "none"
    sigma_omega: float = 1.0

    def to_canonical(self) -> dict[str, Any]:
        return {
            "family": str(self.family),
            "kmax": self.kmax,
            "b": self.b,
            "params": {k: float(v) for k, v in sorted(self.params.items())},
            "weights": list(self.weights) if self.weights is not None else None,
            "conservative": self.conservative,
            "seed": self.seed,
            "normalisation": self.normalisation,
            "sigma_omega": self.sigma_omega,
        }


@dataclass(frozen=True, slots=True)
class CascadeResult:
    """A generated measure with the spec and the audit record that produced it."""

    measure: FloatArray
    spec: CascadeSpec
    provenance: Provenance

    @property
    def total_mass(self) -> float:
        return float(self.measure.sum())

    @property
    def n_cells(self) -> int:
        return int(self.measure.size)

    def trading_time(self, *, t_max: float = 1.0) -> FloatArray:
        """Convenience wrapper around :func:`trading_time`."""
        return trading_time(self.measure, t_max=t_max)


def _require(params: Mapping[str, float], name: str, family: CascadeFamily) -> float:
    if name not in params:
        raise ValueError(f"family {family!s} requires params[{name!r}]; got keys {sorted(params)}")
    return float(params[name])


def generate(spec: CascadeSpec) -> CascadeResult:
    """Realise a cascade from a spec and attach content-addressed provenance.

    The provenance root is a pure function of the family, ``kmax``, ``b``, the
    parameters, the conservativeness flag, the normalisation mode and the seed,
    so two calls with the same spec produce the same root *and* the same array.
    The digest of the produced measure is attached as a non-identifying
    diagnostic, which is what makes the reproducibility claim checkable rather
    than asserted.
    """
    family = CascadeFamily(spec.family)
    p = spec.params
    generator_seed: SeedLike
    omega_seed: SeedLike
    if spec.normalisation == "omega":
        generator_seed, omega_seed = spawn_rngs(spec.seed, 2)
    else:
        generator_seed, omega_seed = spec.seed, None

    if family is CascadeFamily.DETERMINISTIC:
        if spec.weights is None:
            raise ValueError("the deterministic family requires spec.weights")
        mu = deterministic_cascade(spec.kmax, np.asarray(spec.weights, dtype=np.float64), b=spec.b)
    elif family is CascadeFamily.LOGNORMAL:
        mu = lognormal_cascade(
            spec.kmax,
            _require(p, "lam", family),
            float(p["sigma2"]) if "sigma2" in p else None,
            b=spec.b,
            seed=generator_seed,
            conservative=spec.conservative,
            normalise=False,
        )
    elif family is CascadeFamily.LOG_POISSON:
        mu = log_poisson_cascade(
            spec.kmax,
            _require(p, "gamma", family),
            _require(p, "delta", family),
            b=spec.b,
            seed=generator_seed,
            conservative=spec.conservative,
            normalise=False,
        )
    elif family is CascadeFamily.LOG_GAMMA:
        mu = log_gamma_cascade(
            spec.kmax,
            _require(p, "shape", family),
            _require(p, "scale", family),
            b=spec.b,
            seed=generator_seed,
            conservative=spec.conservative,
            normalise=False,
        )
    elif family is CascadeFamily.LOG_STABLE:
        mu = log_stable_cascade(
            spec.kmax,
            _require(p, "alpha", family),
            _require(p, "sigma", family),
            b=spec.b,
            seed=generator_seed,
            conservative=spec.conservative,
            normalise=False,
        )
    else:
        mu = random_binomial_cascade(spec.kmax, b=spec.b, seed=generator_seed)

    if spec.normalisation != "none":
        mu = normalise_cascade(
            mu, mode=spec.normalisation, sigma_omega=spec.sigma_omega, seed=omega_seed
        )

    inputs: tuple[InputRef, ...] = ()
    if spec.weights is not None:
        inputs = (InputRef.from_array("weights", np.asarray(spec.weights, dtype=np.float64)),)

    prov = new_provenance(
        f"cascade.{family!s}",
        estimator_version=MODULE_VERSION,
        params=spec.to_canonical(),
        inputs=inputs,
        seed=seed_info(spec.seed),
        notes=(
            "measure view: tau(q) defined by Z(q,delta) ~ delta^tau(q); "
            "zeta(q) = tau(q) + 1 for this one-dimensional support",
        ),
    ).with_diagnostics(
        measure_digest=array_digest(mu),
        n_cells=int(mu.size),
        total_mass=float(mu.sum()),
    )
    return CascadeResult(measure=mu, spec=spec, provenance=prov)
