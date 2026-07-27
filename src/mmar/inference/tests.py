r"""Hypothesis tests: does a multifractal claim survive contact with a null?

A wide :math:`f(\alpha)`, a curved :math:`\hat\zeta(q)` and a nonlinear
:math:`h(q)` are *not* evidence of multifractality. Every one of them is
produced, in abundance, by processes that are provably monofractal. The three
standard false positives, in decreasing order of how often they are published:

**Heavy tails.** If :math:`\mathbb E|\Delta X|^{q}=\infty` for :math:`q\ge
q_{\mathrm{crit}}` --- as for i.i.d. :math:`\alpha`-stable increments, where
:math:`q_{\mathrm{crit}}=\alpha` --- the sample structure function above
:math:`q_{\mathrm{crit}}` is dominated by the single largest increment and the
measured exponent saturates at :math:`\hat\zeta(q)\to 1`. The resulting
:math:`\hat\zeta` is a concave broken line and :math:`\Delta\alpha` is large,
for a process with i.i.d. increments and no temporal structure whatsoever
(Mandelbrot 1997; Lux 2004). :func:`moment_existence` is the guard rail, and
the shuffle surrogate test is the detector.

**Linear correlation.** A Gaussian linear process with a power-law spectrum
(fGn) has :math:`\zeta(q)=qH` exactly, but at finite :math:`n` its estimated
:math:`\hat\zeta` is curved, with the curvature growing as the memory
lengthens. IAAFT surrogates hold the autocorrelation fixed and so subtract
this.

**A crossover mistaken for a power law.** :math:`R^2=0.99` on a log--log plot
is compatible with visible curvature over two decades. :func:`scaling_range_test`
tests linearity rather than correlation.

The doctrine of this module
---------------------------
A statistic is *not* interpreted against its own sampling distribution under a
parametric model. It is recomputed, by the identical code path, on surrogate
series drawn from an explicit null, and the observed value is placed in that
ensemble. Two nulls matter and they answer different questions:

===========  ==========================================  ==============================
Null         Preserves                                   A small :math:`p` means
===========  ==========================================  ==============================
``shuffle``  the marginal law of the increments          *some* time ordering is needed
``iaaft``    the marginal law **and** the autocovariance *nonlinear* structure is needed
===========  ==========================================  ==============================

Only the second is evidence of genuine multifractal structure, and only the
conjunction of the two is interpretable, which is why
:func:`multifractality_report` runs both.

References
----------
Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., Farmer, J.D. (1992).
Testing for nonlinearity in time series: the method of surrogate data.
*Physica D* **58**, 77--94.

Schreiber, T., Schmitz, A. (1996). Improved surrogate data for nonlinearity
tests. *Phys. Rev. Lett.* **77**, 635--638.

Schreiber, T., Schmitz, A. (2000). Surrogate time series. *Physica D* **142**,
346--382.

Davison, A.C., Hinkley, D.V. (1997). *Bootstrap Methods and their Application*,
§4.2 (the :math:`(1+\#)/(1+B)` Monte Carlo p-value).

Hill, B.M. (1975). A simple general approach to inference about the tail of a
distribution. *Ann. Statist.* **3**, 1163--1174.

Drees, H., de Haan, L., Resnick, S. (2000). How to make a Hill plot.
*Ann. Statist.* **28**, 254--274.

Ramsey, J.B. (1969). Tests for specification errors in classical linear
least-squares regression analysis. *J. R. Stat. Soc. B* **31**, 350--371.

Benjamini, Y., Hochberg, Y. (1995). Controlling the false discovery rate.
*J. R. Stat. Soc. B* **57**, 289--300.

Lux, T. (2004). Detecting multifractal properties in asset returns: the failure
of the "scaling estimator". *Int. J. Mod. Phys. C* **15**, 481--491.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mmar.core.legendre import concave_envelope, concavity_defect, spectrum_from_exponents
from mmar.core.provenance import InputRef, Provenance, SeedInfo, new_provenance
from mmar.core.regression import loglog_fit, select_scaling_range
from mmar.core.rng import SeedLike, make_rng, seed_info, spawn_rngs
from mmar.core.types import FitMethod, FloatArray, ScalingExponents, StructureFunctions
from mmar.core.validate import validate_series

__all__ = [
    "Estimator",
    "MomentDiagnostic",
    "MultifractalityReport",
    "Statistic",
    "SurrogateBasis",
    "SurrogateKind",
    "TestResult",
    "concavity_test",
    "empirical_p_value",
    "moment_existence",
    "monofractal_test",
    "multifractality_report",
    "multifractality_test",
    "scaling_range_test",
    "spectrum_statistic",
]

MODULE_VERSION: Final = "1"

Estimator = Callable[[FloatArray], ScalingExponents]
r"""Anything that maps a series to fitted exponents :math:`\hat\zeta(q)`.

Deliberately a callable rather than a named estimator: the surrogate test must
apply *exactly* the same code path, the same :math:`q` grid and the same scaling
range to the data and to every surrogate. Passing a ``functools.partial`` of the
estimator actually used on the data is the only way to guarantee that.
"""

Statistic = Literal["width", "c2", "hq_range"]
SurrogateKind = Literal["shuffle", "iaaft"]
SurrogateBasis = Literal["increments", "values"]

_LEVEL: Final = 0.05
"""Reference level used only to phrase the ``interpretation`` string."""

_MAX_SURROGATE_FAILURES: Final = 0.2
"""Refuse to report a p-value if more than this fraction of surrogates failed."""


# --------------------------------------------------------------------------- #
# Result carriers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TestResult:
    """One hypothesis test, with the null it was taken against.

    ``statistic`` is always oriented so that **larger means more multifractal**
    and ``p_value`` is always the corresponding upper-tail probability. That
    convention is what makes results from different statistics comparable and
    keeps the sign of the evidence unambiguous.

    ``null_distribution`` is retained in full: a p-value without the ensemble
    behind it cannot be re-levelled, plotted, or checked for degeneracy (a null
    distribution with zero spread produces a meaningless p-value, and the only
    way to notice is to look at it).
    """

    statistic: float
    p_value: float
    null_distribution: FloatArray
    n_surrogates: int
    interpretation: str
    name: str = ""
    null_model: str = ""
    dof: float | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    provenance: Provenance | None = None

    @property
    def null_mean(self) -> float:
        d = self.null_distribution
        return float(np.mean(d)) if d.size else float("nan")

    @property
    def null_sd(self) -> float:
        d = self.null_distribution
        return float(np.std(d, ddof=1)) if d.size > 1 else float("nan")

    @property
    def z_score(self) -> float:
        """Standardised effect size against the null ensemble.

        Reported alongside the p-value because with :math:`B` surrogates the
        smallest attainable p-value is :math:`1/(B+1)`; once the observed
        statistic is outside the ensemble entirely, only :math:`z` still
        discriminates between "just outside" and "absurdly outside".
        """
        sd = self.null_sd
        return (self.statistic - self.null_mean) / sd if sd > 0 else float("nan")

    def rejects(self, level: float = _LEVEL) -> bool:
        return bool(self.p_value <= level)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "null_model": self.null_model,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n_surrogates": self.n_surrogates,
            "dof": self.dof,
        }

    def summary(self) -> dict[str, Any]:
        return {
            **self.to_canonical(),
            "null_mean": self.null_mean,
            "null_sd": self.null_sd,
            "z": self.z_score,
            "interpretation": self.interpretation,
        }


@dataclass(frozen=True, slots=True)
class MultifractalityReport:
    """The two surrogate tests together; neither is interpretable alone.

    See :attr:`verdict` for the 2x2 reading. The pair is the minimum publishable
    unit: rejecting the shuffle null establishes only that the time ordering
    matters, and failing to reject it invalidates any reading of the IAAFT test.
    """

    statistic: str
    shuffle: TestResult
    iaaft: TestResult
    provenance: Provenance | None = None

    @property
    def verdict(self) -> str:
        """The joint reading of the two nulls."""
        s, i = self.shuffle.rejects(), self.iaaft.rejects()
        if not s and not i:
            return (
                "NO EVIDENCE: a series with the same marginal distribution and no time "
                "ordering at all already reproduces the statistic. Heavy tails, not "
                "multifractality."
            )
        if s and not i:
            return (
                "LINEAR: the time ordering matters, but a Gaussian linear process with the "
                "same marginal and the same autocovariance reproduces the statistic. No "
                "evidence of nonlinear multifractal structure."
            )
        if s and i:
            return (
                "NONLINEAR: neither the marginal distribution nor the linear "
                "autocorrelation reproduces the statistic, so some nonlinear "
                "dependence is required. This is the only combination consistent "
                "with a cascade -- but it is NECESSARY, not sufficient: a GARCH(1,1) "
                "reaches it too, with genuine nonlinear dependence and no "
                "multiplicative cascade anywhere in it. To go further, compare the "
                "estimated f(alpha) against a calibrated cascade's closed form "
                "(mmar.calibration) rather than resting on this p-value."
            )
        return (
            "ANOMALOUS: the IAAFT null is rejected while the (weaker) shuffle null is not. "
            "The linear autocorrelation moves the statistic away from the observed value "
            "rather than towards it; suspect a scaling-range or moment-existence problem "
            "before reading this as structure."
        )

    def summary(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "p_shuffle": self.shuffle.p_value,
            "p_iaaft": self.iaaft.p_value,
            "observed": self.shuffle.statistic,
            "verdict": self.verdict,
        }


@dataclass(frozen=True, slots=True)
class MomentDiagnostic:
    r"""Which of the requested moments the sample can actually support.

    ``q_crit`` estimates the tail index :math:`\alpha` of :math:`|\Delta X|`, so
    :math:`\mathbb E|\Delta X|^q<\infty` requires :math:`q<\alpha`.
    ``q_crit_negative`` is the analogous bound for negative moments, obtained
    from the tail index of :math:`1/|\Delta X|`, i.e. from the concentration of
    increments *near zero*.
    """

    q: FloatArray
    exists: NDArray[np.bool_]
    q_crit: float
    q_crit_ci: tuple[float, float]
    q_crit_negative: float
    k_selected: int
    k_grid: NDArray[np.int64]
    hill: FloatArray
    """The Hill plot :math:`\\hat\\alpha(k)`, aligned with ``k_grid``. Look at it."""
    n_used: int
    n_zero_increments: int
    provenance: Provenance | None = None

    @property
    def admissible_range(self) -> tuple[float, float]:
        """The open interval of :math:`q` with finite moments."""
        return (-self.q_crit_negative, self.q_crit)

    def rejected_q(self) -> FloatArray:
        r"""The requested ``q`` ruled out by the fitted tail index.

        .. warning::

            This is *not* a certificate that the moment diverges. ``q_crit`` is
            a Hill estimate, and the Hill estimator returns a finite number for
            any sample, including one drawn from a law with all moments finite:
            for Gaussian increments the estimate drifts upward with :math:`k`
            and settles near :math:`\hat\alpha\approx5` at :math:`n=2\times10^4`,
            so ``rejected_q`` then lists :math:`q=6,8,12` --- all of which exist.
            Treat a rejection as "the sample cannot support this moment", which
            is the operationally relevant statement, and read
            :attr:`hill` before concluding anything about the law. Only when the
            Hill plot has a genuine plateau is ``q_crit`` an index rather than a
            sample-size-dependent floor.
        """
        return np.asarray(self.q[~self.exists], dtype=np.float64)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "q_crit": self.q_crit,
            "q_crit_ci": list(self.q_crit_ci),
            "q_crit_negative": self.q_crit_negative,
            "k_selected": self.k_selected,
            "n_used": self.n_used,
        }

    def summary(self) -> dict[str, Any]:
        return {**self.to_canonical(), "rejected_q": self.rejected_q().tolist()}


# --------------------------------------------------------------------------- #
# p-values
# --------------------------------------------------------------------------- #


def empirical_p_value(observed: float, null: FloatArray) -> float:
    r"""Upper-tail Monte Carlo p-value with the :math:`(1+\#)/(1+B)` correction.

    .. math::
        \hat p = \frac{1 + \#\{b : T_b \ge T_{\mathrm{obs}}\}}{1 + B}.

    The observed value is counted as one draw from the null, which is what makes
    :math:`\hat p` exactly valid (:math:`\Pr(\hat p\le u)\le u` for every
    :math:`u`) at finite :math:`B` under exchangeability of the surrogates with
    the data (Davison & Hinkley 1997, §4.2). The naive
    :math:`\#\{T_b\ge T_{\mathrm{obs}}\}/B` can return exactly zero, which is not
    a probability and cannot be corrected for multiplicity afterwards.

    Non-finite surrogate values are dropped from both the numerator and the
    denominator; the caller is responsible for reporting how many.
    """
    d = np.asarray(null, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    return float((1.0 + np.count_nonzero(d >= observed)) / (1.0 + d.size))


def _benjamini_hochberg(p: FloatArray, level: float) -> NDArray[np.bool_]:
    """BH step-up rejections at FDR ``level``; NaN p-values are never rejected."""
    out = np.zeros(p.shape, dtype=bool)
    finite = np.isfinite(p)
    m = int(finite.sum())
    if m == 0:
        return out
    idx = np.nonzero(finite)[0]
    order = idx[np.argsort(p[idx], kind="stable")]
    thresh = level * np.arange(1, m + 1, dtype=np.float64) / m
    passing = np.nonzero(p[order] <= thresh)[0]
    if passing.size:
        out[order[: int(passing[-1]) + 1]] = True
    return out


# --------------------------------------------------------------------------- #
# Statistics of a fitted exponent curve
# --------------------------------------------------------------------------- #


def _log_cumulants_c2(q: FloatArray, zeta: FloatArray) -> float:
    r"""Least-squares :math:`c_2` in :math:`\zeta(q)=c_1 q + \tfrac12 c_2 q^2`.

    The expansion :math:`\zeta(q)=\sum_{p\ge1}c_p q^p/p!` has no constant term
    (:math:`\zeta(0)=0` identically), so the fit is through the origin; adding an
    intercept would absorb part of the curvature and bias :math:`c_2` towards
    zero. Truncating at :math:`p=2` is a *definition of the statistic*, not a
    claim that the process is lognormal: what matters for a surrogate test is
    that the same functional is applied to the data and to every surrogate.
    """
    design = np.column_stack([q, 0.5 * q**2])
    coef, *_ = np.linalg.lstsq(design, zeta, rcond=None)
    return float(coef[1])


def spectrum_statistic(exponents: ScalingExponents, statistic: Statistic = "width") -> float:
    r"""A scalar measure of multifractality, oriented so that larger is stronger.

    ``"width"``
        :math:`\Delta\alpha=\alpha_{\max}-\alpha_{\min}` of the
        Legendre--Fenchel spectrum built by
        :func:`~mmar.core.legendre.spectrum_from_exponents`. Zero for an exactly
        linear :math:`\zeta`. The most common statistic, and the most sensitive
        to the :math:`q` window: it measures the spread of :math:`\zeta'` over
        the *fitted* :math:`q` range only, so it is not an estimate of the true
        support of :math:`f` unless :math:`q` is wide enough. **NaN when the
        apex of** :math:`f` **lies at or below the level the support is measured
        at**, i.e. when
        :attr:`~mmar.core.types.SpectrumDescriptors.support_truncated` is set by
        an unsupported apex: :math:`\hat\zeta` is then convex, the Legendre
        transform is a tent lying wholly below the axis, both crossings collapse
        onto the apex and ``width`` comes back as exactly zero --- which is also
        what an exact monofractal reports. Handing that zero to
        :func:`multifractality_test` puts a *collapsed estimate* into the null
        comparison as though it were a measurement. Measured on fBm at
        :math:`H=0.3` concatenated with fBm at :math:`H=0.8` (n = 8192,
        non-stationary, Gaussian increments, not multifractal anywhere):
        :math:`f_{\max}=-0.208`, observed width 0.0, all 49 shuffle surrogates
        strictly larger (null mean 0.059, minimum 0.016), :math:`p=1.000`, and
        an ``interpretation`` reading "a randomly reordered copy ... reproduces
        the observed statistic" --- which no surrogate did --- followed by the
        verdict "Heavy tails, not multifractality" for a Gaussian path. The NaN
        routes it to the "not finite on the data itself" guard instead.
    ``"c2"``
        The intermittency coefficient :math:`-c_2\ge0` from
        :func:`_log_cumulants_c2`. Exactly zero for a monofractal, and the least
        noisy of the three because it is a global least-squares functional of
        :math:`\hat\zeta` rather than an extremum of it.
    ``"hq_range"``
        :math:`\max_q h(q)-\min_q h(q)` with :math:`h(q)=\zeta(q)/q`, the
        generalised-Hurst range. Undefined at :math:`q=0`, which is dropped.

    All three are *statistics*, not estimators: they are compared with their own
    surrogate ensembles, so their finite-sample bias --- which is large and
    positive for all three --- cancels to first order provided the surrogates go
    through the identical code path.
    """
    if statistic == "width":
        d = spectrum_from_exponents(exponents).descriptors()
        return float(d.width) if d.f_max > 0.0 else float("nan")
    if statistic == "c2":
        return -_log_cumulants_c2(exponents.q, exponents.zeta)
    if statistic == "hq_range":
        h = exponents.hq
        h = h[np.isfinite(h)]
        if h.size < 2:
            return float("nan")
        return float(h.max() - h.min())
    raise ValueError(f"unknown statistic {statistic!r}; use 'width', 'c2' or 'hq_range'")


# --------------------------------------------------------------------------- #
# Surrogates
# --------------------------------------------------------------------------- #


def _decompose(x: FloatArray, basis: SurrogateBasis) -> tuple[FloatArray, float]:
    """Split the series into the randomisable part and the level to restore."""
    if basis == "increments":
        return np.diff(x), float(x[0])
    if basis == "values":
        return x, 0.0
    raise ValueError(f"unknown basis {basis!r}; use 'increments' or 'values'")


def _recompose(u: FloatArray, x0: float, basis: SurrogateBasis) -> FloatArray:
    if basis == "increments":
        out = np.empty(u.size + 1, dtype=np.float64)
        out[0] = x0
        np.cumsum(u, out=out[1:])
        out[1:] += x0
        return out
    return np.ascontiguousarray(u, dtype=np.float64)


def _iaaft(u: FloatArray, rng: np.random.Generator, *, n_iterations: int) -> FloatArray:
    r"""Iteratively refined amplitude-adjusted Fourier transform surrogate.

    Alternates two projections until they stop fighting (Schreiber & Schmitz
    1996):

    1. *spectral*: replace the Fourier amplitudes of the current iterate by the
       target amplitudes :math:`|\hat u_k|` of the data, keeping its phases;
    2. *amplitude*: rank-order the result onto the sorted values of the data,
       which restores the empirical marginal **exactly**.

    The two constraints are generically incompatible at finite :math:`n`, so the
    loop converges to a compromise. The amplitude step is applied *last*, so the
    returned surrogate has the data's marginal distribution exactly and its
    periodogram approximately --- the standard choice, since a mismatch in the
    marginal is a first-order violation of the null while a small spectral
    mismatch is second-order.

    Failure mode worth knowing: for a strongly non-Gaussian marginal combined
    with long memory, the residual spectral discrepancy is systematic, and the
    test becomes conservative in a way no number of iterations removes. Check
    the reported ``spectral_mismatch``.
    """
    n = u.size
    sorted_values = np.sort(u)
    target = np.abs(np.fft.rfft(u))
    y = rng.permutation(u)
    previous: NDArray[np.intp] | None = None
    for _ in range(n_iterations):
        spectrum = np.fft.rfft(y)
        with np.errstate(invalid="ignore", divide="ignore"):
            phase = np.exp(1j * np.angle(spectrum))
        y = np.fft.irfft(target * phase, n=n)
        # Rank-ordering by scatter rather than a double argsort: y[order] =
        # sorted_values assigns the i-th smallest data value to the position of
        # the i-th smallest iterate, which is the amplitude projection.
        order = np.argsort(y, kind="stable")
        y = np.empty(n, dtype=np.float64)
        y[order] = sorted_values
        if previous is not None and np.array_equal(order, previous):
            break
        previous = order
    return np.ascontiguousarray(y, dtype=np.float64)


def _replayable(seed: SeedLike) -> SeedLike:
    """A seed that spawns the *same* children however many times it is reused.

    :meth:`numpy.random.SeedSequence.spawn` advances an internal counter, so the
    same :class:`~numpy.random.SeedSequence` object passed to two calls produces
    two different ensembles --- while :func:`~mmar.core.rng.seed_info` records
    the same ``(entropy, spawn_key)`` both times and the provenance root
    therefore certifies as one computation two runs that gave different numbers.
    Rebuilding the sequence from exactly the fields the manifest records
    restores the invariant "same root implies same result"; integer, sequence
    and ``None`` seeds already satisfy it, and a live ``Generator`` never can.
    """
    if isinstance(seed, np.random.SeedSequence):
        return np.random.SeedSequence(entropy=seed.entropy, spawn_key=seed.spawn_key)
    return seed


def _child_seeds(seed: SeedLike, n: int) -> list[SeedLike]:
    """``n`` independent child *seeds*, each still recoverable in a manifest.

    :func:`~mmar.core.rng.spawn_rngs` returns live generators, from which
    :func:`~mmar.core.rng.seed_info` can recover nothing, so a sub-analysis
    seeded that way records ``seed=None``. Spawning
    :class:`~numpy.random.SeedSequence` children instead keeps the parent
    entropy and the spawn key in every child's provenance while giving the same
    guarantee of non-overlapping streams. A live ``Generator`` parent has no
    recoverable seed to begin with, and is passed through as such.
    """
    if isinstance(seed, np.random.Generator):
        return list(spawn_rngs(seed, n))
    if isinstance(seed, np.random.SeedSequence):
        parent = np.random.SeedSequence(entropy=seed.entropy, spawn_key=seed.spawn_key)
    elif seed is None:
        parent = np.random.SeedSequence()
    elif isinstance(seed, int):
        parent = np.random.SeedSequence(seed)
    else:
        parent = np.random.SeedSequence(list(seed))
    return list(parent.spawn(n))


def _surrogate(
    x: FloatArray,
    kind: SurrogateKind,
    rng: np.random.Generator,
    *,
    basis: SurrogateBasis,
    iaaft_iterations: int,
) -> FloatArray:
    u, x0 = _decompose(x, basis)
    if kind == "shuffle":
        v = rng.permutation(u)
    elif kind == "iaaft":
        v = _iaaft(u, rng, n_iterations=iaaft_iterations)
    else:
        raise ValueError(f"unknown surrogate kind {kind!r}; use 'shuffle' or 'iaaft'")
    return _recompose(np.asarray(v, dtype=np.float64), x0, basis)


def _spectral_mismatch(x: FloatArray, y: FloatArray, basis: SurrogateBasis) -> float:
    """Relative :math:`L^2` error between the two periodograms; 0 for an exact match."""
    u, _ = _decompose(x, basis)
    v, _ = _decompose(y, basis)
    pu = np.abs(np.fft.rfft(u)) ** 2
    pv = np.abs(np.fft.rfft(v)) ** 2
    denom = float(np.sum(pu**2))
    return float(np.sqrt(np.sum((pu - pv) ** 2) / denom)) if denom > 0 else float("nan")


# --------------------------------------------------------------------------- #
# The surrogate test
# --------------------------------------------------------------------------- #


def _interpretation(kind: SurrogateKind, statistic: Statistic, p: float, observed: float) -> str:
    rejected = p <= _LEVEL
    head = f"{statistic} = {observed:.4f}, p = {p:.4f} against {kind} surrogates. "
    if kind == "shuffle" and rejected:
        return head + (
            "The statistic is larger than a series with the same marginal distribution and "
            "no time ordering can produce, so some temporal structure is required. This is "
            "NOT yet evidence of multifractality: linear autocorrelation alone reproduces "
            "it routinely. Run the IAAFT test before claiming anything."
        )
    if kind == "shuffle":
        return head + (
            "A randomly reordered copy of the increments -- no temporal structure of any "
            "kind -- reproduces the observed statistic. Whatever the log-log plots look "
            "like, this sample carries no evidence of multifractality; the width is a "
            "property of the increment distribution (heavy tails do exactly this) and not "
            "of the process. Check moment_existence."
        )
    if rejected:
        return head + (
            "Neither the marginal distribution nor the linear autocorrelation function "
            "reproduces the observed statistic, so the excess is attributable to nonlinear "
            "structure. This is the necessary condition for genuine multifractality. It is "
            "not sufficient: confirm the shuffle test also rejects, that the moments exist "
            "over the fitted q range, and that the log-log relation is linear."
        )
    return head + (
        "A Gaussian linear process with the same marginal distribution and the same power "
        "spectrum reproduces the observed statistic. The apparent multifractality is "
        "explained by the autocorrelation together with the marginal; there is no evidence "
        "of nonlinear multifractal structure."
    )


def multifractality_test(
    x: Sequence[float] | NDArray[Any],
    estimator: Estimator,
    *,
    kind: SurrogateKind = "shuffle",
    n_surrogates: int = 200,
    statistic: Statistic = "width",
    seed: SeedLike = None,
    basis: SurrogateBasis = "increments",
    iaaft_iterations: int = 200,
    min_length: int = 64,
) -> TestResult:
    r"""Surrogate test of a multifractality statistic against an explicit null.

    The observed statistic :math:`T_{\mathrm{obs}}` is computed from
    ``estimator(x)``, the same statistic is recomputed on ``n_surrogates``
    surrogate series drawn from the null, and the upper-tail Monte Carlo p-value
    :func:`empirical_p_value` is returned.

    Interpretation --- the point of the whole module
    ------------------------------------------------
    **A small p against SHUFFLE surrogates means the width is not explained by
    the marginal distribution alone.** The shuffle null destroys every temporal
    dependence while preserving the increment distribution exactly, so it is the
    null under which heavy tails --- and nothing else --- generate the observed
    statistic. Failing to reject it is decisive: the apparent multifractality is
    a property of the histogram of increments, not of the process.

    **A small p against IAAFT surrogates means the width is not explained by the
    linear autocorrelation either.** The IAAFT null preserves the marginal *and*
    the autocovariance, i.e. everything a Gaussian linear process can express.
    Only this rejection is evidence of genuine nonlinear multifractal structure.

    Neither result is interpretable alone; :func:`multifractality_report`
    returns both, together with the joint verdict.

    Parameters
    ----------
    x
        The series. With ``basis="increments"`` (the default) it is treated as a
        *path* (a price, a profile, a cumulated measure): the surrogate
        randomises :math:`\Delta X` and re-integrates, preserving ``x[0]``. Pass
        ``basis="values"`` when ``x`` is already a return/increment series, or
        the surrogate would randomise second differences.
    estimator
        ``x -> ScalingExponents``. Bind the *identical* configuration used on the
        data (q grid, scales, fit method, scaling-range selection); a surrogate
        analysed differently from the data tests nothing.
    kind
        ``"shuffle"`` or ``"iaaft"``.
    n_surrogates
        :math:`B`. The smallest attainable p-value is :math:`1/(B+1)`, so
        :math:`B=200` cannot support a claim below :math:`p\approx 0.005`; use
        :math:`B\ge999` for anything that will be corrected for multiplicity.
    seed
        Threaded through :func:`~mmar.core.rng.spawn_rngs`, so the ensemble is
        reproducible from the manifest and the :math:`B` streams are provably
        non-overlapping. Incrementing an integer seed per surrogate --- the usual
        shortcut --- gives correlated PCG64 streams and understates the null
        spread.

    Returns
    -------
    TestResult
        With the whole null ensemble attached.

    Raises
    ------
    ValueError
        If the estimator fails on the data, if more than 20% of surrogates fail,
        or if the null ensemble is degenerate (zero spread), in which case the
        p-value would be an artefact of ties rather than a probability.

    Notes
    -----
    Cost is :math:`B+1` estimator evaluations and is dominated by the estimator.
    """
    xa = validate_series(x, name="x", min_length=min_length)
    if n_surrogates < 1:
        raise ValueError(f"n_surrogates must be at least 1, got {n_surrogates}")

    observed_exponents = estimator(xa)
    observed = spectrum_statistic(observed_exponents, statistic)
    if not math.isfinite(observed):
        raise ValueError(
            f"the {statistic!r} statistic is not finite on the data itself; the exponent "
            "estimate has failed before any surrogate was drawn"
        )

    rngs = spawn_rngs(_replayable(seed), n_surrogates)
    null = np.full(n_surrogates, np.nan, dtype=np.float64)
    mismatch = np.full(n_surrogates, np.nan, dtype=np.float64)
    failures: list[str] = []
    for i, rng in enumerate(rngs):
        surrogate = _surrogate(xa, kind, rng, basis=basis, iaaft_iterations=iaaft_iterations)
        if kind == "iaaft":
            mismatch[i] = _spectral_mismatch(xa, surrogate, basis)
        try:
            null[i] = spectrum_statistic(estimator(surrogate), statistic)
        except (ValueError, np.linalg.LinAlgError) as exc:
            failures.append(f"{type(exc).__name__}: {exc}")

    n_ok = int(np.isfinite(null).sum())
    if n_ok < (1.0 - _MAX_SURROGATE_FAILURES) * n_surrogates:
        raise ValueError(
            f"the estimator failed on {n_surrogates - n_ok}/{n_surrogates} surrogates; "
            f"the null ensemble is not usable. First failure: {failures[0] if failures else '?'}"
        )
    finite_null = null[np.isfinite(null)]
    if float(np.std(finite_null)) <= 0.0:
        raise ValueError(
            "the null distribution is degenerate (zero spread): every surrogate produced "
            "the identical statistic, so the p-value would measure ties, not evidence"
        )

    p = empirical_p_value(observed, null)
    prov = _derive_or_new(
        observed_exponents.provenance,
        "inference.multifractality_test",
        params={
            "kind": str(kind),
            "statistic": str(statistic),
            "basis": str(basis),
            "n_surrogates": int(n_surrogates),
            "iaaft_iterations": int(iaaft_iterations) if kind == "iaaft" else None,
            "estimator": observed_exponents.estimator,
        },
        inputs=(InputRef.from_array("x", xa),),
        seed=seed_info(seed),
    )
    return TestResult(
        statistic=observed,
        p_value=p,
        null_distribution=finite_null,
        n_surrogates=n_ok,
        interpretation=_interpretation(kind, statistic, p, observed),
        name=f"multifractality[{statistic}]",
        null_model=kind,
        diagnostics={
            "n_failed": n_surrogates - n_ok,
            "spectral_mismatch": float(np.nanmedian(mismatch)) if kind == "iaaft" else None,
            "estimator": observed_exponents.estimator,
        },
        provenance=prov,
    )


def multifractality_report(
    x: Sequence[float] | NDArray[Any],
    estimator: Estimator,
    *,
    n_surrogates: int = 200,
    statistic: Statistic = "width",
    seed: SeedLike = None,
    basis: SurrogateBasis = "increments",
    iaaft_iterations: int = 200,
) -> MultifractalityReport:
    """Both nulls, because a single one cannot be interpreted.

    Runs :func:`multifractality_test` with ``kind="shuffle"`` and
    ``kind="iaaft"`` on independently spawned seeds, and pairs them with
    :attr:`MultifractalityReport.verdict`. Only the combination
    *shuffle rejected and IAAFT rejected* supports a multifractality claim; the
    combination *shuffle not rejected* refutes it outright, whatever the IAAFT
    test says.

    Cost is :math:`2B+2` estimator evaluations, the IAAFT half additionally
    paying :math:`O(B\\,n\\log n)` per iteration of the surrogate loop.
    """
    seed_shuffle, seed_iaaft = _child_seeds(seed, 2)
    shuffle = multifractality_test(
        x,
        estimator,
        kind="shuffle",
        n_surrogates=n_surrogates,
        statistic=statistic,
        seed=seed_shuffle,
        basis=basis,
    )
    iaaft = multifractality_test(
        x,
        estimator,
        kind="iaaft",
        n_surrogates=n_surrogates,
        statistic=statistic,
        seed=seed_iaaft,
        basis=basis,
        iaaft_iterations=iaaft_iterations,
    )
    prov = new_provenance(
        "inference.multifractality_report",
        estimator_version=MODULE_VERSION,
        params={"statistic": str(statistic), "n_surrogates": int(n_surrogates)},
        inputs=(InputRef.from_array("x", np.asarray(x, dtype=np.float64)),),
        seed=seed_info(seed),
        notes=("shuffle and IAAFT nulls; read the pair, never one alone",),
    )
    return MultifractalityReport(
        statistic=str(statistic), shuffle=shuffle, iaaft=iaaft, provenance=prov
    )


# --------------------------------------------------------------------------- #
# Monofractal (constant h) test
# --------------------------------------------------------------------------- #


def _zeta_stderr(exponents: ScalingExponents) -> FloatArray:
    """The per-``q`` standard error of :math:`\\hat\\zeta`, from wherever it lives."""
    if exponents.zeta_stderr is not None:
        return np.asarray(exponents.zeta_stderr, dtype=np.float64)
    if exponents.fits and len(exponents.fits) == exponents.q.size:
        return np.asarray([f.slope_stderr for f in exponents.fits], dtype=np.float64)
    raise ValueError(
        "no standard errors on zeta: this test is a weighted distance in units of the "
        "fit uncertainty and cannot be formed without it. Re-run the estimator so that "
        "it populates zeta_stderr, or use the surrogate test, which needs no error bars."
    )


def monofractal_test(exponents: ScalingExponents, *, q_tol: float = 1e-8) -> TestResult:
    r"""Wald / chi-square test of :math:`H_0:\ h(q)\ \text{constant}`.

    Under :math:`H_0` the process is monofractal, :math:`\zeta(q)=qH` for a
    single :math:`H`, and the whole exponent curve is a straight line through
    the origin (the intercept is not free: :math:`\zeta(0)=0` identically). With
    :math:`\hat\sigma_i` the reported standard error of :math:`\hat\zeta(q_i)`,
    the generalised-least-squares estimate under independence is

    .. math::
        \hat H = \frac{\sum_i q_i\hat\zeta_i/\hat\sigma_i^2}
                      {\sum_i q_i^2/\hat\sigma_i^2},

    and the test statistic is the residual weighted sum of squares

    .. math::
        X^2 = \sum_i \frac{\big(\hat\zeta_i - \hat H q_i\big)^2}{\hat\sigma_i^2}
        \;\overset{H_0}{\sim}\; \chi^2_{\,m-1},

    with :math:`m` the number of usable :math:`q` values (one degree of freedom
    is spent on :math:`\hat H`). Points with :math:`|q|\le` ``q_tol`` are dropped
    before the fit: :math:`\zeta(0)=0` holds identically under both hypotheses,
    contributes nothing, and its reported standard error is typically zero or
    meaningless, which would otherwise give it infinite weight.

    .. warning::

        **This test is anti-conservative, often severely.** The
        :math:`\hat\zeta(q_i)` are all fitted from the *same* sample over the
        *same* scales, so their errors are strongly positively correlated ---
        empirically :math:`\rho>0.9` between neighbouring :math:`q` is normal.
        The correct statistic is the full quadratic form
        :math:`(\hat\zeta-q\hat H)^\top \Sigma^{-1} (\hat\zeta-q\hat H)` with
        :math:`\Sigma` the covariance of :math:`\hat\zeta` across :math:`q`,
        which requires a block bootstrap of the underlying series. Treating
        :math:`\Sigma` as diagonal, as here, inflates :math:`X^2` and shrinks the
        p-value, typically by orders of magnitude. Additionally the reported
        :math:`\hat\sigma_i` are regression standard errors that ignore the
        estimation of the scaling range itself.

        Use this as a *descriptive* measure of how far :math:`\hat\zeta` is from
        a straight line in units of its own error bars, and take
        :func:`multifractality_test` --- which needs no covariance model,
        because it recomputes the whole pipeline under the null --- as the
        primary evidence.

    Returns
    -------
    TestResult
        ``statistic`` is :math:`X^2` (larger = further from monofractal),
        ``dof`` is :math:`m-1`, ``null_distribution`` is empty (the null is
        parametric, not simulated) and the diagnostics carry :math:`\hat H`.
    """
    q = np.asarray(exponents.q, dtype=np.float64)
    zeta = np.asarray(exponents.zeta, dtype=np.float64)
    sigma = _zeta_stderr(exponents)
    if sigma.shape != q.shape:
        raise ValueError(f"zeta_stderr{sigma.shape} must match q{q.shape}")

    usable = (np.abs(q) > q_tol) & np.isfinite(sigma) & (sigma > 0.0) & np.isfinite(zeta)
    m = int(usable.sum())
    if m < 3:
        raise ValueError(
            f"only {m} q values carry a usable standard error; at least 3 are needed to "
            "have a residual degree of freedom after fitting H"
        )
    qu, zu, su = q[usable], zeta[usable], sigma[usable]
    w = 1.0 / su**2
    h_hat = float(np.sum(w * qu * zu) / np.sum(w * qu**2))
    residual = zu - h_hat * qu
    chi2 = float(np.sum(w * residual**2))
    dof = m - 1
    p = float(stats.chi2.sf(chi2, dof))

    interpretation = (
        f"X^2 = {chi2:.3g} on {dof} dof, p = {p:.3g}; fitted H = {h_hat:.4f}, "
        f"max |zeta - qH| = {float(np.max(np.abs(residual))):.4g} "
        f"({float(np.max(np.abs(residual) / su)):.1f} standard errors). "
    ) + (
        "H0 (monofractal) is rejected, but the chi-square treats the zeta(q) as "
        "independent when they are strongly positively correlated -- the same sample and "
        "the same scales produced all of them -- so this p-value is an upper bound on the "
        "evidence, not a measurement of it. Confirm with the surrogate test."
        if p <= _LEVEL
        else "H0 (monofractal) is not rejected. Since the test is anti-conservative, a "
        "failure to reject here is strong: a correlation-aware test would be even less "
        "inclined to reject."
    )

    prov = _derive_or_new(
        exponents.provenance,
        "inference.monofractal_test",
        params={"q_tol": float(q_tol), "n_q_used": m, "estimator": exponents.estimator},
        inputs=(InputRef.from_array("q", qu), InputRef.from_array("zeta", zu)),
        seed=None,
    )
    return TestResult(
        statistic=chi2,
        p_value=p,
        null_distribution=np.empty(0, dtype=np.float64),
        n_surrogates=0,
        interpretation=interpretation,
        name="monofractal",
        null_model="chi2(independent zeta errors)",
        dof=float(dof),
        diagnostics={
            "H": h_hat,
            "max_abs_residual": float(np.max(np.abs(residual))),
            "max_studentised_residual": float(np.max(np.abs(residual) / su)),
            "n_q_used": m,
            "caveat": "zeta(q) errors are correlated; the chi-square is anti-conservative",
        },
        provenance=prov,
    )


# --------------------------------------------------------------------------- #
# Moment existence
# --------------------------------------------------------------------------- #


def _hill_plot(z_desc: FloatArray, k_max: int) -> FloatArray:
    r"""Hill estimates :math:`\hat\alpha(k)` for :math:`k=1..k_{\max}`, vectorised.

    With the descending order statistics :math:`Z_{(1)}\ge\dots\ge Z_{(n)}`,

    .. math::
        \hat\gamma(k) = \frac1k\sum_{i=1}^{k}\ln Z_{(i)} - \ln Z_{(k+1)},
        \qquad \hat\alpha(k) = 1/\hat\gamma(k),

    the maximum-likelihood estimate of the tail index in the Pareto model
    :math:`\Pr(Z>z)=Cz^{-\alpha}` restricted to the top :math:`k` order
    statistics (Hill 1975).
    """
    log_z = np.log(z_desc)
    k = np.arange(1, k_max + 1, dtype=np.float64)
    gamma = np.cumsum(log_z[:k_max]) / k - log_z[1 : k_max + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(gamma > 0.0, 1.0 / gamma, np.inf)
    return np.asarray(alpha, dtype=np.float64)


def _rolling_std(a: FloatArray, window: int) -> FloatArray:
    """Standard deviation over a centred window; edges are NaN. Vectorised.

    Non-finite entries are *local*: a window containing one is NaN, every other
    window is unaffected. A cumulative-sum formulation would instead propagate a
    single :math:`\\pm\\infty` --- which the Hill plot produces whenever the top
    order statistics are tied --- to every subsequent window, emit
    ``invalid value encountered in subtract`` and leave the plateau rule with
    nothing to minimise.
    """
    n = a.size
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 2 or n < window:
        return out
    view = np.lib.stride_tricks.sliding_window_view(a, window)
    ok = np.all(np.isfinite(view), axis=1)
    std = np.full(view.shape[0], np.nan, dtype=np.float64)
    if ok.any():
        std[ok] = np.std(view[ok], axis=1)
    out[window // 2 : window // 2 + std.size] = std
    return out


def _hill_tail_index(
    z: FloatArray, *, k_min: int, k_max_fraction: float, confidence: float
) -> tuple[float, tuple[float, float], int, NDArray[np.int64], FloatArray]:
    """Hill plot plus a plateau-selected :math:`k`; see :func:`moment_existence`."""
    z_desc = np.sort(z)[::-1]
    n = z_desc.size
    k_max = min(int(k_max_fraction * n), n - 2)
    if k_max <= k_min:
        raise ValueError(
            f"too few usable increments for a Hill estimate: n={n} allows k_max={k_max} "
            f"upper order statistics but k_min={k_min} was requested"
        )
    alpha = _hill_plot(np.ascontiguousarray(z_desc), k_max)
    k_grid = np.arange(1, k_max + 1, dtype=np.int64)

    # Plateau rule: the Hill plot is unstable for small k (variance ~ alpha^2/k)
    # and biased for large k (order statistics that are not in the tail at all),
    # so take the flattest stretch in between. This is a heuristic; Drees, de Haan
    # and Resnick (2000) show no automatic rule is reliable -- hence hill is returned.
    window = max(5, k_max // 10)
    spread = _rolling_std(alpha, window)
    spread[: max(k_min - 1, 0)] = np.nan
    # Never centre the plateau on a k where the Hill estimate itself is infinite
    # (gamma <= 0, i.e. the top k order statistics are tied): the estimate there
    # is not a tail index and selecting it would report q_crit = inf.
    spread[~np.isfinite(alpha)] = np.nan
    if np.all(~np.isfinite(spread)):
        k_sel = max(k_min, k_max // 2)
    else:
        k_sel = int(k_grid[int(np.nanargmin(spread))])

    alpha_hat = float(alpha[k_sel - 1])
    if not math.isfinite(alpha_hat):
        # Every admissible k is tied: |Z| has bounded support above, so no finite
        # critical moment order exists and the honest answer is "no tail found".
        # Report it as such rather than as a NaN-contaminated interval.
        return float("inf"), (float("inf"), float("inf")), k_sel, k_grid, alpha
    z_crit = float(stats.norm.ppf(0.5 + confidence / 2.0))
    half = z_crit * alpha_hat / math.sqrt(k_sel)
    return alpha_hat, (alpha_hat - half, alpha_hat + half), k_sel, k_grid, alpha


def moment_existence(
    x: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any],
    basis: SurrogateBasis = "increments",
    k_min: int = 10,
    k_max_fraction: float = 0.1,
    confidence: float = 0.95,
    min_length: int = 64,
) -> MomentDiagnostic:
    r"""Which requested moments exist, from the tail index of the increments.

    Fitting a structure function at :math:`q` requires
    :math:`\mathbb E|\Delta X|^q<\infty`. If the increments have a regularly
    varying tail, :math:`\Pr(|\Delta X|>z)\sim Cz^{-\alpha}`, then

    .. math::
        \mathbb E|\Delta X|^{q} < \infty \iff q < \alpha =: q_{\mathrm{crit}} ,

    and for :math:`q\ge q_{\mathrm{crit}}` the sample moment
    :math:`\frac1N\sum|\Delta X|^q` is dominated by the single largest
    increment. It is then not an estimate of anything: it inherits that
    increment's scaling, so :math:`\log S(q,\delta)` is still very nearly linear
    in :math:`\log\delta` --- typically with :math:`R^2>0.99` --- and the fitted
    "exponent" saturates at the value dictated by the sample maximum
    (:math:`\hat\zeta(q)\to 1` for i.i.d. increments). The log--log plot gives no
    warning whatsoever. This is the mechanism behind the standard heavy-tail
    false positive (Mandelbrot 1997; Lux 2004), and v1 of this library had no
    guard against it at all.

    :math:`\alpha` is estimated by the Hill (1975) estimator over the upper
    order statistics, reported as a full Hill plot plus a plateau-selected
    value. The asymptotic law :math:`\sqrt k(\hat\alpha-\alpha)\to
    \mathcal N(0,\alpha^2)` gives the interval
    :math:`\hat\alpha(1\pm z_{1-a/2}/\sqrt k)`.

    Negative moments are governed by the *other* end of the distribution:
    :math:`\mathbb E|\Delta X|^{q}=\mathbb E\,|\Delta X|^{-|q|}` is finite iff
    :math:`|q|` is below the tail index of :math:`1/|\Delta X|`, i.e. iff the
    increments do not concentrate too strongly at zero. That index is estimated
    by the same machinery and reported as ``q_crit_negative``. Gaussian
    increments have density bounded away from zero at the origin, so their
    negative moments diverge for :math:`q\le-1` --- which is why negative-:math:`q`
    structure functions on a differenced series are usually meaningless.

    Parameters
    ----------
    q
        The moment orders the caller intends to fit.
    basis
        ``"increments"`` differences ``x`` first (the default: ``x`` is a path);
        ``"values"`` treats ``x`` as the increment series already.
    k_min, k_max_fraction
        The Hill plot is scanned over :math:`k\in[k_{\min}, \lfloor
        \text{fraction}\cdot n\rfloor]`. The default upper bound of :math:`10\%`
        of the sample is the usual compromise between variance and the bias from
        including non-tail observations.

    Returns
    -------
    MomentDiagnostic
        With ``exists`` a boolean per requested ``q`` and the Hill plot attached.

    Notes
    -----
    Two biases to keep in mind, both in the *permissive* direction for the
    heavy-tailed case and the *restrictive* direction for the light-tailed one:

    * The Hill estimator assumes an exactly Pareto tail. For an
      :math:`\alpha`-stable law the slowly varying factor is not constant at
      finite :math:`k`, and :math:`\hat\alpha` is biased **upward** by
      :math:`O(0.1)` at realistic sample sizes --- so a moment just below
      ``q_crit`` may still fail to exist.
    * A distribution with an exponential or Gaussian tail is not regularly
      varying at all: its Hill plot has no plateau and :math:`\hat\alpha` drifts
      upward with :math:`n`. ``q_crit`` is then a *lower bound* certificate ---
      "moments at least up to here are safe" --- and not an index. Every moment
      of a Gaussian exists.

    **Look at the Hill plot.** No automatic selection rule for :math:`k` is
    reliable (Drees, de Haan & Resnick 2000).

    Raises
    ------
    ValueError
        If fewer than ``k_min + 2`` non-zero increments survive.
    """
    if not 0.0 < confidence < 1.0:
        # norm.ppf(0.5 + confidence/2) is NaN outside (0,1) and negative below 0,
        # so the interval on q_crit would come back NaN or inverted, not raise.
        raise ValueError(
            f"confidence must be a probability in (0, 1), got {confidence!r}; "
            "the 95% level is 0.95, not 95"
        )
    xa = validate_series(x, name="x", min_length=min_length)
    qa = np.asarray(q, dtype=np.float64).ravel()
    if qa.size == 0:
        raise ValueError("q is empty")
    if not np.all(np.isfinite(qa)):
        raise ValueError("q contains non-finite values")

    u, _ = _decompose(xa, basis)
    absu = np.abs(u)
    n_zero = int(np.count_nonzero(absu == 0.0))
    absu = absu[absu > 0.0]
    if absu.size < k_min + 2:
        raise ValueError(
            f"only {absu.size} non-zero increments; a Hill estimate over k_min={k_min} "
            "upper order statistics is impossible"
        )

    q_crit, ci, k_sel, k_grid, hill = _hill_tail_index(
        absu, k_min=k_min, k_max_fraction=k_max_fraction, confidence=confidence
    )
    # The negative-moment bound is the tail index of the reciprocal: subnormal
    # increments would overflow it, and those observations carry no information
    # about the small-value exponent that the surviving ones do not.
    with np.errstate(over="ignore"):
        reciprocal = 1.0 / absu
    q_crit_neg, _, _, _, _ = _hill_tail_index(
        reciprocal[np.isfinite(reciprocal)],
        k_min=k_min,
        k_max_fraction=k_max_fraction,
        confidence=confidence,
    )

    exists = np.where(qa >= 0.0, qa < q_crit, -qa < q_crit_neg)
    exists = np.where(qa == 0.0, True, exists)

    prov = new_provenance(
        "inference.moment_existence",
        estimator_version=MODULE_VERSION,
        params={
            "basis": str(basis),
            "k_min": int(k_min),
            "k_max_fraction": float(k_max_fraction),
            "confidence": float(confidence),
        },
        inputs=(InputRef.from_array("x", xa), InputRef.from_array("q", qa)),
        seed=None,
        notes=("Hill tail index of |increments|; q >= q_crit has no finite moment",),
    )
    return MomentDiagnostic(
        q=qa,
        exists=np.asarray(exists, dtype=bool),
        q_crit=q_crit,
        q_crit_ci=ci,
        q_crit_negative=q_crit_neg,
        k_selected=k_sel,
        k_grid=k_grid,
        hill=hill,
        n_used=int(absu.size),
        n_zero_increments=n_zero,
        provenance=prov,
    )


# --------------------------------------------------------------------------- #
# Scaling-range linearity
# --------------------------------------------------------------------------- #


def scaling_range_test(
    structure: StructureFunctions,
    *,
    r2_threshold: float = 0.995,
    min_points: int = 5,
    level: float = _LEVEL,
    weighted: bool = True,
) -> dict[str, Any]:
    r"""Is the log--log relation actually linear, or merely correlated?

    Three diagnostics per :math:`q`, because :math:`R^2` alone answers the wrong
    question:

    **Per-q :math:`R^2`.** Reported, but treated as descriptive only. A smoothly
    curved :math:`\log S` against :math:`\log\delta` routinely reaches
    :math:`R^2>0.99` over two decades: :math:`R^2` measures how much of the
    variance a line explains, and a monotone curve is mostly a line. This is why
    v1's Pearson correlation test on the log--log points (rejecting
    :math:`\rho=0` for every :math:`q`) filtered nothing.

    **RESET-style curvature test.** Augment the regression with a quadratic term
    (Ramsey 1969),

    .. math::
        \log S(q,\delta) = \beta_0 + \beta_1\log\delta
                         + \beta_2 (\log\delta)^2 + \varepsilon,

    and test :math:`H_0:\beta_2=0` with the usual :math:`t_{n-3}` statistic.
    This asks the right question --- *is a straight line the correct
    specification* --- and has power against exactly the crossover that
    :math:`R^2` hides. Because it is run once per :math:`q`, the p-values are
    additionally reported with a Benjamini--Hochberg correction at ``level``.

    .. warning::

        **The curvature p-value is badly anti-conservative on real structure
        functions, and by a large factor.** The :math:`t_{n-3}` law assumes the
        regression errors are independent across scales. They are not: a
        structure function at scale :math:`\delta` is an average over
        *overlapping* windows of the same sample, so :math:`\log S(q,\delta)`
        and :math:`\log S(q,2\delta)` share most of their data and their errors
        are strongly positively autocorrelated. Measured rejection rate at a
        nominal FDR of :math:`5\%` on a relation that is an exact power law:
        :math:`4\%` at error autocorrelation :math:`\rho=0`, :math:`29\%` at
        :math:`\rho=0.5`, :math:`66\%` at :math:`\rho=0.9`. On 30 fresh Brownian
        paths --- :math:`\zeta(q)=q/2` exactly, no crossover to find --- 24 were
        declared ``CURVED``. Read ``residual_autocorr``: when it is far from
        zero the p-values are not usable, and only a large :math:`|\beta_2|`
        relative to the fitted slope means anything. A calibrated version needs
        a HAC/Newey--West variance or a block bootstrap of the series, which
        lives in :mod:`mmar.inference.bootstrap`.

    **Largest linear window.** :func:`~mmar.core.regression.select_scaling_range`
    is run per :math:`q` to find the widest contiguous window reaching
    ``r2_threshold``, and the intersection over :math:`q` is reported as
    ``common_window``: the only scale range over which *all* the fitted
    exponents are supported by a straight line. A :math:`\tau(q)` assembled from
    per-:math:`q` windows that do not overlap is not a scaling function of one
    object.

    Parameters
    ----------
    structure
        Raw :math:`S(q,\delta)` (positive moments, not logarithms). Weighted by
        ``structure.counts`` when present and ``weighted`` is true, which is the
        correct inverse-variance weighting since a log-moment from :math:`n_j`
        windows has variance :math:`\propto 1/n_j`.

    Returns
    -------
    dict
        Keys: ``q``, ``r_squared``, ``slope``, ``curvature`` (:math:`\beta_2`),
        ``curvature_t``, ``curvature_p``, ``curvature_significant`` (BH at
        ``level``), ``residual_autocorr`` (lag-1 autocorrelation of the
        regression residuals across scales --- the diagnostic for the warning
        above), ``window_s_min``/``window_s_max``/``window_n_scales`` per
        :math:`q`, ``common_window``, ``n_curved``, ``min_r_squared``,
        ``failed_q`` and ``verdict``.

    Raises
    ------
    ValueError
        If fewer than four scales are present (the quadratic augmentation needs
        :math:`n\ge4` for a residual degree of freedom).
    """
    q = np.asarray(structure.q, dtype=np.float64)
    scales = np.asarray(structure.scales, dtype=np.float64)
    values = np.asarray(structure.values, dtype=np.float64)
    n = scales.size
    if n < 4:
        raise ValueError(f"need at least 4 scales for a curvature test, got {n}")
    if np.any(scales <= 0.0):
        raise ValueError("scales must be strictly positive")

    counts = structure.counts
    w = (
        np.asarray(counts, dtype=np.float64)
        if (weighted and counts is not None)
        else np.ones(n, dtype=np.float64)
    )
    if w.shape != scales.shape:
        raise ValueError(f"counts{w.shape} must match scales{scales.shape}")

    lx = np.log(scales)
    ok_rows = np.all(values > 0.0, axis=1) & np.all(np.isfinite(values), axis=1)
    failed_q = q[~ok_rows]
    if not ok_rows.any():
        raise ValueError("every q row contains a non-positive or non-finite moment")

    with np.errstate(divide="ignore"):
        ly = np.where(ok_rows[:, None], np.log(np.where(values > 0.0, values, 1.0)), np.nan)

    # Quadratic augmentation, solved for every q at once: the design matrix is
    # shared across q, so one pseudo-inverse serves all rows.
    design = np.column_stack([np.ones(n), lx, lx**2])
    sw = np.sqrt(w)
    dw = design * sw[:, None]
    yw = ly[ok_rows] * sw[None, :]
    beta, *_ = np.linalg.lstsq(dw, yw.T, rcond=None)
    resid = yw.T - dw @ beta
    dof = n - 3
    s2 = np.sum(resid**2, axis=0) / dof
    xtx_inv = np.linalg.inv(dw.T @ dw)
    se2 = float(xtx_inv[2, 2])
    with np.errstate(invalid="ignore"):
        se = np.sqrt(s2 * se2)
        t_stat = np.where(se > 0.0, beta[2] / se, np.nan)
    p_curv = 2.0 * stats.t.sf(np.abs(t_stat), dof)

    # Lag-1 autocorrelation of the residuals across scales. The t-test above
    # assumes this is zero; on overlapping-window structure functions it is not,
    # and it is the only warning the caller gets that curvature_p is inflated.
    ss = np.sum(resid**2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho1 = np.where(ss > 0.0, np.sum(resid[:-1] * resid[1:], axis=0) / ss, np.nan)

    curvature = np.full(q.size, np.nan)
    curvature_t = np.full(q.size, np.nan)
    curvature_p = np.full(q.size, np.nan)
    residual_autocorr = np.full(q.size, np.nan)
    curvature[ok_rows] = beta[2]
    curvature_t[ok_rows] = t_stat
    curvature_p[ok_rows] = p_curv
    residual_autocorr[ok_rows] = rho1

    r2 = np.full(q.size, np.nan)
    slope = np.full(q.size, np.nan)
    win_lo = np.full(q.size, np.nan)
    win_hi = np.full(q.size, np.nan)
    win_n = np.zeros(q.size, dtype=np.int64)
    method = FitMethod.WLS if (weighted and counts is not None) else FitMethod.OLS
    for i in np.nonzero(ok_rows)[0]:
        fit = loglog_fit(scales, values[i], method=method, weights=w)
        r2[i] = fit.r_squared
        slope[i] = fit.slope
        if n >= min_points:
            sel = select_scaling_range(
                scales,
                values[i],
                min_points=min_points,
                r2_threshold=r2_threshold,
                weights=w,
                method=method,
            )
            win_lo[i] = sel.scaling_range.s_min
            win_hi[i] = sel.scaling_range.s_max
            win_n[i] = sel.scaling_range.n_scales

    significant = _benjamini_hochberg(curvature_p, level)
    have_windows = np.isfinite(win_lo) & np.isfinite(win_hi)
    if have_windows.any():
        lo = float(np.max(win_lo[have_windows]))
        hi = float(np.min(win_hi[have_windows]))
        inside = int(np.count_nonzero((scales >= lo) & (scales <= hi)))
        common: tuple[float, float, int] | None = (lo, hi, inside) if hi > lo else None
    else:
        common = None

    n_curved = int(significant.sum())
    min_r2 = float(np.nanmin(r2)) if np.any(np.isfinite(r2)) else float("nan")
    rho_max = (
        float(np.nanmax(np.abs(residual_autocorr)))
        if np.any(np.isfinite(residual_autocorr))
        else float("nan")
    )
    correlated = (
        " NOTE: the residuals are autocorrelated across scales "
        f"(max |lag-1 rho| = {rho_max:.2f}), which the t_(n-3) null forbids; the "
        "curvature p-values are anti-conservative and this verdict may be spurious. "
        "Judge by the size of beta2 against the slope, not by the p-value."
        if math.isfinite(rho_max) and rho_max > 0.3
        else ""
    )
    if n_curved == 0:
        verdict = (
            f"Linear: no q shows significant curvature at FDR {level:g} "
            f"(min R^2 = {min_r2:.4f}). The power-law reading is supported."
        )
    else:
        verdict = (
            f"CURVED: {n_curved}/{int(ok_rows.sum())} q values reject linearity at FDR "
            f"{level:g} despite min R^2 = {min_r2:.4f}. R^2 near 1 does not mean linear; "
            "the fitted exponents mix two regimes. Restrict to the common window "
            + (
                f"[{common[0]:g}, {common[1]:g}] ({common[2]} scales) before quoting them."
                if common is not None
                else "-- which is empty, so no single scale range supports all q."
            )
            + correlated
        )

    return {
        "q": q,
        "r_squared": r2,
        "slope": slope,
        "curvature": curvature,
        "curvature_t": curvature_t,
        "curvature_p": curvature_p,
        "curvature_significant": significant,
        "residual_autocorr": residual_autocorr,
        "window_s_min": win_lo,
        "window_s_max": win_hi,
        "window_n_scales": win_n,
        "common_window": common,
        "n_curved": n_curved,
        "min_r_squared": min_r2,
        "n_scales": n,
        "failed_q": failed_q,
        "weighted": bool(weighted and counts is not None),
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# Concavity
# --------------------------------------------------------------------------- #


def concavity_test(
    exponents: ScalingExponents,
    *,
    n_surrogates: int = 999,
    seed: SeedLike = None,
) -> TestResult:
    r"""Is :math:`\hat\zeta` concave to within its own noise?

    The multifractal formalism *defines* :math:`f` as the Legendre--Fenchel
    transform of :math:`\zeta`, and a Legendre transform sees only the concave
    envelope of its argument: any non-concave stretch of :math:`\hat\zeta` is
    invisible to :math:`f(\alpha)` and is therefore silently discarded. The
    discarded amount is
    :func:`~mmar.core.legendre.concavity_defect`,

    .. math::
        D = \max_q\ \big(\operatorname{conc}\hat\zeta(q) - \hat\zeta(q)\big) \ \ge 0 ,

    with :math:`\operatorname{conc}` the least concave majorant. :math:`D=0`
    exactly when :math:`\hat\zeta` is concave; a large :math:`D` means the
    spectrum plotted from this sample is an artefact of the transform, not a
    summary of the estimate.

    The test asks whether :math:`D` is large *relative to the noise in*
    :math:`\hat\zeta`. Null: :math:`\zeta` is exactly concave and equals its own
    envelope, observed with independent Gaussian errors of the reported sizes
    :math:`\hat\sigma_i`,

    .. math::
        \tilde\zeta^{(b)}_i = \operatorname{conc}\hat\zeta(q_i)
                              + \hat\sigma_i\varepsilon^{(b)}_i ,
        \qquad \varepsilon^{(b)}_i \sim \mathcal N(0,1) \ \text{i.i.d.},

    and :math:`p` is the fraction of :math:`D(\tilde\zeta^{(b)})` at least as
    large as :math:`D(\hat\zeta)`.

    Direction of the error, stated explicitly: the real
    :math:`\hat\zeta(q_i)` are *positively correlated* across :math:`q`, so their
    differences fluctuate less than the independent null assumes. The simulated
    defects are therefore too large and this test is **conservative** --- it
    under-rejects. A rejection is consequently meaningful; a non-rejection is
    weak evidence.

    Returns
    -------
    TestResult
        ``statistic`` is :math:`D`; the diagnostics carry the ratio
        :math:`D/\operatorname{median}(\hat\sigma)` and a verdict, and the
        ``interpretation`` states whether the multifractal formalism applies to
        this sample at all.
    """
    q = np.asarray(exponents.q, dtype=np.float64)
    zeta = np.asarray(exponents.zeta, dtype=np.float64)
    sigma = _zeta_stderr(exponents)
    if sigma.shape != q.shape:
        raise ValueError(f"zeta_stderr{sigma.shape} must match q{q.shape}")
    finite = np.isfinite(sigma) & (sigma > 0.0)
    if not finite.any():
        raise ValueError("no positive standard errors on zeta; the noise scale is unknown")
    scale = np.where(finite, sigma, float(np.median(sigma[finite])))

    defect = concavity_defect(q, zeta)
    envelope = concave_envelope(q, zeta)

    rng = make_rng(seed)
    noise = rng.standard_normal((n_surrogates, q.size)) * scale[None, :]
    replicates = envelope[None, :] + noise
    null = np.asarray(
        [concavity_defect(q, np.ascontiguousarray(row)) for row in replicates],
        dtype=np.float64,
    )
    p = empirical_p_value(defect, null)
    ratio = defect / float(np.median(scale))

    if p <= _LEVEL:
        verdict = "not-concave"
        text = (
            "The estimated zeta(q) is further from concave than independent noise of the "
            "reported size can explain, and this test is conservative, so the departure is "
            "real. The multifractal formalism does not apply to this sample: the Legendre "
            "transform will discard the non-concave part and return a plausible-looking "
            "f(alpha) that no longer corresponds to the estimate. Suspect a crossover in "
            "the scaling range, non-existent moments at the ends of the q grid, or a "
            "sample too short for the widest scales."
        )
    elif ratio > 1.0:
        verdict = "marginal"
        text = (
            "The concavity defect is not significant but exceeds the typical standard "
            "error of zeta. Treat the spectrum as provisional and widen the sample or "
            "narrow the q grid before quoting a width."
        )
    else:
        verdict = "concave"
        text = (
            "The estimate is concave to within its own noise; the Legendre transform "
            "discards nothing material and f(alpha) faithfully represents zeta(q)."
        )

    prov = _derive_or_new(
        exponents.provenance,
        "inference.concavity_test",
        params={"n_surrogates": int(n_surrogates), "estimator": exponents.estimator},
        inputs=(InputRef.from_array("q", q), InputRef.from_array("zeta", zeta)),
        seed=seed_info(seed),
    )
    return TestResult(
        statistic=defect,
        p_value=p,
        null_distribution=null,
        n_surrogates=int(n_surrogates),
        interpretation=f"defect = {defect:.4g} ({ratio:.2f} x median SE), p = {p:.3f}. " + text,
        name="concavity",
        null_model="concave envelope + independent Gaussian noise",
        diagnostics={
            "verdict": verdict,
            "defect_over_median_stderr": ratio,
            "median_stderr": float(np.median(scale)),
        },
        provenance=prov,
    )


# --------------------------------------------------------------------------- #
# Provenance plumbing
# --------------------------------------------------------------------------- #


def _derive_or_new(
    parent: Provenance | None,
    estimator: str,
    *,
    params: Mapping[str, Any],
    inputs: tuple[InputRef, ...],
    seed: SeedInfo | None,
) -> Provenance:
    """Chain onto the upstream record when there is one, else start a new root."""
    if parent is not None:
        return parent.derive(
            estimator,
            estimator_version=MODULE_VERSION,
            params=params,
            extra_inputs=inputs,
            seed=seed,
        )
    return new_provenance(
        estimator, estimator_version=MODULE_VERSION, params=params, inputs=inputs, seed=seed
    )
