r"""Chhabra--Jensen direct determination of :math:`f(\alpha)`.

The Legendre route to a singularity spectrum estimates :math:`\tau(q)`, then
differentiates it to get :math:`\alpha=\tau'(q)`. Numerical differentiation of a
noisy fitted curve is an ill-posed operation, and it is the reason estimated
:math:`f(\alpha)` curves develop loops, backward folds and spurious tails.

Chhabra and Jensen (1989) removed the differentiation entirely. Given a positive
measure :math:`\mu` partitioned at box size :math:`\delta` into cell masses
:math:`\mu_i(\delta)`, define the one-parameter family of *tilted* (normalised)
measures

.. math::
    \mu_i(q,\delta) \;=\; \frac{\mu_i(\delta)^q}{\sum_j \mu_j(\delta)^q},
    \qquad \sum_i \mu_i(q,\delta) = 1 ,

a softmax over :math:`q\ln\mu_i`. The Hausdorff dimension of the set carrying
exponent :math:`\alpha(q)`, and the exponent itself, are then *averages against
that tilted measure*:

.. math::
    \boxed{\;
    \alpha(q) = \lim_{\delta\to0}
       \frac{\sum_i \mu_i(q,\delta)\,\ln \mu_i(\delta)}{\ln\delta},
    \qquad
    f(q) = \lim_{\delta\to0}
       \frac{\sum_i \mu_i(q,\delta)\,\ln \mu_i(q,\delta)}{\ln\delta} \; }

*Why this is the spectrum.* Fix :math:`q`. The tilted measure concentrates, as
:math:`\delta\to0`, on the subset :math:`S_q` of cells whose coarse exponent is
:math:`\ln\mu_i/\ln\delta \approx \alpha(q)`; the first line is precisely the
:math:`\mu(q,\cdot)`-average of that coarse exponent, hence :math:`\alpha(q)`.
The second line is :math:`-H(q,\delta)/\ln(1/\delta)` with :math:`H` the Shannon
entropy of :math:`\mu(q,\cdot)`, and by the Shannon--McMillan--Breiman /
Billingsley argument the entropy of a measure concentrated on :math:`S_q` grows
like :math:`\dim_H(S_q)\ln(1/\delta)`, so the second line is
:math:`f(\alpha(q))=\dim_H S_q`. No Legendre transform, no derivative of a fit.

*Exact internal identity.* Writing :math:`\ln Z(q,\delta)=\ln\sum_i\mu_i^q`, the
numerators satisfy, at every single scale and with no asymptotics,

.. math::
    F(q,\delta) \;\equiv\; \sum_i \mu_i(q)\ln\mu_i(q)
    \;=\; q\,A(q,\delta) - \ln Z(q,\delta),
    \qquad A(q,\delta)\equiv\sum_i\mu_i(q)\ln\mu_i .

Since the least-squares slope is a *linear* functional of the response, fitting
all three against :math:`\ln\delta` with one design matrix and one weight vector
gives :math:`f = q\alpha - \tau` **exactly** (to floating point), where
:math:`\tau` is the directly fitted slope of :math:`\ln Z`. The library exploits
this: it reports :math:`\tau = q\alpha - f` as mandated by the convention, and
records the residual against the direct :math:`\ln Z` fit as a diagnostic. The
identity fails by construction for the robust fitters (``THEILSEN``, ``HUBER``),
which are not linear in the response; the diagnostic then quantifies the gap.

Convention
----------
This estimator is **native to the measure view**: it consumes cell masses and
produces :math:`\tau(q)`, defined by :math:`Z(q,\delta)\sim\delta^{\tau(q)}`. The
canonical carrier :class:`~mmar.core.types.ScalingExponents` stores the signal
exponent, so the conversion applied here is

.. math::
    \zeta(q) \;=\; \tau(q) + d \;=\; q\alpha(q) - f(q) + d ,

with :math:`d` = ``d_support`` (1 for a measure on an interval). Anchors, both
tested: :math:`\tau(0) = -f(0) = -d` because the tilted measure at :math:`q=0`
is uniform over the :math:`\delta^{-d}` boxes; and :math:`\tau(1)=0` for a
conservative measure, because :math:`\mu_i(1)=\mu_i` and
:math:`A(1,\delta)=F(1,\delta)`.

Estimation, not evaluation
--------------------------
Both limits above are estimated as the **slope of the numerator against**
:math:`\ln\delta` over a range of scales, never as the ratio at a single
:math:`\delta`. The single-scale ratio is what most published implementations
compute; it has no error bar, it is biased by the intercept (the numerators are
affine in :math:`\ln\delta`, not proportional to it), and it cannot detect the
failure of scaling. Here each of :math:`\alpha`, :math:`f` and :math:`\tau`
comes from its own weighted regression and carries a standard error and a
confidence interval.

Known biases
------------
1. **Empty boxes.** :math:`\mu_i=0` makes :math:`\ln\mu_i` undefined and
   :math:`\mu_i^q` infinite for :math:`q<0`. Empty boxes are *excluded* and the
   count is recorded per scale in ``provenance.diagnostics["n_empty"]``.
   Exclusion is not neutral: it removes exactly the cells the negative-:math:`q`
   tilt would weight most, so :math:`\alpha(q)` for :math:`q\ll0` is biased
   **downward** and the right branch of :math:`f(\alpha)` is truncated. The
   effect grows as :math:`\delta\to0` (more boxes are empty at fine scales),
   which also tilts the regression. If ``n_empty`` is not identically zero, the
   negative-:math:`q` branch should be read as a lower bound on
   :math:`\alpha_{\max}`.
2. **Finite :math:`|q|`.** :math:`\alpha(q)\to\alpha_{\min/\max}` only as
   :math:`q\to\pm\infty`. On :math:`q\in[-5,5]` roughly the central 75--80% of a
   binomial support is reached; the estimated spectrum is a *sub-interval* of
   the true one, and its endpoints are not :math:`f=0`.
3. **Tail truncation from coarse-graining.** When the box factor does not divide
   the number of cells, the final partial box is dropped. The dropped fraction
   :math:`(n \bmod s)/n` grows with :math:`s`, so it is not a constant offset
   and can tilt the fit for a strongly inhomogeneous measure. Counts are in
   ``provenance.diagnostics["n_truncated"]``. Prefer a measure whose length is a
   power of the branching number, which the default scale ladder detects and
   exploits.
4. **Overlapping information across scales.** Successive dyadic coarse-grainings
   of the same measure are nested, so the residuals of the log--log fit are
   strongly correlated. The reported standard errors are conditional on
   independence across scales and are therefore *optimistic*; for honest
   intervals resample the measure, not the scales.
5. **The fine end of the default ladder manufactures width.** The default scale
   ladder starts at the smallest admissible box factor, :math:`s=2`, matching
   :mod:`~mmar.estimators.box_counting`. A box of two cells is not an average,
   it is a pair, so the coarse exponent field there reflects the *marginal*
   heterogeneity of the cells rather than any multiplicative structure. On an
   exactly self-similar cascade this costs nothing (the numerators are affine in
   :math:`\ln\delta` at every scale, so the recovered width is identical for
   every cutoff). On anything else it dominates. Measured on i.i.d.
   :math:`|N(0,1)|` masses, :math:`n=2^{16}`, :math:`q\in[-5,5]` --- a measure
   with *no* multifractality whatsoever:

   ======================  =======================
   fine-scale cutoff       recovered :math:`\Delta\alpha`
   ======================  =======================
   :math:`s_{\min}=2`      1.818  (the default)
   :math:`s_{\min}=16`     0.132
   :math:`s_{\min}=64`     0.031
   :math:`s_{\min}=1024`   0.003
   ======================  =======================

   with :math:`R^2>0.997` and :math:`\mathrm{SE}(\alpha)<0.02` throughout: the
   artefact is smooth, confident and entirely spurious. It does not shrink with
   :math:`n` (the same 0.13 at :math:`s_{\min}=16` for
   :math:`n=2^{12},2^{14},2^{16}`), only with the cutoff. **Pass an explicit
   ``scales`` ladder with a fine cutoff of at least a few tens of cells for any
   measure that is not a synthetic cascade, and report the cutoff.** A
   :math:`\Delta\alpha` quoted without its :math:`s_{\min}` is not a measurement.
   The corollary for surrogate testing: on a shuffled binomial cascade --- the
   standard null, which destroys the hierarchy while preserving the mass
   histogram --- the default ladder returns :math:`\Delta\alpha=0.926` against
   the true cascade's 0.449, so the naive shuffle test comes out *backwards*.
   The ordering is restored at :math:`s_{\min}\ge16` (0.165 versus 0.449).

References
----------
Chhabra, A. and Jensen, R. V. (1989). Direct determination of the
:math:`f(\alpha)` singularity spectrum. *Phys. Rev. Lett.* **62**, 1327--1330.

Chhabra, A., Meneveau, C., Jensen, R. V. and Sreenivasan, K. R. (1989). Direct
determination of the :math:`f(\alpha)` singularity spectrum and its application
to fully developed turbulence. *Phys. Rev. A* **40**, 5284--5294.

Halsey, T. C., Jensen, M. H., Kadanoff, L. P., Procaccia, I. and Shraiman, B. I.
(1986). Fractal measures and their singularities. *Phys. Rev. A* **33**, 1141.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
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
    HolderSpectrum,
    LogLogFit,
    ScalingExponents,
    ScalingRange,
    StructureFunctions,
)
from mmar.core.validate import (
    NanPolicy,
    check_scales,
    validate_measure,
    validate_q_grid,
    validate_series,
)

__all__ = [
    "ChhabraJensenFits",
    "ChhabraJensenSurface",
    "chhabra_jensen",
    "chhabra_jensen_fits",
    "chhabra_jensen_from_surface",
    "chhabra_jensen_series",
    "chhabra_jensen_structure",
    "chhabra_jensen_surface",
    "coarse_grain",
]

ESTIMATOR = "chhabra_jensen"
ESTIMATOR_VERSION = "1"

_MIN_BOXES = 4
"""Fewest boxes a scale may leave. Below this the box average is three numbers."""

_MIN_SCALES = 3
"""Fewest scales admissible: a slope with a standard error needs three points."""

_CHUNK = 1 << 22
"""Element budget for the (q, box) tilt matrix, ~32 MB of float64 per block."""

_NOTE = (
    "Chhabra-Jensen direct method: alpha and f are fitted slopes of the tilted "
    "averages against ln(delta); measure view, tau(q) = q*alpha - f, zeta = tau + d"
)


# --------------------------------------------------------------------------- #
# Coarse-graining
# --------------------------------------------------------------------------- #


def coarse_grain(mu: FloatArray, factor: int, *, truncate: bool = False) -> FloatArray:
    r"""Aggregate a discrete measure into boxes of ``factor`` consecutive cells.

    The box partition of a measure is a *sum*, not an average:
    :math:`\mu_i(s\delta_0)=\sum_{j\in \text{box } i}\mu_j(\delta_0)`. Implemented
    as one reshape and one row sum, so the whole ladder of scales costs
    :math:`O(n)` per scale with no Python loop over cells.

    Parameters
    ----------
    mu
        Non-negative cell masses at the finest available resolution.
    factor
        Number of fine cells per box, :math:`\ge 1`.
    truncate
        When ``factor`` does not divide ``len(mu)``, ``False`` (the default for
        this primitive) raises, and ``True`` discards the final
        ``len(mu) % factor`` cells. Truncation loses mass from the tail of the
        record and the loss grows with ``factor``; the estimators that call this
        record the dropped count per scale in their provenance diagnostics.

    Returns
    -------
    FloatArray
        Shape ``(len(mu) // factor,)``.
    """
    a = np.ascontiguousarray(mu, dtype=np.float64).ravel()
    f = int(factor)
    if f < 1:
        raise ValueError(f"the coarse-graining factor must be >= 1, got {factor!r}")
    n_boxes = a.size // f
    if n_boxes < 1:
        raise ValueError(f"factor={f} exceeds the measure length {a.size}")
    remainder = a.size - n_boxes * f
    if remainder and not truncate:
        raise ValueError(
            f"factor={f} does not divide len(mu)={a.size} ({remainder} cells left over); "
            "pass truncate=True to drop the trailing partial box, and be aware that the "
            "dropped fraction grows with the factor"
        )
    return np.asarray(a[: n_boxes * f].reshape(n_boxes, f).sum(axis=1), dtype=np.float64)


def _divisor_ladder(n: int) -> NDArray[np.int64]:
    """Divisors ``s`` of ``n`` with ``2 <= s`` and ``n // s >= _MIN_BOXES``."""
    small: list[int] = []
    large: list[int] = []
    root = math.isqrt(n)
    for d in range(1, root + 1):
        if n % d:
            continue
        small.append(d)
        if d != n // d:
            large.append(n // d)
    divisors = np.array(sorted(small + large), dtype=np.int64)
    return np.asarray(divisors[(divisors >= 2) & ((n // divisors) >= _MIN_BOXES)])


def _geometric_subsample(values: NDArray[np.int64], n_target: int) -> NDArray[np.int64]:
    """Thin a sorted integer ladder to ~``n_target`` geometrically spaced entries."""
    if values.size <= n_target:
        return values
    picks = np.geomspace(float(values[0]), float(values[-1]), num=n_target)
    idx = np.unique(np.abs(values[None, :] - picks[:, None]).argmin(axis=1))
    return np.asarray(values[idx])


def _default_scales(n: int) -> NDArray[np.int64]:
    r"""Geometric ladder of box factors, preferring exact divisors of ``n``.

    A measure of length :math:`b^{k}` admits an *exact* box partition at every
    :math:`b`-adic scale, and for such a measure the coarse-grained cells are the
    genuine level-:math:`k'` cells of the underlying construction. The ladder of
    divisors is therefore used whenever it is rich enough; otherwise a plain
    geometric ladder is returned and the caller must accept tail truncation.
    """
    div = _divisor_ladder(n)
    if div.size >= 5:
        return _geometric_subsample(div, 20)
    hi = max(2, n // _MIN_BOXES)
    raw = np.geomspace(2.0, float(hi), num=20)
    return np.asarray(np.unique(np.round(raw).astype(np.int64)))


def _resolve_scales(
    scales: Sequence[int] | NDArray[Any] | None, n: int, *, truncate: bool
) -> NDArray[np.int64]:
    raw = _default_scales(n) if scales is None else np.asarray(scales, dtype=np.int64).ravel()
    sc = check_scales(raw, n, min_windows=_MIN_BOXES, name="box factors")
    if not truncate:
        keep = (n % sc) == 0
        if int(keep.sum()) < _MIN_SCALES:
            raise ValueError(
                f"only {int(keep.sum())} of {sc.size} box factors divide the measure length "
                f"n={n}, fewer than the {_MIN_SCALES} needed for a slope. Pass truncate=True "
                "to allow the trailing partial box to be dropped, or use a measure whose "
                "length is a power of the branching number"
            )
        sc = np.asarray(sc[keep])
    if sc.size < _MIN_SCALES:
        raise ValueError(f"need at least {_MIN_SCALES} box factors, got {sc.size}")
    return np.ascontiguousarray(sc)


# --------------------------------------------------------------------------- #
# The raw scaling surface
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ChhabraJensenSurface:
    r"""The three Chhabra--Jensen numerators on the :math:`(q,\delta)` grid.

    Keeping this separate from the fit is what lets the exponents be refitted
    over a different scaling range, with different weights or a different
    estimator, without recomputing a single partition sum.

    Attributes
    ----------
    alpha_numerator
        :math:`A(q,\delta)=\sum_i\mu_i(q,\delta)\ln\mu_i(\delta)`, shape
        ``(n_q, n_scales)``. Its slope against :math:`\ln\delta` is
        :math:`\alpha(q)`.
    f_numerator
        :math:`F(q,\delta)=\sum_i\mu_i(q,\delta)\ln\mu_i(q,\delta)`. Its slope is
        :math:`f(q)`. Equals :math:`qA-\ln Z` identically.
    log_partition
        :math:`\ln Z(q,\delta)=\ln\sum_i\mu_i(\delta)^q`. Its slope is
        :math:`\tau(q)`. Stored logged: :math:`Z` itself overflows float64 for
        moderately negative :math:`q` on a deep cascade.
    counts
        Non-empty boxes per scale; the inverse-variance weight of the log-moment
        at that scale.
    """

    q: FloatArray
    scales: NDArray[np.int64]
    box_size: FloatArray
    alpha_numerator: FloatArray
    f_numerator: FloatArray
    log_partition: FloatArray
    counts: NDArray[np.int64]
    n_boxes: NDArray[np.int64]
    n_empty: NDArray[np.int64]
    n_truncated: NDArray[np.int64]
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        shape = (self.q.size, self.scales.size)
        for name in ("alpha_numerator", "f_numerator", "log_partition"):
            got = getattr(self, name).shape
            if got != shape:
                raise ValueError(f"{name}{got} must have shape {shape}")

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Non-identifying counts to be attached to a provenance record."""
        return {
            "n_boxes": [int(v) for v in self.n_boxes],
            "n_empty": [int(v) for v in self.n_empty],
            "n_truncated": [int(v) for v in self.n_truncated],
            "n_empty_total": int(self.n_empty.sum()),
            "n_truncated_total": int(self.n_truncated.sum()),
        }

    def to_structure_functions(self) -> StructureFunctions:
        r"""The box partition function :math:`Z(q,\delta)` as the canonical surface.

        The slope of :math:`\ln Z` against :math:`\ln\delta` is :math:`\tau(q)`,
        the **measure**-view exponent --- add ``d_support`` for :math:`\zeta`.
        Raises rather than storing infinities when :math:`Z` overflows float64,
        which happens for strongly negative :math:`q` on a deep cascade; the
        internal fits are unaffected because they use ``log_partition``.
        """
        with np.errstate(over="ignore"):
            values = np.exp(self.log_partition)
        if not np.all(np.isfinite(values)):
            worst = int(np.argmax(self.log_partition.max(axis=1)))
            raise ValueError(
                f"the partition function Z(q, delta) overflows float64 at q={self.q[worst]!r} "
                f"(max ln Z = {float(self.log_partition.max()):.1f} > 709). The fitted "
                "exponents are still available from the surface, which is stored in the log "
                "domain; narrow the q grid to materialise Z itself"
            )
        return StructureFunctions(
            q=self.q,
            scales=self.box_size,
            values=np.asarray(values, dtype=np.float64),
            counts=self.counts,
            provenance=self.provenance,
        )


def _tilted_moments(masses: FloatArray, q: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Return :math:`(A, F, \ln Z)` at one scale, for the whole ``q`` grid.

    Everything is done in the log domain. The tilt weights are
    :math:`\ln w_i = q\ln\mu_i - \operatorname{logsumexp}(q\ln\mu)`, which are
    :math:`\le 0` by construction, so ``exp`` cannot overflow; forming
    :math:`\mu_i^q` directly would overflow at :math:`q=-5` on any cascade deeper
    than about fifteen levels. ``F`` is accumulated as
    :math:`\sum_i w_i \ln w_i` rather than through the algebraic identity
    :math:`qA-\ln Z`, whose two terms are individually large and nearly equal for
    :math:`|q|` large.
    """
    log_mu = np.log(masses)
    n_q = q.size
    a = np.empty(n_q, dtype=np.float64)
    f = np.empty(n_q, dtype=np.float64)
    lz = np.empty(n_q, dtype=np.float64)
    step = max(1, _CHUNK // max(masses.size, 1))
    for lo in range(0, n_q, step):
        hi = min(lo + step, n_q)
        x = q[lo:hi, None] * log_mu[None, :]
        lse = np.asarray(logsumexp(x, axis=1), dtype=np.float64)
        log_w = x - lse[:, None]
        w = np.exp(log_w)
        a[lo:hi] = w @ log_mu
        f[lo:hi] = np.einsum("ij,ij->i", w, log_w)
        lz[lo:hi] = lse
    return a, f, lz


def chhabra_jensen_surface(
    mu: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    truncate: bool = True,
    provenance: Provenance | None = None,
) -> ChhabraJensenSurface:
    r"""Compute the Chhabra--Jensen numerators without fitting anything.

    Parameters
    ----------
    mu
        Non-negative cell masses. Normalised to unit total mass; because the
        total is the same at every box size, that division shifts the intercept
        of every log--log fit and leaves all three slopes invariant.
    q
        Moment grid; defaults to 41 points on :math:`[-5,5]`. Negative
        :math:`q` is required --- it is what resolves :math:`\alpha>\alpha_0`.
    scales
        Box factors in units of fine cells. Defaults to a geometric ladder of
        exact divisors of ``len(mu)`` when at least five exist, else a plain
        geometric ladder. Factors leaving fewer than four boxes are dropped.
    truncate
        Allow the trailing partial box to be dropped when a factor does not
        divide ``len(mu)``; see :func:`coarse_grain`.
    """
    m = validate_measure(mu, name="measure", normalise=True, allow_zero=True)
    n = m.size
    if n < 16:
        raise ValueError(f"a measure of {n} cells cannot support a scaling fit")
    qa = validate_q_grid(q)
    sc = _resolve_scales(scales, n, truncate=truncate)

    n_s = sc.size
    a = np.empty((qa.size, n_s), dtype=np.float64)
    f = np.empty((qa.size, n_s), dtype=np.float64)
    lz = np.empty((qa.size, n_s), dtype=np.float64)
    counts = np.empty(n_s, dtype=np.int64)
    n_boxes = np.empty(n_s, dtype=np.int64)
    n_empty = np.empty(n_s, dtype=np.int64)
    n_trunc = np.empty(n_s, dtype=np.int64)

    for j, s in enumerate(sc):
        step = int(s)
        boxes = coarse_grain(m, step, truncate=True)
        nonzero = boxes > 0.0
        n_boxes[j] = boxes.size
        n_empty[j] = int(boxes.size - nonzero.sum())
        n_trunc[j] = n - boxes.size * step
        live = boxes[nonzero]
        if live.size < 2:
            raise ValueError(
                f"box factor s={step} leaves {live.size} non-empty boxes; the tilted measure "
                "is degenerate there. Reduce the largest scale."
            )
        counts[j] = live.size
        a[:, j], f[:, j], lz[:, j] = _tilted_moments(live, qa)

    return ChhabraJensenSurface(
        q=qa,
        scales=sc,
        box_size=np.asarray(sc / float(n), dtype=np.float64),
        alpha_numerator=a,
        f_numerator=f,
        log_partition=lz,
        counts=counts,
        n_boxes=n_boxes,
        n_empty=n_empty,
        n_truncated=n_trunc,
        provenance=provenance,
    )


def chhabra_jensen_structure(
    mu: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    truncate: bool = True,
) -> StructureFunctions:
    r"""The raw scaling surface alone: :math:`Z(q,\delta)=\sum_i\mu_i(\delta)^q`.

    Provided so exponents can be refitted over a different scaling range without
    recomputing any partition sum. Note the convention: the slope of
    :math:`\ln Z` against :math:`\ln\delta` is :math:`\tau`, **not**
    :math:`\zeta`; the two differ by ``d_support``.
    """
    return chhabra_jensen_surface(
        mu, q=q, scales=scales, truncate=truncate
    ).to_structure_functions()


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ChhabraJensenFits:
    r"""One regression per :math:`q` for each of :math:`\alpha`, :math:`f`, :math:`\tau`.

    The point of the Chhabra--Jensen construction is that :math:`\alpha` and
    :math:`f` are each *measured*, so each carries its own standard error and
    confidence interval instead of inheriting one through a numerical
    derivative. Those live here.
    """

    q: FloatArray
    alpha_fits: tuple[LogLogFit, ...]
    f_fits: tuple[LogLogFit, ...]
    tau_fits: tuple[LogLogFit, ...]
    scaling_range: ScalingRange

    def slopes(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``(alpha, f, tau_direct)`` as arrays over the ``q`` grid."""
        return (
            np.array([fit.slope for fit in self.alpha_fits], dtype=np.float64),
            np.array([fit.slope for fit in self.f_fits], dtype=np.float64),
            np.array([fit.slope for fit in self.tau_fits], dtype=np.float64),
        )

    def stderrs(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``(alpha_se, f_se, tau_se)`` as arrays over the ``q`` grid."""
        return (
            np.array([fit.slope_stderr for fit in self.alpha_fits], dtype=np.float64),
            np.array([fit.slope_stderr for fit in self.f_fits], dtype=np.float64),
            np.array([fit.slope_stderr for fit in self.tau_fits], dtype=np.float64),
        )


def chhabra_jensen_fits(
    surface: ChhabraJensenSurface,
    *,
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
    min_points: int = 5,
) -> ChhabraJensenFits:
    r"""Regress all three numerators against :math:`\ln\delta`.

    The scaling range is chosen **once**, on the :math:`\alpha`-numerator row
    nearest :math:`q=0`, and then imposed on every :math:`q` and on all three
    quantities. Choosing it independently per row would break the exact identity
    :math:`f=q\alpha-\tau` (different design matrices) and would let each
    :math:`q` shop for its own window, which is a well-known way to manufacture
    curvature in :math:`\tau`. The :math:`q\approx0` row is the natural pivot: at
    :math:`q=0` the tilt is uniform, so the fitted quantity is the plain average
    coarse exponent, free of any moment-selection effect.

    Weights are the non-empty box counts (``FitMethod.WLS``, the default): the
    variance of a log-moment averaged over :math:`n_j` boxes falls like
    :math:`1/n_j`, and an unweighted fit over a geometric ladder is dominated by
    the handful of coarse scales holding four boxes.
    """
    delta = surface.box_size
    weights_all = weighted_by_counts(surface.counts)
    pivot = int(np.argmin(np.abs(surface.q)))

    if select_range:
        if delta.size < min_points:
            raise ValueError(
                f"select_range needs at least min_points={min_points} scales, got {delta.size}; "
                "widen the scale ladder or fit the full range"
            )
        chosen = select_scaling_range(
            delta,
            surface.alpha_numerator[pivot],
            min_points=min_points,
            weights=weights_all,
            method=fit_method,
            log_y=False,
        )
        mask = chosen.mask
        srange = chosen.scaling_range
    else:
        mask = np.ones(delta.size, dtype=bool)
        srange = ScalingRange(
            s_min=float(delta.min()),
            s_max=float(delta.max()),
            n_scales=int(delta.size),
            selection="manual",
        )

    x = delta[mask]
    w = weights_all[mask]

    def fit_row(y: FloatArray) -> LogLogFit:
        return loglog_fit(
            x, y[mask], method=fit_method, weights=w, log_y=False, scaling_range=srange
        )

    return ChhabraJensenFits(
        q=surface.q,
        alpha_fits=tuple(fit_row(row) for row in surface.alpha_numerator),
        f_fits=tuple(fit_row(row) for row in surface.f_numerator),
        tau_fits=tuple(fit_row(row) for row in surface.log_partition),
        scaling_range=srange,
    )


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def chhabra_jensen_from_surface(
    surface: ChhabraJensenSurface,
    *,
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
    d_support: float = 1.0,
    min_points: int = 5,
    provenance: Provenance | None = None,
) -> HolderSpectrum:
    r"""Turn a precomputed surface into a spectrum, with the implied exponents.

    ``provenance`` overrides the record carried by the surface; the estimators
    use it to reference the *original* input (a price series, say) rather than
    the derived measure.

    When the :math:`q` grid is wide enough that :math:`Z(q,\delta)` overflows
    float64 (:math:`\ln Z>709`, reachable for :math:`q\ll0` on a deep cascade)
    the exponents are still returned --- they are fitted from the log surface
    --- and only :attr:`ScalingExponents.structure` is dropped, with the reason
    and :math:`\max\ln Z` recorded in ``provenance.diagnostics``. Call
    :meth:`ChhabraJensenSurface.to_structure_functions` directly to get the
    explicit error instead.
    """
    fits = chhabra_jensen_fits(
        surface, fit_method=fit_method, select_range=select_range, min_points=min_points
    )
    alpha, f_alpha, tau_direct = fits.slopes()
    alpha_se, f_se, tau_se = fits.stderrs()
    q = surface.q
    d = float(d_support)

    # The mandated conversion: this estimator is native to the measure view, and
    # the canonical carrier stores the signal exponent.
    tau = q * alpha - f_alpha
    zeta = tau + d
    identity_gap = float(np.max(np.abs(tau - tau_direct)))

    # Every fit above lives in the log domain and is therefore immune to the
    # overflow of Z itself. Materialising Z for the audit surface is not, so a
    # q grid wide enough to overflow it must cost the surface, never the
    # exponents; the omission is recorded rather than left to be inferred.
    structure: StructureFunctions | None
    overflow: dict[str, Any] = {}
    try:
        structure = surface.to_structure_functions()
    except ValueError as exc:
        if "overflows float64" not in str(exc):
            raise
        structure = None
        overflow = {
            "structure_omitted": "Z(q, delta) overflows float64; refit from the log surface",
            "log_partition_max": float(surface.log_partition.max()),
        }

    prov = provenance if provenance is not None else surface.provenance
    if prov is not None:
        prov = prov.with_diagnostics(
            **surface.diagnostics,
            **overflow,
            tau_identity_max_abs_gap=identity_gap,
            n_scales_fitted=int(fits.scaling_range.n_scales),
            decades=float(fits.scaling_range.decades),
        )

    ci_lo = np.array([fit.slope_ci[0] for fit in fits.tau_fits], dtype=np.float64) + d
    ci_hi = np.array([fit.slope_ci[1] for fit in fits.tau_fits], dtype=np.float64) + d

    exponents = ScalingExponents(
        q=q,
        zeta=np.asarray(zeta, dtype=np.float64),
        d_support=d,
        zeta_stderr=np.asarray(tau_se, dtype=np.float64),
        zeta_ci=(ci_lo, ci_hi),
        fits=fits.tau_fits,
        scaling_range=fits.scaling_range,
        structure=structure,
        estimator=ESTIMATOR,
        provenance=prov,
    )
    return HolderSpectrum(
        alpha=np.asarray(alpha, dtype=np.float64),
        f_alpha=np.asarray(f_alpha, dtype=np.float64),
        q=q,
        alpha_stderr=np.asarray(alpha_se, dtype=np.float64),
        f_stderr=np.asarray(f_se, dtype=np.float64),
        d_support=d,
        method="chhabra-jensen",
        exponents=exponents,
        provenance=prov,
    )


def _provenance(
    estimator: str,
    inputs: tuple[InputRef, ...],
    params: dict[str, Any],
    seed: SeedLike,
) -> Provenance:
    return new_provenance(
        estimator,
        estimator_version=ESTIMATOR_VERSION,
        params=params,
        inputs=inputs,
        # The method is exactly deterministic; recording a freshly drawn seed
        # sequence for seed=None would make the root differ between identical runs.
        seed=seed_info(seed) if seed is not None else None,
        notes=(_NOTE,),
    )


def _base_params(
    q: FloatArray,
    scales: NDArray[np.int64],
    *,
    fit_method: FitMethod,
    select_range: bool,
    d_support: float,
    truncate: bool,
    min_points: int,
    seed: SeedLike,
) -> dict[str, Any]:
    return {
        "q": [float(v) for v in q],
        "scales": [int(v) for v in scales],
        "fit_method": str(fit_method),
        "select_range": bool(select_range),
        "d_support": float(d_support),
        "truncate": bool(truncate),
        "min_points": int(min_points),
        "seed": None if seed is None else repr(seed),
    }


def chhabra_jensen(
    mu: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
    d_support: float = 1.0,
    truncate: bool = True,
    min_points: int = 5,
    seed: SeedLike = None,
) -> HolderSpectrum:
    r"""Direct determination of :math:`f(\alpha)` for a discrete measure.

    Estimates :math:`\alpha(q)` and :math:`f(q)` as the slopes of the tilted
    averages :math:`A(q,\delta)` and :math:`F(q,\delta)` against
    :math:`\ln\delta` (module docstring for the derivation), and reports the
    implied measure exponent :math:`\tau=q\alpha-f` on the returned
    :attr:`~mmar.core.types.HolderSpectrum.exponents`, converted to the
    canonical signal exponent :math:`\zeta=\tau+d`.

    Parameters
    ----------
    mu
        Non-negative cell masses at the finest resolution. Zero cells are
        allowed and are excluded box-by-box after coarse-graining; the count is
        recorded (see the module docstring, bias 1).
    q
        Moment grid, default 41 points on :math:`[-5,5]`. **Negative** :math:`q`
        is mandatory for the right branch :math:`\alpha>\alpha_0`; a
        :math:`q\ge0` grid cannot reach it in principle, not merely poorly.
    scales
        Box factors in fine cells; see :func:`chhabra_jensen_surface`. The
        default ladder reaches down to ``s=2``, which is safe on an exactly
        self-similar cascade and *not* safe on anything else --- see bias 5 in
        the module docstring, where an i.i.d. null returns a confident
        :math:`\Delta\alpha=1.82` under that default.
    fit_method
        ``WLS`` by default, weighted by the non-empty box count at each scale.
        ``THEILSEN`` and ``HUBER`` are available for a contaminated ladder but
        break the exact identity :math:`f=q\alpha-\tau`, whose residual is then
        reported in ``provenance.diagnostics["tau_identity_max_abs_gap"]``.
    select_range
        Choose the scaling window automatically, once, from the
        :math:`q\approx0` row; see :func:`chhabra_jensen_fits`.
    d_support
        Topological dimension of the support, used only for
        :math:`\zeta=\tau+d`. The spectrum itself is already in the measure
        convention, so a space-filling measure has :math:`\max_\alpha f = d`
        *as an estimate*, not by construction --- which makes it a genuine
        check rather than a normalisation.
    seed
        Accepted for signature uniformity across the estimator family. This
        method is exactly deterministic and consumes no randomness; the value is
        recorded in the provenance parameters and nowhere else.

    Returns
    -------
    HolderSpectrum
        ``alpha``, ``f_alpha`` with per-:math:`q` standard errors, and
        ``exponents`` carrying :math:`\zeta`, its own standard errors and
        confidence intervals from the direct :math:`\ln Z` fits, the fitted
        scaling range, and the partition-function surface.

    Notes
    -----
    Sanity anchors, both tested against the closed form: :math:`f(q{=}0)=d` and
    hence :math:`\tau(0)=-d`; :math:`\tau(1)=0` for a conservative measure. On
    the deterministic binomial cascade the numerators are *exactly affine* in
    :math:`\ln\delta` at every :math:`b`-adic scale, so the recovery is exact to
    floating point rather than asymptotic.

    References
    ----------
    Chhabra, A. and Jensen, R. V. (1989). *Phys. Rev. Lett.* **62**, 1327.
    """
    surface = chhabra_jensen_surface(mu, q=q, scales=scales, truncate=truncate)
    prov = _provenance(
        ESTIMATOR,
        (InputRef.from_array("mu", np.asarray(mu, dtype=np.float64).ravel()),),
        _base_params(
            surface.q,
            surface.scales,
            fit_method=fit_method,
            select_range=select_range,
            d_support=d_support,
            truncate=truncate,
            min_points=min_points,
            seed=seed,
        ),
        seed,
    )
    return chhabra_jensen_from_surface(
        surface,
        fit_method=fit_method,
        select_range=select_range,
        d_support=d_support,
        min_points=min_points,
        provenance=prov,
    )


def chhabra_jensen_series(
    x: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
    d_support: float = 1.0,
    truncate: bool = True,
    min_points: int = 5,
    nan_policy: NanPolicy = "raise",
    seed: SeedLike = None,
) -> HolderSpectrum:
    r"""Chhabra--Jensen applied to the **measure induced by the increments** of a series.

    The measure analysed is

    .. math::
        \mu_i \;=\; \frac{|X_{i+1}-X_i|}{\sum_j |X_{j+1}-X_j|},

    i.e. the normalised total-variation measure of :math:`X`, coarse-grained by
    *summation* like any other measure. This is a well-defined multifractal
    object with its own spectrum, and it is **not** the structure-function or
    wavelet-leader spectrum of :math:`X`. The distinction is not cosmetic:

    *When they coincide.* If :math:`X` is itself the primitive of a positive
    measure --- monotone, :math:`dX\ge0`, e.g. a trading time
    :math:`\theta(t)=\mu([0,t])` or a devil's staircase --- then
    :math:`\mu` above *is* that measure, the two descriptions are of the same
    object, and this estimator returns its :math:`\tau_\mu` with
    :math:`\tau_\mu(1)=0`. This is the intended use for MMAR trading time.

    *When they do not.* For a signed, oscillating signal the box mass is a sum of
    :math:`s` absolute increments, not the increment over the box:

    .. math::
        \mu_i(s) = \sum_{j\in\text{box}}|\Delta X_j| \;\ne\; |X(t+s)-X(t)| .

    For fractional Brownian motion the summands have no multiplicative
    structure, so :math:`\mu_i(s) = s\,\mathbb E|\Delta X|\,(1+O(s^{H-1}))` and
    the measure is asymptotically **uniform**: :math:`\alpha\equiv1`,
    :math:`f(1)=1`, for *every* :math:`H`. The Hurst exponent is not identifiable
    from the total-variation measure, and the residual spectral width
    :math:`O(s^{H-1})` is the estimator's finite-size noise floor, which widens
    with :math:`H` because long memory makes the box sums fluctuate more. Use
    a structure-function or wavelet-leader estimator for a signal; use this one
    for a measure.

    A related trap: had the box mass been defined as the :math:`\delta`-increment
    :math:`|X(t+\delta)-X(t)|` instead, the construction would not be additive,
    :math:`\sum_i\mu_i` would itself depend on :math:`\delta`, and the
    normalisation would shift every exponent by :math:`\tau_X(1)=\zeta_X(1)-1`.
    That shift is exactly why the increment version is *not* implemented here.

    Parameters
    ----------
    x
        The series. Its first difference must not vanish identically.
    nan_policy
        Passed to :func:`~mmar.core.validate.validate_series`; ``"raise"`` by
        default because interpolation manufactures regularity.

    Other parameters are as in :func:`chhabra_jensen`.
    """
    xa = validate_series(x, name="series", nan_policy=nan_policy, allow_constant=False)
    mu = np.abs(np.diff(xa))
    if float(mu.sum()) <= 0.0:
        raise ValueError("the series has zero total variation; its increment measure is empty")

    surface = chhabra_jensen_surface(mu, q=q, scales=scales, truncate=truncate)
    params = _base_params(
        surface.q,
        surface.scales,
        fit_method=fit_method,
        select_range=select_range,
        d_support=d_support,
        truncate=truncate,
        min_points=min_points,
        seed=seed,
    )
    params["nan_policy"] = str(nan_policy)
    params["measure"] = "total_variation_of_increments"
    prov = _provenance(
        f"{ESTIMATOR}_series",
        (InputRef.from_array("series", xa),),
        params,
        seed,
    )
    return chhabra_jensen_from_surface(
        surface,
        fit_method=fit_method,
        select_range=select_range,
        d_support=d_support,
        min_points=min_points,
        provenance=prov,
    )
