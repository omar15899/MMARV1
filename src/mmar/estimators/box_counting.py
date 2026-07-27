r"""Box counting: coarse Hölder exponents, the histogram method, generalised dimensions.

This is the *measure-side* branch of the multifractal formalism, and the oldest
one. Everything here operates on a discrete positive measure :math:`\mu`
supported on :math:`[0,1]`, partitioned into boxes of side :math:`\delta`, and
everything is derived from the two objects

.. math::
    \mu_i(\delta) = \mu\big([i\delta,(i+1)\delta)\big),
    \qquad
    Z(q,\delta) = \sum_{i:\,\mu_i>0} \mu_i(\delta)^{\,q} \sim \delta^{\tau(q)} .

The *coarse Hölder exponent* of a box is

.. math::
    \alpha_i(\delta) = \frac{\ln \mu_i(\delta)}{\ln \delta},

and the *singularity spectrum* is the exponent of the number of boxes carrying a
given :math:`\alpha`,

.. math::
    N_\delta(\alpha)\,\mathrm{d}\alpha \sim \delta^{-f(\alpha)}\,\mathrm{d}\alpha .

Two routes to :math:`f` are implemented and they are **not** the same estimator:

*Indirect (Legendre).* Fit :math:`\tau(q)` from :math:`Z(q,\delta)`, then
:math:`f=\inf_q\{q\alpha-\tau(q)\}`. Stable, but it can only ever return a
concave curve, and the :math:`q` window silently bounds the recoverable
:math:`\alpha` range.

*Direct (histogram).* Bin the :math:`\alpha_i(\delta)` and read :math:`f` off the
scaling of the bin occupancies. It sees non-concave and truncated spectra, and
it converges appallingly slowly --- quantified in
:func:`histogram_spectrum`.

Conventions (see :mod:`mmar.core.types`, which is law)
------------------------------------------------------
The box partition of a measure natively produces :math:`\tau(q)`. Every
:class:`~mmar.core.types.ScalingExponents` in this library stores :math:`\zeta`,
so the estimators below return

.. math::
    \zeta(q) = \tau(q) + d ,

with :math:`d=` ``d_support`` (:math:`1` for a measure on an interval). The
:math:`d` is added once, in :func:`generalised_dimensions`, and nowhere else.

Scales are given as **integer aggregation factors** :math:`s` --- the number of
native cells merged into one box --- and the box side is :math:`\delta=s/n` with
:math:`n=\operatorname{len}(\mu)`, i.e. the support is :math:`[0,1]`. Fitting
against :math:`\ln\delta` rather than :math:`\ln s` changes only the intercept of
:math:`\tau`, but it is the *absolute* :math:`\delta` that enters
:math:`\alpha=\ln\mu/\ln\delta`, so the two must not be confused. That confusion,
one level deep, is legacy defect D8 (:doc:`LEGACY_AUDIT`), which inflated
:math:`f(\alpha)` by :math:`k/(k-1)`; :func:`coarse_holder` carries an explicit
regression test for it.

References
----------
Halsey, T. C., Jensen, M. H., Kadanoff, L. P., Procaccia, I., Shraiman, B. I.
(1986). Fractal measures and their singularities: the characterization of
strange sets. *Phys. Rev. A* **33**, 1141--1151.

Evertsz, C. J. G. and Mandelbrot, B. B. (1992). Multifractal measures.
Appendix B of H.-O. Peitgen, H. Jürgens, D. Saupe, *Chaos and Fractals*,
Springer, 921--953.

Chhabra, A. and Jensen, R. V. (1989). Direct determination of the
:math:`f(\alpha)` singularity spectrum. *Phys. Rev. Lett.* **62**, 1327--1330.

Meneveau, C. and Sreenivasan, K. R. (1991). The multifractal nature of
turbulent energy dissipation. *J. Fluid Mech.* **224**, 429--484.

Grassberger, P. (1988). Finite sample corrections to entropy and dimension
estimates. *Phys. Lett. A* **128**, 369--373.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from mmar.core.legendre import spectrum_from_exponents
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
from mmar.core.validate import check_scales, validate_measure, validate_q_grid

__all__ = [
    "AlphaLocator",
    "GeneralisedDimensions",
    "SpectrumMethod",
    "box_partition",
    "coarse_holder",
    "dq",
    "generalised_dimensions",
    "generalised_dimensions_structure",
    "histogram_spectrum",
    "information_dimension",
    "multifractal_measure_spectrum",
]

MODULE_VERSION = "1"

AlphaLocator = Literal["weighted", "centre"]
SpectrumMethod = Literal["legendre", "histogram"]

_REFERENCE_Q = 2.0
"""The ``q`` whose log--log plot drives automatic scaling-range selection.

One window must be shared by every ``q`` --- otherwise ``tau`` is assembled from
slopes fitted over different scale ranges and is not the exponent of any single
scaling law. ``q=2`` is the conventional reference: it is the best-conditioned
moment that still sees the multifractality.
"""


# --------------------------------------------------------------------------- #
# Coarse graining
# --------------------------------------------------------------------------- #


def _aggregate(cells: FloatArray, factor: int) -> FloatArray:
    """Merge ``factor`` neighbouring cells by summation (exact box partition)."""
    if factor == 1:
        return cells
    if cells.size % factor != 0:
        raise ValueError(
            f"cannot aggregate {cells.size} cells in groups of {factor}: the last box "
            "would be partial. Box scales must divide the number of cells; truncating "
            "instead is legacy defect D6."
        )
    return np.asarray(cells.reshape(-1, factor).sum(axis=1), dtype=np.float64)


def _default_scales(n: int, b: int, min_boxes: int) -> NDArray[np.int64]:
    r"""Geometric :math:`b`-adic aggregation factors :math:`1,b,b^2,\dots`.

    Stops as soon as fewer than ``min_boxes`` boxes would remain, or as soon as
    the factor stops dividing ``n``.
    """
    out: list[int] = []
    s = 1
    while s <= n and n % s == 0 and n // s >= min_boxes:
        out.append(s)
        s *= b
    if not out:
        raise ValueError(f"no {b}-adic box scale leaves >= {min_boxes} boxes for n={n} cells")
    return np.array(out, dtype=np.int64)


def _resolve_scales(
    scales: Sequence[int] | NDArray[Any] | None,
    n: int,
    b: int,
    min_boxes: int,
) -> NDArray[np.int64]:
    r"""Validate box scales: divisors of ``n`` leaving at least ``min_boxes`` boxes.

    ``check_scales`` is the spine's guard and is used for every factor
    :math:`s\ge 2`. It rejects :math:`s=1` by construction (a "scale" of one
    sample is meaningless for a *fluctuation* estimator), but for a box partition
    :math:`s=1` is the native resolution of the measure and is perfectly
    admissible, so it is admitted here under the same ``min_boxes`` rule.
    """
    if b < 2:
        raise ValueError(f"the aggregation base must be >= 2, got {b}")
    if min_boxes < 4:
        raise ValueError(
            f"min_boxes={min_boxes} < 4: a slope through fewer than four boxes per "
            "scale is not an estimate"
        )
    raw = _default_scales(n, b, min_boxes) if scales is None else np.asarray(scales).ravel()
    s = np.unique(np.asarray(raw, dtype=np.int64))
    if s.size == 0:
        raise ValueError("box scales is empty")
    if np.any(s < 1):
        raise ValueError(f"box scales must be >= 1, got {s.min()}")
    big = s[s >= 2]
    kept = (
        check_scales(big, n, min_windows=min_boxes, name="box scales")
        if big.size
        else np.empty(0, dtype=np.int64)
    )
    if s[0] == 1:
        if n < min_boxes:
            raise ValueError(f"n={n} cells cannot supply {min_boxes} boxes at s=1")
        kept = np.concatenate((np.ones(1, dtype=np.int64), kept))
    bad = kept[n % kept != 0]
    if bad.size:
        raise ValueError(
            f"box scales {bad.tolist()} do not divide n={n}; a box partition with a "
            "partial last box has a different exponent at every scale"
        )
    if kept.size < 3:
        raise ValueError(
            f"only {kept.size} admissible box scales for n={n}; a power-law fit with "
            "error bars needs at least three"
        )
    return np.ascontiguousarray(kept)


@dataclass(frozen=True, slots=True)
class _BoxSurface:
    """Internal log-domain scaling surface; never leaves the module unlogged."""

    q: FloatArray
    scales: NDArray[np.int64]
    delta: FloatArray
    log_z: FloatArray
    """Shape ``(n_q, n_scales)``, natural log of :math:`Z(q,\\delta)`."""
    counts: NDArray[np.int64]
    """Occupied boxes at each scale: the effective sample size of the log-moment."""
    n_empty: NDArray[np.int64]
    entropy: FloatArray
    r""":math:`\sum_i \mu_i\ln\mu_i` at each scale (minus the Shannon entropy)."""


def _box_surface(
    m: FloatArray,
    q: FloatArray,
    scales: NDArray[np.int64],
) -> _BoxSurface:
    r"""Partition function on the log scale, by successive aggregation.

    Empty boxes are dropped from :math:`Z`. For :math:`q>0` that is a no-op
    (:math:`0^q=0`); for :math:`q\le 0` it is the box-counting convention --- the
    sum over *all* boxes would be :math:`+\infty` --- and the number dropped is
    recorded so the caller can see how much of the support is being ignored.
    """
    n = m.size
    n_s = scales.size
    log_z = np.empty((q.size, n_s), dtype=np.float64)
    counts = np.empty(n_s, dtype=np.int64)
    n_empty = np.empty(n_s, dtype=np.int64)
    entropy = np.empty(n_s, dtype=np.float64)

    current = m
    current_s = 1
    for j in range(n_s):
        s = int(scales[j])
        current = _aggregate(current, s // current_s) if s % current_s == 0 else _aggregate(m, s)
        current_s = s
        occupied = current > 0.0
        cells = current[occupied]
        if cells.size < 2:
            raise ValueError(
                f"box scale s={s} leaves {cells.size} occupied boxes; the partition "
                "function degenerates"
            )
        counts[j] = cells.size
        n_empty[j] = current.size - cells.size
        log_cells = np.log(cells)
        entropy[j] = float(np.sum(cells * log_cells))
        for i in range(q.size):
            log_z[i, j] = float(logsumexp(float(q[i]) * log_cells))

    return _BoxSurface(
        q=q,
        scales=scales,
        delta=np.asarray(scales, dtype=np.float64) / float(n),
        log_z=log_z,
        counts=counts,
        n_empty=n_empty,
        entropy=entropy,
    )


# --------------------------------------------------------------------------- #
# Coarse Hölder exponents
# --------------------------------------------------------------------------- #


def _moments_from_log(log_z: FloatArray) -> FloatArray:
    r"""Exponentiate the log-domain surface, refusing to store ``inf`` or ``0``.

    :class:`~mmar.core.types.StructureFunctions` stores the moments :math:`Z`
    themselves, not their logarithms, so a :math:`q` grid wide enough to push
    :math:`\ln Z` outside :math:`[\ln\text{tiny}, \ln\text{max}]` cannot be
    represented. Overflow would store ``inf`` (with a bare ``RuntimeWarning``)
    and underflow would store an exact ``0`` *silently*; either poisons any
    later refit, which takes a logarithm of what is stored. The fit itself never
    sees this --- it works on ``log_z`` --- so the only honest response is to
    refuse the grid rather than hand back a surface that cannot be refitted.
    """
    with np.errstate(over="ignore", under="ignore"):
        values = np.exp(log_z)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        lo, hi = float(log_z.min()), float(log_z.max())
        raise ValueError(
            f"Z(q, delta) spans log-range [{lo:.1f}, {hi:.1f}] nats and does not fit in "
            "float64. Narrow the q grid or use fewer cascade levels."
        )
    return np.asarray(values, dtype=np.float64)


def coarse_holder(
    mu: Sequence[float] | NDArray[Any],
    delta: float,
    *,
    normalise: bool = True,
) -> FloatArray:
    r"""Coarse Hölder exponent of every non-empty box of side :math:`\delta`.

    .. math::
        \alpha_i(\delta) \;=\; \frac{\ln \mu_i(\delta)}{\ln \delta},
        \qquad \mu_i(\delta)=\mu\big([i\delta,(i+1)\delta)\big) .

    **The denominator is the box side, not the side of the previous
    generation.** For a :math:`b`-adic cascade observed at level :math:`k` the
    boxes have :math:`\delta=b^{-k}` and the denominator is :math:`-k\ln b`.
    Using :math:`b^{-(k-1)}` --- the predecessor's bug, legacy defect D8 ---
    multiplies every :math:`\alpha`, and hence the whole spectrum, by
    :math:`k/(k-1)`: a :math:`9\%` inflation at :math:`k=12` that survives every
    plot unnoticed because the curve keeps its shape.

    Parameters
    ----------
    mu
        Cell masses on a uniform partition of :math:`[0,1]` into
        ``len(mu)`` cells. The array is coarse-grained to boxes of side
        ``delta`` by summation, which requires ``round(1/delta)`` to divide
        ``len(mu)``.
    delta
        Box side in :math:`(0,1)`. :math:`\delta=1` is rejected: the single box
        carries the whole mass and :math:`\alpha=0/0`.
    normalise
        Rescale to unit total mass first. :math:`\alpha` is defined for a
        probability measure; a total mass :math:`M\neq 1` shifts every exponent
        by :math:`\ln M/\ln\delta`, uniformly and invisibly.

    Returns
    -------
    FloatArray
        One exponent per **occupied** box, in box order. Empty boxes have
        :math:`\alpha=+\infty` and are omitted rather than returned as ``inf``;
        their number is ``round(1/delta) - out.size``.

    Notes
    -----
    Bias. None: this is a definition, not an estimate. What it *is* subject to
    is discretisation --- at level :math:`k` of a :math:`b`-adic cascade with
    :math:`b` distinct multipliers, :math:`\alpha` takes only :math:`k+1` values,
    on a lattice of spacing :math:`(\alpha_{\max}-\alpha_{\min})/k`. Any
    histogram built on these values must use bins wider than that lattice or it
    will alias (see :func:`histogram_spectrum`).

    Exactness on the binomial. With weights :math:`(m_0,m_1)` at level :math:`k`,
    the cell whose index has :math:`j` zero bits has mass
    :math:`m_0^{\,j}m_1^{\,k-j}` and therefore

    .. math::
        \alpha \;=\; -\,\frac{j\ln m_0 + (k-j)\ln m_1}{k\ln b},

    with multiplicity :math:`\binom{k}{j}`. This identity is combinatorial, not
    asymptotic, and is asserted to :math:`10^{-12}` in the test suite.

    References
    ----------
    Evertsz and Mandelbrot (1992), Appendix B, eq. (B.9).
    """
    m = validate_measure(mu, name="measure", normalise=normalise)
    d = float(delta)
    if not math.isfinite(d) or not 0.0 < d < 1.0:
        raise ValueError(
            f"delta must lie strictly in (0, 1), got {delta!r}; delta = 1 is the single "
            "box carrying the whole mass, where the exponent is 0/0"
        )
    n_boxes = round(1.0 / d)
    if n_boxes < 2 or abs(n_boxes * d - 1.0) > 1e-9:
        raise ValueError(
            f"delta={d!r} does not correspond to an integer number of boxes (1/delta = {1.0 / d!r})"
        )
    if m.size % n_boxes != 0:
        raise ValueError(
            f"a partition into {n_boxes} boxes does not divide the {m.size} cells of the "
            "measure; the last box would be partial"
        )
    boxes = _aggregate(m, m.size // n_boxes)
    occupied = boxes > 0.0
    return np.asarray(np.log(boxes[occupied]) / math.log(d), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Partition function
# --------------------------------------------------------------------------- #


def box_partition(
    mu: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    b: int = 2,
    min_boxes: int = 4,
) -> StructureFunctions:
    r"""The raw scaling surface :math:`Z(q,\delta)=\sum_i \mu_i(\delta)^q`.

    The measure is coarse-grained by successive aggregation: the boxes at scale
    :math:`s` are built from those at the previous scale whenever the previous
    scale divides it, so the whole surface costs :math:`O(n)` per scale and the
    box sums are *exact* (no re-binning error).

    .. math::
        Z(q,\delta) \sim \delta^{\tau(q)},
        \qquad \delta = s/n .

    Returned separately from the exponents so that :math:`\tau` can be refitted
    over a different scaling range without recomputing the surface, which is the
    expensive half.

    Parameters
    ----------
    mu
        Cell masses; normalised to unit total mass.
    q
        Moment grid; default 41 points on :math:`[-5,5]`, which includes
        :math:`q=0` and :math:`q=1` exactly so that the anchors
        :math:`\tau(0)=-D_0` and :math:`\tau(1)=0` can be read directly.
    scales
        Integer aggregation factors, each a divisor of ``len(mu)``. Default: the
        :math:`b`-adic ladder :math:`1,b,b^2,\dots` truncated at ``min_boxes``.
    b
        Base of the default ladder.
    min_boxes
        Scales leaving fewer boxes than this are dropped by
        :func:`~mmar.core.validate.check_scales`.

    Returns
    -------
    StructureFunctions
        ``values[i, j]`` :math:`= Z(q_i,\delta_j)`, ``scales`` the box sides
        :math:`\delta` (not the integer factors), ``counts`` the number of
        occupied boxes at each scale.

    Raises
    ------
    ValueError
        If any :math:`Z(q,\delta)` overflows or underflows float64. The sum is
        formed by :func:`scipy.special.logsumexp`, so the *fit* never sees an
        overflow, but :class:`~mmar.core.types.StructureFunctions` stores the
        moments themselves; a grid of :math:`|q|` large enough to break that
        representation is refused rather than silently stored as ``inf``.
    """
    m = validate_measure(mu, name="measure", normalise=True)
    qa = validate_q_grid(q)
    s = _resolve_scales(scales, m.size, b, min_boxes)
    surface = _box_surface(m, qa, s)
    return StructureFunctions(
        q=qa,
        scales=surface.delta,
        values=_moments_from_log(surface.log_z),
        counts=surface.counts,
    )


def generalised_dimensions_structure(
    mu: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    b: int = 2,
    min_boxes: int = 4,
) -> StructureFunctions:
    """Standard-named alias of :func:`box_partition`.

    Every estimator in :mod:`mmar.estimators` exports ``<name>_structure``
    returning only the raw scaling surface; for this family that surface is the
    box partition function, whose own name is older and more descriptive.
    """
    return box_partition(mu, q=q, scales=scales, b=b, min_boxes=min_boxes)


# --------------------------------------------------------------------------- #
# tau(q), zeta(q) and the generalised dimensions
# --------------------------------------------------------------------------- #


def _select_mask(
    surface: _BoxSurface, fit_method: FitMethod, select_range: bool
) -> tuple[NDArray[np.bool_], ScalingRange]:
    """One scale window, shared by every ``q``; see :data:`_REFERENCE_Q`."""
    n_s = surface.delta.size
    weights = weighted_by_counts(surface.counts)
    ref = int(np.argmin(np.abs(surface.q - _REFERENCE_Q)))
    if not select_range:
        fit = loglog_fit(
            surface.delta,
            surface.log_z[ref],
            method=fit_method,
            weights=weights,
            log_y=False,
        )
        sr = ScalingRange(
            s_min=float(surface.delta.min()),
            s_max=float(surface.delta.max()),
            n_scales=n_s,
            selection="manual",
            r_squared=fit.r_squared,
            mask=(True,) * n_s,
        )
        return np.ones(n_s, dtype=bool), sr
    if n_s < 5:
        raise ValueError(
            f"select_range=True needs at least 5 box scales, got {n_s}; pass a deeper "
            "measure or select_range=False"
        )
    result = select_scaling_range(
        surface.delta,
        surface.log_z[ref],
        weights=weights,
        method=fit_method,
        log_y=False,
    )
    return result.mask, result.scaling_range


def generalised_dimensions(
    mu: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    b: int = 2,
    min_boxes: int = 4,
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
    d_support: float = 1.0,
    seed: SeedLike = None,
) -> ScalingExponents:
    r"""Box-counting exponents of a measure: :math:`\tau(q)`, carried as :math:`\zeta`.

    **What is fitted.** The slope of :math:`\ln Z(q,\delta)` against
    :math:`\ln\delta`,

    .. math::
        Z(q,\delta) = \sum_{i:\,\mu_i>0}\mu_i(\delta)^q \sim \delta^{\tau(q)} ,

    so this estimator natively produces the **measure** exponent :math:`\tau`.
    The returned object stores the signal-view exponent, obtained by the one
    conversion this library allows,

    .. math::
        \zeta(q) = \tau(q) + d, \qquad d = \texttt{d\_support},

    because a :math:`d`-dimensional support carries :math:`\delta^{-d}` boxes.
    Read :math:`\tau` back off the result as ``result.tau``; the generalised
    dimensions are :math:`D_q=\tau(q)/(q-1)` (:func:`dq`).

    **Weighting.** The number of occupied boxes falls like :math:`\delta^{-D_0}`,
    so the coarse scales carry orders of magnitude less information than the fine
    ones. The default is :class:`~mmar.core.types.FitMethod.WLS` with
    :math:`w_j=N(\delta_j)`, the occupied-box count, which is inverse-variance
    weighting for a log-moment. An unweighted fit over these scales is dominated
    by the four-box end of the ladder (legacy defect D5).

    Parameters
    ----------
    mu
        Cell masses of a positive measure; normalised internally.
    q
        Moment grid; default 41 points on :math:`[-5,5]`. **Negative** :math:`q`
        is on by default and is not optional in practice: the right branch
        :math:`\alpha>\alpha_0` of the spectrum is generated by :math:`q<0`
        alone, and a :math:`q\ge0` grid cannot reach it *in principle* (legacy
        defect D2).
    scales, b, min_boxes
        As in :func:`box_partition`.
    fit_method
        Regression used for every slope.
    select_range
        Choose the scaling window automatically from the :math:`q=2` row and
        apply it to every :math:`q`. One window is shared deliberately: slopes
        fitted over different windows are not the exponents of a single scaling
        law.
    d_support
        Topological dimension of the support; :math:`1` for a measure on an
        interval.
    seed
        Recorded in provenance for interface uniformity. This estimator is
        deterministic and the seed does not enter the result.

    Returns
    -------
    ScalingExponents
        With ``zeta``, ``zeta_stderr``, ``zeta_ci``, one
        :class:`~mmar.core.types.LogLogFit` per :math:`q`, the shared
        ``scaling_range``, the full ``structure`` surface, and provenance.

    Notes
    -----
    **Bias 1 --- finite-size linearisation at large** :math:`|q|`. With
    :math:`N` boxes, :math:`Z(q,\delta)` for large positive :math:`q` is
    dominated by the single densest box: :math:`\ln Z \to q\ln\mu_{\max}`, so
    :math:`\hat\tau(q)\to q\,\alpha_{\min}^{\text{obs}}` becomes *linear* and the
    Legendre transform collapses the left branch of :math:`f` onto a point.
    Symmetrically, :math:`q\to-\infty` is dominated by the emptiest occupied box
    and linearises the right branch. The crossover sits where the number of boxes
    contributing to :math:`Z` drops to :math:`O(1)`, i.e. around
    :math:`|q|\gtrsim \ln N/(\alpha_{\max}-\alpha_{\min})/\ln(1/\delta)`; the
    remedy is a bounded :math:`q` window, never a longer extrapolation
    (Grassberger 1988).

    **Bias 2 --- negative** :math:`q` **is dominated by the sparsest box.** The
    relative fluctuation of :math:`\mu_{\min}` is :math:`O(1)` however fine the
    partition, so :math:`\hat\tau(q<0)` has both the largest variance and the
    slowest convergence on the grid. It is nonetheless *necessary*, see ``q``
    above.

    **Bias 3 --- empty boxes.** Boxes with :math:`\mu_i=0` are excluded from
    :math:`Z`; including them makes :math:`Z(q\le 0)` infinite. The number
    excluded per scale is recorded in ``provenance.diagnostics["n_empty_boxes"]``
    and is zero for a full-support measure.

    References
    ----------
    Halsey et al. (1986), eq. (2.4); Grassberger (1988).
    """
    m = validate_measure(mu, name="measure", normalise=True)
    qa = validate_q_grid(q)
    s = _resolve_scales(scales, m.size, b, int(min_boxes))
    surface = _box_surface(m, qa, s)
    # Checked before any fitting: the surface is part of the returned contract, so a
    # q grid whose moments cannot be stored is refused here exactly as in box_partition
    # rather than silently yielding inf (overflow) or 0 (underflow) in `structure`.
    values = _moments_from_log(surface.log_z)
    mask, scaling_range = _select_mask(surface, fit_method, select_range)

    x = surface.delta[mask]
    w = weighted_by_counts(surface.counts[mask])
    fits: list[LogLogFit] = []
    tau = np.empty(qa.size, dtype=np.float64)
    stderr = np.empty(qa.size, dtype=np.float64)
    ci_lo = np.empty(qa.size, dtype=np.float64)
    ci_hi = np.empty(qa.size, dtype=np.float64)
    for i in range(qa.size):
        fit = loglog_fit(
            x,
            surface.log_z[i][mask],
            method=fit_method,
            weights=w,
            log_y=False,
            scaling_range=scaling_range,
        )
        fits.append(fit)
        tau[i] = fit.slope
        stderr[i] = fit.slope_stderr
        ci_lo[i], ci_hi[i] = fit.slope_ci

    d = float(d_support)
    zeta = tau + d
    prov = new_provenance(
        f"{__name__}.generalised_dimensions",
        estimator_version=MODULE_VERSION,
        params={
            "b": int(b),
            "min_boxes": int(min_boxes),
            "fit_method": str(fit_method),
            "select_range": bool(select_range),
            "d_support": d,
            "n_cells": int(m.size),
            "reference_q": _REFERENCE_Q,
        },
        inputs=(
            InputRef.from_array("mu", m),
            InputRef.from_array("q", qa),
            InputRef.from_array("scales", s),
        ),
        seed=None if seed is None else seed_info(seed),
        notes=(
            "box partition of a measure: natively tau(q); stored as zeta = tau + d",
            "deterministic: the seed is recorded but does not enter the computation",
        ),
    ).with_diagnostics(
        n_empty_boxes=surface.n_empty.tolist(),
        occupied_boxes=surface.counts.tolist(),
        scales_used=int(mask.sum()),
    )
    return ScalingExponents(
        q=qa,
        zeta=np.asarray(zeta, dtype=np.float64),
        d_support=d,
        zeta_stderr=stderr,
        zeta_ci=(ci_lo + d, ci_hi + d),
        fits=tuple(fits),
        scaling_range=scaling_range,
        structure=StructureFunctions(
            q=qa,
            scales=surface.delta,
            values=values,
            counts=surface.counts,
        ),
        estimator="box_counting.generalised_dimensions",
        provenance=prov,
    )


def information_dimension(
    mu: Sequence[float] | NDArray[Any],
    *,
    scales: Sequence[int] | NDArray[Any] | None = None,
    b: int = 2,
    min_boxes: int = 4,
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
) -> LogLogFit:
    r""":math:`D_1` from the Shannon-entropy slope, not from :math:`\tau(1+\epsilon)`.

    :math:`D_q=\tau(q)/(q-1)` has a removable singularity at :math:`q=1`, where
    :math:`\tau(1)=0` for a conservative measure. L'Hôpital gives

    .. math::
        D_1 \;=\; \lim_{q\to1}\frac{\tau(q)}{q-1} \;=\; \tau'(1)
              \;=\; \lim_{\delta\to0}
                \frac{\sum_i \mu_i(\delta)\ln \mu_i(\delta)}{\ln \delta} ,

    i.e. :math:`D_1` is the slope of :math:`\sum_i\mu_i\ln\mu_i` --- minus the
    Shannon entropy --- against :math:`\ln\delta`. That is what is fitted here.

    Evaluating :math:`\tau(1+\epsilon)/\epsilon` instead is a numerical trap: the
    numerator is a fitted slope with standard error :math:`\sigma`, so the
    quotient carries error :math:`\sigma/\epsilon`, which for the
    :math:`\epsilon\sim10^{-6}` that "looks like a limit" is a millionfold
    amplification of the fit noise. The entropy slope has the accuracy of an
    ordinary regression.

    Returns
    -------
    LogLogFit
        ``slope`` is :math:`D_1`, with its standard error, confidence interval
        and :math:`R^2`. The response is passed already logged
        (``log_y=False``): :math:`\sum\mu\ln\mu` is negative.

    Notes
    -----
    Bias. The entropy of a finite partition is biased *downward* by roughly
    :math:`(N_{\text{occ}}-1)/(2 N_{\text{samples}})` nats when the measure is an
    empirical histogram of finitely many samples (Grassberger 1988). For a
    measure given exactly --- a cascade, a density --- there is no such bias and
    the fit is exact at every scale for a self-similar measure.

    Exactness on the binomial: at level :math:`l`,
    :math:`\sum\mu\ln\mu = l\sum_i m_i\ln m_i` and
    :math:`\ln\delta=-l\ln b`, so the ratio is
    :math:`-\sum_i m_i\log_b m_i` at *every* level, with no residual.
    """
    m = validate_measure(mu, name="measure", normalise=True)
    s = _resolve_scales(scales, m.size, b, int(min_boxes))
    surface = _box_surface(m, np.array([0.0, 1.0, 2.0]), s)
    mask, scaling_range = _select_mask(surface, fit_method, select_range)
    return loglog_fit(
        surface.delta[mask],
        surface.entropy[mask],
        method=fit_method,
        weights=weighted_by_counts(surface.counts[mask]),
        log_y=False,
        scaling_range=scaling_range,
    )


@dataclass(frozen=True, slots=True)
class GeneralisedDimensions:
    r"""The Rényi dimension curve :math:`q\mapsto D_q` with its error bars.

    :math:`D_0` is the box dimension of the support, :math:`D_1` the information
    dimension, :math:`D_2` the correlation dimension, and :math:`q\mapsto D_q` is
    non-increasing whenever :math:`\tau` is concave with :math:`\tau(1)=0`
    (:math:`D_q` is the slope of the chord of :math:`\tau` from :math:`(1,0)`).
    """

    q: FloatArray
    dq: FloatArray
    dq_stderr: FloatArray
    d1: float
    d1_stderr: float
    d1_fit: LogLogFit
    exponents: ScalingExponents
    provenance: Provenance | None = None

    @property
    def d0(self) -> float:
        r""":math:`D_0=-\tau(0)`, the box dimension of the support."""
        return float(-(self.exponents.at(0.0) - self.exponents.d_support))

    @property
    def d2(self) -> float:
        r""":math:`D_2=\tau(2)`, the correlation dimension."""
        return float(self.exponents.at(2.0) - self.exponents.d_support)

    def at(self, q: float) -> float:
        """:math:`D_q` interpolated on the fitted grid."""
        return float(np.interp(q, self.q, self.dq))


def dq(
    mu: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    b: int = 2,
    min_boxes: int = 4,
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
    d_support: float = 1.0,
    q1_tol: float = 1e-3,
    seed: SeedLike = None,
) -> GeneralisedDimensions:
    r"""Generalised (Rényi) dimensions :math:`D_q=\tau(q)/(q-1)`.

    .. math::
        D_q \;=\; \frac{\tau(q)}{q-1}
            \;=\; \lim_{\delta\to0}\frac{1}{q-1}\,
                  \frac{\ln\sum_i\mu_i(\delta)^q}{\ln\delta},
        \qquad q\neq 1 ,

    with the :math:`q\to1` value supplied by :func:`information_dimension` ---
    the Shannon-entropy slope --- for every grid point inside ``q1_tol`` of 1.

    Parameters
    ----------
    q1_tol
        Half-width of the substitution window around :math:`q=1`. The quotient
        amplifies the standard error of :math:`\hat\tau` by :math:`1/|q-1|`, so
        inside :math:`|q-1|<10^{-3}` the quotient is at least a thousand times
        noisier than the entropy slope and is never the better estimate. Outside
        it the two agree to within their error bars, which is itself a useful
        consistency check.
    mu, scales, b, min_boxes, fit_method, select_range, d_support, seed
        As in :func:`generalised_dimensions`.

    Returns
    -------
    GeneralisedDimensions
        :math:`D_q` with standard errors, the separately fitted :math:`D_1`, and
        the underlying :class:`~mmar.core.types.ScalingExponents`.

    Notes
    -----
    Bias. :math:`D_q` inherits every bias of :math:`\hat\tau`
    (:func:`generalised_dimensions`) and adds one of its own: dividing by
    :math:`q-1` multiplies the error by :math:`1/|q-1|`, so the curve is least
    reliable exactly where it is most quoted. Monotonicity
    :math:`D_{q_1}\ge D_{q_2}` for :math:`q_1<q_2` is a theorem for the true
    :math:`\tau` and a *diagnostic* for the estimate: a violation larger than the
    error bars means the concavity of :math:`\hat\tau` has failed.

    References
    ----------
    Hentschel, H. G. E. and Procaccia, I. (1983). The infinite number of
    generalized dimensions of fractals and strange attractors. *Physica D*
    **8**, 435--444. Halsey et al. (1986).
    """
    exponents = generalised_dimensions(
        mu,
        q=q,
        scales=scales,
        b=b,
        min_boxes=min_boxes,
        fit_method=fit_method,
        select_range=select_range,
        d_support=d_support,
        seed=seed,
    )
    d1_fit = information_dimension(
        mu,
        scales=scales,
        b=b,
        min_boxes=min_boxes,
        fit_method=fit_method,
        select_range=select_range,
    )
    tol = float(q1_tol)
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"q1_tol must be finite and positive, got {q1_tol!r}")

    qa = exponents.q
    tau = exponents.tau
    se = exponents.zeta_stderr
    assert se is not None  # populated by generalised_dimensions
    near_one = np.abs(qa - 1.0) < tol
    denom = np.where(near_one, 1.0, qa - 1.0)
    values = np.where(near_one, d1_fit.slope, tau / denom)
    errors = np.where(near_one, d1_fit.slope_stderr, se / np.abs(denom))

    prov = (
        exponents.provenance.derive(
            f"{__name__}.dq",
            estimator_version=MODULE_VERSION,
            params={"q1_tol": tol},
            notes=("D_1 from the entropy slope, not from tau(1 + eps)",),
        )
        if exponents.provenance is not None
        else None
    )
    return GeneralisedDimensions(
        q=qa,
        dq=np.asarray(values, dtype=np.float64),
        dq_stderr=np.asarray(errors, dtype=np.float64),
        d1=float(d1_fit.slope),
        d1_stderr=float(d1_fit.slope_stderr),
        d1_fit=d1_fit,
        exponents=exponents,
        provenance=prov,
    )


# --------------------------------------------------------------------------- #
# The direct (histogram) method
# --------------------------------------------------------------------------- #


def histogram_spectrum(
    mu: Sequence[float] | NDArray[Any],
    *,
    scales: Sequence[int] | NDArray[Any] | None = None,
    bins: int = 21,
    b: int = 2,
    alpha_range: tuple[float, float] | None = None,
    min_boxes: int = 8,
    min_scales: int = 3,
    alpha_locator: AlphaLocator = "weighted",
    fit_method: FitMethod = FitMethod.WLS,
    d_support: float = 1.0,
    seed: SeedLike = None,
) -> HolderSpectrum:
    r"""Direct histogram spectrum: :math:`f(\alpha)` from box counts, no Legendre transform.

    **Definition.** Let :math:`N_\delta(\alpha)` be the *density* of boxes in
    coarse Hölder exponent, so that the number with
    :math:`\alpha_i(\delta)\in[\alpha,\alpha+\mathrm{d}\alpha)` is
    :math:`N_\delta(\alpha)\,\mathrm{d}\alpha`. The spectrum is defined by

    .. math::
        N_\delta(\alpha) \sim \delta^{-f(\alpha)} .

    **The bin-width term.** A histogram returns the *count*
    :math:`C_\delta(\alpha)=N_\delta(\alpha)\,\mathrm{d}\alpha`, not the density.
    Substituting the count for the density gives

    .. math::
        \hat f_{\text{naive}}(\alpha) = -\frac{\ln C_\delta(\alpha)}{\ln\delta}
        = f(\alpha) \;-\; \frac{\ln \mathrm{d}\alpha}{\ln \delta} ,

    and since :math:`\mathrm{d}\alpha<1` and :math:`\delta<1` the correction
    :math:`-\ln\mathrm{d}\alpha/\ln\delta` is strictly **positive**: the naive
    single-scale estimator is biased *low* by exactly that amount, uniformly in
    :math:`\alpha`. At :math:`\mathrm{d}\alpha=0.03`,
    :math:`\delta=2^{-14}` the bias is :math:`0.36` --- a third of the height of
    the spectrum. The correct single-scale form is

    .. math::
        \hat f(\alpha)
        = -\frac{\ln\!\big(C_\delta(\alpha)/\mathrm{d}\alpha\big)}{\ln\delta} .

    **Why a slope, not a single scale.** Writing the same relation at several
    :math:`\delta` with the bin edges held *fixed*,

    .. math::
        \ln\frac{C_\delta(\alpha)}{\mathrm{d}\alpha}
        = -f(\alpha)\,\ln\delta + \text{const},

    so :math:`f(\alpha)` is minus the slope of :math:`\ln(C_\delta/\mathrm{d}\alpha)`
    against :math:`\ln\delta`. The bin width is then a pure *intercept* term and
    cancels from the estimate **exactly**, which is the real reason to fit a
    slope rather than to divide a single logarithm.

    That matters because the bin-width term is only one of two
    :math:`O(1/\ln(1/\delta))` corrections at a single scale. The other is the
    combinatorial prefactor: at level :math:`l=\log_b(1/\delta)` of a
    :math:`b`-adic cascade Stirling gives
    :math:`C \simeq b^{\,l f}\,\mathrm{d}\alpha\,/\sqrt{l}`, so

    .. math::
        \hat f_{\text{single}}(\alpha) - f(\alpha)
        \;=\; \underbrace{\frac{\ln\mathrm{d}\alpha}{\ln\delta}}_{\text{bin width}}
        \;+\; \underbrace{\frac{\ln\sqrt{l}}{l\ln b}}_{\text{prefactor}}
        \;+\; O(l^{-1}),

    two terms of *opposite sign*. Removing only the first can therefore make a
    single-scale estimate worse --- observed at :math:`k_{\max}=18`, 8 bins:
    mean absolute error 0.064 uncorrected, 0.183 with the bin-width correction
    alone, against 0.060 for the slope form used here. In the slope both are
    intercepts and vanish; what survives is the derivative
    :math:`\mathrm{d}(-\tfrac12\ln l)/\mathrm{d}l = -1/(2l)`, an
    :math:`O(1/l)` bias with coefficient :math:`1/2` where the single-scale bin
    term has coefficient :math:`|\ln\mathrm{d}\alpha|/\ln b\approx 3.8` at these
    settings --- roughly eight times smaller.

    Parameters
    ----------
    mu
        Cell masses; normalised internally.
    scales, b, min_boxes
        Box scales as in :func:`box_partition`. ``min_boxes`` defaults to 8 here:
        a histogram over four boxes is not a histogram.
    bins
        Number of equal-width :math:`\alpha` bins, **shared by every scale**.
        There is a hard constraint the caller must respect: at level :math:`l` of
        a :math:`b`-adic cascade the coarse exponents live on a lattice of
        spacing :math:`(\alpha_{\max}-\alpha_{\min})/l`, so a bin narrower than
        that lattice is empty at that scale and the fit aliases. Take
        :math:`\text{bins} \lesssim l_{\min}`, the coarsest level in ``scales``.
    alpha_range
        Bin support; default the observed :math:`[\min\alpha,\max\alpha]` over
        all scales.
    min_scales
        A bin must be occupied at this many scales to be fitted; bins failing it
        are dropped and counted in ``provenance.diagnostics``.
    alpha_locator
        Where to place the bin on the :math:`\alpha` axis.

        ``"weighted"`` (default)
            The count-weighted mean of the exponents actually observed in the
            bin. Because the occupancy of a bin is dominated by its edge nearest
            the apex of :math:`f`, the *growth rate* measured for the bin is
            :math:`f` at that edge, not at the centre; reporting the weighted
            mean places the ordinate and the abscissa at the same point and
            cancels the leading :math:`O(\mathrm{d}\alpha)` bin-width bias.
        ``"centre"``
            The bin midpoint. Textbook, and biased by
            :math:`\tfrac12 |f'(\alpha)|\,\mathrm{d}\alpha`, which does **not**
            vanish as :math:`\delta\to0` at fixed ``bins``.
    fit_method, d_support, seed
        As in :func:`generalised_dimensions`.

    Returns
    -------
    HolderSpectrum
        ``alpha`` and ``f_alpha`` over the surviving bins, with ``f_stderr`` from
        the regression. ``q`` is ``None``: the direct method never forms a
        moment, which is precisely its advantage --- it can return a non-concave
        or truncated :math:`f`, which no Legendre transform can.

    Notes
    -----
    **Convergence rate, honestly.** Three error terms, in decreasing order of
    nastiness.

    1. *Statistical.* The occupancy of a bin is a count, so its relative error is
       :math:`N_\delta(\alpha)^{-1/2}` and

       .. math::
           \operatorname{sd}\hat f(\alpha) \;\approx\;
           \frac{1}{\sqrt{N_\delta(\alpha)}\;\ln(1/\delta)} .

       At the apex :math:`N_\delta\sim\delta^{-d}` is the total box count, giving
       the advertised :math:`1/\sqrt{\#\text{boxes}}`; in the tails
       :math:`N_\delta=O(1)` at every :math:`\delta` and the error decays only
       like :math:`1/\ln(1/\delta)`, i.e. like :math:`1/k` in cascade levels.
       **The tails of a histogram spectrum do not converge at any useful rate.**
       This is the defect the Chhabra--Jensen canonical method was invented to
       remove, and it is what the predecessor's ``Test3`` was in fact measuring.

    2. *Lattice.* See ``bins``. Aliasing produces empty bins at coarse scales and
       a slope fitted through a comb.

    3. *Bin width.* :math:`O(\mathrm{d}\alpha)` with ``alpha_locator="centre"``,
       suppressed to :math:`O(\mathrm{d}\alpha^2)` by the weighted locator.
       It does not vanish with :math:`\delta`: to converge, ``bins`` must grow
       with the depth of the measure, slower than the lattice
       (:math:`\mathrm{d}\alpha\sim k^{-1/2}` is the usual compromise).

    References
    ----------
    Meneveau and Sreenivasan (1991), §3; Evertsz and Mandelbrot (1992),
    Appendix B; Chhabra and Jensen (1989) for the canonical alternative.
    """
    m = validate_measure(mu, name="measure", normalise=True)
    n = m.size
    s = _resolve_scales(scales, n, b, int(min_boxes))
    n_bins = int(bins)
    if n_bins < 3:
        raise ValueError(f"need at least three alpha bins, got {n_bins}")
    if int(min_scales) < 3:
        raise ValueError(f"min_scales={min_scales} < 3: a slope with error bars needs three points")
    if alpha_locator not in ("weighted", "centre"):
        raise ValueError(f"unknown alpha_locator {alpha_locator!r}")

    alphas: list[FloatArray] = []
    current = m
    current_s = 1
    for si in s:
        step = int(si)
        current = (
            _aggregate(current, step // current_s) if step % current_s == 0 else _aggregate(m, step)
        )
        current_s = step
        occupied = current > 0.0
        alphas.append(np.log(current[occupied]) / math.log(step / n))

    if alpha_range is None:
        lo = min(float(a.min()) for a in alphas)
        hi = max(float(a.max()) for a in alphas)
    else:
        lo, hi = float(alpha_range[0]), float(alpha_range[1])
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        raise ValueError(f"degenerate alpha range [{lo}, {hi}]; the measure may be uniform")
    edges = np.linspace(lo, hi, n_bins + 1)
    d_alpha = float(edges[1] - edges[0])

    counts = np.zeros((n_bins, s.size), dtype=np.float64)
    alpha_sums = np.zeros((n_bins, s.size), dtype=np.float64)
    for j, a in enumerate(alphas):
        counts[:, j] = np.histogram(a, bins=edges)[0]
        alpha_sums[:, j] = np.histogram(a, bins=edges, weights=a)[0]

    delta = np.asarray(s, dtype=np.float64) / float(n)
    centres = 0.5 * (edges[:-1] + edges[1:])
    alpha_out: list[float] = []
    f_out: list[float] = []
    f_se: list[float] = []
    n_dropped = 0
    for i in range(n_bins):
        usable = counts[i] > 0.0
        if int(usable.sum()) < int(min_scales):
            n_dropped += 1
            continue
        c = counts[i][usable]
        fit = loglog_fit(
            delta[usable],
            np.log(c / d_alpha),
            method=fit_method,
            weights=c,
            log_y=False,
        )
        total = float(c.sum())
        alpha_out.append(
            float(alpha_sums[i][usable].sum() / total)
            if alpha_locator == "weighted"
            else float(centres[i])
        )
        f_out.append(-fit.slope)
        f_se.append(fit.slope_stderr)

    if len(f_out) < 3:
        raise ValueError(
            f"only {len(f_out)} alpha bins survived (min_scales={min_scales}); reduce "
            "`bins` so they are wider than the coarse-Hoelder lattice, or use more scales"
        )

    prov = new_provenance(
        f"{__name__}.histogram_spectrum",
        estimator_version=MODULE_VERSION,
        params={
            "bins": n_bins,
            "b": int(b),
            "min_boxes": int(min_boxes),
            "min_scales": int(min_scales),
            "alpha_locator": str(alpha_locator),
            "fit_method": str(fit_method),
            "alpha_range": [lo, hi],
            "d_support": float(d_support),
            "n_cells": int(n),
        },
        inputs=(InputRef.from_array("mu", m), InputRef.from_array("scales", s)),
        seed=None if seed is None else seed_info(seed),
        notes=(
            "direct histogram method: f = -slope of ln(count/bin_width) vs ln(delta)",
            "the bin width cancels from the slope; it does not from a single scale",
        ),
    ).with_diagnostics(
        bin_width=d_alpha,
        bins_dropped=n_dropped,
        bins_kept=len(f_out),
        boxes_per_scale=[int(a.size) for a in alphas],
    )
    return HolderSpectrum(
        alpha=np.asarray(alpha_out, dtype=np.float64),
        f_alpha=np.asarray(f_out, dtype=np.float64),
        q=None,
        f_stderr=np.asarray(f_se, dtype=np.float64),
        d_support=float(d_support),
        method=f"box-counting-histogram({alpha_locator})",
        provenance=prov,
    )


# --------------------------------------------------------------------------- #
# The whole family, tied together
# --------------------------------------------------------------------------- #


def multifractal_measure_spectrum(
    mu: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    b: int = 2,
    method: SpectrumMethod = "legendre",
    bins: int = 21,
    min_boxes: int = 4,
    n_alpha: int = 201,
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
    d_support: float = 1.0,
    seed: SeedLike = None,
) -> HolderSpectrum:
    r"""Singularity spectrum of a measure by the box-counting family.

    ``method="legendre"`` (default)
        Fit :math:`\tau` with :func:`generalised_dimensions`, then take the
        Legendre--Fenchel transform :math:`f(\alpha)=\inf_q\{q\alpha-\tau(q)\}`.
        Low variance, and the standard choice --- but the transform of *any*
        function is concave, so this route cannot report the non-concavity that
        would reveal a failure of the scaling assumption, and the recoverable
        :math:`\alpha` range is bounded by :math:`[\tau'(q_{\max}),
        \tau'(q_{\min})]`: widening ``q`` widens the spectrum.

    ``method="histogram"``
        :func:`histogram_spectrum`. Higher variance, far slower convergence in
        the tails, and the only one of the two that can see a truncated or
        non-concave spectrum.

    The two are different estimators of the same object and are expected to
    disagree in the tails; the size of the disagreement is a useful diagnostic,
    not a bug.

    Parameters
    ----------
    n_alpha
        Resolution of the :math:`\alpha` grid for the Legendre route.
    q, scales, b, bins, min_boxes, fit_method, select_range, d_support, seed
        Passed through; see the two underlying estimators.

    Returns
    -------
    HolderSpectrum
        Measure-view spectrum: :math:`\max_\alpha f = d` for a full-support
        measure.
    """
    if method == "histogram":
        return histogram_spectrum(
            mu,
            scales=scales,
            bins=bins,
            b=b,
            min_boxes=max(int(min_boxes), 8),
            fit_method=fit_method,
            d_support=d_support,
            seed=seed,
        )
    if method != "legendre":
        raise ValueError(f"unknown method {method!r}; use 'legendre' or 'histogram'")
    exponents = generalised_dimensions(
        mu,
        q=q,
        scales=scales,
        b=b,
        min_boxes=min_boxes,
        fit_method=fit_method,
        select_range=select_range,
        d_support=d_support,
        seed=seed,
    )
    return spectrum_from_exponents(exponents, method="fenchel", n_alpha=n_alpha)
