r"""Wavelet leaders: the multifractal formalism as Jaffard states it.

Increment- and coefficient-based structure functions are heuristics. The
*wavelet-leader* formalism (Jaffard 2004) is a theorem: the Legendre transform of
the leader scaling function is an upper bound for the singularity spectrum, with
equality for the self-similar processes the multifractal literature actually
models, and --- crucially --- the formalism is valid for :math:`q<0`, which no
coefficient- or increment-based estimator is.

Construction
------------
Let :math:`\lambda(j,k)=[k2^j,(k+1)2^j)` be a dyadic interval, and
:math:`3\lambda(j,k)=\lambda(j,k-1)\cup\lambda(j,k)\cup\lambda(j,k+1)` the same
interval flanked by its two neighbours. With :math:`d(\lambda')` the
:math:`L^1`-normalised discrete wavelet coefficient of the interval
:math:`\lambda'`, the *leader* is

.. math::
    L(j,k) \;=\; \sup_{\lambda' \subset 3\lambda(j,k)} |d(\lambda')| ,

the supremum running over **all finer scales** :math:`j'\le j` contained in the
union. Its structure function

.. math::
    S_L(q,j) \;=\; \frac{1}{n_j}\sum_k L(j,k)^q \;\sim\; 2^{\,j\zeta(q)},
    \qquad 2^j = \delta \to 0,

produces :math:`\zeta` *natively in the signal convention* of
:mod:`mmar.core.types` --- :math:`S_L` is an average over :math:`n_j` leaders,
exactly the :math:`S(q,\delta)=\langle|\Delta_\delta X|^q\rangle` of that module,
so no conversion is needed and :math:`\tau(q)=\zeta(q)-d` is derived by
:class:`~mmar.core.types.ScalingExponents` as usual.

Why the supremum matters
------------------------
Two properties, both false for raw coefficients:

1. **Local regularity.** If :math:`X` has pointwise Hölder exponent :math:`h(t_0)`
   at :math:`t_0` and a uniform exponent :math:`h_{\min}>0`, then
   :math:`L(j,k_j(t_0)) \sim 2^{\,j\,h(t_0)}` --- an equivalence, not a bound.
   A single wavelet coefficient only gives :math:`|d|\lesssim 2^{jh}`: it may be
   arbitrarily small by accidental cancellation against the wavelet's
   oscillation, so :math:`|d|` does *not* characterise :math:`h`.
2. **Negative moments exist.** :math:`L(j,k)\ge|d(j,k)|` and is a supremum over
   :math:`\mathcal O(2^{j})` coefficients, so it is bounded away from zero
   whenever the signal is non-constant on :math:`3\lambda`. The moment
   :math:`\langle L^q\rangle` therefore converges for :math:`q<0`, whereas
   :math:`\langle|d|^q\rangle` is dominated by whichever coefficient happens to
   be closest to zero and diverges as the sample grows. This is the technical
   content of "the right branch of :math:`f(\alpha)` needs :math:`q<0`" and the
   direct fix for legacy defect D2.

Log-cumulants
-------------
Because :math:`S_L(q,j)=\langle e^{q\ln L}\rangle`, expanding the cumulant
generating function of :math:`\ln L(j,\cdot)` gives

.. math::
    \ln S_L(q,j) \;=\; \sum_{p\ge1} \frac{C_p(j)}{p!}\,q^p ,
    \qquad C_p(j) = \mathrm{cum}_p\big[\ln L(j,\cdot)\big],

and if :math:`C_p(j) = c_p^{0} + c_p\, j\ln 2` then

.. math::
    \zeta(q) \;=\; \sum_{p\ge1} \frac{c_p}{p!}\,q^p .

So :math:`c_p` is simply the slope of :math:`C_p(j)` against :math:`j\ln 2`.
:math:`c_1` is the most probable Hölder exponent :math:`\alpha_0`,
:math:`c_2\le 0` is the multifractality strength (**exactly zero for a
monofractal**), and :math:`c_3` is the spectrum's asymmetry. This is far better
conditioned than a Legendre transform: it estimates the curvature of
:math:`\zeta` from a *variance*, at :math:`q=0`, instead of differencing slopes
fitted at large :math:`|q|` where the moments are least reliable. It is the
headline diagnostic of this module.

Known biases, and their causes
------------------------------
*Vanishing moments.* A wavelet with :math:`N_\psi` vanishing moments annihilates
polynomials of degree :math:`<N_\psi`, hence
:math:`|d(j,k)| \sim 2^{\,j\min(h,\,N_\psi)}`: **exponents saturate at**
:math:`N_\psi`. If the estimated :math:`\alpha_{\max}` approaches
:math:`N_\psi`, the spectrum is truncated by the analysing wavelet and not by the
data. This module never silently tolerates that: it records the margin
:math:`N_\psi-\hat\alpha_{\max}` in the provenance diagnostics and flags the
result when the margin falls below ``saturation_tolerance``.

*Uniform regularity.* Jaffard's theorem requires :math:`h_{\min}>0`. For a
signal with :math:`h_{\min}\le0` (a measure density, white noise, a series with
jumps) the leaders do not converge and the estimate is meaningless; the
diagnostic ``alpha_min_estimated`` is recorded so this is checkable. Integrating
the signal once (and subtracting 1 from :math:`\zeta`) is the standard remedy.
On white noise this implementation returns
:math:`\hat\alpha_{\min}\approx0`, not the true :math:`-1/2`: the supremum
over the :math:`\mathcal O(2^j)` fine coefficients in :math:`3\lambda` cancels
the :math:`2^{-j/2}` decay against the growth of a Gaussian maximum, leaving a
:math:`\sqrt{j}` factor and a flat :math:`\zeta`. A near-zero
:math:`\hat\alpha_{\min}` is therefore itself the diagnostic that the
hypothesis has failed.

*Finite-size bias at extreme* :math:`|q|`. :math:`S_L(q,j)` at large positive
:math:`q` is carried by the largest few leaders and at large negative :math:`q`
by the smallest few. With :math:`n_j = n/2^j` leaders the sample extremes are
systematically interior to the population extremes, so :math:`|\hat\zeta(q)|` is
biased *downwards* at both ends, i.e. the recovered :math:`f(\alpha)` is too
narrow in the tails. The bias shrinks like :math:`1/\log n`, not like
:math:`1/\sqrt n`. Measured values for this implementation are in
``tests/unit/test_wavelet_leaders.py``.

*Coarse-octave leverage.* The number of leaders halves with every octave, so
count-weighted least squares (the default, and the statistically correct
inverse-variance choice) puts almost all weight on the two or three finest
octaves. Those are also the octaves where the discrete wavelet transform has not
yet reached its asymptotic regime, which is why ``j_min`` defaults to 3 rather
than 1 and why ``select_range=True`` exists.

References
----------
Jaffard, S. (2004). Wavelet techniques in multifractal analysis. In
*Fractal Geometry and Applications*, Proc. Symp. Pure Math. **72**, 91--151.

Lashermes, B., Roux, S.G., Abry, P., Jaffard, S. (2008). Comprehensive
multifractal analysis of turbulent velocity using the wavelet leaders.
*Eur. Phys. J. B* **61**, 201--215.

Wendt, H., Abry, P., Jaffard, S. (2007). Bootstrap for empirical multifractal
analysis. *IEEE Signal Processing Magazine* **24**(4), 38--48.

Wendt, H., Roux, S.G., Jaffard, S., Abry, P. (2009). Wavelet leaders and
bootstrap for multifractal analysis of images. *Signal Processing* **89**,
1100--1114.

Daubechies, I. (1992). *Ten Lectures on Wavelets*. SIAM.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from mmar.core.provenance import InputRef, Provenance, SeedInfo, new_provenance
from mmar.core.regression import loglog_fit, select_scaling_range, weighted_by_counts
from mmar.core.rng import SeedLike, seed_info
from mmar.core.types import (
    FitMethod,
    FloatArray,
    LogCumulants,
    LogLogFit,
    ScalingExponents,
    ScalingRange,
    StructureFunctions,
    dyadic_scales,
)
from mmar.core.validate import check_scales, validate_q_grid, validate_series

__all__ = [
    "LeaderPyramid",
    "exponents_from_structure",
    "leader_cumulants",
    "leader_pyramid",
    "wavelet_coefficients_exponents",
    "wavelet_coefficients_structure",
    "wavelet_leaders",
    "wavelet_leaders_structure",
]


def _seed_record(seed: SeedLike) -> SeedInfo | None:
    """Provenance seed field, ``None`` when no seed was supplied.

    These estimators are deterministic, so ``seed`` is accepted only for
    signature uniformity. Passing it straight to
    :func:`~mmar.core.rng.seed_info` would be a bug: ``seed_info(None)`` draws
    *fresh OS entropy*, and the resulting ``SeedInfo`` would change the
    provenance root on every call, making two identical computations
    unrecognisable as identical.
    """
    return None if seed is None else seed_info(seed)


ESTIMATOR_LEADERS = "wavelet_leaders"
ESTIMATOR_COEFFS = "wavelet_coefficients"
ESTIMATOR_CUMULANTS = "wavelet_leaders.cumulants"
_VERSION = "1"

_TILT_BUDGET = 1 << 17
r"""Cells above which :func:`_log_structure` reduces one :math:`q` at a time.

:math:`10^5` float64 cells is under a megabyte, so below the budget the
broadcast form costs nothing; above it the temporary grows like
:math:`n_q N/8` and dominates the whole estimator's footprint.
"""


# --------------------------------------------------------------------------- #
# Wavelet transform and the leader pyramid
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LeaderPyramid:
    r"""The multiresolution quantity an estimator is fitted to, plus its audit trail.

    ``values[i]`` holds the surviving quantities at octave ``octaves[i]``: the
    leaders :math:`L(j,\cdot)` when built by :func:`leader_pyramid`, or the bare
    :math:`L^1`-normalised coefficients :math:`|d(j,\cdot)|` when
    ``leaders=False``. Boundary-contaminated entries have already been removed
    and counted.
    """

    octaves: NDArray[np.int64]
    values: tuple[FloatArray, ...]
    n_border_dropped: NDArray[np.int64]
    """Coefficients discarded at each octave (both ends combined)."""
    n_zero_dropped: NDArray[np.int64]
    """Exactly-zero entries dropped before taking logarithms, per octave."""
    wavelet: str
    n_vanishing_moments: int
    boundary_margin: int
    leaders: bool

    @property
    def scales(self) -> FloatArray:
        return np.asarray(2.0 ** self.octaves.astype(np.float64), dtype=np.float64)

    @property
    def counts(self) -> NDArray[np.int64]:
        return np.asarray([v.size for v in self.values], dtype=np.int64)


def _resolve_wavelet(wavelet: str) -> tuple[Any, int, int]:
    """Return ``(pywt.Wavelet, N_psi, filter_length)``, importing PyWavelets lazily.

    PyWavelets is an optional dependency (``mmar[wavelets]``); the import error is
    re-raised with the install instruction rather than propagating a bare
    ``ModuleNotFoundError`` from deep inside an estimator.
    """
    try:
        import pywt  # noqa: PLC0415 - optional extra; a top-level import would make
        # `import mmar.estimators` fail for anyone who installed without `mmar[wavelets]`.
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the wavelet-leader estimators require PyWavelets; install with "
            "`pip install 'mmar[wavelets]'`"
        ) from exc
    try:
        w = pywt.Wavelet(wavelet)
    except ValueError as exc:
        raise ValueError(f"unknown wavelet {wavelet!r}: {exc}") from exc
    n_psi = int(w.vanishing_moments_psi)
    if n_psi < 1:
        raise ValueError(
            f"wavelet {wavelet!r} has {n_psi} vanishing moments; at least one is required "
            "or the coefficients do not decay with the local regularity at all"
        )
    return w, n_psi, int(w.dec_len)


def _l1_details(x: FloatArray, wavelet: str, j_max: int) -> list[FloatArray]:
    r""":math:`L^1`-normalised detail coefficients, finest octave first.

    PyWavelets returns the **orthonormal** (:math:`L^2`) coefficients
    :math:`d^{(2)}(j,k)=\langle X,\psi_{j,k}\rangle` with
    :math:`\psi_{j,k}(t)=2^{-j/2}\psi(2^{-j}t-k)`. For a signal with local
    exponent :math:`h` a change of variable gives
    :math:`d^{(2)} \sim 2^{\,j(h+1/2)}`, so the exponent read off an
    :math:`L^2`-normalised coefficient is :math:`h+\tfrac12` --- a silent
    half-unit offset in every reported Hölder exponent. The multifractal
    formalism is stated for the :math:`L^1` normalisation

    .. math::
        d(j,k) \;=\; 2^{-j/2}\,d^{(2)}(j,k) \;\sim\; 2^{\,jh},

    which is what this function returns.

    ``mode="periodization"`` is used so that :math:`n_j` is exactly
    :math:`\lceil n_{j-1}/2\rceil`; any other extension mode inflates the
    coefficient count by the filter length at every octave and destroys the exact
    parent--child index map :math:`k\mapsto(2k,2k+1)` that the leader recursion
    depends on. The wrap-around contamination it introduces is confined to the
    first and last :math:`N_\psi` coefficients of each octave and is removed by
    :func:`leader_pyramid`.
    """
    import pywt  # noqa: PLC0415 - optional extra, see _resolve_wavelet

    coeffs = pywt.wavedec(x, wavelet, mode="periodization", level=j_max)
    # coeffs = [cA_J, cD_J, ..., cD_1]; reverse the detail part to get cD_1 first.
    details = coeffs[:0:-1]
    return [
        np.asarray(np.abs(np.asarray(d, dtype=np.float64)) * 2.0 ** (-(j + 1) / 2.0))
        for j, d in enumerate(details)
    ]


def _max_octave(n: int, filter_len: int, margin: int, min_leaders: int) -> int:
    """Coarsest octave with at least ``min_leaders`` uncontaminated leaders.

    Two independent ceilings apply. PyWavelets refuses (with a warning) to
    decompose past ``dwt_max_level``, where every coefficient overlaps the
    boundary. Independently, octave :math:`j` retains
    :math:`n/2^j - 2\\,\\text{margin}` leaders, which must not fall below
    ``min_leaders``: a moment estimated from three numbers is not an estimate.
    """
    import pywt  # noqa: PLC0415 - optional extra, see _resolve_wavelet

    hard = int(pywt.dwt_max_level(n, filter_len))
    j = 0
    while j + 1 <= hard and (n >> (j + 1)) - 2 * margin >= min_leaders:
        j += 1
    return j


def leader_pyramid(
    x: Sequence[float] | NDArray[Any],
    *,
    wavelet: str = "db3",
    j_min: int = 3,
    j_max: int | None = None,
    boundary_margin: int | None = None,
    min_leaders: int = 8,
    leaders: bool = True,
) -> LeaderPyramid:
    r"""Build the wavelet-leader (or bare-coefficient) multiresolution pyramid.

    The recursion is the standard one. Writing :math:`L^{\max}(j,k)` for the
    supremum restricted to :math:`\lambda(j,k)` alone,

    .. math::
        L^{\max}(1,k) &= |d(1,k)|, \\
        L^{\max}(j,k) &= \max\big\{|d(j,k)|,\;L^{\max}(j-1,2k),\;L^{\max}(j-1,2k+1)\big\},

    since :math:`\lambda(j,k)` is the disjoint union of its two children. The
    leader then extends the supremum sideways over the three-interval
    neighbourhood,

    .. math::
        L(j,k) \;=\; \max\big\{L^{\max}(j,k-1),\,L^{\max}(j,k),\,L^{\max}(j,k+1)\big\},

    which is what makes :math:`L(j,k)\sim2^{jh}` an equivalence rather than an
    upper bound: the flanking intervals guarantee the singularity is enclosed
    even when it sits at an interval boundary.

    Boundary handling
    -----------------
    The equivalent filter of :math:`\psi_{j,k}` spans roughly
    :math:`[(k-N_\psi+1)2^j,\,(k+N_\psi)2^j]` in samples, so a coefficient is
    contaminated by the periodic wrap exactly when :math:`k<N_\psi-1` or
    :math:`k>n_j-N_\psi`. That is a *constant index margin per octave*, which is
    the fortunate part: the same margin at octave :math:`j` also covers every
    finer coefficient the leader inherits, because a fixed index margin at octave
    :math:`j'` occupies the same time interval as at octave :math:`j`. The
    default margin is :math:`N_\psi+1` --- :math:`N_\psi` for the wrap and one
    more for the three-interval neighbourhood --- and the number discarded is
    recorded per octave.

    Parameters
    ----------
    j_min, j_max
        Octave window. ``j_max=None`` takes the coarsest octave leaving
        ``min_leaders`` uncontaminated leaders. ``j_min`` defaults to 3: the two
        finest octaves of a discrete wavelet transform are dominated by the
        discretisation transient (the sampled signal is not the continuous one
        below a few samples per oscillation) and, because leaders propagate
        suprema upward, that transient contaminates *every* coarser octave if it
        is included.
    leaders
        ``False`` returns the bare :math:`|d(j,k)|` instead, for the
        coefficient-based comparison estimator. See
        :func:`wavelet_coefficients_exponents` for why that variant is invalid
        at :math:`q<0`.

    Raises
    ------
    ValueError
        If no octave in ``[j_min, j_max]`` survives, which means the series is
        too short for the requested wavelet.
    """
    xa = validate_series(x, name="signal", min_length=64)
    _w, n_psi, filter_len = _resolve_wavelet(wavelet)
    margin = int(n_psi + 1 if boundary_margin is None else boundary_margin)
    if margin < 0:
        raise ValueError(f"boundary_margin must be non-negative, got {boundary_margin!r}")
    if min_leaders < 3:
        raise ValueError(
            f"min_leaders={min_leaders} is below 3; the third log-cumulant is undefined "
            "and no moment estimated from two numbers is meaningful"
        )
    if j_min < 1:
        raise ValueError(f"j_min must be at least 1, got {j_min}")

    top = _max_octave(xa.size, filter_len, margin, min_leaders)
    if j_max is not None:
        top = min(top, int(j_max))
    if top < j_min:
        raise ValueError(
            f"no admissible octave: n={xa.size} with wavelet {wavelet!r} (N_psi={n_psi}, "
            f"margin={margin}, min_leaders={min_leaders}) supports octaves up to j={top}, "
            f"below j_min={j_min}. Use a shorter wavelet, lower j_min, or a longer series."
        )

    # The admissible grid is geometric by construction (scales are 2^j), and is
    # built through dyadic_scales + check_scales so that the >= 4-windows floor is
    # applied by the same code path as everywhere else in the library.
    admissible = set(
        np.round(
            np.log2(
                check_scales(
                    dyadic_scales(xa.size, j_min=j_min, j_max=top), xa.size, min_windows=4
                ).astype(np.float64)
            )
        )
        .astype(np.int64)
        .tolist()
    )

    details = _l1_details(xa, wavelet, top)

    octaves: list[int] = []
    kept: list[FloatArray] = []
    dropped: list[int] = []
    running_max: FloatArray | None = None
    for j in range(1, top + 1):
        a = details[j - 1]
        cur = a.copy()
        if leaders and running_max is not None:
            # Children of interval k at octave j are 2k and 2k+1 at octave j-1.
            m = min(a.size, running_max.size // 2)
            np.maximum(
                cur[:m],
                np.maximum(running_max[0 : 2 * m : 2], running_max[1 : 2 * m : 2]),
                out=cur[:m],
            )
        running_max = cur
        if leaders:
            lam = cur.copy()
            np.maximum(lam[:-1], cur[1:], out=lam[:-1])
            np.maximum(lam[1:], cur[:-1], out=lam[1:])
        else:
            lam = cur
        if j not in admissible:
            continue
        if lam.size <= 2 * margin:
            continue
        core = np.ascontiguousarray(lam[margin : lam.size - margin])
        if core.size < min_leaders:
            continue
        octaves.append(j)
        kept.append(core)
        dropped.append(int(lam.size - core.size))

    if len(octaves) < 3:
        raise ValueError(
            f"only {len(octaves)} octave(s) survived in [{j_min}, {top}] for n={xa.size} with "
            f"wavelet {wavelet!r} (N_psi={n_psi}, margin={margin}, min_leaders={min_leaders}); "
            "at least three are needed for a slope with error bars. Lower j_min, lower "
            "min_leaders, use a shorter wavelet, or take a longer series."
        )

    zeros = [int(np.count_nonzero(v == 0.0)) for v in kept]
    nonzero = tuple(np.ascontiguousarray(v[v > 0.0]) for v in kept)
    for j, v in zip(octaves, nonzero, strict=True):
        if v.size < min_leaders:
            raise ValueError(
                f"octave j={j} has only {v.size} strictly positive "
                f"{'leaders' if leaders else 'coefficients'} after dropping exact zeros; "
                "the signal is locally constant over most of its support"
            )

    return LeaderPyramid(
        octaves=np.asarray(octaves, dtype=np.int64),
        values=nonzero,
        n_border_dropped=np.asarray(dropped, dtype=np.int64),
        n_zero_dropped=np.asarray(zeros, dtype=np.int64),
        wavelet=wavelet,
        n_vanishing_moments=n_psi,
        boundary_margin=margin,
        leaders=leaders,
    )


# --------------------------------------------------------------------------- #
# Scale-grid plumbing
# --------------------------------------------------------------------------- #


def _octaves_from_scales(
    scales: Sequence[int] | NDArray[Any] | None, n: int, j_min: int
) -> NDArray[np.int64] | None:
    """Translate a requested scale grid into octave indices, or ``None`` for 'all'."""
    if scales is None:
        return None
    s = check_scales(scales, n, min_windows=4, name="scales")
    j = np.log2(s.astype(np.float64))
    if not np.allclose(j, np.round(j)):
        raise ValueError(
            f"wavelet-leader scales must be exact powers of two (the dyadic grid the "
            f"transform is defined on); got {s.tolist()}. Use "
            f"mmar.core.types.dyadic_scales(n, j_min={j_min})."
        )
    return np.asarray(np.round(j).astype(np.int64), dtype=np.int64)


def _default_scales(pyramid: LeaderPyramid) -> NDArray[np.int64]:
    return np.asarray(2**pyramid.octaves, dtype=np.int64)


def _select_pyramid(pyramid: LeaderPyramid, octaves: NDArray[np.int64] | None) -> LeaderPyramid:
    if octaves is None:
        return pyramid
    keep = np.isin(pyramid.octaves, octaves)
    if int(keep.sum()) < 3:
        raise ValueError(
            f"only {int(keep.sum())} of the requested octaves {octaves.tolist()} are "
            f"available (the pyramid holds {pyramid.octaves.tolist()}); at least three "
            "are needed for a slope with error bars"
        )
    idx = np.nonzero(keep)[0]
    return LeaderPyramid(
        octaves=pyramid.octaves[idx],
        values=tuple(pyramid.values[i] for i in idx),
        n_border_dropped=pyramid.n_border_dropped[idx],
        n_zero_dropped=pyramid.n_zero_dropped[idx],
        wavelet=pyramid.wavelet,
        n_vanishing_moments=pyramid.n_vanishing_moments,
        boundary_margin=pyramid.boundary_margin,
        leaders=pyramid.leaders,
    )


# --------------------------------------------------------------------------- #
# Structure functions
# --------------------------------------------------------------------------- #


def _log_structure(pyramid: LeaderPyramid, q: FloatArray) -> FloatArray:
    r"""``log S_L(q, j)`` on the natural logarithm scale, shape ``(n_q, n_scales)``.

    Formed as :math:`\operatorname{logsumexp}_k(q\ln L) - \ln n_j` rather than
    ``(L**q).mean()``. The direct form overflows: on the trading time of a
    depth-20 cascade the leaders are :math:`\mathcal O(10^{-6})` and
    :math:`L^{-5}` is :math:`\mathcal O(10^{30})` per term before summation, and
    for :math:`q` beyond :math:`\pm 8` it overflows float64 outright. The
    log-domain form is exact to machine precision throughout, and it is the
    quantity the regression consumes anyway.

    Above :data:`_TILT_BUDGET` cells the reduction runs one :math:`q` at a time.
    The broadcast :math:`(n_q\times n_j)` tilt matrix is bit-for-bit identical
    either way, but materialising it costs: :math:`n_{j_{\min}}\approx N/8`, so
    with the default 41-point grid the peak allocation was 33 times the series
    (17.2 MB at :math:`N=2^{16}`, and 1.1 GB at :math:`N=2^{22}`). The coarse
    octaves stay vectorised, where the matrix is at most a megabyte and a Python
    loop over :math:`q` would be pure call overhead.
    """
    out = np.empty((q.size, pyramid.octaves.size), dtype=np.float64)
    for i, v in enumerate(pyramid.values):
        ln = np.log(v)
        log_n = math.log(v.size)
        if q.size * ln.size > _TILT_BUDGET:
            for k, qk in enumerate(q):
                out[k, i] = float(logsumexp(float(qk) * ln)) - log_n
        else:
            out[:, i] = np.asarray(logsumexp(q[:, None] * ln[None, :], axis=1)) - log_n
    return out


def _params(
    *,
    q: FloatArray,
    scales: NDArray[np.int64],
    wavelet: str,
    j_min: int,
    j_max: int | None,
    boundary_margin: int,
    min_leaders: int,
    fit_method: FitMethod,
    select_range: bool,
    d_support: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    p: dict[str, Any] = {
        "q": q,
        "scales": scales,
        "wavelet": wavelet,
        "j_min": int(j_min),
        "j_max": None if j_max is None else int(j_max),
        "boundary_margin": int(boundary_margin),
        "min_leaders": int(min_leaders),
        "fit_method": str(fit_method),
        "select_range": bool(select_range),
        "d_support": float(d_support),
    }
    if extra:
        p.update(extra)
    return p


def _pyramid_diagnostics(pyramid: LeaderPyramid) -> dict[str, Any]:
    return {
        "octaves": pyramid.octaves.tolist(),
        "n_leaders_per_octave": pyramid.counts.tolist(),
        "n_border_dropped_per_octave": pyramid.n_border_dropped.tolist(),
        "n_border_dropped_total": int(pyramid.n_border_dropped.sum()),
        "n_zero_dropped_per_octave": pyramid.n_zero_dropped.tolist(),
        "n_zero_dropped_total": int(pyramid.n_zero_dropped.sum()),
        "n_vanishing_moments": pyramid.n_vanishing_moments,
        "boundary_margin": pyramid.boundary_margin,
        "wavelet": pyramid.wavelet,
    }


def _structure_from_pyramid(
    pyramid: LeaderPyramid, q: FloatArray, prov: Provenance
) -> tuple[StructureFunctions, FloatArray]:
    log_s = _log_structure(pyramid, q)
    # exp() can overflow for extreme |q| on a very singular signal; the fits use
    # log_s and never the exponentiated surface, so the overflow is recorded and
    # the inf is kept rather than silently rescaled.
    with np.errstate(over="ignore"):
        values = np.asarray(np.exp(log_s), dtype=np.float64)
    n_overflow = int(np.count_nonzero(~np.isfinite(values)))
    structure = StructureFunctions(
        q=q,
        scales=pyramid.scales,
        values=values,
        log_base=math.e,
        counts=pyramid.counts,
        provenance=prov.with_diagnostics(
            **_pyramid_diagnostics(pyramid), n_overflow_cells=n_overflow
        ),
    )
    return structure, log_s


def wavelet_leaders_structure(
    x: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    wavelet: str = "db3",
    j_min: int = 3,
    j_max: int | None = None,
    boundary_margin: int | None = None,
    min_leaders: int = 8,
    seed: SeedLike = None,
) -> StructureFunctions:
    r"""The raw leader scaling surface :math:`S_L(q,2^j)`, with no exponent fitted.

    Kept separate so that :math:`\zeta` can be refitted over a different scaling
    range --- via :func:`exponents_from_structure` --- without recomputing the
    wavelet transform, which is the expensive part.
    """
    xa = validate_series(x, name="signal", min_length=64)
    qa = validate_q_grid(q)
    pyramid = leader_pyramid(
        xa,
        wavelet=wavelet,
        j_min=j_min,
        j_max=j_max,
        boundary_margin=boundary_margin,
        min_leaders=min_leaders,
        leaders=True,
    )
    pyramid = _select_pyramid(pyramid, _octaves_from_scales(scales, xa.size, j_min))
    prov = new_provenance(
        f"{ESTIMATOR_LEADERS}.structure",
        estimator_version=_VERSION,
        params=_params(
            q=qa,
            scales=_default_scales(pyramid),
            wavelet=wavelet,
            j_min=j_min,
            j_max=j_max,
            boundary_margin=pyramid.boundary_margin,
            min_leaders=min_leaders,
            fit_method=FitMethod.WLS,
            select_range=False,
            d_support=1.0,
        ),
        inputs=(InputRef.from_array("x", xa),),
        seed=_seed_record(seed),
        notes=("wavelet leaders; S_L(q,2^j) = mean_k L(j,k)^q ~ 2^{j zeta(q)} (signal view)",),
    )
    return _structure_from_pyramid(pyramid, qa, prov)[0]


def wavelet_coefficients_structure(
    x: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    wavelet: str = "db3",
    j_min: int = 3,
    j_max: int | None = None,
    boundary_margin: int | None = None,
    min_leaders: int = 8,
    seed: SeedLike = None,
) -> StructureFunctions:
    r"""Bare-coefficient scaling surface :math:`S_d(q,2^j)=\langle|d(j,\cdot)|^q\rangle`.

    The comparison object for :func:`wavelet_leaders_structure`. Valid for
    :math:`q>0` only; see :func:`wavelet_coefficients_exponents`.
    """
    xa = validate_series(x, name="signal", min_length=64)
    qa = validate_q_grid(q)
    pyramid = leader_pyramid(
        xa,
        wavelet=wavelet,
        j_min=j_min,
        j_max=j_max,
        boundary_margin=boundary_margin,
        min_leaders=min_leaders,
        leaders=False,
    )
    pyramid = _select_pyramid(pyramid, _octaves_from_scales(scales, xa.size, j_min))
    prov = new_provenance(
        f"{ESTIMATOR_COEFFS}.structure",
        estimator_version=_VERSION,
        params=_params(
            q=qa,
            scales=_default_scales(pyramid),
            wavelet=wavelet,
            j_min=j_min,
            j_max=j_max,
            boundary_margin=pyramid.boundary_margin,
            min_leaders=min_leaders,
            fit_method=FitMethod.WLS,
            select_range=False,
            d_support=1.0,
        ),
        inputs=(InputRef.from_array("x", xa),),
        seed=_seed_record(seed),
        notes=(
            "bare DWT coefficients; S_d(q,2^j) = mean_k |d(j,k)|^q. "
            "NOT a valid multifractal formalism for q < 0",
        ),
    )
    return _structure_from_pyramid(pyramid, qa, prov)[0]


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


def _reference_row(q: FloatArray) -> int:
    """Index of the :math:`q` used to choose one common scaling range.

    The range is selected **once**, on :math:`q\\approx2`, and then applied to
    every :math:`q`. Selecting per :math:`q` would let each moment order pick its
    own window, which manufactures concavity in :math:`\\hat\\zeta` out of
    windowing noise; it also fails outright at :math:`q=0`, where
    :math:`\\log S_L\\equiv0` and :math:`R^2` is undefined.
    """
    return int(np.argmin(np.abs(q - 2.0)))


def _fit_rows(
    scales: FloatArray,
    log_y: FloatArray,
    counts: NDArray[np.int64],
    *,
    fit_method: FitMethod,
    select_range: bool,
    reference: int,
    reference_name: str,
) -> tuple[tuple[LogLogFit, ...], ScalingRange, list[str]]:
    """Fit every row of ``log_y`` against ``log(scales)`` over one common window.

    ``reference_name`` names the row the window was selected on, for the audit
    record. It is the moment order (``"q=2"``), never its index into the grid: a
    label reading ``q=28`` on a grid that stops at 5 misreports the fit.
    """
    weights = weighted_by_counts(counts)
    notes: list[str] = []
    mask = np.ones(scales.size, dtype=bool)
    selection = "manual"
    if select_range:
        if scales.size < 5:
            notes.append(
                f"select_range requested but only {scales.size} octaves are available "
                "(5 are needed); the full octave range was used"
            )
        else:
            sel = select_scaling_range(
                scales,
                log_y[reference],
                min_points=5,
                weights=weights,
                method=fit_method,
                log_y=False,
            )
            mask = sel.mask
            selection = f"auto-r2@{reference_name}"

    sx = scales[mask]
    sw = weights[mask]
    fits = tuple(
        loglog_fit(sx, log_y[i][mask], method=fit_method, weights=sw, log_y=False)
        for i in range(log_y.shape[0])
    )
    r2 = float(np.nanmedian([f.r_squared for f in fits]))
    scaling_range = ScalingRange(
        s_min=float(sx.min()),
        s_max=float(sx.max()),
        n_scales=int(sx.size),
        selection=selection,
        r_squared=r2,
        mask=tuple(bool(m) for m in mask),
    )
    return fits, scaling_range, notes


def _saturation_diagnostics(
    q: FloatArray, zeta: FloatArray, n_psi: int, tolerance: float
) -> dict[str, Any]:
    r"""Flag exponents pressing against the wavelet's :math:`N_\psi` ceiling.

    :math:`\alpha=\zeta'(q)` is the Hölder exponent the moment order :math:`q`
    probes, and no wavelet with :math:`N_\psi` vanishing moments can report
    :math:`\alpha>N_\psi`. A small margin therefore means the *analysing wavelet*
    truncated the spectrum, and the correct response is to re-run with a larger
    ``N_psi`` --- not to report the truncated result.
    """
    alpha = np.gradient(zeta, q)
    a_max = float(np.max(alpha))
    a_min = float(np.min(alpha))
    margin = float(n_psi) - a_max
    diag: dict[str, Any] = {
        "alpha_max_estimated": a_max,
        "alpha_min_estimated": a_min,
        "n_vanishing_moments": int(n_psi),
        "saturation_margin": margin,
        "saturated": bool(margin < tolerance),
    }
    if margin < tolerance:
        diag["saturation_warning"] = (
            f"estimated alpha_max={a_max:.3f} is within {tolerance} of the wavelet's "
            f"{n_psi} vanishing moments: exponents saturate at N_psi, so the right branch "
            "of f(alpha) is truncated by the wavelet, not by the data. Re-run with a "
            f"wavelet of at least {math.ceil(a_max + 1)} vanishing moments."
        )
    if a_min <= 0.0:
        diag["uniform_regularity_warning"] = (
            f"estimated alpha_min={a_min:.3f} <= 0: the wavelet-leader formalism assumes a "
            "uniform Holder exponent h_min > 0. Integrate the signal once (and subtract 1 "
            "from zeta) before trusting these exponents."
        )
    return diag


def _exponents(
    *,
    qa: FloatArray,
    pyramid: LeaderPyramid,
    log_s: FloatArray,
    structure: StructureFunctions,
    fit_method: FitMethod,
    select_range: bool,
    d_support: float,
    saturation_tolerance: float,
    estimator: str,
    prov: Provenance,
) -> ScalingExponents:
    fits, scaling_range, notes = _fit_rows(
        pyramid.scales,
        log_s,
        pyramid.counts,
        fit_method=fit_method,
        select_range=select_range,
        reference=_reference_row(qa),
        reference_name=f"q={float(qa[_reference_row(qa)]):g}",
    )
    zeta = np.asarray([f.slope for f in fits], dtype=np.float64)
    stderr = np.asarray([f.slope_stderr for f in fits], dtype=np.float64)
    lo = np.asarray([f.slope_ci[0] for f in fits], dtype=np.float64)
    hi = np.asarray([f.slope_ci[1] for f in fits], dtype=np.float64)

    diagnostics: dict[str, Any] = {
        **_pyramid_diagnostics(pyramid),
        **_saturation_diagnostics(qa, zeta, pyramid.n_vanishing_moments, saturation_tolerance),
        "scaling_range": scaling_range.to_canonical(),
        "median_r_squared": scaling_range.r_squared,
        "zeta_at_zero": float(zeta[int(np.argmin(np.abs(qa)))]),
    }
    if notes:
        diagnostics["notes"] = notes
    if not pyramid.leaders:
        diagnostics["negative_q_invalid"] = bool(np.any(qa < 0.0))

    return ScalingExponents(
        q=qa,
        zeta=zeta,
        d_support=float(d_support),
        zeta_stderr=stderr,
        zeta_ci=(lo, hi),
        fits=fits,
        scaling_range=scaling_range,
        structure=structure,
        estimator=estimator,
        provenance=prov.with_diagnostics(**diagnostics),
    )


def exponents_from_structure(
    structure: StructureFunctions,
    *,
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
    scale_range: tuple[float, float] | None = None,
    d_support: float = 1.0,
    saturation_tolerance: float = 0.5,
    estimator: str = "wavelet_leaders.refit",
) -> ScalingExponents:
    r"""Refit :math:`\zeta` from a stored scaling surface, over a new scale window.

    The whole point of persisting :class:`~mmar.core.types.StructureFunctions`:
    the scaling range is a modelling choice, and changing it must not require
    recomputing a wavelet transform. Requires ``structure.counts`` (present on
    everything this module produces) for the count weighting.

    ``saturation_tolerance`` has the same meaning as in :func:`wavelet_leaders`:
    the refit runs the same :math:`N_\psi` ceiling check, because an exponent
    fitted over a narrower window saturates just as readily as one fitted over
    the full octave range, and the two paths must not disagree about whether the
    answer is the wavelet's or the data's.
    """
    if structure.counts is None:
        raise ValueError("the structure functions carry no window counts; cannot weight the fit")
    keep = np.ones(structure.scales.size, dtype=bool)
    if scale_range is not None:
        keep = (structure.scales >= scale_range[0]) & (structure.scales <= scale_range[1])
        if int(keep.sum()) < 3:
            raise ValueError(
                f"scale_range {scale_range} retains {int(keep.sum())} scales; at least 3 needed"
            )
    scales = structure.scales[keep]
    counts = np.asarray(structure.counts, dtype=np.int64)[keep]
    with np.errstate(divide="ignore"):
        log_s = np.log(structure.values[:, keep])
    if not np.all(np.isfinite(log_s)):
        raise ValueError(
            "the stored scaling surface contains non-positive or overflowed moments; "
            "restrict q or the scale range before refitting"
        )
    fits, scaling_range, notes = _fit_rows(
        scales,
        log_s,
        counts,
        fit_method=fit_method,
        select_range=select_range,
        reference=_reference_row(structure.q),
        reference_name=f"q={float(structure.q[_reference_row(structure.q)]):g}",
    )
    zeta = np.asarray([f.slope for f in fits], dtype=np.float64)
    prov = (
        structure.provenance.derive(
            estimator,
            estimator_version=_VERSION,
            params={
                "fit_method": str(fit_method),
                "select_range": bool(select_range),
                "scale_range": list(scale_range) if scale_range is not None else None,
                "d_support": float(d_support),
            },
            notes=tuple(notes),
        )
        if structure.provenance is not None
        else None
    )
    if prov is not None and structure.provenance is not None:
        # `derive` starts a fresh diagnostics map. Without this the cheap refit
        # path returns a record with no wavelet, no octave leader counts and no
        # scaling range -- so a result rebuilt from a stored surface cannot be
        # told apart from one fitted over a different window with a different
        # filter. Diagnostics are non-identifying, so `root` is untouched.
        diagnostics: dict[str, Any] = {
            **dict(structure.provenance.diagnostics),
            "scaling_range": scaling_range.to_canonical(),
            "median_r_squared": scaling_range.r_squared,
            "zeta_at_zero": float(zeta[int(np.argmin(np.abs(structure.q)))]),
        }
        n_psi = structure.provenance.diagnostics.get("n_vanishing_moments")
        if isinstance(n_psi, int):
            diagnostics.update(
                _saturation_diagnostics(structure.q, zeta, n_psi, saturation_tolerance)
            )
        if notes:
            diagnostics["notes"] = notes
        prov = prov.with_diagnostics(**diagnostics)
    return ScalingExponents(
        q=structure.q,
        zeta=zeta,
        d_support=float(d_support),
        zeta_stderr=np.asarray([f.slope_stderr for f in fits], dtype=np.float64),
        zeta_ci=(
            np.asarray([f.slope_ci[0] for f in fits], dtype=np.float64),
            np.asarray([f.slope_ci[1] for f in fits], dtype=np.float64),
        ),
        fits=fits,
        scaling_range=scaling_range,
        structure=structure,
        estimator=estimator,
        provenance=prov,
    )


# --------------------------------------------------------------------------- #
# Public estimators
# --------------------------------------------------------------------------- #


def wavelet_leaders(
    x: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    wavelet: str = "db3",
    j_min: int = 3,
    j_max: int | None = None,
    boundary_margin: int | None = None,
    min_leaders: int = 8,
    saturation_tolerance: float = 0.5,
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
    d_support: float = 1.0,
    seed: SeedLike = None,
) -> ScalingExponents:
    r"""Scaling exponents from wavelet leaders. The reference estimator of this library.

    .. math::
        S_L(q,2^j) = \frac{1}{n_j}\sum_k L(j,k)^q \sim 2^{\,j\zeta(q)},
        \qquad
        \hat\zeta(q) = \frac{\partial \log_2 S_L(q,2^j)}{\partial j}.

    **Convention.** The estimator is native in :math:`\zeta` (the signal view):
    :math:`S_L` is an *average* of :math:`n_j` leaders, matching
    :math:`S(q,\delta)=\langle|\Delta_\delta X|^q\rangle\sim\delta^{\zeta(q)}`
    exactly, so the fitted slope *is* :math:`\zeta` and no offset is applied
    anywhere. :math:`\tau(q)=\zeta(q)-d` follows from
    :attr:`~mmar.core.types.ScalingExponents.tau`. Two anchors hold: fBm of Hurst
    :math:`H` gives :math:`\zeta(q)=qH`, and the binomial cascade with
    multipliers :math:`(m_0,m_1)` (analysed through its trading-time integral)
    gives :math:`\zeta(q)=1-\log_2(m_0^q+m_1^q)`. Note
    :math:`\hat\zeta(0)=0` *identically*: :math:`S_L(0,2^j)=1` for every
    :math:`j`, so :math:`\hat\tau(0)=-d` is exact by construction rather than
    estimated --- unlike the increment-based estimators, where it is a genuine
    consistency check.

    Parameters
    ----------
    q
        Moment grid, default :math:`41` points on :math:`[-5,5]`. **Negative
        :math:`q` is the point of this estimator**: the right branch of
        :math:`f(\alpha)` (:math:`\alpha>\alpha_0`, the smoothest parts of the
        signal) is generated by :math:`q<0` and is unreachable otherwise. See
        the module docstring for why leaders, unlike coefficients, make those
        moments converge.
    scales
        Must be exact powers of two. Default: every octave from ``j_min`` up to
        the coarsest one leaving ``min_leaders`` uncontaminated leaders.
    wavelet
        Any orthogonal PyWavelets name. ``"db3"`` has :math:`N_\psi=3`, which
        covers :math:`\alpha<3` --- ample for financial and turbulent data
        (:math:`\alpha_{\max}\approx1.5`) and for the standard cascades. **It is
        a hard ceiling**: exponents saturate at :math:`N_\psi` and the
        provenance diagnostics flag it (see ``saturation_tolerance``).
    j_min
        Finest octave. Defaults to 3, not 1: the two finest octaves carry the
        discretisation transient of the DWT, and because the leader recursion
        propagates suprema upward, including them contaminates every coarser
        octave as well.
    saturation_tolerance
        Flag the result when :math:`N_\psi-\hat\alpha_{\max}` falls below this.
    fit_method
        ``WLS`` by default, weighted by the leader count :math:`n_j`, which is
        the inverse-variance weighting for a log-moment estimated from
        :math:`n_j` samples. Since :math:`n_j` halves per octave this puts
        :math:`\sim2/3` of the weight on the finest octave; if the log--log plot
        curves at fine scales, use ``select_range=True`` or raise ``j_min``.
    select_range
        Choose the widest linear window automatically, once, on the
        :math:`q\approx2` row, and apply it to every :math:`q`.
    seed
        Unused --- the estimator is deterministic --- but accepted and recorded
        so that every estimator in the library has one signature.

    Returns
    -------
    ScalingExponents
        With ``fits`` (one per :math:`q`), ``structure`` (the full surface, so
        the exponents can be refitted by :func:`exponents_from_structure`), and
        provenance diagnostics covering the octaves used, the boundary and
        zero-leader counts, and the :math:`N_\psi` saturation margin.

    Notes
    -----
    Bias. See the module docstring: the dominant term is the finite-sample
    contraction of :math:`\hat\zeta` at large :math:`|q|`, worst on
    *deterministic* cascades where the wavelet filter's cancellations are
    perfectly correlated across the dyadic tree rather than averaged out. On the
    binomial measure with :math:`m_0=0.6` at depth 16 this reaches
    :math:`|\Delta\zeta|\approx0.6` at :math:`|q|=5` while staying near
    :math:`0.1` for randomised cascades of the same :math:`\tau`; the numbers are
    tabulated in the test suite. Prefer :func:`leader_cumulants` when a scalar
    summary suffices --- it is far better conditioned.
    """
    xa = validate_series(x, name="signal", min_length=64)
    qa = validate_q_grid(q)
    pyramid = leader_pyramid(
        xa,
        wavelet=wavelet,
        j_min=j_min,
        j_max=j_max,
        boundary_margin=boundary_margin,
        min_leaders=min_leaders,
        leaders=True,
    )
    pyramid = _select_pyramid(pyramid, _octaves_from_scales(scales, xa.size, j_min))
    prov = new_provenance(
        ESTIMATOR_LEADERS,
        estimator_version=_VERSION,
        params=_params(
            q=qa,
            scales=_default_scales(pyramid),
            wavelet=wavelet,
            j_min=j_min,
            j_max=j_max,
            boundary_margin=pyramid.boundary_margin,
            min_leaders=min_leaders,
            fit_method=fit_method,
            select_range=select_range,
            d_support=d_support,
            extra={"saturation_tolerance": float(saturation_tolerance)},
        ),
        inputs=(InputRef.from_array("x", xa),),
        seed=_seed_record(seed),
        notes=(
            "wavelet leaders (Jaffard 2004); zeta is native (signal view): "
            "S_L(q,2^j) = mean_k L(j,k)^q ~ 2^{j zeta(q)}, tau = zeta - d",
        ),
    )
    structure, log_s = _structure_from_pyramid(pyramid, qa, prov)
    return _exponents(
        qa=qa,
        pyramid=pyramid,
        log_s=log_s,
        structure=structure,
        fit_method=fit_method,
        select_range=select_range,
        d_support=d_support,
        saturation_tolerance=saturation_tolerance,
        estimator=ESTIMATOR_LEADERS,
        prov=prov,
    )


def wavelet_coefficients_exponents(
    x: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    wavelet: str = "db3",
    j_min: int = 3,
    j_max: int | None = None,
    boundary_margin: int | None = None,
    min_leaders: int = 8,
    saturation_tolerance: float = 0.5,
    strict_positive_q: bool = False,
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
    d_support: float = 1.0,
    seed: SeedLike = None,
) -> ScalingExponents:
    r"""Plain-DWT structure functions :math:`\langle|d(j,\cdot)|^q\rangle`. For comparison only.

    .. math::
        S_d(q,2^j) = \frac{1}{n_j}\sum_k |d(j,k)|^q \sim 2^{\,j\zeta_d(q)} .

    Identical machinery to :func:`wavelet_leaders` with the supremum removed.
    Provided so that the leader correction can be *measured* rather than
    asserted, and because for :math:`q>0` on a monofractal the two agree and this
    one is cheaper.

    **Invalid for :math:`q<0`, and not merely inaccurate.** The wavelet
    coefficient is an inner product :math:`\langle X,\psi_{j,k}\rangle`, so it
    vanishes wherever :math:`X` happens to be locally orthogonal to the analysing
    wavelet --- an event with positive probability that has nothing to do with
    the local regularity. Hence:

    * :math:`|d(j,k)|\lesssim 2^{jh}` is an upper bound only; there is no matching
      lower bound, so a small :math:`|d|` does *not* imply a large :math:`h`.
    * The negative moment :math:`\langle|d|^{-|q|}\rangle` is dominated by the
      single smallest coefficient. As :math:`n_j` grows, that minimum approaches
      zero, the moment diverges, and the fitted slope drifts with sample size
      instead of converging. Its log--log plot nonetheless looks straight.

    A leader :math:`L(j,k)=\sup_{\lambda'\subset3\lambda}|d(\lambda')|` cannot be
    small unless *every* coefficient over a whole neighbourhood and all finer
    scales is small, which does imply local smoothness. That is the entire
    difference, and it is why :func:`wavelet_leaders` is the estimator to use.

    Parameters
    ----------
    strict_positive_q
        Raise on any :math:`q<0` instead of computing a meaningless number. Off
        by default only so that the default grid :math:`[-5,5]` remains usable
        for the side-by-side comparison this function exists for; when it is off
        the provenance diagnostic ``negative_q_invalid`` is set.
    """
    xa = validate_series(x, name="signal", min_length=64)
    qa = validate_q_grid(q)
    if strict_positive_q and bool(np.any(qa < 0.0)):
        raise ValueError(
            f"{int(np.sum(qa < 0.0))} negative moment orders were requested. Wavelet "
            "coefficients admit no negative moments (they can be arbitrarily close to "
            "zero by cancellation against the wavelet, independently of the local "
            "regularity), so the fitted exponent would drift with sample size. Use "
            "wavelet_leaders, or pass strict_positive_q=False to compute it anyway."
        )
    pyramid = leader_pyramid(
        xa,
        wavelet=wavelet,
        j_min=j_min,
        j_max=j_max,
        boundary_margin=boundary_margin,
        min_leaders=min_leaders,
        leaders=False,
    )
    pyramid = _select_pyramid(pyramid, _octaves_from_scales(scales, xa.size, j_min))
    prov = new_provenance(
        ESTIMATOR_COEFFS,
        estimator_version=_VERSION,
        params=_params(
            q=qa,
            scales=_default_scales(pyramid),
            wavelet=wavelet,
            j_min=j_min,
            j_max=j_max,
            boundary_margin=pyramid.boundary_margin,
            min_leaders=min_leaders,
            fit_method=fit_method,
            select_range=select_range,
            d_support=d_support,
            extra={
                "saturation_tolerance": float(saturation_tolerance),
                "strict_positive_q": bool(strict_positive_q),
            },
        ),
        inputs=(InputRef.from_array("x", xa),),
        seed=_seed_record(seed),
        notes=(
            "bare DWT coefficients; zeta is native (signal view). "
            "NOT a valid multifractal formalism for q < 0: |d| has no lower bound in 2^{jh}",
        ),
    )
    structure, log_s = _structure_from_pyramid(pyramid, qa, prov)
    return _exponents(
        qa=qa,
        pyramid=pyramid,
        log_s=log_s,
        structure=structure,
        fit_method=fit_method,
        select_range=select_range,
        d_support=d_support,
        saturation_tolerance=saturation_tolerance,
        estimator=ESTIMATOR_COEFFS,
        prov=prov,
    )


# --------------------------------------------------------------------------- #
# Log-cumulants
# --------------------------------------------------------------------------- #


def _log_cumulant_curves(pyramid: LeaderPyramid, order: int) -> FloatArray:
    r"""``C[p-1, i]`` = the :math:`p`-th cumulant of :math:`\ln L` at octave ``i``.

    Unbiased :math:`k`-statistics are used, not the plug-in moments. At the
    coarsest octaves :math:`n_j` is :math:`\mathcal O(10)` and the plug-in
    variance is biased by :math:`-\sigma^2/n_j`, which tilts the :math:`C_2(j)`
    line and inflates :math:`|\hat c_2|` --- i.e. it manufactures
    multifractality out of small samples, exactly the artefact :math:`c_2` is
    supposed to detect. The estimators used are

    .. math::
        k_2 = \frac{n}{n-1}m_2, \qquad
        k_3 = \frac{n^2}{(n-1)(n-2)}m_3,

    with :math:`m_p` the plug-in central moments. Beyond :math:`p=3` the plug-in
    central moment is used and the residual bias is :math:`O(1/n_j)`; the fourth
    log-cumulant is rarely identifiable on real data anyway.
    """
    out = np.zeros((order, pyramid.octaves.size), dtype=np.float64)
    for i, v in enumerate(pyramid.values):
        y = np.log(v)
        n = float(y.size)
        centred = y - y.mean()
        out[0, i] = float(y.mean())
        if order >= 2:
            out[1, i] = float(np.sum(centred**2) / (n - 1.0))
        if order >= 3:
            m3 = float(np.sum(centred**3) / n)
            out[2, i] = n * n / ((n - 1.0) * (n - 2.0)) * m3
        for p in range(4, order + 1):
            out[p - 1, i] = float(np.sum(centred**p) / n)
    return out


def leader_cumulants(
    x: Sequence[float] | NDArray[Any],
    *,
    order: int = 3,
    scales: Sequence[int] | NDArray[Any] | None = None,
    wavelet: str = "db3",
    j_min: int = 3,
    j_max: int | None = None,
    boundary_margin: int | None = None,
    min_leaders: int = 8,
    saturation_tolerance: float = 0.5,
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
    d_support: float = 1.0,
    seed: SeedLike = None,
) -> LogCumulants:
    r"""Log-cumulants :math:`c_p` of the wavelet leaders. The best-conditioned diagnostic.

    .. math::
        C_p(j) \;=\; \mathrm{cum}_p\big[\ln L(j,\cdot)\big]
                 \;=\; c_p^{0} + c_p\, j\ln 2
        \quad\Longrightarrow\quad
        c_p = \frac{\partial C_p(j)}{\partial (j\ln 2)},

    so each :math:`c_p` is a single slope: :math:`C_1(j)=\langle\ln L\rangle`,
    :math:`C_2(j)=\operatorname{Var}\ln L`, :math:`C_3(j)` the third central
    moment. The identification is exact, not an approximation --- expanding
    :math:`\ln S_L(q,j)=\ln\langle e^{q\ln L}\rangle=\sum_p C_p(j)q^p/p!` and
    matching against :math:`\ln S_L = j\ln2\,\zeta(q) + \text{const}` gives
    :math:`\zeta(q)=\sum_p c_p q^p/p!` term by term.

    Interpretation
    --------------
    ================  =====================================================
    :math:`c_1`       :math:`=\alpha_0`, the most probable Hölder exponent.
    :math:`c_2`       :math:`\le0`; the multifractality strength. **Exactly
                      zero for a monofractal**, since :math:`\zeta(q)=qH` has
                      no quadratic term. This is *the* monofractality test.
    :math:`c_3`       Asymmetry of :math:`f(\alpha)`; zero for the lognormal
                      cascade, which is exactly second order.
    ================  =====================================================

    Near the apex, :math:`f(\alpha)\approx d+(\alpha-c_1)^2/(2c_2)`, available as
    :meth:`~mmar.core.types.LogCumulants.parabolic_spectrum`.

    Why prefer this to a Legendre transform. The Legendre route infers curvature
    by differencing slopes fitted at large :math:`|q|`, where the moments are
    dominated by a handful of extreme leaders; the cumulant route reads the same
    curvature off a *variance at* :math:`q=0`, using every leader with equal
    leverage. The variance of :math:`\hat c_2` is therefore smaller by roughly an
    order of magnitude at the sample sizes that occur in practice, and it does
    not inherit the concavity pathologies of :math:`\hat\zeta`.

    Bias
    ----
    Unbiased :math:`k`-statistics remove the leading :math:`-\sigma^2/n_j` bias
    of the plug-in variance, which would otherwise tilt :math:`C_2(j)` and report
    multifractality on monofractal data. What remains is (i) the finite-size
    positive bias of :math:`\hat c_2` on short series --- on exact fBm at
    :math:`n=2^{16}` this implementation returns :math:`|\hat c_2|\lesssim0.01`,
    which is the resolution floor below which a :math:`c_2` claim is not
    meaningful --- and (ii) the fact that :math:`\ln L` is a supremum, hence
    slightly right-skewed, which biases :math:`\hat c_3` positive.

    Parameters
    ----------
    order
        Number of cumulants; :math:`\ge1`. Beyond 3 they are rarely identifiable.
    d_support
        Recorded on the result so that
        :meth:`~mmar.core.types.LogCumulants.parabolic_spectrum` has the right
        apex height :math:`f(c_1)=d`.

    Returns
    -------
    LogCumulants
        ``c[0] = c1``, ``c[1] = c2``, ... with standard errors from the
        per-cumulant regressions.
    """
    if order < 1:
        raise ValueError(f"order must be at least 1, got {order}")
    xa = validate_series(x, name="signal", min_length=64)
    pyramid = leader_pyramid(
        xa,
        wavelet=wavelet,
        j_min=j_min,
        j_max=j_max,
        boundary_margin=boundary_margin,
        min_leaders=min_leaders,
        leaders=True,
    )
    pyramid = _select_pyramid(pyramid, _octaves_from_scales(scales, xa.size, j_min))
    if order >= 3 and int(pyramid.counts.min()) < 3:
        raise ValueError(
            f"the third log-cumulant needs at least 3 leaders per octave; the coarsest "
            f"octave has {int(pyramid.counts.min())}"
        )

    curves = _log_cumulant_curves(pyramid, order)
    # C_p(j) is affine in j*ln2, and log(scale) = j*ln2 exactly, so the slope of
    # C_p against log(scale) is c_p with no further conversion.
    fits, scaling_range, notes = _fit_rows(
        pyramid.scales,
        curves,
        pyramid.counts,
        fit_method=fit_method,
        select_range=select_range,
        reference=0,
        reference_name="c1",
    )
    c = np.asarray([f.slope for f in fits], dtype=np.float64)
    stderr = np.asarray([f.slope_stderr for f in fits], dtype=np.float64)

    diagnostics: dict[str, Any] = {
        **_pyramid_diagnostics(pyramid),
        "scaling_range": scaling_range.to_canonical(),
        "r_squared_per_cumulant": [f.r_squared for f in fits],
        "c1_vs_vanishing_moments": float(pyramid.n_vanishing_moments) - float(c[0]),
    }
    if float(pyramid.n_vanishing_moments) - float(c[0]) < saturation_tolerance:
        diagnostics["saturation_warning"] = (
            f"c1={float(c[0]):.3f} is within {saturation_tolerance} of the wavelet's "
            f"{pyramid.n_vanishing_moments} vanishing moments; exponents saturate at "
            "N_psi. Re-run with a longer Daubechies wavelet."
        )
    if order >= 2 and float(c[1]) > 0.0:
        diagnostics["c2_positive"] = (
            f"c2={float(c[1]):.4g} > 0 is theoretically impossible (zeta is concave); "
            "it is estimator noise, and its magnitude is the resolution floor below "
            "which a multifractality claim from this sample is not supported"
        )
    if notes:
        diagnostics["notes"] = notes

    prov = new_provenance(
        ESTIMATOR_CUMULANTS,
        estimator_version=_VERSION,
        params=_params(
            q=np.zeros(0, dtype=np.float64),
            scales=_default_scales(pyramid),
            wavelet=wavelet,
            j_min=j_min,
            j_max=j_max,
            boundary_margin=pyramid.boundary_margin,
            min_leaders=min_leaders,
            fit_method=fit_method,
            select_range=select_range,
            d_support=d_support,
            extra={"order": int(order), "saturation_tolerance": float(saturation_tolerance)},
        ),
        inputs=(InputRef.from_array("x", xa),),
        seed=_seed_record(seed),
        notes=(
            "log-cumulants of the wavelet leaders: zeta(q) = sum_p c_p q^p / p!; "
            "c1 = alpha_0, c2 <= 0 the multifractality strength, c2 = 0 iff monofractal",
        ),
    ).with_diagnostics(**diagnostics)

    return LogCumulants(
        c=c,
        stderr=stderr,
        order=int(order),
        d_support=float(d_support),
        provenance=prov,
    )
