r"""Pointwise Hölder exponents: the field :math:`t\mapsto h(t)`, not its distribution.

Everything else in :mod:`mmar.estimators` estimates the *distribution* of local
regularity --- :math:`\tau(q)`, :math:`f(\alpha)`, the log-cumulants. This module
estimates the regularity itself, sample by sample, so that a singular event can
be located in time rather than merely counted.

Definition
----------
A function :math:`f` belongs to :math:`C^{h}(t_0)`, :math:`h\ge 0`, if there are
a constant :math:`C`, a radius :math:`\rho>0` and a polynomial :math:`P` with
:math:`\deg P < h` such that

.. math::
    |f(t) - P(t-t_0)| \le C\,|t-t_0|^{h}, \qquad |t-t_0|<\rho .

The **pointwise Hölder exponent** is

.. math::
    h(t_0) = \sup\{\,h\ge 0 : f\in C^{h}(t_0)\,\}.

Small :math:`h(t_0)` means a rough point (a jump has :math:`h=0`, a cusp
:math:`|t-t_0|^{a}` has :math:`h=a`); :math:`h(t_0)\ge 1` means differentiable
behaviour at :math:`t_0`. The polynomial :math:`P` is why every estimator here
either differences the signal (oscillation) or analyses it with a wavelet having
:math:`N_\psi > h` vanishing moments: both annihilate :math:`P` and leave the
singular part.

The three estimators
--------------------
=====================  ==========================================  ==============
Estimator              Multiresolution quantity                     Scales as
=====================  ==========================================  ==============
:func:`oscillation_holder`  :math:`\operatorname{osc}(t,\delta)=
                            \sup_{I}f-\inf_{I}f`, :math:`|I|=\delta`  :math:`\delta^{h(t)}`
:func:`wavelet_holder`      :math:`\sup_{|b-t|\le \kappa a}|T(a,b)|`  :math:`a^{h(t)+1/2}`
:func:`leader_holder`       :math:`L(j,k)=\sup_{\lambda'\subset
                            3\lambda_{j,k}}|d_{\lambda'}|`           :math:`2^{\,j h(t)}`
=====================  ==========================================  ==============

They are genuinely independent --- a different transform, a different definition
of "local", a different bias --- and they do **not** agree to three decimals on a
finite sample. Section *Bias* of each docstring states the direction and the
measured magnitude; ``tests/unit/test_pointwise.py`` records the numbers.

Why the leaders are the theoretically sound one
-----------------------------------------------
Wavelet *coefficients* only bound the exponent from above: :math:`|d_{j,k}|
\lesssim 2^{j(h+1/2)}` is necessary but not sufficient, because a coefficient can
be accidentally small (the wavelet can be nearly orthogonal to the singularity)
and the decay rate of :math:`|d_{j,k}|` alone can therefore *overestimate*
:math:`h`. Jaffard (2004) shows that the wavelet **leaders**, the local suprema
of the coefficients over the three adjacent dyadic intervals at all finer scales,
satisfy a two-sided bound: for a signal with uniform regularity
:math:`h_{\min}>0`,

.. math::
    h(t) = \liminf_{j\to\infty} \frac{\log_2 L(j, k_j(t))}{j},

with :math:`\lambda_{j,k_j(t)}\ni t`, and the relation is an equivalence, not an
inequality. Equivalently the leaders bound the local oscillation from both sides,
which the coefficients do not. That is the whole reason wavelet leaders replaced
WTMM in the modern literature.

Weighting (and why the structure-function rule does not transfer)
-----------------------------------------------------------------
For a structure function the fit *must* be weighted by the window count, because
:math:`S(q,\delta)` is an average over :math:`n_\delta \propto \delta^{-1}`
windows and its log-variance falls like :math:`1/n_\delta`. Here every cell
:math:`(t,\delta)` of the multiresolution surface is **one** extremal statistic,
not an average: :math:`\operatorname{osc}(t,\delta)` is a single range,
:math:`L(j,k)` a single supremum. Their log-variance is scale-invariant to
leading order, so count-weighting would be a misspecification, and the default is
therefore ordinary least squares over a geometric scale grid. ``weights`` is
exposed for callers who want to down-weight the coarse (least local) scales
deliberately; the choice is recorded in provenance either way.

Degeneracies are recorded, never hidden
---------------------------------------
A constant stretch has :math:`\operatorname{osc}=0`; a wavelet coefficient can be
exactly zero; a leader can vanish on a locally polynomial signal. The logarithm
is undefined there. Those cells are dropped from the regression, the number
dropped is written to ``provenance.diagnostics["n_nonpositive_cells"]``, and a
point with fewer than ``min_points`` usable scales gets ``h = nan`` rather than a
fabricated slope.

References
----------
Jaffard, S. (1991). Pointwise smoothness, two-microlocalization and wavelet
coefficients. *Publicacions Matemàtiques* 35, 155--168.

Mallat, S. and Hwang, W.L. (1992). Singularity detection and processing with
wavelets. *IEEE Transactions on Information Theory* 38(2), 617--643.

Tricot, C. (1995). *Curves and Fractal Dimension*. Springer. (Oscillation
definition of the local exponent, Ch. 5.)

Jaffard, S. (2004). Wavelet techniques in multifractal analysis. In *Fractal
Geometry and Applications*, Proc. Sympos. Pure Math. 72(2), 91--151.

Wendt, H., Abry, P. and Jaffard, S. (2007). Bootstrap for empirical multifractal
analysis. *IEEE Signal Processing Magazine* 24(4), 38--48.

Peltier, R.F. and Lévy Véhel, J. (1995). Multifractional Brownian motion.
*INRIA Research Report* 2645. (The mBm ground truth: :math:`h(t)` prescribed.)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from mmar.core.provenance import InputRef, Provenance, new_provenance
from mmar.core.regression import loglog_fit
from mmar.core.types import FitMethod, FloatArray, LogLogFit, PointwiseHolder
from mmar.core.validate import check_scales, validate_series

if TYPE_CHECKING:  # pragma: no cover - typing only
    from types import ModuleType

__all__ = [
    "PointwiseMethod",
    "RegularityMap",
    "WindowShape",
    "holder_at",
    "leader_holder",
    "local_regularity_map",
    "oscillation_holder",
    "wavelet_holder",
]

_VERSION = "1"

PointwiseMethod = Literal["oscillation", "wavelet", "leaders"]
WindowShape = Literal["centred", "causal"]

_MIN_POINTS = 4
"""Fewest usable scales for a pointwise slope. Below this the "power law" is a
line through three points and its standard error is not estimable."""

_WRAP_MARGIN_OCTAVES = 3.0
r"""Boundary margin of the wavelet estimators, in units of :math:`2^{j_{\max}}`.

Measured, not guessed. Both :func:`pywt.swt` and ``pywt.wavedec(mode=
"periodization")`` extend the record *periodically*, so the two ends see the
wrap-around jump :math:`x_{n-1}-x_0`. Isolating that jump (subtract the chord
:math:`x_{n-1}-x_0`, which ``db3`` annihilates, and difference the two fields)
on exact fBm, eight seeds, :math:`n=2^{14}`, the induced error in the field
averaged over successive blocks of :math:`2^{j_{\max}}` samples inwards from
each end is

=======  ==============================  ==============================
block    wavelet (:math:`H=0.7`)         leaders (:math:`H=0.7`)
=======  ==============================  ==============================
0        :math:`+0.23 / +0.31`           :math:`+0.33 / +0.27`
1        :math:`+0.04 / +0.13`           :math:`+0.16 / +0.07`
2        :math:`-0.00 / +0.01`           :math:`+0.03 / -0.00`
3        :math:`-0.00 / -0.00`           :math:`-0.00 / -0.00`
=======  ==============================  ==============================

so three blocks must go and the fourth is already clean; the same three-block
figure is obtained at :math:`j_{\max}=9`, confirming that the margin scales with
:math:`2^{j_{\max}}` and is not an artefact of one octave window. A margin of
one block --- the analysing scale itself --- is *not* enough: it leaves a
systematic :math:`+0.06\ldots+0.19` in the outer quarter of the retained field
on fBm, and up to :math:`+0.5` on a record with a large end-to-end drift, which
is every log-price.
"""


# --------------------------------------------------------------------------- #
# Products
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RegularityMap:
    r"""The :math:`(t,\text{scale})` surface of *local* log--log slopes.

    Where :class:`~mmar.core.types.PointwiseHolder` collapses each row of the
    multiresolution surface to one exponent, this keeps the whole surface. The
    slope between consecutive scales,

    .. math::
        \hat h(t, a_{j+1/2}) =
        \frac{\log Y(t,a_{j+1}) - \log Y(t,a_j)}{\log a_{j+1} - \log a_j}
        - \nu ,

    with :math:`\nu` the estimator's normalisation offset (:math:`1/2` for the
    L2-normalised wavelet transform, :math:`0` otherwise), is the natural object
    to plot: a genuine power law is a horizontal band, a crossover is a step, and
    a scale range that has stopped being local curves away at the right.
    """

    t: FloatArray
    scales: FloatArray
    """Analysing scales in *samples*, geometrically ordered."""
    log_values: FloatArray
    """Shape ``(n_t, n_scales)``: natural log of the multiresolution quantity."""
    slopes: FloatArray
    """Shape ``(n_t, n_scales - 1)``: offset-corrected consecutive-scale slopes."""
    scale_midpoints: FloatArray
    """Geometric midpoints :math:`\\sqrt{a_j a_{j+1}}` labelling ``slopes``."""
    method: str
    offset: float
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        if self.log_values.shape != (self.t.size, self.scales.size):
            raise ValueError(
                f"log_values{self.log_values.shape} does not match "
                f"(n_t={self.t.size}, n_scales={self.scales.size})"
            )


@dataclass(frozen=True, slots=True)
class _Multiresolution:
    """Internal carrier: a strictly positive surface ``Y(t, a) ~ a**(h + offset)``."""

    t: FloatArray
    scales: FloatArray
    values: FloatArray
    offset: float
    method: str
    n_nonpositive: int
    trim: int
    """Width, in entries of ``t``, of the boundary-contaminated margin."""
    trim_left: bool
    trim_right: bool
    extra: dict[str, Any]


# --------------------------------------------------------------------------- #
# Vectorised per-point regression
# --------------------------------------------------------------------------- #


def _row_slopes(
    log_x: FloatArray,
    log_y: FloatArray,
    weights: FloatArray,
    *,
    min_points: int,
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.int64]]:
    """Weighted least squares of every row of ``log_y`` on the shared ``log_x``.

    One design matrix, ``n_t`` responses: the whole field is four array
    reductions, never a Python loop over samples. Cells that are not finite (the
    logarithm of a vanished oscillation or coefficient) receive zero weight, and
    a row with fewer than ``min_points`` survivors returns NaN throughout.
    """
    ok = np.isfinite(log_y)
    n_used = ok.sum(axis=1).astype(np.int64)
    w = np.where(ok, weights[None, :], 0.0)
    y = np.where(ok, log_y, 0.0)

    s0 = w.sum(axis=1)
    s0_safe = np.where(s0 > 0.0, s0, 1.0)
    xbar = (w * log_x[None, :]).sum(axis=1) / s0_safe
    ybar = (w * y).sum(axis=1) / s0_safe

    dx = log_x[None, :] - xbar[:, None]
    dy = y - ybar[:, None]
    sxx = (w * dx * dx).sum(axis=1)
    sxy = (w * dx * dy).sum(axis=1)

    good = (n_used >= min_points) & (sxx > 0.0) & (s0 > 0.0)
    slope_safe = np.where(good, sxy / np.where(sxx > 0.0, sxx, 1.0), 0.0)

    intercept = ybar - slope_safe * xbar
    fitted = slope_safe[:, None] * log_x[None, :] + intercept[:, None]
    resid = np.where(ok, y - fitted, 0.0)
    ss_res = (w * resid * resid).sum(axis=1)
    ss_tot = (w * dy * dy).sum(axis=1)

    dof = np.maximum(n_used - 2, 1)
    var = ss_res / dof
    stderr = np.sqrt(var / np.where(sxx > 0.0, sxx, 1.0))
    r2 = 1.0 - ss_res / np.where(ss_tot > 0.0, ss_tot, 1.0)

    nan = np.full(log_y.shape[0], np.nan)
    return (
        np.where(good, slope_safe, nan),
        np.where(good, stderr, nan),
        np.where(good & (ss_tot > 0.0), r2, nan),
        n_used,
    )


def _resolve_weights(
    weights: Sequence[float] | FloatArray | None,
    n_scales: int,
    fit_method: FitMethod,
) -> FloatArray:
    if fit_method not in (FitMethod.OLS, FitMethod.WLS):
        raise ValueError(
            f"fit_method={fit_method!r} is not available for the pointwise field: "
            "Theil-Sen and Huber are iterative per-row estimators and would cost one "
            "robust regression per sample. Use holder_at() for a single location, "
            "where the full mmar.core.regression machinery (including those methods) "
            "is applied with proper diagnostics."
        )
    if weights is None:
        if fit_method is FitMethod.WLS:
            raise ValueError(
                "FitMethod.WLS requires explicit weights here. A pointwise "
                "multiresolution cell is a single extremal statistic, not an average "
                "over n_delta windows, so there is no count-based inverse-variance "
                "weight to default to (see the module docstring)."
            )
        return np.ones(n_scales, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != n_scales:
        raise ValueError(f"weights has {w.size} entries but there are {n_scales} scales")
    if np.any(w <= 0.0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and strictly positive")
    return np.ascontiguousarray(w)


def _apply_trim(h: FloatArray, trim: int, *, left: bool = True, right: bool = True) -> FloatArray:
    """Blank the boundary margin. Edge extension manufactures regularity."""
    out = h.copy()
    k = min(int(trim), out.size)
    if left and k > 0:
        out[:k] = np.nan
    if right and k > 0:
        out[out.size - k :] = np.nan
    return out


def _finalise(
    mr: _Multiresolution,
    *,
    fit_method: FitMethod,
    weights: Sequence[float] | FloatArray | None,
    trim_edges: bool,
    min_points: int,
    prov: Provenance,
) -> PointwiseHolder:
    w = _resolve_weights(weights, mr.scales.size, fit_method)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_y = np.log(mr.values)
    log_y = np.where(np.isfinite(log_y), log_y, np.nan)
    slope, stderr, r2, n_used = _row_slopes(np.log(mr.scales), log_y, w, min_points=min_points)
    h = slope - mr.offset
    if trim_edges:
        h = _apply_trim(h, mr.trim, left=mr.trim_left, right=mr.trim_right)
        stderr = _apply_trim(stderr, mr.trim, left=mr.trim_left, right=mr.trim_right)
        r2 = _apply_trim(r2, mr.trim, left=mr.trim_left, right=mr.trim_right)

    finite = np.isfinite(h)
    prov = prov.with_diagnostics(
        n_nonpositive_cells=int(mr.n_nonpositive),
        n_points=int(h.size),
        n_finite=int(finite.sum()),
        n_undetermined=int(h.size - finite.sum()),
        min_usable_scales=int(n_used.min()),
        trim_samples=int(mr.trim) if trim_edges else 0,
        normalisation_offset=float(mr.offset),
        **mr.extra,
    )
    return PointwiseHolder(
        t=mr.t,
        h=np.asarray(h, dtype=np.float64),
        stderr=np.asarray(stderr, dtype=np.float64),
        r_squared=np.asarray(r2, dtype=np.float64),
        method=mr.method,
        scales=np.asarray(mr.scales, dtype=np.float64),
        provenance=prov,
    )


def _time_grid(n: int) -> FloatArray:
    r"""Normalised time :math:`t_i = (i+1)/n`, the grid of :func:`mmar.models.fbm.fbm`."""
    return np.asarray((np.arange(n, dtype=np.float64) + 1.0) / n, dtype=np.float64)


# --------------------------------------------------------------------------- #
# 1. Oscillation
# --------------------------------------------------------------------------- #


def _oscillation_surface(
    x: FloatArray,
    scales: NDArray[np.int64],
    window: WindowShape,
) -> tuple[FloatArray, int]:
    """Sliding max minus sliding min, one ``scipy.ndimage`` pass per scale."""
    out = np.empty((x.size, scales.size), dtype=np.float64)
    for j, s in enumerate(scales):
        size = int(s)
        # scipy's default window for output[i] is [i - size//2, i - size//2 + size - 1].
        # The causal shift puts it at [i - size + 1, i], which is what an online
        # estimator can actually see; it is always inside scipy's legal origin range.
        origin = 0 if window == "centred" else size // 2 - size + 1
        hi = maximum_filter1d(x, size, mode="nearest", origin=origin)
        lo = minimum_filter1d(x, size, mode="nearest", origin=origin)
        out[:, j] = hi - lo
    return out, int((out <= 0.0).sum())


def _oscillation_multiresolution(
    x: FloatArray,
    scales: NDArray[np.int64],
    window: WindowShape,
) -> _Multiresolution:
    values, n_zero = _oscillation_surface(x, scales, window)
    s_max = int(scales.max())
    trim = s_max // 2 + 1 if window == "centred" else s_max
    return _Multiresolution(
        t=_time_grid(x.size),
        scales=scales.astype(np.float64),
        values=values,
        offset=0.0,
        method="oscillation",
        n_nonpositive=n_zero,
        trim=trim,
        trim_left=True,
        # A causal window only ever reaches backwards, so the right-hand margin
        # is uncontaminated and keeping it is free information.
        trim_right=window == "centred",
        extra={"window": str(window)},
    )


def _default_oscillation_scales(n: int) -> NDArray[np.int64]:
    r"""Geometric grid :math:`8 \le \delta \le n/32`, sixteen scales.

    The cap at :math:`n/32` is a *locality* constraint, not the usual
    :math:`n/4` window-count constraint: an oscillation measured over a quarter of
    the record is not a local quantity, whatever its window count.
    """
    s_max = max(32, n // 32)
    raw = np.geomspace(8.0, float(s_max), num=16)
    return np.unique(np.round(raw).astype(np.int64))


def oscillation_holder(
    x: Sequence[float] | NDArray[Any],
    *,
    scales: Sequence[int] | NDArray[Any] | None = None,
    window: WindowShape = "centred",
    fit_method: FitMethod = FitMethod.OLS,
    weights: Sequence[float] | FloatArray | None = None,
    trim_edges: bool = True,
    min_points: int = _MIN_POINTS,
) -> PointwiseHolder:
    r"""Pointwise exponent from the local oscillation (Tricot 1995).

    Defining equation. For a window :math:`I_\delta(t)` of width :math:`\delta`
    containing :math:`t`,

    .. math::
        \operatorname{osc}(t,\delta)
            = \sup_{u\in I_\delta(t)} x(u) - \inf_{u\in I_\delta(t)} x(u),
        \qquad
        \operatorname{osc}(t,\delta) \asymp \delta^{\,h(t)} \;\; (\delta\to0),

    so :math:`h(t)` is the slope of :math:`\log\operatorname{osc}` against
    :math:`\log\delta`. The oscillation is exactly the quantity the definition of
    :math:`C^h(t_0)` controls when :math:`\deg P = 0`, which is why this estimator
    needs no wavelet and no transform: it is the definition, discretised.

    Sanity check by hand: for the cusp :math:`x(t)=|t-t_0|^{a}` with
    :math:`0<a<1` and a *centred* window,
    :math:`\operatorname{osc}(t_0,\delta)=(\delta/2)^{a}`, so the log--log slope
    is :math:`a` **exactly**, at every scale. The estimator is unbiased on the
    canonical cusp; the measured error at :math:`n=2^{14}` is
    :math:`+0.002\ldots+0.004` and is pure grid discretisation.

    Parameters
    ----------
    x
        Real series, :math:`n\ge 64`.
    scales
        Window widths :math:`\delta` in samples. Default: sixteen geometric
        scales on :math:`[8, n/32]` (see :func:`_default_oscillation_scales`).
        Scales leaving fewer than four windows are dropped by
        :func:`~mmar.core.validate.check_scales`.
    window
        ``"centred"`` (default) or ``"causal"``. The causal window uses only the
        past, :math:`I_\delta(t)=[t-\delta+1,\,t]`, and is the only admissible
        choice for an online/streaming regularity monitor; it costs a factor
        :math:`2^{h}` in amplitude but not in exponent.
    fit_method
        ``OLS`` (default; see the module docstring on why count-weighting does
        not apply) or ``WLS`` with explicit ``weights``.
    trim_edges
        Blank the :math:`\delta_{\max}/2` samples at each end, where the
        ``mode="nearest"`` edge extension flattens the signal and therefore
        *inflates* :math:`h`. On by default.
    min_points
        Minimum number of scales with a strictly positive oscillation.

    Returns
    -------
    PointwiseHolder
        ``h``, ``stderr``, ``r_squared`` per sample; ``method="oscillation"``.

    Bias
    ----
    Two effects, in opposite directions and both quantified in the tests.

    1. **Discrete-range bias, positive, the dominant one on random signals.**
       The oscillation of :math:`\delta` *samples* of a continuous process
       underestimates the oscillation of the continuous path by roughly one
       increment, :math:`\operatorname{osc}_{\text{disc}}(\delta) \approx
       c\,\delta^{h}-O(1)` in units of the unit-lag increment. Differentiating
       the logarithm gives a slope excess of order
       :math:`h\,\delta^{-h}/c`, positive and *larger for small* :math:`h`.
       Measured on exact fBm at :math:`n=2^{14}`, default grid, mean over eight
       seeds of the spatial mean of :math:`\hat h`:

       ==========  ==========  ===========================
       :math:`H`   bias        within-path s.d. of `h(t)`
       ==========  ==========  ===========================
       0.3         :math:`+0.115`  0.094
       0.5         :math:`+0.074`  0.126
       0.7         :math:`+0.049`  0.165
       ==========  ==========  ===========================

       The ordering (bias falling with :math:`H`) is exactly what
       :math:`h\,\delta^{-h}` predicts, and it is the reason this estimator is
       the *least* trustworthy of the three on noisy data despite being the most
       accurate on a clean cusp.
    2. **Ceiling at** :math:`h=1`. The oscillation only annihilates constants
       (:math:`\deg P = 0`), so a :math:`C^\infty` signal with non-zero
       derivative gives :math:`\operatorname{osc}\propto\delta` and
       :math:`\hat h = 1`, never more (measured: :math:`1.038` on a sine).
       Exponents above 1 are *not* measurable by oscillation; use
       :func:`wavelet_holder` with a wavelet having :math:`N_\psi>h` vanishing
       moments.

    Notes
    -----
    Deterministic: no randomness, no seed. Degenerate cells
    (:math:`\operatorname{osc}=0`, e.g. a locally constant stretch) are dropped
    from the regression and counted in
    ``provenance.diagnostics["n_nonpositive_cells"]``.

    References
    ----------
    Tricot, C. (1995). *Curves and Fractal Dimension*, Ch. 5. Springer.
    """
    a = validate_series(x, name="series")
    if window not in ("centred", "causal"):
        raise ValueError(f"window must be 'centred' or 'causal', got {window!r}")
    raw = _default_oscillation_scales(a.size) if scales is None else scales
    s = check_scales(raw, a.size, name="scales")
    if s.size < min_points:
        raise ValueError(f"need at least {min_points} scales, got {s.size}")

    mr = _oscillation_multiresolution(a, s, window)
    prov = new_provenance(
        "estimators.pointwise.oscillation",
        estimator_version=_VERSION,
        params={
            "scales": [int(v) for v in s],
            "window": str(window),
            "fit_method": str(fit_method),
            "weights": None if weights is None else [float(v) for v in np.asarray(weights).ravel()],
            "trim_edges": bool(trim_edges),
            "min_points": int(min_points),
        },
        inputs=(InputRef.from_array("series", a),),
        seed=None,
        notes=("osc(t, delta) ~ delta**h(t); slope is h directly (offset 0)",),
    )
    return _finalise(
        mr,
        fit_method=fit_method,
        weights=weights,
        trim_edges=trim_edges,
        min_points=min_points,
        prov=prov,
    )


# --------------------------------------------------------------------------- #
# Wavelet plumbing
# --------------------------------------------------------------------------- #


def _pywt() -> ModuleType:
    try:
        import pywt  # noqa: PLC0415  (optional extra: imported on demand, not at module load)
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the wavelet-based pointwise estimators need PyWavelets; "
            "install the 'wavelets' extra (pip install mmar[wavelets])"
        ) from exc
    module: ModuleType = pywt
    return module


def _octave_range(n: int, j_min: int, j_max: int | None) -> NDArray[np.int64]:
    r"""Usable octaves :math:`j_{\min}\ldots j_{\max}`.

    Two ceilings apply to the default, and the binding one wins.

    1. **Locality.** Stopping three octaves below :math:`\log_2 n` leaves at
       least eight dyadic intervals at the coarsest scale; a "local" exponent
       read off two intervals is a global exponent with a misleading name.
    2. **Affordable boundary margin.** The periodic extension used by both
       transforms contaminates :math:`3\cdot 2^{j_{\max}}` samples at each end
       (see :data:`_WRAP_MARGIN_OCTAVES`). An octave whose honest margin would
       consume more than half the record buys resolution the estimator then has
       to throw away, so the default also requires
       :math:`2\cdot 3\cdot 2^{j_{\max}} \le n/2`.

    At :math:`n=2^{14}` with ``db3`` this gives :math:`j_{\max}=10`, matching the
    ceiling :func:`mmar.estimators.wavelet_leaders.leader_pyramid` independently
    arrives at from its own ``boundary_margin`` budget. An explicit ``j_max``
    overrides both: the margin is still applied, so the field simply shrinks.
    """
    if j_max is None:
        top = min(
            math.floor(math.log2(n)) - 3,
            math.floor(math.log2(n / (4.0 * _WRAP_MARGIN_OCTAVES))),
        )
    else:
        top = int(j_max)
    lo = int(j_min)
    if lo < 1:
        raise ValueError(f"j_min must be at least 1, got {j_min}")
    if top < lo + _MIN_POINTS - 1:
        raise ValueError(
            f"only octaves {lo}..{top} are available for n={n}; a pointwise slope "
            f"needs at least {_MIN_POINTS}. Lengthen the series or lower j_min."
        )
    return np.arange(lo, top + 1, dtype=np.int64)


def _pad_to_multiple(x: FloatArray, block: int) -> tuple[FloatArray, int]:
    """Reflect-pad on the right to a multiple of ``block``; the pad is trimmed later."""
    pad = (-x.size) % block
    if pad == 0:
        return x, 0
    reps = math.ceil(pad / max(x.size - 1, 1))
    tail = np.tile(x[-2::-1], reps)[:pad] if x.size > 1 else np.full(pad, x[0])
    return np.concatenate([x, tail]), pad


# --------------------------------------------------------------------------- #
# 2. Wavelet coefficients along the cone of influence
# --------------------------------------------------------------------------- #


def _wavelet_multiresolution(
    x: FloatArray,
    wavelet: str,
    octaves: NDArray[np.int64],
    cone: float,
) -> _Multiresolution:
    pywt = _pywt()
    j_max = int(octaves.max())
    padded, pad = _pad_to_multiple(x, 1 << j_max)
    # norm=False is the L2 normalisation: pywt's undecimated transform then
    # reproduces T(a,b) = a^{-1/2} <x, psi((.-b)/a)> and hence a^{h+1/2}.
    coeffs = pywt.swt(padded, wavelet, level=j_max, norm=False, trim_approx=True)
    details = coeffs[1:][::-1]  # details[j-1] is octave j (finest first)

    values = np.empty((x.size, octaves.size), dtype=np.float64)
    for col, j in enumerate(octaves):
        half = max(1, round(cone * float(1 << int(j))))
        d = np.abs(np.asarray(details[int(j) - 1], dtype=np.float64))
        values[:, col] = maximum_filter1d(d, 2 * half + 1, mode="nearest")[: x.size]

    # pywt.swt extends periodically, so everything within reach of the wrap-around
    # jump is contaminated. The reach is the analysing support at j_max -- for a
    # db-N filter that is (2N-1)*2**j_max samples, far more than 2**j_max -- widened
    # again by the cone. _WRAP_MARGIN_OCTAVES is the measured figure; the cone term
    # keeps the margin honest for kappa > 2.
    trim = math.ceil(max(_WRAP_MARGIN_OCTAVES, 2.0 + cone) * (1 << j_max)) + pad
    return _Multiresolution(
        t=_time_grid(x.size),
        scales=np.asarray(2.0 ** octaves.astype(np.float64), dtype=np.float64),
        values=values,
        offset=0.5,
        method="wavelet-maxima",
        n_nonpositive=int((values <= 0.0).sum()),
        trim=trim,
        trim_left=True,
        trim_right=True,
        extra={"wavelet": str(wavelet), "cone": float(cone), "pad_samples": int(pad)},
    )


def wavelet_holder(
    x: Sequence[float] | NDArray[Any],
    *,
    wavelet: str = "db3",
    j_min: int = 3,
    j_max: int | None = None,
    cone: float = 1.0,
    fit_method: FitMethod = FitMethod.OLS,
    weights: Sequence[float] | FloatArray | None = None,
    trim_edges: bool = True,
    min_points: int = _MIN_POINTS,
) -> PointwiseHolder:
    r"""Pointwise exponent from the wavelet modulus along the cone of influence.

    Defining equation. With the **L2 normalisation**

    .. math::
        T(a,b) = \frac{1}{\sqrt a}\int x(u)\,
                 \overline{\psi\!\left(\frac{u-b}{a}\right)}\,\mathrm du ,

    Jaffard (1991) and Mallat & Hwang (1992) give, for a wavelet with
    :math:`N_\psi > h(t)` vanishing moments and along a modulus-maxima line
    converging to :math:`t`,

    .. math::
        |T(a,b)| \sim a^{\,h(t) + 1/2}, \qquad a \to 0 .

    **The** :math:`+1/2` **is the normalisation, not the regularity.** It is the
    :math:`a^{-1/2}` in the definition above: with the L1 normalisation
    :math:`a^{-1}\int\ldots` the same singularity gives :math:`a^{h}`. This
    function fits the L2 quantity and subtracts :math:`1/2`; the offset actually
    used is recorded in ``provenance.diagnostics["normalisation_offset"]``.

    Implementation. The transform is the undecimated dyadic ("à trous")
    transform at :math:`a=2^j`, which is precisely the discrete realisation
    Mallat & Hwang analyse, computed by :func:`pywt.swt` with ``norm=False``
    (verified to reproduce :math:`a^{h+1/2}` on the exact cusp). Instead of
    chaining modulus-maxima lines — fragile, and the business of
    :mod:`mmar.estimators.wtmm` — the estimator takes the supremum of
    :math:`|T(a,\cdot)|` over the **cone of influence**
    :math:`\{|b-t| \le \kappa a\}`. That is the standard robustification: it
    contains every maxima line converging to :math:`t`, and it removes the
    catastrophic downward spikes produced when :math:`\psi` happens to be nearly
    orthogonal to the singularity at exactly :math:`b=t`.

    Parameters
    ----------
    wavelet
        Any orthogonal PyWavelets name. ``"db3"`` (three vanishing moments)
        resolves exponents up to :math:`h<3`; use a longer filter only if
        exponents above that are expected, since the analysing support grows with
        the filter and worsens the boundary margin.
    j_min, j_max
        Octave window for the fit. ``j_max`` defaults to the smaller of
        :math:`\lfloor\log_2 n\rfloor - 3` and the coarsest octave whose
        boundary margin still leaves half the record (see
        :func:`_octave_range`); at :math:`n=2^{14}` that is 10.
        ``j_min=3`` discards the two finest octaves, where a six-tap filter on a
        sampled signal is measuring the interpolation, not the signal.
    cone
        Cone aperture :math:`\kappa`. ``0.0`` reads :math:`|T(a,t)|` itself:
        unbiased in principle, but the accidental near-zeros then dominate the
        log and both bias and dispersion blow up (measured on the
        :math:`a=0.5` cusp: :math:`\hat h = 0.343` at :math:`\kappa=0` against
        :math:`0.481` at :math:`\kappa=1`; on fBm at :math:`H=0.5` the
        within-path dispersion falls from :math:`0.197` to :math:`0.088`).
        :math:`\kappa\ge 1` saturates — the estimate is identical at
        :math:`\kappa=1` and :math:`\kappa=2` on an isolated cusp — because the
        maxima line is already inside the cone.

    Returns
    -------
    PointwiseHolder
        ``method="wavelet-maxima"``.

    Bias
    ----
    On the exact cusp :math:`|t-t_0|^{a}` at :math:`n=2^{14}` (:math:`j\in[3,10]`)
    the measured error is :math:`-0.049` at :math:`a=0.3`, :math:`-0.019` at
    :math:`0.5`, :math:`-0.008` at :math:`0.7`: a small **negative** bias from the
    curvature of :math:`\log|T|` against :math:`\log a` over the finite octaves
    available, which shrinks as the octave window moves up (at :math:`n=2^{16}`,
    :math:`j\in[3,12]`, it is :math:`-0.034 / -0.013 / -0.005`, and with
    :math:`j_{\min}=5` the :math:`a=0.3` error falls to :math:`-0.009`).

    On exact fBm at :math:`n=2^{14}` the estimator is close to unbiased: mean
    over eight seeds of the spatial mean of :math:`\hat h` minus :math:`H` is
    :math:`-0.012` at :math:`H=0.3`, :math:`-0.008` at :math:`0.5`,
    :math:`-0.008` at :math:`0.7`, with a realisation-to-realisation spread of
    about :math:`\pm0.04` (long memory makes the *spatial* mean itself a random
    variable) and a within-path dispersion of :math:`0.087`--:math:`0.090`. That
    dispersion is not error to be removed: a single fBm path genuinely has no
    more information about :math:`h(t)` at one instant than that.

    Notes
    -----
    Deterministic. :func:`pywt.swt` requires a length divisible by
    :math:`2^{j_{\max}}`; the series is reflect-padded on the right and the pad is
    both trimmed from the output and folded into the boundary margin.

    References
    ----------
    Jaffard, S. (1991). *Publicacions Matemàtiques* 35, 155--168.

    Mallat, S. and Hwang, W.L. (1992). *IEEE Trans. Inform. Theory* 38(2), 617--643.
    """
    a = validate_series(x, name="series")
    if not math.isfinite(cone) or cone < 0.0:
        raise ValueError(f"cone must be finite and non-negative, got {cone!r}")
    octaves = _octave_range(a.size, j_min, j_max)
    if octaves.size < min_points:
        raise ValueError(f"need at least {min_points} octaves, got {octaves.size}")

    mr = _wavelet_multiresolution(a, wavelet, octaves, cone)
    prov = new_provenance(
        "estimators.pointwise.wavelet",
        estimator_version=_VERSION,
        params={
            "wavelet": str(wavelet),
            "j_min": int(octaves.min()),
            "j_max": int(octaves.max()),
            "cone": float(cone),
            "fit_method": str(fit_method),
            "weights": None if weights is None else [float(v) for v in np.asarray(weights).ravel()],
            "trim_edges": bool(trim_edges),
            "min_points": int(min_points),
        },
        inputs=(InputRef.from_array("series", a),),
        seed=None,
        notes=("|T(a,b)| ~ a**(h + 1/2) under the L2 normalisation; 1/2 is subtracted",),
    )
    return _finalise(
        mr,
        fit_method=fit_method,
        weights=weights,
        trim_edges=trim_edges,
        min_points=min_points,
        prov=prov,
    )


# --------------------------------------------------------------------------- #
# 3. Wavelet leaders
# --------------------------------------------------------------------------- #


def _leader_field(
    x: FloatArray,
    wavelet: str,
    j_max: int,
    gamma: float,
) -> list[FloatArray]:
    r"""Leaders :math:`L(j,k)` for :math:`j=1..j_{\max}`, index ``j-1``.

    Two recursions, both :math:`O(n)`. First the running supremum inside one
    dyadic interval,
    :math:`M(j,k)=\max\{|d_{j,k}|, M(j-1,2k), M(j-1,2k+1)\}`, then the widening
    to the three adjacent intervals, :math:`L(j,k)=\max_{i\in\{-1,0,1\}}M(j,k+i)`.
    """
    pywt = _pywt()
    coeffs = pywt.wavedec(x, wavelet, mode="periodization", level=j_max)
    # coeffs = [cA_J, cD_J, ..., cD_1]; the L1 normalisation d = 2**(-j/2) * c
    # turns c ~ 2**(j(h+1/2)) into d ~ 2**(jh). The extra 2**(j*gamma) is the
    # fractional integration of order gamma, undone by the caller.
    d = [
        np.abs(np.asarray(coeffs[-j], dtype=np.float64)) * 2.0 ** (-j * (0.5 - gamma))
        for j in range(1, j_max + 1)
    ]
    running: list[FloatArray] = [d[0]]
    for j in range(1, j_max):
        prev = running[j - 1]
        k = prev.size // 2
        running.append(np.maximum(d[j][:k], np.maximum(prev[0 : 2 * k : 2], prev[1 : 2 * k : 2])))
    return [np.maximum(m, np.maximum(np.roll(m, 1), np.roll(m, -1))) for m in running]


def _leader_multiresolution(
    x: FloatArray,
    wavelet: str,
    octaves: NDArray[np.int64],
    gamma: float,
) -> _Multiresolution:
    j_max = int(octaves.max())
    padded, pad = _pad_to_multiple(x, 1 << j_max)
    leaders = _leader_field(padded, wavelet, j_max, gamma)

    # One exponent per finest dyadic interval (two samples wide), truncated back
    # to the unpadded record.
    n_base = min(leaders[0].size, math.ceil(x.size / 2))
    base = np.arange(n_base)
    values = np.empty((n_base, octaves.size), dtype=np.float64)
    for col, j in enumerate(octaves):
        lj = leaders[int(j) - 1]
        idx = np.minimum(base >> (int(j) - 1), lj.size - 1)
        values[:, col] = lj[idx]

    t = (base.astype(np.float64) + 0.5) * 2.0 / x.size
    # Same periodic wrap as the SWT, and the leader inherits the supremum from every
    # finer octave inside 3*lambda, so the reach is if anything wider. Expressed in
    # entries of t, which are two input samples each.
    trim = math.ceil((_WRAP_MARGIN_OCTAVES * (1 << j_max) + pad) / 2.0)
    return _Multiresolution(
        t=np.asarray(t, dtype=np.float64),
        scales=np.asarray(2.0 ** octaves.astype(np.float64), dtype=np.float64),
        values=values,
        offset=gamma,
        method="wavelet-leaders",
        n_nonpositive=int((values <= 0.0).sum()),
        trim=trim,
        trim_left=True,
        trim_right=True,
        extra={"wavelet": str(wavelet), "gamma": float(gamma), "pad_samples": int(pad)},
    )


def leader_holder(
    x: Sequence[float] | NDArray[Any],
    *,
    wavelet: str = "db3",
    j_min: int = 3,
    j_max: int | None = None,
    gamma: float = 0.5,
    fit_method: FitMethod = FitMethod.OLS,
    weights: Sequence[float] | FloatArray | None = None,
    trim_edges: bool = True,
    min_points: int = _MIN_POINTS,
) -> PointwiseHolder:
    r"""Pointwise exponent from the wavelet leaders (Jaffard 2004).

    Defining equation. Let :math:`\lambda_{j,k}=[k2^{j},(k+1)2^{j})` and let
    :math:`d_{j,k}=2^{-j/2}\langle x,\psi_{j,k}\rangle` be the **L1-normalised**
    orthonormal coefficients, so :math:`d_{j,k}\sim 2^{jh}` (this is the same
    :math:`\pm1/2` bookkeeping as in :func:`wavelet_holder`, resolved once here
    rather than at the end). With
    :math:`3\lambda_{j,k}=\lambda_{j,k-1}\cup\lambda_{j,k}\cup\lambda_{j,k+1}`
    the leader is

    .. math::
        L(j,k) \;=\; \sup_{\lambda' \subset 3\lambda_{j,k}} |d_{\lambda'}| ,

    the supremum being over **all** dyadic intervals :math:`\lambda'` at every
    scale finer than :math:`j`. Then, for :math:`t\in\lambda_{j,k_j(t)}`,

    .. math::
        h(t) \;=\; \liminf_{j\to\infty}\; \frac{\log_2 L\big(j,k_j(t)\big)}{j} ,

    and the estimator is the slope of :math:`\log_2 L` against :math:`j` along the
    nested intervals :math:`\lambda_{j,k_j(t)}\ni t`.

    Why this is the sound one. A wavelet coefficient gives only
    :math:`|d_{j,k}|\lesssim 2^{jh}`, an upper bound: the coefficient can be
    accidentally small and the decay rate can therefore *overstate* the
    regularity. The leader is a supremum over a neighbourhood at all finer
    scales, which is equivalent (up to constants) to the local oscillation of the
    :math:`N_\psi`-fold "derivative" of the signal; it bounds the exponent from
    **both** sides, so the :math:`\liminf` above is an equality, not an
    inequality (Jaffard 2004, Thm. 3). This two-sidedness is why leaders, not
    coefficients and not WTMM, underpin the modern multifractal formalism.

    Parameters
    ----------
    gamma
        Order of fractional integration, :math:`d^{(\gamma)}_{j,k} = 2^{j\gamma}
        d_{j,k}`; the fitted slope is :math:`h+\gamma` and :math:`\gamma` is
        subtracted back. **The default is non-zero on purpose.** The leader is a
        supremum over all finer scales, so it is pinned by the largest
        :math:`d_{j',k'}` beneath it. On a *sampled* signal the one or two finest
        octaves carry a discretisation transient that behaves like a negative
        local exponent, and the supremum then locks onto it, returning
        :math:`\hat h \equiv 0` regardless of the true exponent — observed
        exactly, :math:`\hat h = -0.0000` on the :math:`a=0.3` cusp with
        ``gamma=0.0``. Lifting every exponent by :math:`\gamma` restores
        monotonicity in :math:`j` and the supremum is again attained at the
        current octave. Set ``gamma=0.0`` for the textbook leader, but check
        :math:`h_{\min}>0` first (Wendt, Abry & Jaffard 2007).

        **The floor is exponent-dependent, not universal.** Measured on the cusp
        at :math:`n=2^{14}`, :math:`j\in[3,10]`
        (``test_leaders_without_fractional_integration_collapse_only_at_small_h``):

        ====  ==============  ================
        a     ``gamma=0``     ``gamma=0.5``
        ====  ==============  ================
        0.2   :math:`+0.000`  :math:`-0.094`
        0.3   :math:`+0.000`  :math:`+0.082`
        0.5   :math:`+0.245`  :math:`+0.325`
        0.7   :math:`+0.493`  :math:`+0.508`
        0.9   :math:`+0.674`  :math:`+0.674`
        ====  ==============  ================

        So the collapse bites below roughly :math:`a=0.4` and is irrelevant by
        :math:`a=0.9`, where the two settings agree exactly. Note also that at
        :math:`a=0.2` *neither* works: the default returns a negative exponent,
        which no continuous function has. Below :math:`h\approx0.25` this
        estimator has no resolution at this length, whatever :math:`\gamma`.
    j_min, j_max, wavelet, fit_method, weights, trim_edges, min_points
        As in :func:`wavelet_holder`.

    Returns
    -------
    PointwiseHolder
        ``method="wavelet-leaders"``. Resolution is one exponent per **finest
        dyadic interval** (two input samples), so ``t`` has :math:`n/2` entries,
        not :math:`n`: the leader is defined on intervals, and resampling it to
        the sample grid would advertise a resolution the estimator does not have.

    Bias
    ----
    Strongly **negative** on an isolated cusp and the worst of the three there:
    at :math:`n=2^{14}` with the default octave window :math:`j\in[3,10]` the
    measured errors are :math:`-0.218` at :math:`a=0.3`, :math:`-0.175` at
    :math:`0.5`, :math:`-0.192` at :math:`0.7`. The cause is structural, not a
    coding defect: :math:`\log_2 L(j,\cdot)` is *convex* in :math:`j` because the
    supremum takes several octaves to escape the discretisation floor, and a
    straight line through a convex curve underestimates its asymptotic slope.
    Restricting to :math:`j\ge 5` roughly halves it
    (:math:`-0.098, -0.103, -0.117`) and :math:`j\ge 6` shrinks it again
    (:math:`-0.070, -0.074, -0.085`): a finite-size, slowly converging bias, and
    the price of the two-sided bound.

    A second, honest caveat specific to the *decimated* transform: the leader
    grid is not translation invariant. Shifting the same :math:`a=0.5` cusp by
    0--3 samples moves :math:`\hat h` over :math:`0.325, 0.371, 0.442, 0.439`.
    For an isolated singularity whose position matters, prefer
    :func:`wavelet_holder`, whose undecimated transform is shift-equivariant by
    construction.

    On fBm, where the singularity is dense rather than isolated, the leaders are
    competitive: bias :math:`-0.026 / -0.016 / -0.014` at
    :math:`H=0.3/0.5/0.7` (eight seeds) with a within-path dispersion of
    :math:`0.079`--:math:`0.093`, and on the mBm ramp they give the smallest
    residual dispersion against the prescribed profile.

    Notes
    -----
    Deterministic. Uses ``mode="periodization"`` so that octave :math:`j` has
    exactly :math:`n/2^{j}` coefficients and the nesting
    :math:`k_j(t)=\lfloor t/2^{j}\rfloor` is exact; the wrap-around that this
    causes at the two ends is removed by ``trim_edges``.

    References
    ----------
    Jaffard, S. (2004). Wavelet techniques in multifractal analysis.
    *Proc. Sympos. Pure Math.* 72(2), 91--151.

    Wendt, H., Abry, P. and Jaffard, S. (2007). *IEEE Signal Processing Magazine*
    24(4), 38--48.
    """
    a = validate_series(x, name="series")
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError(f"gamma must be finite and non-negative, got {gamma!r}")
    octaves = _octave_range(a.size, j_min, j_max)
    if octaves.size < min_points:
        raise ValueError(f"need at least {min_points} octaves, got {octaves.size}")

    mr = _leader_multiresolution(a, wavelet, octaves, gamma)
    prov = new_provenance(
        "estimators.pointwise.leaders",
        estimator_version=_VERSION,
        params={
            "wavelet": str(wavelet),
            "j_min": int(octaves.min()),
            "j_max": int(octaves.max()),
            "gamma": float(gamma),
            "fit_method": str(fit_method),
            "weights": None if weights is None else [float(v) for v in np.asarray(weights).ravel()],
            "trim_edges": bool(trim_edges),
            "min_points": int(min_points),
        },
        inputs=(InputRef.from_array("series", a),),
        seed=None,
        notes=("L(j,k) ~ 2**(j*(h+gamma)) with L1-normalised coefficients; gamma subtracted",),
    )
    return _finalise(
        mr,
        fit_method=fit_method,
        weights=weights,
        trim_edges=trim_edges,
        min_points=min_points,
        prov=prov,
    )


# --------------------------------------------------------------------------- #
# Shared entry points
# --------------------------------------------------------------------------- #


def _build(
    x: FloatArray,
    method: PointwiseMethod,
    *,
    scales: Sequence[int] | NDArray[Any] | None,
    window: WindowShape,
    wavelet: str,
    j_min: int,
    j_max: int | None,
    cone: float,
    gamma: float,
) -> _Multiresolution:
    if method == "oscillation":
        raw = _default_oscillation_scales(x.size) if scales is None else scales
        return _oscillation_multiresolution(x, check_scales(raw, x.size, name="scales"), window)
    if method == "wavelet":
        return _wavelet_multiresolution(x, wavelet, _octave_range(x.size, j_min, j_max), cone)
    if method == "leaders":
        return _leader_multiresolution(x, wavelet, _octave_range(x.size, j_min, j_max), gamma)
    raise ValueError(f"unknown method {method!r}; use 'oscillation', 'wavelet' or 'leaders'")


def local_regularity_map(
    x: Sequence[float] | NDArray[Any],
    *,
    method: PointwiseMethod = "oscillation",
    scales: Sequence[int] | NDArray[Any] | None = None,
    window: WindowShape = "centred",
    wavelet: str = "db3",
    j_min: int = 3,
    j_max: int | None = None,
    cone: float = 1.0,
    gamma: float = 0.5,
) -> RegularityMap:
    r"""The full :math:`(t,\text{scale})` surface of local log--log slopes.

    No fit is performed: this is the object one *looks at* before choosing a
    scaling range. Genuine pointwise scaling shows as a horizontal band in
    :math:`j`; a crossover shows as a step; loss of locality (the analysing
    window swallowing a neighbouring singularity) shows as a drift at the coarse
    end. Reading a single exponent off a surface that is not flat is how
    "multifractality" gets manufactured.

    Returns
    -------
    RegularityMap
        ``slopes[i, j]`` is the offset-corrected slope between scales ``j`` and
        ``j+1`` at time ``t[i]``, NaN where either cell is degenerate.
    """
    a = validate_series(x, name="series")
    mr = _build(
        a,
        method,
        scales=scales,
        window=window,
        wavelet=wavelet,
        j_min=j_min,
        j_max=j_max,
        cone=cone,
        gamma=gamma,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_y = np.log(mr.values)
    log_y = np.where(np.isfinite(log_y), log_y, np.nan)
    log_a = np.log(mr.scales)
    slopes = np.diff(log_y, axis=1) / np.diff(log_a)[None, :] - mr.offset

    prov = new_provenance(
        "estimators.pointwise.regularity_map",
        estimator_version=_VERSION,
        params={
            "method": str(method),
            "window": str(window),
            "wavelet": str(wavelet),
            "j_min": int(j_min),
            "j_max": None if j_max is None else int(j_max),
            "cone": float(cone),
            "gamma": float(gamma),
            "scales": None if scales is None else [int(v) for v in np.asarray(scales).ravel()],
        },
        inputs=(InputRef.from_array("series", a),),
        seed=None,
    ).with_diagnostics(n_nonpositive_cells=int(mr.n_nonpositive), trim_samples=int(mr.trim))

    return RegularityMap(
        t=mr.t,
        scales=mr.scales,
        log_values=np.asarray(log_y, dtype=np.float64),
        slopes=np.asarray(slopes, dtype=np.float64),
        scale_midpoints=np.asarray(np.sqrt(mr.scales[:-1] * mr.scales[1:]), dtype=np.float64),
        method=mr.method,
        offset=mr.offset,
        provenance=prov,
    )


def holder_at(
    x: Sequence[float] | NDArray[Any],
    t0: float,
    *,
    method: PointwiseMethod = "oscillation",
    scales: Sequence[int] | NDArray[Any] | None = None,
    window: WindowShape = "centred",
    wavelet: str = "db3",
    j_min: int = 3,
    j_max: int | None = None,
    cone: float = 1.0,
    gamma: float = 0.5,
    fit_method: FitMethod = FitMethod.OLS,
    weights: Sequence[float] | FloatArray | None = None,
) -> tuple[float, LogLogFit]:
    r"""The exponent at one location, with the full regression diagnostics.

    Unlike the field estimators, this routes the single column through
    :func:`mmar.core.regression.loglog_fit`, so the whole :class:`FitMethod`
    menu is available (including ``THEILSEN``, the right choice when the local
    log--log plot is contaminated by a crossover) and the returned
    :class:`~mmar.core.types.LogLogFit` carries the residuals, the confidence
    interval and the scaling range. An exponent quoted without them is a number,
    not an estimate.

    Parameters
    ----------
    t0
        Location in the same normalised time as
        :attr:`~mmar.core.types.PointwiseHolder.t`, i.e. :math:`t\in(0,1]`. The
        nearest available grid point is used; for ``method="leaders"`` the grid
        is the finest dyadic interval, not the sample.

    Returns
    -------
    tuple[float, LogLogFit]
        :math:`\hat h(t_0)` (slope minus the estimator's normalisation offset)
        and the fit it came from.

    Raises
    ------
    ValueError
        If fewer than three scales survive at ``t0`` because the multiresolution
        quantity vanishes there — a locally constant signal has no measurable
        exponent, and this says so instead of returning a slope through the
        remaining points.
    """
    a = validate_series(x, name="series")
    if not math.isfinite(t0):
        raise ValueError(f"t0 must be finite, got {t0!r}")
    mr = _build(
        a,
        method,
        scales=scales,
        window=window,
        wavelet=wavelet,
        j_min=j_min,
        j_max=j_max,
        cone=cone,
        gamma=gamma,
    )
    idx = int(np.argmin(np.abs(mr.t - float(t0))))
    y = mr.values[idx]
    ok = y > 0.0
    if int(ok.sum()) < 3:
        raise ValueError(
            f"only {int(ok.sum())} of {y.size} scales have a strictly positive "
            f"{mr.method} statistic at t0={t0!r}; the signal is locally degenerate "
            "there and no pointwise exponent is defined"
        )
    w: FloatArray | None = None
    if weights is not None:
        w_all = np.asarray(weights, dtype=np.float64).ravel()
        if w_all.size != mr.scales.size:
            raise ValueError(
                f"weights has {w_all.size} entries but there are {mr.scales.size} scales"
            )
        w = w_all[ok]
    fit = loglog_fit(mr.scales[ok], y[ok], method=fit_method, weights=w)
    return float(fit.slope - mr.offset), fit
