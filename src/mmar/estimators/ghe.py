r"""Generalised Hurst exponent and the classical monofractal baselines.

This module is the *reference bench*. Every multifractal claim made elsewhere in
:mod:`mmar` has to be compared against these estimators, because they are cheap,
they have sixty years of known failure modes, and three of them (R/S, aggregated
variance, DFA) estimate a single number :math:`H` under an explicitly
monofractal model. A spectrum that does not survive the comparison --- an
:math:`f(\alpha)` width that is not larger than the spread of the panel below,
or a :math:`h(2)` that disagrees with DFA1 by more than the panel spread --- is
not evidence of multifractality.

Conventions (see :mod:`mmar.core.types`, which is law)
------------------------------------------------------
The generalised Hurst estimator natively produces a **fluctuation exponent**
:math:`H(q)`, defined by :math:`\langle|\Delta_\tau X|^q\rangle \sim
\tau^{qH(q)}`. The carrier of this library is the structure-function exponent
:math:`\zeta`, so the conversion is

.. math::
    \zeta(q) = q\,H(q), \qquad \tau_{\text{measure}}(q) = \zeta(q) - d,

and it is performed once, explicitly, in :func:`generalised_hurst`. The fitted
log--log slope *is* :math:`\zeta(q)` --- no offset is applied anywhere --- and
:math:`H(q)=\zeta(q)/q` is the derived quantity, which is why the estimator is
well defined at :math:`q=0` (where :math:`\zeta(0)=0` exactly) even though
:math:`H(0)` is not.

Anchors the tests hit:

* fBm with Hurst :math:`H`: :math:`\zeta(q)=qH`, so :math:`H(q)\equiv H` is flat.
* Binomial cascade :math:`(m_0,1-m_0)` read through its trading time:
  :math:`\zeta(q)=1-\log_2(m_0^q+m_1^q)`, so :math:`H(q)` is strictly decreasing.
* :math:`\zeta(0)=0` identically, hence :math:`\tau(0)=-d=-1`.
* Conservative measure: :math:`\tau(1)=0`, i.e. :math:`\zeta(1)=1`.

Input convention
----------------
Every entry point takes the **level** series :math:`X` (the self-affine motion:
a log-price, an fBm path, a trading time) by default, matching
:func:`mmar.models.fbm.fbm`. Estimators that are classically defined on the
increment sequence (R/S, aggregated variance, DFA, GPH) difference internally.
Pass ``series_kind="increments"`` to hand them an fGn directly. Getting this
wrong shifts every estimate by exactly one: an fGn analysed as a level returns
:math:`H-1`, not :math:`H`.

References
----------
Di Matteo, T., Aste, T. and Dacorogna, M.M. (2005). Long-term memories of
developed and emerging markets: using the scaling analysis to characterize their
stage of development. *Journal of Banking & Finance* **29**, 827--851.

Di Matteo, T. (2007). Multi-scaling in finance. *Quantitative Finance* **7**,
21--36.

Hurst, H.E. (1951). Long-term storage capacity of reservoirs.
*Transactions of the American Society of Civil Engineers* **116**, 770--808.

Mandelbrot, B.B. and Wallis, J.R. (1969). Computer experiments with fractional
Gaussian noises. *Water Resources Research* **5**, 228--267.

Anis, A.A. and Lloyd, E.H. (1976). The expected value of the adjusted rescaled
Hurst range of independent normal summands. *Biometrika* **63**, 111--116.

Peters, E.E. (1994). *Fractal Market Analysis*. Wiley, chapter 5.

Teverovsky, V., Taqqu, M.S. and Willinger, W. (1999). A critical look at Lo's
modified R/S statistic. *Journal of Statistical Planning and Inference* **80**,
211--227.

Beran, J. (1994). *Statistics for Long-Memory Processes*. Chapman & Hall
(aggregated variance, section 4.4).

Peng, C.-K., Buldyrev, S.V., Havlin, S., Simons, M., Stanley, H.E. and
Goldberger, A.L. (1994). Mosaic organization of DNA nucleotides.
*Physical Review E* **49**, 1685--1689.

Kantelhardt, J.W., Koscielny-Bunde, E., Rego, H.H.A., Havlin, S. and Bunde, A.
(2001). Detecting long-range correlations with detrended fluctuation analysis.
*Physica A* **295**, 441--454.

Geweke, J. and Porter-Hudak, S. (1983). The estimation and application of long
memory time series models. *Journal of Time Series Analysis* **4**, 221--238.

Hurvich, C.M., Deo, R. and Brodsky, J. (1998). The mean squared error of Geweke
and Porter-Hudak's estimator of the memory parameter of a long-memory time
series. *Journal of Time Series Analysis* **19**, 19--46.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.special import gammaln, logsumexp

from mmar.core.provenance import InputRef, new_provenance
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
from mmar.core.validate import check_scales, validate_q_grid, validate_series

__all__ = [
    "SeriesKind",
    "aggregated_variance",
    "anis_lloyd_expected_rs",
    "detrended_fluctuation",
    "generalised_hurst",
    "generalised_hurst_structure",
    "hurst_consensus",
    "periodogram_hurst",
    "rescaled_range",
]

SeriesKind = Literal["level", "increments"]

_VERSION = "1"

_CONSENSUS_PANEL = ("ghe_h2", "rescaled_range", "aggregated_variance", "dfa1", "gph")
"""Members whose spread defines the consensus disagreement. See :func:`hurst_consensus`."""


# --------------------------------------------------------------------------- #
# Shared plumbing
# --------------------------------------------------------------------------- #


def _level_and_increments(
    x: Sequence[float] | NDArray[Any], kind: SeriesKind, *, name: str
) -> tuple[FloatArray, FloatArray]:
    """Return ``(level, increments)`` whatever the caller handed in.

    The pair is derived once so that no estimator in this module can silently
    disagree with another about what the input was.
    """
    a = validate_series(x, name=name, min_length=64)
    if kind == "level":
        return a, np.ascontiguousarray(np.diff(a))
    if kind == "increments":
        return np.ascontiguousarray(np.concatenate(([0.0], np.cumsum(a)))), a
    raise ValueError(f"unknown series_kind {kind!r}; use 'level' or 'increments'")


def _resolve_scales(
    n: int,
    scales: Sequence[int] | NDArray[Any] | None,
    *,
    s_min: int,
    s_max: int | None,
    n_scales: int,
    min_windows: int = 4,
) -> NDArray[np.int64]:
    """Geometric integer scale grid, filtered by :func:`check_scales`.

    Geometric spacing is not cosmetic: on a linear grid the points pile up in
    the top half-decade of :math:`\\log\\delta` and an unweighted fit becomes a
    fit of the coarsest scales alone (legacy defect D5).
    """
    if scales is None:
        raw: NDArray[np.int64] = log_spaced_scales(n, s_min=s_min, s_max=s_max, n_scales=n_scales)
    else:
        raw = np.asarray(scales, dtype=np.int64).ravel()
    return check_scales(raw, n, min_windows=min_windows)


def _log_moments(values: FloatArray, q: FloatArray, *, n_total: int | None = None) -> FloatArray:
    r"""``log <|values|^q>`` for every ``q``, evaluated in the log domain.

    :math:`\log\langle|v|^q\rangle = \operatorname{logsumexp}(q\log|v|) - \log N`.
    Forming ``(|v|**q).mean()`` directly overflows float64 around
    :math:`q\log|v| \approx 709`, which for :math:`q=-5` is reached by a single
    increment of size :math:`10^{-62}`; the log-domain form never does.

    ``n_total`` is the size of the sample *before* exact zeros were removed, and
    it is the divisor used for :math:`q>0`. This is not bookkeeping pedantry: a
    zero contributes :math:`|0|^q=0` to a positive moment, so the population mean
    is over the whole sample, and dividing by the surviving count instead
    multiplies :math:`\langle|\Delta_\tau X|^q\rangle` by
    :math:`N_{\text{tot}}(\tau)/N_{\text{nz}}(\tau)`. On a tick-discretised price
    the zero fraction *falls with* :math:`\tau`, so that ratio is a power of
    :math:`\tau` and the artefact lands directly in the slope --- the same
    :math:`\Delta` subtracted from every :math:`\zeta(q)`, hence
    :math:`-\Delta/q` added to :math:`h(q)`, which is a manufactured, perfectly
    smooth multiscaling curve on a process that is exactly monofractal.

    For :math:`q\le 0` no such repair exists: with an atom at zero the moment
    :math:`\mathbb E|v|^{q}` is infinite, and the zero-free subsample is the only
    computable object. It is used, and the divisor is then the surviving count so
    that :math:`K_0(\tau)\equiv1` and :math:`\zeta(0)=0` stay exact.

    The reduction runs one :math:`q` at a time. Broadcasting to the full
    ``(n_q, n)`` tilt matrix instead is bit-for-bit identical but materialises
    it, and :func:`scipy.special.logsumexp` keeps several copies alive: on
    :math:`n=2^{20}` with the default 41-point grid that peaks at 2.1 GB against
    the 8 MB the series itself occupies, for no gain in speed.
    """
    kept = values.size
    total = kept if n_total is None else int(n_total)
    with np.errstate(divide="ignore"):
        lv = np.log(np.abs(values))
    ls = np.array([logsumexp(float(qi) * lv) for qi in q], dtype=np.float64)
    log_n = np.where(q > 0.0, math.log(total), math.log(kept))
    return np.asarray(ls - log_n, dtype=np.float64)


def _increments_at_lag(x: FloatArray, tau: int, *, overlapping: bool) -> FloatArray:
    r""":math:`X(t+\tau)-X(t)`, either at every :math:`t` or on disjoint blocks.

    Disjoint blocks start at index zero, so on a :math:`b`-adic trading time
    with :math:`\tau=b^{j}` the increments are *exactly* the level-:math:`j` box
    masses and the partition function is exact rather than asymptotic.
    """
    if overlapping:
        return np.ascontiguousarray(x[tau:] - x[:-tau])
    k = (x.size - 1) // tau
    idx = np.arange(k + 1, dtype=np.int64) * tau
    return np.ascontiguousarray(x[idx[1:]] - x[idx[:-1]])


def _independent_windows(n: int, s: int) -> int:
    """Number of *disjoint* windows of length ``s``; the honest weight at scale ``s``.

    Overlapping increments raise the number of terms averaged but not the
    information content: neighbouring increments of a long-memory process share
    :math:`\\tau-1` of their :math:`\\tau` summands. Weighting by the overlapping
    count would claim an accuracy the sample does not have.
    """
    return max(int(n // s), 1)


# --------------------------------------------------------------------------- #
# Generalised Hurst exponent
# --------------------------------------------------------------------------- #


def _ghe_surface(
    level: FloatArray,
    q: FloatArray,
    scales: NDArray[np.int64],
    *,
    normalise: bool,
    overlapping: bool,
    drop_zeros: bool,
) -> tuple[FloatArray, NDArray[np.int64], dict[str, Any]]:
    r"""Log of :math:`K_q(\tau)` on the ``(q, tau)`` grid, plus window counts.

    Returns the *logarithm* because the fit consumes it directly: exponentiating
    and re-logging loses the large-:math:`|q|` tails to under/overflow.
    """
    n = level.size
    log_k = np.empty((q.size, scales.size), dtype=np.float64)
    counts = np.empty(scales.size, dtype=np.int64)
    zeros_increments = 0
    zeros_level = 0

    if normalise:
        lvl = level
        if drop_zeros:
            nz = lvl != 0.0
            zeros_level = int((~nz).sum())
            lvl = lvl[nz]
        if lvl.size == 0:
            raise ValueError("every level value is zero; the GHE normalisation is undefined")
        log_denominator = _log_moments(lvl, q, n_total=level.size)
    else:
        log_denominator = np.zeros(q.size, dtype=np.float64)

    zero_fraction: list[float] = []
    for j, tau in enumerate(scales):
        inc = _increments_at_lag(level, int(tau), overlapping=overlapping)
        n_at_tau = inc.size
        nz = inc != 0.0
        n_zero = int((~nz).sum())
        # Reported whether or not they are dropped: the diagnostic exists so a
        # reader can see how the zero fraction varies with tau, which is what
        # decides whether the negative-q exponents mean anything.
        zero_fraction.append(n_zero / n_at_tau)
        if drop_zeros:
            zeros_increments += n_zero
            inc = inc[nz]
        if inc.size < 2:
            raise ValueError(
                f"only {inc.size} non-zero increments survive at lag tau={int(tau)}; "
                "the series is (locally) constant and no moment can be estimated there"
            )
        log_k[:, j] = _log_moments(inc, q, n_total=n_at_tau) - log_denominator
        counts[j] = _independent_windows(n, int(tau))

    diagnostics: dict[str, Any] = {
        "n_zero_increments_dropped": zeros_increments,
        "n_zero_levels_dropped": zeros_level,
        "n_samples": int(n),
        "overlapping_increments": bool(overlapping),
        # The zero fraction as a function of tau. It is reported per scale rather
        # than as a total because only its *variation with tau* matters: a
        # constant fraction is harmless, a falling one (any tick-discretised
        # price) is what corrupts the negative-q moments the repair cannot reach.
        "zero_fraction_by_scale": [round(float(v), 6) for v in zero_fraction],
        "max_zero_fraction": round(float(max(zero_fraction, default=0.0)), 6),
    }
    return log_k, counts, diagnostics


def generalised_hurst_structure(
    x: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    tau_min: int = 2,
    tau_max: int = 19,
    n_scales: int = 24,
    normalise: bool = True,
    overlapping: bool = True,
    drop_zeros: bool = True,
    series_kind: SeriesKind = "level",
) -> StructureFunctions:
    r"""The raw GHE scaling surface :math:`K_q(\tau)`, with no exponent fitted.

    .. math::
        K_q(\tau) \;=\;
        \frac{\big\langle |X(t+\tau)-X(t)|^{\,q}\big\rangle_t}
             {\big\langle |X(t)|^{\,q}\big\rangle_t}
        \;\sim\; \tau^{\,q H(q)} .

    Exposed separately from :func:`generalised_hurst` so that exponents can be
    refitted over a different scaling range, with a different weighting, or with
    a robust regression, without recomputing the moments --- which is the
    expensive part and the part that must not change when only the fit changes.

    Parameters
    ----------
    x
        Level series (or increments, with ``series_kind="increments"``).
    q
        Moment grid. Defaults to 41 points on :math:`[-5,5]`; **the negative
        half is not optional**, it is the only source of the right branch of
        :math:`f(\alpha)`.
    scales
        Explicit integer lags :math:`\tau`. Defaults to a geometric grid on
        :math:`[\tau_{\min},\tau_{\max}]`.
    normalise
        Divide by :math:`\langle|X|^q\rangle`; see :func:`generalised_hurst` for
        what that does and, more importantly, what it does not do.
    overlapping
        Average over every :math:`t` (Di Matteo's definition) or over disjoint
        blocks only. Disjoint blocks are exact on a :math:`b`-adic measure.
    drop_zeros
        Exact zeros make :math:`|v|^{q}` infinite for :math:`q<0`. They are
        removed --- from the increments and from the level --- and the number
        removed, together with the zero fraction at every :math:`\tau`, is
        recorded in ``provenance.diagnostics``. Nothing is dropped silently.
        The :math:`q>0` moments are still divided by the *full* count, because a
        zero contributes :math:`|0|^q=0` there; see :func:`generalised_hurst` for
        why the alternative fabricates multiscaling on a tick-discretised price.

    Returns
    -------
    StructureFunctions
        ``values[i, j]`` is :math:`K_{q_i}(\tau_j)` in linear units. Very large
        :math:`|q|` can push these past float64 range; the fit inside
        :func:`generalised_hurst` uses the log-moments directly and is
        unaffected, and ``provenance.diagnostics['n_nonfinite_values']`` reports
        the loss.
    """
    level, _ = _level_and_increments(x, series_kind, name="series")
    qa = validate_q_grid(q)
    sc = _resolve_scales(
        level.size, scales, s_min=int(tau_min), s_max=int(tau_max), n_scales=n_scales
    )
    log_k, counts, diag = _ghe_surface(
        level, qa, sc, normalise=normalise, overlapping=overlapping, drop_zeros=drop_zeros
    )
    with np.errstate(over="ignore", under="ignore"):
        values = np.asarray(np.exp(log_k), dtype=np.float64)
    diag["n_nonfinite_values"] = int((~np.isfinite(values)).sum())

    prov = new_provenance(
        "ghe.structure",
        estimator_version=_VERSION,
        params={
            "tau_min": int(tau_min),
            "tau_max": int(tau_max),
            "n_scales": int(n_scales),
            "normalise": bool(normalise),
            "overlapping": bool(overlapping),
            "drop_zeros": bool(drop_zeros),
            "series_kind": str(series_kind),
        },
        inputs=(
            InputRef.from_array("x", np.asarray(x, dtype=np.float64)),
            InputRef.from_array("q", qa),
            InputRef.from_array("scales", sc),
        ),
        notes=("K_q(tau) = <|dX|^q>/<|X|^q> ~ tau^(q H(q)); slope of log K = zeta(q)",),
    ).with_diagnostics(**diag)

    return StructureFunctions(
        q=qa,
        scales=np.asarray(sc, dtype=np.float64),
        values=values,
        counts=counts,
        provenance=prov,
    )


def _cutoff_indices(
    n_scales: int, min_points: int, grid: Sequence[int] | None, scales: NDArray[np.int64]
) -> list[int]:
    """Indices of the upper cut-offs used by the :math:`\\tau_{\\max}` averaging device."""
    if grid is not None:
        wanted = np.asarray(grid, dtype=np.int64).ravel()
        idx = [int(np.searchsorted(scales, w, side="right")) - 1 for w in wanted]
        keep = sorted({i for i in idx if i >= min_points - 1})
        if not keep:
            raise ValueError(
                f"no entry of tau_max_grid={list(wanted)} leaves at least min_scales_per_fit="
                f"{min_points} scales inside the grid {scales.tolist()}"
            )
        return keep
    lo = min(min_points - 1, n_scales - 1)
    return list(range(lo, n_scales))


def generalised_hurst(
    x: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    tau_min: int = 2,
    tau_max: int = 19,
    n_scales: int = 24,
    tau_max_grid: Sequence[int] | None = None,
    average_tau_max: bool = True,
    min_scales_per_fit: int = 5,
    normalise: bool = True,
    overlapping: bool = True,
    drop_zeros: bool = True,
    series_kind: SeriesKind = "level",
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
    d_support: float = 1.0,
    seed: SeedLike = None,
) -> ScalingExponents:
    r"""Generalised Hurst exponent (Di Matteo, Aste & Dacorogna 2005).

    **Definition.** For a level series :math:`X` and integer lag :math:`\tau`,

    .. math::
        K_q(\tau) \;=\;
        \frac{\big\langle |X(t+\tau)-X(t)|^{\,q}\big\rangle_t}
             {\big\langle |X(t)|^{\,q}\big\rangle_t}
        \;\sim\; \tau^{\,q H(q)},

    and :math:`H(q)` is the log--log slope divided by :math:`q`. A process is
    *uniscaling* iff :math:`H(q)` is constant --- :math:`H(q)\equiv H` for fBm ---
    and *multiscaling* iff :math:`H(q)` decreases; :math:`H(2)` is the classical
    Hurst exponent and :math:`qH(q)` at :math:`q=1` is the exponent Di Matteo
    uses to rank markets by development stage.

    **Which convention this produces.** The fitted slope of
    :math:`\log K_q` against :math:`\log\tau` is a moment of increments, i.e.
    the *structure-function* exponent :math:`\zeta(q)=qH(q)` of
    :mod:`mmar.core.types`. It is stored as ``zeta`` unmodified; the measure
    exponent is the derived ``tau = zeta - d_support``, and ``hq = zeta/q``.
    No offset is added anywhere.

    **The normalisation, honestly.** :math:`\langle|X(t)|^q\rangle` does not
    depend on :math:`\tau`. Dividing by it therefore shifts
    :math:`\log K_q(\tau)` by a :math:`\tau`-independent constant and is
    **exactly neutral for** :math:`H(q)`: it changes the intercept of every
    log--log fit and nothing else. What it genuinely buys is threefold, and none
    of it concerns the exponent:

    1. :math:`K_q` is dimensionless, so the *amplitude* is comparable across
       assets quoted in different units and at different price levels --- the
       reason Di Matteo can pool 32 markets in one table.
    2. Numerator and denominator carry the same power of the series' scale, so
       the ratio stays :math:`O(1)` even at :math:`|q|=5`, where the raw moment
       of a series with increments of order :math:`10^{-4}` is :math:`10^{20}`.
       This is a conditioning gain, not a statistical one.
    3. It fixes a reference level at which the log--log line is anchored, which
       makes the intercept interpretable as a volatility scale.

    Any claim that the normalisation makes the *exponent* robust is false, and
    it is checkable: run with ``normalise=False`` and the ``zeta`` array is
    identical to float round-off.

    **The** :math:`\tau_{\max}` **device.** Di Matteo's own recommendation
    (2005, §3; 2007, §2) is not to trust one upper cut-off. With
    ``average_tau_max=True`` the fit is repeated over every nested window
    :math:`[\tau_{\min},\tau_c]` with at least ``min_scales_per_fit`` scales,
    and :math:`\zeta(q)` is the mean of the slopes while their standard
    deviation is reported as the cut-off sensitivity. Averaging the slopes and
    averaging :math:`H(q)=\text{slope}/q` are the same operation, so this is
    well defined at :math:`q=0` too.

    ``zeta_stderr`` is the quadrature sum of (i) the weighted-regression
    standard error on the full window and (ii) that cut-off spread. The two
    components are recorded separately in ``provenance.diagnostics``. The spread
    is *not* a sampling standard error --- the nested fits share almost all
    their data and are strongly dependent --- it is a sensitivity, and it
    captures exactly the crossover risk that a single-window standard error is
    blind to. Reporting their quadrature sum is deliberately conservative.

    **Known bias, and its causes.**

    * *Moment truncation.* For :math:`|q|` beyond the largest finite moment the
      sample :math:`\langle|\Delta X|^q\rangle` is dominated by a handful of
      extreme increments; the log--log plot stays straight and :math:`H(q)`
      bends *downward*, mimicking multifractality. On an :math:`\alpha`-stable
      Lévy motion --- monofractal by construction --- this estimator returns
      :math:`\zeta(q)=\min(q/\alpha,1)`, a textbook false positive. Keep
      :math:`|q|\le 5` and check moment existence separately.
    * *Short cut-off.* The default :math:`\tau\le 19` is Di Matteo's, chosen for
      daily financial series of a few thousand points. Over 0.98 decades any
      short-range dependence (bid--ask bounce, AR(1) microstructure) is not
      asymptotically negligible and biases :math:`H(2)` toward :math:`1/2` for
      persistent series and away from it for anti-persistent ones. On long
      series raise ``tau_max``.
    * *Drift.* A non-zero mean increment :math:`\mu` makes
      :math:`\Delta_\tau X \approx \mu\tau` at large :math:`\tau`, driving
      :math:`H(q)\to 1`. Demean, or keep :math:`\tau` short.
    * *Overlap.* Overlapping increments correlate the moments across :math:`t`,
      so the regression standard error --- computed as if the scales were
      independent --- understates the sampling error. The weights use the number
      of *disjoint* windows :math:`n/\tau` precisely to avoid compounding this.

    **Negative** :math:`q`, **explicitly.** The right branch of
    :math:`f(\alpha)` --- everything with :math:`\alpha>\alpha_0` --- is
    generated by :math:`q<0` and by nothing else, which is why the default grid
    is symmetric (legacy defect D2 recovered 24 % of a known support by taking
    :math:`q\in[0,3]`). Negative moments are also where this estimator is most
    fragile, in two distinct ways that must not be conflated:

    1. *Numerical.* :math:`|\Delta_\tau X|^{q}` is infinite at an exactly zero
       increment. Such increments are removed --- from the numerator and from
       the level in the denominator --- and the counts appear as
       ``n_zero_increments_dropped`` and ``n_zero_levels_dropped`` in
       ``provenance.diagnostics``. Nothing is dropped silently. Moments are then
       accumulated by :func:`scipy.special.logsumexp`, so a merely *small*
       increment costs accuracy but never overflows.

       **The divisor is the trap, not the dropping.** For :math:`q>0` the
       positive moment is a mean over the *whole* sample --- a zero contributes
       :math:`0` --- so the surviving values are divided by
       :math:`N_{\text{tot}}(\tau)`, not by :math:`N_{\text{nz}}(\tau)`.
       Dividing by the surviving count instead multiplies :math:`K_q(\tau)` by
       :math:`N_{\text{tot}}(\tau)/N_{\text{nz}}(\tau)`, and on any
       tick-discretised price that ratio is itself a power of :math:`\tau`
       (the zero fraction falls as the lag grows). The artefact is then a single
       constant :math:`\Delta` subtracted from *every* :math:`\zeta(q)`, i.e.
       :math:`-\Delta/q` added to :math:`h(q)` --- a smooth, monotone, entirely
       spurious multiscaling curve. Measured on a Gaussian random walk rounded
       to a coarse tick (76 % zero increments at :math:`\tau=1`, :math:`n=2^{14}`):
       with the wrong divisor :math:`\Delta=-0.39`, :math:`h(q)` runs from
       :math:`0.103` at :math:`q=0.5` to :math:`0.279` at :math:`q=5` and
       ``is_monofractal`` returns ``False``; with the divisor above,
       :math:`h(q)` is flat again. ``zero_fraction_by_scale`` in
       ``provenance.diagnostics`` is what lets a reader check this.

       For :math:`q\le0` there is no repair: with an atom at zero the moment
       does not exist, and the zero-free subsample --- divided by its own count,
       which is what keeps :math:`\zeta(0)=0` exact --- is the only computable
       object. Read ``max_zero_fraction`` before trusting any negative-\ :math:`q`
       exponent.
    2. *Statistical, and not fixable here.* For a **Gaussian** process
       :math:`\mathbb{E}|Z|^q` diverges for :math:`q\le-1`, because the density
       is positive at the origin. On fBm the sample :math:`\hat H(q)` therefore
       collapses as :math:`q\to-1^+` --- measured here as :math:`h(-0.9)=0.644`
       against :math:`h(-0.5)=0.690` at a true :math:`H=0.687` --- and below
       :math:`q=-1` it estimates a quantity that does not exist. Multiplicative
       cascades have multipliers bounded away from zero and suffer none of this,
       which is exactly why the default grid reaching :math:`q=-5` is right for
       a measure and wrong for an fBm null. Restrict to :math:`q>-1` when the
       object under test may be Gaussian.

    Parameters
    ----------
    q
        Moment grid; default 41 points on :math:`[-5,5]`, negative half included.
    tau_min, tau_max, n_scales, scales
        The lag grid. ``tau_min`` is 2 rather than Di Matteo's 1 because
        :func:`mmar.core.validate.check_scales` refuses scales below 2: a
        one-sample lag has no window structure and cannot be weighted.
    tau_max_grid
        Explicit cut-offs for the averaging device (e.g. ``range(5, 20)``).
        Each is snapped down to the nearest available scale.
    select_range
        Choose the scaling window automatically. The window is selected **once**,
        on the row nearest :math:`q=2`, and then used for every :math:`q`: a
        scaling range that varies with :math:`q` produces a :math:`\hat\zeta`
        assembled from different physics at each moment order.
    seed
        Unused --- this estimator is deterministic --- and accepted only so that
        every estimator in the library shares one signature. It is recorded in
        provenance when explicitly supplied, and left ``None`` otherwise so that
        the provenance root of a default call is reproducible.

    Returns
    -------
    ScalingExponents
        With ``estimator="ghe"``, one :class:`~mmar.core.types.LogLogFit` per
        :math:`q` over the full window, and the complete
        :class:`~mmar.core.types.StructureFunctions` attached.
    """
    level, _ = _level_and_increments(x, series_kind, name="series")
    qa = validate_q_grid(q)
    sc = _resolve_scales(
        level.size, scales, s_min=int(tau_min), s_max=int(tau_max), n_scales=n_scales
    )
    log_k, counts, diag = _ghe_surface(
        level, qa, sc, normalise=normalise, overlapping=overlapping, drop_zeros=drop_zeros
    )

    sc_f = np.asarray(sc, dtype=np.float64)
    weights = weighted_by_counts(counts)

    mask = np.ones(sc.size, dtype=bool)
    selection = "manual"
    if select_range:
        pivot = int(np.argmin(np.abs(qa - 2.0)))
        sel = select_scaling_range(
            sc_f,
            log_k[pivot],
            min_points=max(min_scales_per_fit, 3),
            weights=weights,
            method=fit_method,
            log_y=False,
        )
        mask = np.asarray(sel.mask, dtype=bool)
        selection = f"{sel.scaling_range.selection}@q={qa[pivot]:.3g}"
    sc_fit = sc_f[mask]
    w_fit = weights[mask]
    log_fit = log_k[:, mask]
    if sc_fit.size < 3:
        raise ValueError(f"only {sc_fit.size} scales survive selection; a slope needs at least 3")

    cutoffs = _cutoff_indices(sc_fit.size, min_scales_per_fit, tau_max_grid, sc[mask])
    averaged = bool(average_tau_max) and len(cutoffs) > 1

    zeta = np.empty(qa.size, dtype=np.float64)
    reg_se = np.empty(qa.size, dtype=np.float64)
    cut_spread = np.zeros(qa.size, dtype=np.float64)
    fits: list[LogLogFit] = []

    for i in range(qa.size):
        full = loglog_fit(sc_fit, log_fit[i], method=fit_method, weights=w_fit, log_y=False)
        fits.append(full)
        reg_se[i] = full.slope_stderr
        if averaged:
            slopes = np.array(
                [
                    loglog_fit(
                        sc_fit[: c + 1],
                        log_fit[i, : c + 1],
                        method=fit_method,
                        weights=w_fit[: c + 1],
                        log_y=False,
                    ).slope
                    for c in cutoffs
                ],
                dtype=np.float64,
            )
            zeta[i] = float(slopes.mean())
            cut_spread[i] = float(slopes.std(ddof=1))
        else:
            zeta[i] = full.slope

    stderr = np.sqrt(np.where(np.isfinite(reg_se), reg_se, 0.0) ** 2 + cut_spread**2)
    dof = max(sc_fit.size - 2, 1)
    t_crit = float(stats.t.ppf(0.975, dof))
    ci = (zeta - t_crit * stderr, zeta + t_crit * stderr)

    scaling_range = ScalingRange(
        s_min=float(sc_fit.min()),
        s_max=float(sc_fit.max()),
        n_scales=int(sc_fit.size),
        selection=selection,
        # nanmedian, not median: K_0(tau) = 1 bit-exactly (the q <= 0 divisor is the
        # surviving count), so the q = 0 row has zero total sum of squares and its
        # R^2 is NaN by construction. A plain median would poison the fit-quality
        # summary of every grid containing q = 0, the library default included.
        r_squared=float(np.nanmedian([f.r_squared for f in fits])),
        mask=tuple(bool(m) for m in mask),
    )

    params: dict[str, Any] = {
        "tau_min": int(tau_min),
        "tau_max": int(tau_max),
        "n_scales": int(n_scales),
        "tau_max_grid": None if tau_max_grid is None else [int(t) for t in tau_max_grid],
        "average_tau_max": bool(average_tau_max),
        "min_scales_per_fit": int(min_scales_per_fit),
        "normalise": bool(normalise),
        "overlapping": bool(overlapping),
        "drop_zeros": bool(drop_zeros),
        "series_kind": str(series_kind),
        "fit_method": str(fit_method),
        "select_range": bool(select_range),
        "d_support": float(d_support),
    }
    with np.errstate(over="ignore", under="ignore"):
        values = np.asarray(np.exp(log_k), dtype=np.float64)

    prov = new_provenance(
        "ghe.generalised_hurst",
        estimator_version=_VERSION,
        params=params,
        inputs=(
            InputRef.from_array("x", np.asarray(x, dtype=np.float64)),
            InputRef.from_array("q", qa),
            InputRef.from_array("scales", sc),
        ),
        seed=seed_info(seed) if seed is not None else None,
        notes=(
            "native exponent: zeta(q) = q*H(q) (structure-function convention); "
            "tau(q) = zeta(q) - d_support",
            "the <|X|^q> normalisation is tau-independent and therefore exactly "
            "slope-neutral; it fixes the intercept only",
        ),
    ).with_diagnostics(
        **diag,
        tau_max_cutoffs=[int(sc[mask][c]) for c in cutoffs],
        averaged_over_tau_max=averaged,
        regression_stderr=[float(v) for v in reg_se],
        cutoff_spread=[float(v) for v in cut_spread],
        n_nonfinite_values=int((~np.isfinite(values)).sum()),
    )

    structure = StructureFunctions(q=qa, scales=sc_f, values=values, counts=counts, provenance=prov)

    return ScalingExponents(
        q=qa,
        zeta=zeta,
        d_support=float(d_support),
        zeta_stderr=stderr,
        zeta_ci=ci,
        fits=tuple(fits),
        scaling_range=scaling_range,
        structure=structure,
        estimator="ghe",
        provenance=prov,
    )


# --------------------------------------------------------------------------- #
# Rescaled range
# --------------------------------------------------------------------------- #


def anis_lloyd_expected_rs(n: int) -> float:
    r"""Expected rescaled range of :math:`n` iid Gaussian summands.

    Anis and Lloyd (1976):

    .. math::
        \mathbb{E}\big[(R/S)_n\big] =
        \frac{n-\tfrac12}{n}\,
        \frac{\Gamma\!\big(\frac{n-1}{2}\big)}{\sqrt\pi\,\Gamma\!\big(\frac n2\big)}
        \sum_{i=1}^{n-1}\sqrt{\frac{n-i}{i}},

    with the gamma ratio replaced by its Stirling limit
    :math:`(n\pi/2)^{-1/2}` for :math:`n>340`, where the ratio itself overflows.
    The leading behaviour is :math:`\sqrt{n\pi/2}`, i.e. slope :math:`1/2`; the
    :math:`O(n^{-1/2})` correction is what makes the *finite-sample* slope
    exceed :math:`1/2` and is the whole reason for the Peters correction in
    :func:`rescaled_range`.

    Evaluated with :func:`scipy.special.gammaln`, so no branch overflows.
    """
    m = int(n)
    if m < 2:
        raise ValueError(f"the expected rescaled range needs n >= 2, got {n!r}")
    i = np.arange(1, m, dtype=np.float64)
    tail = float(np.sum(np.sqrt((m - i) / i)))
    if m <= 340:
        log_ratio = gammaln(0.5 * (m - 1)) - 0.5 * math.log(math.pi) - gammaln(0.5 * m)
        front = math.exp(log_ratio)
    else:
        front = 1.0 / math.sqrt(0.5 * math.pi * m)
    return (m - 0.5) / m * front * tail


def _rs_at_scale(increments: FloatArray, s: int) -> tuple[float, int]:
    r"""Mean :math:`R/S` over the disjoint windows of length ``s``, and their count.

    Within a window :math:`w`, :math:`Z_k=\sum_{j\le k}(w_j-\bar w)` is the
    mean-adjusted partial sum, :math:`R=\max Z-\min Z` and :math:`S` the
    population standard deviation. Constant windows have :math:`S=0` and are
    dropped, which is reported by the caller.
    """
    k = increments.size // s
    blocks = increments[: k * s].reshape(k, s)
    dev = blocks - blocks.mean(axis=1, keepdims=True)
    z = np.cumsum(dev, axis=1)
    r = z.max(axis=1) - z.min(axis=1)
    sd = blocks.std(axis=1)
    ok = sd > 0.0
    if not ok.any():
        raise ValueError(f"every window of length {s} is constant; R/S is undefined there")
    return float(np.mean(r[ok] / sd[ok])), int(ok.sum())


def rescaled_range(
    x: Sequence[float] | NDArray[Any],
    *,
    scales: Sequence[int] | NDArray[Any] | None = None,
    corrected: bool = True,
    s_min: int = 8,
    s_max: int | None = None,
    n_scales: int = 24,
    series_kind: SeriesKind = "level",
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
) -> LogLogFit:
    r"""Classical rescaled-range analysis (Hurst 1951; Mandelbrot & Wallis 1969).

    For disjoint windows of length :math:`s` of the increment sequence
    :math:`\{w_j\}`, with :math:`\bar w` the window mean,

    .. math::
        R(s) = \max_{1\le k\le s}\sum_{j=1}^{k}(w_j-\bar w)
             - \min_{1\le k\le s}\sum_{j=1}^{k}(w_j-\bar w),
        \qquad
        S(s) = \Big(\tfrac1s\sum_j (w_j-\bar w)^2\Big)^{1/2},

    and :math:`\langle R/S\rangle_s \sim c\,s^{H}`. **The returned slope is
    :math:`H` directly** (no conversion), which is the estimator's whole appeal:
    it is scale-free, distribution-free, and needs no spectral assumption.

    **Known bias, and its cause.** For iid Gaussian input the *exact* finite-:math:`s`
    expectation is not :math:`\sqrt{s\pi/2}` but the Anis--Lloyd formula
    :func:`anis_lloyd_expected_rs`, whose :math:`O(s^{-1/2})` correction is
    negative and decays; regressing the uncorrected statistic therefore returns
    :math:`\hat H>1/2` on white noise, by roughly :math:`0.05`--:math:`0.1` at
    the window lengths a few thousand points allow. This is the single
    best-known artefact in the long-memory literature and the reason a bare R/S
    "Hurst exponent" of 0.6 is worthless as evidence of persistence. With
    ``corrected=True`` the Peters (1994) adjustment

    .. math::
        (R/S)^{\text{corr}}(s) = (R/S)(s) - \mathbb{E}[(R/S)_s] + \sqrt{s\pi/2}

    restores the asymptotic null so that iid input regresses to :math:`H\approx1/2`.

    **The correction has its own bias, and it is worse than the one it fixes.**
    :math:`\mathbb{E}[(R/S)_s]` is the expectation under *iid* summands, so the
    adjustment is exact only at :math:`H=1/2` and over-corrects away from it,
    pulling :math:`\hat H` toward :math:`1/2`. Measured on 30 replicas of exact
    fBm at :math:`n=2^{14}` (``tests/unit/test_ghe.py``):

    ======  ==================  ==================
    True H  bias, uncorrected   bias, corrected
    ======  ==================  ==================
    0.20    :math:`+0.150`      :math:`+0.079`
    0.35    :math:`+0.111`      :math:`+0.030`
    0.50    :math:`+0.074`      :math:`-0.013`
    0.65    :math:`+0.031`      :math:`-0.059`
    0.80    :math:`-0.023`      :math:`-0.113`
    0.95    :math:`-0.092`      :math:`-0.182`
    ======  ==================  ==================

    The correction is therefore the right choice for *testing the null* --- it
    is what makes "R/S says 0.5" mean something --- and the wrong choice for
    *measuring* an :math:`H` already known to be far from :math:`1/2`. Neither
    column is usable as a point estimate at :math:`H\ge0.8`; use DFA1
    (bias :math:`\le0.023` everywhere) and keep R/S as a corroborating,
    distribution-free member of the panel.

    R/S is additionally **sensitive to short-range dependence**: an AR(1) with
    :math:`\phi=0.5` inflates :math:`\hat H` well above :math:`1/2` at small
    :math:`s` even with no long memory at all (Teverovsky, Taqqu & Willinger
    1999). Nothing in this function detects that; it is why R/S belongs in a
    panel and not on its own.

    Parameters
    ----------
    corrected
        Apply the Anis--Lloyd/Peters small-sample correction. Default ``True``;
        the uncorrected statistic is available for reproducing legacy results
        and for measuring the bias itself.
    fit_method
        Weighted least squares by default, with weights equal to the number of
        disjoint windows: at :math:`s=n/4` the mean :math:`R/S` is an average of
        four numbers and must not carry the same weight as the average of a
        thousand.

    Returns
    -------
    LogLogFit
        ``slope`` is :math:`\hat H`.
    """
    _, inc = _level_and_increments(x, series_kind, name="series")
    sc = _resolve_scales(inc.size, scales, s_min=s_min, s_max=s_max, n_scales=n_scales)

    rs = np.empty(sc.size, dtype=np.float64)
    counts = np.empty(sc.size, dtype=np.int64)
    for j, s in enumerate(sc):
        rs[j], counts[j] = _rs_at_scale(inc, int(s))

    if corrected:
        expected = np.array([anis_lloyd_expected_rs(int(s)) for s in sc], dtype=np.float64)
        asymptotic = np.sqrt(0.5 * math.pi * sc.astype(np.float64))
        rs = rs - expected + asymptotic
        if np.any(rs <= 0.0):
            raise ValueError(
                f"the Anis-Lloyd/Peters correction drove {int((rs <= 0).sum())} of "
                f"{rs.size} rescaled ranges non-positive; the series is far below the "
                "iid null at those scales and the correction does not apply. "
                "Re-run with corrected=False and report the raw bias instead."
            )

    weights = weighted_by_counts(counts)
    if select_range:
        return select_scaling_range(
            sc.astype(np.float64), rs, weights=weights, method=fit_method
        ).fit
    return loglog_fit(sc.astype(np.float64), rs, method=fit_method, weights=weights)


# --------------------------------------------------------------------------- #
# Aggregated variance
# --------------------------------------------------------------------------- #


def aggregated_variance(
    x: Sequence[float] | NDArray[Any],
    *,
    scales: Sequence[int] | NDArray[Any] | None = None,
    s_min: int = 8,
    s_max: int | None = None,
    n_scales: int = 24,
    series_kind: SeriesKind = "level",
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
) -> LogLogFit:
    r"""Variance of block means: :math:`\operatorname{Var}\bar w^{(m)} \sim m^{2H-2}`.

    Split the increment sequence into disjoint blocks of length :math:`m`, take
    the block means :math:`\bar w^{(m)}_k = m^{-1}\sum_{j} w_{(k-1)m+j}`, and
    regress their sample variance on :math:`m`. For a second-order self-similar
    process with Hurst exponent :math:`H`,

    .. math::
        \operatorname{Var}\big[\bar w^{(m)}\big] = \sigma^2 m^{2H-2},

    directly from :math:`\operatorname{Var}\big[\sum_{j\le m} w_j\big]
    = \sigma^2 m^{2H}`, which is self-similarity itself. **The slope is
    :math:`2H-2`, so** :math:`H = 1 + \text{slope}/2` --- this is the conversion
    most often botched, and it is applied here only inside
    :func:`hurst_consensus`, never silently inside the returned fit.

    **Known bias, and its cause.** The estimator is a variance, so it is
    quadratic in the data and therefore *not* robust: a single jump, a level
    shift, or a slowly varying trend adds a term to the block variance that
    decays like :math:`m^{0}` rather than :math:`m^{2H-2}`, flattening the
    log--log line and driving :math:`\hat H\to 1`. Beran (1994, §4.4) recommends
    reading it only jointly with a *differenced* variant for exactly this
    reason. Conversely, additive white measurement noise contributes
    :math:`\sigma_\varepsilon^2 m^{-1}` and pulls :math:`\hat H` toward
    :math:`1/2` at small :math:`m`.

    Returns
    -------
    LogLogFit
        ``slope`` is :math:`2H-2`, weighted by the number of blocks.
    """
    _, inc = _level_and_increments(x, series_kind, name="series")
    sc = _resolve_scales(inc.size, scales, s_min=s_min, s_max=s_max, n_scales=n_scales)

    var = np.empty(sc.size, dtype=np.float64)
    counts = np.empty(sc.size, dtype=np.int64)
    for j, s in enumerate(sc):
        m = int(s)
        k = inc.size // m
        means = inc[: k * m].reshape(k, m).mean(axis=1)
        v = float(means.var(ddof=1))
        if v <= 0.0:
            raise ValueError(
                f"block means at scale {m} have zero variance; the series is degenerate there"
            )
        var[j] = v
        counts[j] = k

    weights = weighted_by_counts(counts)
    if select_range:
        return select_scaling_range(
            sc.astype(np.float64), var, weights=weights, method=fit_method
        ).fit
    return loglog_fit(sc.astype(np.float64), var, method=fit_method, weights=weights)


# --------------------------------------------------------------------------- #
# Detrended fluctuation analysis (the q = 2 special case)
# --------------------------------------------------------------------------- #


def _dfa_fluctuation(profile: FloatArray, s: int, order: int) -> tuple[float, int]:
    """RMS residual of a degree-``order`` polynomial fit over disjoint windows.

    Windows are laid out from both ends, so the :math:`n \\bmod s` samples the
    forward pass truncates are still used --- the standard Kantelhardt (2001)
    two-directional partition.
    """
    n = profile.size
    k = n // s
    forward = profile[: k * s].reshape(k, s)
    backward = profile[n - k * s :].reshape(k, s)
    blocks = np.vstack((forward, backward))

    t = np.arange(s, dtype=np.float64)
    design = np.vander(t, order + 1)
    coef, *_ = np.linalg.lstsq(design, blocks.T, rcond=None)
    resid = blocks.T - design @ coef
    return float(math.sqrt(float(np.mean(resid**2)))), int(2 * k)


def detrended_fluctuation(
    x: Sequence[float] | NDArray[Any],
    *,
    scales: Sequence[int] | NDArray[Any] | None = None,
    order: int = 1,
    s_min: int = 8,
    s_max: int | None = None,
    n_scales: int = 24,
    series_kind: SeriesKind = "level",
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
) -> LogLogFit:
    r"""Detrended fluctuation analysis, the :math:`q=2` case (Peng et al. 1994).

    Form the profile of the mean-removed increments,
    :math:`Y_k=\sum_{j\le k}(w_j-\bar w)`; partition it into disjoint windows of
    length :math:`s` from both ends; remove a degree-``order`` polynomial from
    each; and set

    .. math::
        F(s) = \Big(\tfrac{1}{2N_s s}\sum_{\nu=1}^{2N_s}\sum_{j=1}^{s}
                \big(Y_{\nu,j}-P_{\nu,\text{order}}(j)\big)^2\Big)^{1/2}
        \;\sim\; s^{\,\mathcal H} .

    For fGn, :math:`\mathcal H = H` exactly, so **the slope is :math:`H`**
    with no conversion. DFA1 --- linear detrending --- is the baseline this
    module exists to provide: a multifractal analysis that cannot beat DFA1 has
    produced nothing.

    **Duplication, stated.** This is the :math:`q=2`, monofractal case of
    multifractal DFA. It is implemented here rather than imported from
    :mod:`mmar.estimators.mfdfa` so that the baseline panel has no dependency on
    the multifractal machinery it is meant to check --- a baseline that shares
    an implementation with the thing it validates checks nothing. The two must
    agree numerically at :math:`q=2`, and that agreement is a cross-module test,
    not an assumption.

    **Known bias, and its cause.** Polynomial detrending of order :math:`p`
    removes power at low frequencies inside each window, so :math:`F(s)`
    is biased at small :math:`s`, with the bias growing in :math:`p`; the
    standard remedy is :math:`s \gtrsim 10(p+1)`. The default ``s_min=8`` does
    **not** meet it at ``order=1`` (which would need :math:`s\ge20`), and that
    gap *is* the DFA1 positive bias reported in the accuracy table: measured
    over 20 exact-fBm replicas at :math:`n=2^{14}`, ``s_min=8`` gives bias
    :math:`+0.021/+0.011/+0.008` at :math:`H=0.2/0.5/0.8` while ``s_min=20``
    gives :math:`+0.004/+0.004/-0.000` at roughly 1.6x the replica sd. The
    default is kept at 8 to preserve a full 2.7 decades of scaling range;
    pass ``s_min=10*(order+1)`` when bias matters more than range.
    Detrending also caps the resolvable exponent:
    order-:math:`p` DFA saturates at :math:`\mathcal H = p+1`, so DFA1 cannot
    see :math:`H>2` and, more relevantly, reports :math:`\mathcal H\approx1`
    for any series with an unremoved linear trend. At large :math:`s` the number
    of windows collapses and :math:`F(s)` becomes noisy; the WLS weighting by
    :math:`2N_s` is what stops those points dominating the slope.

    Returns
    -------
    LogLogFit
        ``slope`` is :math:`\hat H`.
    """
    if int(order) < 1:
        raise ValueError(f"the detrending order must be at least 1, got {order!r}")
    _, inc = _level_and_increments(x, series_kind, name="series")
    profile = np.ascontiguousarray(np.cumsum(inc - inc.mean()))
    sc = _resolve_scales(
        profile.size, scales, s_min=max(s_min, 2 * int(order) + 2), s_max=s_max, n_scales=n_scales
    )
    if int(sc.min()) <= int(order) + 1:
        raise ValueError(
            f"scale {int(sc.min())} cannot support a degree-{order} fit "
            f"(it needs more than {order + 1} points)"
        )

    fluct = np.empty(sc.size, dtype=np.float64)
    counts = np.empty(sc.size, dtype=np.int64)
    for j, s in enumerate(sc):
        fluct[j], counts[j] = _dfa_fluctuation(profile, int(s), int(order))
    if np.any(fluct <= 0.0):
        raise ValueError("the detrended fluctuation vanished at some scale; the fit is undefined")

    weights = weighted_by_counts(counts)
    if select_range:
        return select_scaling_range(
            sc.astype(np.float64), fluct, weights=weights, method=fit_method
        ).fit
    return loglog_fit(sc.astype(np.float64), fluct, method=fit_method, weights=weights)


# --------------------------------------------------------------------------- #
# Log-periodogram (GPH)
# --------------------------------------------------------------------------- #


def periodogram_hurst(
    x: Sequence[float] | NDArray[Any],
    *,
    bandwidth_exponent: float = 0.5,
    n_frequencies: int | None = None,
    series_kind: SeriesKind = "level",
    fit_method: FitMethod = FitMethod.OLS,
) -> LogLogFit:
    r"""Log-periodogram (Geweke--Porter-Hudak 1983) estimate of :math:`d`, hence :math:`H`.

    A stationary long-memory process has spectral density
    :math:`f(\lambda) \sim c|\lambda|^{-2d}` as :math:`\lambda\to0`. Writing the
    fractional-differencing filter exactly,
    :math:`f(\lambda)=|1-e^{i\lambda}|^{-2d}f_u(\lambda)
    = \big(4\sin^2\tfrac\lambda2\big)^{-d}f_u(\lambda)`, and taking logs of the
    periodogram :math:`I(\lambda_j)` at the Fourier frequencies
    :math:`\lambda_j=2\pi j/n`, :math:`j=1,\dots,m`, gives the regression

    .. math::
        \log I(\lambda_j) \;=\; c \;-\; d\,\log\!\Big(4\sin^2\tfrac{\lambda_j}{2}\Big)
        \;+\; \varepsilon_j .

    **The returned slope is** :math:`-d`, so :math:`H = d + \tfrac12
    = \tfrac12 - \text{slope}`. The conversion is applied only in
    :func:`hurst_consensus`.

    This is the one estimator in the panel that lives in the frequency domain.
    That makes it a genuinely independent check: R/S, aggregated variance and
    DFA all fail together in the presence of a slow trend, because they all
    estimate a time-domain second moment of the same windows, whereas GPH fails
    differently (a trend contaminates the lowest frequencies specifically, and
    can be excluded by trimming them).

    **Known bias, and its cause.** The regression is valid only in the
    asymptotic band :math:`\lambda\to0`; including frequencies where the
    short-memory factor :math:`f_u` is not flat biases :math:`\hat d`, and the
    bias is :math:`O((m/n)^{2})` under smoothness of :math:`f_u` (Hurvich, Deo &
    Brodsky 1998). The variance is :math:`\pi^2/(24m)`, so bandwidth trades bias
    against variance directly; the conventional :math:`m=\lfloor n^{1/2}\rfloor`
    used here is variance-heavy and *not* MSE-optimal --- the optimal rate is
    :math:`m\propto n^{4/5}`. Expect a standard deviation of roughly
    :math:`\pi/\sqrt{24 m} \approx 0.05` at :math:`n=2^{14}`, an order of
    magnitude worse than DFA1. It is in the panel for independence, not for
    precision. Additionally, :math:`\hat d` is only consistent for
    :math:`|d|<1/2`, i.e. :math:`0<H<1`, and a non-stationary input
    (:math:`H\ge1` after differencing) breaks it silently.

    Parameters
    ----------
    bandwidth_exponent
        :math:`m = \lfloor n^{\text{exponent}} \rfloor`. The literature default
        is :math:`1/2`.
    n_frequencies
        Overrides the bandwidth rule with an explicit :math:`m`.
    fit_method
        OLS by default: unlike the time-domain estimators, every periodogram
        ordinate carries the same (asymptotically exponential) distribution, so
        there is no count to weight by.

    Returns
    -------
    LogLogFit
        ``slope`` is :math:`-\hat d`; ``slope_stderr`` is the regression
        standard error, which is close to the theoretical
        :math:`\pi/\sqrt{24m}`.
    """
    _, inc = _level_and_increments(x, series_kind, name="series")
    n = inc.size
    m = int(n_frequencies) if n_frequencies is not None else math.floor(n**bandwidth_exponent)
    if m < 3:
        raise ValueError(f"the GPH bandwidth m={m} leaves fewer than 3 frequencies")
    if m >= n // 2:
        raise ValueError(f"the GPH bandwidth m={m} must stay below n/2={n // 2}")

    centred = inc - inc.mean()
    spectrum = np.abs(np.fft.rfft(centred)) ** 2 / (2.0 * math.pi * n)
    intensity = np.asarray(spectrum[1 : m + 1], dtype=np.float64)
    if np.any(intensity <= 0.0):
        raise ValueError(
            f"{int((intensity <= 0).sum())} periodogram ordinates vanish; the series has an "
            "exact spectral null in the GPH band and the log-regression is undefined"
        )

    lam = 2.0 * math.pi * np.arange(1, m + 1, dtype=np.float64) / n
    regressor = 4.0 * np.sin(0.5 * lam) ** 2
    return loglog_fit(regressor, intensity, method=fit_method)


# --------------------------------------------------------------------------- #
# The panel
# --------------------------------------------------------------------------- #


def hurst_consensus(
    x: Sequence[float] | NDArray[Any],
    *,
    scales: Sequence[int] | NDArray[Any] | None = None,
    tau_max: int = 19,
    dfa_order: int = 1,
    series_kind: SeriesKind = "level",
) -> dict[str, float]:
    r"""Run every baseline in this module and report the panel and its spread.

    Five estimators of one number, each converted to :math:`H` on the way out:

    ========================  =====================  =============================
    Key                       Fitted slope           Conversion
    ========================  =====================  =============================
    ``ghe_h2``                :math:`\zeta(2)=2H(2)` :math:`H=\text{slope}/2`
    ``rescaled_range``        :math:`H`              identity
    ``aggregated_variance``   :math:`2H-2`           :math:`H=1+\text{slope}/2`
    ``dfa1``                  :math:`H`              identity
    ``gph``                   :math:`-d`             :math:`H=\tfrac12-\text{slope}`
    ========================  =====================  =============================

    ``ghe_h1``, ``rescaled_range_uncorrected`` and the GPH memory parameter
    ``gph_d`` are also returned, but are **not** part of the spread: the first
    is redundant with ``ghe_h2`` under the monofractal hypothesis the panel is
    testing, and the other two are diagnostics.

    **How to read the spread.** Under a genuine self-similar scaling law all
    five estimate the *same* :math:`H`, so

    .. math::
        \text{spread} = \max_k \hat H_k - \min_k \hat H_k

    is a specification test. What counts as large must be calibrated against
    what these estimators do on data that *is* exactly self-similar, and the
    measured answer (30 replicas of exact fBm at :math:`n=2^{14}`, table in
    ``tests/unit/test_ghe.py``) is less flattering than the folklore:

    * ``dfa1`` and ``ghe_h2`` are tight --- bias :math:`\le 0.023`, sd
      :math:`\le 0.023`;
    * ``aggregated_variance`` has sd up to :math:`0.029` and bias
      :math:`-0.075` at :math:`H=0.95`;
    * ``gph`` is unbiased but has sd :math:`\approx 0.06`, its theoretical
      :math:`\pi/\sqrt{24m}`;
    * ``rescaled_range`` is the outlier: the Anis--Lloyd/Peters correction is
      derived under the iid null, so it is accurate at :math:`H=1/2` (bias
      :math:`-0.013`) and *over-corrects* away from it, by :math:`+0.079` at
      :math:`H=0.2` and :math:`-0.182` at :math:`H=0.95`.

    So on exact fBm the observed spread is about :math:`0.05` near
    :math:`H=1/2` and can reach :math:`0.2` at :math:`H=0.95` **with nothing
    wrong at all**. The honest thresholds are therefore
    :math:`\text{spread}\gtrsim0.15`, or :math:`\text{spread\_time\_domain}
    \gtrsim0.08` for the four time-domain members, and in either case the
    *identity of the outlier* must be inspected before concluding anything: an
    R/S member far from the rest at extreme :math:`H` is the correction's known
    bias, not evidence against scaling.

    When the disagreement is real, the usual causes are a trend or level shift
    (aggregated variance and R/S run high, DFA1 does not), a crossover so that
    the short-lag GHE and the long-lag DFA1 see different regimes, heavy tails
    which break the variance-based members but not R/S, or plain
    non-stationarity --- in which case no single :math:`H` exists, and a
    multifractal reading of the same data is not rescued by being multifractal.
    A large spread is a reason to stop, not a reason to average.

    Returns
    -------
    dict[str, float]
        The panel, plus ``mean``, ``median``, ``min``, ``max``, ``spread`` over
        the five members above, and ``spread_time_domain`` over the four
        excluding GPH (whose variance dominates the full spread).
    """
    ghe = generalised_hurst(
        x,
        q=np.array([1.0, 2.0, 3.0]),
        tau_max=tau_max,
        series_kind=series_kind,
    )
    rs = rescaled_range(x, scales=scales, corrected=True, series_kind=series_kind)
    rs_raw = rescaled_range(x, scales=scales, corrected=False, series_kind=series_kind)
    av = aggregated_variance(x, scales=scales, series_kind=series_kind)
    dfa = detrended_fluctuation(x, scales=scales, order=dfa_order, series_kind=series_kind)
    gph = periodogram_hurst(x, series_kind=series_kind)

    panel: dict[str, float] = {
        "ghe_h1": float(ghe.at(1.0)),
        "ghe_h2": float(ghe.h2),
        "rescaled_range": float(rs.slope),
        "rescaled_range_uncorrected": float(rs_raw.slope),
        "aggregated_variance": 1.0 + 0.5 * float(av.slope),
        "dfa1": float(dfa.slope),
        "gph": 0.5 - float(gph.slope),
        "gph_d": -float(gph.slope),
    }
    members = np.array([panel[k] for k in _CONSENSUS_PANEL], dtype=np.float64)
    time_domain = np.array([panel[k] for k in _CONSENSUS_PANEL if k != "gph"], dtype=np.float64)
    panel["mean"] = float(members.mean())
    panel["median"] = float(np.median(members))
    panel["min"] = float(members.min())
    panel["max"] = float(members.max())
    panel["spread"] = float(members.max() - members.min())
    panel["spread_time_domain"] = float(time_domain.max() - time_domain.min())
    return panel
