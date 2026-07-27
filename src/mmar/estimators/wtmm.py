r"""Wavelet Transform Modulus Maxima (WTMM).

The oldest wavelet-based multifractal formalism that actually works for
:math:`q<0`, and still the best-resolved method when the singularities are
*isolated* rather than dense.

The construction (Muzy, Bacry & Arneodo 1991, 1993)
---------------------------------------------------
Let :math:`\psi` be a real analysing wavelet with :math:`n_\psi \ge 1`
vanishing moments and let

.. math::
    T_\psi[x](a,b) \;=\; \frac1a \int x(t)\,
        \psi\!\Big(\frac{t-b}{a}\Big)\,\mathrm dt

be the continuous wavelet transform in the :math:`L^1` normalisation. If
:math:`x` is Hölder-:math:`\alpha` at :math:`t_0` with
:math:`\alpha < n_\psi`, then along the ridge :math:`b\to t_0`

.. math::
    |T_\psi[x](a,b(a))| \;\sim\; a^{\alpha(t_0)}, \qquad a \to 0^+ .

Naively summing :math:`|T|^q` over *all* :math:`(a,b)` is hopeless for
:math:`q<0`, because :math:`T` has zeros everywhere and :math:`|T|^{q}` then
diverges. WTMM fixes this in two steps, and both are essential:

1. **Restrict to the skeleton.** At each scale keep only the local maxima of
   :math:`b\mapsto|T(a,b)|` and *chain* them across scales into connected
   **maxima lines** :math:`l`. Only lines that survive down to the finest
   analysed scale are retained: a maximum that appears at one scale and
   evaporates at the next is a numerical accident, not a singularity.

2. **Take the supremum along the line.** The partition function is

   .. math::
       \mathcal Z(q,a) \;=\; \sum_{l \in \mathcal L(a)}
           \Big(\sup_{a' \le a,\ (a',b)\in l} |T_\psi[x](a',b)|\Big)^{q}
       \;\sim\; a^{\tau(q)} ,

   where :math:`\mathcal L(a)` is the set of maxima lines present at scale
   :math:`a`. The supremum is what makes the negative branch well posed: it is
   bounded away from zero by the line's own largest coefficient, so
   :math:`(\cdot)^q` cannot blow up even when :math:`|T(a,b)|` happens to be
   small at the scale :math:`a` itself. Replacing the supremum by
   :math:`|T(a,b)|` --- which many implementations quietly do --- destroys the
   whole method for :math:`q<0` and is the commonest WTMM bug.

Convention
----------
:math:`\mathcal Z` is a **sum over lines**, exactly analogous to a box-counting
partition function :math:`\sum_i \mu_i^q`, so the fitted slope is
:math:`\tau(q)` in the **measure** convention of :mod:`mmar.core.types`. This
module therefore converts on output,

.. math::
    \zeta(q) \;=\; \tau(q) + d ,

with :math:`d = 1` for a time series. Two anchors follow and are tested:

* the number of maxima lines scales like :math:`\mathcal Z(0,a)\sim a^{-d}`, so
  :math:`\tau(0)\simeq-1` and :math:`\zeta(0)\simeq0`;
* for fBm of Hurst :math:`H`, :math:`|T|\sim a^{H}` along every line, hence
  :math:`\tau(q)=qH-1` and :math:`\zeta(q)=qH`;
* for the binomial measure with multipliers :math:`(m_0,m_1)`, applied to its
  primitive (the devil's staircase / trading time), integrating by parts gives
  :math:`T(a,b) = -\int \Psi((t-b)/a)\,\mathrm d\mu(t) \sim \mu(B(b,a))`, so
  :math:`\zeta(q) = 1-\log_2(m_0^q+m_1^q)`.

Known bias and failure modes (state these before quoting a number)
------------------------------------------------------------------
* **Cost.** :math:`O(N\log N)` per scale, so :math:`O(N\log N\cdot J)` overall,
  against :math:`O(N)` per scale for a structure function or MFDFA --- WTMM
  does asymptotically more work than either, and it also pays for the chaining
  pass. Measured for this implementation on an M-series laptop, 20 scales and
  41 moments: :math:`n=2^{14}` takes **12.8 ms** end to end (4.8 ms CWT, 6.8 ms
  skeleton, the remainder the partition function and 41 regressions) and
  :math:`n=2^{16}` takes **39.0 ms**.
* **Chaining tolerance.** The result depends on ``chain_tol``: too tight and
  lines die, thinning :math:`\mathcal L(a)` and steepening :math:`\tau`; too
  loose and distinct singularities get welded into one line, flattening it.
  There is no principled value; ``1.0`` (link to a maximum within one coarse
  scale) is the usual choice and is what is used here. Sensitivity should be
  reported, not hidden.
* **Negative** :math:`q` **needs a right branch to exist.** The supremum makes
  :math:`\mathcal Z(q,a)` *finite* for :math:`q<0`; it does not make it
  *meaningful*. Write the line suprema as :math:`s_l = C_l\,a^{\alpha_l}`. If
  the singularity spectrum is bounded, :math:`\alpha \le \alpha_{\max} <
  \infty`, the negative moments of :math:`C` converge and :math:`\hat\tau(q)`
  is excellent: on a binomial cascade (:math:`m_0=0.6`, :math:`k=14`, dyadic
  scales, ``n_vanishing=1``) the observed error is :math:`\le 0.04` over the
  whole of :math:`q\in[-5,5]`. On a **monofractal** there is no right branch at
  all, the density of :math:`C` is positive at the origin, :math:`\mathbb
  E[C^q]=\infty` for :math:`q\le-1`, and the sum collapses onto its smallest
  term. Since the minimum of :math:`N_a\sim a^{-1}` such draws is of order
  :math:`a`, the estimator returns the *pathology law*

  .. math::
      \hat h(q) \;=\; \frac{\hat\zeta(q)}{q} \;\approx\; H + 1 + \frac1q ,
      \qquad q \ll 0 ,

  and this is what is observed: on fBm at :math:`n=2^{14}`, :math:`H=0.5`,
  :math:`\hat h(-5)` ranges over :math:`1.12\text{--}1.27` across six seeds
  (mean :math:`\approx1.18`) against a predicted :math:`1.30` --- the law gets
  the size and the sign of the divergence right, not its third digit.
  A negative-:math:`q` branch that tracks :math:`H+1+1/q` is therefore a
  *monofractality diagnostic*, not an estimate. Compare against fBm before
  reading any width off the right branch.
* **Choice of wavelet matters more than usual for measures.** Analysing the
  primitive of a measure with :math:`n_\psi` vanishing moments is, after one
  integration by parts, analysing the measure itself with the kernel
  :math:`\Psi=\int\psi`. For :math:`n_\psi=1` that kernel is a *Gaussian* ---
  strictly positive, so :math:`|T|` is a smoothed mass and there is no
  cancellation. For the Mexican hat (:math:`n_\psi=2`) it is the odd first
  derivative of a Gaussian, :math:`T` is a smoothed left--right mass
  *difference*, and the resulting cancellations cost an order of magnitude in
  accuracy: on the binomial cascade above, ``n_vanishing=2`` gives errors of
  :math:`-0.52` at :math:`q=5` and :math:`-1.04` at :math:`q=-5` where
  ``n_vanishing=1`` gives :math:`-0.036` and :math:`+0.001`. **Use
  ``n_vanishing=1`` for measures**; the Mexican hat default is for signals,
  where the extra vanishing moment kills a linear trend.
* **Edge effects.** The transform is computed on a reflection-padded signal, so
  there is no circular wrap, but maxima within ``edge_factor * a`` of either
  boundary are still discarded. That exclusion removes a band whose width grows
  with :math:`a` and therefore biases :math:`\tau(0)` slightly below
  :math:`-1`; keep ``edge_factor`` small and ``s_max`` well below :math:`n/8`.
  With the default ``edge_factor=0.5`` the observed :math:`\hat\tau(0)` on fBm
  at :math:`n=2^{14}` stays within :math:`0.008` of :math:`-1` over 8 seeds,
  and on the binomial measure it is :math:`-1.00000`.
* **Coarse scales are line-starved.** Unlike a structure function, WTMM's
  effective sample size at scale :math:`a` is the *number of lines*, which can
  fall to single digits. Scales with fewer than ``min_lines`` lines are dropped
  and the fit is weighted by the line count. Residual coarse-scale curvature
  remains at large :math:`|q|`, but it is mild over the recommended window: on
  the binomial cascade the local slope at :math:`q=5` drifts monotonically from
  :math:`3.477` (over :math:`a=8\to16`) to :math:`3.39` (:math:`512\to1024`)
  and only reaches :math:`3.30` on :math:`1024\to2048`, against a true
  :math:`3.506`. That last decade is what the :math:`n/16` cap excludes.
* **Positive** :math:`q` **is biased upward, and the bias does not vanish with
  more data.** :math:`h(2)` on fBm is biased **upward** by
  :math:`+0.064,\,+0.038,\,+0.022` at :math:`H=0.3,\,0.5,\,0.7`, and those
  figures are the same at :math:`n=2^{14}`, :math:`2^{16}` and :math:`2^{18}`
  (3 seeds each; the seed-to-seed sd falls like :math:`n^{-1/2}` but the mean
  does not move). Over the same paths MFDFA and the structure function
  converge to :math:`|{\rm bias}|<0.005` at :math:`n=2^{18}`, so this is the
  estimator's, not the generator's.

  The cause is the **running supremum**, not the record length and not the
  chaining. :math:`\sup_{a'\le a}` is taken from the finest *analysed* scale
  upward, so the inflation it applies to :math:`|T(a,b)|` grows from
  :math:`1` at :math:`a=a_{\min}` to an :math:`a`-independent asymptote; that
  transient tilts :math:`\log\mathcal Z` and steepens :math:`\hat\tau`.
  Rebuilding :math:`\mathcal Z` from the node moduli instead of the suprema on
  the *same* skeleton removes it entirely --- bias :math:`\le0.006` at every
  :math:`H` and every :math:`n` tested, :math:`\le0.002` at :math:`n=2^{18}`
  (:func:`~tests.unit.test_wtmm.test_positive_q_bias_is_the_running_supremum`).
  Dropping the supremum is **not** the fix: it is what makes :math:`q<0` well
  posed (see above). The fix is to read :math:`h(2)` off a
  fluctuation-based estimator and use WTMM for the negative branch, the
  spectrum width and the pointwise exponents.

References
----------
Muzy, J.F., Bacry, E. and Arneodo, A. (1991). Wavelets and multifractal
formalism for singular signals: application to turbulence data.
*Phys. Rev. Lett.* **67**, 3515--3518.

Muzy, J.F., Bacry, E. and Arneodo, A. (1993). Multifractal formalism for
fractal signals: the structure-function approach versus the wavelet-transform
modulus-maxima method. *Phys. Rev. E* **47**, 875--884.

Arneodo, A., Bacry, E. and Muzy, J.F. (1995). The thermodynamics of fractals
revisited with wavelets. *Physica A* **213**, 232--275.

Mallat, S. and Hwang, W.L. (1992). Singularity detection and processing with
wavelets. *IEEE Trans. Inform. Theory* **38**, 617--643.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq
from scipy.special import gammaln, logsumexp

from mmar.core.provenance import InputRef, Provenance, new_provenance
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
    "WTMMSkeleton",
    "cwt_dog",
    "line_exponent",
    "maxima_lines",
    "wtmm",
    "wtmm_structure",
]

_VERSION = "1"

_DEFAULT_S_MIN = 4
"""Below ~4 samples the sampled derivative-of-Gaussian is not resolved."""

_DEFAULT_MAX_FRACTION = 0.0625
"""Coarsest scale :math:`n/16`. WTMM's effective sample size is the number of
maxima lines, which collapses long before the structure-function window count
does; :math:`n/4` (the library default) leaves a handful of lines."""


# --------------------------------------------------------------------------- #
# Continuous wavelet transform
# --------------------------------------------------------------------------- #


def _dog_filter(omega: FloatArray, a: float, n_vanishing: int) -> NDArray[np.complex128]:
    r"""Fourier multiplier :math:`\overline{\hat\psi(a\omega)}` of the DOG wavelet.

    With :math:`\psi = \mathrm d^{n}/\mathrm dt^{n}\,e^{-t^2/2}` (up to sign),

    .. math::
        \hat\psi(\omega) \;=\; \frac{(i\omega)^{n}}{\sqrt{\Gamma(n+\tfrac12)}}\,
                               e^{-\omega^2/2},

    the constant being the :math:`L^2` normalisation of Torrence & Compo; it is
    :math:`a`-independent and therefore affects only the intercept of every
    log--log fit, never an exponent. The multiplier is Hermitian
    (:math:`F(-\omega)=\overline{F(\omega)}`), which is what allows the inverse
    transform to be taken with :func:`scipy.fft.irfft`.
    """
    aw = a * omega
    norm = math.exp(-0.5 * gammaln(n_vanishing + 0.5))
    with np.errstate(under="ignore"):
        envelope = norm * aw**n_vanishing * np.exp(-0.5 * aw * aw)
    return np.asarray((-1j) ** n_vanishing * envelope, dtype=np.complex128)


def cwt_dog(
    x: FloatArray, scales: FloatArray | Sequence[float], *, n_vanishing: int = 2
) -> FloatArray:
    r"""Continuous wavelet transform against the :math:`n`-th derivative of a Gaussian.

    Computed by FFT convolution, :math:`O(N\log N)` per scale, on a
    reflection-padded copy of the signal so that no coefficient inside the
    original support is contaminated by a circular wrap. ``n_vanishing=2`` is
    the Mexican hat.

    Parameters
    ----------
    x
        Real signal, shape ``(n,)``.
    scales
        Analysing scales :math:`a`, in samples.
    n_vanishing
        Number of vanishing moments :math:`n_\psi`. A Hölder exponent
        :math:`\alpha` is only detectable when :math:`\alpha < n_\psi`; a
        polynomial trend of degree :math:`< n_\psi` is annihilated. Raise it to
        remove trends, at the cost of a wider wavelet support and worse
        time localisation.

    Returns
    -------
    FloatArray
        Shape ``(n_scales, n)``, the :math:`L^1`-normalised coefficients
        :math:`T_\psi[x](a,b)` so that :math:`|T|\sim a^{\alpha}` on a ridge.
    """
    a = np.asarray(scales, dtype=np.float64).ravel()
    if a.size == 0:
        raise ValueError("scales is empty")
    if np.any(a <= 0.0) or not np.all(np.isfinite(a)):
        raise ValueError("scales must be finite and strictly positive")
    if int(n_vanishing) != n_vanishing or n_vanishing < 1:
        raise ValueError(f"n_vanishing must be a positive integer, got {n_vanishing!r}")

    n = int(x.size)
    # Five e-folding widths of the widest wavelet, capped at the signal length:
    # beyond that the Gaussian envelope is below double precision anyway.
    pad = int(min(n, math.ceil(5.0 * float(a.max()))))
    n_ext = int(next_fast_len(n + 2 * pad))
    left = np.asarray(x[pad:0:-1] if pad > 0 else x[:0], dtype=np.float64)
    right = np.asarray(x[-2 : -2 - (n_ext - n - left.size) : -1], dtype=np.float64)
    if left.size + n + right.size != n_ext:
        # Signals shorter than the requested padding cannot be reflected once;
        # fall back to edge-value extension, which is continuous and adds no
        # spurious singularity.
        ext = np.concatenate(
            [
                np.full(pad, x[0]),
                x,
                np.full(n_ext - n - pad, x[-1]),
            ]
        )
        left_size = pad
    else:
        ext = np.concatenate([left, x, right])
        left_size = left.size

    spectrum = rfft(ext)
    omega = 2.0 * math.pi * np.asarray(rfftfreq(n_ext), dtype=np.float64)
    out = np.empty((a.size, n), dtype=np.float64)
    for j, aj in enumerate(a):
        filt = _dog_filter(omega, float(aj), int(n_vanishing))
        if n_ext % 2 == 0:
            # irfft treats the Nyquist bin as real; the DOG multiplier is purely
            # imaginary there for odd n_vanishing. The envelope is ~exp(-(a*pi)^2/2),
            # i.e. exactly zero in float64 for any a >= 4, so this is bookkeeping.
            filt[-1] = filt[-1].real
        conv = irfft(spectrum * filt, n=n_ext)
        out[j] = conv[left_size : left_size + n]
    return out


# --------------------------------------------------------------------------- #
# The skeleton
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class WTMMSkeleton:
    r"""The maxima-line skeleton: WTMM's most informative diagnostic.

    A *node* is one modulus maximum at one scale. Nodes are stored flat and
    grouped by ``line_id``; a *line* is the set of nodes sharing a ``line_id``,
    one per scale, tracing a curve :math:`a\mapsto b(a)` in the
    time--scale plane. Only lines reaching the finest analysed scale are kept,
    and ``line_id`` is the index of the line's node at that finest scale.

    Plotting ``position`` against ``scales[scale_index]`` on a log scale axis
    *is* the skeleton picture: singularities show up as lines converging to
    a point of the time axis, and the local slope of
    :math:`\log|T|` against :math:`\log a` along one line is that point's Hölder
    exponent (:func:`line_exponent`).
    """

    scales: FloatArray
    scale_index: NDArray[np.int64]
    line_id: NDArray[np.int64]
    position: FloatArray
    modulus: FloatArray
    running_sup: FloatArray
    r"""``max`` of ``modulus`` over the line from the finest scale up to this node."""
    n_lines: NDArray[np.int64]
    """Number of surviving lines present at each scale; the effective sample size."""
    n_maxima: NDArray[np.int64]
    """Number of raw modulus maxima at each scale, before chaining."""
    n_series: int = 0
    wavelet: str = "dog2"
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        sizes = {
            self.scale_index.size,
            self.line_id.size,
            self.position.size,
            self.modulus.size,
            self.running_sup.size,
        }
        if len(sizes) != 1:
            raise ValueError(f"node arrays have inconsistent lengths: {sorted(sizes)}")
        if self.n_lines.size != self.scales.size or self.n_maxima.size != self.scales.size:
            raise ValueError("per-scale counts must match the number of scales")

    @property
    def n_nodes(self) -> int:
        return int(self.position.size)

    @property
    def n_lines_total(self) -> int:
        """Number of distinct maxima lines, i.e. of maxima at the finest scale."""
        return int(np.unique(self.line_id).size)

    def line(self, line_id: int) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``(scales, positions, moduli)`` of one line, ordered from fine to coarse."""
        sel = np.nonzero(self.line_id == int(line_id))[0]
        if sel.size == 0:
            raise KeyError(f"no maxima line with id {line_id}")
        order = sel[np.argsort(self.scale_index[sel])]
        return (
            np.asarray(self.scales[self.scale_index[order]], dtype=np.float64),
            np.asarray(self.position[order], dtype=np.float64),
            np.asarray(self.modulus[order], dtype=np.float64),
        )

    def line_nearest(self, position: float) -> int:
        """Id of the line whose finest-scale node sits closest to ``position``.

        The natural way to interrogate an isolated singularity: give the time at
        which it sits and get back the line that converges onto it.
        """
        base = np.nonzero(self.scale_index == 0)[0]
        if base.size == 0:  # pragma: no cover - the finest scale always carries nodes
            raise ValueError("the skeleton has no nodes at the finest scale")
        i = int(np.argmin(np.abs(self.position[base] - float(position))))
        return int(self.line_id[base[i]])

    def longest_lines(self, k: int = 10) -> NDArray[np.int64]:
        """Ids of the ``k`` lines spanning the most scales, coarsest-reaching first."""
        ids, counts = np.unique(self.line_id, return_counts=True)
        order = np.argsort(-counts, kind="stable")[: int(k)]
        return np.asarray(ids[order], dtype=np.int64)


def line_exponent(
    skeleton: WTMMSkeleton,
    line_id: int,
    *,
    fit_method: FitMethod = FitMethod.OLS,
    min_points: int = 4,
) -> LogLogFit:
    r"""Pointwise Hölder exponent from one maxima line: slope of :math:`\log|T|` vs :math:`\log a`.

    This is WTMM's unique strength and the reason it survived the arrival of
    faster methods: for an *isolated* singularity :math:`|t-t_0|^{\alpha}` the
    single line converging onto :math:`t_0` obeys
    :math:`|T(a,b(a))| = C\,a^{\alpha}` with no ensemble average and no
    partition function, so :math:`\alpha` is recovered from one regression
    (Mallat & Hwang 1992).

    The fit is unweighted by default: the nodes of a line are one coefficient
    each, not an average over windows, so there is no count to weight by.
    """
    a, _, modulus = skeleton.line(line_id)
    if a.size < max(3, int(min_points)):
        raise ValueError(
            f"line {line_id} spans only {a.size} scales; at least "
            f"{max(3, int(min_points))} are needed for a slope with error bars"
        )
    return loglog_fit(a, modulus, method=fit_method, log_y=True)


# --------------------------------------------------------------------------- #
# Maxima detection and chaining
# --------------------------------------------------------------------------- #


def _local_maxima(v: FloatArray) -> NDArray[np.int64]:
    """Strict interior local maxima of ``v``, vectorised.

    Strict on both sides: a plateau is not a singularity, and admitting one
    would create duplicate nodes that the chaining step cannot separate.
    """
    if v.size < 3:
        return np.empty(0, dtype=np.int64)
    interior = (v[1:-1] > v[:-2]) & (v[1:-1] > v[2:])
    return np.asarray(np.nonzero(interior)[0] + 1, dtype=np.int64)


def _link(pos_coarse: FloatArray, pos_fine: FloatArray, tol: float) -> NDArray[np.int64]:
    """Injective nearest-neighbour map from coarse-scale to fine-scale maxima.

    Returns, for each coarse maximum, the index of the fine maximum it is
    chained to, or ``-1`` when no candidate lies within ``tol``. Injectivity is
    enforced greedily by distance: in the continuum two distinct maxima lines
    never coincide, so a collision means the discretisation cannot separate
    them and the losing line is terminated rather than duplicated.
    """
    m = pos_coarse.size
    if m == 0 or pos_fine.size == 0:
        return np.full(m, -1, dtype=np.int64)

    k = np.searchsorted(pos_fine, pos_coarse)
    lo = np.clip(k - 1, 0, pos_fine.size - 1)
    hi = np.clip(k, 0, pos_fine.size - 1)
    d_lo = np.abs(pos_coarse - pos_fine[lo])
    d_hi = np.abs(pos_coarse - pos_fine[hi])
    take_hi = d_hi < d_lo
    nearest = np.where(take_hi, hi, lo)
    dist = np.where(take_hi, d_hi, d_lo)

    out = np.full(m, -1, dtype=np.int64)
    cand = np.nonzero(dist <= tol)[0]
    if cand.size == 0:
        return out
    order = cand[np.lexsort((dist[cand], nearest[cand]))]
    _, first = np.unique(nearest[order], return_index=True)
    winners = order[first]
    out[winners] = nearest[winners]
    return out


def _resolve_scales(n: int, scales: Sequence[int] | NDArray[Any] | None) -> NDArray[np.int64]:
    if scales is None:
        raw = log_spaced_scales(
            n, s_min=_DEFAULT_S_MIN, n_scales=20, max_fraction=_DEFAULT_MAX_FRACTION
        )
    else:
        raw = np.asarray(scales, dtype=np.int64).ravel()
    s = check_scales(raw, n, name="scales")
    if s.size < 3:
        raise ValueError(
            f"WTMM needs at least three usable scales, got {s.size} for n={n}; "
            "the series is too short or the scale grid too narrow"
        )
    return s


def _build_skeleton(
    coeffs: FloatArray,
    scales: NDArray[np.int64],
    *,
    chain_tol: float,
    edge_factor: float,
) -> tuple[list[FloatArray], list[FloatArray], list[NDArray[np.int64]], NDArray[np.int64], int]:
    """Detect, chain and validate maxima; returns per-scale surviving nodes.

    Returns ``(positions, running_sups, line_ids, n_maxima_raw, n_zero_dropped)``
    with one entry per scale, finest first.
    """
    n = coeffs.shape[1]
    j_max = scales.size

    pos: list[FloatArray] = []
    mod: list[FloatArray] = []
    n_raw = np.zeros(j_max, dtype=np.int64)
    n_zero = 0
    for j in range(j_max):
        v = np.abs(coeffs[j])
        idx = _local_maxima(v)
        n_raw[j] = idx.size
        margin = edge_factor * float(scales[j])
        keep = (idx.astype(np.float64) >= margin) & (idx.astype(np.float64) <= n - 1 - margin)
        idx = idx[keep]
        # Exact zeros make (.)^q infinite for q < 0. They are dropped, and the
        # count is recorded in provenance diagnostics rather than swallowed.
        nonzero = v[idx] > 0.0
        n_zero += int((~nonzero).sum())
        idx = idx[nonzero]
        pos.append(np.asarray(idx, dtype=np.float64))
        mod.append(np.asarray(v[idx], dtype=np.float64))

    if pos[0].size == 0:
        raise ValueError(
            f"no modulus maxima survive at the finest scale a={scales[0]}; "
            "reduce edge_factor or the smallest scale"
        )

    # Chain coarse -> fine, then propagate validity, the running supremum and the
    # line identity fine -> coarse. Both sweeps are loops over scales only.
    parent: list[NDArray[np.int64]] = [np.empty(0, dtype=np.int64)]
    for j in range(1, j_max):
        parent.append(_link(pos[j], pos[j - 1], chain_tol * float(scales[j])))

    valid: list[NDArray[np.bool_]] = [np.ones(pos[0].size, dtype=bool)]
    sup: list[FloatArray] = [mod[0].copy()]
    ident: list[NDArray[np.int64]] = [np.arange(pos[0].size, dtype=np.int64)]
    for j in range(1, j_max):
        p = parent[j]
        has = p >= 0
        safe = np.where(has, p, 0)
        ok = has & valid[j - 1][safe] if pos[j - 1].size else np.zeros(p.size, dtype=bool)
        valid.append(np.asarray(ok, dtype=bool))
        sup.append(np.maximum(mod[j], np.where(has, sup[j - 1][safe], 0.0)))
        ident.append(np.where(has, ident[j - 1][safe], -1).astype(np.int64))

    out_pos = [pos[j][valid[j]] for j in range(j_max)]
    out_sup = [sup[j][valid[j]] for j in range(j_max)]
    out_id = [ident[j][valid[j]] for j in range(j_max)]
    return out_pos, out_sup, out_id, n_raw, n_zero


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _provenance(
    name: str,
    x: FloatArray,
    *,
    q: FloatArray | None,
    scales: NDArray[np.int64],
    params: dict[str, Any],
    seed: SeedLike,
) -> Provenance:
    inputs = [InputRef.from_array("x", x), InputRef.from_array("scales", scales)]
    if q is not None:
        inputs.insert(1, InputRef.from_array("q", q))
    return new_provenance(
        name,
        estimator_version=_VERSION,
        params=params,
        inputs=inputs,
        # WTMM draws no random numbers. Recording a SeedInfo for seed=None would
        # bake fresh OS entropy into the digest and make the root unreproducible.
        seed=None if seed is None else seed_info(seed),
        notes=(
            "WTMM partition function sums over maxima lines, so the fitted slope is "
            "tau(q) in the measure convention; zeta(q) = tau(q) + d_support",
            "supremum taken along each line over all scales at or below a",
        ),
    )


def maxima_lines(
    x: Sequence[float] | NDArray[Any],
    *,
    scales: Sequence[int] | NDArray[Any] | None = None,
    n_vanishing: int = 2,
    chain_tol: float = 1.0,
    edge_factor: float = 0.5,
    seed: SeedLike = None,
) -> WTMMSkeleton:
    r"""Build the WTMM skeleton of ``x`` without computing any exponent.

    Parameters
    ----------
    scales
        Geometric grid of analysing scales in samples. ``None`` uses
        :func:`~mmar.core.types.log_spaced_scales` from :math:`a=4` to
        :math:`n/16` --- see :data:`_DEFAULT_MAX_FRACTION` for why the coarse
        end is well below the library-wide default.
    n_vanishing
        Vanishing moments of the derivative-of-Gaussian wavelet; ``2`` is the
        Mexican hat.
    chain_tol
        A maximum at scale :math:`a` is chained to the nearest maximum at the
        next finer scale if they are within ``chain_tol * a`` samples. The
        single most influential free parameter of the method.
    edge_factor
        Maxima within ``edge_factor * a`` of either boundary are discarded.
    seed
        Accepted for signature uniformity and recorded in provenance; WTMM is
        deterministic and draws no random numbers.

    Returns
    -------
    WTMMSkeleton
        Nodes of every line that reaches the finest scale.
    """
    xa = validate_series(x, name="x")
    s = _resolve_scales(xa.size, scales)
    if not math.isfinite(chain_tol) or chain_tol <= 0.0:
        raise ValueError(f"chain_tol must be finite and strictly positive, got {chain_tol!r}")
    if not math.isfinite(edge_factor) or edge_factor < 0.0:
        raise ValueError(f"edge_factor must be finite and non-negative, got {edge_factor!r}")

    coeffs = cwt_dog(xa, s.astype(np.float64), n_vanishing=n_vanishing)
    pos, sup, ident, n_raw, n_zero = _build_skeleton(
        coeffs, s, chain_tol=chain_tol, edge_factor=edge_factor
    )

    n_lines = np.array([p.size for p in pos], dtype=np.int64)
    scale_index = np.concatenate([np.full(p.size, j, dtype=np.int64) for j, p in enumerate(pos)])
    flat_pos = np.concatenate(pos) if pos else np.empty(0, dtype=np.float64)
    flat_sup = np.concatenate(sup) if sup else np.empty(0, dtype=np.float64)
    flat_id = np.concatenate(ident) if ident else np.empty(0, dtype=np.int64)
    # modulus at the node itself, recoverable from the CWT at (scale, position)
    flat_mod = np.concatenate([np.abs(coeffs[j][pos[j].astype(np.int64)]) for j in range(s.size)])

    params = {
        "n_vanishing": int(n_vanishing),
        "chain_tol": float(chain_tol),
        "edge_factor": float(edge_factor),
        "n": int(xa.size),
    }
    prov = _provenance(
        "wtmm.skeleton", xa, q=None, scales=s, params=params, seed=seed
    ).with_diagnostics(
        n_maxima_per_scale=[int(v) for v in n_raw],
        n_lines_per_scale=[int(v) for v in n_lines],
        n_zero_modulus_dropped=int(n_zero),
        n_lines_total=int(n_lines[0]) if n_lines.size else 0,
    )
    return WTMMSkeleton(
        scales=np.asarray(s, dtype=np.float64),
        scale_index=scale_index,
        line_id=flat_id,
        position=flat_pos,
        modulus=flat_mod,
        running_sup=flat_sup,
        n_lines=n_lines,
        n_maxima=n_raw,
        n_series=int(xa.size),
        wavelet=f"dog{int(n_vanishing)}",
        provenance=prov,
    )


def wtmm_structure(
    x: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    n_vanishing: int = 2,
    chain_tol: float = 1.0,
    edge_factor: float = 0.5,
    min_lines: int = 4,
    seed: SeedLike = None,
) -> StructureFunctions:
    r"""The raw WTMM scaling surface :math:`\mathcal Z(q,a)`, unfitted.

    Separated from :func:`wtmm` so that exponents can be refitted over a
    different scaling range --- or with a different regression --- without
    recomputing the wavelet transform and the skeleton, which is where all the
    cost is.

    ``min_lines`` drops scales carrying fewer than that many maxima lines. A
    partition function over three lines is not an estimate; silently keeping
    such scales is how WTMM log--log plots acquire their characteristic
    upward hook at the coarse end.

    Notes
    -----
    :math:`\mathcal Z` is accumulated as
    :math:`\log\mathcal Z = \operatorname{logsumexp}(q\log s_l)` over line
    suprema :math:`s_l`, never as ``(sup**q).sum()``: at :math:`q=-5` and
    :math:`n=2^{14}` the direct form is within a few orders of magnitude of
    overflow and, for finer grids, past it.
    """
    xa = validate_series(x, name="x")
    qa = validate_q_grid(q)
    skeleton = maxima_lines(
        xa,
        scales=scales,
        n_vanishing=n_vanishing,
        chain_tol=chain_tol,
        edge_factor=edge_factor,
        seed=seed,
    )
    if int(min_lines) < 1:
        raise ValueError(f"min_lines must be at least 1, got {min_lines!r}")

    keep = np.nonzero(skeleton.n_lines >= int(min_lines))[0]
    if keep.size < 3:
        raise ValueError(
            f"only {keep.size} scales carry >= {min_lines} maxima lines "
            f"(line counts {skeleton.n_lines.tolist()}); WTMM cannot fit a power law. "
            "Lengthen the series, lower min_lines, or narrow the scale range."
        )
    # Scales must stay contiguous: a power law is fitted over a window, not over
    # a set of scales that happen to have survived.
    if np.any(np.diff(keep) != 1):
        keep = keep[: int(np.nonzero(np.diff(keep) != 1)[0][0]) + 1]
        if keep.size < 3:
            raise ValueError(
                "the scales carrying enough maxima lines are not contiguous; "
                "narrow the scale range explicitly"
            )

    log_z = np.empty((qa.size, keep.size), dtype=np.float64)
    for jj, j in enumerate(keep):
        sel = skeleton.scale_index == j
        with np.errstate(divide="ignore"):
            log_sup = np.log(skeleton.running_sup[sel])
        log_z[:, jj] = logsumexp(qa[:, None] * log_sup[None, :], axis=1)

    with np.errstate(over="ignore"):
        values = np.exp(log_z)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "the WTMM partition function overflowed float64; the |q| range is too "
            "wide for the dynamic range of the maxima-line suprema. Restrict q."
        )

    counts = np.asarray(skeleton.n_lines[keep], dtype=np.int64)
    prov = (
        skeleton.provenance.derive(
            "wtmm.structure",
            estimator_version=_VERSION,
            params={"min_lines": int(min_lines), "n_q": int(qa.size)},
            extra_inputs=(InputRef.from_array("q", qa),),
        )
        if skeleton.provenance is not None
        else None
    )
    if prov is not None and skeleton.provenance is not None:
        # `derive` starts a fresh diagnostics map; the skeleton's counts and the
        # number of discarded zero-modulus maxima must survive into the result.
        prov = prov.with_diagnostics(
            **dict(skeleton.provenance.diagnostics),
            n_lines_per_fitted_scale=[int(v) for v in counts],
            scales_dropped=int(skeleton.scales.size - keep.size),
        )
    return StructureFunctions(
        q=qa,
        scales=np.asarray(skeleton.scales[keep], dtype=np.float64),
        values=values,
        counts=counts,
        provenance=prov,
    )


def wtmm(
    x: Sequence[float] | NDArray[Any],
    *,
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    n_vanishing: int = 2,
    chain_tol: float = 1.0,
    edge_factor: float = 0.5,
    min_lines: int = 4,
    fit_method: FitMethod = FitMethod.WLS,
    select_range: bool = False,
    d_support: float = 1.0,
    seed: SeedLike = None,
) -> ScalingExponents:
    r"""Scaling exponents by the Wavelet Transform Modulus Maxima method.

    Fits :math:`\log\mathcal Z(q,a) = \tau(q)\log a + c_q` over the scale grid,
    weighted by the number of maxima lines at each scale (which falls roughly
    like :math:`1/a`, so an unweighted fit hands the slope to the two or three
    coarsest, line-starved scales).

    **The fitted slope is** :math:`\tau(q)`, the measure-convention exponent,
    because :math:`\mathcal Z` is a sum over lines rather than an average over
    windows. The returned :class:`~mmar.core.types.ScalingExponents` carries

    .. math::
        \zeta(q) \;=\; \tau(q) + d ,

    so ``result.zeta`` is directly comparable with
    :func:`mmar.models.theory.fbm_zeta` and ``result.tau`` with
    :func:`mmar.models.theory.cascade_tau`. The ``fits`` tuple stores the raw
    regressions, whose ``slope`` is :math:`\tau`, not :math:`\zeta`.

    Parameters
    ----------
    x
        The signal. To analyse a *measure*, pass its primitive (the cumulative
        sum / trading time) **and set** ``n_vanishing=1``: WTMM is defined for
        functions, the exponent of the primitive at a point is the local
        dimension of the measure there, and one vanishing moment makes the
        effective kernel a positive Gaussian rather than a sign-changing
        derivative (see the module docstring for the accuracy cost of getting
        this wrong).
    q
        Moment grid; ``None`` gives 41 points on :math:`[-5,5]`. Negative
        :math:`q` are not optional --- the right branch of :math:`f(\alpha)`
        exists only there --- but see the module docstring for their bias.
    scales, n_vanishing, chain_tol, edge_factor
        As in :func:`maxima_lines`.
    min_lines
        As in :func:`wtmm_structure`.
    fit_method
        ``WLS`` by default, weighted by the line count.
    select_range
        Choose the scaling window automatically. The window is selected once,
        on the row nearest :math:`q=2`, and applied to every :math:`q`: fitting
        each moment over a different window makes :math:`\hat\tau` a function
        of :math:`q` *and* of the selection noise, and destroys concavity.
    d_support
        Topological dimension of the support; ``1`` for a time series.
    seed
        Recorded for provenance only; the estimator is deterministic.

    Returns
    -------
    ScalingExponents
        With ``estimator = "wtmm.dog<n>"``, the full structure surface attached,
        and provenance whose diagnostics carry the per-scale line counts.

    Examples
    --------
    >>> import numpy as np
    >>> from mmar.models.fbm import fbm
    >>> from mmar.estimators.wtmm import wtmm
    >>> path = fbm(2**13, 0.7, seed=0)
    >>> res = wtmm(path, q=np.linspace(-3, 3, 13))
    >>> bool(abs(res.h2 - 0.7) < 0.1)
    True
    """
    xa = validate_series(x, name="x")
    qa = validate_q_grid(q)
    structure = wtmm_structure(
        xa,
        q=qa,
        scales=scales,
        n_vanishing=n_vanishing,
        chain_tol=chain_tol,
        edge_factor=edge_factor,
        min_lines=min_lines,
        seed=seed,
    )
    a = structure.scales
    counts = structure.counts
    if counts is None:  # pragma: no cover - wtmm_structure always sets counts
        raise ValueError("the WTMM structure surface carries no line counts")
    weights = weighted_by_counts(counts)
    with np.errstate(divide="ignore"):
        log_z = np.log(structure.values)

    mask = np.ones(a.size, dtype=bool)
    selection = "manual"
    if select_range:
        pivot = int(np.argmin(np.abs(qa - 2.0)))
        sel = select_scaling_range(
            a,
            log_z[pivot],
            weights=weights,
            method=fit_method,
            log_y=False,
            min_points=min(8, a.size),
        )
        mask = np.asarray(sel.mask, dtype=bool)
        selection = f"auto-r2@q={qa[pivot]:.3g}"

    a_fit = a[mask]
    w_fit = weights[mask]
    fits: list[LogLogFit] = []
    for i in range(qa.size):
        fits.append(
            loglog_fit(a_fit, log_z[i][mask], method=fit_method, weights=w_fit, log_y=False)
        )

    tau = np.array([f.slope for f in fits], dtype=np.float64)
    stderr = np.array([f.slope_stderr for f in fits], dtype=np.float64)
    ci_lo = np.array([f.slope_ci[0] for f in fits], dtype=np.float64)
    ci_hi = np.array([f.slope_ci[1] for f in fits], dtype=np.float64)
    zeta = tau + float(d_support)

    scaling_range = ScalingRange(
        s_min=float(a_fit.min()),
        s_max=float(a_fit.max()),
        n_scales=int(a_fit.size),
        selection=selection,
        r_squared=float(np.median([f.r_squared for f in fits])),
        mask=tuple(bool(m) for m in mask),
    )

    params = {
        "n_vanishing": int(n_vanishing),
        "chain_tol": float(chain_tol),
        "edge_factor": float(edge_factor),
        "min_lines": int(min_lines),
        "fit_method": str(fit_method),
        "select_range": bool(select_range),
        "d_support": float(d_support),
        "n": int(xa.size),
    }
    prov = _provenance(
        f"wtmm.dog{int(n_vanishing)}",
        xa,
        q=qa,
        scales=np.asarray(np.round(a), dtype=np.int64),
        params=params,
        seed=seed,
    )
    diag: dict[str, Any] = {"scaling_range": scaling_range.to_canonical()}
    if structure.provenance is not None:
        diag.update(dict(structure.provenance.diagnostics))
    prov = prov.with_diagnostics(**diag)

    return ScalingExponents(
        q=qa,
        zeta=zeta,
        d_support=float(d_support),
        zeta_stderr=stderr,
        zeta_ci=(ci_lo + float(d_support), ci_hi + float(d_support)),
        fits=tuple(fits),
        scaling_range=scaling_range,
        structure=structure,
        estimator=f"wtmm.dog{int(n_vanishing)}",
        provenance=prov,
    )
