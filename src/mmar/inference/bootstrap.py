r"""Resampling confidence intervals for scaling exponents and Hölder spectra.

Why resampling, and not the delta method
----------------------------------------
Every exponent this library produces is a slope, and a slope has a textbook
standard error (:mod:`mmar.core.regression`). Propagating that standard error
into :math:`f(\alpha)` is where the textbook stops being usable: the spectrum is
a Legendre--Fenchel transform

.. math::
    f(\alpha) = \inf_q\{q\alpha - \zeta(q)\} + d ,

an *infimum*, hence non-smooth exactly at the interesting points. Its derivative
with respect to :math:`\hat\zeta` does not exist where the minimising :math:`q`
switches, and a first-order expansion around a kink is not an approximation of
anything. Worse, the map is a contraction of the non-concave part of
:math:`\hat\zeta` (see :func:`mmar.core.legendre.concave_envelope`), so
delta-method bars are simultaneously wrong in form and, empirically, far too
narrow. Resampling is the honest route: perturb the *data*, re-run the *whole*
pipeline, and look at what the estimator actually does.

Dependence is the entire difficulty
-----------------------------------
The iid bootstrap is inadmissible here. A multifractal series is by construction
long-range dependent; resampling its observations one at a time destroys every
correlation and returns a sampling distribution appropriate to white noise. The
resulting intervals are not slightly optimistic, they are optimistic by a factor
that diverges with the sample size: for a stationary sequence with
:math:`\gamma(k)\sim c\,k^{2H-2}`, :math:`H>1/2`,

.. math::
    \operatorname{Var}(\bar X_n) \sim \frac{c'}{n^{2-2H}}
    \qquad\text{against}\qquad \frac{\sigma^2}{n}\ \text{under iid},

a ratio :math:`n^{2H-1}\to\infty`. At :math:`H=0.7`, :math:`n=4096` that is a
factor of :math:`\approx 12` in variance, i.e. iid intervals about
:math:`3.5\times` too short.

The block bootstrap (Künsch 1989; Liu & Singh 1992) keeps blocks of
:math:`b` consecutive observations intact, so dependence at lags below
:math:`b` survives resampling and only the coupling across block joins is
destroyed. The block length is therefore the single parameter that matters, and
:func:`optimal_block_length` chooses it by the automatic rule of Politis & White
(2004) rather than by taste.

Failure modes, stated plainly
-----------------------------
* **Long memory beyond the block.** The block bootstrap is consistent for
  short-memory processes. On a long-memory series the resampled paths lose all
  dependence at lags :math:`> b`, so a statistic that reads scales above
  :math:`b` — which a multifractal estimator does by design — sees a resample
  whose exponent is pulled toward the uncorrelated value. Keep the estimator's
  largest scale well below :math:`b`, or read the resulting interval as a lower
  bound on the true uncertainty. Lahiri (2003, ch. 10) is the reference.
* **Path versus increments.** Block-resampling a self-affine *path* splices
  unrelated levels together and injects a jump at every block join; those jumps
  are :math:`\alpha=0` singularities and they bias :math:`\hat\zeta(q)`
  downward. Pass ``resample="increments"`` for anything that is a cumulated
  process (fBm, a log-price), which differences first and re-cumulates after.
* **A biased statistic.** :math:`\hat\zeta` is biased on finite samples, so a
  percentile interval — which is centred on the resample distribution, not on
  the truth — is miscalibrated. BCa (Efron 1987) corrects the median bias and
  the first-order skewness, and is the default here. When its acceleration is
  not computable the code falls back to percentile and *says so* in the result,
  rather than silently returning a differently-defined interval.

References
----------
Künsch, H.R. (1989). The jackknife and the bootstrap for general stationary
observations. *Annals of Statistics* 17(3), 1217--1241.

Politis, D.N. and Romano, J.P. (1992). A circular block-resampling procedure for
stationary data. In *Exploring the Limits of Bootstrap*, Wiley, 263--270.

Politis, D.N. and White, H. (2004). Automatic block-length selection for the
dependent bootstrap. *Econometric Reviews* 23(1), 53--70.

Patton, A., Politis, D.N. and White, H. (2009). Correction to "Automatic
block-length selection for the dependent bootstrap". *Econometric Reviews* 28(4),
372--375.

Efron, B. (1987). Better bootstrap confidence intervals. *JASA* 82(397),
171--185.

Efron, B. and Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*.
Chapman & Hall.

Lahiri, S.N. (2003). *Resampling Methods for Dependent Data*. Springer.

Freedman, D.A. (1981). Bootstrapping regression models. *Annals of Statistics*
9(6), 1218--1228.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfft, next_fast_len, rfft
from scipy.stats import norm

from mmar.core.legendre import spectrum_from_exponents
from mmar.core.provenance import InputRef, Provenance, new_provenance
from mmar.core.rng import SeedLike, seed_info, spawn_rngs
from mmar.core.types import FloatArray, LogLogFit, ScalingExponents
from mmar.core.validate import validate_series

__all__ = [
    "BlockLength",
    "BlockMethod",
    "BootstrapResult",
    "DescriptorBootstrap",
    "Estimator",
    "EstimatorContractError",
    "IntervalMethod",
    "ResampleTarget",
    "ResidualBootstrap",
    "SpectrumBootstrap",
    "bootstrap_descriptors",
    "bootstrap_exponents",
    "bootstrap_spectrum",
    "circular_block_bootstrap",
    "moving_block_bootstrap",
    "optimal_block_length",
    "politis_white_block_length",
    "residual_bootstrap_fit",
]

_VERSION = "1"

type Estimator = Callable[[FloatArray], ScalingExponents]
"""Anything that maps a series to fitted exponents. Deliberately a callable:
this module must not depend on which estimator produced the number."""

type Statistic = Callable[[FloatArray], FloatArray]
"""A vector-valued functional of the series; the object actually resampled."""

type BlockMethod = Literal["moving", "circular"]
type ResampleTarget = Literal["series", "increments"]
type IntervalMethod = Literal["percentile", "bca"]

DESCRIPTOR_NAMES: tuple[str, ...] = (
    "alpha_0",
    "width",
    "asymmetry",
    "alpha_min",
    "alpha_max",
    "f_max",
)

_ESTIMATOR_FAILURES = (ValueError, ArithmeticError, np.linalg.LinAlgError)
"""Exceptions treated as "this resample defeated the estimator" and recorded as
NaN. A :class:`EstimatorContractError` is deliberately *not* one of them."""


class EstimatorContractError(RuntimeError):
    """The estimator broke the contract the bootstrap needs (e.g. a moving q grid).

    Not a subclass of :class:`ValueError`, and so not swallowed as a per-resample
    failure: a contract violation is a bug in the caller's estimator and must
    surface immediately rather than being averaged into a NaN count.
    """


# --------------------------------------------------------------------------- #
# Block-length selection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class BlockLength:
    r"""Automatic block length with the intermediate quantities that produced it.

    The intermediates are not decoration: :attr:`bandwidth` hitting
    :attr:`bandwidth_max` means the correlogram never fell inside the noise band,
    which is the signature of long memory and the situation in which the whole
    rule is being applied outside its assumptions.
    """

    n: int
    m_hat: int
    """Smallest lag past which the correlogram stays inside the noise band."""
    bandwidth: int
    """:math:`M=\\min(2\\hat m, M_{\\max})`, the flat-top lag-window bandwidth."""
    bandwidth_max: int
    g_hat: float
    r""":math:`\hat G=\sum_j \lambda(j/M)\,|j|\,\hat R(j)`."""
    spectrum_at_zero: float
    r""":math:`\hat g(0)=\sum_j \lambda(j/M)\,\hat R(j)`, the long-run variance."""
    moving: int
    """Optimal block for the moving/circular block bootstrap."""
    stationary: int
    """Optimal mean block for the stationary bootstrap (Politis--Romano 1994)."""
    truncated: bool
    """Whether the raw optimum exceeded :math:`\\min(3\\sqrt n, n/3)`."""

    def to_canonical(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "m_hat": self.m_hat,
            "bandwidth": self.bandwidth,
            "moving": self.moving,
            "stationary": self.stationary,
            "truncated": self.truncated,
        }


def _autocovariance(x: FloatArray, max_lag: int) -> FloatArray:
    r"""Biased sample autocovariance :math:`\hat R(k)=n^{-1}\sum (x_t-\bar x)(x_{t+k}-\bar x)`.

    The :math:`1/n` (rather than :math:`1/(n-k)`) normalisation is required, not
    incidental: it is the one that makes :math:`(\hat R(|i-j|))_{ij}` positive
    semidefinite, hence :math:`\hat g(0)\ge 0` in the Politis--White formulas.
    Computed by FFT, so the cost is :math:`O(n\log n)` for all lags at once.
    """
    n = x.size
    xc = x - x.mean()
    nfft = next_fast_len(2 * n)
    spec = rfft(xc, nfft)
    acov = irfft(spec * np.conjugate(spec), nfft)[: max_lag + 1] / n
    return np.asarray(acov, dtype=np.float64)


def _flat_top_window(m: int) -> FloatArray:
    r"""Trapezoidal ("flat-top") lag window of Politis & Romano (1995) at lags :math:`0..M`.

    .. math::
        \lambda(s) = \begin{cases}
            1, & 0\le|s|\le \tfrac12,\\
            2(1-|s|), & \tfrac12<|s|\le 1,\\
            0, & |s|>1.
        \end{cases}

    Flat-top windows have bias of *arbitrarily high* order, which is what lets
    the block-length rule below be consistent without a bandwidth that must be
    tuned by hand.
    """
    s = np.arange(m + 1, dtype=np.float64) / float(m)
    return np.asarray(np.where(s <= 0.5, 1.0, np.maximum(2.0 * (1.0 - s), 0.0)), dtype=np.float64)


def politis_white_block_length(x: Sequence[float] | NDArray[Any], *, c: float = 2.0) -> BlockLength:
    r"""Automatic block length by the Politis--White (2004) rule, with diagnostics.

    The MSE-optimal block for the circular/moving block bootstrap of the sample
    mean is

    .. math::
        b^\ast = \left(\frac{2\,\hat G^2}{\hat D}\right)^{1/3} n^{1/3},
        \qquad
        \hat G = \sum_{|j|\le M}\lambda\!\Big(\frac{j}{M}\Big)|j|\hat R(j),
        \qquad
        \hat D = \tfrac43\,\hat g(0)^2,

    with :math:`\hat g(0)=\sum_{|j|\le M}\lambda(j/M)\hat R(j)` the flat-top
    estimate of the spectral density at the origin (:math:`2\pi` times it, in the
    convention where :math:`\hat D=\tfrac43 g(0)^2`; the constant is
    :math:`2 g(0)^2` for the stationary bootstrap, which is the only difference
    between the two rules). :math:`\hat G` is the first-order bias constant: it
    measures how much correlation mass sits at *long* lags, so a series with more
    memory gets a longer block, monotonically.

    Bandwidth selection follows Patton, Politis & White (2009):
    :math:`M = \min(2\hat m, M_{\max})` where :math:`\hat m` is the smallest lag
    after which :math:`|\hat\rho|` stays below :math:`c\sqrt{\log_{10}n/n}` for
    :math:`K_n=\max(5,\lceil\sqrt{\log_{10}n}\rceil)` consecutive lags, and
    :math:`M_{\max}=\lceil\sqrt n\rceil+K_n`. The optimum is truncated at
    :math:`\lceil\min(3\sqrt n,\,n/3)\rceil`.

    Assumptions and failure modes
    -----------------------------
    The rule is derived for the sample *mean* of a short-memory (absolutely
    summable autocovariance) process. Two consequences for this library:

    1. Applied to a long-memory series (:math:`H>1/2`) the correlogram never
       enters the noise band, :math:`\hat m` saturates at :math:`M_{\max}`, and
       :math:`b^\ast` **under**-states the block needed. :attr:`BlockLength.truncated`
       and ``bandwidth == bandwidth_max`` flag this.
    2. A scaling exponent is not a sample mean. It is a smooth function of
       log-moments across scales, and the optimal block for it grows with the
       largest scale the estimator reads. Treat :math:`b^\ast` as a floor and
       raise it above the coarsest scale used.

    Parameters
    ----------
    x
        The stationary series to be resampled (increments, not a path).
    c
        Multiplier of the noise band; ``2.0`` is the value in the paper.

    Returns
    -------
    BlockLength
        With ``moving`` and ``stationary`` both at least ``1``.
    """
    a = validate_series(x, name="series", min_length=8, allow_constant=False)
    n = a.size
    log10n = math.log10(n)
    k_n = max(5, math.ceil(math.sqrt(max(log10n, 1.0))))
    m_max = min(math.ceil(math.sqrt(n)) + k_n, n - 1)
    b_max = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))

    acov = _autocovariance(a, m_max)
    rho = np.abs(acov[1:] / acov[0])
    threshold = c * math.sqrt(log10n / n)

    # m_hat: first lag beginning a run of k_n consecutive insignificant
    # correlations. If no such run exists the correlogram never dies, and the
    # PPW (2009) fallback is the last significant lag.
    below = rho < threshold
    m_hat = m_max
    if below.size >= k_n:
        runs = np.lib.stride_tricks.sliding_window_view(below, k_n).all(axis=1)
        hit = np.nonzero(runs)[0]
        if hit.size > 0:
            m_hat = int(hit[0]) + 1
        else:
            significant = np.nonzero(~below)[0]
            m_hat = int(significant[-1]) + 1 if significant.size > 0 else 1
    bandwidth = max(1, min(2 * m_hat, m_max))

    lam = _flat_top_window(bandwidth)
    lags = np.arange(bandwidth + 1, dtype=np.float64)
    r = acov[: bandwidth + 1]
    g_hat = float(2.0 * np.sum(lam[1:] * lags[1:] * r[1:]))
    g_zero = float(r[0] + 2.0 * np.sum(lam[1:] * r[1:]))

    def _optimum(d_hat: float) -> tuple[int, bool]:
        if not math.isfinite(d_hat) or d_hat <= 0.0 or not math.isfinite(g_hat):
            return 1, False
        raw = (2.0 * g_hat**2 / d_hat) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
        if not math.isfinite(raw):  # pragma: no cover - guarded by d_hat > 0
            return 1, False
        return max(1, min(math.ceil(raw), b_max)), raw > b_max

    b_cb, trunc_cb = _optimum((4.0 / 3.0) * g_zero**2)
    b_sb, _ = _optimum(2.0 * g_zero**2)

    return BlockLength(
        n=n,
        m_hat=m_hat,
        bandwidth=bandwidth,
        bandwidth_max=m_max,
        g_hat=g_hat,
        spectrum_at_zero=g_zero,
        moving=b_cb,
        stationary=b_sb,
        truncated=trunc_cb,
    )


def optimal_block_length(
    x: Sequence[float] | NDArray[Any], *, method: BlockMethod = "moving", c: float = 2.0
) -> int:
    r"""Politis--White (2004) block length as a single integer.

    ``"moving"`` and ``"circular"`` return the same value: Lahiri (1999) shows
    the two schemes share the same MSE-optimal block to first order, the
    difference between them being an :math:`O(b/n)` edge effect. See
    :func:`politis_white_block_length` for the derivation, the diagnostics and
    the long-memory caveat.
    """
    if method not in ("moving", "circular"):
        raise ValueError(f"unknown block method {method!r}; use 'moving' or 'circular'")
    return politis_white_block_length(x, c=c).moving


# --------------------------------------------------------------------------- #
# Block resampling schemes
# --------------------------------------------------------------------------- #


def _check_block(block: int, n: int, *, circular: bool) -> int:
    b = int(block)
    if b < 1:
        raise ValueError(f"block length must be at least 1, got {block!r}")
    if b > n:
        raise ValueError(f"block length {b} exceeds the series length {n}")
    if not circular and b == n:
        raise ValueError(
            f"block length {b} equals the series length: the moving-block scheme has a "
            "single admissible block and every resample is the original series"
        )
    return b


def _block_indices(
    rng: np.random.Generator, n: int, block: int, *, circular: bool
) -> NDArray[np.intp]:
    """Indices of one resample: ``ceil(n/b)`` blocks concatenated and truncated."""
    n_blocks = -(-n // block)
    n_starts = n if circular else n - block + 1
    starts = rng.integers(0, n_starts, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    if circular:
        idx %= n
    return np.asarray(idx, dtype=np.intp)


def moving_block_bootstrap(
    x: Sequence[float] | NDArray[Any],
    *,
    block: int,
    n_resamples: int,
    seed: SeedLike,
) -> Iterator[FloatArray]:
    r"""Moving-block bootstrap of Künsch (1989).

    The :math:`n-b+1` overlapping blocks
    :math:`B_i=(x_i,\dots,x_{i+b-1})`, :math:`i=1,\dots,n-b+1`, are drawn
    :math:`\lceil n/b\rceil` times with replacement and concatenated; the result
    is truncated to length :math:`n`. Dependence at lags :math:`<b` is preserved
    inside a block and destroyed across joins, so the scheme is consistent for
    smooth functionals of a short-memory process with :math:`b\to\infty`,
    :math:`b/n\to 0` (Künsch 1989, Thm. 3.1).

    Two known biases, neither of which this function hides:

    * **Edge effect.** Observations near the ends of the series belong to fewer
      of the :math:`n-b+1` blocks (:math:`x_1` and :math:`x_n` to exactly one),
      so they are under-represented. :func:`circular_block_bootstrap` removes it.
    * **Non-stationarity of the resample.** The resampled sequence is not
      stationary — its law depends on position within a block — which is why the
      bootstrap distribution of a *mean* is centred on a slightly shrunk value.

    Parameters
    ----------
    block
        Block length :math:`b`; see :func:`optimal_block_length`.
    n_resamples
        Number of resamples the iterator will yield.
    seed
        Threaded through :func:`~mmar.core.rng.spawn_rngs`, so resample ``k``
        depends only on ``(seed, k)`` — never on how many were drawn before it,
        and never on the global numpy state.

    Yields
    ------
    FloatArray
        Fresh arrays of length ``len(x)``. Each is a new allocation; the caller
        may keep them.
    """
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    if a.ndim != 1:  # pragma: no cover - ravel guarantees 1-D
        raise ValueError("x must be one-dimensional")
    b = _check_block(block, a.size, circular=False)
    if int(n_resamples) < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples!r}")
    for rng in spawn_rngs(seed, int(n_resamples)):
        yield a[_block_indices(rng, a.size, b, circular=False)]


def circular_block_bootstrap(
    x: Sequence[float] | NDArray[Any],
    *,
    block: int,
    n_resamples: int,
    seed: SeedLike,
) -> Iterator[FloatArray]:
    r"""Circular block bootstrap of Politis & Romano (1992).

    Identical to :func:`moving_block_bootstrap` except that the series is wrapped
    onto a circle, :math:`x_{n+j}:=x_j`, so there are :math:`n` blocks rather
    than :math:`n-b+1` and **every observation belongs to exactly** :math:`b`
    **of them**. Equal resampling probability makes the resample mean unbiased
    for :math:`\bar x`, which removes the :math:`O(b/n)` centring bias of the
    moving-block scheme.

    The price is an artificial join between :math:`x_n` and :math:`x_1`. For a
    series with a trend or a level shift that seam is a large spurious
    increment, so wrap only what is plausibly stationary.
    """
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    b = _check_block(block, a.size, circular=True)
    if int(n_resamples) < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples!r}")
    for rng in spawn_rngs(seed, int(n_resamples)):
        yield a[_block_indices(rng, a.size, b, circular=True)]


# --------------------------------------------------------------------------- #
# Resampling target: path versus increments
# --------------------------------------------------------------------------- #


def _units_and_rebuild(
    x: FloatArray, target: ResampleTarget
) -> tuple[FloatArray, Callable[[FloatArray], FloatArray]]:
    """Split the series into the objects that may be permuted and the way back.

    ``"increments"`` is the correct choice for any cumulated process: the
    stationary object is :math:`\\Delta x`, and the path is rebuilt by
    :math:`x_0 + \\operatorname{cumsum}`.
    """
    if target == "series":
        return x, lambda u: u
    if target == "increments":
        first = float(x[0])
        return np.diff(x), lambda u: np.concatenate(([first], first + np.cumsum(u)))
    raise ValueError(f"unknown resample target {target!r}; use 'series' or 'increments'")


def _resample_iterator(
    units: FloatArray,
    *,
    block: int,
    n_resamples: int,
    seed: SeedLike,
    method: BlockMethod,
) -> Iterator[FloatArray]:
    if method == "moving":
        return moving_block_bootstrap(units, block=block, n_resamples=n_resamples, seed=seed)
    if method == "circular":
        return circular_block_bootstrap(units, block=block, n_resamples=n_resamples, seed=seed)
    raise ValueError(f"unknown block method {method!r}; use 'moving' or 'circular'")


# --------------------------------------------------------------------------- #
# Replicate collection
# --------------------------------------------------------------------------- #


def _collect(
    stat: Statistic, series: Iterator[FloatArray], k: int, n_resamples: int
) -> tuple[FloatArray, int]:
    """Evaluate the statistic on every resample; raised failures become NaN rows.

    An estimator can legitimately fail on a resample (a negative moment that no
    longer exists, a scaling range that collapses). Recording NaN and counting it
    is the honest response; dropping the resample silently would condition the
    bootstrap distribution on success and shrink the interval.

    Returns the ``(n_resamples, k)`` matrix and the number of resamples on which
    the statistic *raised*. That count is deliberately distinct from the number
    of NaN entries: a spectrum replicate is NaN wherever the resample's own
    :math:`\\alpha` support misses a grid point, which is information about the
    spectrum, not an error.
    """
    out = np.full((n_resamples, k), np.nan, dtype=np.float64)
    n_failed = 0
    for i, y in enumerate(series):
        try:
            v = np.asarray(stat(y), dtype=np.float64)
        except _ESTIMATOR_FAILURES:
            n_failed += 1
            continue
        if v.shape != (k,):
            raise ValueError(f"statistic returned shape {v.shape} on a resample, expected ({k},)")
        out[i] = v
    return out, n_failed


def _block_jackknife(
    units: FloatArray,
    rebuild: Callable[[FloatArray], FloatArray],
    stat: Statistic,
    *,
    block: int,
    k: int,
    max_blocks: int | None,
) -> FloatArray:
    r"""Delete-one-block jackknife values, shape ``(n_blocks, k)``.

    The ordinary delete-one-*observation* jackknife is invalid under dependence:
    removing a single point leaves the correlation structure intact and the
    resulting influence values understate the variability of the statistic
    (Künsch 1989, §3). The delete-one-block version partitions the series into
    :math:`\lfloor m/b \rfloor` contiguous blocks and deletes one at a time,
    which is the jackknife matched to the block bootstrap.

    ``max_blocks`` caps the number of deletions by taking an evenly spaced
    subset. The acceleration constant is a *ratio* of third to :math:`3/2`-power
    of second empirical moments of the same values, so a representative subset
    estimates it consistently; the cap trades a little precision in :math:`\hat a`
    for a linear reduction in estimator calls.
    """
    m = units.size
    n_blocks = m // block
    if n_blocks < 3:
        return np.empty((0, k), dtype=np.float64)
    starts = np.arange(n_blocks) * block
    if max_blocks is not None and n_blocks > max_blocks:
        starts = starts[np.linspace(0, n_blocks - 1, int(max_blocks)).round().astype(np.intp)]
    keep = np.ones(m, dtype=bool)
    values = np.full((starts.size, k), np.nan, dtype=np.float64)
    for i, s in enumerate(starts):
        keep[s : s + block] = False
        try:
            v = np.asarray(stat(rebuild(units[keep])), dtype=np.float64)
            if v.shape == (k,):
                values[i] = v
        except _ESTIMATOR_FAILURES:
            pass
        keep[s : s + block] = True
    return values


# --------------------------------------------------------------------------- #
# Interval construction
# --------------------------------------------------------------------------- #


def _quantile_columns(ordered: FloatArray, counts: NDArray[np.int64], p: FloatArray) -> FloatArray:
    """Column-wise quantile of a NaN-padded, column-sorted matrix.

    ``ordered`` must be ``np.sort(reps, axis=0)`` (NaN sorts last) and ``counts``
    the number of finite entries per column. Linear interpolation between order
    statistics, i.e. numpy's default ``'linear'`` method, evaluated at a
    *different* probability per column — which is exactly what BCa needs and what
    :func:`numpy.nanquantile` cannot do.
    """
    k = ordered.shape[1]
    m = counts.astype(np.float64)
    h = np.clip((m - 1.0) * np.clip(p, 0.0, 1.0), 0.0, None)
    lo = np.floor(h).astype(np.intp)
    hi = np.minimum(lo + 1, np.maximum(counts.astype(np.intp) - 1, 0))
    cols = np.arange(k)
    a, b = ordered[lo, cols], ordered[hi, cols]
    out = a + (h - lo) * (b - a)
    return np.asarray(np.where(counts > 0, out, np.nan), dtype=np.float64)


def _column_moments(values: FloatArray) -> tuple[FloatArray, FloatArray]:
    """NaN-aware column mean and sample sd, without ever touching an empty slice.

    :func:`numpy.nanmean` and :func:`numpy.nanstd` emit a ``RuntimeWarning`` on an
    all-NaN column, which is a routine occurrence here (a grid point no resample
    reached) and must not be treated as an anomaly.
    """
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    safe = np.where(finite, values, 0.0)
    mean = np.where(counts > 0, safe.sum(axis=0) / np.maximum(counts, 1), np.nan)
    dev = np.where(finite, (values - mean[None, :]) ** 2, 0.0)
    var = np.where(counts > 1, dev.sum(axis=0) / np.maximum(counts - 1, 1), np.nan)
    return (
        np.asarray(mean, dtype=np.float64),
        np.asarray(np.sqrt(var), dtype=np.float64),
    )


def _acceleration(jack: FloatArray) -> FloatArray:
    r"""Jackknife acceleration :math:`\hat a` per column.

    .. math::
        \hat a = \frac{\sum_i (\bar\theta_{(\cdot)}-\theta_{(i)})^3}
                      {6\big[\sum_i (\bar\theta_{(\cdot)}-\theta_{(i)})^2\big]^{3/2}} ,

    the standardised skewness of the influence values (Efron 1987, eq. 6.6). It
    corrects the first-order effect of a standard error that varies with the
    parameter — the reason a percentile interval is asymmetric in the wrong
    direction for a variance-like statistic. Returns NaN where the denominator
    vanishes or the jackknife failed.
    """
    if jack.shape[0] < 3:
        return np.full(jack.shape[1], np.nan, dtype=np.float64)
    finite = np.isfinite(jack)
    counts = finite.sum(axis=0)
    centred = np.where(finite, jack, np.nan)
    with np.errstate(invalid="ignore"):
        mean = np.where(counts > 0, np.nansum(centred, axis=0) / np.maximum(counts, 1), np.nan)
        dev = np.where(finite, mean[None, :] - centred, 0.0)
        s2 = np.sum(dev**2, axis=0)
        s3 = np.sum(dev**3, axis=0)
        a = np.where(s2 > 0.0, s3 / (6.0 * np.power(np.maximum(s2, 1e-300), 1.5)), np.nan)
    return np.asarray(np.where(counts >= 3, a, np.nan), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _Intervals:
    lower: FloatArray
    upper: FloatArray
    z0: FloatArray
    acceleration: FloatArray
    per_point_method: tuple[str, ...]
    counts: NDArray[np.int64]


def _intervals(
    reps: FloatArray,
    point: FloatArray,
    jack: FloatArray | None,
    *,
    ci: float,
    interval: IntervalMethod,
    min_valid: int,
) -> _Intervals:
    r"""Percentile or BCa endpoints, column by column, with per-column fallback.

    BCa (Efron 1987) replaces the nominal probabilities
    :math:`(\alpha/2, 1-\alpha/2)` by

    .. math::
        \alpha_1 = \Phi\!\left(\hat z_0 +
            \frac{\hat z_0 + z_{\alpha/2}}{1-\hat a(\hat z_0+z_{\alpha/2})}\right),
        \qquad
        \alpha_2 = \Phi\!\left(\hat z_0 +
            \frac{\hat z_0 + z_{1-\alpha/2}}{1-\hat a(\hat z_0+z_{1-\alpha/2})}\right),

    with the bias correction
    :math:`\hat z_0=\Phi^{-1}\big(\#\{\hat\theta^*_b<\hat\theta\}/B\big)` and the
    acceleration :math:`\hat a` from :func:`_acceleration`. It is
    transformation-respecting and second-order accurate, against first-order for
    the percentile interval — which matters precisely because
    :math:`\hat\zeta` is biased.

    Falls back to percentile, per column and on the record, when: the column has
    too few valid replicates; every replicate lies on one side of the point
    estimate (:math:`\hat z_0=\pm\infty`); the acceleration is not computable; or
    the BCa denominator :math:`1-\hat a(\hat z_0+z)` is non-positive, where the
    map ceases to be increasing and the "interval" would be inverted.
    """
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must lie strictly in (0, 1), got {ci!r}")
    k = reps.shape[1]
    finite = np.isfinite(reps)
    counts = finite.sum(axis=0).astype(np.int64)
    ordered = np.sort(np.where(finite, reps, np.nan), axis=0)

    p_lo = (1.0 - ci) / 2.0
    p_hi = 1.0 - p_lo
    prob_lo = np.full(k, p_lo, dtype=np.float64)
    prob_hi = np.full(k, p_hi, dtype=np.float64)
    z0 = np.full(k, np.nan, dtype=np.float64)
    accel = np.full(k, np.nan, dtype=np.float64)
    usable = counts >= min_valid
    method = np.where(usable, "percentile", "insufficient")

    if interval == "bca" and jack is not None:
        below = np.count_nonzero(np.where(finite, reps, np.inf) < point[None, :], axis=0)
        frac = np.where(counts > 0, below / np.maximum(counts, 1), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            z0 = np.asarray(norm.ppf(frac), dtype=np.float64)
        accel = _acceleration(jack)
        z_lo, z_hi = float(norm.ppf(p_lo)), float(norm.ppf(p_hi))
        with np.errstate(invalid="ignore", divide="ignore"):
            den_lo = 1.0 - accel * (z0 + z_lo)
            den_hi = 1.0 - accel * (z0 + z_hi)
            adj_lo = z0 + (z0 + z_lo) / den_lo
            adj_hi = z0 + (z0 + z_hi) / den_hi
        ok = (
            usable
            & np.isfinite(z0)
            & np.isfinite(accel)
            & (den_lo > 0.0)
            & (den_hi > 0.0)
            & np.isfinite(adj_lo)
            & np.isfinite(adj_hi)
        )
        prob_lo = np.where(ok, norm.cdf(adj_lo), prob_lo)
        prob_hi = np.where(ok, norm.cdf(adj_hi), prob_hi)
        method = np.where(ok, "bca", method)
    elif interval not in ("percentile", "bca"):
        raise ValueError(f"unknown interval {interval!r}; use 'percentile' or 'bca'")

    lower = _quantile_columns(ordered, counts, prob_lo)
    upper = _quantile_columns(ordered, counts, prob_hi)
    blank = ~usable
    lower = np.where(blank, np.nan, lower)
    upper = np.where(blank, np.nan, upper)
    return _Intervals(
        lower=np.asarray(lower, dtype=np.float64),
        upper=np.asarray(upper, dtype=np.float64),
        z0=z0,
        acceleration=accel,
        per_point_method=tuple(str(s) for s in method),
        counts=counts,
    )


def _summary_notes(iv: _Intervals, interval: IntervalMethod, n_failed: int) -> tuple[str, ...]:
    notes: list[str] = []
    n_bca = sum(1 for m in iv.per_point_method if m == "bca")
    n_pct = sum(1 for m in iv.per_point_method if m == "percentile")
    n_none = sum(1 for m in iv.per_point_method if m == "insufficient")
    if interval == "bca":
        notes.append(f"BCa at {n_bca} point(s); percentile fallback at {n_pct}")
        if n_pct:
            notes.append(
                "percentile fallback means the acceleration or the bias correction was "
                "not computable there; those intervals are only first-order accurate"
            )
    else:
        notes.append(f"percentile interval at {n_pct} point(s)")
    if n_none:
        notes.append(f"{n_none} point(s) reported as NaN: too few valid resamples")
    if n_failed:
        notes.append(f"{n_failed} resample(s) failed the estimator and were recorded as NaN")
    return tuple(notes)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    r"""Bootstrap distribution and confidence band for :math:`\zeta(q)`.

    The replicate matrix is kept, not just its summary: a bimodal or heavily
    truncated bootstrap distribution invalidates any interval read off it, and
    that is only visible in the raw replicates.
    """

    q: FloatArray
    point: FloatArray
    r"""The estimate on the original series, :math:`\hat\zeta(q)`."""
    replicates: FloatArray
    """Shape ``(n_resamples, n_q)``; NaN where the estimator failed."""
    lower: FloatArray
    upper: FloatArray
    ci_level: float
    interval: IntervalMethod
    per_point_method: tuple[str, ...]
    """``"bca"``, ``"percentile"`` or ``"insufficient"`` for each ``q``."""
    z0: FloatArray
    acceleration: FloatArray
    n_valid: NDArray[np.int64]
    n_resamples: int
    n_failed: int
    block: int
    method: BlockMethod
    resample_target: ResampleTarget
    notes: tuple[str, ...]
    provenance: Provenance | None = None

    @property
    def bias(self) -> FloatArray:
        r"""Bootstrap bias estimate :math:`\overline{\hat\zeta^*}-\hat\zeta`.

        Reported, never subtracted: bias correction inflates the variance, and a
        corrected point estimate paired with an uncorrected interval is
        incoherent. BCa already accounts for the *median* bias.
        """
        return np.asarray(_column_moments(self.replicates)[0] - self.point, dtype=np.float64)

    @property
    def stderr(self) -> FloatArray:
        """Bootstrap standard error: the sd of the replicate distribution."""
        return _column_moments(self.replicates)[1]

    @property
    def width(self) -> FloatArray:
        return np.asarray(self.upper - self.lower, dtype=np.float64)

    def at(self, q: float) -> tuple[float, float, float]:
        """``(point, lower, upper)`` at the grid ``q`` nearest the request."""
        i = int(np.argmin(np.abs(self.q - q)))
        return float(self.point[i]), float(self.lower[i]), float(self.upper[i])

    def covers(self, q: float, value: float) -> bool:
        """Whether the interval at ``q`` contains ``value``; False if it is NaN."""
        _, lo, hi = self.at(q)
        return bool(math.isfinite(lo) and math.isfinite(hi) and lo <= value <= hi)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "n_resamples": self.n_resamples,
            "block": self.block,
            "method": str(self.method),
            "interval": str(self.interval),
            "ci_level": self.ci_level,
            "n_failed": self.n_failed,
        }


@dataclass(frozen=True, slots=True)
class SpectrumBootstrap:
    r"""Confidence band for :math:`f(\alpha)` on a fixed :math:`\alpha` grid.

    Each resample produces a spectrum on *its own* :math:`\alpha` support, so the
    band only exists after every replicate has been interpolated onto one common
    grid. :attr:`coverage` is the fraction of resamples whose support actually
    contained each grid point: near the edges of the spectrum this falls away,
    and where it falls below the required threshold the interval is NaN rather
    than a narrow band computed from the handful of resamples that happened to
    reach that far. A narrow interval built from few replicates is the most
    misleading object this module could return.
    """

    alpha: FloatArray
    point: FloatArray
    replicates: FloatArray
    """Shape ``(n_resamples, n_alpha)``; NaN outside a resample's own support."""
    lower: FloatArray
    upper: FloatArray
    coverage: FloatArray
    """Fraction of resamples covering each grid point, in :math:`[0,1]`."""
    min_coverage: float
    ci_level: float
    interval: IntervalMethod
    per_point_method: tuple[str, ...]
    n_valid: NDArray[np.int64]
    n_resamples: int
    n_failed: int
    block: int
    method: BlockMethod
    resample_target: ResampleTarget
    notes: tuple[str, ...]
    provenance: Provenance | None = None

    @property
    def resolved(self) -> NDArray[np.bool_]:
        """Grid points with a usable interval."""
        return np.asarray(np.isfinite(self.lower) & np.isfinite(self.upper))

    def to_canonical(self) -> dict[str, Any]:
        return {
            "n_resamples": self.n_resamples,
            "block": self.block,
            "method": str(self.method),
            "interval": str(self.interval),
            "ci_level": self.ci_level,
            "min_coverage": self.min_coverage,
            "n_resolved": int(self.resolved.sum()),
        }


@dataclass(frozen=True, slots=True)
class DescriptorBootstrap:
    r"""Intervals for the scalar summaries of a spectrum.

    These — :math:`\Delta\alpha`, :math:`\alpha_0`, the asymmetry — are what a
    paper quotes, and they are functionals of the *whole* estimated curve, so
    their sampling distribution cannot be read off pointwise bands on
    :math:`f(\alpha)`. They have to be resampled directly, which is what this is.
    """

    names: tuple[str, ...]
    point: FloatArray
    replicates: FloatArray
    lower: FloatArray
    upper: FloatArray
    ci_level: float
    interval: IntervalMethod
    per_point_method: tuple[str, ...]
    z0: FloatArray
    acceleration: FloatArray
    n_valid: NDArray[np.int64]
    n_resamples: int
    n_failed: int
    block: int
    method: BlockMethod
    resample_target: ResampleTarget
    notes: tuple[str, ...]
    provenance: Provenance | None = None

    @property
    def stderr(self) -> FloatArray:
        """Bootstrap standard error of each descriptor."""
        return _column_moments(self.replicates)[1]

    def interval_for(self, name: str) -> tuple[float, float, float]:
        """``(point, lower, upper)`` for one descriptor."""
        if name not in self.names:
            raise KeyError(f"no descriptor {name!r}; have {list(self.names)}")
        i = self.names.index(name)
        return float(self.point[i]), float(self.lower[i]), float(self.upper[i])

    def table(self) -> dict[str, tuple[float, float, float]]:
        return {n: self.interval_for(n) for n in self.names}

    def to_canonical(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "n_resamples": self.n_resamples,
            "block": self.block,
            "interval": str(self.interval),
            "ci_level": self.ci_level,
        }


@dataclass(frozen=True, slots=True)
class ResidualBootstrap:
    """Residual-bootstrap distribution of one log--log slope."""

    slope: float
    replicates: FloatArray
    lower: float
    upper: float
    stderr: float
    ci_level: float
    n_resamples: int
    n_points: int
    notes: tuple[str, ...]
    provenance: Provenance | None = None

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def to_canonical(self) -> dict[str, Any]:
        return {
            "slope": self.slope,
            "lower": self.lower,
            "upper": self.upper,
            "n_resamples": self.n_resamples,
            "ci_level": self.ci_level,
        }


# --------------------------------------------------------------------------- #
# Public bootstrap drivers
# --------------------------------------------------------------------------- #


def _check_ci(ci: float) -> float:
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must lie strictly in (0, 1), got {ci!r}")
    return float(ci)


def _resolve_block(units: FloatArray, block: int | None, method: BlockMethod) -> tuple[int, str]:
    """The block length actually used, and where it came from."""
    if block is not None:
        return _check_block(block, units.size, circular=method == "circular"), "manual"
    return optimal_block_length(units, method=method), "politis-white"


def _provenance_for(
    name: str,
    x: FloatArray,
    *,
    params: dict[str, Any],
    seed: SeedLike,
    notes: tuple[str, ...],
    parent: Provenance | None,
) -> Provenance:
    inputs = [InputRef.from_array("x", x)]
    if parent is not None:
        inputs.append(InputRef.from_provenance("point_estimate", parent))
    return new_provenance(
        name,
        estimator_version=_VERSION,
        params=params,
        inputs=inputs,
        seed=seed_info(seed),
        notes=notes,
    )


def bootstrap_exponents(
    x: Sequence[float] | NDArray[Any],
    estimator: Estimator,
    *,
    n_resamples: int = 200,
    block: int | None = None,
    seed: SeedLike,
    ci: float = 0.95,
    method: BlockMethod = "moving",
    resample: ResampleTarget = "series",
    interval: IntervalMethod = "bca",
    jackknife_blocks: int | None = 40,
    min_valid_fraction: float = 0.5,
) -> BootstrapResult:
    r"""Block-bootstrap confidence band for :math:`\zeta(q)`.

    The *series* is resampled and the estimator re-run on each resample, so the
    interval reflects everything the estimator does — the scaling-range
    selection, the weighting, the fits — and not merely the regression step. This
    is the only construction that can be quoted without a caveat about which part
    of the pipeline the uncertainty refers to.

    Parameters
    ----------
    x
        The series, exactly as it would be passed to ``estimator``.
    estimator
        ``x -> ScalingExponents``. Must use the same ``q`` grid for every input;
        a data-dependent grid makes the replicates incomparable and is rejected.
    n_resamples
        Number of block resamples :math:`B`. 200 suffices for a standard error,
        but a BCa *interval* reads the tails of the replicate distribution and
        wants :math:`B\ge 1000` for a 95% level; below that the endpoints are
        themselves noisy.
    block
        Block length. ``None`` calls :func:`optimal_block_length` on the
        resampling units. Read the long-memory caveat there: for a multifractal
        series this is a floor, not an answer.
    seed
        Required. Resample ``k`` is a pure function of ``(seed, k)``.
    ci
        Nominal two-sided coverage.
    method
        ``"moving"`` (Künsch) or ``"circular"`` (Politis--Romano).
    resample
        ``"series"`` resamples what you passed. ``"increments"`` differences
        first and re-cumulates, which is mandatory for a self-affine path — see
        the module docstring.
    interval
        ``"bca"`` (default) or ``"percentile"``. BCa costs one delete-one-block
        jackknife pass, i.e. up to ``jackknife_blocks`` extra estimator calls.
    jackknife_blocks
        Cap on the number of deleted blocks. ``None`` uses all of them.
    min_valid_fraction
        A ``q`` whose replicates failed more often than this gets a NaN interval.

    Returns
    -------
    BootstrapResult
        Carrying the point estimate, the full replicate matrix, the band, the
        per-``q`` interval method actually used, and provenance.

    Raises
    ------
    EstimatorContractError
        If the estimator returns a different ``q`` grid on a resample.
    ValueError
        If every resample failed the estimator.
    """
    a = validate_series(x, name="series")
    ci = _check_ci(ci)
    point_exp = estimator(a)
    q = np.asarray(point_exp.q, dtype=np.float64)
    k = q.size
    point = np.asarray(point_exp.zeta, dtype=np.float64)

    def stat(y: FloatArray) -> FloatArray:
        out = estimator(y)
        if out.q.shape != q.shape or not np.allclose(out.q, q):
            raise EstimatorContractError(
                "the estimator returned a different q grid on a resample; bootstrap "
                "replicates must be comparable point by point. Fix the q grid."
            )
        return np.asarray(out.zeta, dtype=np.float64)

    units, rebuild = _units_and_rebuild(a, resample)
    b, block_source = _resolve_block(units, block, method)
    reps, n_failed = _collect(
        stat,
        (
            rebuild(u)
            for u in _resample_iterator(
                units, block=b, n_resamples=int(n_resamples), seed=seed, method=method
            )
        ),
        k,
        int(n_resamples),
    )
    if n_failed == int(n_resamples):
        raise ValueError(
            f"every one of the {n_resamples} resamples failed the estimator; the block "
            f"length ({b}) or the estimator's scale range is incompatible with the resample"
        )

    jack = (
        _block_jackknife(units, rebuild, stat, block=b, k=k, max_blocks=jackknife_blocks)
        if interval == "bca"
        else None
    )
    iv = _intervals(
        reps,
        point,
        jack,
        ci=ci,
        interval=interval,
        min_valid=max(3, int(min_valid_fraction * n_resamples)),
    )
    notes = _summary_notes(iv, interval, n_failed)
    if jack is not None and jack.shape[0] == 0:
        notes = (*notes, "delete-one-block jackknife unavailable: fewer than three blocks")
    params: dict[str, Any] = {
        "n_resamples": int(n_resamples),
        "block": int(b),
        "block_source": block_source,
        "method": str(method),
        "resample": str(resample),
        "interval": str(interval),
        "ci": float(ci),
        "jackknife_blocks": jackknife_blocks,
        "estimator": point_exp.estimator,
    }
    return BootstrapResult(
        q=q,
        point=point,
        replicates=reps,
        lower=iv.lower,
        upper=iv.upper,
        ci_level=float(ci),
        interval=interval,
        per_point_method=iv.per_point_method,
        z0=iv.z0,
        acceleration=iv.acceleration,
        n_valid=iv.counts,
        n_resamples=int(n_resamples),
        n_failed=n_failed,
        block=int(b),
        method=method,
        resample_target=resample,
        notes=notes,
        provenance=_provenance_for(
            "inference.bootstrap_exponents",
            a,
            params=params,
            seed=seed,
            notes=notes,
            parent=point_exp.provenance,
        ),
    )


def bootstrap_spectrum(
    x: Sequence[float] | NDArray[Any],
    estimator: Estimator,
    *,
    n_resamples: int = 200,
    block: int | None = None,
    seed: SeedLike,
    ci: float = 0.95,
    method: BlockMethod = "moving",
    resample: ResampleTarget = "series",
    interval: IntervalMethod = "percentile",
    alpha_grid: FloatArray | None = None,
    n_alpha: int = 101,
    min_coverage: float = 0.8,
    spectrum_method: str = "fenchel",
    smoothing: float | None = None,
    jackknife_blocks: int | None = 40,
) -> SpectrumBootstrap:
    r"""Confidence band for :math:`f(\alpha)` on a fixed :math:`\alpha` grid.

    Each resample yields its own :math:`\hat\zeta`, hence its own spectrum on its
    own support :math:`[\min\hat\zeta',\max\hat\zeta']`. Those supports differ —
    that variation *is* part of the uncertainty — so the replicates are
    interpolated onto one common grid, with NaN outside each replicate's own
    support (:meth:`~mmar.core.types.HolderSpectrum.evaluate`), and the fraction
    of resamples reaching each grid point is recorded as :attr:`coverage`.

    A grid point covered by only a few resamples is reported as NaN. Doing
    otherwise produces the classic artefact of published :math:`f(\alpha)`
    figures: error bars that *narrow* toward the tails of the spectrum, because
    only the few resamples that reached that far are being averaged.

    Parameters
    ----------
    alpha_grid
        The common grid. ``None`` spans the point-estimate spectrum's support
        with ``n_alpha`` points.
    min_coverage
        Minimum fraction of resamples that must reach a grid point for an
        interval to be reported there.
    interval
        Percentile by default. BCa is available and uses the same delete-one-block
        jackknife, but the spectrum is an infimum of the resampled exponents, so
        its influence values are non-smooth and the acceleration is correspondingly
        less stable than for :math:`\zeta` itself.
    spectrum_method, smoothing
        Passed to :func:`~mmar.core.legendre.spectrum_from_exponents`. Keep them
        identical to whatever produced the point estimate being reported.
    """
    a = validate_series(x, name="series")
    ci = _check_ci(ci)
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(f"min_coverage must lie in [0, 1], got {min_coverage!r}")
    point_exp = estimator(a)
    point_spec = spectrum_from_exponents(
        point_exp, method=spectrum_method, smoothing=smoothing, n_alpha=max(n_alpha, 3)
    )
    if alpha_grid is None:
        finite = np.isfinite(point_spec.alpha) & np.isfinite(point_spec.f_alpha)
        if int(finite.sum()) < 3:
            raise ValueError("the point-estimate spectrum has fewer than three finite points")
        lo = float(point_spec.alpha[finite].min())
        hi = float(point_spec.alpha[finite].max())
        grid = np.linspace(lo, hi, int(n_alpha))
    else:
        grid = np.ascontiguousarray(np.asarray(alpha_grid, dtype=np.float64).ravel())
    k = grid.size
    point = point_spec.evaluate(grid)

    def stat(y: FloatArray) -> FloatArray:
        spec = spectrum_from_exponents(
            estimator(y), method=spectrum_method, smoothing=smoothing, n_alpha=max(n_alpha, 3)
        )
        return spec.evaluate(grid)

    units, rebuild = _units_and_rebuild(a, resample)
    b, block_source = _resolve_block(units, block, method)
    reps, n_failed = _collect(
        stat,
        (
            rebuild(u)
            for u in _resample_iterator(
                units, block=b, n_resamples=int(n_resamples), seed=seed, method=method
            )
        ),
        k,
        int(n_resamples),
    )
    if n_failed == int(n_resamples):
        raise ValueError(f"every one of the {n_resamples} resamples failed the estimator")

    coverage = np.isfinite(reps).sum(axis=0) / float(n_resamples)
    jack = (
        _block_jackknife(units, rebuild, stat, block=b, k=k, max_blocks=jackknife_blocks)
        if interval == "bca"
        else None
    )
    iv = _intervals(
        reps,
        point,
        jack,
        ci=ci,
        interval=interval,
        min_valid=max(3, math.ceil(min_coverage * n_resamples)),
    )
    lower = np.where(coverage >= min_coverage, iv.lower, np.nan)
    upper = np.where(coverage >= min_coverage, iv.upper, np.nan)
    n_dropped = int(np.count_nonzero(coverage < min_coverage))
    notes = (
        *_summary_notes(iv, interval, n_failed),
        f"{n_dropped}/{k} grid point(s) below the {min_coverage:.0%} coverage floor -> NaN",
    )
    params: dict[str, Any] = {
        "n_resamples": int(n_resamples),
        "block": int(b),
        "block_source": block_source,
        "method": str(method),
        "resample": str(resample),
        "interval": str(interval),
        "ci": float(ci),
        "min_coverage": float(min_coverage),
        "spectrum_method": spectrum_method,
        "smoothing": smoothing,
        "n_alpha": int(k),
    }
    return SpectrumBootstrap(
        alpha=grid,
        point=point,
        replicates=reps,
        lower=np.asarray(lower, dtype=np.float64),
        upper=np.asarray(upper, dtype=np.float64),
        coverage=np.asarray(coverage, dtype=np.float64),
        min_coverage=float(min_coverage),
        ci_level=float(ci),
        interval=interval,
        per_point_method=iv.per_point_method,
        n_valid=iv.counts,
        n_resamples=int(n_resamples),
        n_failed=n_failed,
        block=int(b),
        method=method,
        resample_target=resample,
        notes=notes,
        provenance=_provenance_for(
            "inference.bootstrap_spectrum",
            a,
            params=params,
            seed=seed,
            notes=notes,
            parent=point_exp.provenance,
        ),
    )


def bootstrap_descriptors(
    x: Sequence[float] | NDArray[Any],
    estimator: Estimator,
    *,
    n_resamples: int = 200,
    block: int | None = None,
    seed: SeedLike,
    ci: float = 0.95,
    method: BlockMethod = "moving",
    resample: ResampleTarget = "series",
    interval: IntervalMethod = "bca",
    names: Sequence[str] = ("alpha_0", "width", "asymmetry"),
    spectrum_method: str = "fenchel",
    smoothing: float | None = None,
    n_alpha: int = 201,
    jackknife_blocks: int | None = 40,
    min_valid_fraction: float = 0.5,
) -> DescriptorBootstrap:
    r"""Intervals for the scalar spectrum summaries — the numbers papers quote.

    Available ``names`` are the fields of
    :class:`~mmar.core.types.SpectrumDescriptors`:
    ``alpha_0``, ``width``, ``asymmetry``, ``alpha_min``, ``alpha_max``,
    ``f_max``.

    Two warnings about these functionals specifically:

    * :math:`\Delta\alpha` is a *range*, i.e. a maximum minus a minimum, and
      extremes converge slowly and non-normally. Its bootstrap distribution is
      skewed and its lower endpoint is the informative one — a width whose
      interval includes a neighbourhood of zero is not evidence of
      multifractality.
    * The asymmetry :math:`B=(\alpha_0-\alpha_{\min})/(\alpha_{\max}-\alpha_0)`
      is a ratio whose denominator can approach zero, so replicates can be
      enormous or infinite. Non-finite replicates are recorded as NaN and
      excluded, which is visible in ``n_valid``.
    """
    unknown = [n for n in names if n not in DESCRIPTOR_NAMES]
    if unknown:
        raise ValueError(f"unknown descriptor(s) {unknown}; available: {list(DESCRIPTOR_NAMES)}")
    wanted = tuple(names)
    if not wanted:
        raise ValueError("at least one descriptor must be requested")

    a = validate_series(x, name="series")
    ci = _check_ci(ci)
    point_exp = estimator(a)

    def stat(y: FloatArray) -> FloatArray:
        spec = spectrum_from_exponents(
            estimator(y), method=spectrum_method, smoothing=smoothing, n_alpha=n_alpha
        )
        d = spec.descriptors()
        vals = np.array([float(getattr(d, n)) for n in wanted], dtype=np.float64)
        # An infinite asymmetry is a degenerate spectrum, not a datum.
        return np.asarray(np.where(np.isfinite(vals), vals, np.nan), dtype=np.float64)

    k = len(wanted)
    point = stat(a)
    units, rebuild = _units_and_rebuild(a, resample)
    b, block_source = _resolve_block(units, block, method)
    reps, n_failed = _collect(
        stat,
        (
            rebuild(u)
            for u in _resample_iterator(
                units, block=b, n_resamples=int(n_resamples), seed=seed, method=method
            )
        ),
        k,
        int(n_resamples),
    )
    if n_failed == int(n_resamples):
        raise ValueError(f"every one of the {n_resamples} resamples failed the estimator")

    jack = (
        _block_jackknife(units, rebuild, stat, block=b, k=k, max_blocks=jackknife_blocks)
        if interval == "bca"
        else None
    )
    iv = _intervals(
        reps,
        point,
        jack,
        ci=ci,
        interval=interval,
        min_valid=max(3, int(min_valid_fraction * n_resamples)),
    )
    notes = _summary_notes(iv, interval, n_failed)
    params: dict[str, Any] = {
        "n_resamples": int(n_resamples),
        "block": int(b),
        "block_source": block_source,
        "method": str(method),
        "resample": str(resample),
        "interval": str(interval),
        "ci": float(ci),
        "names": list(wanted),
        "spectrum_method": spectrum_method,
        "smoothing": smoothing,
    }
    return DescriptorBootstrap(
        names=wanted,
        point=point,
        replicates=reps,
        lower=iv.lower,
        upper=iv.upper,
        ci_level=float(ci),
        interval=interval,
        per_point_method=iv.per_point_method,
        z0=iv.z0,
        acceleration=iv.acceleration,
        n_valid=iv.counts,
        n_resamples=int(n_resamples),
        n_failed=n_failed,
        block=int(b),
        method=method,
        resample_target=resample,
        notes=notes,
        provenance=_provenance_for(
            "inference.bootstrap_descriptors",
            a,
            params=params,
            seed=seed,
            notes=notes,
            parent=point_exp.provenance,
        ),
    )


def residual_bootstrap_fit(
    fit: LogLogFit,
    *,
    n_resamples: int = 1000,
    seed: SeedLike,
    ci: float = 0.95,
    centre: bool = True,
) -> ResidualBootstrap:
    r"""Cheap interval for a single log--log slope by resampling its residuals.

    Freedman's (1981) residual bootstrap holds the design fixed and resamples the
    fitted residuals:

    .. math::
        \log y_j^{*} = \hat\beta\log x_j + \hat c + \varepsilon^{*}_j,
        \qquad \varepsilon^{*}\ \text{drawn i.i.d. from}\ \{\hat\varepsilon_j\},

    and refits :math:`\hat\beta^{*}` on each synthetic response. Because the
    design matrix is fixed the refit is a single closed-form contraction, so all
    :math:`B` slopes are computed in one vectorised expression with no loop.

    **This is not a substitute for** :func:`bootstrap_exponents`. It treats the
    residuals across scales as exchangeable, and they are not: the moments
    :math:`S(q,\delta)` at neighbouring :math:`\delta` are computed from heavily
    overlapping windows of the *same* series, so their errors are strongly
    positively correlated. The interval is therefore optimistic — typically by a
    factor of two or more — and it says nothing at all about the sampling
    variability of the underlying series, only about the scatter of the points
    already in hand. Use it for a quick sanity bar on one exponent; quote the
    block bootstrap.

    Parameters
    ----------
    fit
        A fit that retained ``log_x``, ``log_y`` and ``residuals``.
    centre
        Subtract the (weighted) mean residual before resampling. With an
        intercept in the model the OLS residuals already sum to zero, but a
        Theil--Sen or Huber fit's do not, and an uncentred draw would then shift
        the whole replicate distribution.

    Returns
    -------
    ResidualBootstrap
        Percentile interval, the replicate slopes, and provenance.
    """
    lx = np.asarray(fit.log_x, dtype=np.float64)
    ly = np.asarray(fit.log_y, dtype=np.float64)
    resid = np.asarray(fit.residuals, dtype=np.float64)
    if lx.size < 3 or lx.shape != ly.shape or resid.shape != lx.shape:
        raise ValueError(
            "the fit does not carry a usable (log_x, log_y, residuals) triple; "
            f"got shapes {lx.shape}, {ly.shape}, {resid.shape}. Refit with loglog_fit."
        )
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must lie strictly in (0, 1), got {ci!r}")
    n = lx.size
    b = int(n_resamples)
    if b < 2:
        raise ValueError(f"n_resamples must be at least 2, got {n_resamples!r}")

    w = np.ones(n) if fit.weights is None else np.asarray(fit.weights, dtype=np.float64)
    sw = w / w.sum()
    xbar = float(np.sum(sw * lx))
    sxx = float(np.sum(w * (lx - xbar) ** 2))
    if sxx <= 0.0:
        raise ValueError("all scales coincide in log space; the slope is not identified")

    e = resid - float(np.sum(sw * resid)) if centre else resid
    rng = spawn_rngs(seed, 1)[0]
    draws = e[rng.integers(0, n, size=(b, n))]
    fitted = fit.slope * lx + fit.intercept
    ly_star = fitted[None, :] + draws
    # Closed-form weighted slope for every synthetic response at once.
    ybar = ly_star @ sw
    slopes = ((ly_star - ybar[:, None]) * (w * (lx - xbar))[None, :]).sum(axis=1) / sxx

    p_lo, p_hi = (1.0 - ci) / 2.0, 1.0 - (1.0 - ci) / 2.0
    lo, hi = (float(v) for v in np.quantile(slopes, [p_lo, p_hi]))
    notes = (
        "residuals across scales are dependent (overlapping windows); this interval "
        "is optimistic and is not a substitute for bootstrap_exponents",
    )
    params: dict[str, Any] = {
        "n_resamples": b,
        "ci": float(ci),
        "centre": bool(centre),
        "n_points": n,
        "method": str(fit.method),
        "slope": float(fit.slope),
    }
    return ResidualBootstrap(
        slope=float(fit.slope),
        replicates=np.asarray(slopes, dtype=np.float64),
        lower=lo,
        upper=hi,
        stderr=float(np.std(slopes, ddof=1)),
        ci_level=float(ci),
        n_resamples=b,
        n_points=n,
        notes=notes,
        provenance=new_provenance(
            "inference.residual_bootstrap_fit",
            estimator_version=_VERSION,
            params=params,
            inputs=(InputRef.from_array("log_x", lx), InputRef.from_array("log_y", ly)),
            seed=seed_info(seed),
            notes=notes,
        ),
    )
