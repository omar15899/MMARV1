r"""Surrogate data: telling genuine multifractality from its two impostors.

A wide singularity spectrum :math:`f(\alpha)` is not evidence of a cascade. Three
mechanisms produce one, and no estimator can distinguish them from a single
sample:

1. **Genuine multifractality** --- a nonlinear temporal dependence structure, of
   the kind a multiplicative cascade generates: the *ordering* of the sample
   carries the scaling.
2. **A fat-tailed marginal with no temporal structure at all.** For iid draws
   from a law with :math:`\mathbb E|X|^q=\infty` beyond some :math:`q_c`, the
   empirical structure function saturates and
   :math:`\hat\zeta(q)\approx\min(q/\alpha,1)`: a concave broken line, which the
   Legendre transform turns into a broad :math:`f(\alpha)`
   (Mandelbrot 1997; see :func:`mmar.models.fbm.levy_stable_increments`).
3. **Non-stationarity** --- trends, level shifts, a slowly varying variance.

Surrogates resolve the ambiguity by *destroying one mechanism at a time* and
re-estimating. Each generator below is a map
:math:`x \mapsto x^{(s)}` that preserves some features of :math:`x` exactly and
randomises the rest; the statistic of interest (usually the spectrum width
:math:`\Delta\alpha`) is recomputed on an ensemble of surrogates and the
observed value is placed in that null distribution
(:func:`surrogate_test`).

What each surrogate preserves
-----------------------------
======================  =========================  ==========================
Generator               Preserved exactly          Destroyed
======================  =========================  ==========================
:func:`shuffle_surrogate`          marginal law    all temporal dependence
:func:`phase_randomised_surrogate` power spectrum  nonlinearity; marginal
                                   (hence the
                                   linear ACF)     is Gaussianised
:func:`aaft_surrogate`             marginal law    nonlinearity (ACF only
                                                   approximately preserved)
:func:`iaaft_surrogate`            marginal law    nonlinearity (ACF preserved
                                                   to a reported tolerance)
:func:`block_shuffle_surrogate`    marginal law;   dependence beyond one block
                                   dependence
                                   inside a block
======================  =========================  ==========================

The shuffle is the single most informative of the five. If the estimated width
:math:`\Delta\alpha` **survives shuffling**, the multifractality is a property of
the marginal distribution --- mechanism (2) --- and no amount of cascade
modelling is warranted. If it collapses, the width came from the ordering, which
is mechanism (1) or (3). Sweeping the block length in
:func:`block_shuffle_surrogate` then localises the *scale* at which that
ordering matters: the width recovers once the block exceeds the correlation
length of the generating mechanism, and the recovery curve is a far more
informative diagnostic than any single p-value. Zhou (2009) uses exactly this
decomposition to show that the empirical multifractality of financial returns is
mostly, but not entirely, a fat-tail effect.

Failure modes, stated plainly
-----------------------------
* A surrogate test is a test of **its own null**, not of "multifractality".
  Rejecting the shuffle null says the ordering matters; it does not say the
  ordering is a cascade rather than a trend. Combine with
  :func:`phase_randomised_surrogate` (which keeps the linear ACF) to separate
  linear memory from nonlinear structure.
* **AAFT is biased.** The rank remap to a Gaussian and back is not
  spectrum-preserving: the resulting surrogates have a *flatter* spectrum than
  the data, i.e. their autocorrelation is biased toward white noise, and the
  bias grows with the distance of the marginal from Gaussian. A test using AAFT
  surrogates therefore over-rejects when the data are both correlated and
  strongly non-Gaussian. :func:`iaaft_surrogate` exists to fix precisely this.
* An **IAAFT surrogate that did not converge is not an IAAFT surrogate**. The
  achieved relative spectrum discrepancy is reported in the provenance
  diagnostics of :func:`surrogate_ensemble_with_provenance` and must be checked,
  not assumed.
* All of these are *constrained realisations* in the sense of Theiler &
  Prichard (1996): the constraints are imposed exactly rather than in
  distribution, so the resulting test is conservative for pivotal statistics and
  only approximately calibrated otherwise.

References
----------
Theiler, J., Eubank, S., Longtin, A., Galdrikian, B. and Farmer, J.D. (1992).
Testing for nonlinearity in time series: the method of surrogate data.
*Physica D* **58**, 77--94.

Theiler, J. and Prichard, D. (1996). Constrained-realization Monte-Carlo method
for hypothesis testing. *Physica D* **94**, 221--235.

Schreiber, T. and Schmitz, A. (1996). Improved surrogate data for nonlinearity
tests. *Phys. Rev. Lett.* **77**, 635--638.

Schreiber, T. and Schmitz, A. (2000). Surrogate time series.
*Physica D* **142**, 346--382.

Kantelhardt, J.W., Zschiegner, S.A., Koscielny-Bunde, E., Havlin, S., Bunde, A.
and Stanley, H.E. (2002). Multifractal detrended fluctuation analysis of
nonstationary time series. *Physica A* **316**, 87--114.

Zhou, W.-X. (2009). The components of empirical multifractality in financial
returns. *EPL* **88**, 28004.

Mandelbrot, B. (1997). *Fractals and Scaling in Finance*. Springer.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from mmar.core.legendre import spectrum_from_exponents
from mmar.core.provenance import InputRef, Provenance, array_digest, new_provenance
from mmar.core.rng import SeedLike, make_rng, seed_info, spawn_rngs
from mmar.core.types import FloatArray, ScalingExponents
from mmar.core.validate import validate_series

__all__ = [
    "SurrogateKind",
    "SurrogateResult",
    "SurrogateTest",
    "aaft_surrogate",
    "amplitude_spectrum",
    "block_shuffle_surrogate",
    "iaaft_surrogate",
    "phase_randomised_surrogate",
    "shuffle_surrogate",
    "spectrum_error",
    "spectrum_width",
    "surrogate_ensemble",
    "surrogate_ensemble_with_provenance",
    "surrogate_test",
]

MODULE_VERSION: Final = "1"

_MIN_LENGTH: Final = 16
"""Below this the Fourier grid has too few free phases for a surrogate to be random."""

_IMAG_TOL: Final = 1e-12
"""Relative bound on the imaginary residual of an inverse FFT that must be real."""

_CARRIED_DIAGNOSTICS: Final = frozenset(
    {
        "spectrum_error_max",
        "spectrum_error_mean",
        "iterations_max",
        "n_not_converged",
        "imag_residual_max",
        "values_digest",
    }
)
"""Ensemble diagnostics that must survive into a derived test record.

``n_not_converged`` in particular: it is the flag that says whether the null the
p-value was computed against is the one the caller asked for.
"""

ComplexArray = NDArray[np.complex128]


class SurrogateKind(StrEnum):
    """Which null hypothesis a surrogate ensemble embodies."""

    SHUFFLE = "shuffle"
    """iid draws from the empirical marginal: no temporal structure whatsoever."""

    PHASE = "phase"
    """A stationary linear Gaussian process with the observed power spectrum."""

    AAFT = "aaft"
    """A monotone nonlinear transform of a linear Gaussian process."""

    IAAFT = "iaaft"
    """As ``AAFT``, with the spectrum constraint enforced iteratively."""

    BLOCK = "block"
    """Dependence retained inside blocks of a fixed length, destroyed across them."""


# --------------------------------------------------------------------------- #
# Spectral helpers
# --------------------------------------------------------------------------- #


def amplitude_spectrum(x: FloatArray) -> FloatArray:
    r"""Amplitude spectrum :math:`|\hat x_k|`, :math:`k=0,\dots,\lfloor n/2\rfloor`.

    Its square is the (unnormalised) periodogram, and by Wiener--Khinchin it is
    an exact reparametrisation of the sample autocovariance: preserving
    :math:`|\hat x_k|` is *identical* to preserving the circular ACF, which is
    why the Fourier surrogates constrain this object and nothing else.
    """
    return np.asarray(np.abs(np.fft.rfft(x)), dtype=np.float64)


def _relative_spectrum_error(target: FloatArray, achieved: FloatArray) -> float:
    r""":math:`\|A^{\text{ach}}-A^{\text{tgt}}\|_2 / \|A^{\text{tgt}}\|_2`."""
    denom = float(np.linalg.norm(target))
    if denom <= 0.0:  # pragma: no cover - a non-constant series has positive norm
        return float("nan")
    return float(np.linalg.norm(achieved - target) / denom)


def spectrum_error(reference: FloatArray, surrogate: FloatArray) -> float:
    r"""Relative :math:`L^2` discrepancy between two amplitude spectra.

    .. math::
        \varepsilon = \frac{\big\|\,|\hat y| - |\hat x|\,\big\|_2}{\big\||\hat x|\big\|_2}.

    Zero for a phase-randomised surrogate (up to round-off), of order
    :math:`10^{-3}`--:math:`10^{-2}` for a converged IAAFT surrogate, and larger
    --- typically by an order of magnitude --- for AAFT, whose rank remap has no
    mechanism to restore the spectrum it distorts. This is the number that
    decides whether an IAAFT surrogate is fit to be used.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(surrogate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"series must have the same length; got {a.shape} and {b.shape}")
    return _relative_spectrum_error(amplitude_spectrum(a), amplitude_spectrum(b))


def _ranks(a: FloatArray) -> NDArray[np.intp]:
    """Rank of every entry, ties broken by position (stable sort)."""
    order = np.argsort(a, kind="stable")
    out = np.empty(a.size, dtype=np.intp)
    out[order] = np.arange(a.size, dtype=np.intp)
    return out


def _unit_phase(spec: ComplexArray) -> ComplexArray:
    """:math:`e^{i\\arg z}`, with the (measure-zero) zero coefficients sent to 1."""
    mag = np.abs(spec)
    safe = np.where(mag > 0.0, mag, 1.0)
    return np.asarray(np.where(mag > 0.0, spec / safe, 1.0 + 0.0j), dtype=np.complex128)


def _check_series(x: FloatArray | Sequence[float]) -> FloatArray:
    return validate_series(x, name="x", min_length=_MIN_LENGTH, allow_constant=False)


# --------------------------------------------------------------------------- #
# One draw, plus the diagnostics that make it auditable
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Draw:
    """One surrogate and the numbers needed to certify it."""

    values: FloatArray
    spectrum_error: float = float("nan")
    """Relative amplitude-spectrum discrepancy; NaN when the kind does not constrain it."""
    iterations: int = 0
    converged: bool = True
    imag_residual: float = float("nan")
    """Largest imaginary part of an inverse FFT that the construction forces to be real."""


# --------------------------------------------------------------------------- #
# The five generators
# --------------------------------------------------------------------------- #


def shuffle_surrogate(x: FloatArray | Sequence[float], *, seed: SeedLike) -> FloatArray:
    r"""Random permutation: the iid null with the empirical marginal.

    Returns :math:`x^{(s)}_i = x_{\pi(i)}` for :math:`\pi` uniform on
    :math:`S_n`. The multiset :math:`\{x_i\}` --- hence every moment, quantile
    and tail index of the marginal --- is preserved **exactly**, and every
    temporal structure, linear and nonlinear, is destroyed: the surrogate is an
    exchangeable sequence.

    This is the decisive diagnostic of the module. Under the shuffle null the
    process is iid, so its true spectrum is the degenerate monofractal one; any
    :math:`\Delta\alpha>0` measured on a shuffled sample is the estimator's
    response to the *marginal law alone*. Comparing it with the width measured
    on the data therefore decomposes the observed multifractality into a
    distributional component and a correlation component (Kantelhardt et al.
    2002; Zhou 2009). A width that survives shuffling is a fat tail, not a
    cascade.

    Parameters
    ----------
    x
        Real series, at least :data:`_MIN_LENGTH` finite non-constant samples.
    seed
        Any :data:`~mmar.core.rng.SeedLike`. The global numpy state is never
        touched, so the surrogate is bit-for-bit reproducible from ``seed``.

    Returns
    -------
    FloatArray
        A permutation of ``x``, same length, same dtype.
    """
    a = _check_series(x)
    return np.ascontiguousarray(make_rng(seed).permutation(a))


def _phase_randomise(x: FloatArray, rng: np.random.Generator) -> tuple[FloatArray, float]:
    r"""Core of the FT surrogate: randomise phases, enforce Hermitian symmetry.

    Returns the real series and the imaginary residual of the inverse transform,
    which is the numerical certificate that the symmetry was imposed correctly.
    """
    n = x.size
    spec = np.asarray(np.fft.fft(x), dtype=np.complex128)
    out = spec.copy()

    # Free positive frequencies are k = 1..m; k = 0 (the mean) and, for even n,
    # k = n/2 (Nyquist) are their own conjugates and admit no free phase.
    m = (n - 1) // 2
    if m > 0:
        phase = np.exp(1j * rng.uniform(0.0, 2.0 * math.pi, size=m))
        out[1 : m + 1] = spec[1 : m + 1] * phase
        # x_hat[n-k] = conj(x_hat[k]) is what makes the inverse transform real.
        out[n - m :] = np.conj(out[1 : m + 1][::-1])
    if n % 2 == 0:
        # The only unimodular multiplier that keeps a real coefficient real.
        out[n // 2] = spec[n // 2] * (1.0 if rng.random() < 0.5 else -1.0)

    y = np.asarray(np.fft.ifft(out), dtype=np.complex128)
    residual = float(np.max(np.abs(y.imag)))
    scale = max(1.0, float(np.max(np.abs(y.real))))
    if residual > _IMAG_TOL * scale:  # pragma: no cover - a symmetry bug, not a data case
        raise ValueError(
            f"phase-randomised series has imaginary part {residual:.3e} (scale {scale:.3e}); "
            "the Hermitian symmetry of the randomised spectrum is broken, so the "
            "construction — not the data — is at fault"
        )
    return np.ascontiguousarray(y.real, dtype=np.float64), residual


def phase_randomised_surrogate(x: FloatArray | Sequence[float], *, seed: SeedLike) -> FloatArray:
    r"""Fourier-transform (FT) surrogate: Theiler et al. (1992).

    Writes :math:`\hat x_k = |\hat x_k| e^{i\varphi_k}` and replaces the phases
    of the free positive frequencies by iid :math:`\mathrm{Unif}(0,2\pi)` draws,
    mirroring them onto the negative frequencies by
    :math:`\hat y_{n-k}=\overline{\hat y_k}`:

    .. math::
        \hat y_k = |\hat x_k|\,e^{i\psi_k},\quad
        \psi_{n-k}=-\psi_k,\quad \psi_0 = 0,\quad
        \psi_{n/2}\in\{0,\pi\}\ (n \text{ even}),

    then inverts. Two bins are *not* free: :math:`k=0` (the sample mean) and,
    for even :math:`n`, the Nyquist bin :math:`k=n/2`. Both coefficients are
    real, and any phase other than a sign destroys the conjugate symmetry and
    yields a complex "series". Getting this wrong is the standard FT-surrogate
    bug; here the symmetry is imposed explicitly and the residual imaginary part
    of the inverse transform is checked against
    :math:`10^{-12}\max(1,\|y\|_\infty)` before the real part is returned.

    Because :math:`|\hat y_k| = |\hat x_k|` for every :math:`k`, the periodogram
    --- equivalently, by Wiener--Khinchin, the entire circular autocovariance
    function --- is preserved **exactly**. What is destroyed is every
    higher-order structure: the null is a stationary *linear Gaussian* process
    with the observed second-order statistics. In particular the surrogate's
    marginal is Gaussianised, by the central limit theorem applied to the sum of
    :math:`n/2` independently phased components. That Gaussianisation is a
    feature when testing for nonlinearity and a defect when the statistic is
    sensitive to the marginal --- which the multifractal width emphatically is.
    Use :func:`iaaft_surrogate` when the marginal must be held fixed.

    Returns
    -------
    FloatArray
        Real series of the same length and the same power spectrum as ``x``,
        with the same mean (the DC bin is untouched).
    """
    a = _check_series(x)
    return _phase_randomise(a, make_rng(seed))[0]


def aaft_surrogate(x: FloatArray | Sequence[float], *, seed: SeedLike) -> FloatArray:
    r"""Amplitude-adjusted FT surrogate: Theiler et al. (1992), §5.

    Three steps. Let :math:`r_i` be the rank of :math:`x_i` and :math:`g_{(1)}
    \le\dots\le g_{(n)}` the order statistics of an iid standard Gaussian sample:

    1. **Gaussianise by rank.** :math:`u_i = g_{(r_i+1)}`. This is the monotone
       map that carries the empirical marginal onto a Gaussian one while keeping
       the ordering of the sample.
    2. **Phase-randomise** :math:`u` (:func:`phase_randomised_surrogate`).
    3. **Invert the remap by rank.** :math:`y_i = x_{(\rho_i+1)}` where
       :math:`\rho` are the ranks of the phase-randomised series.

    The composition preserves the marginal law **exactly** (step 3 returns the
    original multiset) and the autocorrelation only **approximately**: this is
    the whole content of the method's known bias.

    Bias
    ----
    The rank remaps are nonlinear, and a nonlinear map does not commute with the
    spectral constraint. Schreiber & Schmitz (1996, 2000) show that the
    composition systematically *flattens* the spectrum toward white noise: the
    surrogates are less correlated than the data, by an amount that grows with
    the departure of the marginal from Gaussian. A nonlinearity test built on
    AAFT surrogates therefore rejects too often for data that are both
    correlated and heavy-tailed --- exactly the regime of financial returns, and
    exactly the regime in which multifractality is claimed. Quantify the bias
    with :func:`spectrum_error` before trusting the test, and prefer
    :func:`iaaft_surrogate`.

    Returns
    -------
    FloatArray
        A permutation of ``x`` (hence an exact match of its marginal).
    """
    a = _check_series(x)
    rng = make_rng(seed)
    x_sorted = np.sort(a)
    gaussian_order_stats = np.sort(rng.standard_normal(a.size))
    gaussianised = gaussian_order_stats[_ranks(a)]
    randomised = _phase_randomise(gaussianised, rng)[0]
    return np.ascontiguousarray(x_sorted[_ranks(randomised)])


def _iaaft(
    x: FloatArray, rng: np.random.Generator, *, max_iter: int, tol: float
) -> tuple[FloatArray, float, int, bool]:
    """IAAFT alternating projection. Returns ``(y, spectrum_error, iterations, converged)``."""
    n = x.size
    target = amplitude_spectrum(x)
    x_sorted = np.sort(x)

    y = np.asarray(rng.permutation(x), dtype=np.float64)
    err = _relative_spectrum_error(target, amplitude_spectrum(y))
    iterations = 0
    converged = False
    for iterations in range(1, max_iter + 1):  # noqa: B007 - the count is the diagnostic
        # (a) project onto the spectrum constraint: impose |y_hat| = |x_hat|,
        #     keeping the current phases. Exact, but breaks the marginal.
        filtered = np.fft.irfft(target * _unit_phase(np.fft.rfft(y)), n=n)
        # (b) project onto the distribution constraint: reimpose the empirical
        #     marginal by rank. Exact, but perturbs the spectrum again.
        y_next = np.asarray(x_sorted[_ranks(np.asarray(filtered, dtype=np.float64))])
        err_next = _relative_spectrum_error(target, amplitude_spectrum(y_next))
        improvement = err - err_next
        y, err = y_next, err_next
        if improvement <= tol:
            converged = True
            break
    return np.ascontiguousarray(y), err, iterations, converged


def iaaft_surrogate(
    x: FloatArray | Sequence[float],
    *,
    seed: SeedLike,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> FloatArray:
    r"""Iterative AAFT surrogate: Schreiber & Schmitz (1996).

    Alternating projection between the two constraint sets that AAFT can only
    satisfy one at a time. Starting from a random shuffle :math:`y^{(0)}` of the
    data, iterate

    .. math::
        \tilde y^{(k)} = \mathcal F^{-1}\!\Big[\,|\hat x|\;
            e^{i\arg \widehat{y^{(k)}}}\,\Big],
        \qquad
        y^{(k+1)} = \operatorname{rank-remap}\big(\tilde y^{(k)};\,x\big),

    i.e. project onto :math:`\{y : |\hat y| = |\hat x|\}` and then onto
    :math:`\{y : \operatorname{sort}(y)=\operatorname{sort}(x)\}`. The second set
    is finite (the :math:`n!` permutations of :math:`x`) so the scheme is not a
    convex alternating projection and has no convergence guarantee; in practice
    the discrepancy

    .. math::
        \varepsilon_k = \frac{\big\|\,|\widehat{y^{(k)}}|-|\hat x|\,\big\|_2}
                             {\big\||\hat x|\big\|_2}

    decreases monotonically to a nonzero plateau, at which the ranking stops
    changing and the iteration is at a fixed point. The marginal is exact at
    every step; the spectrum is exact only in the limit, so
    :math:`\varepsilon_\infty` is the honest measure of how good the surrogate
    is. It is typically an order of magnitude below the corresponding AAFT
    value.

    Parameters
    ----------
    max_iter
        Cap on the number of alternating projections.
    tol
        Stop when one full sweep improves :math:`\varepsilon` by no more than
        this. A stalled iteration (exact fixed point) improves by zero and stops
        immediately.

    Returns
    -------
    FloatArray
        A permutation of ``x``. **The achieved** :math:`\varepsilon` **is not
        returned here**: use :func:`surrogate_ensemble_with_provenance`, whose
        provenance diagnostics carry ``spectrum_error_max``,
        ``iterations_max`` and ``n_not_converged``. An IAAFT surrogate whose
        discrepancy was never checked is an AAFT surrogate with extra steps.
    """
    a = _check_series(x)
    if int(max_iter) < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter!r}")
    if not math.isfinite(tol) or tol < 0.0:
        raise ValueError(f"tol must be finite and non-negative, got {tol!r}")
    return _iaaft(a, make_rng(seed), max_iter=int(max_iter), tol=float(tol))[0]


def block_shuffle_surrogate(
    x: FloatArray | Sequence[float], *, block: int, seed: SeedLike
) -> FloatArray:
    r"""Permute contiguous blocks: dependence survives up to the block length.

    Cut the series into :math:`\lceil n/L\rceil` consecutive blocks of length
    :math:`L` (the last one short when :math:`L\nmid n`), permute the blocks
    uniformly at random, and concatenate. The marginal is preserved exactly, as
    for :func:`shuffle_surrogate`, and so is every dependence *internal* to a
    block: a fraction :math:`(L-1)/L` of the adjacent pairs of the original
    series survive intact, so the lag-:math:`\ell` autocorrelation is retained
    with weight :math:`\max(0, 1-\ell/L)` and destroyed beyond it.

    Sweeping :math:`L` is the point. Estimating :math:`\Delta\alpha` on
    block-shuffled ensembles as :math:`L` grows from 1 to :math:`n` interpolates
    between the iid null (:math:`L=1`, identical in law to
    :func:`shuffle_surrogate`) and the data (:math:`L=n`, the identity map). The
    :math:`L` at which the width recovers is the **correlation length of the
    mechanism that generates the multifractality** --- a cascade with
    :math:`k_{\max}` levels on :math:`n=b^{k_{\max}}` points has structure at
    every scale, so its recovery curve rises slowly across the whole range,
    whereas a short-memory artefact recovers abruptly. Most packages omit this
    diagnostic; it is more informative than the p-value it generalises.

    Parameters
    ----------
    block
        Block length :math:`L\in[1,n]`. ``block=1`` is a full shuffle;
        ``block=n`` returns ``x`` unchanged (one block cannot be permuted), which
        is the correct degenerate limit and not an error.

    Returns
    -------
    FloatArray
        A permutation of ``x``. The ragged final block is permuted along with
        the rest, so it can land in the interior; the alternative (pinning it to
        the end) would leave a deterministic segment in every surrogate.
    """
    a = _check_series(x)
    n = a.size
    L = int(block)
    if L < 1 or n < L:
        raise ValueError(f"block must lie in [1, n] = [1, {n}], got {block!r}")

    n_blocks = -(-n // L)  # ceil
    lengths = np.full(n_blocks, L, dtype=np.intp)
    lengths[-1] = n - L * (n_blocks - 1)
    starts = np.arange(n_blocks, dtype=np.intp) * L

    perm = make_rng(seed).permutation(n_blocks)
    out_lengths = lengths[perm]
    out_starts = np.concatenate(([0], np.cumsum(out_lengths)[:-1])).astype(np.intp)
    # Position within the destination block, mapped back to the source block.
    offsets = np.arange(n, dtype=np.intp) - np.repeat(out_starts, out_lengths)
    index = np.repeat(starts[perm], out_lengths) + offsets
    return np.ascontiguousarray(a[index])


# --------------------------------------------------------------------------- #
# Dispatch and ensembles
# --------------------------------------------------------------------------- #


def _draw(
    x: FloatArray,
    kind: SurrogateKind,
    rng: np.random.Generator,
    *,
    block: int | None,
    max_iter: int,
    tol: float,
) -> _Draw:
    """One surrogate of the requested kind, with its per-draw diagnostics."""
    if kind is SurrogateKind.SHUFFLE:
        return _Draw(values=np.ascontiguousarray(rng.permutation(x)))
    if kind is SurrogateKind.PHASE:
        values, residual = _phase_randomise(x, rng)
        return _Draw(values=values, imag_residual=residual)
    if kind is SurrogateKind.AAFT:
        values = aaft_surrogate(x, seed=rng)
        return _Draw(values=values, spectrum_error=spectrum_error(x, values))
    if kind is SurrogateKind.IAAFT:
        values, err, iterations, converged = _iaaft(x, rng, max_iter=max_iter, tol=tol)
        return _Draw(values=values, spectrum_error=err, iterations=iterations, converged=converged)
    if kind is SurrogateKind.BLOCK:
        if block is None:
            raise ValueError("kind='block' requires an explicit block length")
        return _Draw(values=block_shuffle_surrogate(x, block=block, seed=rng))
    raise ValueError(f"unhandled surrogate kind {kind!r}")  # pragma: no cover


def surrogate_ensemble(
    x: FloatArray | Sequence[float],
    kind: SurrogateKind | str,
    n: int,
    *,
    seed: SeedLike,
    block: int | None = None,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> FloatArray:
    r"""``n`` independent surrogates of the requested kind, shape ``(n, len(x))``.

    The replicas are driven by :func:`~mmar.core.rng.spawn_rngs`, i.e. by
    children of one :class:`numpy.random.SeedSequence`, **not** by incrementing
    an integer seed: consecutive integer seeds initialise PCG64 states that are
    close in the generator's 128-bit space and produce detectably correlated
    streams, which in a surrogate ensemble means a null distribution narrower
    than it should be and a p-value that is too small. The whole ensemble
    remains reproducible from the single parent ``seed``.

    Parameters
    ----------
    kind
        A :class:`SurrogateKind` or its string value.
    n
        Number of replicas. For a rank-based p-value at resolution
        :math:`1/(n+1)`, use ``n = 19`` (0.05), ``99`` (0.01) or ``999``.
    block, max_iter, tol
        Forwarded to :func:`block_shuffle_surrogate` and
        :func:`iaaft_surrogate` respectively; ignored by the other kinds.

    Returns
    -------
    FloatArray
        Shape ``(n, len(x))``. Row ``i`` comes from the ``i``-th spawned stream.
    """
    a = _check_series(x)
    k = SurrogateKind(kind)
    m = int(n)
    if m < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    rngs = spawn_rngs(seed, m)
    draws = [_draw(a, k, r, block=block, max_iter=max_iter, tol=tol) for r in rngs]
    return np.ascontiguousarray(np.stack([d.values for d in draws]))


@dataclass(frozen=True, slots=True)
class SurrogateResult:
    """A surrogate ensemble with the audit record that certifies it."""

    values: FloatArray
    """Shape ``(n_members, n_samples)``."""
    kind: SurrogateKind
    provenance: Provenance

    @property
    def n_members(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[1])

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": str(self.kind),
            "n_members": self.n_members,
            "n_samples": self.n_samples,
        }
        out.update(dict(self.provenance.diagnostics))
        return out


def _aggregate(draws: Sequence[_Draw]) -> dict[str, Any]:
    """Ensemble-level diagnostics: the worst case is the one that matters."""
    out: dict[str, Any] = {}
    errors = np.array([d.spectrum_error for d in draws], dtype=np.float64)
    if np.isfinite(errors).any():
        out["spectrum_error_max"] = float(np.nanmax(errors))
        out["spectrum_error_mean"] = float(np.nanmean(errors))
    iterations = np.array([d.iterations for d in draws], dtype=np.int64)
    if int(iterations.max(initial=0)) > 0:
        out["iterations_max"] = int(iterations.max())
        out["n_not_converged"] = int(sum(not d.converged for d in draws))
    residuals = np.array([d.imag_residual for d in draws], dtype=np.float64)
    if np.isfinite(residuals).any():
        out["imag_residual_max"] = float(np.nanmax(residuals))
    return out


def surrogate_ensemble_with_provenance(
    x: FloatArray | Sequence[float],
    kind: SurrogateKind | str,
    n: int,
    *,
    seed: SeedLike,
    block: int | None = None,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> SurrogateResult:
    """Audited form of :func:`surrogate_ensemble`.

    The provenance ``root`` is a pure function of the *recipe* --- input digest,
    kind, ensemble size, block length, IAAFT settings, seed, environment --- so
    two runs with the same seed share a root and the ensemble is regenerable
    from the manifest alone. The realised ensemble's digest and the convergence
    diagnostics are attached as non-identifying diagnostics, which is what makes
    the reproducibility claim checkable rather than asserted.

    For ``kind="iaaft"`` the diagnostics carry ``spectrum_error_max``,
    ``spectrum_error_mean``, ``iterations_max`` and ``n_not_converged``. Read
    them. A surrogate test run on unconverged IAAFT surrogates is a test against
    an unspecified null.
    """
    a = _check_series(x)
    k = SurrogateKind(kind)
    m = int(n)
    if m < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    rngs = spawn_rngs(seed, m)
    draws = [_draw(a, k, r, block=block, max_iter=max_iter, tol=tol) for r in rngs]
    values = np.ascontiguousarray(np.stack([d.values for d in draws]))

    prov = new_provenance(
        f"inference.surrogates.{k!s}",
        estimator_version=MODULE_VERSION,
        params={
            "kind": str(k),
            "n": m,
            "block": None if block is None else int(block),
            "max_iter": int(max_iter),
            "tol": float(tol),
        },
        inputs=(InputRef.from_array("x", a),),
        seed=seed_info(seed),
        notes=(
            "surrogate ensemble; the null hypothesis is the invariance stated in "
            f"SurrogateKind.{k.name}",
        ),
    ).with_diagnostics(
        values_digest=array_digest(values),
        n_members=m,
        n_samples=int(a.size),
        **_aggregate(draws),
    )
    return SurrogateResult(values=values, kind=k, provenance=prov)


# --------------------------------------------------------------------------- #
# The test
# --------------------------------------------------------------------------- #


def spectrum_width(exponents: ScalingExponents) -> float:
    r"""Default test statistic: the spectrum width :math:`\Delta\alpha`.

    Computed through the Legendre--Fenchel transform, so it is the width of a
    genuinely concave spectrum and not of a folded parametric curve. Exactly
    zero for a monofractal, and the standard scalar summary of multifractality.
    """
    return spectrum_from_exponents(exponents).descriptors().width


@dataclass(frozen=True, slots=True)
class SurrogateTest:
    r"""The outcome of one surrogate test: the observed statistic in its null."""

    statistic: float
    """Value on the data."""
    null: FloatArray
    """Values on the surrogate ensemble, shape ``(n_surrogates,)``."""
    p_value: float
    kind: SurrogateKind
    statistic_name: str
    provenance: Provenance

    @property
    def n_surrogates(self) -> int:
        return int(self.null.size)

    @property
    def z_score(self) -> float:
        r""":math:`(s_{\text{obs}}-\bar s)/\operatorname{sd}(s)`, for reporting only.

        The p-value is the rank statistic, not this: the null distribution of a
        spectrum width is skewed and bounded below by zero, so a Gaussian tail
        probability derived from ``z_score`` would be wrong. Quote ``z_score``
        as an effect size, never as significance.
        """
        sd = float(np.std(self.null, ddof=1))
        if sd <= 0.0:
            return float("nan")
        return float((self.statistic - float(np.mean(self.null))) / sd)

    def summary(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "statistic_name": self.statistic_name,
            "statistic": self.statistic,
            "null_mean": float(np.mean(self.null)),
            "null_sd": float(np.std(self.null, ddof=1)) if self.null.size > 1 else float("nan"),
            "p_value": self.p_value,
            "n_surrogates": self.n_surrogates,
            "z_score": self.z_score,
        }


def surrogate_test(
    x: FloatArray | Sequence[float],
    estimator: Callable[[FloatArray], ScalingExponents],
    *,
    seed: SeedLike,
    kind: SurrogateKind | str = SurrogateKind.SHUFFLE,
    n_surrogates: int = 99,
    statistic: Callable[[ScalingExponents], float] = spectrum_width,
    block: int | None = None,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> SurrogateTest:
    r"""Place an estimated statistic in the null distribution of a surrogate ensemble.

    Computes :math:`s_{\text{obs}} = T(\hat\zeta[x])` and
    :math:`s_i = T(\hat\zeta[x^{(i)}])` for :math:`n` surrogates, and reports the
    one-sided rank p-value

    .. math::
        p = \frac{1 + \#\{\,i : s_i \ge s_{\text{obs}}\,\}}{n+1},

    which is exactly calibrated (uniform on :math:`\{1,2,\dots,n+1\}/(n+1)`)
    whenever the data are exchangeable with the surrogates under the null, and
    conservative otherwise. The :math:`+1` in both places counts the observation
    itself and is not a continuity correction: omitting it makes :math:`p=0`
    attainable, which is never true.

    Parameters
    ----------
    estimator
        Any map from a series to :class:`~mmar.core.types.ScalingExponents` ---
        MFDFA, wavelet leaders, structure functions. Passed as a callable rather
        than selected by name so that the test is agnostic to which estimator
        produced the exponents, and so that the *same* estimator with the *same*
        settings is applied to the data and to every surrogate. Applying
        different settings is the commonest way to fake a significant result.
    statistic
        Scalar functional of the exponents. Defaults to
        :func:`spectrum_width`; :math:`-c_2` or :math:`h(-5)-h(5)` are equally
        valid choices, and the choice must be made *before* seeing the data.
    kind
        Which null. ``"shuffle"`` (default) tests "the multifractality is a
        property of the marginal alone"; ``"phase"`` tests "the data are a
        linear Gaussian process"; ``"iaaft"`` tests "the data are a monotone
        nonlinear transform of one"; ``"block"`` tests "the dependence beyond
        ``block`` samples is irrelevant".

    Returns
    -------
    SurrogateTest

    Raises
    ------
    ValueError
        If the estimator returns a non-finite statistic on the data or on any
        surrogate. A p-value computed over the subset on which an estimator
        happened to succeed is conditional on that success and is not the
        quantity anyone intends to report.

    Notes
    -----
    The estimator is evaluated ``n_surrogates + 1`` times; with a cubic-cost
    estimator this dominates the runtime, and the ensemble is the natural place
    to parallelise (the surrogates are independent by construction).
    """
    a = _check_series(x)
    k = SurrogateKind(kind)
    m = int(n_surrogates)
    if m < 1:
        raise ValueError(f"n_surrogates must be a positive integer, got {n_surrogates!r}")

    observed = float(statistic(estimator(a)))
    if not math.isfinite(observed):
        raise ValueError(f"the statistic is {observed} on the data itself; nothing to test")

    ensemble = surrogate_ensemble_with_provenance(
        a, k, m, seed=seed, block=block, max_iter=max_iter, tol=tol
    )
    null = np.array([float(statistic(estimator(row))) for row in ensemble.values], dtype=np.float64)
    bad = int((~np.isfinite(null)).sum())
    if bad:
        raise ValueError(
            f"{bad}/{m} surrogate statistics are non-finite: the estimator failed on the "
            "null ensemble, so any p-value would be conditional on its success. Fix the "
            "estimator's settings (usually the q range or the scaling range) and re-run."
        )

    p_value = float((1 + int(np.sum(null >= observed))) / (m + 1))
    # The whole ensemble audit record travels with the test, not just the
    # spectrum errors: a p-value computed against surrogates that never
    # converged (``n_not_converged``) is a p-value against an unspecified null,
    # and dropping the flag here would leave the manifest unable to say so.
    carried = {
        key: value
        for key, value in ensemble.provenance.diagnostics.items()
        if key in _CARRIED_DIAGNOSTICS
    }
    prov = ensemble.provenance.derive(
        "inference.surrogates.surrogate_test",
        estimator_version=MODULE_VERSION,
        params={
            "kind": str(k),
            "n_surrogates": m,
            "statistic": getattr(statistic, "__name__", repr(statistic)),
            "estimator": getattr(estimator, "__name__", repr(estimator)),
            "block": None if block is None else int(block),
        },
        notes=("one-sided rank p-value: p = (1 + #{s_i >= s_obs}) / (n + 1)",),
    ).with_diagnostics(
        statistic=observed,
        p_value=p_value,
        null_mean=float(np.mean(null)),
        **carried,
    )
    return SurrogateTest(
        statistic=observed,
        null=null,
        p_value=p_value,
        kind=k,
        statistic_name=getattr(statistic, "__name__", "statistic"),
        provenance=prov,
    )
