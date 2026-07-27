r"""Exact simulators for self-similar Gaussian processes, and the fat-tailed null.

These are the *monofractal nulls*. Every bias claim made about an estimator in
this library is measured against a process whose exponents are known in closed
form, so the simulators must be exact in distribution rather than merely
"asymptotically right": an approximate synthesis leaks a spurious curvature into
:math:`\hat\zeta(q)` that is indistinguishable from genuine multifractality.

Closed-form anchors
-------------------
Fractional Brownian motion :math:`B_H` with Hurst exponent :math:`H\in(0,1)` is
self-similar, :math:`B_H(at)\overset{d}{=}a^H B_H(t)`, with stationary Gaussian
increments. Hence for every :math:`q>0`

.. math::
    S(q,\delta) = \big\langle|B_H(t+\delta)-B_H(t)|^q\big\rangle
                = \sigma^q\,c_q\,\delta^{qH},
    \qquad c_q = \tfrac{2^{q/2}}{\sqrt\pi}\Gamma\!\big(\tfrac{q+1}{2}\big),

so in the conventions of :mod:`mmar.core.types`

.. math::
    \zeta(q) = qH, \qquad \tau(q) = qH - 1, \qquad
    f(\alpha) = \begin{cases}1 & \alpha = H\\ -\infty & \text{otherwise,}\end{cases}

a degenerate spectrum concentrated at the single exponent :math:`\alpha=H`. Any
estimated width :math:`\Delta\alpha>0` on an fBm sample is finite-size bias, and
quantifying it is the whole point of these generators.

The :math:`\alpha`-stable Lévy motion is the other, more dangerous null. Its
increments are iid, so it is *monofractal* (self-similar with
:math:`H=1/\alpha`), yet because :math:`\mathbb{E}|X|^q=\infty` for
:math:`q\ge\alpha` the empirical structure function saturates,

.. math::
    \hat\zeta(q) = \begin{cases} q/\alpha, & q<\alpha,\\ 1, & q>\alpha,\end{cases}

a concave broken line that every moment-based estimator reports as strong
multifractality (Mandelbrot 1997). Surrogate tests that do not include this null
cannot distinguish heavy tails from a cascade.

References
----------
Davies, R.B. and Harte, D.S. (1987). Tests for Hurst effect. *Biometrika* 74(1),
95--101.

Wood, A.T.A. and Chan, G. (1994). Simulation of stationary Gaussian processes in
:math:`[0,1]^d`. *Journal of Computational and Graphical Statistics* 3(4), 409--432.

Dietrich, C.R. and Newsam, G.N. (1997). Fast and exact simulation of stationary
Gaussian processes through circulant embedding of the covariance matrix.
*SIAM Journal on Scientific Computing* 18(4), 1088--1107.

Hosking, J.R.M. (1984). Modeling persistence in hydrological time series using
fractional differencing. *Water Resources Research* 20(12), 1898--1908.

Craigmile, P.F. (2003). Simulating a class of stationary Gaussian processes using
the Davies--Harte algorithm. *Journal of Time Series Analysis* 24(5), 505--511.

Chambers, J.M., Mallows, C.L. and Stuck, B.W. (1976). A method for simulating
stable random variables. *JASA* 71(354), 340--344.

Weron, R. (1996). On the Chambers--Mallows--Stuck method for simulating skewed
stable random variables. *Statistics & Probability Letters* 28(2), 165--171.

Peltier, R.F. and Lévy Véhel, J. (1995). Multifractional Brownian motion:
definition and preliminary results. *INRIA Research Report* 2645.

Coeurjolly, J.-F. (2000). Simulation and identification of the fractional
Brownian motion: a bibliographical and comparative study. *Journal of
Statistical Software* 5(7).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np

from mmar.core.provenance import Provenance, array_digest, new_provenance
from mmar.core.rng import SeedLike, make_rng, seed_info
from mmar.core.types import FloatArray

__all__ = [
    "FgnMethod",
    "brownian_bridge",
    "fbm",
    "fbm_with_provenance",
    "fgn",
    "fgn_acf",
    "fgn_hosking",
    "levy_stable_increments",
    "mbm",
]

type FgnMethod = Literal["davies-harte", "hosking"]

_VERSION = "1"

# Relative slack allowed on the circulant eigenvalues before the embedding is
# declared non-positive-definite. Round-off in a length-2^k real FFT is O(eps *
# log2(M) * max|lambda|); anything larger is a genuine embedding failure.
_EIGENVALUE_SLACK = 1e-9


def _check_hurst(H: float, name: str = "H") -> float:
    h = float(H)
    if not math.isfinite(h) or not (0.0 < h < 1.0):
        raise ValueError(f"{name} must lie strictly inside (0, 1); got {H!r}")
    return h


def _check_n(n: int) -> int:
    m = int(n)
    if m < 1:
        raise ValueError(f"n must be a positive integer; got {n!r}")
    return m


def _check_positive(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive; got {value!r}")
    return v


# --------------------------------------------------------------------------- #
# Autocovariance
# --------------------------------------------------------------------------- #


def fgn_acf(n: int, H: float, sigma: float = 1.0) -> FloatArray:
    r"""Exact autocovariance of unit-lag fractional Gaussian noise, lags :math:`0..n-1`.

    Differencing :math:`B_H` at unit spacing gives the stationary sequence
    :math:`G_k = B_H(k+1)-B_H(k)`, whose covariance follows from
    :math:`\operatorname{Cov}(B_H(s),B_H(t)) =
    \tfrac{\sigma^2}{2}(s^{2H}+t^{2H}-|t-s|^{2H})`:

    .. math::
        \gamma(k) = \frac{\sigma^2}{2}\Big(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H}\Big).

    In particular :math:`\gamma(0)=\sigma^2`, and
    :math:`\gamma(k)\sim \sigma^2 H(2H-1)k^{2H-2}` as :math:`k\to\infty`. The
    sign of :math:`H-\tfrac12` therefore decides everything: :math:`H>1/2` gives
    non-summable positive correlations (long memory), :math:`H<1/2` gives
    negative correlations with :math:`\sum_k\gamma(k)=0`, and :math:`H=1/2`
    collapses to white noise exactly.

    Parameters
    ----------
    n
        Number of lags returned, ``0`` through ``n-1``.
    H
        Hurst exponent in :math:`(0,1)`.
    sigma
        Standard deviation of a unit-lag increment.

    Returns
    -------
    FloatArray
        ``gamma[k]`` for ``k = 0, ..., n-1``.
    """
    n = _check_n(n)
    H = _check_hurst(H)
    s2 = _check_positive(sigma, "sigma") ** 2
    k = np.arange(n, dtype=np.float64)
    two_h = 2.0 * H
    return np.asarray(
        0.5 * s2 * ((k + 1.0) ** two_h - 2.0 * k**two_h + np.abs(k - 1.0) ** two_h),
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- #
# Davies--Harte circulant embedding
# --------------------------------------------------------------------------- #


def _embedding_size(n: int) -> int:
    """Half-size ``m`` of the circulant, the next power of two with ``m >= n``.

    Exactness requires only :math:`m \\ge n-1` (all lags realised in the sample
    must be below the circulant's fold point), but a power of two keeps the
    length-:math:`2m` FFT on its fast path.
    """
    return 1 << max(1, int(n - 1).bit_length())


def _circulant_eigenvalues(m: int, H: float, sigma: float) -> FloatArray:
    r"""Eigenvalues of the circulant embedding of the fGn covariance.

    The symmetric first row :math:`c=(\gamma_0,\dots,\gamma_m,\gamma_{m-1},\dots,
    \gamma_1)` of length :math:`M=2m` generates a circulant :math:`C=F^*\Lambda F`
    whose eigenvalues are the DFT of :math:`c`; they are real because :math:`c`
    is even. Positive semidefiniteness of :math:`C` is what makes the synthesis
    exact, and it is *not* automatic for a general covariance — Craigmile (2003)
    proves it for fGn with :math:`H\le 1/2`, and it is observed (Dietrich &
    Newsam 1997) for :math:`H>1/2` once :math:`m\ge n`.
    """
    gamma = fgn_acf(m + 1, H, sigma=sigma)
    row = np.concatenate([gamma, gamma[m - 1 : 0 : -1]])
    lam = np.asarray(np.fft.fft(row).real, dtype=np.float64)

    lam_max = float(lam.max())
    floor = -_EIGENVALUE_SLACK * max(lam_max, 1.0)
    worst = float(lam.min())
    if worst < floor:
        raise ValueError(
            f"circulant embedding is not positive semidefinite for H={H} at "
            f"embedding size 2m={2 * m} (min eigenvalue {worst:.3e}, max {lam_max:.3e}). "
            "Davies-Harte cannot be applied without silently distorting the covariance; "
            "use method='hosking' (exact, O(n^2)) instead."
        )
    return np.maximum(lam, 0.0)


def _spectral_normals(rng: np.random.Generator, m: int) -> tuple[FloatArray, FloatArray]:
    """Real and imaginary standard normals indexed by frequency ``0..m``.

    Kept separate from the synthesis so that :func:`mbm` can drive several
    :math:`H` values with one shared draw, which is what makes the interpolated
    path continuous in :math:`H`.
    """
    return (
        np.asarray(rng.standard_normal(m + 1), dtype=np.float64),
        np.asarray(rng.standard_normal(m + 1), dtype=np.float64),
    )


def _synthesise(lam: FloatArray, z_re: FloatArray, z_im: FloatArray, n: int) -> FloatArray:
    r"""Davies--Harte synthesis step: spectral weights, one FFT, truncate to ``n``.

    With :math:`M=2m`, set

    .. math::
        w_0 = \sqrt{\lambda_0/M}\,V_0, \qquad w_m = \sqrt{\lambda_m/M}\,V_m,
        \qquad w_k = \sqrt{\lambda_k/(2M)}\,(V_k^{(1)} + iV_k^{(2)}),

    for :math:`0<k<m`, and :math:`w_{M-k}=\overline{w_k}`. Then
    :math:`X_j=\sum_k w_k e^{-2\pi ijk/M}` is real, and since
    :math:`\mathbb{E}[w_k\overline{w_k}]=\lambda_k/M` with all other second
    moments vanishing,

    .. math::
        \mathbb{E}[X_jX_l]
        = \frac1M\sum_k \lambda_k e^{-2\pi i (j-l)k/M} = c_{|j-l|} = \gamma(|j-l|),

    the exact target covariance for :math:`|j-l|\le m`. The construction is
    therefore exact in distribution, not asymptotic.
    """
    m = lam.size // 2
    M = 2 * m
    w = np.zeros(M, dtype=np.complex128)
    w[0] = math.sqrt(lam[0] / M) * z_re[0]
    w[m] = math.sqrt(lam[m] / M) * z_re[m]
    k = np.arange(1, m)
    amp = np.sqrt(lam[k] / (2.0 * M))
    w[k] = amp * (z_re[k] + 1j * z_im[k])
    w[M - k] = np.conj(w[k])
    return np.ascontiguousarray(np.fft.fft(w).real[:n], dtype=np.float64)


def _fgn_davies_harte(n: int, H: float, sigma: float, rng: np.random.Generator) -> FloatArray:
    m = _embedding_size(n)
    lam = _circulant_eigenvalues(m, H, sigma)
    z_re, z_im = _spectral_normals(rng, m)
    return _synthesise(lam, z_re, z_im, n)


# --------------------------------------------------------------------------- #
# Hosking / Levinson--Durbin
# --------------------------------------------------------------------------- #


def fgn_hosking(n: int, H: float, *, seed: SeedLike, sigma: float = 1.0) -> FloatArray:
    r"""Exact fGn by Hosking's recursive (Durbin--Levinson) method. :math:`O(n^2)`.

    Draws :math:`G_k` sequentially from its exact conditional law
    :math:`G_k \mid G_{0:k-1} \sim \mathcal N(\mu_k, v_k)`, with the conditional
    mean and variance propagated by the Durbin--Levinson recursion on the
    partial autocorrelations :math:`\phi_{k,k}`:

    .. math::
        \phi_{k,k} = \frac{\gamma(k)-\sum_{j=1}^{k-1}\phi_{k-1,j}\gamma(k-j)}{v_{k-1}},
        \qquad v_k = v_{k-1}\big(1-\phi_{k,k}^2\big),

    :math:`\phi_{k,j}=\phi_{k-1,j}-\phi_{k,k}\phi_{k-1,k-j}` and
    :math:`\mu_k=\sum_{j=1}^{k}\phi_{k,j}G_{k-j}`.

    Exact like Davies--Harte but by a completely different route, which is why
    it is the cross-check in the test suite: an error in the circulant embedding
    (wrong normalisation, wrong fold point) would have to be reproduced
    identically here to escape detection.

    Notes
    -----
    The recursion is genuinely sequential — each sample conditions on all its
    predecessors — so the outer loop over samples cannot be vectorised. Cost is
    :math:`O(n^2)` time and :math:`O(n)` memory; use it up to a few thousand
    points. For :math:`H` close to :math:`1` the innovation variance
    :math:`v_k \sim k^{1-2H}` decays and the recursion loses precision; that is
    reported as an error rather than returned as a distorted path.
    """
    n = _check_n(n)
    H = _check_hurst(H)
    sigma = _check_positive(sigma, "sigma")

    gamma = fgn_acf(n, H, sigma=sigma)
    z = make_rng(seed).standard_normal(n)

    x = np.empty(n, dtype=np.float64)
    phi = np.zeros(n, dtype=np.float64)
    nxt = np.zeros(n, dtype=np.float64)

    v = float(gamma[0])
    x[0] = math.sqrt(v) * z[0]

    for k in range(1, n):
        # phi[:k-1] holds phi_{k-1, 1..k-1}; the reversed gamma slice pairs
        # phi_{k-1,j} with gamma(k-j).
        acc = float(np.dot(phi[: k - 1], gamma[k - 1 : 0 : -1])) if k > 1 else 0.0
        phi_kk = (float(gamma[k]) - acc) / v
        if k > 1:
            nxt[: k - 1] = phi[: k - 1] - phi_kk * phi[k - 2 :: -1]
        nxt[k - 1] = phi_kk
        phi, nxt = nxt, phi

        v *= 1.0 - phi_kk * phi_kk
        if v <= 0.0 or not math.isfinite(v):
            raise ValueError(
                f"Durbin-Levinson innovation variance collapsed to {v:.3e} at step {k} "
                f"(H={H}, n={n}); the recursion has lost positive definiteness to "
                "round-off. Use the Davies-Harte method instead."
            )
        mu = float(np.dot(phi[:k], x[k - 1 :: -1]))
        x[k] = mu + math.sqrt(v) * z[k]

    return x


# --------------------------------------------------------------------------- #
# Public generators
# --------------------------------------------------------------------------- #


def fgn(
    n: int,
    H: float,
    *,
    seed: SeedLike,
    sigma: float = 1.0,
    method: FgnMethod = "davies-harte",
) -> FloatArray:
    r"""Exact fractional Gaussian noise of length ``n``.

    The stationary increment sequence :math:`G_k=B_H(k+1)-B_H(k)`, Gaussian with
    mean zero and autocovariance :func:`fgn_acf`, so :math:`\operatorname{Var}
    G_k=\sigma^2` and :math:`\zeta(q)=qH` for the cumulated process.

    Parameters
    ----------
    n
        Length of the returned sequence.
    H
        Hurst exponent in :math:`(0,1)`. ``H=0.5`` returns iid
        :math:`\mathcal N(0,\sigma^2)` exactly.
    seed
        Any :data:`~mmar.core.rng.SeedLike`; the stream is threaded through
        :func:`~mmar.core.rng.make_rng`, never the global numpy state.
    sigma
        Standard deviation of one unit-lag increment.
    method
        ``"davies-harte"`` (default): circulant embedding, :math:`O(n\log n)`,
        exact. ``"hosking"``: Durbin--Levinson recursion, :math:`O(n^2)`, exact;
        the documented fallback when the circulant embedding fails to be
        positive semidefinite.

    Returns
    -------
    FloatArray
        Shape ``(n,)``, mean zero.

    Raises
    ------
    ValueError
        If ``H`` is outside :math:`(0,1)`, ``sigma`` is not positive, or the
        circulant embedding is not positive semidefinite (the eigenvalues are
        never silently zeroed: doing so changes the covariance and biases every
        exponent estimated from the sample).
    """
    n = _check_n(n)
    H = _check_hurst(H)
    sigma = _check_positive(sigma, "sigma")
    if method == "hosking":
        return fgn_hosking(n, H, seed=seed, sigma=sigma)
    if method != "davies-harte":
        raise ValueError(f"unknown method {method!r}; use 'davies-harte' or 'hosking'")
    return _fgn_davies_harte(n, H, sigma, make_rng(seed))


def fbm(
    n: int,
    H: float,
    *,
    seed: SeedLike,
    sigma: float = 1.0,
    t_max: float = 1.0,
    method: FgnMethod = "davies-harte",
) -> FloatArray:
    r"""Exact fractional Brownian motion sampled on a uniform grid of :math:`(0,t_{\max}]`.

    Returns :math:`B_H(t_i)` at :math:`t_i = i\,t_{\max}/n`, :math:`i=1,\dots,n`.
    The initial value :math:`B_H(0)=0` is *not* included, so the array has
    exactly ``n`` entries and ``out[-1]`` is :math:`B_H(t_{\max})`.

    Scaling. Unit-lag fGn has increment variance :math:`\sigma^2`; over a grid
    step :math:`\Delta t=t_{\max}/n` self-similarity demands increment variance
    :math:`\sigma^2\Delta t^{2H}`. Hence the path is
    :math:`B_H(t_i)=\Delta t^{H}\sum_{k\le i}G_k` and

    .. math::
        \operatorname{Var} B_H(t) = \sigma^2 t^{2H}, \qquad t\in(0,t_{\max}],

    exactly, for every :math:`n`. Getting this power of :math:`\Delta t` wrong
    is the classic way to produce a process whose measured Hurst exponent is
    right but whose amplitude is not, which then silently rescales any
    calibration performed against it.

    Exponents (see :mod:`mmar.core.types` for the conventions):
    :math:`\zeta(q)=qH`, :math:`\tau(q)=qH-1`, and the singularity spectrum is
    the single point :math:`f(H)=1`.

    Parameters
    ----------
    n, H, seed, sigma, method
        As in :func:`fgn`.
    t_max
        Right endpoint of the time interval.

    Returns
    -------
    FloatArray
        Shape ``(n,)``.
    """
    H = _check_hurst(H)
    t_max = _check_positive(t_max, "t_max")
    increments = fgn(n, H, seed=seed, sigma=sigma, method=method)
    dt = t_max / increments.size
    return np.asarray(np.cumsum(increments) * dt**H, dtype=np.float64)


def fbm_with_provenance(
    n: int,
    H: float,
    *,
    seed: SeedLike,
    sigma: float = 1.0,
    t_max: float = 1.0,
    method: FgnMethod = "davies-harte",
) -> tuple[FloatArray, Provenance]:
    """Audit path for :func:`fbm`: the path plus its content-addressed record.

    The provenance ``root`` is a pure function of ``(n, H, sigma, t_max, method,
    seed, environment)`` — the *recipe*, not the realisation — so two runs with
    the same seed share a root and the path can be regenerated from the manifest
    alone. The digest of the realised path is attached as a non-identifying
    diagnostic (``output_digest``), which is what a verifier compares against.

    Passing a live :class:`numpy.random.Generator` as ``seed`` leaves the seed
    field ``None``: such a stream is not recoverable, and the record says so
    rather than pretending otherwise.
    """
    path = fbm(n, H, seed=seed, sigma=sigma, t_max=t_max, method=method)
    prov = new_provenance(
        "models.fbm",
        estimator_version=_VERSION,
        params={
            "n": int(n),
            "H": float(H),
            "sigma": float(sigma),
            "t_max": float(t_max),
            "method": str(method),
        },
        seed=seed_info(seed),
        notes=("exact fBm; zeta(q) = q*H, tau(q) = q*H - 1, f(alpha) = {(H, 1)}",),
    ).with_diagnostics(output_digest=array_digest(path), n_samples=int(path.size))
    return path, prov


def brownian_bridge(
    n: int,
    *,
    seed: SeedLike,
    H: float = 0.5,
    sigma: float = 1.0,
    t_max: float = 1.0,
    method: FgnMethod = "davies-harte",
) -> FloatArray:
    r"""Fractional bridge on :math:`(0,t_{\max}]`, pinned to zero at both ends.

    Constructed by the pinning (conditioning) map

    .. math::
        \tilde B(t) = B_H(t) - \frac{t}{t_{\max}}\,B_H(t_{\max}),

    so ``out[-1]`` is exactly zero. For :math:`H=1/2` this is the standard
    Brownian bridge and equals :math:`B\mid\{B(t_{\max})=0\}` in law, with
    :math:`\operatorname{Cov}=\sigma^2(s\wedge t - st/t_{\max})`.

    For :math:`H\ne1/2` the linear pinning is **not** the conditional law of
    :math:`B_H` given :math:`B_H(t_{\max})=0` — that conditioning has a
    non-linear (kriging) drift. It is still a legitimate zero-mean Gaussian
    process with the same local regularity :math:`H`, which is all a
    detrending/edge-effect study needs, but do not call it "the" fractional
    bridge.
    """
    t_max = _check_positive(t_max, "t_max")
    path = fbm(n, H, seed=seed, sigma=sigma, t_max=t_max, method=method)
    t = np.arange(1, path.size + 1, dtype=np.float64) / path.size
    return np.asarray(path - t * path[-1], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Multifractional Brownian motion
# --------------------------------------------------------------------------- #


def mbm(
    n: int,
    h_func: Callable[[FloatArray], FloatArray],
    *,
    seed: SeedLike,
    sigma: float = 1.0,
    t_max: float = 1.0,
    n_h: int = 33,
) -> FloatArray:
    r"""Multifractional Brownian motion: fBm whose Hurst exponent varies in time.

    mBm (Peltier & Lévy Véhel 1995; Benassi, Jaffard & Roux 1997) replaces the
    constant :math:`H` of fBm by a Hölder-continuous function :math:`h(t)`. Its
    defining property, and the reason it is the ground truth for *pointwise*
    Hölder estimators, is that the local regularity is prescribed:
    almost surely

    .. math::
        \alpha_{X}(t) = h(t) \quad\text{for every } t,

    where :math:`\alpha_X(t)` is the pointwise Hölder exponent, provided
    :math:`h` is itself :math:`\beta`-Hölder with
    :math:`\beta>\sup_t h(t)`. Globally mBm has no single scaling exponent, so
    the *multifractal* formalism does not describe it; only pointwise estimators
    (oscillation, wavelet-based local exponents) recover :math:`h`.

    Method (Wood--Chan "krigeage" interpolation)
    --------------------------------------------
    Exact synthesis of mBm is expensive because the process is non-stationary in
    a way circulant embedding cannot exploit. This implementation uses the
    standard and honestly *approximate* construction of Chan & Wood (1998),
    also described by Coeurjolly (2000):

    1. Fix a grid :math:`H_1<\dots<H_K` spanning :math:`[\min h,\max h]`.
    2. Synthesise :math:`B_{H_j}` for every :math:`j` by Davies--Harte, **driving
       every one of them with the same spectral Gaussian draw**. Each
       :math:`B_{H_j}` then has exactly the right fBm law, and the family is a
       path-continuous function of :math:`H`.
    3. Return :math:`X(t_i)` by linear interpolation in :math:`H` between the two
       bracketing grid paths at :math:`H=h(t_i)`.

    Limitations, stated plainly:

    * The one-dimensional marginals are only approximately correct. A convex
      combination of two *correlated* Gaussians is Gaussian with variance
      differing from the target :math:`\sigma^2 t^{2h(t)}` at order
      :math:`O\big((\Delta H\,\log t)^2\big)`, :math:`\Delta H=(\max h-\min
      h)/(K-1)`. Raise ``n_h`` to shrink it.
    * The joint law is *not* the law of mBm: sharing one Gaussian draw across
      the :math:`H`-grid induces a coupling the true process does not have. What
      is faithfully reproduced is the local regularity, which is what pointwise
      estimators are validated against.
    * When :math:`h` is constant the construction is exact and reduces bit-for-bit
      to :func:`fbm` with the same ``seed``.

    Parameters
    ----------
    n
        Number of samples, on the grid :math:`t_i=i\,t_{\max}/n`, :math:`i=1..n`.
    h_func
        Vectorised map :math:`u\mapsto h(u)` from normalised time
        :math:`u=t/t_{\max}\in(0,1]` to :math:`(0,1)`. It is called once, on the
        whole grid, and must return an array of the same shape.
    seed, sigma, t_max
        As in :func:`fbm`.
    n_h
        Size :math:`K` of the :math:`H` grid. Ignored when :math:`h` is constant.

    Returns
    -------
    FloatArray
        Shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``h_func`` is not vectorised (wrong output shape), or returns values
        outside :math:`(0,1)`, or ``n_h < 2``.
    """
    n = _check_n(n)
    sigma = _check_positive(sigma, "sigma")
    t_max = _check_positive(t_max, "t_max")
    if int(n_h) < 2:
        raise ValueError(f"n_h must be at least 2; got {n_h!r}")

    u = np.arange(1, n + 1, dtype=np.float64) / n
    h = np.asarray(h_func(u), dtype=np.float64)
    if h.shape != u.shape:
        raise ValueError(
            f"h_func must be vectorised: called on shape {u.shape} it returned {h.shape}. "
            "Wrap a scalar function with numpy.vectorize if necessary."
        )
    if not np.all(np.isfinite(h)):
        raise ValueError("h_func returned non-finite Hurst exponents")
    if float(h.min()) <= 0.0 or float(h.max()) >= 1.0:
        raise ValueError(
            f"h_func must map into the open interval (0, 1); got range "
            f"[{float(h.min()):.4g}, {float(h.max()):.4g}]"
        )

    rng = make_rng(seed)
    m = _embedding_size(n)
    z_re, z_im = _spectral_normals(rng, m)
    dt = t_max / n

    h_lo, h_hi = float(h.min()), float(h.max())
    if h_hi - h_lo <= 1e-12:
        lam = _circulant_eigenvalues(m, h_lo, sigma)
        return np.asarray(np.cumsum(_synthesise(lam, z_re, z_im, n)) * dt**h_lo, dtype=np.float64)

    grid = np.linspace(h_lo, h_hi, int(n_h))
    paths = np.empty((grid.size, n), dtype=np.float64)
    for j, hj in enumerate(grid):
        lam = _circulant_eigenvalues(m, float(hj), sigma)
        paths[j] = np.cumsum(_synthesise(lam, z_re, z_im, n)) * dt ** float(hj)

    lo = np.clip(np.searchsorted(grid, h, side="right") - 1, 0, grid.size - 2)
    w = (h - grid[lo]) / (grid[lo + 1] - grid[lo])
    cols = np.arange(n)
    return np.asarray((1.0 - w) * paths[lo, cols] + w * paths[lo + 1, cols], dtype=np.float64)


# --------------------------------------------------------------------------- #
# alpha-stable increments
# --------------------------------------------------------------------------- #


def levy_stable_increments(
    n: int,
    alpha: float,
    *,
    seed: SeedLike,
    beta: float = 0.0,
    scale: float = 1.0,
) -> FloatArray:
    r"""iid :math:`S_\alpha(\text{scale},\beta,0)` draws by Chambers--Mallows--Stuck.

    The cumulative sum of these increments is an :math:`\alpha`-stable Lévy
    motion: self-similar with :math:`H=1/\alpha`, with iid increments, and
    therefore **monofractal**. It is nevertheless the indispensable null for
    surrogate testing, because moment-based estimators report it as strongly
    multifractal: for :math:`q\ge\alpha` the moment
    :math:`\mathbb{E}|X|^q` is infinite, the sample structure function is
    dominated by the single largest increment, and the measured exponent
    saturates at :math:`\hat\zeta(q)=\min(q/\alpha,\,1)`. A concave broken line
    in :math:`\hat\zeta` is the fingerprint of heavy tails, not of a cascade.

    Algorithm (Weron's 1996 correction of Chambers--Mallows--Stuck, in the
    :math:`S^0`-free "1-parametrisation"): with
    :math:`U\sim\mathrm{Unif}(-\pi/2,\pi/2)` and :math:`W\sim\mathrm{Exp}(1)`
    independent, for :math:`\alpha\ne1`

    .. math::
        B = \frac{\arctan(\beta\tan\tfrac{\pi\alpha}{2})}{\alpha},\qquad
        S = \Big(1+\beta^2\tan^2\tfrac{\pi\alpha}{2}\Big)^{1/(2\alpha)},

    .. math::
        X = S\,\frac{\sin\big(\alpha(U+B)\big)}{(\cos U)^{1/\alpha}}
            \left(\frac{\cos\big(U-\alpha(U+B)\big)}{W}\right)^{(1-\alpha)/\alpha},

    and for :math:`\alpha=1`

    .. math::
        X = \frac{2}{\pi}\left[\Big(\frac\pi2+\beta U\Big)\tan U
            - \beta\log\frac{\tfrac\pi2 W\cos U}{\tfrac\pi2+\beta U}\right].

    The :math:`\pi/2` inside that logarithm is not cosmetic: dropping it
    translates every draw by :math:`\tfrac2\pi\beta\log\tfrac\pi2\approx
    0.2875\,\beta`, which is invisible at :math:`\beta=0` and is a location
    error of a quarter of the scale at :math:`\beta=1`.

    Convention. The law returned is the *1-parametrisation* (Nolan's
    :math:`S(\alpha,\beta,c,0;1)`, equivalently ``scipy.stats.levy_stable`` with
    ``parameterization="S1"``), whose characteristic function at
    :math:`\alpha=1` is

    .. math::
        \mathbb{E}e^{itX}
        = \exp\!\Big(-c|t|\big(1 + i\beta\tfrac2\pi\operatorname{sgn}(t)
          \log(c|t|)\big)\Big),

    so ``scale`` acts by plain multiplication for every :math:`\alpha`. Under
    the Samorodnitsky--Taqqu convention (:math:`\log|t|` in place of
    :math:`\log(c|t|)`) the :math:`\alpha=1` scaling would instead require the
    extra shift :math:`\tfrac2\pi\beta c\log c`; the two agree at :math:`c=1`
    and nowhere else.

    At :math:`\alpha=2` the formula degenerates correctly to
    :math:`\mathcal N(0,2)`, i.e. standard deviation :math:`\sqrt2\,`\ ``scale``
    — the stable "scale" is not the standard deviation, and conflating the two
    is a common calibration error.

    Parameters
    ----------
    n
        Number of iid draws.
    alpha
        Stability index in :math:`(0,2]`.
    beta
        Skewness in :math:`[-1,1]`.
    scale
        Stable scale parameter :math:`c>0` (:math:`X\to cX`).

    Returns
    -------
    FloatArray
        Shape ``(n,)``, iid.
    """
    n = _check_n(n)
    a = float(alpha)
    if not math.isfinite(a) or not (0.0 < a <= 2.0):
        raise ValueError(f"alpha must lie in (0, 2]; got {alpha!r}")
    b = float(beta)
    if not math.isfinite(b) or abs(b) > 1.0:
        raise ValueError(f"beta must lie in [-1, 1]; got {beta!r}")
    c = _check_positive(scale, "scale")

    rng = make_rng(seed)
    u = math.pi * (rng.random(n) - 0.5)
    w = rng.exponential(size=n)

    if abs(a - 1.0) < 1e-12:
        # The alpha=1 branch is a genuinely different formula, not a limit that
        # can be evaluated by substitution: tan(pi/2) blows up in the general one.
        arg = 0.5 * math.pi + b * u
        with np.errstate(divide="ignore", invalid="ignore"):
            x = (2.0 / math.pi) * (
                arg * np.tan(u) - b * np.log(0.5 * math.pi * w * np.cos(u) / arg)
            )
    else:
        t = math.tan(0.5 * math.pi * a)
        bshift = math.atan(b * t) / a
        s = (1.0 + (b * t) ** 2) ** (0.5 / a)
        # (1-a)/a is negative for a>1: cos(...)/W can be large, so guard the power.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            x = (
                s
                * np.sin(a * (u + bshift))
                / np.cos(u) ** (1.0 / a)
                * (np.cos(u - a * (u + bshift)) / w) ** ((1.0 - a) / a)
            )

    out = c * np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(
            f"Chambers-Mallows-Stuck produced {int((~np.isfinite(out)).sum())} non-finite "
            f"draws at alpha={a}, beta={b}; this indicates catastrophic cancellation, "
            "not a heavy tail."
        )
    return out
