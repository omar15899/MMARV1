r"""Coverage holes found by mutation: numbers the suite lets you change silently.

Each test here corresponds to one surgical edit to ``src/`` that the whole
existing suite (``tests/unit``, ``tests/theory``, ``tests/integration``, and the
other adversarial files) accepted without a single failure. A mutation nothing
detects is a number nothing pins, and every one of them is a number that reaches
a reported result: a confidence interval, a spectrum width, a Hölder exponent, a
provenance root.

The one entry that is *not* a hole but a live defect is
:func:`test_curvature_is_the_second_derivative_of_the_parabolic_fit`: at the
commit under attack ``SpectrumDescriptors.curvature`` returned the bare
quadratic coefficient while documenting itself as the second derivative, i.e.
exactly half the advertised quantity.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mmar.analysis import _to_level, analyze
from mmar.core.legendre import spectrum_from_exponents
from mmar.core.provenance import array_digest
from mmar.core.regression import loglog_fit
from mmar.core.types import (
    FitMethod,
    FloatArray,
    HolderSpectrum,
    ScalingExponents,
)
from mmar.estimators.mfdfa import mfdfa
from mmar.estimators.wavelet_leaders import LeaderPyramid, _log_cumulant_curves
from mmar.models.cascade import deterministic_cascade, trading_time
from mmar.models.fbm import fgn
from mmar.models.theory import (
    cumulants_from_tau_lognormal,
    lognormal_conservation_sigma2,
    lognormal_falpha,
)

# --------------------------------------------------------------------------- #
# core/regression.py
# --------------------------------------------------------------------------- #

_SCALES = np.array([8.0, 16.0, 32.0, 64.0, 128.0, 256.0])
_MOMENTS = np.array([1.0, 1.9, 4.1, 7.8, 16.5, 31.0])


def test_ols_standard_error_uses_n_minus_two_degrees_of_freedom() -> None:
    r"""Attack: change ``dof = n - 2`` to ``n - 1`` in :func:`loglog_fit`.

    Nothing in the suite fails. A straight-line fit estimates two parameters, so
    the residual variance carries :math:`n-2` degrees of freedom; using
    :math:`n-1` shrinks every reported ``slope_stderr`` by
    :math:`\sqrt{(n-2)/(n-1)}` and narrows every :math:`\zeta` interval by the
    same factor, which is a *coverage* claim, not a cosmetic one. On the six
    scales below the reported error falls from 0.012613 to 0.011281, an 11%
    understatement that no assertion anywhere notices.
    """
    fit = loglog_fit(_SCALES, _MOMENTS)
    lx, ly = np.log(_SCALES), np.log(_MOMENTS)
    n = lx.size
    sxx = float(np.sum((lx - lx.mean()) ** 2))
    slope = float(np.sum((lx - lx.mean()) * (ly - ly.mean())) / sxx)
    intercept = float(ly.mean() - slope * lx.mean())
    rss = float(np.sum((ly - (slope * lx + intercept)) ** 2))

    s2_correct = rss / (n - 2)
    s2_wrong = rss / (n - 1)
    assert fit.slope_stderr == pytest.approx(math.sqrt(s2_correct / sxx), rel=1e-12)
    assert fit.intercept_stderr == pytest.approx(
        math.sqrt(s2_correct * (1.0 / n + lx.mean() ** 2 / sxx)), rel=1e-12
    )
    # The two dof choices are far enough apart that the test cannot pass by luck.
    assert abs(fit.slope_stderr - math.sqrt(s2_wrong / sxx)) > 1e-3 * fit.slope_stderr


def _wls(x: FloatArray, y: FloatArray, w: FloatArray) -> tuple[float, float]:
    """Closed-form weighted least squares with *inverse-variance* weights ``w``."""
    sw = w / w.sum()
    xbar = float(np.sum(sw * x))
    ybar = float(np.sum(sw * y))
    slope = float(np.sum(w * (x - xbar) * (y - ybar)) / np.sum(w * (x - xbar) ** 2))
    return slope, float(ybar - slope * xbar)


def _huber_reference(x: FloatArray, y: FloatArray, w: FloatArray) -> float:
    """IRLS Huber slope, seeded with the WLS fit, written without ``numpy.polyfit``."""
    slope, intercept = _wls(x, y, w)
    for _ in range(30):
        r = y - (slope * x + intercept)
        scale = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        if scale <= 0.0 or not math.isfinite(scale):
            break
        u = np.abs(r) / scale
        hw = np.where(u <= 1.345, 1.0, 1.345 / np.maximum(u, 1e-12))
        new_slope, new_intercept = _wls(x, y, w * hw)
        converged = abs(new_slope - slope) < 1e-12 and abs(new_intercept - intercept) < 1e-12
        slope, intercept = new_slope, new_intercept
        if converged:
            break
    return slope


def test_huber_irls_is_seeded_with_the_weighted_least_squares_fit() -> None:
    r"""Attack: seed ``_huber_irls`` with ``polyfit(w=w)`` instead of ``polyfit(w=sqrt(w))``.

    :func:`numpy.polyfit` multiplies the *residuals* by its ``w``, so
    inverse-variance weights :math:`w_j=n_j` must be handed over as
    :math:`\sqrt{w_j}`; passing them raw squares them. Only the IRLS *seed* is
    affected, which is why nothing catches it: on clean data the iteration walks
    back to the same fixed point. On an outlier-contaminated log--log plot it
    does not — the seed decides which points the Huber weights end up
    discounting. Here the two seeds land on slopes 0.72027 and 0.70830, a 1.7%
    difference in a reported scaling exponent.
    """
    scales = np.geomspace(8.0, 1024.0, 8)
    moments = np.array(
        [
            4.110084909074863,
            6.808538383763522,
            10.31400420811944,
            18.633627597239236,
            30.812688540210477,
            44.92304477908769,
            1303.1775696553348,  # one corrupted coarse moment
            136.08799935444796,
        ]
    )
    counts = np.geomspace(2048.0, 8.0, 8)

    fit = loglog_fit(scales, moments, method=FitMethod.HUBER, weights=counts)
    reference = _huber_reference(np.log(scales), np.log(moments), counts)
    assert fit.slope == pytest.approx(reference, abs=1e-12)
    # The squared-weight seed is a different answer, not numerical noise.
    squared_seed = _huber_reference(np.log(scales), np.log(moments), counts**2)
    assert abs(fit.slope - squared_seed) > 1e-3


# --------------------------------------------------------------------------- #
# core/types.py -- the descriptors that become the headline numbers
# --------------------------------------------------------------------------- #


def _asymmetric_spectrum() -> HolderSpectrum:
    """A tent whose zero-crossings fall strictly between grid points, both sides."""
    alpha = np.array([0.0, 0.4, 1.0, 1.6, 3.0])
    f = np.array([-1.0, 0.2, 1.0, 0.5, -2.0])
    return HolderSpectrum(alpha=alpha, f_alpha=f)


def test_support_crossing_interpolates_from_the_point_above_the_level() -> None:
    r"""Attack: flip the interpolation weight in ``_crossing`` to ``(level - f_lo)/(f_hi - f_lo)``.

    The crossing of :math:`f(\alpha)=0` is what ``width``, ``alpha_min`` and
    ``alpha_max`` *are*, and the flipped weight is its exact complement
    :math:`1-w`, so the crossing lands mirrored inside the same bracket: the
    spectrum below reports :math:`\alpha_{\min}=0.3333` instead of
    :math:`0.0667` and :math:`\alpha_{\max}=1.88` instead of :math:`2.72`, a
    width of 1.547 against 2.653. Both look entirely plausible; nothing in the
    suite pins either.
    """
    d = _asymmetric_spectrum().descriptors()
    # Left: bracket (0.0, -1.0) -> (0.4, 0.2); the crossing sits 1/6 of the way back.
    assert d.alpha_min == pytest.approx(0.4 - (0.2 / 1.2) * 0.4, rel=1e-12)
    # Right: bracket (1.6, 0.5) -> (3.0, -2.0); the crossing sits 1/5 of the way on.
    assert d.alpha_max == pytest.approx(1.6 + (0.5 / 2.5) * 1.4, rel=1e-12)
    assert d.width == pytest.approx(d.alpha_max - d.alpha_min, rel=1e-12)
    assert not d.support_truncated


def test_asymmetry_is_left_over_right_as_documented() -> None:
    r"""Attack: return ``right / left`` for :attr:`SpectrumDescriptors.asymmetry`.

    The documented statistic is
    :math:`B=(\alpha_0-\alpha_{\min})/(\alpha_{\max}-\alpha_0)`, and the sense
    matters: ``B > 1`` is read as a left-truncated, large-fluctuation-dominated
    spectrum. Inverting it turns 0.758 into 1.320 and reverses that reading,
    with no test anywhere to object.
    """
    d = _asymmetric_spectrum().descriptors()
    left = d.alpha_0 - d.alpha_min
    right = d.alpha_max - d.alpha_0
    assert d.asymmetry == pytest.approx(left / right, rel=1e-12)
    assert d.asymmetry < 1.0  # this spectrum is right-heavy; the flip would say otherwise


def test_curvature_is_the_second_derivative_of_the_parabolic_fit() -> None:
    r"""Attack: drop the factor of two from ``curvature``. It was *already* dropped.

    ``SpectrumDescriptors.curvature`` documents itself as "second derivative of
    the parabolic fit at :math:`\alpha_0`", and :mod:`mmar.viz.diagnostics`
    prints its own quadratic curvature as ``2.0 * coeffs[0]``. The descriptor
    returned the bare ``numpy.polyfit`` coefficient, which is :math:`f''/2`.

    Measured on the exact lognormal spectrum
    :math:`f(\alpha)=1-(\alpha-\lambda)^2/(2\kappa)` with :math:`\lambda=1.08`,
    :math:`b=2` (so :math:`\kappa=0.16` and :math:`f''=-1/\kappa=-6.25`):
    ``descriptors().curvature`` came back as :math:`-3.125`, exactly half, and
    therefore disagreed by a factor of two with the library's own
    :math:`c_2`, for which :math:`f''=1/c_2`.
    """
    lam, b = 1.08, 2.0
    sigma2 = lognormal_conservation_sigma2(lam, b)
    _c1, c2 = cumulants_from_tau_lognormal(lam, sigma2, b)
    alpha = np.linspace(lam - 0.3, lam + 0.3, 201)
    spectrum = HolderSpectrum(alpha=alpha, f_alpha=lognormal_falpha(alpha, lam, sigma2, b))

    curvature = spectrum.descriptors().curvature
    assert curvature == pytest.approx(1.0 / c2, rel=1e-9)
    assert curvature == pytest.approx(-1.0 / (sigma2 * math.log(b)), rel=1e-9)


# --------------------------------------------------------------------------- #
# core/legendre.py
# --------------------------------------------------------------------------- #


def test_parametric_alpha_stderr_propagates_two_zeta_errors() -> None:
    r"""Attack: drop the :math:`\sqrt 2` from the delta-method ``alpha_stderr``.

    :math:`\alpha=\zeta'` is a difference of two fitted exponents, so its
    propagated error carries :math:`\sqrt{2}\,\hat\sigma_\zeta/\Delta q`; without
    the factor every error bar on the :math:`\alpha` axis of a parametric
    spectrum is 29% too small. The quantity is exported on
    :class:`HolderSpectrum` and plotted, and nothing pins it.
    """
    q = np.linspace(-3.0, 3.0, 13)
    se = np.full(q.size, 0.02)
    exponents = ScalingExponents(q=q, zeta=0.6 * q - 0.05 * q**2, zeta_stderr=se)
    spectrum = spectrum_from_exponents(exponents, method="parametric")

    assert spectrum.alpha_stderr is not None
    dq = float(q[1] - q[0])
    np.testing.assert_allclose(
        spectrum.alpha_stderr, math.sqrt(2.0) * se / dq, rtol=1e-12, atol=0.0
    )
    assert spectrum.f_stderr is not None
    assert spectrum.q is not None
    np.testing.assert_allclose(
        spectrum.f_stderr, np.abs(spectrum.q) * spectrum.alpha_stderr + se, rtol=1e-12, atol=0.0
    )


# --------------------------------------------------------------------------- #
# core/provenance.py
# --------------------------------------------------------------------------- #


def test_array_digest_separates_arrays_that_differ_only_in_shape() -> None:
    """Attack: drop ``h.update(repr(a.shape))`` from :func:`array_digest`.

    The C-contiguous byte image of ``arange(6)``, ``arange(6).reshape(2, 3)``
    and ``arange(6).reshape(3, 2)`` is identical, so without the shape in the
    hash all three collide. A collision in an *input* digest is a collision in
    the provenance root: two different computations become indistinguishable in
    the audit record, which is the one property the whole module exists to
    provide.
    """
    flat = np.arange(6, dtype=np.float64)
    digests = {
        array_digest(flat),
        array_digest(flat.reshape(2, 3)),
        array_digest(flat.reshape(3, 2)),
    }
    assert len(digests) == 3
    # ... while a mere memory-layout change must NOT move the digest.
    assert array_digest(flat.reshape(2, 3)) == array_digest(
        np.asfortranarray(flat.reshape(2, 3)).T.T
    )


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #


def test_leader_log_cumulants_use_the_unbiased_k_statistic() -> None:
    r"""Attack: divide the second log-cumulant by :math:`n` instead of :math:`n-1`.

    The module states the estimator as :math:`k_2 = \tfrac{n}{n-1}m_2` precisely
    because the plug-in variance is biased by :math:`-\sigma^2/n_j` and
    :math:`n_j` *halves every octave*, so the bias tilts the :math:`C_2(j)` line
    and inflates :math:`|\hat c_2|` — it manufactures multifractality out of
    small samples, which is the one artefact :math:`c_2` exists to rule out. On
    the four-leader octave below the plug-in value is 0.3125 against the
    unbiased 0.4167, a 25% deficit that grows toward the coarse octaves.
    """
    values = (
        np.exp(np.array([0.1, 0.5, -0.3, 1.2, -1.1, 0.7, 0.05, -0.9])),
        np.exp(np.array([0.2, -0.4, 0.9, -1.3, 0.3, 1.0])),
        np.exp(np.array([0.3, -0.2, 0.8, -0.7])),
    )
    pyramid = LeaderPyramid(
        octaves=np.array([3, 4, 5], dtype=np.int64),
        values=values,
        n_border_dropped=np.zeros(3, dtype=np.int64),
        n_zero_dropped=np.zeros(3, dtype=np.int64),
        wavelet="db3",
        n_vanishing_moments=3,
        boundary_margin=4,
        leaders=True,
    )
    curves = _log_cumulant_curves(pyramid, 2)
    for i, v in enumerate(values):
        log_v = np.log(v)
        assert curves[0, i] == pytest.approx(float(np.mean(log_v)), rel=1e-12)
        assert curves[1, i] == pytest.approx(float(np.var(log_v, ddof=1)), rel=1e-12)
        assert curves[1, i] != pytest.approx(float(np.var(log_v, ddof=0)), rel=1e-6)


def test_mfdfa_reports_the_scaling_range_it_selected_on_q_equals_two() -> None:
    r"""Attack: pick the reference row at :math:`q\approx0` instead of :math:`q\approx2`.

    ``_reference_index`` chooses the single moment on which the scaling range —
    and the :math:`R^2` that is reported for the whole fit — is decided. Moving
    it to :math:`q=0` swaps in the geometric-mean row, whose :math:`R^2` is a
    different number and whose straightest sub-window is a different band of
    scales. The reported ``scaling_range.r_squared`` then describes a moment
    nobody asked about, and with ``select_range=True`` every exponent is fitted
    over the wrong window.
    """
    x = fgn(4096, 0.65, seed=11)
    exponents = mfdfa(x, q=np.linspace(-3.0, 3.0, 13), integrate=True)
    assert exponents.scaling_range is not None

    i_two = int(np.argmin(np.abs(exponents.q - 2.0)))
    i_zero = int(np.argmin(np.abs(exponents.q)))
    assert exponents.q[i_two] == pytest.approx(2.0)
    assert exponents.q[i_zero] == pytest.approx(0.0)
    assert exponents.scaling_range.r_squared == pytest.approx(
        exponents.fits[i_two].r_squared, rel=1e-12
    )
    assert exponents.scaling_range.r_squared != pytest.approx(
        exponents.fits[i_zero].r_squared, rel=1e-6
    )


# --------------------------------------------------------------------------- #
# models/cascade.py
# --------------------------------------------------------------------------- #


def test_trading_time_normalises_by_its_own_cumulative_endpoint() -> None:
    r"""Attack: divide the cumulative sum by ``m.sum()`` instead of by ``theta[-1]``.

    :func:`numpy.sum` is pairwise and :func:`numpy.cumsum` is sequential, so on
    a heterogeneous measure they disagree by :math:`O(n\varepsilon)`. Dividing
    the sequential running total by the pairwise one pushes the tail of
    :math:`\theta` above ``t_max``, the guard clip flattens it, and cells of a
    perfectly valid measure come back with an increment of *exactly zero* — the
    trading time silently stops being strictly increasing. On the binomial
    measure :math:`(0.8,0.2)` at :math:`k_{\max}=18` the mutated form produced
    3 zero increments and pinned 4 entries at ``t_max``; the correct form
    produces none and pins exactly one.
    """
    mu = deterministic_cascade(18, [0.8, 0.2])
    theta = trading_time(mu)

    increments = np.diff(theta)
    assert increments.size == mu.size
    assert float(increments.min()) > 0.0, "a strictly positive measure lost a cell"
    assert int(np.count_nonzero(theta == 1.0)) == 1
    assert theta[0] == 0.0
    assert theta[-1] == 1.0


# --------------------------------------------------------------------------- #
# analysis.py
# --------------------------------------------------------------------------- #


def test_increment_input_is_demeaned_before_it_is_integrated() -> None:
    r"""Attack: drop the mean subtraction from ``_to_level(x, "increments")``.

    The profile of Kantelhardt step 1 is :math:`Y(i)=\sum_{k\le i}(x_k-\bar x)`.
    Without the subtraction a drifting return series integrates into a ramp, and
    every estimator that differences the level rather than detrending it reads
    the ramp instead of the process: on fGn with :math:`H=0.35` plus a
    three-sigma drift, ``structure_function`` returned :math:`h(2)=0.9989` for
    the un-demeaned profile against :math:`0.3375` for the correct one. A
    reported Hurst exponent of one on a mean-reverting series is the textbook
    false positive this library is meant to prevent.
    """
    increments = fgn(8192, 0.35, seed=4)
    drifted = increments + 3.0 * float(np.std(increments))

    level = _to_level(drifted, "increments")
    assert abs(float(level[-1])) < 1e-8 * float(np.abs(level).max())

    q = np.linspace(-3.0, 3.0, 13)
    honest = analyze(drifted, estimator="structure_function", kind="increments", q=q)
    ramp = analyze(np.cumsum(drifted), estimator="structure_function", kind="level", q=q)
    assert honest.exponents.h2 < 0.5
    assert ramp.exponents.h2 > 0.9
