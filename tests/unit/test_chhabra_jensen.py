"""Validation of the Chhabra--Jensen direct spectrum estimator.

Every tolerance below is justified in the test that uses it. The headline fact
that licenses the tight ones: for a deterministic multinomial cascade the
coarse-grained measure at every b-adic box factor *is* the level-k' cascade
exactly, so the Chhabra--Jensen numerators A, F and ln Z are exactly affine in
ln(delta). The regression is then exact and the only error is floating point.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from mmar.core.legendre import spectrum_from_exponents
from mmar.core.types import FitMethod, HolderSpectrum
from mmar.core.validate import DataQualityError
from mmar.estimators.chhabra_jensen import (
    chhabra_jensen,
    chhabra_jensen_fits,
    chhabra_jensen_from_surface,
    chhabra_jensen_series,
    chhabra_jensen_structure,
    chhabra_jensen_surface,
    coarse_grain,
)
from mmar.models import theory
from mmar.models.cascade import deterministic_cascade, trading_time
from mmar.models.fbm import fbm

M_BINOMIAL = (0.6, 0.4)
KMAX = 15
SEED = 20260727

# Exact recovery on the deterministic cascade: 1e-9 sits four orders above the
# observed float64 residual (~4e-13) and absorbs BLAS/platform variation in the
# weighted least-squares solve.
EXACT = 1e-9


def alpha_extent(spectrum: HolderSpectrum) -> float:
    """Range of Hoelder exponents the q window actually resolved.

    Deliberately *not* ``descriptors().width``: that measures the support at an
    f-level crossing, a spine-level convention, whereas what is being asserted
    below is a property of this estimator's own output on a given q grid.
    """
    return float(spectrum.alpha.max() - spectrum.alpha.min())


@pytest.fixture(scope="module")
def binomial() -> np.ndarray:
    return deterministic_cascade(KMAX, M_BINOMIAL)


@pytest.fixture(scope="module")
def q_grid() -> np.ndarray:
    return np.linspace(-5.0, 5.0, 41)


# --------------------------------------------------------------------------- #
# (b) Multifractal recovery against the closed form
# --------------------------------------------------------------------------- #


@pytest.mark.theory
def test_binomial_cascade_recovers_closed_form(binomial: np.ndarray, q_grid: np.ndarray) -> None:
    spectrum = chhabra_jensen(binomial, q=q_grid)

    alpha_exact = theory.cascade_alpha(q_grid, M_BINOMIAL)
    _, f_exact = theory.cascade_falpha_parametric(q_grid, M_BINOMIAL)
    tau_exact = theory.cascade_tau(q_grid, M_BINOMIAL)

    err_alpha = float(np.max(np.abs(spectrum.alpha - alpha_exact)))
    err_f = float(np.max(np.abs(spectrum.f_alpha - f_exact)))
    err_tau = float(np.max(np.abs(spectrum.exponents.tau - tau_exact)))

    assert err_alpha < EXACT, f"alpha error {err_alpha:.3e}"
    assert err_f < EXACT, f"f error {err_f:.3e}"
    assert err_tau < EXACT, f"tau error {err_tau:.3e}"

    # zeta = tau + d is the mandated conversion, and must match the oracle.
    zeta_exact = theory.cascade_zeta(q_grid, M_BINOMIAL, d=1.0)
    assert np.max(np.abs(spectrum.exponents.zeta - zeta_exact)) < EXACT


@pytest.mark.theory
def test_falpha_matches_entropy_formula(binomial: np.ndarray, q_grid: np.ndarray) -> None:
    """f evaluated at the recovered alpha must equal the binary-entropy closed form."""
    spectrum = chhabra_jensen(binomial, q=q_grid)
    f_entropy = theory.cascade_falpha(spectrum.alpha, M_BINOMIAL)
    assert np.max(np.abs(spectrum.f_alpha - f_entropy)) < EXACT


@pytest.mark.theory
def test_apex_and_endpoints(binomial: np.ndarray) -> None:
    """Apex f(alpha_0) = 1; the endpoints approach f = 0 once |q| is large enough."""
    q = np.linspace(-30.0, 30.0, 61)
    spectrum = chhabra_jensen(binomial, q=q)
    a_min, a_max = theory.cascade_alpha_range(M_BINOMIAL)

    assert abs(float(np.max(spectrum.f_alpha)) - 1.0) < EXACT
    assert abs(spectrum.alpha_0 - 0.5 * (a_min + a_max)) < 1e-6
    # At |q| = 30 the recovered support is 0.584956 against a true 0.584963.
    assert abs(alpha_extent(spectrum) - (a_max - a_min)) < 1e-5

    # alpha(q) -> alpha_min/alpha_max only as q -> +-infinity; at |q| = 30 the
    # residual gap is O((m1/m0)^30) = O(5e-6), hence the 1e-4 tolerance.
    assert abs(float(spectrum.alpha[-1]) - a_min) < 1e-4
    assert abs(float(spectrum.alpha[0]) - a_max) < 1e-4
    assert float(spectrum.f_alpha[-1]) < 1e-3
    assert float(spectrum.f_alpha[0]) < 1e-3


# --------------------------------------------------------------------------- #
# (c) Convention anchors
# --------------------------------------------------------------------------- #


@pytest.mark.theory
def test_convention_tau0_is_minus_d_and_tau1_is_zero(
    binomial: np.ndarray, q_grid: np.ndarray
) -> None:
    spectrum = chhabra_jensen(binomial, q=q_grid)
    q = spectrum.exponents.q
    tau = spectrum.exponents.tau

    i0 = int(np.argmin(np.abs(q)))
    i1 = int(np.argmin(np.abs(q - 1.0)))
    assert abs(q[i0]) < 1e-12 and abs(q[i1] - 1.0) < 1e-12

    assert abs(float(tau[i0]) + 1.0) < EXACT, "tau(0) must be -d for a space-filling measure"
    assert abs(float(tau[i1])) < EXACT, "tau(1) must vanish for a conservative measure"
    # f(0) = d is measured here, not imposed by a normalisation.
    assert abs(float(spectrum.f_alpha[i0]) - 1.0) < EXACT


@pytest.mark.theory
def test_tau_identity_holds_to_machine_precision(binomial: np.ndarray, q_grid: np.ndarray) -> None:
    """tau = q*alpha - f must equal the directly fitted slope of ln Z (linear fitters)."""
    surface = chhabra_jensen_surface(binomial, q=q_grid)
    fits = chhabra_jensen_fits(surface, fit_method=FitMethod.WLS)
    alpha, f_alpha, tau_direct = fits.slopes()
    gap = float(np.max(np.abs(q_grid * alpha - f_alpha - tau_direct)))
    assert gap < 1e-10, f"identity gap {gap:.3e}"


def test_cantor_measure_zero_cells_are_excluded_and_counted() -> None:
    """Triadic Cantor measure: f = alpha = log_3(2) exactly, with empty boxes present."""
    mu = deterministic_cascade(9, (0.5, 0.0, 0.5))
    q = np.linspace(-5.0, 5.0, 21)
    spectrum = chhabra_jensen(mu, q=q, d_support=1.0)

    d0 = np.log(2.0) / np.log(3.0)
    assert np.max(np.abs(spectrum.alpha - d0)) < EXACT
    assert np.max(np.abs(spectrum.f_alpha - d0)) < EXACT
    # tau(q) = (q - 1) * log_3 2 for a uniform measure on a Cantor support.
    assert np.max(np.abs(spectrum.exponents.tau - (q - 1.0) * d0)) < EXACT

    diag = spectrum.provenance.diagnostics
    assert diag["n_empty_total"] > 0, "the Cantor measure has empty boxes; they must be recorded"
    assert len(diag["n_empty"]) == len(diag["n_boxes"])
    assert all(e < b for e, b in zip(diag["n_empty"], diag["n_boxes"], strict=True))


# --------------------------------------------------------------------------- #
# (f) Negative q: the direct regression test for legacy defect D2
# --------------------------------------------------------------------------- #


@pytest.mark.theory
def test_negative_q_widens_the_recovered_alpha_range(binomial: np.ndarray) -> None:
    q_pos = np.linspace(0.0, 5.0, 21)
    q_full = np.linspace(-5.0, 5.0, 41)

    pos = chhabra_jensen(binomial, q=q_pos)
    full = chhabra_jensen(binomial, q=q_full)
    w_pos, w_full = alpha_extent(pos), alpha_extent(full)

    a_min, a_max = theory.cascade_alpha_range(M_BINOMIAL)
    support = a_max - a_min

    # Observed: w_pos = 0.224413 (38.4% of the 0.584956 support), w_full = 0.448826
    # (76.7%). q >= 0 reaches only the left branch alpha <= alpha(0); q < 0 adds the
    # right branch, which is symmetric for the binomial, hence the exact doubling.
    assert w_full > 1.9 * w_pos
    assert w_pos / support < 0.45
    assert w_full / support > 0.70
    assert float(full.alpha.max()) > float(pos.alpha.max()) + 0.2


# --------------------------------------------------------------------------- #
# (a) Monofractal null. The estimator's documented blind spot, with numbers.
# --------------------------------------------------------------------------- #


@pytest.mark.stochastic
@pytest.mark.parametrize("H", [0.3, 0.5, 0.7])
def test_fbm_total_variation_measure_is_uniform_not_hurst(H: float) -> None:
    """On fBm the increment measure is asymptotically uniform: alpha_0 -> 1 for every H.

    This is the documented property of the method, not a failure: the box mass is
    a sum of s absolute increments, whose relative fluctuation is O(s^{H-1}), so
    the total-variation measure carries no information about H. Observed at
    n = 2**14, q in [-5, 5], box factors 64..4096:

        H=0.3  alpha_0 = 1.00190  alpha extent = 0.03804
        H=0.5  alpha_0 = 1.00176  alpha extent = 0.03549
        H=0.7  alpha_0 = 1.00262  alpha extent = 0.05342
    """
    x = fbm(2**14 + 1, H, seed=SEED)
    scales = [2**j for j in range(6, 13)]
    spectrum = chhabra_jensen_series(x, q=np.linspace(-5.0, 5.0, 41), scales=scales)

    assert abs(spectrum.alpha_0 - 1.0) < 0.01, f"alpha_0 = {spectrum.alpha_0:.5f}"
    assert abs(float(np.max(spectrum.f_alpha)) - 1.0) < 1e-6
    assert alpha_extent(spectrum) < 0.06, f"noise floor = {alpha_extent(spectrum):.5f}"


@pytest.mark.stochastic
def test_fbm_noise_floor_shrinks_with_the_fine_scale_cutoff() -> None:
    """The apparent width on fBm is a fine-scale artefact and must collapse with s_min.

    Observed alpha extents (H = 0.5, n = 2**14, q in [-5, 5]):

        s_min=2    1.53854   <- reading this as multifractality is the trap
        s_min=16   0.12925
        s_min=64   0.03549
        s_min=256  0.01222
    """
    x = fbm(2**14 + 1, 0.5, seed=SEED)
    q = np.linspace(-5.0, 5.0, 41)
    widths = [
        alpha_extent(chhabra_jensen_series(x, q=q, scales=[2**j for j in range(j_min, 13)]))
        for j_min in (1, 4, 6, 8)
    ]
    assert widths[0] > 1.0, "the fine-scale artefact must be reproduced, not hidden"
    assert all(a > b for a, b in itertools.pairwise(widths))
    assert widths[-1] < 0.02


@pytest.mark.stochastic
def test_fbm_alpha0_is_insensitive_to_hurst() -> None:
    """H is not identifiable from the total-variation measure: alpha_0 spread < 1e-2."""
    scales = [2**j for j in range(6, 13)]
    q = np.linspace(-5.0, 5.0, 41)
    a0 = [
        chhabra_jensen_series(fbm(2**14 + 1, H, seed=SEED), q=q, scales=scales).alpha_0
        for H in (0.3, 0.5, 0.7)
    ]
    assert max(a0) - min(a0) < 1e-2, f"alpha_0 spread {max(a0) - min(a0):.2e} over H"


# --------------------------------------------------------------------------- #
# Series entry point on a genuine measure
# --------------------------------------------------------------------------- #


@pytest.mark.theory
def test_series_on_trading_time_reproduces_the_measure(
    binomial: np.ndarray, q_grid: np.ndarray
) -> None:
    """The two entry points describe the same object when the series is a CDF."""
    theta = trading_time(binomial)
    from_measure = chhabra_jensen(binomial, q=q_grid)
    from_series = chhabra_jensen_series(theta, q=q_grid)
    # np.diff of the cumulative sum is the measure up to accumulated rounding.
    assert np.max(np.abs(from_measure.alpha - from_series.alpha)) < 1e-6
    assert np.max(np.abs(from_measure.f_alpha - from_series.f_alpha)) < 1e-6
    assert from_measure.provenance.root != from_series.provenance.root


# --------------------------------------------------------------------------- #
# Cross-check against the Legendre route
# --------------------------------------------------------------------------- #


@pytest.mark.theory
def test_discrepancy_against_the_legendre_route(binomial: np.ndarray, q_grid: np.ndarray) -> None:
    """Direct vs Legendre on identical input; the gap is the differentiation error.

    Observed on the binomial cascade at kmax=15, q in [-5, 5], 41 points:

        max |alpha_CJ - alpha_Legendre(parametric)| = 4.061e-06
        max |f_CJ     - f_Legendre(parametric)|     = 2.031e-05
        rms |f_CJ     - f_Legendre(fenchel)|        = 1.114e-05

    Both routes see the *same* exactly recovered tau; the entire discrepancy is
    the spline derivative used by the Legendre construction. Against the oracle:
    max |alpha - alpha_exact| = 1.865e-14 direct versus 4.061e-06 via Legendre,
    a factor of 2e8 -- the differentiation, not the data, is the error.
    """
    direct = chhabra_jensen(binomial, q=q_grid)
    parametric = spectrum_from_exponents(direct.exponents, method="parametric")

    d_alpha = float(np.max(np.abs(direct.alpha - parametric.alpha)))
    d_f = float(np.max(np.abs(direct.f_alpha - parametric.f_alpha)))

    assert d_alpha < 1e-2 and d_f < 1e-2
    # The Legendre route is strictly worse against the oracle than the direct one.
    alpha_exact = theory.cascade_alpha(q_grid, M_BINOMIAL)
    err_direct = float(np.max(np.abs(direct.alpha - alpha_exact)))
    err_legendre = float(np.max(np.abs(parametric.alpha - alpha_exact)))
    assert err_direct < err_legendre
    assert err_direct < EXACT


# --------------------------------------------------------------------------- #
# (d) Determinism and (e) provenance
# --------------------------------------------------------------------------- #


def test_determinism_is_bitwise(binomial: np.ndarray, q_grid: np.ndarray) -> None:
    a = chhabra_jensen(binomial, q=q_grid, seed=11)
    b = chhabra_jensen(binomial, q=q_grid, seed=11)
    assert np.array_equal(a.alpha, b.alpha)
    assert np.array_equal(a.f_alpha, b.f_alpha)
    assert np.array_equal(a.exponents.zeta, b.exponents.zeta)


def test_provenance_root_is_stable_and_parameter_sensitive(
    binomial: np.ndarray, q_grid: np.ndarray
) -> None:
    base = chhabra_jensen(binomial, q=q_grid)
    assert base.provenance is not None
    root = base.provenance.root
    assert chhabra_jensen(binomial, q=q_grid).provenance.root == root

    variants = {
        "d_support": chhabra_jensen(binomial, q=q_grid, d_support=2.0),
        "fit_method": chhabra_jensen(binomial, q=q_grid, fit_method=FitMethod.OLS),
        "select_range": chhabra_jensen(binomial, q=q_grid, select_range=True),
        "q": chhabra_jensen(binomial, q=np.linspace(-4.0, 4.0, 41)),
        "scales": chhabra_jensen(binomial, q=q_grid, scales=[2**j for j in range(1, 12)]),
        "seed": chhabra_jensen(binomial, q=q_grid, seed=3),
        "min_points": chhabra_jensen(binomial, q=q_grid, min_points=6),
    }
    for name, variant in variants.items():
        assert variant.provenance.root != root, f"root did not change with {name}"

    other = chhabra_jensen(deterministic_cascade(KMAX, (0.7, 0.3)), q=q_grid)
    assert other.provenance.root != root, "root must change with the input measure"


def test_provenance_records_every_parameter(binomial: np.ndarray, q_grid: np.ndarray) -> None:
    spectrum = chhabra_jensen(binomial, q=q_grid, d_support=1.0, truncate=True, seed=7)
    params = dict(spectrum.provenance.params)
    for key in (
        "q",
        "scales",
        "fit_method",
        "select_range",
        "d_support",
        "truncate",
        "min_points",
        "seed",
    ):
        assert key in params, f"parameter {key} missing from provenance"
    assert spectrum.provenance.inputs[0].name == "mu"
    assert spectrum.provenance.estimator == "chhabra_jensen"


# --------------------------------------------------------------------------- #
# Surface reuse, structure functions, and refitting
# --------------------------------------------------------------------------- #


def test_structure_functions_carry_counts_and_refit_identically(
    binomial: np.ndarray, q_grid: np.ndarray
) -> None:
    surface = chhabra_jensen_surface(binomial, q=q_grid)
    structure = surface.to_structure_functions()
    assert structure.values.shape == (q_grid.size, surface.scales.size)
    assert structure.counts is not None
    assert np.array_equal(structure.counts, surface.counts)
    assert np.all(structure.values > 0.0)

    exported = chhabra_jensen_structure(binomial, q=q_grid)
    assert np.array_equal(exported.values, structure.values)

    # Refitting the cached surface must reproduce the full estimator bit for bit.
    reused = chhabra_jensen_from_surface(surface)
    fresh = chhabra_jensen(binomial, q=q_grid)
    assert np.array_equal(reused.alpha, fresh.alpha)
    assert np.array_equal(reused.f_alpha, fresh.f_alpha)


def test_refitting_a_narrower_range_needs_no_recomputation(
    binomial: np.ndarray, q_grid: np.ndarray
) -> None:
    surface = chhabra_jensen_surface(binomial, q=q_grid)
    wide = chhabra_jensen_from_surface(surface, select_range=False)
    narrow = chhabra_jensen_from_surface(surface, select_range=True, min_points=5)
    assert narrow.exponents.scaling_range.selection.startswith("auto")
    assert wide.exponents.scaling_range.selection == "manual"
    # Exact scaling means any sub-window recovers the same exponents.
    assert np.max(np.abs(wide.alpha - narrow.alpha)) < EXACT


def test_fits_expose_error_bars_for_alpha_and_f(binomial: np.ndarray, q_grid: np.ndarray) -> None:
    fits = chhabra_jensen_fits(chhabra_jensen_surface(binomial, q=q_grid))
    assert len(fits.alpha_fits) == len(fits.f_fits) == len(fits.tau_fits) == q_grid.size
    a_se, f_se, t_se = fits.stderrs()
    assert np.all(np.isfinite(a_se)) and np.all(np.isfinite(f_se)) and np.all(np.isfinite(t_se))
    # Exact scaling: the standard errors must be at the floating-point floor.
    assert float(np.max(a_se)) < 1e-9
    assert all(fit.r_squared > 1.0 - 1e-12 for fit in fits.alpha_fits)


def test_weighting_is_applied_by_default(binomial: np.ndarray, q_grid: np.ndarray) -> None:
    fits = chhabra_jensen_fits(chhabra_jensen_surface(binomial, q=q_grid))
    weights = fits.alpha_fits[0].weights
    assert weights is not None, "the default WLS fit must record its weights"
    assert np.all(np.diff(weights) < 0), "box counts must fall with the box factor"


# --------------------------------------------------------------------------- #
# Coarse-graining primitive and input guards
# --------------------------------------------------------------------------- #


def test_coarse_grain_is_reshape_and_sum() -> None:
    mu = np.arange(12, dtype=np.float64)
    assert np.array_equal(coarse_grain(mu, 3), np.array([3.0, 12.0, 21.0, 30.0]))
    assert np.array_equal(coarse_grain(mu, 1), mu)


def test_coarse_grain_requires_divisibility_unless_truncating() -> None:
    mu = np.arange(10, dtype=np.float64)
    with pytest.raises(ValueError, match="does not divide"):
        coarse_grain(mu, 3)
    out = coarse_grain(mu, 3, truncate=True)
    assert out.size == 3
    assert np.array_equal(out, np.array([3.0, 12.0, 21.0]))


def test_truncation_counts_are_recorded() -> None:
    """A prime length admits no exact box partition, so the tail is dropped and counted."""
    mu = np.abs(np.sin(np.arange(1009, dtype=np.float64))) + 1e-3
    spectrum = chhabra_jensen(mu, q=np.linspace(-3.0, 3.0, 13), truncate=True)
    diag = spectrum.provenance.diagnostics
    assert diag["n_truncated_total"] > 0, "1009 is prime; every box factor must truncate"
    assert len(diag["n_truncated"]) == len(diag["n_boxes"])

    with pytest.raises(ValueError, match="truncate=True"):
        chhabra_jensen(mu, q=np.linspace(-3.0, 3.0, 13), truncate=False)


def test_scales_leaving_too_few_boxes_are_rejected(binomial: np.ndarray) -> None:
    with pytest.raises(DataQualityError, match="too short"):
        chhabra_jensen(binomial, q=np.linspace(-3.0, 3.0, 13), scales=[binomial.size])


def test_non_divisor_scales_rejected_without_truncation(binomial: np.ndarray) -> None:
    with pytest.raises(ValueError, match="truncate=True"):
        chhabra_jensen(binomial, q=np.linspace(-3.0, 3.0, 13), scales=[3, 5, 7], truncate=False)


def test_negative_measure_is_rejected() -> None:
    mu = np.ones(256)
    mu[3] = -1.0
    with pytest.raises(DataQualityError, match="not a measure"):
        chhabra_jensen(mu, q=np.linspace(-3.0, 3.0, 13))


def test_select_range_needs_enough_scales(binomial: np.ndarray) -> None:
    with pytest.raises(ValueError, match="min_points"):
        chhabra_jensen(
            binomial,
            q=np.linspace(-3.0, 3.0, 13),
            scales=[2, 4, 8, 16],
            select_range=True,
            min_points=5,
        )


def test_constant_series_is_rejected() -> None:
    with pytest.raises(DataQualityError):
        chhabra_jensen_series(np.ones(1024))


# --------------------------------------------------------------------------- #
# Overflow of the audit surface must cost the surface, never the exponents
# --------------------------------------------------------------------------- #


def test_partition_overflow_drops_the_surface_but_keeps_the_exponents() -> None:
    """ln Z > 709 must not destroy a fit that was done entirely in the log domain."""
    mu = deterministic_cascade(20, M_BINOMIAL)
    q = np.linspace(-45.0, 45.0, 21)

    # The explicit surface export still raises: the caller asked for Z itself.
    with pytest.raises(ValueError, match="overflows float64"):
        chhabra_jensen_structure(mu, q=q)

    spectrum = chhabra_jensen(mu, q=q)
    assert np.all(np.isfinite(spectrum.alpha))
    assert np.all(np.isfinite(spectrum.f_alpha))
    assert abs(float(spectrum.exponents.tau[q.size // 2]) + 1.0) < EXACT
    # At |q| = 45 the tilt has not fully collapsed onto one branch: the residual
    # gap to the endpoint is O((m1/m0)^45) = O(7e-9), which is what is observed.
    a_min, a_max = theory.cascade_alpha_range(M_BINOMIAL)
    assert abs(float(spectrum.alpha[-1]) - a_min) < 1e-7
    assert abs(float(spectrum.alpha[0]) - a_max) < 1e-7

    assert spectrum.exponents.structure is None
    diag = spectrum.provenance.diagnostics
    assert "structure_omitted" in diag, "an omitted audit surface must say so"
    assert diag["log_partition_max"] > 709.0


def test_partition_surface_is_kept_when_it_fits() -> None:
    spectrum = chhabra_jensen(deterministic_cascade(12, M_BINOMIAL), q=np.linspace(-5.0, 5.0, 21))
    assert spectrum.exponents.structure is not None
    assert "structure_omitted" not in spectrum.provenance.diagnostics


# --------------------------------------------------------------------------- #
# Bias 5: the fine end of the DEFAULT ladder manufactures width
# --------------------------------------------------------------------------- #


def test_default_ladder_manufactures_width_on_an_iid_null() -> None:
    """The default s_min=2 returns a confident, smooth, entirely spurious spectrum.

    Reproduced, not hidden: i.i.d. |N(0,1)| cell masses carry no multifractality
    at all, yet the default ladder reports Delta-alpha ~ 1.8 with R^2 > 0.997.
    The artefact is a property of the fine-scale cutoff, not of the sample size.
    """
    mu = np.abs(np.random.default_rng(4242).standard_normal(2**16))
    q = np.linspace(-5.0, 5.0, 41)

    default = chhabra_jensen(mu, q=q)
    assert alpha_extent(default) > 1.0, "the s_min=2 artefact must be reproduced"

    widths = [
        alpha_extent(chhabra_jensen(mu, q=q, scales=[2**j for j in range(j_min, 15)]))
        for j_min in (1, 2, 4, 6, 8, 10)
    ]
    assert all(a > b for a, b in itertools.pairwise(widths)), "width must collapse with the cutoff"
    assert widths[-1] < 0.01, f"clean end of the ladder still shows {widths[-1]:.4f}"

    # Not a sample-size effect: the same cutoff gives the same width at every n.
    fixed = [
        alpha_extent(
            chhabra_jensen(
                np.abs(np.random.default_rng(4242).standard_normal(2**k)),
                q=q,
                scales=[2**j for j in range(4, k - 1)],
            )
        )
        for k in (12, 14, 16)
    ]
    assert max(fixed) - min(fixed) < 0.05, f"widths at s_min=16 over n: {fixed}"


def test_shuffle_surrogate_is_backwards_at_the_default_cutoff() -> None:
    """Shuffling destroys the cascade hierarchy yet WIDENS the default spectrum.

    Guard for the reader as much as for the code: a surrogate test on Delta-alpha
    with the default ladder concludes the opposite of the truth. Restricting the
    ladder to s >= 16 restores the correct ordering.
    """
    true_measure = deterministic_cascade(KMAX, M_BINOMIAL)
    shuffled = np.random.default_rng(0).permutation(true_measure)
    q = np.linspace(-5.0, 5.0, 41)

    assert alpha_extent(chhabra_jensen(shuffled, q=q)) > alpha_extent(
        chhabra_jensen(true_measure, q=q)
    )

    clean = [2**j for j in range(4, 14)]
    w_true = alpha_extent(chhabra_jensen(true_measure, q=q, scales=clean))
    w_shuf = alpha_extent(chhabra_jensen(shuffled, q=q, scales=clean))
    assert w_shuf < 0.5 * w_true, f"true {w_true:.5f} vs shuffled {w_shuf:.5f}"
    # The genuine cascade is exactly self-similar: its width is cutoff-invariant.
    assert abs(w_true - alpha_extent(chhabra_jensen(true_measure, q=q))) < EXACT
