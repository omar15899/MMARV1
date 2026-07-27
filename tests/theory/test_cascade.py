r"""Validation of the cascade generators against closed-form theory.

The deterministic cascade is checked combinatorially and to machine precision:
its partition function is an identity, not an asymptotic law, so anything worse
than ~1e-12 there is a bug rather than sampling noise. The random families are
checked against their annealed moments with reported standard errors, and the
legacy per-cell-Gaussian normalisation has an explicit regression test.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy.special import comb, logsumexp

from mmar.core.rng import make_rng, spawn_rngs
from mmar.core.types import FloatArray
from mmar.core.validate import DataQualityError
from mmar.models import theory
from mmar.models.cascade import (
    CascadeFamily,
    CascadeSpec,
    cascade_measure,
    cascade_moments,
    deterministic_cascade,
    generate,
    log_gamma_multipliers,
    log_poisson_multipliers,
    log_stable_multipliers,
    lognormal_cascade,
    lognormal_multipliers,
    normalise_cascade,
    random_binomial_cascade,
    trading_time,
    uniform_binomial_multipliers,
)

pytestmark = pytest.mark.theory

M0 = 0.6
KMAX = 12
WEIGHTS = np.array([M0, 1.0 - M0])


# --------------------------------------------------------------------------- #
# Deterministic cascade: exact combinatorics
# --------------------------------------------------------------------------- #


def test_binomial_total_mass_is_one() -> None:
    mu = deterministic_cascade(KMAX, WEIGHTS)
    assert mu.size == 2**KMAX
    assert abs(float(mu.sum()) - 1.0) < 1e-12


def test_binomial_value_multiset_is_the_multinomial_one() -> None:
    r"""The cell values must be exactly :math:`\{m_0^j m_1^{k-j}\}` with weights C(k, j).

    The cell indexed by binary digits :math:`(d_1,\dots,d_k)` carries
    :math:`m_0^{\#\{d=0\}} m_1^{\#\{d=1\}}`, so the value depends only on the
    digit *count* and the multiplicity of the :math:`j`-zero value is
    :math:`\binom{k}{j}`. Floating point breaks exact equality only through the
    order of multiplication, so the match is asserted at relative 1e-12.
    """
    mu = deterministic_cascade(KMAX, WEIGHTS)
    theoretical = np.array([M0**j * (1.0 - M0) ** (KMAX - j) for j in range(KMAX + 1)])

    nearest = np.argmin(np.abs(np.log(mu)[:, None] - np.log(theoretical)[None, :]), axis=1)
    rel = np.abs(mu / theoretical[nearest] - 1.0)
    assert rel.max() < 1e-12, f"worst relative deviation {rel.max():.3e}"

    counts = np.bincount(nearest, minlength=KMAX + 1)
    expected = np.array([comb(KMAX, j, exact=True) for j in range(KMAX + 1)])
    assert np.array_equal(counts, expected)
    assert counts.sum() == 2**KMAX

    # The assignment must agree with the digit count, not merely with some binning.
    zeros = KMAX - np.array([int(i).bit_count() for i in range(2**KMAX)])
    assert np.array_equal(nearest, zeros)


def test_binomial_partition_function_is_exact() -> None:
    r"""``cascade_moments`` must reproduce ``theory.cascade_tau`` to machine precision.

    For a deterministic cascade :math:`Z(q,b^{-l}) = (\sum_i m_i^q)^l` exactly,
    so :math:`\log_b Z = -l\,\tau(q)` holds at every level with no asymptotics.
    """
    mu = deterministic_cascade(KMAX, WEIGHTS)
    q = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0])
    log_z = cascade_moments(mu, q, 2)
    tau = theory.cascade_tau(q, WEIGHTS, 2)
    levels = np.arange(KMAX + 1, dtype=np.float64)
    predicted = -np.outer(tau, levels)
    err = float(np.max(np.abs(log_z - predicted)))
    assert err < 1e-10, f"max |log_b Z - (-l tau)| = {err:.3e}"


def test_binomial_partition_function_anchors() -> None:
    """tau(0) = -1 (box counting) and tau(1) = 0 (mass conservation)."""
    mu = deterministic_cascade(8, WEIGHTS)
    log_z = cascade_moments(mu, np.array([0.0, 1.0, 2.0]), 2)
    levels = np.arange(9, dtype=np.float64)
    assert np.allclose(log_z[0], levels, atol=1e-12)  # tau(0) = -1
    assert np.allclose(log_z[1], 0.0, atol=1e-12)  # tau(1) = 0


def test_deterministic_cascade_is_bitwise_reproducible() -> None:
    a = deterministic_cascade(10, WEIGHTS)
    b = deterministic_cascade(10, WEIGHTS)
    assert np.array_equal(a, b)


def test_deterministic_cascade_rejects_non_conserving_weights() -> None:
    with pytest.raises(ValueError, match="conserves mass pathwise"):
        deterministic_cascade(4, np.array([0.6, 0.5]))


def test_deterministic_cascade_multinomial_base_four() -> None:
    m = np.array([0.4, 0.3, 0.2, 0.1])
    mu = deterministic_cascade(6, m)
    assert mu.size == 4**6
    q = np.array([-1.0, 0.0, 1.0, 2.0, 4.0])
    log_z = cascade_moments(mu, q, 4)
    predicted = -np.outer(theory.cascade_tau(q, m, 4), np.arange(7, dtype=np.float64))
    assert float(np.max(np.abs(log_z - predicted))) < 1e-10


# --------------------------------------------------------------------------- #
# Lognormal cascade
# --------------------------------------------------------------------------- #


def test_lognormal_multiplier_log_mean_is_minus_lambda() -> None:
    r""":math:`\log_b M = -V` with :math:`V\sim N(\lambda,\sigma^2)`."""
    lam = 1.05
    sigma2 = theory.lognormal_conservation_sigma2(lam, 2.0)
    sampler = lognormal_multipliers(lam, b=2)
    m = sampler(19, make_rng(7))
    assert m.size == 2**20

    log2_m = np.log2(m)
    mean = float(log2_m.mean())
    se = math.sqrt(sigma2 / m.size)
    assert abs(mean + lam) < 5.0 * se, f"mean log2 M = {mean:.6f}, target {-lam}, se {se:.2e}"
    assert abs(float(log2_m.var()) - sigma2) < 0.01


def test_lognormal_canonical_mass_is_one_in_expectation() -> None:
    r"""E[Omega] = 1 for the canonical cascade under the conservation constraint.

    The total mass is the martingale limit's finite-:math:`k` approximation; its
    variance is finite here (:math:`q_{\rm crit}=\sqrt{2/\sigma^2\ln b}\approx4.5`),
    so the sample mean is asymptotically normal and a 4-sigma band is a fair test.
    """
    lam, kmax, n_rep = 1.05, 10, 400
    masses = np.array(
        [
            float(lognormal_cascade(kmax, lam, b=2, seed=g, normalise=False).sum())
            for g in spawn_rngs(20260727, n_rep)
        ]
    )
    mean = float(masses.mean())
    se = float(masses.std(ddof=1)) / math.sqrt(n_rep)
    assert masses.min() > 0.0
    assert abs(mean - 1.0) < 4.0 * se, f"E[Omega] = {mean:.5f} +/- {se:.5f}"


def test_lognormal_conservative_mass_is_exact_per_level() -> None:
    """Microcanonical cascade: every sibling group sums to the parent mass exactly."""
    sampler = lognormal_multipliers(1.05, b=2, conservative=True)
    rng = make_rng(11)
    for k in range(8):
        m = sampler(k, rng)
        row_sums = m.reshape(-1, 2).sum(axis=1)
        assert float(np.max(np.abs(row_sums - 1.0))) < 1e-14

    mu = lognormal_cascade(12, 1.05, b=2, seed=3, conservative=True, normalise=False)
    assert abs(float(mu.sum()) - 1.0) < 1e-12
    # log_b Z(1, delta) == 0 at every coarse level is exactly per-level conservation.
    log_z1 = cascade_moments(mu, np.array([0.5, 1.0, 2.0]), 2)[1]
    assert float(np.max(np.abs(log_z1))) < 1e-12


def test_lognormal_is_bitwise_reproducible() -> None:
    kwargs = {"b": 2, "seed": 987654321, "normalise": False}
    a = lognormal_cascade(11, 1.1, **kwargs)  # type: ignore[arg-type]
    b = lognormal_cascade(11, 1.1, **kwargs)  # type: ignore[arg-type]
    assert np.array_equal(a, b)
    c = lognormal_cascade(11, 1.1, b=2, seed=987654322, normalise=False)
    assert not np.array_equal(a, c)


def test_lognormal_requires_lambda_above_one_when_deriving_sigma2() -> None:
    with pytest.raises(ValueError, match="conservation constraint"):
        lognormal_cascade(4, 0.9, seed=1)


@pytest.mark.stochastic
@pytest.mark.slow
def test_lognormal_annealed_tau_matches_theory() -> None:
    r"""Ensemble-averaged partition function recovers ``theory.lognormal_tau``.

    The closed form is the exponent of :math:`\mathbb{E}[Z]`, so the average is
    taken over replicas *before* the logarithm. :math:`q` is kept well below
    :math:`q_{\rm crit}\approx 4.5`, above which a single realisation's
    partition function is dominated by cells that almost surely do not occur.
    """
    lam, kmax, n_rep = 1.05, 11, 500
    sigma2 = theory.lognormal_conservation_sigma2(lam, 2.0)
    q = np.array([0.5, 1.0, 1.5, 2.0, 2.5])

    log_z = np.stack(
        [
            cascade_moments(lognormal_cascade(kmax, lam, b=2, seed=g, normalise=False), q, 2)
            for g in spawn_rngs(4242, n_rep)
        ]
    )
    # log_b E[Z] = logsumexp over replicas of (ln b * log_b Z) / ln b - log_b n_rep
    ln_b = math.log(2.0)
    log_mean_z = logsumexp(log_z * ln_b, axis=0) / ln_b - math.log2(n_rep)

    levels = np.arange(kmax + 1, dtype=np.float64)
    fit = levels[3:]
    tau_hat = np.array(
        [-np.polyfit(fit, log_mean_z[i, 3:], 1)[0] for i in range(q.size)], dtype=np.float64
    )
    tau_true = theory.lognormal_tau(q, lam, sigma2, 2.0)
    err = np.abs(tau_hat - tau_true)
    # The residual (~0.02 at q = 2.5) is a finite-ensemble bias, not noise: the
    # sample mean of Z under-resolves E[Z] as q climbs towards q_crit, because
    # the expectation is carried by replicas rarer than 1/n_rep.
    assert err.max() < 0.03, f"tau_hat={tau_hat}, tau_true={tau_true}, err={err}"


# --------------------------------------------------------------------------- #
# The other multiplier families: conservation in expectation is tau(1) = 0
# --------------------------------------------------------------------------- #


@pytest.mark.stochastic
@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("lognormal", lambda: lognormal_multipliers(1.05, b=2)),
        ("log_poisson", lambda: log_poisson_multipliers(0.2, 0.6, b=2)),
        ("log_gamma", lambda: log_gamma_multipliers(2.0, 0.3, b=2)),
        ("log_stable_1.7", lambda: log_stable_multipliers(1.7, 0.25, b=2)),
        ("log_stable_2.0", lambda: log_stable_multipliers(2.0, 0.3, b=2)),
        ("log_stable_0.7", lambda: log_stable_multipliers(0.7, 0.3, b=2)),
        ("uniform_binomial", lambda: uniform_binomial_multipliers(b=2)),
    ],
)
def test_multiplier_mean_is_one_over_b(name: str, factory: Callable[[], object]) -> None:
    """E[M] = 1/b for every family; this is exactly the statement tau(1) = 0."""
    sampler = factory()
    m = sampler(17, make_rng(2024))  # type: ignore[operator]
    assert m.size == 2**18
    assert np.all(m >= 0.0)
    mean = float(m.mean())
    se = float(m.std(ddof=1)) / math.sqrt(m.size)
    assert abs(mean - 0.5) < 5.0 * se + 1e-3, f"{name}: E[M] = {mean:.5f} +/- {se:.5f}"


def test_log_stable_alpha_two_reduces_to_lognormal() -> None:
    r"""alpha = 2 with scale sigma is the lognormal cascade with sigma2 = 2 sigma^2.

    Samorodnitsky--Taqqu's :math:`S_2(\sigma,\beta,0)` is :math:`N(0,2\sigma^2)`,
    so the variance of :math:`\log_b M` must be :math:`2\sigma^2`.
    """
    sigma = 0.3
    m = log_stable_multipliers(2.0, sigma, b=2)(17, make_rng(5))
    var_log2 = float(np.log2(m).var())
    target = 2.0 * sigma**2
    se = target * math.sqrt(2.0 / m.size)
    assert abs(var_log2 - target) < 6.0 * se, f"var log2 M = {var_log2:.5f}, target {target}"


def test_log_stable_rejects_alpha_one() -> None:
    with pytest.raises(ValueError, match="Cauchy"):
        log_stable_multipliers(1.0, 0.3, b=2)


def test_random_binomial_is_exactly_conservative() -> None:
    """The legacy (m, 1-m) cascade is microcanonical: total mass is 1 pathwise."""
    mu = random_binomial_cascade(12, seed=99)
    assert mu.size == 2**12
    assert abs(float(mu.sum()) - 1.0) < 1e-12
    assert np.array_equal(mu, random_binomial_cascade(12, seed=99))


# --------------------------------------------------------------------------- #
# Normalisation: the legacy per-cell Gaussian regression test
# --------------------------------------------------------------------------- #


def _legacy_final_factor(
    rng: np.random.Generator, lam: float, kmax: int, sigma: float
) -> FloatArray:
    """Reproduction of the bug: one independent Gaussian per cell.

    The predecessor multiplied the finest level by
    ``np.random.normal(2**(lam*kmax), sigma, 2**kmax)``.
    """
    return np.asarray(rng.normal(2.0 ** (lam * kmax), sigma, 2**kmax), dtype=np.float64)


def test_legacy_percell_gaussian_produces_negative_cells() -> None:
    """Regression witness: the construction this module replaces is not a measure."""
    lam, kmax, sigma, n_seeds = 0.2, 6, 2.0, 200
    mu = deterministic_cascade(kmax, WEIGHTS)
    negatives = 0
    for g in spawn_rngs(31337, n_seeds):
        broken = mu * _legacy_final_factor(g, lam, kmax, sigma)
        if float(broken.min()) < 0.0:
            negatives += 1
    assert negatives > 0.9 * n_seeds, f"only {negatives}/{n_seeds} seeds went negative"

    # And such an object is rejected downstream rather than silently analysed.
    broken = mu * _legacy_final_factor(make_rng(0), lam, kmax, sigma)
    with pytest.raises(DataQualityError, match="not a measure"):
        normalise_cascade(broken, mode="unit_mass")


def test_omega_normalisation_stays_strictly_positive_over_200_seeds() -> None:
    """The correct canonical normalisation: one positive scalar, never negative cells."""
    kmax, n_seeds = 6, 200
    mu = deterministic_cascade(kmax, WEIGHTS)
    minima = np.empty(n_seeds)
    masses = np.empty(n_seeds)
    for i, g in enumerate(spawn_rngs(31337, n_seeds)):
        out = normalise_cascade(mu, mode="omega", sigma_omega=1.0, seed=g)
        minima[i] = out.min()
        masses[i] = out.sum()
    assert float(minima.min()) > 0.0
    assert np.all(np.isfinite(masses))
    # Omega is a common factor, so the measure's *shape* is untouched.
    ref = normalise_cascade(mu, mode="omega", sigma_omega=1.0, seed=0)
    assert np.allclose(ref / ref.sum(), mu, rtol=0, atol=1e-15)


def test_omega_leaves_tau_invariant() -> None:
    r"""A single common factor shifts :math:`\log Z` by :math:`q\log\Omega`, not its slope."""
    mu = deterministic_cascade(10, WEIGHTS)
    scaled = normalise_cascade(mu, mode="omega", sigma_omega=1.5, seed=77)
    q = np.array([-1.0, 0.5, 1.0, 2.0, 3.0])
    d = cascade_moments(scaled, q, 2) - cascade_moments(mu, q, 2)
    # Each row must be constant across levels (the shift q*log_b Omega).
    assert float(np.max(np.abs(d - d[:, :1]))) < 1e-10


@pytest.mark.stochastic
@pytest.mark.parametrize("sigma_omega", [0.5, 1.0, 1.5])
def test_omega_has_unit_mean(sigma_omega: float) -> None:
    r"""The random total mass must satisfy :math:`\mathbb{E}[\Omega]=1`.

    :math:`\Omega=\exp\mathcal N(-s^2/2, s^2)` has mean
    :math:`e^{-s^2/2+s^2/2}=1`; the :math:`-s^2/2` drift is the whole content of
    the claim that ``"omega"`` is the canonical cascade's *unbiased* total mass
    rather than an arbitrary positive rescaling. Without this test a sign flip
    on that drift --- which multiplies the level of every simulated MMAR path by
    :math:`e^{s^2}` --- passes the entire suite.
    """
    n = 40_000
    mu = np.array([0.25, 0.25, 0.25, 0.25])
    omegas = np.array(
        [
            float(normalise_cascade(mu, mode="omega", sigma_omega=sigma_omega, seed=g).sum())
            for g in spawn_rngs(20260727, n)
        ]
    )
    assert float(omegas.min()) > 0.0
    se = float(omegas.std(ddof=1)) / math.sqrt(n)
    assert abs(float(omegas.mean()) - 1.0) < 4.0 * se, (
        f"E[Omega] = {omegas.mean():.5f} +/- {se:.5f} at sigma_omega={sigma_omega}; "
        f"a flipped drift would give exp(s^2) = {math.exp(sigma_omega**2):.3f}"
    )
    # The median pins the drift itself: median(Omega) = exp(-s^2/2).
    assert abs(float(np.median(omegas)) - math.exp(-0.5 * sigma_omega**2)) < 0.02


def test_normalise_modes() -> None:
    mu = deterministic_cascade(6, WEIGHTS) * 3.0
    assert abs(float(normalise_cascade(mu, mode="unit_mass").sum()) - 1.0) < 1e-14
    same = normalise_cascade(mu, mode="none")
    assert np.array_equal(same, mu) and same is not mu
    with pytest.raises(ValueError, match="unknown mode"):
        normalise_cascade(mu, mode="bogus")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Trading time
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("t_max", [1.0, 250.0])
def test_trading_time_is_a_cdf(t_max: float) -> None:
    mu = lognormal_cascade(10, 1.05, b=2, seed=13, normalise=False)
    theta = trading_time(mu, t_max=t_max)
    assert theta.size == mu.size + 1
    assert theta[0] == 0.0
    assert theta[-1] == t_max
    assert np.all(np.diff(theta) >= 0.0)
    # Cancellation: theta[i+1] - theta[i] subtracts two O(t_max) numbers to get an
    # O(t_max/n) increment, so the relative accuracy of the difference is only
    # ~n * eps, not eps.
    assert np.allclose(np.diff(theta), mu / mu.sum() * t_max, rtol=1e-9, atol=1e-15 * t_max)


def test_trading_time_of_deterministic_cascade_is_singular() -> None:
    """The binomial devil's staircase: strictly increasing but with a null-set support."""
    theta = trading_time(deterministic_cascade(10, WEIGHTS))
    assert np.all(np.diff(theta) > 0.0)
    # Half the mass sits in the first 2^-1... of the interval by construction of m0=0.6.
    assert abs(float(theta[theta.size // 2]) - M0) < 1e-12


@pytest.mark.parametrize(
    ("weights", "kmax"), [((0.9, 0.1), 20), ((0.99, 0.01), 12), ((0.8, 0.2), 20)]
)
def test_trading_time_survives_cumsum_vs_pairwise_sum_disagreement(
    weights: tuple[float, float], kmax: int
) -> None:
    r"""Regression: a valid measure must never be rejected by the monotonicity guard.

    ``np.cumsum`` is sequential and ``np.sum`` is pairwise, so their totals
    disagree by up to :math:`O(n\epsilon)` --- here by :math:`2\times10^{-11}`
    relative for ``(0.9, 0.1)`` at ``kmax = 20``. Normalising the sequential
    cumulative sum by the pairwise total pushed the tail above ``t_max``, and
    pinning ``theta[-1] = t_max`` on top of that produced a negative final
    increment and the bogus error "the input was not a measure".
    """
    mu = deterministic_cascade(kmax, np.array(weights))
    assert float(mu.min()) > 0.0  # every cell is strictly positive: this IS a measure
    theta = trading_time(mu, t_max=250.0)
    assert theta[0] == 0.0
    assert theta[-1] == 250.0
    assert np.all(np.diff(theta) >= 0.0)
    assert theta.size == mu.size + 1


def test_cascade_moments_refuses_a_grid_the_validator_would_reorder() -> None:
    """Rows are matched to ``q`` positionally, so a descending grid must not be answered.

    ``validate_q_grid`` sorts and de-duplicates; ``theory.cascade_tau`` does not.
    Feeding both a descending grid and comparing row-wise used to disagree by
    8.6 in ``log2 Z`` with no diagnostic at all.
    """
    mu = deterministic_cascade(6, WEIGHTS)
    ok = cascade_moments(mu, np.array([0.5, 1.0, 2.0]), 2)
    assert ok.shape == (3, 7)
    with pytest.raises(ValueError, match="strictly increasing"):
        cascade_moments(mu, np.array([2.0, 1.0, 0.5]), 2)
    with pytest.raises(ValueError, match="strictly increasing"):
        cascade_moments(mu, np.array([1.0, 1.0, 2.0, 3.0]), 2)


def test_trading_time_rejects_bad_input() -> None:
    with pytest.raises(DataQualityError):
        trading_time(np.array([1.0, -1.0, 2.0]))
    with pytest.raises(ValueError, match="t_max"):
        trading_time(np.array([1.0, 2.0]), t_max=0.0)


# --------------------------------------------------------------------------- #
# Engine guards and diagnostics
# --------------------------------------------------------------------------- #


def test_cascade_measure_rejects_wrong_multiplier_shape() -> None:
    def bad(k: int, rng: np.random.Generator) -> FloatArray:
        return np.ones(3, dtype=np.float64)

    with pytest.raises(ValueError, match=r"expected b\*\*\(k\+1\)"):
        cascade_measure(4, bad, b=2, seed=0)


def test_cascade_measure_rejects_negative_multipliers() -> None:
    def bad(k: int, rng: np.random.Generator) -> FloatArray:
        return -np.ones(2 ** (k + 1), dtype=np.float64)

    with pytest.raises(ValueError, match="negative multipliers"):
        cascade_measure(3, bad, b=2, seed=0)


def test_cascade_measure_normalises_by_default() -> None:
    mu = cascade_measure(
        5, lambda k, rng: np.full(2 ** (k + 1), 0.7, dtype=np.float64), b=2, seed=0
    )
    assert abs(float(mu.sum()) - 1.0) < 1e-14
    assert np.allclose(mu, 2.0**-5)


def test_cascade_moments_handles_empty_cells() -> None:
    """q < 0 diverges and must raise; q = 0 is box counting and must not."""
    mu = deterministic_cascade(6, np.array([0.0, 1.0]))
    assert int((mu == 0.0).sum()) == 2**6 - 1
    with pytest.raises(ValueError, match="diverges for every q < 0"):
        cascade_moments(mu, np.array([-1.0, 0.0, 1.0]), 2)
    log_z = cascade_moments(mu, np.array([0.0, 1.0, 2.0]), 2)
    # A single non-empty box at every level: Z(0) = 1 and Z(q) = 1 for all q.
    assert np.allclose(log_z, 0.0, atol=1e-14)


def test_cascade_moments_rejects_non_power_of_b_length() -> None:
    with pytest.raises(ValueError, match="not a power of b"):
        cascade_moments(np.ones(10), np.array([1.0, 2.0, 3.0]), 2)


def test_cell_guard_refuses_absurd_sizes() -> None:
    with pytest.raises(ValueError, match="cell guard"):
        deterministic_cascade(40, WEIGHTS)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_generate_records_seed_and_is_reproducible() -> None:
    spec = CascadeSpec(
        family=CascadeFamily.LOGNORMAL,
        kmax=10,
        b=2,
        params={"lam": 1.05},
        seed=20260727,
    )
    a = generate(spec)
    b = generate(spec)
    assert np.array_equal(a.measure, b.measure)
    assert a.provenance.root == b.provenance.root
    assert a.provenance.seed is not None
    assert a.provenance.seed.entropy == 20260727
    assert a.provenance.estimator == "cascade.lognormal"
    assert a.provenance.diagnostics["n_cells"] == 1024
    assert a.n_cells == 1024

    other = generate(
        CascadeSpec(
            family=CascadeFamily.LOGNORMAL, kmax=10, b=2, params={"lam": 1.1}, seed=20260727
        )
    )
    assert other.provenance.root != a.provenance.root


def test_generate_deterministic_family_carries_weight_digest() -> None:
    spec = CascadeSpec(family=CascadeFamily.DETERMINISTIC, kmax=8, b=2, weights=(M0, 1.0 - M0))
    res = generate(spec)
    assert res.provenance.inputs[0].name == "weights"
    assert abs(res.total_mass - 1.0) < 1e-12
    assert np.array_equal(res.measure, deterministic_cascade(8, WEIGHTS))
    assert res.trading_time()[-1] == 1.0


def test_generate_omega_normalisation_uses_an_independent_stream() -> None:
    base = CascadeSpec(
        family=CascadeFamily.LOGNORMAL, kmax=8, params={"lam": 1.05}, seed=5, normalisation="none"
    )
    with_omega = CascadeSpec(
        family=CascadeFamily.LOGNORMAL,
        kmax=8,
        params={"lam": 1.05},
        seed=5,
        normalisation="omega",
        sigma_omega=0.8,
    )
    a, b = generate(base), generate(with_omega)
    assert float(b.measure.min()) > 0.0
    # Different roots, and the omega variant is not a rescaling of the base draw
    # because its cascade stream was spawned rather than reused.
    assert a.provenance.root != b.provenance.root
    ratio = b.measure / a.measure
    assert float(ratio.std()) > 1e-9


def test_generate_missing_parameter_is_loud() -> None:
    with pytest.raises(ValueError, match=r"requires params\['delta'\]"):
        generate(CascadeSpec(family=CascadeFamily.LOG_POISSON, kmax=4, params={"gamma": 0.2}))
    with pytest.raises(ValueError, match=r"requires spec\.weights"):
        generate(CascadeSpec(family=CascadeFamily.DETERMINISTIC, kmax=4))
