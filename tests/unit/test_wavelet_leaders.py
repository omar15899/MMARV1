r"""Validation of :mod:`mmar.estimators.wavelet_leaders` against closed-form oracles.

Every tolerance below is a *measured* number rounded up, never a number chosen
to make a test pass. The measurements, all on the machine that produced this
file, are recorded in the ``OBSERVED`` tables so that a future regression can be
told apart from a re-tuning.

Sample sizes are the honest ones: the leader pyramid loses an octave per
halving, and a fair statement of an estimator's bias needs at least eight
octaves of scaling.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmar.core.legendre import spectrum_from_exponents
from mmar.core.rng import make_rng
from mmar.core.types import FitMethod, dyadic_scales
from mmar.estimators.wavelet_leaders import (
    exponents_from_structure,
    leader_cumulants,
    leader_pyramid,
    wavelet_coefficients_exponents,
    wavelet_coefficients_structure,
    wavelet_leaders,
    wavelet_leaders_structure,
)
from mmar.models import theory
from mmar.models.cascade import (
    cascade_measure,
    deterministic_cascade,
    lognormal_cascade,
    trading_time,
)
from mmar.models.fbm import fbm

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

N_FBM = 1 << 16
SEEDS = (11, 22, 33, 44, 55)
BINOMIAL = (0.6, 0.4)
KMAX = 18
"""Cascade depth for the multifractal-recovery tests: 2**18 = 262144 cells.

Depth 16 also works but the measured error at |q|=5 is 0.44 instead of 0.33;
the leader bias on a *deterministic* cascade decays like 1/log n, so two extra
octaves are worth the 2 MB.
"""
J_MIN_CASCADE = 7
r"""Finest octave used on the cascade.

The default ``j_min=3`` is wrong here and the module says why: count-weighted
least squares puts ~3/4 of its weight on the finest octave, and for a
deterministic cascade the DWT has not reached its asymptotic regime until
:math:`j\approx6`. Measured max :math:`|\Delta\zeta|` over :math:`|q|\le5` at
:math:`k_{\max}=18`: ``j_min=3 -> 0.91``, ``5 -> 0.52``, ``6 -> 0.33``,
``7 -> 0.33``, ``8 -> 0.36``. Seven is the plateau, not a hand-picked optimum.
"""


def binomial_signal(kmax: int = KMAX) -> np.ndarray:
    """Trading time of the deterministic binomial measure, on a power-of-two grid.

    ``trading_time`` returns the CDF on ``n+1`` cell boundaries; the leading zero
    is dropped so the length is a power of two and the dyadic wavelet grid lines
    up with the cascade's own dyadic grid.
    """
    return np.asarray(trading_time(deterministic_cascade(kmax, BINOMIAL))[1:], dtype=np.float64)


def randomised_binomial_signal(kmax: int, seed: int) -> np.ndarray:
    r"""Trading time of the *randomised* binomial cascade: identical :math:`\tau`.

    Each parent hands :math:`m_0` to a uniformly chosen child. The multiplier
    multiset per generation, hence :math:`\tau(q)=-\log_2(m_0^q+m_1^q)`, is
    unchanged; only the correlation between the wavelet filter's phase and the
    dyadic digit string is destroyed. Used to attribute the deterministic
    cascade's excess bias to exactly that correlation.
    """

    def sampler(k: int, rng: np.random.Generator) -> np.ndarray:
        n = 2**k
        left = rng.random(n) < 0.5
        out = np.empty(2 * n, dtype=np.float64)
        out[0::2] = np.where(left, BINOMIAL[0], BINOMIAL[1])
        out[1::2] = np.where(left, BINOMIAL[1], BINOMIAL[0])
        return out

    mu = cascade_measure(kmax, sampler, b=2, seed=seed, normalise=True)
    return np.asarray(trading_time(mu)[1:], dtype=np.float64)


# --------------------------------------------------------------------------- #
# (a) Monofractal recovery
# --------------------------------------------------------------------------- #

OBSERVED_FBM = """
n = 2**16, db3, default j_min=3, WLS by leader count, 5 seeds, q in [-5, 5]:

  H     mean h(q)   bias      max spread   mean c1    bias      max |c2|
  0.3   0.2450      -0.0550   0.0263       0.2431     -0.0569   0.00532
  0.5   0.4723      -0.0277   0.0191       0.4721     -0.0279   0.00472
  0.7   0.6877      -0.0123   0.0155       0.6903     -0.0097   0.00421

The bias is negative and grows as H falls: at small H the leader is a supremum
over rapidly *decaying* fine coefficients, so the fine octaves --- which carry
almost all the regression weight --- sit above the asymptote and flatten the
slope. It is a finite-resolution effect, not a coding error: raising j_min to 5
moves H=0.5 to -0.0066 and H=0.7 to -0.0031, at the cost of two octaves.

|c2| <= 0.0053 across all 15 runs is the monofractality resolution floor of this
estimator at n = 2**16: a measured |c2| below ~0.01 is not evidence of
multifractality.
"""


@pytest.mark.stochastic
@pytest.mark.parametrize("H", [0.3, 0.5, 0.7])
def test_fbm_gives_flat_hq_at_hurst(H: float) -> None:
    """h(q) is flat and centred on H on exact fBm; see OBSERVED_FBM for the numbers."""
    means, spreads, h2s = [], [], []
    for seed in SEEDS:
        r = wavelet_leaders(fbm(N_FBM, H, seed=seed))
        assert r.estimator == "wavelet_leaders"
        hq = r.hq[np.isfinite(r.hq)]
        means.append(float(hq.mean()))
        spreads.append(float(hq.max() - hq.min()))
        h2s.append(r.h2)

    bias = float(np.mean(means)) - H
    assert abs(bias) < 0.08, f"H={H}: h(q) bias {bias:+.4f} (observed -0.055..-0.012)"
    assert max(spreads) < 0.05, f"H={H}: h(q) spread {max(spreads):.4f} (observed <= 0.027)"
    assert abs(float(np.mean(h2s)) - H) < 0.08
    # Flatness is what monofractality *means*; the library's own predicate must agree.
    assert wavelet_leaders(fbm(N_FBM, H, seed=SEEDS[0])).is_monofractal(tol=0.05)


@pytest.mark.stochastic
@pytest.mark.parametrize("H", [0.3, 0.5, 0.7])
def test_fbm_cumulants_c1_is_hurst_and_c2_vanishes(H: float) -> None:
    """c1 ~= H and c2 ~= 0. The magnitude of c2 IS the monofractality test."""
    c1s, c2s, c3s = [], [], []
    for seed in SEEDS:
        lc = leader_cumulants(fbm(N_FBM, H, seed=seed))
        assert lc.order == 3 and lc.c.size == 3
        assert lc.stderr is not None and np.all(np.isfinite(lc.stderr))
        c1s.append(lc.c1)
        c2s.append(lc.c2)
        c3s.append(lc.c3)

    assert abs(float(np.mean(c1s)) - H) < 0.08, f"c1 bias {float(np.mean(c1s)) - H:+.4f}"
    worst_c2 = float(np.max(np.abs(c2s)))
    assert worst_c2 < 0.015, f"H={H}: |c2| = {worst_c2:.5f} on a monofractal (observed <= 0.0053)"
    assert float(np.max(np.abs(c3s))) < 0.02


@pytest.mark.stochastic
def test_fbm_zeta_matches_the_closed_form_q_times_hurst() -> None:
    """The oracle comparison: zeta(q) = qH pointwise, not merely on average."""
    H = 0.7
    r = wavelet_leaders(fbm(N_FBM, H, seed=SEEDS[0]))
    err = np.abs(r.zeta - theory.fbm_zeta(r.q, H))
    # The error grows linearly in |q| because it is a bias on the single slope h.
    assert float(np.max(err / np.maximum(np.abs(r.q), 1.0))) < 0.05


# --------------------------------------------------------------------------- #
# (b) Multifractal recovery
# --------------------------------------------------------------------------- #

OBSERVED_CASCADE = """
Deterministic binomial m0 = 0.6, kmax = 18 (n = 262144), trading-time integral,
db3, j_min = 7, WLS by leader count:

  max |zeta_hat - zeta_theory| over |q| <= 2 : 0.090
                                   |q| <= 3 : 0.167
                                   |q| <= 5 : 0.332
  c1 = 1.0051  (theory alpha_0 = 1.02945)      error -0.024
  c2 = -0.0705 (theory zeta''(0) = -0.05930)   error -0.011
  tau(1) = -0.0332                              error -0.033
  f(alpha) RMS error over {f_theory >= 0.3}: 0.108

The same statistic on the RANDOMISED binomial cascade (identical tau, kmax = 16,
j_min = 6, three seeds) is 0.128 - 0.146 over |q| <= 3, i.e. as good at a
quarter of the length. The excess on the deterministic cascade is therefore
attributable to the perfect correlation between the db3 filter's phase and the
cascade's dyadic digit string, which a random cascade averages away. It is also
strongly wavelet-dependent (max |dzeta| at kmax=16, j_min=3: db1 0.13,
db2 1.09, db3 0.94, db4 0.05, db6 1.43), which is the same phenomenon.
"""


@pytest.mark.theory
def test_binomial_cascade_recovers_zeta() -> None:
    """zeta(q) of the binomial measure, both branches. Tolerances from OBSERVED_CASCADE."""
    r = wavelet_leaders(binomial_signal(), j_min=J_MIN_CASCADE)
    zeta_true = theory.cascade_zeta(r.q, BINOMIAL)
    err = np.abs(r.zeta - zeta_true)

    assert float(err[np.abs(r.q) <= 2].max()) < 0.15, "|q| <= 2 (observed 0.090)"
    assert float(err[np.abs(r.q) <= 3].max()) < 0.25, "|q| <= 3 (observed 0.167)"
    # |q| = 5 is where the finite-sample contraction of the leader extremes bites;
    # the bound is deliberately loose and is a regression guard, not a claim.
    assert float(err.max()) < 0.45, "|q| <= 5 (observed 0.332)"

    # zeta must be increasing and concave-ish: h(q) = zeta/q decreasing in q.
    assert np.all(np.diff(r.zeta) > 0.0)


@pytest.mark.theory
def test_binomial_cascade_recovers_log_cumulants() -> None:
    r"""c1 = alpha_0 and c2 = zeta''(0), both in closed form for the binomial measure.

    :math:`\zeta(q)=1-\log_2(m_0^q+m_1^q)` gives
    :math:`c_1=\zeta'(0)=-\tfrac{1}{2}\log_2(m_0m_1)` and
    :math:`c_2=\zeta''(0)=-\tfrac{1}{4}\log_2 e \,\ln^2(m_0/m_1)`.
    """
    m0, m1 = BINOMIAL
    c1_true = -0.5 * np.log2(m0 * m1)
    c2_true = -0.25 * np.log(m0 / m1) ** 2 / np.log(2.0)
    assert c1_true == pytest.approx(1.029446, abs=1e-5)
    assert c2_true == pytest.approx(-0.059295, abs=1e-5)
    # Cross-check against the oracle's own alpha(q=0), which must equal c1.
    assert float(theory.cascade_alpha([0.0], BINOMIAL)[0]) == pytest.approx(c1_true, abs=1e-12)

    lc = leader_cumulants(binomial_signal(), j_min=J_MIN_CASCADE)
    assert lc.c1 == pytest.approx(c1_true, abs=0.06), f"c1={lc.c1:.4f} (observed 1.0051)"
    assert lc.c2 == pytest.approx(c2_true, abs=0.03), f"c2={lc.c2:.4f} (observed -0.0705)"
    assert lc.c2 < 0.0
    assert lc.intermittency > 0.0


@pytest.mark.theory
def test_binomial_cascade_recovers_falpha_on_both_branches() -> None:
    """The Legendre transform reproduces the entropy formula, left branch and right."""
    r = wavelet_leaders(binomial_signal(), j_min=J_MIN_CASCADE)
    sp = spectrum_from_exponents(r)
    f_true = theory.cascade_falpha(sp.alpha, BINOMIAL)

    core = np.isfinite(f_true) & (f_true >= 0.3)
    assert int(core.sum()) > 100
    rms = float(np.sqrt(np.mean((sp.f_alpha[core] - f_true[core]) ** 2)))
    assert rms < 0.18, f"f(alpha) RMS {rms:.4f} over f_theory >= 0.3 (observed 0.108)"

    d = sp.descriptors()
    a_min_true, a_max_true = theory.cascade_alpha_range(BINOMIAL)
    assert d.f_max == pytest.approx(1.0, abs=0.02)
    assert d.alpha_0 == pytest.approx(0.5 * (a_min_true + a_max_true), abs=0.06)
    # Both branches must be populated: this is the point of the estimator.
    assert int(np.sum(sp.alpha < d.alpha_0)) > 20
    assert int(np.sum(sp.alpha > d.alpha_0)) > 20
    # ...and the recovered support must sit inside the true one, never outside it.
    assert a_min_true - 0.05 < d.alpha_min < d.alpha_0
    assert d.alpha_0 < d.alpha_max < a_max_true + 0.05


@pytest.mark.theory
@pytest.mark.stochastic
def test_lognormal_cascade_recovers_c1_and_c2() -> None:
    r"""The headline claim: :math:`c_1=\lambda` and :math:`c_2=-\sigma^2\ln 2`.

    Observed over 6 seeds at :math:`k_{\max}=16`, ``j_min=5``:
    :math:`\hat c_1 = 1.0425 \pm 0.0232` against :math:`\lambda=1.05`, and
    :math:`\hat c_2 = -0.1046 \pm 0.0180` against :math:`-\sigma^2\ln2 = -0.1000`.
    Both within one standard deviation of the truth, on a *random* cascade with
    no closed-form sample path.
    """
    lam = 1.05
    sigma2 = theory.lognormal_conservation_sigma2(lam, 2.0)
    c1_true, c2_true = theory.cumulants_from_tau_lognormal(lam, sigma2, 2.0)
    assert c1_true == pytest.approx(1.05) and c2_true == pytest.approx(-0.1, abs=1e-12)

    c1s, c2s = [], []
    for seed in range(1, 7):
        signal = trading_time(lognormal_cascade(16, lam, seed=seed))[1:]
        lc = leader_cumulants(signal, j_min=5)
        c1s.append(lc.c1)
        c2s.append(lc.c2)

    assert float(np.mean(c1s)) == pytest.approx(c1_true, abs=0.06)
    assert float(np.mean(c2s)) == pytest.approx(c2_true, abs=0.04)
    assert float(np.std(c1s)) < 0.06
    # Every individual realisation must at least get the sign of c2 right.
    assert all(c < 0.0 for c in c2s)


@pytest.mark.theory
@pytest.mark.stochastic
def test_randomising_the_cascade_removes_most_of_the_deterministic_bias() -> None:
    """Attribution test for the bias claim in OBSERVED_CASCADE.

    Same tau, quarter of the length, coarser octave floor -- and a smaller error.
    If this ever fails, the deterministic-cascade tolerance above is hiding a
    genuine implementation defect rather than a filter/digit-string resonance.
    """
    errs = []
    for seed in (1, 2, 3):
        r = wavelet_leaders(randomised_binomial_signal(16, seed), j_min=6)
        e = np.abs(r.zeta - theory.cascade_zeta(r.q, BINOMIAL))
        errs.append(float(e[np.abs(r.q) <= 3].max()))
    assert max(errs) < 0.25, f"randomised binomial |q|<=3 errors {errs} (observed 0.128-0.146)"

    deterministic = wavelet_leaders(binomial_signal(16), j_min=6)
    e_det = np.abs(deterministic.zeta - theory.cascade_zeta(deterministic.q, BINOMIAL))
    assert float(e_det[np.abs(deterministic.q) <= 3].max()) > float(np.mean(errs))


# --------------------------------------------------------------------------- #
# (c) Convention checks
# --------------------------------------------------------------------------- #


def test_zeta_at_zero_is_exactly_zero_and_tau_at_zero_is_minus_d() -> None:
    r"""``S_L(0, 2^j) = 1`` identically, so :math:`\hat\zeta(0)=0` to machine precision.

    Unlike an increment-based estimator --- where :math:`\hat\tau(0)=-1` is a
    genuine consistency check on the window bookkeeping and legacy v1 failed it
    by 1.7% (defect D6) --- here it is exact by construction, and *that* is the
    property worth pinning: any drift means the count normalisation
    :math:`1/n_j` has been lost.
    """
    for signal in (fbm(N_FBM, 0.6, seed=1), binomial_signal(16)):
        r = wavelet_leaders(signal, j_min=4)
        i0 = int(np.argmin(np.abs(r.q)))
        assert r.q[i0] == 0.0
        assert abs(float(r.zeta[i0])) < 1e-12
        assert float(r.tau[i0]) == pytest.approx(-1.0, abs=1e-12)
        assert np.allclose(r.structure.values[i0], 1.0, atol=1e-12)  # type: ignore[union-attr]

    # d_support is carried, not hard-coded.
    r2 = wavelet_leaders(fbm(N_FBM, 0.6, seed=1), j_min=4, d_support=2.0)
    assert float(r2.tau[int(np.argmin(np.abs(r2.q)))]) == pytest.approx(-2.0, abs=1e-12)


@pytest.mark.theory
def test_tau_at_one_vanishes_for_a_conservative_measure() -> None:
    r"""Mass conservation is exactly :math:`\tau(1)=0`; observed :math:`-0.0332`."""
    r = wavelet_leaders(binomial_signal(), j_min=J_MIN_CASCADE)
    tau1 = float(np.interp(1.0, r.q, r.tau))
    assert abs(tau1) < 0.06, f"tau(1) = {tau1:+.4f} for a conservative cascade"

    # The MMAR identity tau(1/H) = 0 degenerates to H = 1 here (the trading time
    # is the measure itself, not a subordinated motion), so the root of tau on
    # the positive branch must land near q = 1.
    assert r.hurst_tau_root == pytest.approx(1.0, abs=0.06)


# --------------------------------------------------------------------------- #
# (d) Determinism
# --------------------------------------------------------------------------- #


def test_repeated_calls_are_bitwise_identical() -> None:
    x = fbm(1 << 15, 0.65, seed=999)
    a = wavelet_leaders(x, seed=7)
    b = wavelet_leaders(x, seed=7)
    assert np.array_equal(a.zeta, b.zeta)
    assert np.array_equal(a.zeta_stderr, b.zeta_stderr)  # type: ignore[arg-type]
    assert np.array_equal(a.structure.values, b.structure.values)  # type: ignore[union-attr]

    ca, cb = leader_cumulants(x, seed=7), leader_cumulants(x, seed=7)
    assert np.array_equal(ca.c, cb.c)

    # The estimator is deterministic, so the seed must not move a single bit.
    assert np.array_equal(wavelet_leaders(x, seed=1).zeta, wavelet_leaders(x, seed=2).zeta)


def test_generator_inputs_reproduce_through_the_model_seed() -> None:
    """The signal, not the estimator, is the stochastic part; make_rng is the only source."""
    a = wavelet_leaders(make_rng(5).standard_normal(1 << 14).cumsum())
    b = wavelet_leaders(make_rng(5).standard_normal(1 << 14).cumsum())
    assert np.array_equal(a.zeta, b.zeta)


# --------------------------------------------------------------------------- #
# (e) Provenance
# --------------------------------------------------------------------------- #


def test_provenance_root_is_stable_and_parameter_sensitive() -> None:
    x = fbm(1 << 15, 0.65, seed=123)
    base = wavelet_leaders(x)
    assert base.provenance is not None
    root = base.provenance.root
    assert wavelet_leaders(x).provenance.root == root  # type: ignore[union-attr]

    variants = {
        "wavelet": dict(wavelet="db4"),
        "j_min": dict(j_min=4),
        "j_max": dict(j_max=9),
        "min_leaders": dict(min_leaders=16),
        "boundary_margin": dict(boundary_margin=6),
        "fit_method": dict(fit_method=FitMethod.OLS),
        "select_range": dict(select_range=True),
        "d_support": dict(d_support=2.0),
        "saturation_tolerance": dict(saturation_tolerance=0.9),
        "q": dict(q=np.linspace(-4.0, 4.0, 41)),
        "scales": dict(scales=dyadic_scales(1 << 15, j_min=4, j_max=10)),
        "seed": dict(seed=3),
    }
    for name, kwargs in variants.items():
        other = wavelet_leaders(x, **kwargs)  # type: ignore[arg-type]
        assert other.provenance is not None
        assert other.provenance.root != root, f"changing {name} did not change the root"

    # A different input array must change the root too.
    assert wavelet_leaders(fbm(1 << 15, 0.65, seed=124)).provenance.root != root  # type: ignore[union-attr]

    # Diagnostics are non-identifying: they must not perturb the digest.
    assert base.provenance.with_diagnostics(anything="else").root == root


def test_provenance_records_every_parameter_and_the_leader_bookkeeping() -> None:
    x = fbm(1 << 15, 0.65, seed=321)
    r = wavelet_leaders(x, wavelet="db4", j_min=4, min_leaders=12, select_range=False)
    assert r.provenance is not None
    params = dict(r.provenance.params)
    for key in (
        "q",
        "scales",
        "wavelet",
        "j_min",
        "j_max",
        "boundary_margin",
        "min_leaders",
        "fit_method",
        "select_range",
        "d_support",
        "saturation_tolerance",
    ):
        assert key in params, f"parameter {key} is not recorded in provenance"
    assert params["wavelet"] == "db4"

    diag = dict(r.provenance.diagnostics)
    for key in (
        "octaves",
        "n_leaders_per_octave",
        "n_border_dropped_per_octave",
        "n_border_dropped_total",
        "n_zero_dropped_total",
        "n_vanishing_moments",
        "boundary_margin",
        "alpha_min_estimated",
        "alpha_max_estimated",
        "saturation_margin",
        "saturated",
        "scaling_range",
    ):
        assert key in diag, f"diagnostic {key} missing"
    # db4 has N_psi = 4, so the default margin is 5 and 10 coefficients go per octave.
    assert diag["n_vanishing_moments"] == 4
    assert diag["boundary_margin"] == 5
    assert all(n == 10 for n in diag["n_border_dropped_per_octave"])
    assert diag["n_border_dropped_total"] > 0

    assert r.provenance.inputs[0].name == "x"
    assert r.provenance.inputs[0].shape == (x.size,)
    assert r.provenance.to_manifest()["root"] == r.provenance.root


def test_result_is_fully_populated() -> None:
    r = wavelet_leaders(fbm(1 << 15, 0.6, seed=8))
    assert r.zeta_stderr is not None and np.all(np.isfinite(r.zeta_stderr))
    assert r.zeta_ci is not None
    lo, hi = r.zeta_ci
    assert np.all(lo <= r.zeta) and np.all(r.zeta <= hi)
    assert len(r.fits) == r.q.size
    assert r.scaling_range is not None and r.scaling_range.n_scales >= 3
    assert r.structure is not None and r.structure.counts is not None
    assert r.structure.values.shape == (r.q.size, r.structure.scales.size)
    # Counts must fall geometrically: one leader per two, per octave.
    counts = np.asarray(r.structure.counts, dtype=np.float64)
    assert np.all(np.diff(counts) < 0)
    assert r.provenance is not None


# --------------------------------------------------------------------------- #
# (f) Negative q -- the direct regression test for legacy defect D2
# --------------------------------------------------------------------------- #

OBSERVED_NEGATIVE_Q = """
Binomial cascade, kmax = 18, j_min = 7, Legendre-Fenchel spectrum:

  q in [-5, 5] : alpha in [0.7274, 1.2360], width 0.5086
  q in [ 0, 5] : alpha in [0.7273, 1.0050], width 0.2777
  theory       : alpha in [0.7370, 1.3219], width 0.5850

Dropping q < 0 costs 45% of the spectrum and *all* of the right branch, exactly
as the legacy audit measured (24% of the support recovered at q in [0, 3]).
The right branch is not merely noisy without negative q -- it is unreachable.
"""


@pytest.mark.theory
def test_negative_q_widens_the_recovered_alpha_range() -> None:
    signal = binomial_signal()
    full = wavelet_leaders(signal, j_min=J_MIN_CASCADE, q=np.linspace(-5.0, 5.0, 41))
    positive = wavelet_leaders(signal, j_min=J_MIN_CASCADE, q=np.linspace(0.0, 5.0, 21))

    d_full = spectrum_from_exponents(full).descriptors()
    d_pos = spectrum_from_exponents(positive).descriptors()

    assert d_full.width > 1.5 * d_pos.width, (
        f"negative q must widen the spectrum: {d_full.width:.4f} vs {d_pos.width:.4f}"
    )
    # The apex is the boundary of the q >= 0 world; the right branch lies beyond it.
    assert d_pos.alpha_max == pytest.approx(d_pos.alpha_0, abs=0.02)
    assert d_full.alpha_max > d_full.alpha_0 + 0.15

    a_min_true, a_max_true = theory.cascade_alpha_range(BINOMIAL)
    coverage_full = d_full.width / (a_max_true - a_min_true)
    coverage_pos = d_pos.width / (a_max_true - a_min_true)
    assert coverage_full > 0.75, f"only {coverage_full:.0%} of the support recovered"
    assert coverage_pos < 0.55


@pytest.mark.stochastic
def test_leader_negative_moments_converge_where_coefficient_ones_do_not() -> None:
    r"""Why leaders exist, measured.

    On fBm with :math:`H=0.7` the truth is :math:`\zeta(-5)=-3.5`. Observed:

      n        leaders     bare coefficients
      2**14    -3.168      -6.901
      2**16    -3.413      -1.520
      2**18    -3.383      -4.146

    The leader estimate converges; the coefficient estimate wanders by five
    units with no trend, because :math:`\langle|d|^{-5}\rangle` is carried by
    whichever coefficient happens to be nearest zero at that sample size.
    """
    truth = -3.5
    leader_err, coeff_err = [], []
    for n in (1 << 14, 1 << 16, 1 << 18):
        x = fbm(n, 0.7, seed=7)
        leader_err.append(abs(float(wavelet_leaders(x).zeta[0]) - truth))
        coeff_err.append(abs(float(wavelet_coefficients_exponents(x).zeta[0]) - truth))

    assert max(leader_err) < 0.5, f"leader zeta(-5) errors {leader_err}"
    assert max(coeff_err) > 3.0 * max(leader_err), (
        f"coefficient zeta(-5) errors {coeff_err} should be far worse than {leader_err}"
    )


def test_coefficient_variant_can_refuse_negative_q_and_flags_it_otherwise() -> None:
    x = fbm(1 << 15, 0.7, seed=2)
    with pytest.raises(ValueError, match="negative moment orders"):
        wavelet_coefficients_exponents(x, strict_positive_q=True)

    r = wavelet_coefficients_exponents(x)
    assert r.estimator == "wavelet_coefficients"
    assert r.provenance is not None
    assert r.provenance.diagnostics["negative_q_invalid"] is True

    ok = wavelet_coefficients_exponents(x, q=np.linspace(0.5, 5.0, 19), strict_positive_q=True)
    assert ok.provenance is not None
    assert ok.provenance.diagnostics["negative_q_invalid"] is False
    # For q > 0 on a monofractal the two estimators must agree.
    leaders = wavelet_leaders(x, q=np.linspace(0.5, 5.0, 19))
    assert float(np.max(np.abs(ok.hq - leaders.hq))) < 0.1


# --------------------------------------------------------------------------- #
# Vanishing-moment saturation diagnostic
# --------------------------------------------------------------------------- #


def test_saturation_is_flagged_when_alpha_approaches_n_psi() -> None:
    r"""Haar has :math:`N_\psi=1`; on fBm with :math:`H=0.7` the margin is 0.27."""
    x = fbm(N_FBM, 0.7, seed=3)

    haar = wavelet_leaders(x, wavelet="db1")
    assert haar.provenance is not None
    d = haar.provenance.diagnostics
    assert d["n_vanishing_moments"] == 1
    assert d["saturated"] is True
    assert d["saturation_margin"] < 0.5
    assert "saturation_warning" in d
    assert "vanishing moments" in d["saturation_warning"]

    db3 = wavelet_leaders(x, wavelet="db3")
    assert db3.provenance is not None
    assert db3.provenance.diagnostics["saturated"] is False
    assert "saturation_warning" not in db3.provenance.diagnostics

    # The cumulant estimator carries the same guard, keyed on c1.
    assert "saturation_warning" in leader_cumulants(x, wavelet="db1").provenance.diagnostics  # type: ignore[union-attr]


def test_saturation_actually_happens_beyond_n_psi() -> None:
    r"""The bias the diagnostic warns about, demonstrated rather than asserted.

    Integrating fBm twice raises its regularity to :math:`h=H+2=2.7` exactly
    (each integration adds one to the Hölder exponent). Observed :math:`\hat c_1`
    at :math:`n=2^{15}`, ``j_min=3``:

      db1 (N_psi=1) 1.041   db2 (2) 2.032   db3 (3) 2.706   db4 (4) 2.685

    Each short wavelet reports its own :math:`N_\psi`, not the signal's
    exponent -- and does so with a perfectly straight log-log plot. Only
    :math:`N_\psi>h` recovers the truth.
    """
    smooth = np.cumsum(np.cumsum(fbm(1 << 15, 0.7, seed=17)))

    for wavelet, ceiling in (("db1", 1.0), ("db2", 2.0)):
        lc = leader_cumulants(smooth, wavelet=wavelet, j_min=3)
        assert lc.c1 == pytest.approx(ceiling, abs=0.15), f"{wavelet} c1 = {lc.c1:.3f}"
        assert lc.provenance is not None
        assert "saturation_warning" in lc.provenance.diagnostics

    resolved = leader_cumulants(smooth, wavelet="db4", j_min=3)
    assert resolved.c1 == pytest.approx(2.7, abs=0.15), f"db4 c1 = {resolved.c1:.3f}"
    assert resolved.provenance is not None
    assert "saturation_warning" not in resolved.provenance.diagnostics


def test_uniform_regularity_diagnostic_on_white_noise() -> None:
    r"""White noise violates :math:`h_{\min}>0`; the estimate collapses towards zero.

    The leader supremum over the :math:`\mathcal O(2^j)` finest coefficients
    cancels their :math:`2^{-j/2}` decay, so the reported
    :math:`\hat\alpha_{\min}` is near 0 rather than the true :math:`-1/2`. Near
    zero is itself the signal that the formalism's hypothesis has failed.
    """
    r = wavelet_leaders(make_rng(1).standard_normal(N_FBM))
    assert r.provenance is not None
    assert abs(float(r.provenance.diagnostics["alpha_min_estimated"])) < 0.2


# --------------------------------------------------------------------------- #
# Structure functions, refitting, scale grids
# --------------------------------------------------------------------------- #


def test_structure_only_entry_point_matches_the_full_estimator() -> None:
    x = fbm(1 << 15, 0.62, seed=71)
    st = wavelet_leaders_structure(x)
    direct = wavelet_leaders(x)
    assert st.values.shape == direct.structure.values.shape  # type: ignore[union-attr]
    assert np.array_equal(st.values, direct.structure.values)  # type: ignore[union-attr]
    assert np.array_equal(st.counts, direct.structure.counts)  # type: ignore[arg-type, union-attr]

    refit = exponents_from_structure(st)
    assert np.allclose(refit.zeta, direct.zeta, rtol=0, atol=1e-12)

    # The point of keeping the surface: refit over a narrower window, no DWT.
    narrow = exponents_from_structure(st, scale_range=(32.0, 2048.0))
    assert narrow.scaling_range is not None
    assert narrow.scaling_range.s_min == 32.0 and narrow.scaling_range.s_max == 2048.0
    assert not np.allclose(narrow.zeta, direct.zeta)

    coeff_st = wavelet_coefficients_structure(x)
    assert coeff_st.values.shape == st.values.shape
    assert not np.array_equal(coeff_st.values, st.values)


def test_scales_must_be_dyadic_and_are_honoured() -> None:
    x = fbm(1 << 15, 0.6, seed=4)
    with pytest.raises(ValueError, match="powers of two"):
        wavelet_leaders(x, scales=[10, 20, 40, 80])

    scales = dyadic_scales(x.size, j_min=5, j_max=10)
    r = wavelet_leaders(x, scales=scales)
    assert r.structure is not None
    assert np.array_equal(r.structure.scales, scales.astype(np.float64))
    assert r.provenance is not None
    assert r.provenance.diagnostics["octaves"] == [5, 6, 7, 8, 9, 10]

    with pytest.raises(ValueError, match="available"):
        wavelet_leaders(x, scales=dyadic_scales(x.size, j_min=13, j_max=14))


def test_default_scale_grid_is_geometric_and_respects_the_window_floor() -> None:
    x = fbm(1 << 14, 0.6, seed=6)
    p = leader_pyramid(x)
    ratios = np.diff(p.scales) / p.scales[:-1]
    assert np.allclose(ratios, 1.0)  # every step doubles
    assert int(p.octaves.min()) == 3
    # No octave may leave fewer than min_leaders uncontaminated leaders.
    assert int(p.counts.min()) >= 8
    assert np.all(p.counts == (x.size >> p.octaves) - 2 * p.boundary_margin)


def test_select_range_picks_a_window_and_records_it() -> None:
    x = fbm(1 << 16, 0.7, seed=9)
    r = wavelet_leaders(x, select_range=True)
    assert r.scaling_range is not None
    assert r.scaling_range.selection.startswith("auto-")
    assert r.scaling_range.n_scales >= 5
    assert r.scaling_range.decades > 0.5
    # A single window is chosen once and shared by every q; not one per moment.
    assert all(f.n_points == r.scaling_range.n_scales for f in r.fits)


def test_weighting_is_by_leader_count_under_wls() -> None:
    x = fbm(1 << 15, 0.6, seed=10)
    r = wavelet_leaders(x, fit_method=FitMethod.WLS)
    fit = r.fits[0]
    assert fit.method is FitMethod.WLS
    assert fit.weights is not None
    assert np.array_equal(fit.weights, np.asarray(r.structure.counts, dtype=np.float64))  # type: ignore[union-attr]

    ols = wavelet_leaders(x, fit_method=FitMethod.OLS)
    assert ols.fits[0].weights is None
    assert not np.allclose(ols.zeta, r.zeta)


# --------------------------------------------------------------------------- #
# Leader construction, boundaries, zeros
# --------------------------------------------------------------------------- #


def test_leaders_dominate_the_coefficients_they_are_built_from() -> None:
    r"""``L(j,k) >= |d(j,k)|`` and ``L`` is non-decreasing along the recursion."""
    x = fbm(1 << 14, 0.6, seed=12)
    lead = leader_pyramid(x, j_min=3)
    coef = leader_pyramid(x, j_min=3, leaders=False)
    assert np.array_equal(lead.octaves, coef.octaves)
    for a, b in zip(lead.values, coef.values, strict=True):
        assert a.size == b.size
        assert np.all(a >= b - 1e-15)
    # The supremum lifts the small coefficients much more than the large ones.
    assert float(np.min(lead.values[0])) > float(np.min(coef.values[0]))


def test_boundary_coefficients_are_discarded_and_counted() -> None:
    x = fbm(1 << 14, 0.6, seed=13)
    for wavelet, n_psi in (("db1", 1), ("db3", 3), ("db6", 6)):
        p = leader_pyramid(x, wavelet=wavelet, j_min=3)
        assert p.n_vanishing_moments == n_psi
        assert p.boundary_margin == n_psi + 1
        assert np.all(p.n_border_dropped == 2 * (n_psi + 1))

    explicit = leader_pyramid(x, j_min=3, boundary_margin=10)
    assert np.all(explicit.n_border_dropped == 20)
    assert int(explicit.counts[0]) == (x.size >> 3) - 20


def test_exact_zero_leaders_are_dropped_and_recorded() -> None:
    """A half-zero signal has exactly-zero Haar leaders; they must be counted, not hidden."""
    x = np.concatenate([fbm(1 << 15, 0.7, seed=4), np.zeros(1 << 15)])
    p = leader_pyramid(x, wavelet="db1", j_min=2, j_max=8)
    assert int(p.n_zero_dropped.sum()) > 1000
    assert np.all(p.counts > 0)
    for v in p.values:
        assert np.all(v > 0.0)

    r = wavelet_leaders(x, wavelet="db1", j_min=2, j_max=8)
    assert r.provenance is not None
    assert r.provenance.diagnostics["n_zero_dropped_total"] == int(p.n_zero_dropped.sum())
    assert np.all(np.isfinite(r.zeta))


def test_l1_normalisation_is_the_one_that_makes_the_slope_hurst() -> None:
    r"""Regression guard on the :math:`2^{-j/2}` factor.

    With the :math:`L^2` coefficients PyWavelets returns, the slope of
    :math:`\log_2\langle|d|\rangle` against :math:`j` would be :math:`H+1/2`.
    The mean absolute leader must scale as :math:`2^{jH}`, not
    :math:`2^{j(H+1/2)}`.
    """
    H = 0.6
    p = leader_pyramid(fbm(N_FBM, H, seed=15), j_min=3)
    y = np.log2([float(v.mean()) for v in p.values])
    slope = float(np.polyfit(p.octaves.astype(float), y, 1)[0])
    assert slope == pytest.approx(H, abs=0.08)
    assert abs(slope - (H + 0.5)) > 0.3


def test_leader_recursion_matches_a_direct_brute_force_supremum() -> None:
    r"""The vectorised recursion against the definition, on a short signal.

    :math:`L(j,k)=\sup\{|d(j',k')| : \lambda(j',k')\subset 3\lambda(j,k),\,j'\le j\}`
    is here evaluated by explicit enumeration over the dyadic tree.
    """
    x = fbm(1 << 10, 0.6, seed=16)
    coef = leader_pyramid(x, j_min=1, j_max=5, min_leaders=3, boundary_margin=0, leaders=False)
    lead = leader_pyramid(x, j_min=1, j_max=5, min_leaders=3, boundary_margin=0, leaders=True)
    d = {int(j): v for j, v in zip(coef.octaves, coef.values, strict=True)}

    for idx, j in enumerate(lead.octaves.tolist()):
        n_j = d[j].size
        for k in (0, 1, n_j // 2, n_j - 2, n_j - 1):
            best = 0.0
            for jp in range(1, j + 1):
                span = 1 << (j - jp)
                lo = (k - 1) * span
                hi = (k + 2) * span
                sub = d[jp][max(lo, 0) : min(hi, d[jp].size)]
                if sub.size:
                    best = max(best, float(sub.max()))
            assert float(lead.values[idx][k]) == pytest.approx(best, rel=1e-12)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def test_short_series_and_bad_arguments_fail_loudly() -> None:
    with pytest.raises(ValueError, match="octave"):
        wavelet_leaders(fbm(256, 0.5, seed=1))
    with pytest.raises(ValueError, match="unknown wavelet"):
        wavelet_leaders(fbm(1 << 13, 0.5, seed=1), wavelet="not-a-wavelet")
    with pytest.raises(ValueError, match="constant"):
        wavelet_leaders(np.ones(1 << 13))
    with pytest.raises(ValueError, match="min_leaders"):
        wavelet_leaders(fbm(1 << 13, 0.5, seed=1), min_leaders=2)
    with pytest.raises(ValueError, match="j_min"):
        wavelet_leaders(fbm(1 << 13, 0.5, seed=1), j_min=0)
    with pytest.raises(ValueError, match="boundary_margin"):
        wavelet_leaders(fbm(1 << 13, 0.5, seed=1), boundary_margin=-1)
    with pytest.raises(ValueError, match="order"):
        leader_cumulants(fbm(1 << 13, 0.5, seed=1), order=0)
    with pytest.raises(ValueError, match="non-finite"):
        wavelet_leaders(np.r_[fbm(1 << 13, 0.5, seed=1), np.nan])


def test_cumulant_order_is_honoured() -> None:
    x = fbm(1 << 15, 0.6, seed=18)
    for order in (1, 2, 3, 4):
        lc = leader_cumulants(x, order=order)
        assert lc.c.size == order and lc.order == order
        assert lc.stderr is not None and lc.stderr.size == order
    assert leader_cumulants(x, order=1).c1 == pytest.approx(leader_cumulants(x, order=3).c1)
    assert np.isnan(leader_cumulants(x, order=1).c2)


def test_cumulant_parabolic_spectrum_agrees_with_the_legendre_apex() -> None:
    signal = binomial_signal()
    lc = leader_cumulants(signal, j_min=J_MIN_CASCADE, d_support=1.0)
    r = wavelet_leaders(signal, j_min=J_MIN_CASCADE)
    apex = spectrum_from_exponents(r).alpha_0
    assert lc.c1 == pytest.approx(apex, abs=0.05)

    alpha = np.linspace(lc.c1 - 0.1, lc.c1 + 0.1, 11)
    f = lc.parabolic_spectrum(alpha)
    assert float(np.max(f)) == pytest.approx(1.0, abs=1e-12)
    assert np.argmax(f) == 5
