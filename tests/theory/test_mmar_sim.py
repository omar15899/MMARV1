r"""Validation of the MMAR simulator against the closed-form theory.

The load-bearing test in this file is
:func:`test_composition_creates_multifractality`. The MMAR is
:math:`X=B_H\circ\theta`; a simulator that draws :math:`\theta` and
:math:`B_H` and then reports :math:`B_H` --- plotting it against a
trading-time abscissa, say --- returns a plain fBm. That output is exactly
monofractal (:math:`h(q)\equiv H`, Gaussian increments) and looks entirely
plausible on a chart, so only a scaling test catches it. Everything here is
seeded; nothing touches the global numpy state.

Statistic used
--------------
For a path :math:`x` and integer lags :math:`\ell`,
:math:`S_q(\ell)=\langle|x_{i+\ell}-x_i|^q\rangle \sim \ell^{\zeta(q)}` and
:math:`h(q)=\zeta(q)/q`. The MMAR predicts
:math:`h(q)=H\lambda-\tfrac{\sigma^2\ln b}{2}H^2q`, affine and strictly
decreasing; fBm predicts :math:`h(q)\equiv H`. The discriminant is therefore
:math:`h(2)-h(4)`, whose theoretical value under the conservation closure is
:math:`2(\lambda-1)H^2`.

Lags start at 32 cells deliberately. Below that the linear interpolation of
the fBm onto the trading-time grid contributes an error of order
:math:`(\Delta u)^H` that is comparable with the increment itself (see the
*Discretisation bias* section of :mod:`mmar.models.mmar`).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from mmar.core.regression import loglog_fit
from mmar.core.rng import spawn_rngs
from mmar.core.types import FloatArray
from mmar.models.fbm import fbm
from mmar.models.mmar import (
    MMARPath,
    MMARSpec,
    calibrate_mmar,
    mmar_forecast_distribution,
    mmar_zeta_lognormal,
    simulate_ensemble,
    simulate_mmar,
)

pytestmark = pytest.mark.stochastic

H_TRUE = 0.55
ALPHA_0 = 0.65
"""Chosen so that lambda = alpha_0/H = 1.1818 > 1, i.e. sigma^2 > 0."""

KMAX_BIG = 14
KMAX_SMALL = 10
N_REPLICAS = 40
OVERSAMPLE_BIG = 16
OVERSAMPLE_SMALL = 16
SEED = 20260727

LAGS = np.array([32, 64, 128, 256, 512, 1024, 2048], dtype=np.int64)
Q_GRID = (1.0, 2.0, 4.0)


def _spec(kmax: int = KMAX_BIG) -> MMARSpec:
    lam, _ = calibrate_mmar(ALPHA_0, H_TRUE)
    return MMARSpec(kmax=kmax, H=H_TRUE, lam=lam)


def generalised_hurst(
    x: FloatArray, q_values: Sequence[float], lags: NDArray[np.int64]
) -> FloatArray:
    r"""Sample :math:`h(q)=\zeta(q)/q` from overlapping-increment structure functions.

    Overlapping increments are used because they cut the variance of
    :math:`\hat S_q` by roughly the lag without biasing it; the price is a
    correlated regression design, which inflates the nominal standard error of
    the slope. Only the point estimate is used here.
    """
    out = []
    for q in q_values:
        s = np.array(
            [float(np.mean(np.abs(x[int(lag) :] - x[: -int(lag)]) ** q)) for lag in lags],
            dtype=np.float64,
        )
        out.append(loglog_fit(lags.astype(np.float64), s).slope / q)
    return np.asarray(out, dtype=np.float64)


def kurtosis(z: FloatArray) -> float:
    """Non-excess kurtosis; exactly 3 for a Gaussian, hence for fGn."""
    c = z - float(np.mean(z))
    return float(np.mean(c**4) / np.mean(c**2) ** 2)


@dataclass(frozen=True)
class ReplicaStats:
    """Per-replica statistics for the composed MMAR and the uncomposed fBm null."""

    h_composed: FloatArray
    h_legacy: FloatArray
    kurt_composed: FloatArray
    kurt_legacy: FloatArray

    @property
    def gap_composed(self) -> FloatArray:
        return np.asarray(self.h_composed[:, 1] - self.h_composed[:, 2], dtype=np.float64)

    @property
    def gap_legacy(self) -> FloatArray:
        return np.asarray(self.h_legacy[:, 1] - self.h_legacy[:, 2], dtype=np.float64)


@pytest.fixture(scope="module")
def replicas() -> ReplicaStats:
    r"""``N_REPLICAS`` paired draws: the MMAR, and the bug it must not be.

    The "legacy" arm is the un-composed path --- an fBm of the same length,
    same :math:`H`, same parent seed tree --- i.e. precisely what the historic
    implementation produced when it built a trading time, built an fBm, and
    then exponentiated the fBm without ever evaluating it at
    :math:`\theta(t)`.
    """
    spec = _spec(KMAX_BIG)
    streams = spawn_rngs(SEED, 2 * N_REPLICAS)
    hc, hl, kc, kl = [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for i in range(N_REPLICAS):
            path = simulate_mmar(spec, seed=streams[2 * i], oversample=OVERSAMPLE_BIG)
            legacy = np.concatenate(
                (np.zeros(1), fbm(spec.n, spec.H, seed=streams[2 * i + 1], t_max=1.0))
            )
            hc.append(generalised_hurst(path.x, Q_GRID, LAGS))
            hl.append(generalised_hurst(legacy, Q_GRID, LAGS))
            kc.append(kurtosis(path.log_returns))
            kl.append(kurtosis(np.diff(legacy)))
    return ReplicaStats(
        h_composed=np.asarray(hc, dtype=np.float64),
        h_legacy=np.asarray(hl, dtype=np.float64),
        kurt_composed=np.asarray(kc, dtype=np.float64),
        kurt_legacy=np.asarray(kl, dtype=np.float64),
    )


# --------------------------------------------------------------------------- #
# Calibration identities
# --------------------------------------------------------------------------- #


@pytest.mark.theory
@pytest.mark.parametrize("H", [0.3, 0.5, 0.55, 0.8])
@pytest.mark.parametrize("lam", [1.05, 1.2, 1.5])
def test_calibrate_roundtrips(H: float, lam: float) -> None:
    r""":math:`\alpha_0 = H\lambda \Rightarrow \lambda` recovered exactly."""
    lam_hat, sigma2 = calibrate_mmar(H * lam, H)
    assert lam_hat == pytest.approx(lam, rel=1e-12)
    assert sigma2 == pytest.approx(2.0 * (lam - 1.0) / math.log(2.0), rel=1e-12)
    assert sigma2 > 0.0


@pytest.mark.theory
def test_calibrate_conservation_closure_matches_base() -> None:
    r""":math:`\sigma^2=2(\lambda-1)/\ln b` scales with the base, as :math:`1/\ln b`."""
    _, s2_b2 = calibrate_mmar(0.65, 0.5, b=2)
    _, s2_b4 = calibrate_mmar(0.65, 0.5, b=4)
    assert s2_b2 / s2_b4 == pytest.approx(math.log(4.0) / math.log(2.0), rel=1e-12)


def test_calibrate_raises_with_a_diagnosis() -> None:
    """alpha_0 <= H is a modelling contradiction, and the message must say why."""
    with pytest.raises(ValueError) as exc:
        calibrate_mmar(0.40, 0.55)
    msg = str(exc.value)
    assert "alpha_0" in msg and "lambda" in msg and "sigma^2" in msg
    # The numbers that make the contradiction checkable must be in the message.
    assert "0.4" in msg and "0.55" in msg
    assert "0.727" in msg  # lambda = 0.40 / 0.55
    assert "lambda > 1" in msg


def test_calibrate_boundary_alpha0_equals_hurst_raises() -> None:
    r""":math:`\lambda=1` exactly is the monofractal boundary, not a valid MMAR."""
    with pytest.raises(ValueError, match="alpha_0 <= H"):
        calibrate_mmar(0.55, 0.55)


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan")])
def test_calibrate_rejects_nonpositive_alpha0(bad: float) -> None:
    with pytest.raises(ValueError, match="alpha_0 must be"):
        calibrate_mmar(bad, 0.5)


@pytest.mark.parametrize("bad_H", [0.0, 1.0, 1.5, -0.2])
def test_calibrate_rejects_hurst_outside_unit_interval(bad_H: float) -> None:
    with pytest.raises(ValueError, match="H must"):
        calibrate_mmar(0.6, bad_H)


def test_calibrate_warns_on_degenerate_cascade() -> None:
    r""":math:`\lambda\ge2` violates Kahane--Peyriere non-degeneracy."""
    with pytest.warns(UserWarning, match="Kahane"):
        lam, _ = calibrate_mmar(1.2, 0.5)
    assert lam == pytest.approx(2.4)


# --------------------------------------------------------------------------- #
# The specification and its closed-form exponents
# --------------------------------------------------------------------------- #


def independent_tau_x(q: FloatArray, *, H: float, lam: float, sigma2: float, b: int) -> FloatArray:
    r"""The MFC parabola, written out here and derived nowhere else in this file.

    :math:`\tau_X(q)=\tau_\theta(qH)` with
    :math:`\tau_\theta(p)=p\lambda-\tfrac{\sigma^2\ln b}{2}p^2-1`. Hard-coded on
    purpose: asserting ``zeta == tau + 1`` against an implementation that
    *computes* ``zeta`` as ``tau + 1`` cannot fail, and would pass just as well
    with the wrong log base or a dropped :math:`-1`.
    """
    p = np.asarray(q, dtype=np.float64) * H
    return np.asarray(p * lam - 0.5 * sigma2 * math.log(b) * p**2 - 1.0, dtype=np.float64)


@pytest.mark.theory
@pytest.mark.parametrize("b", [2, 3, 5])
@pytest.mark.parametrize("H", [0.3, 0.55, 0.8])
def test_closed_form_matches_an_independently_written_parabola(b: int, H: float) -> None:
    r"""Every sign, the log base and the :math:`-1` offset, against fresh algebra."""
    lam, sigma2 = calibrate_mmar(1.2 * H, H, b=b)
    spec = MMARSpec(kmax=8, H=H, lam=lam, b=b)
    q = np.linspace(-3.0, 5.0, 33)
    np.testing.assert_allclose(
        np.asarray(spec.theoretical_tau(q)),
        independent_tau_x(q, H=H, lam=lam, sigma2=sigma2, b=b),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        np.asarray(spec.theoretical_zeta(q)),
        independent_tau_x(q, H=H, lam=lam, sigma2=sigma2, b=b) + 1.0,
        atol=1e-13,
    )
    # Conservation is what pins the offset: zeta_X(1/H) = tau_theta(1) + 1 = 1.
    assert float(np.atleast_1d(spec.theoretical_zeta(1.0 / H))[0]) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.theory
def test_theoretical_tau_anchors() -> None:
    r""":math:`\tau_X(0)=-d` and :math:`\tau_X(1/H)=0`; :math:`\zeta=\tau+d`."""
    spec = _spec()
    tau, zeta = spec.theoretical_tau, spec.theoretical_zeta
    assert float(np.atleast_1d(tau(0.0))[0]) == pytest.approx(-1.0, abs=1e-12)
    assert float(np.atleast_1d(tau(1.0 / spec.H))[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(np.atleast_1d(zeta(0.0))[0]) == pytest.approx(0.0, abs=1e-12)
    q = np.linspace(-3.0, 5.0, 33)
    np.testing.assert_allclose(np.asarray(zeta(q)), np.asarray(tau(q)) + 1.0, atol=1e-12)


@pytest.mark.theory
def test_theoretical_hq_is_affine_and_decreasing() -> None:
    r""":math:`h(q)=H\lambda-\tfrac{\sigma^2\ln b}{2}H^2q`: the multifractal signature."""
    spec = _spec()
    q = np.array([1.0, 2.0, 3.0, 4.0])
    h = np.asarray(spec.theoretical_zeta(q)) / q
    slope = -0.5 * spec.sigma2_effective * math.log(spec.b) * spec.H**2
    np.testing.assert_allclose(np.diff(h), slope, atol=1e-12)
    assert slope < 0.0
    assert float(h[0]) - float(h[-1]) == pytest.approx(-3.0 * slope, rel=1e-12)


@pytest.mark.theory
def test_alpha_0_is_the_apex_of_the_spectrum() -> None:
    r""":math:`\alpha_0=\tau_X'(0)=H\lambda`, checked by central difference."""
    spec = _spec()
    tau = spec.theoretical_tau
    eps = 1e-6
    slope = (float(np.atleast_1d(tau(eps))[0]) - float(np.atleast_1d(tau(-eps))[0])) / (2 * eps)
    assert slope == pytest.approx(spec.alpha_0, abs=1e-8)
    assert spec.alpha_0 == pytest.approx(ALPHA_0, rel=1e-12)


@pytest.mark.theory
def test_zeta_helper_agrees_with_the_spec_binding() -> None:
    spec = _spec()
    q = np.linspace(0.5, 4.0, 15)
    direct = mmar_zeta_lognormal(q, H=spec.H, lam=spec.lam, sigma2=spec.sigma2_effective, b=spec.b)
    np.testing.assert_allclose(direct, np.asarray(spec.theoretical_zeta(q)), atol=1e-14)


@pytest.mark.theory
def test_critical_and_linearisation_moments_are_ordered() -> None:
    r"""Sample linearisation bites long before the moments actually diverge."""
    spec = _spec()
    assert 1.0 < spec.linearisation_moment < spec.critical_moment
    # f_X(alpha(q_lin)) = 0 by construction.
    q = spec.linearisation_moment
    tau = float(np.atleast_1d(spec.theoretical_tau(q))[0])
    eps = 1e-6
    alpha = (
        float(np.atleast_1d(spec.theoretical_tau(q + eps))[0])
        - float(np.atleast_1d(spec.theoretical_tau(q - eps))[0])
    ) / (2 * eps)
    assert q * alpha - tau == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kmax": 0}, "kmax"),
        ({"b": 1}, "base"),
        ({"H": 0.0}, "H must"),
        ({"H": 1.0}, "H must"),
        ({"price0": 0.0}, "price0"),
        ({"price0": -3.0}, "price0"),
        ({"lam": 1.0}, "lambda > 1"),
        ({"lam": 0.5}, "lambda > 1"),
    ],
)
def test_spec_validation_rejects(kwargs: dict[str, float], match: str) -> None:
    base = {"kmax": 8, "H": 0.55, "lam": 1.2}
    with pytest.raises(ValueError, match=match):
        MMARSpec(**{**base, **kwargs}).validate()  # type: ignore[arg-type]


def test_spec_warns_when_sigma2_breaks_conservation() -> None:
    r"""An explicit :math:`\sigma^2` off the closure kills :math:`\tau_\theta(1)=0`."""
    with pytest.warns(UserWarning, match="conservation closure"):
        MMARSpec(kmax=8, H=0.55, lam=1.2, sigma2=0.9).validate()


def test_spec_warns_when_alpha0_exceeds_one() -> None:
    with pytest.warns(UserWarning, match="alpha_0"):
        MMARSpec(kmax=8, H=0.9, lam=1.3).validate()


def test_spec_n_is_b_to_the_kmax() -> None:
    assert MMARSpec(kmax=10, H=0.5, lam=1.2).n == 1024
    assert MMARSpec(kmax=5, H=0.5, lam=1.2, b=3).n == 243


# --------------------------------------------------------------------------- #
# Structural invariants of a simulated path
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def one_path() -> MMARPath:
    return simulate_mmar(_spec(KMAX_SMALL), seed=4242, oversample=OVERSAMPLE_SMALL)


def test_theta_is_a_clock(one_path: MMARPath) -> None:
    r""":math:`\theta` is non-decreasing, starts at 0, ends at :math:`T`."""
    theta = one_path.theta
    assert theta[0] == 0.0
    assert theta[-1] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(theta) >= 0.0)
    # Singular, not Lebesgue: the cell masses must be wildly unequal.
    d = np.diff(theta)
    assert float(d.max() / d.min()) > 100.0


def test_array_shapes_and_price_definition(one_path: MMARPath) -> None:
    n = one_path.spec.n
    assert one_path.t.shape == (n + 1,)
    assert one_path.theta.shape == (n + 1,)
    assert one_path.x.shape == (n + 1,)
    assert one_path.price.shape == (n + 1,)
    assert one_path.log_returns.shape == (n,)
    np.testing.assert_allclose(one_path.t, np.linspace(0.0, 1.0, n + 1), atol=1e-15)
    np.testing.assert_allclose(
        one_path.price, one_path.spec.price0 * np.exp(one_path.x), rtol=1e-13
    )
    assert one_path.x[0] == 0.0
    assert one_path.price[0] == pytest.approx(one_path.spec.price0)
    assert np.all(np.isfinite(one_path.x))


def test_price0_scales_the_path_multiplicatively() -> None:
    """P(0) is a pure scale: it must not perturb the random stream."""
    base = _spec(KMAX_SMALL)
    scaled = MMARSpec(kmax=base.kmax, H=base.H, lam=base.lam, price0=137.0)
    a = simulate_mmar(base, seed=8, oversample=OVERSAMPLE_SMALL)
    b = simulate_mmar(scaled, seed=8, oversample=OVERSAMPLE_SMALL)
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_allclose(b.price, 137.0 * a.price, rtol=1e-13)


def test_at_interpolates_and_refuses_extrapolation(one_path: MMARPath) -> None:
    assert float(one_path.at(0.0)) == pytest.approx(one_path.spec.price0)
    assert float(one_path.at(1.0)) == pytest.approx(float(one_path.price[-1]))
    mid = float(one_path.at(0.5))
    assert min(one_path.price) <= mid <= max(one_path.price)
    with pytest.raises(ValueError, match="outside the simulated span"):
        one_path.at(1.5)


def test_oversample_must_be_a_positive_integer() -> None:
    with pytest.raises(ValueError, match="oversample"):
        simulate_mmar(_spec(KMAX_SMALL), seed=1, oversample=0)


def test_coarse_fbm_grid_warns_about_chord_interpolation() -> None:
    """The bias diagnostic must fire when the fBm grid cannot resolve the cells."""
    with pytest.warns(UserWarning, match="linear chords"):
        simulate_mmar(_spec(12), seed=2, oversample=1)


@pytest.mark.theory
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("b", [3, 5])
def test_non_dyadic_base_simulates_and_keeps_its_exponents(b: int) -> None:
    r"""Nothing in the pipeline may be secretly hard-wired to :math:`b=2`.

    The whole chain --- conservation closure, cascade sampler, trading time,
    subordination --- is parameterised by the base, and every one of those
    steps has a :math:`\ln b` in it. A formula that is right for
    :math:`b=2` and wrong in general survives every other test in this file.
    """
    lam, sigma2 = calibrate_mmar(ALPHA_0, H_TRUE, b=b)
    assert sigma2 == pytest.approx(2.0 * (lam - 1.0) / math.log(b), rel=1e-12)
    spec = MMARSpec(kmax=9 if b == 3 else 6, H=H_TRUE, lam=lam, b=b)
    path = simulate_mmar(spec, seed=4, oversample=8)

    assert path.theta.shape == (spec.n + 1,)
    assert path.theta[0] == 0.0
    assert float(path.theta[-1]) == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(path.theta) >= 0.0)
    assert np.all(np.isfinite(path.x)) and path.x[0] == 0.0
    assert spec.alpha_0 == pytest.approx(ALPHA_0, rel=1e-12)
    assert float(np.atleast_1d(spec.theoretical_tau(1.0 / H_TRUE))[0]) == pytest.approx(
        0.0, abs=1e-12
    )
    # The realised measure must scale with the base it was built with: at p = 1
    # the partition function is conservation, tau_theta(1) = 0, exactly.
    mu = np.diff(path.theta)
    levels = [mu]
    while levels[-1].size > 1:
        levels.append(levels[-1].reshape(-1, b).sum(axis=1))
    np.testing.assert_allclose([float(c.sum()) for c in levels], 1.0, atol=1e-12)


def test_conservative_cascade_warns_that_it_breaks_the_closed_form() -> None:
    """Microcanonical rescaling changes tau_theta; the spec must not stay silent."""
    with pytest.warns(UserWarning, match="MICROCANONICAL"):
        MMARSpec(kmax=8, H=0.55, lam=1.2, conservative=True).validate()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        MMARSpec(kmax=8, H=0.55, lam=1.2, conservative=False).validate()


def test_provenance_records_the_composition() -> None:
    path = simulate_mmar(_spec(KMAX_SMALL), seed=17, oversample=OVERSAMPLE_SMALL)
    prov = path.provenance
    assert prov.estimator == "mmar.simulate"
    assert prov.seed is not None and prov.seed.entropy == 17
    assert prov.params["oversample"] == OVERSAMPLE_SMALL
    assert prov.params["spec"]["kmax"] == KMAX_SMALL
    assert {ref.name for ref in prov.inputs} == {"trading_time", "fbm_grid"}
    assert any("B_H(theta(t))" in n for n in prov.notes)
    assert 0.0 <= prov.diagnostics["chord_fraction"] <= 1.0
    assert prov.diagnostics["trading_time_total"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Reproducibility and independence
# --------------------------------------------------------------------------- #


def test_reproducible_from_seed_bitwise() -> None:
    spec = _spec(KMAX_SMALL)
    a = simulate_mmar(spec, seed=99, oversample=OVERSAMPLE_SMALL)
    b = simulate_mmar(spec, seed=99, oversample=OVERSAMPLE_SMALL)
    for name in ("t", "theta", "x", "price"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.provenance.root == b.provenance.root


def test_different_seeds_give_different_paths() -> None:
    spec = _spec(KMAX_SMALL)
    a = simulate_mmar(spec, seed=1, oversample=OVERSAMPLE_SMALL)
    b = simulate_mmar(spec, seed=2, oversample=OVERSAMPLE_SMALL)
    assert not np.array_equal(a.x, b.x)
    assert not np.array_equal(a.theta, b.theta)
    assert a.provenance.root != b.provenance.root


def test_ensemble_is_reproducible_and_indexed() -> None:
    spec = _spec(KMAX_SMALL)
    a = simulate_ensemble(spec, 5, seed=7, oversample=OVERSAMPLE_SMALL)
    b = simulate_ensemble(spec, 5, seed=7, oversample=OVERSAMPLE_SMALL)
    for pa, pb in zip(a, b, strict=True):
        np.testing.assert_array_equal(pa.x, pb.x)
        assert pa.provenance.root == pb.provenance.root
    # Distinct paths must carry distinct content addresses.
    assert len({p.provenance.root for p in a}) == 5
    assert [p.provenance.params["path_index"] for p in a] == [0, 1, 2, 3, 4]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_ensemble_paths_are_independent() -> None:
    r"""Pairwise correlation of the return series and of terminal values is ~0.

    This checks independence; it does **not** discriminate spawning from
    seed-incrementing, and an earlier version of this docstring claimed it did.
    Measured over 40 paths at ``kmax=10``: spawned sub-streams give mean
    :math:`\rho=-3.5\times10^{-4}` (median :math:`|\rho|` 0.0198) and
    consecutive integer seeds 1000..1039 give :math:`-9\times10^{-5}` (0.0187) ---
    indistinguishable, because :class:`numpy.random.SeedSequence` entropy-hashes
    an integer seed before it ever reaches PCG64. Spawning is still the right
    construction (non-overlap is *guaranteed*, not merely observed), but this
    statistic is not the evidence for it.
    """
    spec = _spec(KMAX_SMALL)
    n_paths = 40
    paths = simulate_ensemble(spec, n_paths, seed=101, oversample=OVERSAMPLE_SMALL)
    returns = np.array([p.log_returns for p in paths])
    corr = np.corrcoef(returns)[np.triu_indices(n_paths, 1)]
    # The mean over ~800 pairs is tight even though individual pairs are noisy:
    # MMAR returns are so heavy-tailed that a single sample correlation is
    # dominated by a handful of increments.
    assert abs(float(np.mean(corr))) < 0.01
    assert float(np.median(np.abs(corr))) < 0.05

    # Terminal values of two independently seeded ensembles.
    a = np.array([p.x[-1] for p in simulate_ensemble(spec, 200, seed=202, oversample=16)])
    b = np.array([p.x[-1] for p in simulate_ensemble(spec, 200, seed=303, oversample=16)])
    rho = float(np.corrcoef(a, b)[0, 1])
    assert abs(rho) < 3.0 / math.sqrt(200)
    assert len(np.unique(a)) == 200


def test_cascade_and_fbm_streams_are_distinct() -> None:
    r"""Independence of :math:`\theta` and :math:`B_H` is a model assumption.

    If both were drawn from one stream, changing ``oversample`` (which only
    changes how many fBm variates are consumed) would leave the trading time
    untouched only by luck. Here it must be exactly untouched.
    """
    spec = _spec(KMAX_SMALL)
    a = simulate_mmar(spec, seed=55, oversample=8)
    b = simulate_mmar(spec, seed=55, oversample=32)
    np.testing.assert_array_equal(a.theta, b.theta)
    assert not np.array_equal(a.x, b.x)


# --------------------------------------------------------------------------- #
# THE regression test: the composition is the model
# --------------------------------------------------------------------------- #


@pytest.mark.theory
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_x_is_literally_the_fbm_evaluated_at_theta() -> None:
    r"""Reconstruct :math:`B_H\circ\theta` from the seed and demand equality.

    The statistical tests below establish that the output *behaves* like a
    subordinated process. This one asserts the identity itself: with the same
    parent seed, the same two spawned streams and the same grid, the returned
    ``x`` must equal :math:`B_H` interpolated at :math:`\theta(t_i)` to machine
    precision. Any reversion to ``x = B_H(t)`` fails here immediately, with no
    appeal to sampling variability.
    """
    spec = _spec(KMAX_SMALL)
    oversample = 8
    path = simulate_mmar(spec, seed=2718, oversample=oversample)

    rng_cascade, rng_fbm = spawn_rngs(2718, 2)
    del rng_cascade  # consumed by the cascade; theta is taken from the path itself
    n_steps = oversample * spec.n
    grid = np.linspace(0.0, 1.0, n_steps + 1)
    b_path = np.concatenate((np.zeros(1), fbm(n_steps, spec.H, seed=rng_fbm, t_max=1.0)))
    expected = np.interp(path.theta, grid, b_path)

    np.testing.assert_allclose(path.x, expected, atol=1e-15, rtol=0.0)
    # And it is emphatically *not* the un-composed fBm on the physical grid.
    uncomposed = np.interp(path.t, grid, b_path)
    assert float(np.max(np.abs(path.x - uncomposed))) > 1e-3


@pytest.mark.theory
@pytest.mark.slow
def test_composition_creates_multifractality(replicas: ReplicaStats) -> None:
    r"""The composed path is multifractal; the un-composed fBm is not.

    Guards the historic bug in which the simulator built :math:`\theta` and
    :math:`B_H` and returned :math:`P(0)e^{B_H(t)}`, never evaluating
    :math:`B_H` at :math:`\theta(t)`. Two independent discriminants, each
    separating the two arms completely over ``N_REPLICAS`` seeded draws:

    1. :math:`h(2)-h(4)`, zero for a monofractal, :math:`2(\lambda-1)H^2` for
       the MMAR;
    2. the kurtosis of the increments, exactly 3 for fGn.
    """
    spec = _spec(KMAX_BIG)
    predicted_gap = 2.0 * (spec.lam - 1.0) * spec.H**2

    gap_c, gap_l = replicas.gap_composed, replicas.gap_legacy
    med_c, med_l = float(np.median(gap_c)), float(np.median(gap_l))

    # 1. h(q) is non-constant for the composition, and by about the right amount.
    assert med_c == pytest.approx(predicted_gap, rel=0.5), (
        f"h(2)-h(4) = {med_c:.4f}, theory {predicted_gap:.4f}"
    )
    assert 0.5 * predicted_gap < med_c < 1.5 * predicted_gap

    # 2. The un-composed fBm is monofractal: h(q) flat to within the
    #    finite-sample bias of the fourth-order structure function.
    assert abs(med_l) < 0.03
    h_legacy = np.median(replicas.h_legacy, axis=0)
    assert float(h_legacy.max() - h_legacy.min()) < 0.02
    np.testing.assert_allclose(h_legacy, H_TRUE, atol=0.05)

    # 3. Complete, non-parametric separation of the two arms.
    p_gap = float(stats.mannwhitneyu(gap_c, gap_l, alternative="greater").pvalue)
    assert p_gap < 1e-8, f"h(2)-h(4) does not separate the arms (p={p_gap:.3g})"
    assert gap_c.min() > gap_l.max()

    # 4. Leptokurtosis. fGn increments are exactly Gaussian; the subordinated
    #    increments are a variance mixture with E[(dtheta)^{4H}]/E[(dtheta)^{2H}]^2
    #    times the Gaussian kurtosis.
    kc, kl = replicas.kurt_composed, replicas.kurt_legacy
    assert float(np.median(kl)) == pytest.approx(3.0, abs=0.2)
    assert float(np.median(kc)) > 10.0
    assert kc.min() > 3.0 * kl.max()
    p_kurt = float(stats.mannwhitneyu(kc, kl, alternative="greater").pvalue)
    assert p_kurt < 1e-8


@pytest.mark.theory
@pytest.mark.slow
def test_composed_hq_tracks_the_closed_form(replicas: ReplicaStats) -> None:
    r"""Ensemble-median :math:`\hat h(q)` matches :math:`\zeta_X(q)/q`.

    Tolerance is loose on purpose. Two known biases push in opposite
    directions: chord interpolation of the small cells flattens the estimate
    upward, and single-sample linearisation past
    :math:`q_{\rm lin}=1/(H\sqrt{\sigma^2\ln b/2})` pushes the large-``q``
    estimate down. ``q=4`` here sits at 94% of :math:`q_{\rm lin}`.
    """
    spec = _spec(KMAX_BIG)
    q = np.asarray(Q_GRID, dtype=np.float64)
    predicted = np.asarray(spec.theoretical_zeta(q)) / q
    observed = np.median(replicas.h_composed, axis=0)
    np.testing.assert_allclose(observed, predicted, atol=0.05)
    assert np.all(np.diff(observed) < 0.0), "h(q) must decrease in q"


@pytest.mark.theory
def test_degenerate_cascade_reduces_to_fbm() -> None:
    r""":math:`\sigma^2=0,\ \lambda=1`: trading time is Lebesgue and :math:`X=B_H`.

    The monofractal limit of the model. It also proves the subordination is
    wired the right way round: with :math:`\theta(t)=t` the composed output
    must be the fBm itself, not a reshuffling of it.
    """
    spec = MMARSpec(kmax=12, H=0.55, lam=1.0, sigma2=0.0)
    spec.validate()
    path = simulate_mmar(spec, seed=31, oversample=4)
    np.testing.assert_allclose(np.diff(path.theta), 1.0 / spec.n, rtol=1e-12)
    np.testing.assert_allclose(path.theta, path.t, atol=1e-12)
    h = generalised_hurst(path.x, (2.0, 4.0), np.array([16, 32, 64, 128, 256, 512]))
    assert abs(float(h[0] - h[1])) < 0.03
    assert float(np.atleast_1d(spec.theoretical_tau(1.0 / spec.H))[0]) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Forecast distribution
# --------------------------------------------------------------------------- #


def test_forecast_is_a_coherent_distribution() -> None:
    fc = mmar_forecast_distribution(
        _spec(KMAX_SMALL), 0.5, 400, seed=909, oversample=OVERSAMPLE_SMALL
    )
    assert fc.terminal_price.shape == (400,)
    assert np.all(fc.terminal_price > 0.0)
    assert np.all(np.diff(fc.quantiles) >= 0.0)
    assert fc.quantiles[0] < fc.median < fc.quantiles[-1]
    np.testing.assert_allclose(
        fc.terminal_log_return, np.log(fc.terminal_price / fc.price0), rtol=1e-13
    )
    assert fc.std > 0.0
    # Driftless log-price plus Jensen: E[P] > median P = P(0) * exp(median X) ~ P(0).
    assert fc.mean > fc.median
    assert fc.probability_below(fc.median) == pytest.approx(0.5, abs=0.01)
    assert fc.quantile(0.5) == pytest.approx(fc.median, rel=1e-12)
    assert math.isfinite(fc.tail_index_upper) and fc.tail_index_upper > 0.0
    assert math.isfinite(fc.tail_index_lower) and fc.tail_index_lower > 0.0
    assert fc.provenance.estimator == "mmar.forecast"
    assert fc.provenance.seed is not None and fc.provenance.seed.entropy == 909


def test_forecast_is_reproducible() -> None:
    spec = _spec(KMAX_SMALL)
    a = mmar_forecast_distribution(spec, 0.25, 100, seed=5, oversample=8)
    b = mmar_forecast_distribution(spec, 0.25, 100, seed=5, oversample=8)
    np.testing.assert_array_equal(a.terminal_price, b.terminal_price)
    assert a.provenance.root == b.provenance.root
    c = mmar_forecast_distribution(spec, 0.25, 100, seed=6, oversample=8)
    assert a.provenance.root != c.provenance.root


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_forecast_dispersion_grows_with_the_horizon() -> None:
    r"""Var of :math:`X(h)` is :math:`\mathbb E[\theta(h)^{2H}]`, increasing in ``h``."""
    spec = _spec(KMAX_SMALL)
    short = mmar_forecast_distribution(spec, 0.1, 300, seed=13, oversample=16)
    long = mmar_forecast_distribution(spec, 1.0, 300, seed=13, oversample=16)
    assert float(np.std(long.terminal_log_return)) > float(np.std(short.terminal_log_return))


@pytest.mark.parametrize("bad_horizon", [0.0, -0.1, 1.5])
def test_forecast_rejects_horizons_outside_the_span(bad_horizon: float) -> None:
    with pytest.raises(ValueError, match="horizon must lie"):
        mmar_forecast_distribution(_spec(8), bad_horizon, 10, seed=1, oversample=8)


@pytest.mark.parametrize(("n_paths", "match"), [(10, "NaN"), (20, "noise")])
def test_forecast_says_out_loud_when_the_hill_index_is_meaningless(
    n_paths: int, match: str
) -> None:
    """A NaN (or a two-point average) in a returned field must not travel silently.

    ``n_paths=10`` is below the estimator's sample floor, so both tail indices
    are NaN; ``n_paths=20`` clears the floor but asks for
    :math:`\\lceil 0.05\\times20\\rceil = 1` order statistic, which the clamp
    silently promotes to two. Either way the caller is told, rather than
    reading the NaN out of ``summary()`` later.
    """
    with pytest.warns(UserWarning, match=match):
        fc = mmar_forecast_distribution(_spec(8), 0.5, n_paths, seed=3, oversample=8)
    if n_paths < 20:
        assert math.isnan(fc.tail_index_upper) and math.isnan(fc.tail_index_lower)
        assert fc.provenance.diagnostics["hill_order_statistics"] == 0
    else:
        assert fc.provenance.diagnostics["hill_order_statistics"] == 2
    # Everything that does not depend on the tail estimate stays valid.
    assert np.all(np.isfinite(fc.quantiles)) and fc.std > 0.0


def test_forecast_hill_index_is_quiet_when_it_has_enough_order_statistics() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fc = mmar_forecast_distribution(_spec(8), 0.5, 200, seed=3, oversample=OVERSAMPLE_SMALL)
    assert not [w for w in caught if "Hill" in str(w.message) or "tail index" in str(w.message)]
    assert fc.provenance.diagnostics["hill_order_statistics"] == 10
    assert math.isfinite(fc.tail_index_upper)


def test_forecast_rejects_degenerate_ensembles() -> None:
    with pytest.raises(ValueError, match="at least 2 paths"):
        mmar_forecast_distribution(_spec(8), 0.5, 1, seed=1, oversample=8)
    with pytest.raises(ValueError, match="hill_fraction"):
        mmar_forecast_distribution(_spec(8), 0.5, 30, seed=1, oversample=8, hill_fraction=0.9)
    with pytest.raises(ValueError, match="quantile_levels"):
        mmar_forecast_distribution(_spec(8), 0.5, 30, seed=1, oversample=8, quantile_levels=[1.0])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_interior_horizons_are_leptokurtic_and_the_endpoint_is_not() -> None:
    r"""The terminal law is a variance mixture --- except exactly at :math:`h=T`.

    :math:`X(h)\mid\theta\sim\mathcal N(0,\theta(h)^{2H})`, so the
    unconditional terminal log-return is a scale mixture of normals and is
    leptokurtic in proportion to :math:`\operatorname{Var}[\theta(h)^{2H}]`.
    That variance is *zero* at :math:`h=T`, because trading time is normalised
    to :math:`\theta(T)=T` on every path: the endpoint law is exactly Gaussian.
    A simulator that reported a fat-tailed law at :math:`h=T` would be
    mis-normalising the clock; one that reported a Gaussian law at
    :math:`h<T` would not be subordinating at all.
    """
    spec = _spec(KMAX_SMALL)
    interior = mmar_forecast_distribution(spec, 0.25, 800, seed=77, oversample=16)
    endpoint = mmar_forecast_distribution(spec, 1.0, 800, seed=77, oversample=16)
    k_in = kurtosis(interior.terminal_log_return)
    k_end = kurtosis(endpoint.terminal_log_return)
    assert k_in > 3.4, f"interior horizon is not leptokurtic (kurtosis {k_in:.3f})"
    assert abs(k_end - 3.0) < 0.6, f"endpoint law should be Gaussian (kurtosis {k_end:.3f})"
    assert k_in > k_end + 0.5
