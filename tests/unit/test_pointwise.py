r"""Validation of :mod:`mmar.estimators.pointwise` against exact ground truth.

Every tolerance below is a *measured* quantity, not a number chosen until the
test went green. The observed errors are printed by ``pytest -s`` through
:func:`_report` and are reproduced verbatim in the estimators' ``Bias`` sections.

Ground truths used, in decreasing order of strength:

* **Cusp** :math:`x(t)=|t-t_0|^{a}`: :math:`h(t_0)=a` exactly, and
  :math:`h(t)=+\infty` elsewhere. The oscillation of a *centred* window at
  :math:`t_0` is :math:`(\delta/2)^a` in closed form, so this test admits an
  analytic check of the estimator, not merely of its output.
* **Jump**: :math:`h=0` exactly at the discontinuity.
* **fBm** (:mod:`mmar.models.fbm`, exact Davies--Harte): :math:`h(t)=H` for
  every :math:`t`, almost surely.
* **mBm** (Peltier & Lévy Véhel 1995): :math:`h(t)` is prescribed by
  construction; this is the only ground truth in the library with a genuinely
  *varying* pointwise exponent.
* **Binomial cascade trading time** (:math:`m_0=0.6`, :math:`k_{\max}=14`):
  the pointwise exponents fill :math:`[\alpha_{\min},\alpha_{\max}] =
  [0.7370, 1.3219]` with :math:`\alpha_0=1.0294`
  (:func:`mmar.models.theory.cascade_alpha_range`).

On the convention anchors. A pointwise estimator has no moment grid, so
:math:`\tau(0)=-1` and :math:`\tau(1)=0` do not apply to it. Its equivalents,
all tested here, are: a jump has :math:`h=0`; a :math:`C^1` signal has
:math:`h=1` under the oscillation (which annihilates only constants); and on fBm
the pointwise :math:`h` must equal the *same* :math:`H` that
:math:`\zeta(q)=qH` encodes, i.e. :math:`h = \zeta(q)/q` — the two views of the
same exponent must not drift apart.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mmar.core.types import FitMethod, PointwiseHolder
from mmar.estimators.pointwise import (
    RegularityMap,
    holder_at,
    leader_holder,
    local_regularity_map,
    oscillation_holder,
    wavelet_holder,
)
from mmar.models.cascade import deterministic_cascade, trading_time
from mmar.models.fbm import fbm, mbm
from mmar.models.theory import cascade_alpha_range

N = 2**14
SEED = 20260727

ESTIMATORS = {
    "oscillation": oscillation_holder,
    "wavelet": wavelet_holder,
    "leaders": leader_holder,
}


def _grid(n: int = N) -> np.ndarray:
    """The sample grid the estimators report, :math:`t_i=(i+1)/n`."""
    return (np.arange(n, dtype=np.float64) + 1.0) / n


def _cusp(a: float, *, t0: float = 0.5, n: int = N) -> np.ndarray:
    return np.abs(_grid(n) - t0) ** a


def _at(result: PointwiseHolder, t0: float) -> float:
    return float(result.h[int(np.argmin(np.abs(result.t - t0)))])


def _finite(result: PointwiseHolder) -> np.ndarray:
    return np.asarray(result.h[np.isfinite(result.h)], dtype=np.float64)


def _report(label: str, **values: float) -> None:
    """Print the observed number. The tolerance is a claim; this is the evidence."""
    body = "  ".join(f"{k}={v:+.4f}" for k, v in values.items())
    print(f"[pointwise] {label:<34s} {body}")


# --------------------------------------------------------------------------- #
# (a) Cusp: the definitive test, exact ground truth
# --------------------------------------------------------------------------- #


@pytest.mark.theory
@pytest.mark.parametrize("a", [0.3, 0.5, 0.7])
@pytest.mark.parametrize(
    ("name", "tol"),
    [
        # Analytic: osc(t0, delta) = (delta/2)**a exactly, so the only error is
        # grid discretisation of the cusp itself.
        ("oscillation", 0.02),
        # Curvature of log|T| vs log a over the eight octaves j in [3, 10].
        ("wavelet", 0.06),
        # log2 L(j, .) is convex in j; a line through it undershoots. Halves at
        # j_min=5, halves again at j_min=6 (see test_leader_bias_shrinks...).
        # Measured at j in [3, 10]: -0.2184 / -0.1752 / -0.1921.
        ("leaders", 0.23),
    ],
)
def test_cusp_exponent_recovered(name: str, tol: float, a: float) -> None:
    r"""``x(t) = |t - 1/2|**a`` has :math:`h(1/2)=a`; recover it."""
    x = _cusp(a)
    h = _at(ESTIMATORS[name](x), 0.5)
    _report(f"cusp a={a} {name}", h=h, err=h - a)
    assert abs(h - a) < tol, f"{name} recovered h={h:.4f} for a={a} (error {h - a:+.4f})"


@pytest.mark.theory
def test_oscillation_matches_the_closed_form_cusp_law() -> None:
    r"""The estimator must reproduce :math:`\operatorname{osc}(t_0,\delta)=(\delta/2)^a`.

    This checks the *surface*, not just the fitted slope: an estimator can get
    the exponent right from a wrong multiresolution quantity.
    """
    a = 0.5
    x = _cusp(a)
    grid = local_regularity_map(x, method="oscillation")
    row = grid.log_values[int(np.argmin(np.abs(grid.t - 0.5)))]
    predicted = a * np.log(grid.scales / 2.0) - a * math.log(N)
    assert np.allclose(row, predicted, atol=0.05)


@pytest.mark.theory
def test_leader_cusp_bias_shrinks_with_the_octave_window() -> None:
    """The leader bias is finite-size convexity, so it must fall as j_min rises."""
    errors = {}
    for j_min in (3, 5, 6):
        errs = [abs(_at(leader_holder(_cusp(a), j_min=j_min), 0.5) - a) for a in (0.3, 0.5, 0.7)]
        errors[j_min] = float(np.mean(errs))
    _report("leader |err| vs j_min", **{f"j{j}": v for j, v in errors.items()})
    assert errors[6] < errors[5] < errors[3]
    assert errors[6] < 0.08


@pytest.mark.theory
def test_leaders_without_fractional_integration_collapse_only_at_small_h() -> None:
    r"""``gamma=0`` is pinned by the discretisation transient — but only for small
    :math:`h`, which is why the sweep below is the test and a single :math:`a` is not.

    The leader is a supremum over all finer scales, so a discretisation transient
    at the two finest octaves — which behaves like a negative local exponent —
    can pin it. Measured at :math:`n=2^{14}`, :math:`j\in[3,10]`:

    ====  ==============  ================
    a     ``gamma=0``     ``gamma=0.5``
    ====  ==============  ================
    0.2   :math:`+0.000`  :math:`-0.094`
    0.3   :math:`+0.000`  :math:`+0.082`
    0.5   :math:`+0.245`  :math:`+0.325`
    0.7   :math:`+0.493`  :math:`+0.508`
    0.9   :math:`+0.674`  :math:`+0.674`
    ====  ==============  ================

    So the collapse is *not* "to zero regardless of the true exponent": it is a
    floor that bites when the true exponent is below roughly 0.4 and is
    irrelevant by :math:`a=0.9`. The honest statements are that ``gamma=0`` never
    exceeds ``gamma=0.5``, that the gap closes as :math:`a` grows, and that
    **neither** setting resolves :math:`a=0.2` at this length — where the default
    even returns a negative exponent, which no continuous function has.
    """
    table = {}
    for a in (0.2, 0.3, 0.5, 0.7, 0.9):
        x = _cusp(a)
        table[a] = (_at(leader_holder(x, gamma=0.0), 0.5), _at(leader_holder(x, gamma=0.5), 0.5))
    _report("leaders gamma sweep g=0", **{f"a{a}": v[0] for a, v in table.items()})
    _report("leaders gamma sweep g=1/2", **{f"a{a}": v[1] for a, v in table.items()})

    assert abs(table[0.3][0]) < 0.05, "expected the documented gamma=0 collapse at a=0.3"
    assert table[0.3][1] > 0.05, "gamma=1/2 must lift the a=0.3 cusp off the floor"
    # The pathology is exponent-dependent, not universal.
    assert table[0.7][0] > 0.40, "gamma=0 does NOT collapse at a=0.7; do not claim it does"
    # gamma=1/2 dominates above the resolution floor, and the two agree once the
    # discretisation floor stops binding. It does NOT dominate at a=0.2, where
    # gamma=0 clips at exactly 0 and gamma=1/2 goes negative -- asserted below.
    for a, (g0, ghalf) in table.items():
        if a >= 0.3:
            assert ghalf >= g0 - 1e-9, f"gamma=1/2 fell below gamma=0 at a={a}"
    assert table[0.9][1] == pytest.approx(table[0.9][0], abs=1e-9)
    # Honest ceiling on what this estimator can resolve at n=2**14.
    assert table[0.2][1] < 0.0, "a=0.2 is below the resolution floor; that must stay visible"


# --------------------------------------------------------------------------- #
# (b) / (c) Convention anchors: jump -> 0, C^1 -> 1, multifractal ground truth
# --------------------------------------------------------------------------- #


@pytest.mark.theory
@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_jump_has_zero_exponent(name: str) -> None:
    """A step discontinuity is the h = 0 anchor of the pointwise convention."""
    x = (_grid() >= 0.5).astype(np.float64)
    result = ESTIMATORS[name](x)
    h = _at(result, 0.5)
    _report(f"jump {name}", h=h)
    assert abs(h) < 0.05


@pytest.mark.theory
@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_degenerate_cells_are_counted_not_hidden(name: str) -> None:
    """Away from the jump the signal is constant: those cells must be recorded."""
    x = (_grid() >= 0.5).astype(np.float64)
    prov = ESTIMATORS[name](x).provenance
    assert prov is not None
    assert prov.diagnostics["n_nonpositive_cells"] > 0
    assert prov.diagnostics["n_undetermined"] > 0


@pytest.mark.theory
def test_c1_signal_saturates_the_oscillation_ceiling() -> None:
    r"""The oscillation annihilates constants only, so a smooth signal gives h=1.

    Not a defect: it is the documented ceiling. An estimator reporting
    :math:`h>1` from an oscillation would be reporting an artefact.
    """
    x = np.sin(2.0 * math.pi * 3.0 * _grid()) + 3.0
    h = float(np.mean(_finite(oscillation_holder(x))))
    _report("C1 sine oscillation ceiling", h=h)
    assert abs(h - 1.0) < 0.10


@pytest.mark.theory
@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_binomial_cascade_trading_time(name: str) -> None:
    r"""The pointwise field of a multifractal must sit inside the known support.

    Ground truth: :math:`\alpha\in[0.7370, 1.3219]`, :math:`\alpha_0=1.0294`.
    A pointwise estimator sees a *single* realisation of the exponent field on a
    finite grid, so its empirical range is wider than the asymptotic support
    (finite-scale coarse exponents scatter around their limits). The claims
    tested are therefore: the bulk is centred on :math:`\alpha_0`, and the field
    genuinely reaches both branches.
    """
    a_min, a_max = cascade_alpha_range([0.6, 0.4], 2)
    a_0 = 0.5 * (a_min + a_max)
    x = trading_time(deterministic_cascade(14, [0.6, 0.4]))[:-1]
    h = _finite(ESTIMATORS[name](x))
    mean = float(h.mean())
    _report(
        f"cascade {name}",
        mean=mean,
        err=mean - a_0,
        q02=float(np.quantile(h, 0.02)),
        q98=float(np.quantile(h, 0.98)),
    )
    assert abs(mean - a_0) < 0.15, f"{name} centred at {mean:.4f}, alpha_0={a_0:.4f}"
    # Both branches of the spectrum are populated -- see the D2 regression below.
    assert float(np.quantile(h, 0.02)) < a_0 - 0.15
    assert float(np.quantile(h, 0.98)) > a_0 + 0.15


# --------------------------------------------------------------------------- #
# (a') Monofractal recovery on exact fBm
# --------------------------------------------------------------------------- #


@pytest.mark.theory
@pytest.mark.stochastic
@pytest.mark.parametrize("H", [0.3, 0.5, 0.7])
@pytest.mark.parametrize(
    ("name", "lo", "hi"),
    [
        # Positive by construction (discrete-range bias); the interval is one-sided
        # on purpose, because a negative bias here would be a defect, not noise.
        ("oscillation", 0.00, 0.16),
        ("wavelet", -0.10, 0.10),
        ("leaders", -0.10, 0.12),
    ],
)
def test_fbm_pointwise_field_is_flat_at_hurst(name: str, lo: float, hi: float, H: float) -> None:
    r"""fBm has :math:`h(t)\equiv H`; recover it, and quantify the dispersion.

    The dispersion is *not* driven to zero by any tolerance here: pointwise
    estimation on one realisation is irreducibly noisy, and reporting that noise
    is the point of the test.
    """
    h = _finite(ESTIMATORS[name](fbm(N, H, seed=SEED)))
    bias = float(h.mean()) - H
    _report(f"fBm H={H} {name}", bias=bias, sd=float(h.std()), median_err=float(np.median(h)) - H)
    assert lo <= bias <= hi, f"{name} bias {bias:+.4f} outside [{lo}, {hi}] at H={H}"
    assert float(h.std()) < 0.30


@pytest.mark.theory
@pytest.mark.stochastic
def test_fbm_bias_is_systematic_not_a_lucky_seed() -> None:
    """Average the spatial mean over seeds: a bias claim on one path is worthless."""
    seeds = (101, 202, 303, 404, 505, 606, 707, 808)
    for H in (0.3, 0.5, 0.7):
        paths = [fbm(N, H, seed=s) for s in seeds]
        dispersion = {}
        for name, est in ESTIMATORS.items():
            fields = [_finite(est(p)) for p in paths]
            biases = np.array([float(f.mean()) - H for f in fields])
            dispersion[name] = float(np.mean([f.std() for f in fields]))
            _report(
                f"fBm multi-seed H={H} {name}",
                mean=float(biases.mean()),
                lo=float(biases.min()),
                hi=float(biases.max()),
                within_sd=dispersion[name],
            )
        # The documented ordering: the oscillation is the noisiest field, the
        # leaders the tightest. Neither is an artefact of one realisation.
        assert dispersion["oscillation"] > dispersion["wavelet"]
        osc = np.array([float(_finite(oscillation_holder(p)).mean()) - H for p in paths])
        assert osc.min() > 0.0, "the oscillation's positive bias must be systematic"


# --------------------------------------------------------------------------- #
# mBm: the only ground truth with a genuinely varying h(t)
# --------------------------------------------------------------------------- #


@pytest.mark.theory
@pytest.mark.stochastic
@pytest.mark.parametrize(
    ("name", "min_corr", "max_mae", "max_resid_sd"),
    [
        ("oscillation", 0.40, 0.14, 0.16),
        ("wavelet", 0.45, 0.09, 0.11),
        ("leaders", 0.65, 0.09, 0.10),
    ],
)
def test_mbm_ramp_is_tracked(
    name: str, min_corr: float, max_mae: float, max_resid_sd: float
) -> None:
    r"""mBm with :math:`h(t)=0.3+0.4t`: the estimated profile must track it.

    Two claims, because the obvious one is treacherous. Correlation is *not* a
    property of the estimator alone: it is
    :math:`\operatorname{cov}/(\sigma_{\text{truth}}\sigma_{\hat h})`, and the
    boundary margin decides how much of the ramp survives and therefore how large
    :math:`\sigma_{\text{truth}}` is. Measured on the wavelet estimator at
    ``seed=424242``, varying only ``j_max`` (and hence the margin):

    ========  =========  ===============  =========  ==============
    ``j_max``  retained  :math:`\sigma_{\text{truth}}`  corr    resid s.d.
    ========  =========  ===============  =========  ==============
    9          0.812     0.094            0.502      0.128
    10         0.625     0.072            0.540      0.092
    11         0.250     0.029            0.200      0.068
    ========  =========  ===============  =========  ==============

    The estimator gets monotonically *more* accurate (residual s.d. 0.128 →
    0.068) while the correlation collapses. So the residual dispersion is
    asserted too, and it is the claim with teeth; ``min_corr`` only guards
    against a field that has stopped tracking the ramp at all.
    """

    def ramp(u: np.ndarray) -> np.ndarray:
        return np.asarray(0.3 + 0.4 * u, dtype=np.float64)

    result = ESTIMATORS[name](mbm(N, ramp, seed=424242))
    ok = np.isfinite(result.h)
    truth = ramp(result.t)[ok]
    est = result.h[ok]
    corr = float(np.corrcoef(est, truth)[0, 1])
    mae = float(np.mean(np.abs(est - truth)))
    resid_sd = float((est - truth).std())
    _report(
        f"mBm {name}",
        corr=corr,
        mae=mae,
        resid_sd=resid_sd,
        truth_sd=float(truth.std()),
        bias=float(np.mean(est - truth)),
    )
    assert corr > min_corr
    assert mae < max_mae
    assert resid_sd < max_resid_sd


# --------------------------------------------------------------------------- #
# Ordering of two singularities of different strength
# --------------------------------------------------------------------------- #


@pytest.mark.theory
@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_two_cusps_of_different_strength_are_ordered(name: str) -> None:
    r"""Two well-separated cusps, :math:`a=0.3` at :math:`t=1/4` and
    :math:`a=0.8` at :math:`t=3/4`. Absolute accuracy differs by method; the
    *ordering* is the claim a localisation tool has to honour.
    """
    x = np.abs(_grid() - 0.25) ** 0.3 + np.abs(_grid() - 0.75) ** 0.8
    result = ESTIMATORS[name](x)
    rough, smooth = _at(result, 0.25), _at(result, 0.75)
    _report(f"two cusps {name}", h_rough=rough, h_smooth=smooth, gap=smooth - rough)
    assert rough < smooth - 0.2, f"{name}: h(0.25)={rough:.4f} not below h(0.75)={smooth:.4f}"
    assert rough < 0.55 and smooth > 0.55


# --------------------------------------------------------------------------- #
# D2 regression analogue: both branches of the spectrum, no moment grid needed
# --------------------------------------------------------------------------- #


@pytest.mark.theory
def test_both_branches_of_the_spectrum_are_reachable() -> None:
    r"""Legacy defect D2, in its pointwise form.

    v1 could only reach :math:`\alpha<\alpha_0` because its moment grid started
    at :math:`q=0`: the right branch of :math:`f(\alpha)` requires :math:`q<0`
    and was unreachable *in principle*, recovering 24% of the true support of the
    binomial measure. A pointwise estimator has no moment grid at all, so the
    obstruction cannot exist — and this test pins that down quantitatively
    against the same measure v1 failed on.
    """
    a_min, a_max = cascade_alpha_range([0.6, 0.4], 2)
    a_0 = 0.5 * (a_min + a_max)
    x = trading_time(deterministic_cascade(14, [0.6, 0.4]))[:-1]

    covered = {}
    for name, est in ESTIMATORS.items():
        h = _finite(est(x))
        lo, hi = float(np.quantile(h, 0.01)), float(np.quantile(h, 0.99))
        overlap = min(hi, a_max) - max(lo, a_min)
        covered[name] = overlap / (a_max - a_min)
        _report(f"D2 coverage {name}", lo=lo, hi=hi, fraction=covered[name])
        assert lo < a_0, "left branch (alpha < alpha_0) not reached"
        assert hi > a_0, "right branch (alpha > alpha_0) not reached"
        # v1's q in [0, 3] reached 24% of the support; q in [-3, 3] reached 52%.
        assert covered[name] > 0.80, f"{name} covers only {covered[name]:.0%} of the support"


# --------------------------------------------------------------------------- #
# Determinism and provenance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_bitwise_determinism(name: str) -> None:
    """Same input, same parameters, bitwise identical field. No hidden state."""
    x = fbm(N, 0.6, seed=SEED)
    first, second = ESTIMATORS[name](x), ESTIMATORS[name](x)
    assert np.array_equal(first.h, second.h, equal_nan=True)
    assert np.array_equal(first.stderr, second.stderr, equal_nan=True)  # type: ignore[arg-type]
    assert np.array_equal(first.r_squared, second.r_squared, equal_nan=True)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_provenance_root_is_stable(name: str) -> None:
    x = fbm(N, 0.6, seed=SEED)
    roots = {ESTIMATORS[name](x).provenance.root for _ in range(3)}  # type: ignore[union-attr]
    assert len(roots) == 1


def test_provenance_root_changes_with_every_parameter() -> None:
    """A parameter that does not move the root is a parameter missing from it."""
    x = fbm(N, 0.6, seed=SEED)

    def root(result: PointwiseHolder) -> str:
        assert result.provenance is not None
        return result.provenance.root

    base = root(oscillation_holder(x))
    assert root(oscillation_holder(x, window="causal")) != base
    assert root(oscillation_holder(x, trim_edges=False)) != base
    assert root(oscillation_holder(x, min_points=5)) != base
    assert root(oscillation_holder(x, scales=[8, 16, 32, 64, 128])) != base
    assert root(oscillation_holder(x[:-1])) != base

    w_base = root(wavelet_holder(x))
    assert root(wavelet_holder(x, cone=2.0)) != w_base
    assert root(wavelet_holder(x, wavelet="db4")) != w_base
    assert root(wavelet_holder(x, j_min=4)) != w_base

    l_base = root(leader_holder(x))
    assert root(leader_holder(x, gamma=0.0)) != l_base
    assert root(leader_holder(x, j_max=9)) != l_base


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_provenance_records_the_input_and_the_offset(name: str) -> None:
    x = fbm(N, 0.6, seed=SEED)
    prov = ESTIMATORS[name](x).provenance
    assert prov is not None
    assert [ref.name for ref in prov.inputs] == ["series"]
    assert prov.inputs[0].shape == (N,)
    assert prov.seed is None, "these estimators are deterministic; no seed to record"
    expected_offset = 0.0 if name == "oscillation" else 0.5
    assert prov.diagnostics["normalisation_offset"] == pytest.approx(expected_offset)


# --------------------------------------------------------------------------- #
# holder_at and local_regularity_map
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["oscillation", "wavelet", "leaders"])
def test_holder_at_agrees_with_the_field(method: str) -> None:
    """The single-point route and the vectorised field must be the same estimate."""
    x = _cusp(0.5)
    point, fit = holder_at(x, 0.5, method=method)
    field = _at(ESTIMATORS[method](x), 0.5)
    assert point == pytest.approx(field, abs=1e-9)
    assert fit.n_points >= 4
    assert fit.scaling_range is not None
    assert math.isfinite(fit.slope_stderr)
    assert fit.slope_ci[0] < fit.slope < fit.slope_ci[1]


def test_holder_at_supports_robust_fitters_the_field_cannot() -> None:
    """Theil-Sen is available for one location and refused for the whole field."""
    x = _cusp(0.5)
    robust, fit = holder_at(x, 0.5, fit_method=FitMethod.THEILSEN)
    assert fit.method is FitMethod.THEILSEN
    assert abs(robust - 0.5) < 0.05
    with pytest.raises(ValueError, match="not available for the pointwise field"):
        oscillation_holder(x, fit_method=FitMethod.THEILSEN)


def test_holder_at_refuses_a_locally_degenerate_point() -> None:
    """A locally constant signal has no exponent; say so instead of fitting one."""
    x = np.concatenate([np.zeros(N // 2), np.ones(N // 2)])
    with pytest.raises(ValueError, match="locally degenerate"):
        holder_at(x, 0.25)


@pytest.mark.parametrize("method", ["oscillation", "wavelet", "leaders"])
def test_regularity_map_shape_and_flatness(method: str) -> None:
    r"""On a pure cusp the local-slope surface must be a flat band at :math:`h=a`."""
    a = 0.5
    grid: RegularityMap = local_regularity_map(_cusp(a), method=method)
    assert grid.slopes.shape == (grid.t.size, grid.scales.size - 1)
    assert grid.scale_midpoints.size == grid.scales.size - 1
    assert np.all(np.diff(grid.scales) > 0)
    row = grid.slopes[int(np.argmin(np.abs(grid.t - 0.5)))]
    assert abs(float(np.nanmedian(row)) - a) < 0.25


def test_regularity_map_offset_matches_the_normalisation() -> None:
    x = _cusp(0.5)
    assert local_regularity_map(x, method="oscillation").offset == 0.0
    assert local_regularity_map(x, method="wavelet").offset == 0.5
    assert local_regularity_map(x, method="leaders", gamma=0.25).offset == 0.25


# --------------------------------------------------------------------------- #
# Options, edges and refusals
# --------------------------------------------------------------------------- #


def test_causal_window_recovers_the_cusp_and_keeps_the_right_edge() -> None:
    r"""A backward-looking window sees only the past; the exponent is unchanged."""
    x = _cusp(0.3)
    causal = oscillation_holder(x, window="causal")
    h = _at(causal, 0.5)
    _report("cusp a=0.3 causal", h=h, err=h - 0.3)
    assert abs(h - 0.3) < 0.05
    assert np.isfinite(causal.h[-1]), "a causal window cannot be contaminated on the right"
    assert not np.isfinite(causal.h[0])


def test_trim_edges_blanks_the_contaminated_margin() -> None:
    x = fbm(N, 0.5, seed=SEED)
    trimmed = oscillation_holder(x, trim_edges=True)
    kept = oscillation_holder(x, trim_edges=False)
    assert np.isnan(trimmed.h[0]) and np.isnan(trimmed.h[-1])
    assert np.isfinite(kept.h[0]) and np.isfinite(kept.h[-1])
    assert np.count_nonzero(np.isfinite(kept.h)) > np.count_nonzero(np.isfinite(trimmed.h))


def test_weighted_least_squares_requires_explicit_weights() -> None:
    """There is no count-based default here, and inventing one would be wrong."""
    x = _cusp(0.5)
    with pytest.raises(ValueError, match="requires explicit weights"):
        oscillation_holder(x, fit_method=FitMethod.WLS)
    weighted = oscillation_holder(x, fit_method=FitMethod.WLS, weights=np.linspace(1.0, 2.0, 16))
    assert abs(_at(weighted, 0.5) - 0.5) < 0.03


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_non_dyadic_length_is_handled(name: str) -> None:
    """Real series are not powers of two; the padding must not move the exponent."""
    n = 5000
    x = np.abs(_grid(n) - 0.5) ** 0.5
    result = ESTIMATORS[name](x)
    assert result.t[0] > 0.0 and result.t[-1] <= 1.0 + 1e-12
    assert abs(_at(result, 0.5) - 0.5) < 0.20


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_series_too_short_is_refused(name: str) -> None:
    with pytest.raises(ValueError):
        ESTIMATORS[name](np.arange(32, dtype=np.float64))


def test_unknown_options_are_refused() -> None:
    x = _cusp(0.5)
    with pytest.raises(ValueError, match="centred"):
        oscillation_holder(x, window="triangular")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown method"):
        local_regularity_map(x, method="wtmm")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cone"):
        wavelet_holder(x, cone=-1.0)
    with pytest.raises(ValueError, match="gamma"):
        leader_holder(x, gamma=-0.5)
    with pytest.raises(ValueError, match="j_min"):
        wavelet_holder(x, j_min=0)


def test_leaders_report_one_exponent_per_finest_dyadic_interval() -> None:
    """Resolution is advertised honestly: intervals, not samples.

    The coarsest octave is 10, not ``log2(n) - 3 = 11``: the default octave
    ceiling is also bounded by the boundary margin it would have to pay for
    (:data:`mmar.estimators.pointwise._WRAP_MARGIN_OCTAVES`), which is the same
    ceiling :mod:`mmar.estimators.wavelet_leaders` reaches independently.
    """
    result = leader_holder(_cusp(0.5))
    assert result.t.size == N // 2
    assert result.scales.tolist() == [2.0**j for j in range(3, 11)]


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_r_squared_is_reported_and_high_on_a_clean_power_law(name: str) -> None:
    """A caller must be able to see where the local power law is trustworthy."""
    result = ESTIMATORS[name](_cusp(0.5))
    assert result.r_squared is not None
    r2 = result.r_squared[int(np.argmin(np.abs(result.t - 0.5)))]
    _report(f"cusp r2 {name}", r2=float(r2))
    assert r2 > 0.95


def test_singular_times_locates_the_rough_point() -> None:
    """The whole purpose of the field: find *where* the signal is singular."""
    x = np.abs(_grid() - 0.25) ** 0.3 + np.abs(_grid() - 0.75) ** 0.8
    result = wavelet_holder(x)
    idx = result.singular_times(0.45)
    assert idx.size > 0
    times = result.t[idx]
    assert abs(float(np.median(times)) - 0.25) < 0.05


# --------------------------------------------------------------------------- #
# Documented caveats, pinned so they cannot silently change
# --------------------------------------------------------------------------- #


def test_leader_grid_is_not_translation_invariant() -> None:
    r"""The decimated DWT is shift-sensitive; the docstring says so, so pin it.

    Shifting the same cusp by a few samples moves the leader estimate by ~0.1.
    This is a property of the orthonormal dyadic grid, not a bug, and it is the
    reason :func:`wavelet_holder` (undecimated, shift-equivariant) is preferred
    for an isolated singularity whose position matters.
    """
    values = []
    for shift in range(4):
        x = np.abs((np.arange(N, dtype=np.float64) + 1.0 - shift) / N - 0.5) ** 0.5
        values.append(_at(leader_holder(x), 0.5))
    _report("leader shift sensitivity", **{f"s{i}": v for i, v in enumerate(values)})
    spread = max(values) - min(values)
    assert 0.02 < spread < 0.20, f"observed shift spread {spread:.4f}"


def test_wavelet_cone_saturates_and_kappa_zero_is_worse() -> None:
    r"""The cone is a variance-reduction device with a documented saturation."""
    x = _cusp(0.5)
    bare = _at(wavelet_holder(x, cone=0.0), 0.5)
    one = _at(wavelet_holder(x, cone=1.0), 0.5)
    two = _at(wavelet_holder(x, cone=2.0), 0.5)
    _report("cone kappa", k0=bare, k1=one, k2=two)
    assert abs(one - 0.5) < abs(bare - 0.5), "the cone must improve the isolated cusp"
    assert one == pytest.approx(two, abs=1e-12), "kappa >= 1 must saturate on a lone cusp"

    noisy = fbm(N, 0.5, seed=SEED)
    sd_bare = float(_finite(wavelet_holder(noisy, cone=0.0)).std())
    sd_cone = float(_finite(wavelet_holder(noisy, cone=1.0)).std())
    _report("cone dispersion on fBm", k0=sd_bare, k1=sd_cone)
    assert sd_cone < sd_bare
