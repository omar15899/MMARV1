r"""Surrogate generators: each must preserve *exactly* what it claims to preserve.

The value of a surrogate test rests entirely on the invariance its null encodes,
so every invariance here is asserted at machine precision rather than
statistically: the shuffle family must reproduce the sorted sample bit for bit,
and the Fourier family must reproduce the amplitude spectrum to round-off. Where
a statistical claim is unavoidable (a destroyed correlation, a p-value) the
assertion is made against a sampling bound derived from the ensemble itself, and
every draw is seeded, so a failure is a regression and not a coin flip.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import pytest
from scipy.special import logsumexp

from mmar.core.regression import loglog_fit
from mmar.core.types import FloatArray, ScalingExponents, dyadic_scales
from mmar.core.validate import DataQualityError
from mmar.inference.surrogates import (
    SurrogateKind,
    aaft_surrogate,
    amplitude_spectrum,
    block_shuffle_surrogate,
    iaaft_surrogate,
    phase_randomised_surrogate,
    shuffle_surrogate,
    spectrum_error,
    spectrum_width,
    surrogate_ensemble,
    surrogate_ensemble_with_provenance,
    surrogate_test,
)
from mmar.models.cascade import lognormal_cascade
from mmar.models.fbm import fgn, levy_stable_increments

pytestmark = pytest.mark.stochastic

SEED = 20260727


# --------------------------------------------------------------------------- #
# Fixtures: one correlated Gaussian series, one correlated non-Gaussian series
# --------------------------------------------------------------------------- #


def _fgn(n: int = 1024, H: float = 0.8, seed: int = SEED) -> FloatArray:
    """Long-memory Gaussian noise: strong linear correlation, Gaussian marginal."""
    return fgn(n, H, seed=seed)


def _lognormal_memory(n: int = 1024, H: float = 0.8, seed: int = SEED) -> FloatArray:
    """Correlated *and* strongly non-Gaussian: the regime where AAFT breaks."""
    return np.asarray(np.exp(fgn(n, H, seed=seed)), dtype=np.float64)


def _acf1(x: FloatArray) -> float:
    """Lag-one sample autocorrelation."""
    y = x - float(np.mean(x))
    return float(np.dot(y[:-1], y[1:]) / np.dot(y, y))


# --------------------------------------------------------------------------- #
# A self-contained estimator, so this file depends on no estimator module
# --------------------------------------------------------------------------- #


def _toy_estimator(x: FloatArray) -> ScalingExponents:
    r"""Structure functions of the integrated series, in the MFDFA convention.

    Takes an *increment* series, forms the profile
    :math:`Y_t=\sum_{i\le t}(x_i-\bar x)`, and fits
    :math:`\log S(q,s)=\log\langle|Y_{t+s}-Y_t|^q\rangle` against :math:`\log s`
    over dyadic scales. Surrogating the increments and integrating inside the
    estimator is the standard composition: shuffling a *path* would destroy the
    random-walk structure itself and test nothing about the increments.

    Deliberately minimal — no scaling-range selection, no weighting — because its
    only job here is to be a fixed, deterministic map from a series to
    :math:`\hat\zeta(q)` that the surrogate machinery can be exercised against.
    The grid is :math:`q>0` only: on a centred profile the increments pass
    through zero, so negative moments are dominated by near-cancellations rather
    than by the small-:math:`\mu` cells they are meant to probe. Moments are
    accumulated in the log domain with :func:`scipy.special.logsumexp`.
    """
    q = np.linspace(0.2, 4.0, 13)
    profile = np.cumsum(x - float(np.mean(x)))
    scales = dyadic_scales(profile.size, j_min=2, j_max=int(math.log2(profile.size)) - 3)
    log_s = np.empty((q.size, scales.size), dtype=np.float64)
    for j, s in enumerate(scales):
        inc = np.abs(profile[int(s) :] - profile[: -int(s)])
        li = np.log(inc[inc > 0.0])
        log_s[:, j] = logsumexp(q[:, None] * li[None, :], axis=1) - math.log(li.size)
    zeta = np.array(
        [loglog_fit(scales.astype(np.float64), log_s[i], log_y=False).slope for i in range(q.size)],
        dtype=np.float64,
    )
    return ScalingExponents(q=q, zeta=zeta, estimator="toy.structure_function")


# --------------------------------------------------------------------------- #
# Shuffle
# --------------------------------------------------------------------------- #


def test_shuffle_preserves_the_marginal_exactly() -> None:
    x = _lognormal_memory()
    y = shuffle_surrogate(x, seed=SEED)
    assert np.array_equal(np.sort(x), np.sort(y)), "the shuffle must be a permutation, bit for bit"
    assert y.dtype == np.float64


def test_shuffle_destroys_the_lag_one_autocorrelation() -> None:
    x = _fgn(n=4096, H=0.85)
    observed = _acf1(x)
    assert observed > 0.5, f"the fixture must be strongly correlated, got acf1={observed:.3f}"

    ensemble = surrogate_ensemble(x, SurrogateKind.SHUFFLE, 32, seed=SEED)
    acf = np.array([_acf1(row) for row in ensemble])
    # Under the iid null, acf1 has standard error ~1/sqrt(n); four of them is the bound.
    bound = 4.0 / math.sqrt(x.size)
    assert np.max(np.abs(acf)) < bound, f"max |acf1| = {np.max(np.abs(acf)):.4f} >= {bound:.4f}"


# --------------------------------------------------------------------------- #
# Phase randomisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [1024, 1023])
def test_phase_randomisation_preserves_the_periodogram(n: int) -> None:
    """Both parities: an even-length series has a Nyquist bin, an odd one does not."""
    x = _fgn(n=n)
    y = phase_randomised_surrogate(x, seed=SEED)
    a, b = amplitude_spectrum(x), amplitude_spectrum(y)
    rel = float(np.max(np.abs(b - a)) / np.max(a))
    assert rel < 1e-10, f"amplitude spectrum drifted by {rel:.3e}"
    assert spectrum_error(x, y) < 1e-12


@pytest.mark.parametrize("n", [1024, 1023])
def test_phase_randomisation_output_is_real_to_machine_precision(n: int) -> None:
    x = _fgn(n=n) / float(np.std(fgn(n, 0.8, seed=SEED)))
    result = surrogate_ensemble_with_provenance(x, "phase", 8, seed=SEED)
    assert np.isrealobj(result.values)
    residual = float(result.provenance.diagnostics["imag_residual_max"])
    assert residual < 1e-12, f"imaginary residual {residual:.3e} of an inverse FFT forced real"


def test_phase_randomisation_preserves_the_mean_and_kills_the_ordering() -> None:
    x = _fgn(n=2048, H=0.85)
    y = phase_randomised_surrogate(x, seed=SEED)
    assert float(np.mean(y)) == pytest.approx(float(np.mean(x)), abs=1e-12)
    # The linear ACF is a function of the amplitude spectrum alone, hence preserved.
    assert _acf1(y) == pytest.approx(_acf1(x), abs=5e-3)
    assert not np.array_equal(np.sort(x), np.sort(y)), "the marginal is Gaussianised, not preserved"


# --------------------------------------------------------------------------- #
# AAFT and IAAFT
# --------------------------------------------------------------------------- #


def test_aaft_and_iaaft_preserve_the_marginal_exactly() -> None:
    x = _lognormal_memory()
    for y in (aaft_surrogate(x, seed=SEED), iaaft_surrogate(x, seed=SEED)):
        assert np.array_equal(np.sort(x), np.sort(y))


def test_iaaft_reproduces_the_spectrum_better_than_aaft() -> None:
    """The defining claim of Schreiber & Schmitz (1996), measured, not assumed."""
    x = _lognormal_memory(n=1024, H=0.85)
    n_rep = 8
    aaft = surrogate_ensemble(x, "aaft", n_rep, seed=SEED)
    iaaft = surrogate_ensemble(x, "iaaft", n_rep, seed=SEED, max_iter=400)
    err_aaft = np.array([spectrum_error(x, row) for row in aaft])
    err_iaaft = np.array([spectrum_error(x, row) for row in iaaft])
    print(
        f"\nspectrum error   AAFT  mean={err_aaft.mean():.5f} max={err_aaft.max():.5f}"
        f"\n                 IAAFT mean={err_iaaft.mean():.5f} max={err_iaaft.max():.5f}"
        f"\n                 ratio mean={err_aaft.mean() / err_iaaft.mean():.2f}x"
    )
    assert err_iaaft.max() < err_aaft.min(), "every IAAFT draw must beat every AAFT draw"
    assert err_iaaft.mean() < 0.2 * err_aaft.mean()


def test_iaaft_convergence_is_reported_and_reached() -> None:
    x = _lognormal_memory(n=1024)
    result = surrogate_ensemble_with_provenance(x, "iaaft", 4, seed=SEED, max_iter=400)
    diag = result.provenance.diagnostics
    print(
        f"\nIAAFT iterations_max={diag['iterations_max']} "
        f"spectrum_error_max={diag['spectrum_error_max']:.5f} "
        f"mean={diag['spectrum_error_mean']:.5f}"
    )
    assert diag["n_not_converged"] == 0
    assert 0 < int(diag["iterations_max"]) <= 400
    # A nonzero plateau is expected: the two constraint sets do not intersect.
    assert 0.0 < float(diag["spectrum_error_max"]) < 0.05


def test_iaaft_rejects_a_degenerate_iteration_budget() -> None:
    x = _fgn(n=256)
    with pytest.raises(ValueError, match="max_iter"):
        iaaft_surrogate(x, seed=SEED, max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        iaaft_surrogate(x, seed=SEED, tol=-1.0)


# --------------------------------------------------------------------------- #
# Block shuffle
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("block", [1, 7, 64, 300, 1024])
def test_block_shuffle_is_always_a_permutation(block: int) -> None:
    """Including the ragged case: 1024 is not divisible by 7 or by 300."""
    x = _lognormal_memory(n=1024)
    y = block_shuffle_surrogate(x, block=block, seed=SEED)
    assert y.size == x.size
    assert np.array_equal(np.sort(x), np.sort(y))


def test_block_shuffle_at_full_length_is_the_identity() -> None:
    x = _fgn(n=256)
    assert np.array_equal(block_shuffle_surrogate(x, block=x.size, seed=SEED), x)


def test_block_shuffle_rejects_an_out_of_range_block() -> None:
    x = _fgn(n=256)
    for bad in (0, -3, 257):
        with pytest.raises(ValueError, match="block must lie"):
            block_shuffle_surrogate(x, block=bad, seed=SEED)


def test_block_length_controls_how_much_correlation_survives() -> None:
    r"""Lag-one correlation is retained with weight :math:`(L-1)/L`.

    This monotone recovery is what makes a block-length sweep a scale
    diagnostic rather than a single yes/no test.
    """
    x = _fgn(n=4096, H=0.85)
    target = _acf1(x)
    retained = {}
    for block in (1, 4, 16, 64, 256):
        ensemble = surrogate_ensemble(x, "block", 16, seed=SEED, block=block)
        retained[block] = float(np.mean([_acf1(row) for row in ensemble])) / target
    print(f"\nretained lag-1 acf by block length: {retained}")
    fractions = [retained[b] for b in (1, 4, 16, 64, 256)]
    assert all(a < b for a, b in pairwise(fractions)), "must be monotone"
    assert abs(retained[1]) < 0.05, "block=1 is a full shuffle"
    # Theory: 1 - 1/L, up to the sampling noise of a 16-member ensemble.
    for block in (4, 16, 64, 256):
        assert retained[block] == pytest.approx(1.0 - 1.0 / block, abs=0.03)


# --------------------------------------------------------------------------- #
# Reproducibility and ensembles
# --------------------------------------------------------------------------- #

_KINDS: Sequence[tuple[str, dict[str, int]]] = (
    ("shuffle", {}),
    ("phase", {}),
    ("aaft", {}),
    ("iaaft", {}),
    ("block", {"block": 32}),
)


@pytest.mark.parametrize(("kind", "kwargs"), _KINDS)
def test_every_surrogate_is_bitwise_reproducible_from_its_seed(
    kind: str, kwargs: dict[str, int]
) -> None:
    x = _lognormal_memory(n=512)
    a = surrogate_ensemble(x, kind, 3, seed=SEED, **kwargs)
    b = surrogate_ensemble(x, kind, 3, seed=SEED, **kwargs)
    assert np.array_equal(a, b), "identical seeds must give identical bytes"
    c = surrogate_ensemble(x, kind, 3, seed=SEED + 1, **kwargs)
    assert not np.array_equal(a, c), "a different seed must give a different ensemble"


@pytest.mark.parametrize(("kind", "kwargs"), _KINDS)
def test_ensemble_members_are_distinct_and_mutually_independent(
    kind: str, kwargs: dict[str, int]
) -> None:
    x = _fgn(n=2048, H=0.7)
    m = 12
    ensemble = surrogate_ensemble(x, kind, m, seed=SEED, **kwargs)
    assert ensemble.shape == (m, x.size)

    digests = {row.tobytes() for row in ensemble}
    assert len(digests) == m, "spawn_rngs must not hand two members the same stream"

    corr = np.asarray(np.corrcoef(ensemble), dtype=np.float64)
    pairs = corr[np.triu_indices(m, k=1)]
    # Independence is a statement about the *mean* pairwise correlation. Its
    # spread is not 1/sqrt(n): Fourier-based members share an amplitude
    # spectrum, so for a long-memory series only the few dominant frequencies
    # are effectively independent and individual pairs fluctuate much more.
    # The bound is therefore taken from the sample's own standard error.
    stderr = float(np.std(pairs, ddof=1)) / math.sqrt(pairs.size)
    assert abs(float(np.mean(pairs))) < 4.0 * stderr, (
        f"mean pairwise corr = {np.mean(pairs):+.4f} exceeds 4 SE = {4 * stderr:.4f}"
    )
    assert float(np.max(np.abs(pairs))) < 0.5, "no two members may be near-duplicates"


def test_ensemble_rejects_a_non_positive_size_and_a_missing_block_length() -> None:
    x = _fgn(n=256)
    with pytest.raises(ValueError, match="n must be a positive integer"):
        surrogate_ensemble(x, "shuffle", 0, seed=SEED)
    with pytest.raises(ValueError, match="requires an explicit block length"):
        surrogate_ensemble(x, "block", 2, seed=SEED)
    with pytest.raises(ValueError, match="is not a valid SurrogateKind"):
        surrogate_ensemble(x, "bootstrap", 2, seed=SEED)


def test_short_or_degenerate_input_is_rejected_loudly() -> None:
    with pytest.raises(DataQualityError):
        shuffle_surrogate(np.arange(8, dtype=np.float64), seed=SEED)
    with pytest.raises(DataQualityError):
        shuffle_surrogate(np.ones(64), seed=SEED)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_provenance_root_is_a_pure_function_of_the_recipe() -> None:
    x = _fgn(n=512)
    a = surrogate_ensemble_with_provenance(x, "iaaft", 2, seed=SEED)
    b = surrogate_ensemble_with_provenance(x, "iaaft", 2, seed=SEED)
    assert a.provenance.root == b.provenance.root
    assert np.array_equal(a.values, b.values)
    assert a.provenance.diagnostics["values_digest"] == b.provenance.diagnostics["values_digest"]

    different = (
        surrogate_ensemble_with_provenance(x, "iaaft", 2, seed=SEED + 1),
        surrogate_ensemble_with_provenance(x, "aaft", 2, seed=SEED),
        surrogate_ensemble_with_provenance(x, "iaaft", 3, seed=SEED),
        surrogate_ensemble_with_provenance(x + 1.0, "iaaft", 2, seed=SEED),
    )
    for other in different:
        assert other.provenance.root != a.provenance.root


def test_diagnostics_do_not_enter_the_root() -> None:
    x = _fgn(n=512)
    result = surrogate_ensemble_with_provenance(x, "iaaft", 2, seed=SEED)
    stripped = result.provenance.with_diagnostics()
    assert stripped.root == result.provenance.root
    assert result.summary()["n_members"] == 2


# --------------------------------------------------------------------------- #
# The test itself: the fat-tail impostor must NOT be called multifractal
# --------------------------------------------------------------------------- #


def _cascade_increments(kmax: int = 12, lam: float = 1.10, seed: int = SEED) -> FloatArray:
    """Cell masses of a lognormal cascade: genuine multifractality, by construction.

    Integrating them (which :func:`_toy_estimator` does) gives the trading time of
    :func:`~mmar.models.cascade.trading_time` up to a normalisation, so this is
    the MMAR's multifractal component itself.
    """
    return lognormal_cascade(kmax, lam, seed=seed)


def _levy_increments(n: int = 4096, alpha: float = 1.5, seed: int = SEED) -> FloatArray:
    """iid alpha-stable draws: monofractal, but with a wide *estimated* spectrum."""
    return levy_stable_increments(n, alpha, seed=seed)


def test_shuffle_test_separates_a_cascade_from_a_fat_tailed_iid_process() -> None:
    r"""One worked example of the discrimination, pinned as a regression.

    Both series have a wide estimated :math:`f(\alpha)`. Only one of them has it
    *because of temporal structure*, and only the surrogate test can tell.

    Note what the numbers say about the cascade: the shuffled null keeps roughly
    70% of the observed width. That residual is the contribution of the
    cascade's own fat-tailed marginal, and it is real --- the correlation
    component is the *excess* over it, not the whole of it. This is exactly
    Zhou's (2009) decomposition, and it is why quoting :math:`\Delta\alpha`
    without a surrogate baseline overstates multifractality even when the
    process genuinely is multifractal.

    **This is a single draw, and a single draw is not the capability claim.**
    Measured across 20 independent cascade realisations at these very settings,
    ``p <= 0.05`` holds for only about half of them and
    ``null_mean < 0.85 * statistic`` for about half; at ``kmax=14, lam=1.30`` the
    latter held for none of 12 draws, because a more strongly multifractal
    cascade has a fatter marginal and its *shuffled* width grows faster than its
    observed width. The test is therefore under-powered per realisation by
    construction, and the honest statement of the effect --- a paired,
    systematically positive width excess that fat-tailed iid data do not show ---
    is asserted in
    :func:`test_the_width_excess_over_the_shuffled_null_is_systematic`.
    """
    cascade = surrogate_test(
        _cascade_increments(), _toy_estimator, seed=SEED, kind="shuffle", n_surrogates=99
    )
    levy = surrogate_test(
        _levy_increments(), _toy_estimator, seed=SEED, kind="shuffle", n_surrogates=99
    )
    print(
        f"\ncascade: width={cascade.statistic:.4f} null_mean={cascade.null.mean():.4f} "
        f"p={cascade.p_value:.3f} z={cascade.z_score:.2f}"
        f"\nlevy   : width={levy.statistic:.4f} null_mean={levy.null.mean():.4f} "
        f"p={levy.p_value:.3f} z={levy.z_score:.2f}"
    )
    assert cascade.statistic > 0.2, "the cascade fixture must look multifractal to begin with"
    assert levy.statistic > 0.2, "so must the fat-tailed one — that is the point"

    # Genuine multifractality: part of the width does not survive shuffling.
    assert cascade.p_value <= 0.05
    assert cascade.null.mean() < 0.85 * cascade.statistic

    # The impostor: shuffling changes nothing, because there was nothing to destroy.
    assert levy.p_value > 0.20
    assert levy.null.mean() > 0.90 * levy.statistic


def test_the_width_excess_over_the_shuffled_null_is_systematic() -> None:
    r"""The capability claim, stated over independent realisations rather than one.

    A single surrogate test on a single cascade is under-powered: the shuffled
    null retains most of the width, so the excess
    :math:`\Delta\alpha - \overline{\Delta\alpha^{(s)}}` is small relative to its
    sampling spread and only about half of the draws reach :math:`p\le 0.05`.
    What is *not* ambiguous is the sign of that excess across draws. Here it is
    measured as a paired quantity --- each realisation supplies its own null ---
    over independent cascades and independent alpha-stable series:

    * for cascades the excess must be systematically positive (the ordering
      carries width that shuffling destroys);
    * for iid alpha-stable draws it must not be, because there is no ordering to
      destroy and the entire estimated width is a property of the marginal.

    This is the assertion that would fail if the machinery were broken, and it
    is the one a reader should take as the module's claim.
    """
    reps, n_surrogates = 10, 19

    def excess(series: FloatArray, seed: int) -> float:
        r = surrogate_test(
            series, _toy_estimator, seed=seed, kind="shuffle", n_surrogates=n_surrogates
        )
        return r.statistic - float(r.null.mean())

    cascade = np.array(
        [excess(_cascade_increments(seed=50_000 + i), 1_200 + i) for i in range(reps)]
    )
    levy = np.array([excess(_levy_increments(seed=60_000 + i), 1_300 + i) for i in range(reps)])

    def t_stat(d: FloatArray) -> float:
        return float(d.mean() / (float(np.std(d, ddof=1)) / math.sqrt(d.size)))

    print(
        f"\nwidth excess over the shuffled null (paired, {reps} draws each)"
        f"\n  cascade: mean={cascade.mean():+.4f} positive={int((cascade > 0).sum())}/{reps} "
        f"t={t_stat(cascade):+.2f}"
        f"\n  levy   : mean={levy.mean():+.4f} positive={int((levy > 0).sum())}/{reps} "
        f"t={t_stat(levy):+.2f}"
    )

    assert cascade.mean() > 0.0, "a cascade must lose width when its ordering is destroyed"
    assert t_stat(cascade) > 2.5, f"the excess must be systematic, got t={t_stat(cascade):+.2f}"
    assert int((cascade > 0).sum()) >= 8, "the sign must hold for the large majority of draws"

    # The impostor: no ordering, hence no excess. Two-sided, because an estimator
    # that manufactured a *negative* excess on iid data would be just as broken.
    assert abs(t_stat(levy)) < 2.5, (
        f"iid alpha-stable shows a spurious excess, t={t_stat(levy):+.2f}"
    )
    assert abs(levy.mean()) < 0.25 * cascade.mean(), (
        f"the impostor's excess {levy.mean():+.4f} is not negligible against the "
        f"cascade's {cascade.mean():+.4f}"
    )


def test_surrogate_test_validates_its_inputs() -> None:
    x = _fgn(n=512)
    with pytest.raises(ValueError, match="n_surrogates"):
        surrogate_test(x, _toy_estimator, seed=SEED, n_surrogates=0)
    with pytest.raises(ValueError, match="nothing to test"):
        surrogate_test(
            x, _toy_estimator, seed=SEED, n_surrogates=2, statistic=lambda _: float("nan")
        )


def test_surrogate_test_carries_a_chained_provenance() -> None:
    x = _cascade_increments(kmax=10)
    result = surrogate_test(x, _toy_estimator, seed=SEED, kind="shuffle", n_surrogates=9)
    assert result.n_surrogates == 9
    assert 0.0 < result.p_value <= 1.0
    assert result.provenance.params["statistic"] == "spectrum_width"
    # The ensemble is the parent: the test is a Merkle child of the surrogates it used.
    ensemble = surrogate_ensemble_with_provenance(x, "shuffle", 9, seed=SEED)
    assert result.provenance.inputs[0].digest == ensemble.provenance.root
    assert result.summary()["p_value"] == result.p_value
    assert spectrum_width(_toy_estimator(x)) == pytest.approx(result.statistic)


def test_the_test_record_keeps_the_convergence_flag_of_its_own_null() -> None:
    """A p-value against unconverged IAAFT surrogates must say so in its manifest.

    The module's own doctrine is that an IAAFT surrogate whose spectrum
    discrepancy was never checked is an AAFT surrogate with extra steps. The
    fixture is deliberately hard --- a strongly skewed long-memory series at the
    *default* iteration budget --- so that some members genuinely fail to
    converge and the flag has something to report.
    """
    hard = np.asarray(np.exp(2.0 * fgn(2048, 0.9, seed=SEED)), dtype=np.float64)
    ensemble = surrogate_ensemble_with_provenance(hard, "iaaft", 4, seed=SEED)
    result = surrogate_test(hard, _toy_estimator, seed=SEED, kind="iaaft", n_surrogates=4)

    for key in ("spectrum_error_max", "spectrum_error_mean", "iterations_max", "n_not_converged"):
        assert key in result.provenance.diagnostics, f"{key} was dropped from the test record"
        assert result.provenance.diagnostics[key] == ensemble.provenance.diagnostics[key]
    diag = result.provenance.diagnostics
    print(
        f"\nIAAFT@default(max_iter=200) n_not_converged={diag['n_not_converged']}/4 "
        f"iterations_max={diag['iterations_max']} "
        f"spectrum_error_max={diag['spectrum_error_max']:.5f}"
    )
    # The flag must be doing work, not sitting at a constant zero: at the default
    # budget this fixture genuinely does not reach a fixed point, which is the
    # whole reason the manifest has to carry the number.
    assert int(diag["n_not_converged"]) > 0, "fixture no longer exercises the flag"
    assert int(diag["iterations_max"]) == 200

    phase = surrogate_test(hard, _toy_estimator, seed=SEED, kind="phase", n_surrogates=4)
    assert "imag_residual_max" in phase.provenance.diagnostics
