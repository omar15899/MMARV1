"""Scale behaviour and estimator self-consistency.

Every test here attacks an invariance the multifractal formalism *guarantees*,
so a violation is an implementation error and not a statistical one:

* :math:`\\zeta` is a slope, so ``x -> c*x`` and ``x -> x + a`` may move the
  intercept of every log--log fit but never its slope;
* the exponent at one :math:`q` is a pointwise estimate, so refining the
  :math:`q` grid cannot move the exponents at the :math:`q` values both grids
  share;
* ``<estimator>_structure`` followed by a refit is *the same computation* as
  ``<estimator>``, so the two must agree bit for bit;
* a stationary-increment process has time-reversal-symmetric exponents;
* the content address of a deterministic estimator must not carry anything
  session-local.

The growth tests pin the cost model: a peak allocation that scales with the
:math:`q` grid rather than with the series is a resource defect that only shows
up on the record sizes users actually have.
"""

from __future__ import annotations

import functools
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from mmar.analysis import analyze
from mmar.core.rng import make_rng
from mmar.core.types import FloatArray, ScalingExponents
from mmar.estimators.ghe import generalised_hurst
from mmar.estimators.mfdfa import mfdfa, mfdfa_from_structure, mfdfa_structure
from mmar.estimators.mfdma import mfdma, mfdma_exponents_from_structure, mfdma_structure
from mmar.estimators.structure_function import (
    exponents_from_structure as mfc_from_structure,
)
from mmar.estimators.structure_function import (
    mfc_exponents,
    mfc_exponents_structure,
)
from mmar.estimators.wavelet_leaders import (
    exponents_from_structure as leaders_from_structure,
)
from mmar.estimators.wavelet_leaders import (
    wavelet_leaders,
    wavelet_leaders_structure,
)
from mmar.estimators.wtmm import wtmm
from mmar.models.cascade import deterministic_cascade, trading_time
from mmar.models.fbm import fbm

N = 4096
Q = np.linspace(-3.0, 3.0, 13)
Q_POSITIVE = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
BINOMIAL = (0.65, 0.35)

Estimator = Callable[..., ScalingExponents]


def _mfdfa_on_level(x: Any, **kwargs: Any) -> ScalingExponents:
    """MF-DFA is the only estimator whose default re-integrates its input.

    ``analyze`` pins ``integrate=False`` for the same reason: every test here
    hands over a level series, and integrating it twice shifts h by exactly one.
    """
    return mfdfa(x, integrate=False, **kwargs)


ESTIMATORS: dict[str, Estimator] = {
    "mfdfa": _mfdfa_on_level,
    "mfdma": mfdma,
    "structure_function": mfc_exponents,
    "wavelet_leaders": wavelet_leaders,
    "ghe": generalised_hurst,
    "wtmm": wtmm,
}
NAMES = tuple(ESTIMATORS)


def fbm_path(seed: int = 11, n: int = N, hurst: float = 0.7) -> FloatArray:
    return np.asarray(fbm(n, hurst, seed=seed), dtype=np.float64)


def cascade_path(kmax: int = 12) -> FloatArray:
    """Trading time of the deterministic binomial measure.

    Reversing it in time gives the trading time of the :math:`(m_1,m_0)`
    cascade, whose :math:`\\tau(q)=1-\\log_2(m_0^q+m_1^q)` is *identical*: the
    formula is symmetric under swapping the multipliers. Any exponent shift
    under reversal is therefore estimator error, never signal.
    """
    return np.asarray(trading_time(deterministic_cascade(kmax, m=BINOMIAL)), dtype=np.float64)


def crossover_path(n: int, h_small: float, h_large: float, s_break: int, seed: int) -> FloatArray:
    """Level series whose spectral exponent breaks at ``s_break``.

    Built by filtering white noise with an amplitude
    :math:`|f|^{(1-2H)/2}` that switches Hurst exponent at
    :math:`f_c=1/s_{\\text{break}}`, then integrating. Small scales (high
    frequencies) carry ``h_small``, large scales ``h_large``, and the two
    branches are matched at :math:`f_c` so the spectrum is continuous.
    """
    rng = make_rng(seed)
    white = rng.standard_normal(n)
    freq = np.fft.rfftfreq(n, d=1.0)
    freq[0] = freq[1]
    f_c = 1.0 / s_break
    p_small, p_large = (1.0 - 2.0 * h_small) / 2.0, (1.0 - 2.0 * h_large) / 2.0
    match = f_c**p_small / f_c**p_large
    amp = np.where(freq >= f_c, freq**p_small, match * freq**p_large)
    amp[0] = 0.0
    noise = np.fft.irfft(np.fft.rfft(white) * amp, n=n)
    return np.asarray(np.cumsum(noise - noise.mean()), dtype=np.float64)


# --------------------------------------------------------------------------- #
# (a) Affine invariance of the exponents
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("c", [1.0e3, 1.0e-3])
def test_rescaling_the_series_leaves_zeta_unchanged(name: str, c: float) -> None:
    r"""``x -> c*x`` multiplies :math:`S(q,\delta)` by :math:`c^q`.

    That is a :math:`\delta`-independent factor, so it shifts the intercept of
    every log--log fit and leaves every slope alone. The structure-function
    module states this in its own overflow message ("rescale the series ...
    shifts only the intercept, never the slope"), so an estimator whose
    :math:`\zeta` moves under a change of units is contradicting the library's
    own advice, and a user who reports exponents in cents rather than pounds
    would get different physics.
    """
    fn = ESTIMATORS[name]
    x = fbm_path()
    base = fn(x, q=Q).zeta
    scaled = fn(c * x, q=Q).zeta
    assert np.max(np.abs(scaled - base)) < 1.0e-8, f"{name}: zeta moved under x -> {c:g}*x"


@pytest.mark.parametrize("name", NAMES)
def test_an_additive_offset_leaves_zeta_unchanged(name: str) -> None:
    """Every estimator here is increment- or wavelet-based, so a DC level cancels.

    MF-DFA and MF-DMA subtract a local polynomial or moving average; the
    structure function and GHE difference; the wavelet families use filters with
    vanishing moments. If any of them lets a constant through, an analysis of a
    price and of the same price quoted relative to a reference would disagree.
    """
    fn = ESTIMATORS[name]
    x = fbm_path()
    offset = 10.0 * float(np.std(x))
    base = fn(x, q=Q).zeta
    shifted = fn(x + offset, q=Q).zeta
    assert np.max(np.abs(shifted - base)) < 1.0e-5, f"{name}: zeta moved under x -> x + {offset:g}"


def test_analyze_summary_is_affine_invariant() -> None:
    """The invariance must survive the whole pipeline, not just the estimator.

    ``analyze`` runs the Legendre transform and the log-cumulants on top of the
    exponents. Both are built from ``zeta``, so ``alpha_0``, ``width``, ``c1``
    and ``c2`` inherit the invariance; a leak would mean the spectrum, not the
    exponents, is where the units got in.
    """
    x = fbm_path(seed=17)
    keys = ("h2", "alpha_0", "width", "c1", "c2")
    for estimator in ("wavelet_leaders", "mfdfa", "mfdma", "structure_function", "ghe"):
        base = analyze(x, estimator=estimator, q=Q).summary()
        scaled = analyze(100.0 * x, estimator=estimator, q=Q).summary()
        shifted = analyze(x + 50.0, estimator=estimator, q=Q).summary()
        for key in keys:
            assert abs(scaled[key] - base[key]) < 1.0e-9, f"{estimator}.{key} under rescaling"
            assert abs(shifted[key] - base[key]) < 1.0e-8, f"{estimator}.{key} under offset"


# --------------------------------------------------------------------------- #
# (b) The q grid is a set of independent pointwise estimates
# --------------------------------------------------------------------------- #

_Q_COARSE = np.linspace(-4.0, 4.0, 9)
_Q_FINE = np.linspace(-4.0, 4.0, 17)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("select_range", [False, True])
def test_refining_the_q_grid_cannot_move_the_shared_exponents(
    name: str, select_range: bool
) -> None:
    r"""One regression per :math:`q`: doubling the grid must change nothing.

    :math:`\hat\zeta(q)` is the slope of a single row of the scaling surface, so
    the value at :math:`q=2` cannot depend on whether :math:`q=1.5` was also
    requested. Anything that couples the rows --- a shared normalisation, a
    reference row chosen by position rather than by value, a per-grid
    renormalisation of the weights --- would show up here as a grid-dependent
    exponent, which is the sort of defect that makes two analyses of the same
    data disagree for no visible reason. Run with ``select_range`` on as well,
    because the scaling window is chosen on one row and imposed on all of them.
    """
    fn = ESTIMATORS[name]
    x = fbm_path(seed=5)
    coarse = fn(x, q=_Q_COARSE, select_range=select_range)
    fine = fn(x, q=_Q_FINE, select_range=select_range)
    shared = np.searchsorted(fine.q, _Q_COARSE)
    np.testing.assert_array_equal(
        coarse.zeta, fine.zeta[shared], err_msg=f"{name}: exponents depend on the q grid"
    )


# --------------------------------------------------------------------------- #
# (c) Refit consistency: the stored surface is the same computation
# --------------------------------------------------------------------------- #

_REFIT: dict[str, tuple[Callable[..., Any], Callable[..., ScalingExponents], dict[str, Any]]] = {
    "mfdfa": (mfdfa_structure, mfdfa_from_structure, {"integrate": False}),
    "mfdma": (mfdma_structure, mfdma_exponents_from_structure, {}),
    "structure_function": (mfc_exponents_structure, mfc_from_structure, {}),
    "wavelet_leaders": (wavelet_leaders_structure, leaders_from_structure, {}),
}


@pytest.mark.parametrize("name", sorted(_REFIT))
def test_refit_from_the_stored_surface_reproduces_the_direct_exponents(name: str) -> None:
    """``<estimator>_structure`` + refit is advertised as free and identical.

    The whole point of persisting ``StructureFunctions`` is that the scaling
    range can be revised without recomputing the moments. If the two paths
    disagree at all, one of them is not the documented estimator, and an audit
    trail rebuilt from the stored surface would not reproduce the published
    number. Bit equality is the right bar: the arithmetic is meant to be the
    same arithmetic.
    """
    structure_fn, refit_fn, kwargs = _REFIT[name]
    x = fbm_path(seed=7)
    direct = ESTIMATORS[name](x, q=Q).zeta
    refit = refit_fn(structure_fn(x, q=Q, **kwargs)).zeta
    np.testing.assert_array_equal(refit, direct, err_msg=f"{name}: refit != direct")


def test_leader_refit_over_a_subrange_equals_a_direct_run_on_those_octaves() -> None:
    """Restricting the stored surface must equal recomputing on the same octaves.

    ``exponents_from_structure(..., scale_range=...)`` is the only refit path in
    the library that takes a window. If it disagreed with ``wavelet_leaders``
    run over exactly those octaves, the cheap path and the honest path would be
    different estimators wearing the same name.
    """
    x = fbm_path(seed=9, n=1 << 14)
    refit = leaders_from_structure(wavelet_leaders_structure(x, q=Q), scale_range=(8.0, 128.0)).zeta
    direct = wavelet_leaders(x, q=Q, j_min=3, j_max=7).zeta
    np.testing.assert_array_equal(refit, direct)


# --------------------------------------------------------------------------- #
# (d) Time reversal
# --------------------------------------------------------------------------- #


def _reversal_gap(fn: Estimator, x: FloatArray, q: FloatArray, **kwargs: Any) -> float:
    forward = fn(x, q=q, **kwargs).zeta
    backward = fn(np.ascontiguousarray(x[::-1]), q=q, **kwargs).zeta
    return float(np.max(np.abs(forward - backward)))


@pytest.mark.parametrize("signal", ["fbm", "cascade"])
def test_mfdfa_is_exactly_time_reversal_invariant(signal: str) -> None:
    """MF-DFA tiles the profile from both ends, so reversal permutes its windows.

    The forward sweep of the reversed series is the reversed backward sweep of
    the original, and least-squares polynomial detrending commutes with
    reversing a window, so the multiset of :math:`F^2(v,s)` is identical and
    :math:`\\hat\\zeta` must agree to round-off. This is the sharpest available
    check that the second sweep is present and correctly indexed: dropping it,
    or tiling the tail from the wrong offset, breaks the equality immediately.
    """
    x = fbm_path() if signal == "fbm" else cascade_path()
    gap = _reversal_gap(ESTIMATORS["mfdfa"], x, Q)
    assert gap < 1.0e-11, f"MF-DFA time-reversal gap {gap:.3e}"


@pytest.mark.parametrize("signal", ["fbm", "cascade"])
def test_stride_one_increment_estimators_are_exactly_time_reversal_invariant(
    signal: str,
) -> None:
    """With overlapping increments the reversed sample is a permutation.

    :math:`|X(t+\\delta)-X(t)|` over every :math:`t` is the same multiset read
    either way, so GHE (``overlapping=True`` by default) and the structure
    function with ``overlapping=True`` must be invariant to round-off. The
    *default* structure function uses stride-:math:`\\delta` increments and is
    not --- see ``test_disjoint_increments_cost_time_reversal_symmetry``, which
    measures the price.
    """
    x = fbm_path() if signal == "fbm" else cascade_path()
    ghe_gap = _reversal_gap(ESTIMATORS["ghe"], x, Q)
    assert ghe_gap < 1.0e-11, f"GHE time-reversal gap {ghe_gap:.3e}"
    mfc_gap = _reversal_gap(mfc_exponents, x, Q_POSITIVE, overlapping=True)
    assert mfc_gap < 1.0e-11, f"structure function (overlapping) gap {mfc_gap:.3e}"


def test_disjoint_increments_cost_time_reversal_symmetry() -> None:
    r"""Quantify the asymmetry the default stride-:math:`\delta` tiling buys.

    ``overlapping=False`` starts its tiling at index 0 and drops the
    :math:`(n-1) \bmod \delta` samples at the tail; reversal drops the head
    instead, so the two runs average different sub-samples. On positive
    :math:`q` the effect is a few hundredths; on :math:`q<0` a single
    near-cancelling increment entering or leaving the sample moves
    :math:`\hat\zeta(-4)` by more than one, which is why the negative branch of
    this estimator is a diagnostic and not a measurement. Both bounds are
    measured, not derived: they exist so a future change that makes the
    asymmetry *worse* is visible.
    """
    x = fbm_path()
    positive = _reversal_gap(mfc_exponents, x, Q_POSITIVE)
    assert 1.0e-6 < positive < 0.10, f"positive-q asymmetry {positive:.4f}"

    q_negative = np.array([-4.0, -3.0, -2.0, -1.0])
    negative = _reversal_gap(mfc_exponents, x, q_negative)
    assert negative > 0.5, (
        f"the q<0 branch of the disjoint structure function was reversal-stable "
        f"({negative:.4f}); either the estimator changed or the guard is stale"
    )


# --------------------------------------------------------------------------- #
# (e) The reported scaling range must be the one that was fitted
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("select_range", [False, True])
def test_the_reported_scaling_range_is_the_range_actually_fitted(
    name: str, select_range: bool
) -> None:
    """A slope quoted over the wrong window is the library's stated failure mode.

    ``ScalingRange`` is what a reader uses to judge whether a power law was
    measured over a decade or over five points, and it is what
    ``decades`` and the manifest report. Cross-check it against the ``log_x``
    the regression actually consumed: if selection narrowed the window but the
    record kept the full grid (or vice versa), every published ``decades`` from
    this estimator is wrong.
    """
    exponents = ESTIMATORS[name](fbm_path(seed=13, n=1 << 14), q=Q, select_range=select_range)
    scaling_range = exponents.scaling_range
    assert scaling_range is not None
    fitted = np.exp(exponents.fits[0].log_x)
    assert fitted.size == scaling_range.n_scales
    assert float(fitted.min()) == pytest.approx(scaling_range.s_min, rel=1e-12)
    assert float(fitted.max()) == pytest.approx(scaling_range.s_max, rel=1e-12)
    if scaling_range.mask:
        assert int(np.sum(scaling_range.mask)) == scaling_range.n_scales


def test_selection_label_names_the_moment_order_not_the_row_index() -> None:
    r"""``selection`` is audit metadata; it must not report an array index as a q.

    ``mfdma`` and ``wtmm`` write ``auto-r2@q=2``. ``wavelet_leaders`` wrote
    ``auto-r2@q={reference}`` with ``reference`` the *row number*, so on the
    library default grid ``linspace(-5, 5, 41)`` the manifest claimed the
    scaling range had been selected on :math:`q=28` --- a moment order 5.6
    times outside the grid it was computed on. The string lands in
    ``provenance.diagnostics['scaling_range']`` and in every stored manifest,
    so a reader reconstructing the analysis would reconstruct the wrong one.
    """
    x = fbm_path(seed=3, n=1 << 14)
    for estimator in (mfdma, wavelet_leaders, wtmm):
        exponents = estimator(x, select_range=True)
        assert exponents.scaling_range is not None
        label = exponents.scaling_range.selection
        if "@q=" not in label:
            continue
        reported = float(label.split("@q=")[1])
        assert float(exponents.q.min()) <= reported <= float(exponents.q.max()), (
            f"{estimator.__name__} reports selection on q={reported:g}, outside the grid "
            f"[{exponents.q.min():g}, {exponents.q.max():g}]"
        )
        assert reported == pytest.approx(2.0, abs=0.25), (
            f"{estimator.__name__}: the range is documented as selected on q ~ 2, "
            f"the label says q={reported:g}"
        )


# --------------------------------------------------------------------------- #
# (f) Scaling-range selection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", NAMES)
def test_range_selection_barely_moves_the_positive_q_exponents_on_clean_fbm(
    name: str,
) -> None:
    """On a pure power law there is no window to find, so selection must be a no-op.

    A selector that reliably shortens the range on clean fBm is not finding a
    scaling regime, it is chasing noise, and every width it reports afterwards
    is a property of the selector. Only :math:`q>0` is checked: the negative
    branch of WTMM and of the increment estimators is a documented pathology on
    a monofractal and moves for reasons that have nothing to do with selection.
    """
    fn = ESTIMATORS[name]
    x = fbm_path(seed=3, n=1 << 13)
    full = fn(x, q=Q_POSITIVE, select_range=False)
    selected = fn(x, q=Q_POSITIVE, select_range=True)
    gap = float(np.max(np.abs(full.zeta - selected.zeta)))
    assert gap < 0.12, f"{name}: select_range moved zeta(q>0) by {gap:.4f} on clean fBm"


def test_range_selection_on_a_deliberate_crossover_is_reported_not_hidden() -> None:
    r"""Glue :math:`H=0.3` below :math:`s=200` to :math:`H=0.8` above it.

    A fit across the break returns an exponent that exists nowhere in the
    series. The ``r2`` selector takes the *widest* window clearing
    :math:`R^2\ge0.995` and the full 23-scale MF-DMA window clears it at
    :math:`R^2=0.9957`, so selection does **not** split this crossover: it
    returns :math:`h(2)\approx0.6`, between the two regimes. That is the
    documented behaviour of the criterion, and the defence against it is that
    the window is reported. So the contract this test pins is the weaker but
    checkable one: whatever window is chosen must be recorded, must be a
    contiguous sub-window of the grid, and MF-DFA and the wavelet leaders ---
    which do split it --- must land on the small-scale regime rather than
    between the two.
    """
    x = crossover_path(1 << 15, h_small=0.3, h_large=0.8, s_break=200, seed=7)
    for name in ("mfdfa", "mfdma", "structure_function", "wavelet_leaders"):
        selected = ESTIMATORS[name](x, q=Q_POSITIVE, select_range=True)
        chosen = selected.scaling_range
        assert chosen is not None
        assert chosen.selection.startswith("auto"), f"{name}: selection not recorded"
        assert chosen.n_scales >= 5
        assert chosen.s_min < chosen.s_max

    for name in ("mfdfa", "wavelet_leaders"):
        selected = ESTIMATORS[name](x, q=Q_POSITIVE, select_range=True)
        assert selected.scaling_range is not None
        assert selected.scaling_range.s_max <= 600.0, (
            f"{name}: selection kept scales past the s=200 break "
            f"(s_max={selected.scaling_range.s_max:g})"
        )
        assert abs(selected.h2 - 0.3) < 0.15, (
            f"{name}: h(2)={selected.h2:.4f} after selection, neither 0.3 nor 0.8"
        )


# --------------------------------------------------------------------------- #
# (g) Sub-sampling stability
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", NAMES)
def test_the_two_halves_agree_with_the_whole_on_a_stationary_process(name: str) -> None:
    r"""fBm has stationary increments, so any window estimates the same :math:`H`.

    A split-half disagreement much larger than the sampling scatter would mean
    the estimator is reading something about *where* in the record it is --- a
    boundary term, a scale grid that depends on the offset, an accumulator that
    drifts. The bound is loose on purpose: the point is the shape of the
    failure, not a tight coverage claim. The reported ``zeta_stderr`` is *not*
    used as the yardstick, because the library documents it as regression
    scatter that understates the true spread several-fold.
    """
    fn = ESTIMATORS[name]
    x = fbm_path(seed=21, n=1 << 15)
    half = x.size // 2
    whole = fn(x, q=Q_POSITIVE).h2
    first = fn(np.ascontiguousarray(x[:half]), q=Q_POSITIVE).h2
    second = fn(np.ascontiguousarray(x[half:]), q=Q_POSITIVE).h2
    for label, value in (("first", first), ("second", second)):
        assert abs(value - whole) < 0.10, (
            f"{name}: h(2) on the {label} half is {value:.4f} against {whole:.4f} on the whole"
        )


# --------------------------------------------------------------------------- #
# (h) Reproducibility of the content address
# --------------------------------------------------------------------------- #

_CHILD = """
import json
import numpy as np
from mmar.models.fbm import fbm
from mmar.estimators.mfdfa import mfdfa
from mmar.estimators.mfdma import mfdma
from mmar.estimators.structure_function import mfc_exponents
from mmar.estimators.wavelet_leaders import wavelet_leaders
from mmar.estimators.ghe import generalised_hurst
from mmar.estimators.wtmm import wtmm
from mmar.analysis import analyze

x = fbm(2048, 0.7, seed=11)
q = np.linspace(-3.0, 3.0, 13)
roots = {
    "mfdfa": mfdfa(x, q=q, integrate=False).provenance.root,
    "mfdma": mfdma(x, q=q).provenance.root,
    "structure_function": mfc_exponents(x, q=q).provenance.root,
    "wavelet_leaders": wavelet_leaders(x, q=q).provenance.root,
    "ghe": generalised_hurst(x, q=q).provenance.root,
    "wtmm": wtmm(x, q=q).provenance.root,
    "analyze": analyze(x, estimator="mfdfa", q=q).provenance.root,
}
print(json.dumps(roots))
"""


def test_provenance_roots_are_reproduced_in_a_fresh_interpreter() -> None:
    """A content address that carries session state is not a content address.

    ``Provenance.root`` is documented as a pure function of estimator, params,
    inputs, seed and environment. Anything session-local leaking into it --- a
    dict iteration order, ``id()``, a default ``SeedSequence`` minted from OS
    entropy, a wall-clock field --- would make two identical computations
    unrecognisable as identical, which is exactly the guarantee the audit chain
    rests on. The child process is launched with a different ``PYTHONHASHSEED``
    so that any string-hash dependence shows up too.
    """
    import json
    import os

    in_process = json.loads(
        subprocess.run(
            [sys.executable, "-c", _CHILD],
            capture_output=True,
            check=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": "0"},
        ).stdout
    )
    fresh = json.loads(
        subprocess.run(
            [sys.executable, "-c", _CHILD],
            capture_output=True,
            check=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": "98765"},
        ).stdout
    )
    assert in_process == fresh, "provenance roots differ between processes"

    x = fbm(2048, 0.7, seed=11)
    here = mfdfa(x, q=np.linspace(-3.0, 3.0, 13), integrate=False).provenance
    assert here is not None
    assert here.root == in_process["mfdfa"]


# --------------------------------------------------------------------------- #
# (i) Cost model
# --------------------------------------------------------------------------- #


def _peak_bytes(fn: Callable[[], Any]) -> int:
    tracemalloc.start()
    try:
        fn()
        return int(tracemalloc.get_traced_memory()[1])
    finally:
        tracemalloc.stop()


def test_ghe_peak_memory_does_not_scale_with_the_q_grid() -> None:
    r"""GHE materialised the whole :math:`(n_q \times n)` tilt matrix.

    ``_log_moments`` evaluated ``logsumexp(q[:, None] * log|v|[None, :], axis=1)``
    at every lag. ``logsumexp`` keeps roughly six copies of its argument alive,
    so the peak was :math:`6.2\,n_q n` float64 --- **2.14 GB** on
    :math:`n=2^{20}` with the default 41-point grid, against 53 MB for MF-DFA on
    the same series and 8 MB for the series itself. Reducing one :math:`q` at a
    time is bit-for-bit identical (verified against the vectorised form on fBm
    and on a cascade, for every combination of ``normalise``/``overlapping``/
    ``average_tau_max``) and no slower, so the matrix bought nothing.

    The bound is stated relative to the tilt matrix, not as an absolute figure:
    anything below it proves the matrix is not being formed.
    """
    n = 1 << 16
    q = np.linspace(-5.0, 5.0, 41)
    x = fbm_path(seed=1, n=n, hurst=0.6)
    peak = _peak_bytes(lambda: generalised_hurst(x, q=q))
    tilt_matrix = 8 * q.size * n
    assert peak < 0.5 * tilt_matrix, (
        f"GHE peaked at {peak / 1e6:.1f} MB = {peak / tilt_matrix:.1f} x the "
        f"(n_q x n) tilt matrix; it is forming the matrix again"
    )


@pytest.mark.parametrize("name", NAMES)
def test_one_more_q_point_costs_less_than_one_copy_of_the_series(name: str) -> None:
    r"""Bound the *marginal* peak allocation per extra moment order.

    Every estimator reduces over samples at each ``(q, scale)`` cell, so the
    :math:`q` grid should cost :math:`O(n_q)` scalars, not :math:`O(n_q n)`
    arrays. Measured marginal cost between an 11-point and an 81-point grid at
    :math:`n=2^{16}`, in units of one float64 copy of the series::

        mfdfa 0.00   structure_function 0.00   ghe 0.00
        wtmm  0.35   wavelet_leaders    0.75   mfdma 0.77

    MF-DMA, the wavelet leaders and WTMM tilt an array of segment or
    coefficient log-magnitudes for all :math:`q` at once, which is bounded by
    the finest scale's segment count and stays under one series-copy per
    :math:`q`. GHE used to tilt the *full-length* increment array at every lag
    and scored 6.2 --- 2.14 GB on :math:`n=2^{20}` with the default grid. The
    threshold sits between the two, so the class of defect cannot come back
    anywhere.
    """
    n = 1 << 16
    x = fbm_path(seed=2, n=n, hurst=0.6)
    fn = ESTIMATORS[name]
    small, large = 11, 81
    peak_small = _peak_bytes(functools.partial(fn, x, q=np.linspace(-5.0, 5.0, small)))
    peak_large = _peak_bytes(functools.partial(fn, x, q=np.linspace(-5.0, 5.0, large)))
    marginal = (peak_large - peak_small) / ((large - small) * 8 * n)
    assert marginal < 1.5, (
        f"{name}: every extra q point costs {marginal:.2f} copies of the series "
        f"({peak_small / 1e6:.1f} MB at n_q={small} -> {peak_large / 1e6:.1f} MB at n_q={large})"
    )


@pytest.mark.slow
def test_runtime_and_memory_grow_at_most_like_n_log_n() -> None:
    """Fit the empirical cost exponent over ``n = 2**12, 2**14, 2**16``.

    Anything quadratic --- a Python loop over samples, a pairwise maxima-line
    match, an accidental ``O(n^2)`` broadcast --- shows up as an exponent well
    above 1. WTMM is the one that could plausibly go quadratic, because it
    chains maxima across scales; the measured exponents on this machine are all
    below 0.5 (fixed overheads still dominate at these sizes) and memory is
    linear for every estimator. The bound is 1.35, which admits :math:`n\\log n`
    with room to spare and rejects :math:`n^{1.5}`.
    """
    sizes = (1 << 12, 1 << 14, 1 << 16)
    q = np.linspace(-4.0, 4.0, 17)
    paths = {n: fbm_path(seed=1, n=n, hurst=0.6) for n in sizes}
    log_n = np.log(np.asarray(sizes, dtype=np.float64))

    report: list[str] = []
    for name in NAMES:
        fn = ESTIMATORS[name]
        fn(paths[sizes[0]], q=q)  # warm the lazy imports out of the measurement
        times, peaks = [], []
        for n in sizes:
            start = time.perf_counter()
            fn(paths[n], q=q)
            times.append(max(time.perf_counter() - start, 1e-6))
            peaks.append(float(_peak_bytes(functools.partial(fn, paths[n], q=q))))
        t_exp = float(np.polyfit(log_n, np.log(times), 1)[0])
        m_exp = float(np.polyfit(log_n, np.log(peaks), 1)[0])
        report.append(f"{name:20s} runtime n^{t_exp:+.2f}  memory n^{m_exp:+.2f}")
        assert t_exp < 1.35, f"{name}: runtime exponent {t_exp:.2f}, times={times}"
        assert m_exp < 1.15, f"{name}: memory exponent {m_exp:.2f}, peaks={peaks}"
    print("\n" + "\n".join(report))
