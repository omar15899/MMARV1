"""Aliasing, shared state and frozen things that are not frozen.

Four attacks, in decreasing order of what they cost a user:

1. an unseeded run writes a seed record into its manifest that regenerates a
   *different* stream, and makes two identical deterministic analyses carry two
   different content addresses;
2. the caller's moment grid is stored by reference inside the result, so a
   later ``q[0] = ...`` rewrites a curve that has already been published;
3. ``frozen=True`` stops attribute rebinding but not array mutation, so a
   result can be edited in place while its provenance root stands still;
4. ``Provenance.params`` is typed ``Mapping`` but is a live ``dict``.

(1) and (2) are fixed here; (3) and (4) are characterised so that changing them
has to be deliberate.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from mmar.analysis import analyze
from mmar.calibration.mmar_calibration import CalibrationResult
from mmar.core.legendre import spectrum_from_exponents
from mmar.core.provenance import (
    EnvFingerprint,
    InputRef,
    Provenance,
    SeedInfo,
    array_digest,
    new_provenance,
)
from mmar.core.rng import SeedLike, make_rng, seed_info
from mmar.core.types import (
    FloatArray,
    HolderSpectrum,
    LogCumulants,
    LogLogFit,
    MultifractalReport,
    PointwiseHolder,
    ScalingExponents,
    ScalingRange,
    SpectrumDescriptors,
    StructureFunctions,
)
from mmar.core.validate import validate_q_grid
from mmar.estimators.mfdfa import mfdfa
from mmar.estimators.structure_function import mfc_exponents
from mmar.models.fbm import fbm, fbm_with_provenance

Q = np.linspace(-4.0, 4.0, 17)


def _series(n: int = 2048) -> FloatArray:
    return np.ascontiguousarray(fbm(n, H=0.6, seed=20260727), dtype=np.float64)


def _exponents() -> ScalingExponents:
    return ScalingExponents(q=np.linspace(-2.0, 2.0, 5), zeta=np.linspace(-1.0, 1.0, 5))


def _spectrum() -> HolderSpectrum:
    return HolderSpectrum(alpha=np.linspace(0.3, 0.7, 5), f_alpha=np.linspace(0.0, 1.0, 5))


def _provenance() -> Provenance:
    return new_provenance("probe", params={"a": 1}, inputs=(InputRef.from_array("x", Q),))


FROZEN_CASES: list[tuple[Any, str, Any]] = [
    (_exponents(), "zeta", np.zeros(5)),
    (_spectrum(), "f_alpha", np.zeros(5)),
    (
        LogLogFit(
            slope=1.0,
            intercept=0.0,
            slope_stderr=0.1,
            intercept_stderr=0.1,
            slope_ci=(0.9, 1.1),
            r_squared=0.99,
            n_points=8,
        ),
        "slope",
        2.0,
    ),
    (
        StructureFunctions(
            q=np.linspace(-1.0, 1.0, 3), scales=np.array([2.0, 4.0]), values=np.ones((3, 2))
        ),
        "values",
        np.zeros((3, 2)),
    ),
    (LogCumulants(c=np.array([0.5, -0.02])), "c", np.zeros(2)),
    (PointwiseHolder(t=np.arange(4.0), h=np.full(4, 0.5)), "h", np.zeros(4)),
    (MultifractalReport(exponents=_exponents(), spectrum=_spectrum()), "spectrum", _spectrum()),
    (ScalingRange(s_min=2.0, s_max=64.0, n_scales=6), "s_max", 1.0),
    (
        SpectrumDescriptors(
            alpha_min=0.3,
            alpha_max=0.7,
            alpha_0=0.5,
            width=0.4,
            f_max=1.0,
            asymmetry=1.0,
            curvature=-1.0,
        ),
        "width",
        99.0,
    ),
    (_provenance(), "estimator", "forged"),
    (InputRef.from_array("x", Q), "digest", "0" * 64),
    (SeedInfo(entropy=7), "entropy", 8),
    (EnvFingerprint.capture(), "numpy", "0.0.0"),
    (
        CalibrationResult(
            hurst=0.55,
            alpha_0=0.6,
            lam=1.1,
            sigma2=0.29,
            base=2.0,
            admissible=True,
            diagnosis=(),
            q_critical=10.0,
            width_implied=0.5,
            width_observed=0.5,
        ),
        "admissible",
        False,
    ),
]


@pytest.mark.parametrize(
    ("obj", "attr", "value"), FROZEN_CASES, ids=[type(c[0]).__name__ for c in FROZEN_CASES]
)
def test_every_result_type_refuses_attribute_rebinding(obj: Any, attr: str, value: Any) -> None:
    """Rebind one field on each declared-frozen result type.

    The boundary the next test probes from the other side: the dataclasses stop
    a field being *replaced*, which is the only part of immutability the
    declaration buys. A type that quietly lost ``frozen=True`` would let a
    caller retag an estimator name or a digest on a stored result.
    """
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, attr, value)


def test_frozen_results_do_not_protect_the_arrays_they_carry() -> None:
    """Mutate ``exponents.zeta`` in place and re-read the headline numbers.

    ``frozen=True`` is shallow. Every reported quantity — :math:`h(2)`, the
    spectrum, its width — is derived from ``zeta`` on demand, so overwriting the
    array rewrites the whole analysis, while ``provenance.root`` (a digest over
    the *inputs and settings*, never over the produced curve) does not move. The
    result is an audit record that certifies a curve which no longer exists,
    which is precisely the failure mode content addressing is sold as
    preventing. Characterised, not fixed: marking the arrays read-only is a
    library-wide decision.
    """
    e = mfdfa(_series(), q=Q, seed=11)
    assert e.provenance is not None
    root_before = e.provenance.root
    h2_before = e.h2
    width_before = spectrum_from_exponents(e).descriptors().width

    assert e.zeta.flags.writeable, "arrays inside a frozen result are still writeable"
    e.zeta[:] = e.zeta * 2.0

    assert e.h2 == pytest.approx(2.0 * h2_before)
    assert spectrum_from_exponents(e).descriptors().width != pytest.approx(width_before)
    assert e.provenance.root == root_before, (
        "the content address did not move although the reported curve did"
    )


def test_the_provenance_record_and_the_report_tests_are_live_dicts() -> None:
    """Assign into ``prov.params`` and into ``report.tests`` after construction.

    Both are typed :class:`collections.abc.Mapping` but are plain ``dict``. Since
    ``root`` is a property recomputed from ``params`` on every access, editing
    the dict silently re-addresses a record that has already been written to
    disk under the old digest; and ``report.summary()`` will happily report an
    injected test result. Characterised, not fixed.
    """
    prov = new_provenance("probe", params={"scale_max": 64})
    root_before = prov.root
    assert isinstance(prov.params, dict)
    prov.params["scale_max"] = 8  # type: ignore[index]
    assert prov.root != root_before

    report = MultifractalReport(exponents=_exponents(), spectrum=_spectrum(), tests={"defect": 0.0})
    assert isinstance(report.tests, dict)
    report.tests["defect"] = -1.0  # type: ignore[index]
    assert report.summary()["tests"] == {"defect": -1.0}


def test_the_moment_grid_is_copied_out_of_the_caller_s_hands() -> None:
    """Hand an estimator a ``q`` array, then overwrite it and re-read the result.

    ``validate_q_grid`` used to hand back ``np.ascontiguousarray(q)``, which for
    an already-contiguous float64 input is the caller's own array. That object
    was then stored verbatim as ``ScalingExponents.q`` *and* inside the
    provenance params, so a caller reusing one grid buffer across a sweep would
    retroactively rewrite the abscissa of every curve already computed — and
    with it the content address that was supposed to pin them apart.
    """
    q = np.linspace(-4.0, 4.0, 17)
    grid = validate_q_grid(q)
    assert not np.shares_memory(grid, q)

    for fn in (mfdfa, mfc_exponents):
        e = fn(_series(), q=q, seed=11)
        assert e.provenance is not None
        assert not np.shares_memory(e.q, q), f"{fn.__name__} aliased the caller's q grid"
        root_before, q_before = e.provenance.root, e.q.copy()
        q[0] = -99.0
        np.testing.assert_array_equal(e.q, q_before)
        assert e.provenance.root == root_before
        q[0] = -4.0


@pytest.mark.parametrize("seed", [None, "generator"])
def test_a_non_recoverable_stream_is_recorded_as_absent(seed: str | None) -> None:
    """Ask for the seed record of the two streams that cannot be replayed.

    A live :class:`~numpy.random.Generator` was already handled. ``seed=None``
    was not: ``seed_info(None)`` minted a *fresh* ``SeedSequence`` from OS
    entropy and returned it, but every consumer builds its own — so the number
    written into the manifest described a draw that never happened. Recording an
    unusable 128-bit entropy is strictly worse than recording nothing, because
    it reads as a reproducibility guarantee.
    """
    value: SeedLike = make_rng(5) if seed == "generator" else None
    if seed == "generator":
        assert make_rng(value) is value
    assert seed_info(value) is None


def test_an_unseeded_path_manifest_does_not_claim_a_seed_that_replays_elsewhere() -> None:
    """Generate an unseeded fBm with provenance, then take its record at its word.

    ``fbm_with_provenance`` documents that "the path can be regenerated from the
    manifest alone". Feeding the recorded entropy back through :func:`fbm` used
    to produce a completely different path whose digest did not match the
    ``output_digest`` diagnostic sitting in the same record — a false claim, not
    a missing one. Either the record is silent or it must replay.
    """
    path, prov = fbm_with_provenance(512, 0.6, seed=None)
    if prov.seed is None:
        return
    entropy = prov.seed.entropy
    assert isinstance(entropy, int)
    regenerated = fbm(512, 0.6, seed=entropy)
    assert array_digest(regenerated) == prov.diagnostics["output_digest"], (
        "the recorded seed regenerates a different path from the one it accompanies"
    )
    np.testing.assert_array_equal(regenerated, path)


def test_analyze_reproduces_its_root_on_an_unseeded_deterministic_rerun() -> None:
    """Run the same deterministic analysis twice with the default ``seed=None``.

    The estimators guard this case explicitly (``_seed_record``); ``analyze``
    called :func:`seed_info` unguarded, so the OS entropy drawn per call landed
    in the identifying part of the record. Two byte-identical analyses of the
    same series then carried two different content addresses — the library's
    central promise ("two runs share a root iff they were the same computation")
    inverted, and a de-duplicating store made useless.
    """
    x = _series()
    first = analyze(x, estimator="mfdfa", q=Q, cumulants=False)
    second = analyze(x, estimator="mfdfa", q=Q, cumulants=False)

    assert first.provenance is not None and second.provenance is not None
    np.testing.assert_array_equal(first.exponents.zeta, second.exponents.zeta)
    assert math.isfinite(first.exponents.h2)
    assert first.provenance.root == second.provenance.root
