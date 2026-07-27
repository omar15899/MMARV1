"""Aliasing, shared state, and frozen things that are not frozen.

The attacks, in decreasing order of what they cost a user:

1. an unseeded run writes a seed record into its manifest that regenerates a
   *different* stream, and makes two identical deterministic analyses carry two
   different content addresses;
2. building a result out of the caller's arrays takes them over --- either by
   storing them by reference (a later ``q[0] = ...`` rewrites a published
   curve) or, the other way round, by revoking the caller's own write access to
   a buffer it still owns;
3. ``frozen=True`` stops attribute rebinding but not array mutation, so a
   result can be edited in place while its provenance root stands still;
4. two computations that differ in a setting the caller chose share one content
   address, because that setting never reached the digest;
5. ``Provenance.params`` is typed ``Mapping`` but the object the record keeps is
   a live ``dict``, so writing into it re-addresses a record already on disk.

(1)--(4) are fixed; (5) is characterised, so that changing it has to be
deliberate.
"""

from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mmar.analysis import analyze
from mmar.calibration.mmar_calibration import CalibrationResult
from mmar.core.legendre import spectrum_from_exponents
from mmar.core.provenance import (
    ArtifactStore,
    EnvFingerprint,
    InputRef,
    Provenance,
    SeedInfo,
    array_digest,
    new_provenance,
)
from mmar.core.rng import SeedLike, make_rng, seed_info, spawn_rngs
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
from mmar.estimators.ghe import generalised_hurst
from mmar.estimators.mfdfa import mfdfa
from mmar.estimators.mfdma import mfdma
from mmar.estimators.structure_function import mfc_exponents
from mmar.models.fbm import fbm, fbm_with_provenance

Q = np.linspace(-4.0, 4.0, 17)
_SERIES: FloatArray = np.ascontiguousarray(fbm(2048, H=0.6, seed=20260727), dtype=np.float64)


def _series() -> FloatArray:
    return _SERIES.copy()


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

    One side of immutability: the dataclasses must stop a field being
    *replaced*. A type that quietly lost ``frozen=True`` would let a caller
    retag an estimator name or a digest on a stored result.
    """
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, attr, value)


def test_the_arrays_a_result_exposes_cannot_be_written_through_it() -> None:
    """Mutate ``exponents.zeta`` in place and re-read the headline numbers.

    The other side, and the one ``frozen=True`` does *not* buy: every reported
    quantity --- :math:`h(2)`, the spectrum, its width --- is derived from
    ``zeta`` on demand, while ``provenance.root`` digests the inputs and
    settings and never the produced curve. So ``e.zeta[:] = 2 * e.zeta`` used to
    move ``h2`` from 0.70 to 1.40 with the content address unchanged: an audit
    record certifying a curve that no longer existed, which is the exact failure
    mode content addressing is sold as preventing.
    """
    e = mfdfa(_series(), q=Q, seed=11)
    assert e.provenance is not None
    h2_before, root_before = e.h2, e.provenance.root

    for arr in (e.zeta, e.q):
        assert not arr.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 999.0

    spectrum = spectrum_from_exponents(e)
    for arr in (spectrum.alpha, spectrum.f_alpha):
        with pytest.raises(ValueError, match="read-only"):
            arr[:] = 0.0

    assert e.structure is not None
    with pytest.raises(ValueError, match="read-only"):
        e.structure.values[0, 0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        e.structure.row(1.0)[0] = 0.0

    assert e.h2 == h2_before
    assert e.provenance.root == root_before


def test_building_a_result_does_not_revoke_the_caller_s_write_access() -> None:
    """Construct results from caller-owned buffers, then write to the buffers.

    Making the result's arrays read-only must be done through a *view*: setting
    ``writeable = False`` on the array the caller handed in reaches back through
    the reference the caller still holds and confiscates it. The symptom is a
    ``ValueError: assignment destination is read-only`` raised nowhere near the
    constructor --- on a scratch buffer reused across a sweep, the second
    iteration dies with a message that names neither the buffer nor the result
    that froze it, and code that filled the buffer *after* constructing the
    result loses the write silently under a try/except.
    """
    q, zeta = np.linspace(-2.0, 2.0, 5), np.linspace(-1.2, 1.2, 5)
    ScalingExponents(q=q, zeta=zeta)
    assert q.flags.writeable and zeta.flags.writeable
    zeta[0] = -1.5

    alpha, f = np.linspace(0.3, 0.7, 5), np.linspace(0.0, 1.0, 5)
    HolderSpectrum(alpha=alpha, f_alpha=f)
    assert alpha.flags.writeable and f.flags.writeable

    t, h = np.arange(4.0), np.full(4, 0.5)
    PointwiseHolder(t=t, h=h)
    assert t.flags.writeable and h.flags.writeable

    scratch = np.empty(5)
    widths = []
    for i in range(3):
        scratch[:] = np.linspace(-1.0, 1.0, 5) * (i + 1.0)
        widths.append(ScalingExponents(q=np.linspace(-2.0, 2.0, 5), zeta=scratch).zeta[-1])
    assert widths == [1.0, 2.0, 3.0]


def test_the_spectrum_does_not_take_over_the_caller_s_alpha_grid() -> None:
    """Pass an explicit ``alpha_grid``, then keep using the array.

    ``legendre_fenchel`` returned the caller's grid object as
    ``HolderSpectrum.alpha``. Two consequences, and both are silent: a caller
    that reuses one grid buffer across a sweep of exponent sets rewrites the
    abscissa of every spectrum already built (their ``descriptors().width`` and
    ``alpha_0`` move with it, while the provenance root does not), and once the
    result types freeze what they carry, the caller's own grid is confiscated by
    a function it only passed it to.
    """
    e = mfdfa(_series(), q=Q, seed=11)
    grid = np.linspace(0.2, 0.9, 201)
    spectrum = spectrum_from_exponents(e, alpha_grid=grid)

    assert not np.shares_memory(spectrum.alpha, grid)
    assert grid.flags.writeable
    a0_before = spectrum.alpha_0
    grid += 10.0
    assert spectrum.alpha_0 == a0_before
    assert float(spectrum.alpha.max()) < 5.0


def test_a_setting_that_moves_the_spectrum_moves_its_content_address() -> None:
    """Transform one exponent set twice, changing only a keyword.

    ``alpha_grid`` fixes the abscissa the spectrum is reported on --- hence its
    width, its apex and its truncation flag --- and ``clip_negative`` deletes
    every :math:`f<0` point from the curve. Neither reached the derived
    provenance params, so two demonstrably different spectra came back under one
    content address: a store keyed on the root keeps whichever was written
    first, and ``verify()`` blesses it.
    """
    e = mfdfa(_series(), q=Q, seed=11)
    grid = np.linspace(0.0, 3.0, 201)
    narrow = spectrum_from_exponents(e, alpha_grid=np.linspace(1.4, 1.9, 201))
    wide = spectrum_from_exponents(e, alpha_grid=grid)
    assert narrow.provenance is not None and wide.provenance is not None
    assert not np.array_equal(narrow.alpha, wide.alpha)
    assert narrow.descriptors().width != pytest.approx(wide.descriptors().width)
    assert narrow.provenance.root != wide.provenance.root

    clipped = spectrum_from_exponents(e, alpha_grid=grid, clip_negative=True)
    assert clipped.provenance is not None
    assert int((wide.f_alpha < 0.0).sum()) > 0
    assert int(np.isnan(clipped.f_alpha).sum()) == int((wide.f_alpha < 0.0).sum())
    assert clipped.provenance.root != wide.provenance.root


def test_two_runs_of_one_estimator_share_nothing_and_touch_nothing() -> None:
    """Run the same estimator twice on one input and cross-check the buffers.

    Three ways a result can stop being its own: sharing memory with the input
    (a later edit of the series rewrites a stored curve), sharing memory with
    the previous result (a module-level buffer reused across calls), or writing
    into the caller's series. The input is handed in read-only so that an
    in-place write is a failure rather than a silent edit.
    """
    x = _series()
    x.flags.writeable = False
    for fn in (mfdfa, mfdma, mfc_exponents, generalised_hurst):
        first, second = fn(x, q=Q, seed=11), fn(x, q=Q, seed=11)
        assert not np.shares_memory(first.zeta, x)
        assert not np.shares_memory(first.zeta, second.zeta)
        assert not np.shares_memory(first.q, second.q)
        np.testing.assert_array_equal(first.zeta, second.zeta)


def test_the_same_estimator_called_from_two_threads_returns_one_answer() -> None:
    """Four threads, one input array, one seed; compare the results bit for bit.

    A module-level scratch buffer or a mutable default would show up here as two
    threads reporting different exponents from the same call, and only under
    load. The roots must collapse to one for the same reason.
    """
    x = _series()
    x.flags.writeable = False
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: mfdfa(x, q=Q, seed=11), range(4)))
    reference = mfdfa(x, q=Q, seed=11)
    for r in results:
        np.testing.assert_array_equal(r.zeta, reference.zeta)
        assert r.provenance is not None and reference.provenance is not None
        assert r.provenance.root == reference.provenance.root


def test_the_moment_grid_is_copied_out_of_the_caller_s_hands() -> None:
    """Hand an estimator a ``q`` array, then overwrite it and re-read the result.

    ``validate_q_grid`` used to hand back ``np.ascontiguousarray(q)``, which for
    an already-contiguous float64 input is the caller's own array. That object
    was then stored verbatim as ``ScalingExponents.q`` *and* inside the
    provenance params, so a caller reusing one grid buffer across a sweep would
    retroactively rewrite the abscissa of every curve already computed --- and
    with it the content address that was supposed to pin them apart.
    """
    q = np.linspace(-4.0, 4.0, 17)
    grid = validate_q_grid(q)
    assert not np.shares_memory(grid, q)

    for fn in (mfdfa, mfc_exponents):
        e = fn(_series(), q=q, seed=11)
        assert e.provenance is not None
        assert not np.shares_memory(e.q, q), f"{fn.__name__} aliased the caller's q grid"
        root_before, q_before = e.provenance.root, np.array(e.q)
        q[0] = -99.0
        np.testing.assert_array_equal(e.q, q_before)
        assert e.provenance.root == root_before
        q[0] = -4.0


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


def test_the_copy_taken_of_the_params_reaches_the_nested_containers() -> None:
    """Put a list, an array and a nested dict inside ``params``, then mutate them.

    ``Provenance.__post_init__`` copies the mapping precisely so that the
    caller's dict cannot re-address a written record. A *shallow* copy stops the
    guarantee at the first level: the values that carry the numbers --- a scale
    ladder, a nested option dict, a ``q`` array --- stay shared, and editing one
    moves ``root`` exactly as replacing a key would, so a record already written
    to disk stops matching the address it was filed under.
    """
    scales = [8, 16, 32]
    grid = np.array([1.0, 2.0])
    nested = {"window": [1, 2]}
    prov = new_provenance("probe", params={"scales": scales, "grid": grid, "nested": nested})
    root = prov.root

    scales.append(64)
    grid[0] = 99.0
    nested["window"].append(3)
    assert prov.root == root
    assert prov.params["scales"] == [8, 16, 32]


def test_with_diagnostics_holds_the_root_still_and_derive_moves_it() -> None:
    """Attach diagnostics, then chain a child, and watch the address.

    The split is the whole point of the record: ``diagnostics`` is
    non-identifying (timings, counts, an output digest) and must not re-address
    a result, while ``derive`` starts a *different* computation and must. Were
    ``diagnostics`` identifying, the same analysis run on a slower machine would
    file under a different digest; were ``derive`` not, a spectrum would collide
    with the exponents it came from.
    """
    prov = _provenance()
    root = prov.root
    annotated = prov.with_diagnostics(elapsed_s=1.5, n_windows=12)
    assert annotated.root == root
    assert annotated.diagnostics == {"elapsed_s": 1.5, "n_windows": 12}
    assert prov.diagnostics == {}

    child = prov.derive("legendre", params={"method": "fenchel"})
    assert child.root != root
    assert child.root == prov.derive("legendre", params={"method": "fenchel"}).root
    assert child.inputs[0].digest == root
    assert child.derive("calibration").root != child.root


def test_the_store_persists_a_snapshot_and_refuses_a_rewritten_payload(tmp_path: Path) -> None:
    """Write a result, mutate the caller's arrays, then rewrite the payload on disk.

    Two failure modes with one shape. If ``put`` kept a reference rather than
    writing the bytes, a later edit of the caller's array would change what a
    verified artefact contains; and if ``verify`` only checked that a file
    exists, swapping the payload for a well-formed archive of different numbers
    --- not a corruption, a substitution --- would pass.
    """
    store = ArtifactStore(tmp_path / "cas")
    zeta = np.linspace(-2.0, 2.0, 16)
    prov = new_provenance("probe", params={"a": 1}, inputs=(InputRef.from_array("z", zeta),))
    root = store.put(prov, arrays={"zeta": zeta}, summary={"h2": 0.6})
    assert store.verify(root)

    zeta[0] = 999.0
    assert store.verify(root), "the store kept a reference instead of a snapshot"
    np.testing.assert_array_equal(store.get_arrays(root)["zeta"], np.linspace(-2.0, 2.0, 16))

    np.savez_compressed(store.arrays / f"{root}.npz", zeta=np.zeros(16))
    assert not store.verify(root)


def test_spawned_streams_are_independent_and_leave_the_parent_where_it_was() -> None:
    """Spawn children from a live generator and compare the parent's next draws.

    Replica seeding is the one place where a mistake is invisible in the output
    and fatal to the inference: children drawn from a *consumed* parent, or
    children that overlap, make a bootstrap ensemble correlated with itself and
    its confidence interval too narrow, with no diagnostic anywhere. Spawning
    must also leave the parent's own stream untouched, or the surrogate drawn
    after an ensemble depends on how many replicas were requested.
    """
    reference = make_rng(20260727).random(5)
    parent = make_rng(20260727)
    children = spawn_rngs(parent, 4)
    np.testing.assert_array_equal(parent.random(5), reference)

    draws = [c.random(64) for c in children]
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            assert not np.array_equal(draws[i], draws[j])
            assert abs(float(np.corrcoef(draws[i], draws[j])[0, 1])) < 0.35

    again = [c.random(64) for c in spawn_rngs(20260727, 4)]
    first = [c.random(64) for c in spawn_rngs(20260727, 4)]
    for a, b in zip(again, first, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("seed", [None, "generator"])
def test_a_non_recoverable_stream_is_recorded_as_absent(seed: str | None) -> None:
    """Ask for the seed record of the two streams that cannot be replayed.

    A live :class:`~numpy.random.Generator` was already handled. ``seed=None``
    was not: ``seed_info(None)`` minted a *fresh* ``SeedSequence`` from OS
    entropy and returned it, but every consumer builds its own --- so the number
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
    ``output_digest`` diagnostic sitting in the same record --- a false claim,
    not a missing one. Either the record is silent or it must replay.
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
    same series then carried two different content addresses --- the library's
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
