"""The traceability layer must make reproducibility a testable property."""

from __future__ import annotations

import numpy as np
import pytest

from mmar.core.legendre import spectrum_from_exponents
from mmar.core.provenance import ArtifactStore, InputRef, new_provenance
from mmar.core.types import ScalingExponents
from mmar.tracing import Run, diff_manifests, load_manifest, replay_plan


def _exponents(H: float = 0.7, *, npoints: int = 41, seed: int = 0) -> ScalingExponents:
    x = np.linspace(0.0, 1.0, 512)
    q = np.linspace(-5.0, 5.0, npoints)
    return ScalingExponents(
        q=q,
        zeta=q * H,
        d_support=1.0,
        estimator="synthetic",
        provenance=new_provenance(
            "synthetic",
            params={"H": H, "npoints": npoints, "seed": seed},
            inputs=[InputRef.from_array("x", x)],
        ),
    )


def test_root_is_a_pure_function_of_the_computation() -> None:
    a, b = _exponents(), _exponents()
    assert a.provenance is not None and b.provenance is not None
    assert a.provenance.root == b.provenance.root, "identical computations must share a root"
    identifying = a.provenance._identifying()
    assert "created_utc" not in identifying, "wall-clock time must not enter the digest"


def test_root_changes_when_any_parameter_changes() -> None:
    base = _exponents(H=0.7).provenance
    assert base is not None
    for changed in (_exponents(H=0.71), _exponents(npoints=42), _exponents(seed=1)):
        assert changed.provenance is not None
        assert changed.provenance.root != base.root


def test_root_changes_when_the_input_data_changes() -> None:
    q = np.linspace(-3.0, 3.0, 21)
    roots = []
    for n in (512, 513):
        prov = new_provenance(
            "synthetic", inputs=[InputRef.from_array("x", np.arange(n, dtype=float))]
        )
        roots.append(ScalingExponents(q=q, zeta=q * 0.5, provenance=prov).provenance.root)  # type: ignore[union-attr]
    assert roots[0] != roots[1]


def test_diagnostics_do_not_change_the_root() -> None:
    prov = new_provenance("synthetic", params={"H": 0.7})
    assert prov.with_diagnostics(n_dropped=3, note="x").root == prov.root


def test_run_records_the_dag_and_chains_parents(tmp_path) -> None:
    with Run("demo", store=tmp_path / "artifacts") as run:
        exps = run.record(_exponents(), name="exponents")
        run.record(spectrum_from_exponents(exps), name="spectrum")

    manifest = run.manifest
    assert [n.name for n in manifest.nodes] == ["exponents", "spectrum"]
    assert manifest.by_name("spectrum").parents == (manifest.by_name("exponents").root,)
    assert [n.name for n in manifest.leaves()] == ["spectrum"]
    assert [n.name for n in manifest.ancestors("spectrum")] == ["exponents"]


def test_session_root_is_reproducible_across_runs(tmp_path) -> None:
    roots = []
    for i in range(2):
        with Run("demo", store=tmp_path / f"run{i}") as run:
            exps = run.record(_exponents(), name="exponents")
            run.record(spectrum_from_exponents(exps), name="spectrum")
        roots.append(run.manifest_root)
    assert roots[0] == roots[1], "a re-run of the same pipeline must yield the same session root"


def test_manifest_round_trips_and_detects_tampering(tmp_path) -> None:
    with Run("demo", store=tmp_path / "artifacts") as run:
        run.record(_exponents(), name="exponents")
    path = tmp_path / "artifacts" / "run-demo.json"
    assert load_manifest(path).root == run.manifest_root

    text = path.read_text().replace('"H": "0.7"', '"H": "0.9"')
    assert text != path.read_text(), "the tamper target must actually be present"
    path.write_text(text)
    with pytest.raises(ValueError, match="manifest root mismatch"):
        load_manifest(path)


def test_stored_arrays_verify_against_their_digests(tmp_path) -> None:
    with Run("demo", store=tmp_path / "artifacts") as run:
        run.record(_exponents(), name="exponents")
    assert run.verify() == {"exponents": True}


def test_untraceable_results_are_refused(tmp_path) -> None:
    with (
        Run("demo", store=tmp_path / "artifacts") as run,
        pytest.raises(TypeError, match="no Provenance"),
    ):
        run.record(np.arange(10), name="raw")


def test_diff_reports_the_parameter_that_changed(tmp_path) -> None:
    manifests = []
    for H in (0.70, 0.55):
        with Run("demo", store=tmp_path / f"h{H}") as run:
            run.record(_exponents(H=H), name="exponents")
        manifests.append(run.manifest)

    report = diff_manifests(*manifests)
    assert report["identical"] is False
    assert report["only_in_a"] == [] and report["only_in_b"] == []
    assert report["changed"]["exponents"]["params"] == {"H": {"a": "0.7", "b": "0.55"}}


def test_diff_of_a_run_against_itself_is_clean(tmp_path) -> None:
    with Run("demo", store=tmp_path / "a") as run:
        run.record(_exponents(), name="exponents")
    report = diff_manifests(run.manifest, run.manifest)
    assert report["identical"] and not report["changed"] and not report["env_changed"]


def test_replay_plan_is_ordered_by_dependency(tmp_path) -> None:
    with Run("demo", store=tmp_path / "artifacts") as run:
        exps = run.record(_exponents(), name="exponents")
        run.record(spectrum_from_exponents(exps), name="spectrum")
    plan = replay_plan(run.manifest)
    assert plan.index("exponents =") < plan.index("spectrum ="), "parents must be emitted first"
    assert run.manifest.by_name("exponents").root in plan


def test_store_deduplicates_identical_computations(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "cas")
    for _ in range(3):
        prov = _exponents().provenance
        assert prov is not None
        store.put(prov, arrays={"q": np.arange(5.0)})
    assert len(store.list_roots()) == 1, "the same computation must not be stored twice"


def test_run_without_a_store_still_builds_the_dag() -> None:
    with Run("in-memory") as run:
        run.record(_exponents(), name="exponents")
    assert len(run) == 1 and run.verify() == {}


def test_in_memory_and_round_tripped_manifests_compare_equal(tmp_path) -> None:
    """A manifest read back from disk must diff clean against the live one.

    Regression: params were canonicalised on write but not in memory, so every
    float parameter looked changed after a save/load cycle.
    """
    with Run("demo", store=tmp_path / "artifacts") as run:
        run.record(_exponents(H=0.7), name="exponents")
    reloaded = load_manifest(tmp_path / "artifacts" / "run-demo.json")
    report = diff_manifests(run.manifest, reloaded)
    assert report["identical"] and not report["changed"]
