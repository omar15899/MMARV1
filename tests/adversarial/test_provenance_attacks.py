"""Attacks on the audit trail: collision, missed change, forgery, corruption.

The library's central claim is that a provenance root is a content address:
*two runs share a root if and only if they were the same computation*. Every
test here tries to break one direction of that biconditional, or to hand-edit a
stored artifact without the store noticing. A defeated audit trail is worse than
no audit trail, because the root is what a reader trusts instead of re-running.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mmar.calibration import calibrate_from_spectrum
from mmar.core.legendre import spectrum_from_exponents
from mmar.core.provenance import (
    ArtifactStore,
    EnvFingerprint,
    InputRef,
    Provenance,
    SeedInfo,
    array_digest,
    canonical_json,
    digest_document,
    digest_mapping,
    new_provenance,
)
from mmar.core.rng import make_rng
from mmar.core.types import FloatArray, ScalingExponents
from mmar.estimators.mfdfa import mfdfa
from mmar.models.fbm import fbm_with_provenance
from mmar.tracing import Run, load_manifest

_Q: FloatArray = np.linspace(-4.0, 4.0, 17)


def _series(n: int = 4096, seed: int = 7) -> FloatArray:
    return np.asarray(make_rng(seed).standard_normal(n).cumsum(), dtype=np.float64)


def _provenance(**over: Any) -> Provenance:
    base: dict[str, Any] = {
        "estimator": "attack",
        "estimator_version": "1",
        "params": {"a": 1, "b": "x"},
        "inputs": (InputRef.from_array("x", np.arange(8.0)),),
        "env": EnvFingerprint.capture(),
        "seed": SeedInfo(entropy=11),
    }
    base.update(over)
    return Provenance(**base)


def _put(store: ArtifactStore, **over: Any) -> str:
    return store.put(
        _provenance(**over),
        arrays={"zeta": np.arange(32.0)},
        summary={"h2": 0.5},
    )


@pytest.mark.parametrize(
    ("left", "right", "why"),
    [
        (1.5, "1.5", "a float is emitted as a bare repr, so it is its own string"),
        (float("nan"), "__nan__", "the NaN sentinel is an ordinary string value"),
        (float("inf"), "__inf__", "the infinity sentinel is an ordinary string value"),
        (
            np.arange(3.0),
            {
                "__array__": array_digest(np.arange(3.0)),
                "shape": [3],
                "dtype": np.arange(3.0).dtype.str,
            },
            "the array sentinel is an ordinary dict shape",
        ),
    ],
)
def test_canonicalisation_is_injective(left: Any, right: Any, why: str) -> None:
    """Two params that a user can distinguish must not canonicalise the same.

    `_canonicalise` maps a float onto its bare `repr` and tags NaN, infinity and
    arrays with plain strings/dicts that a user parameter can reproduce exactly.
    A run recorded with ``q_max=1.5`` and one recorded with ``q_max="1.5"`` are
    then the same content address, so an artifact store deduplicates two results
    that are not the same result. Fix: tag the float branch (``{"__float__":
    repr(x)}``) and the sentinels so no user value can spell them.
    """
    assert digest_mapping({"p": left}) != digest_mapping({"p": right}), why


@pytest.mark.xfail(strict=True, reason="Mapping keys are str()-ed, so distinct keys collide")
def test_canonicalising_a_mapping_never_drops_a_parameter() -> None:
    """A parameter that vanishes from the digest is a parameter nobody can audit.

    `_canonicalise` rebuilds a mapping as ``{str(k): ...}``. A dict holding both
    ``1`` and ``"1"`` therefore loses one entry outright: the manifest shows a
    single parameter, and two computations that differ in the dropped one share
    a root.
    """
    canonical = json.loads(canonical_json({"weights": {1: "a", "1": "b"}}))
    assert len(canonical["weights"]) == 2, canonical


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (1, 1.0),
        (1, True),
        (0.0, -0.0),
        (None, "None"),
        (None, 0),
        ({}, []),
        ("", None),
        (np.arange(3.0), np.arange(3.0, dtype=np.float32)),
        (np.arange(3.0), np.arange(3.0).reshape(3, 1)),
    ],
)
def test_scalar_forms_that_must_never_collide(left: Any, right: Any) -> None:
    """Lock in the discriminations the canonicaliser does get right.

    These are the cheap type-punning attacks. They fail today; a future
    "simplification" of `_canonicalise` (dropping the bool branch, coercing
    numpy scalars, normalising signed zero) would silently merge two distinct
    computations onto one address.
    """
    assert digest_mapping({"p": left}) != digest_mapping({"p": right})


def test_array_digest_ignores_memory_layout() -> None:
    """The same numbers must address the same, however numpy chose to store them.

    A digest that moved with C/Fortran order or with a strided view would make
    an artifact store miss every legitimate cache hit and would make roots
    machine-dependent — the failure the module's docstring rules out.
    """
    a = np.arange(12.0).reshape(3, 4)
    strided = np.arange(24.0).reshape(3, 8)[:, ::2].copy()
    view = np.arange(24.0).reshape(3, 8)[:, ::2]
    assert array_digest(a) == array_digest(np.asfortranarray(a))
    assert array_digest(strided) == array_digest(view)
    assert array_digest(a.ravel()) == array_digest(a.ravel().tolist())


def test_array_digest_binds_dtype_shape_and_normalises_byte_order() -> None:
    """Precision and shape are part of the computation; endianness is not.

    float32 and float64 inputs give different numbers, so they must give
    different roots. A big-endian array holds the same numbers as its
    little-endian twin, so a manifest written on one machine must verify on the
    other.
    """
    a = np.arange(12.0).reshape(3, 4)
    assert array_digest(a) != array_digest(a.astype(np.float32))
    assert array_digest(a) != array_digest(a.reshape(4, 3))
    assert array_digest(a.ravel()) != array_digest(a.ravel().reshape(12, 1))
    assert array_digest(np.arange(3)) != array_digest(np.arange(3.0))
    assert array_digest(a) == array_digest(a.astype(a.dtype.newbyteorder(">")))


def test_only_the_identifying_fields_move_the_root() -> None:
    """Every causal field must move the root; every incidental one must not.

    Both halves matter. A parameter outside the digest is an unauditable degree
    of freedom; a timestamp inside it makes reproducibility untestable.
    """
    base = _provenance()
    moved = {
        "estimator": _provenance(estimator="other"),
        "estimator_version": _provenance(estimator_version="2"),
        "params": _provenance(params={"a": 1, "b": "y"}),
        "params_extra": _provenance(params={"a": 1, "b": "x", "c": None}),
        "inputs": _provenance(inputs=(InputRef.from_array("x", np.arange(9.0)),)),
        "input_name": _provenance(inputs=(InputRef.from_array("y", np.arange(8.0)),)),
        "seed": _provenance(seed=SeedInfo(entropy=12)),
        "spawn_key": _provenance(seed=SeedInfo(entropy=11, spawn_key=(0,))),
    }
    for name, other in moved.items():
        assert other.root != base.root, f"root did not move when {name} changed"

    unmoved = {
        "created_utc": _provenance(created_utc="2026-01-01T00:00:00+00:00"),
        "notes": _provenance(notes=("a note",)),
        "diagnostics": base.with_diagnostics(n_dropped=3),
    }
    for name, other in unmoved.items():
        assert other.root == base.root, f"root moved on the non-identifying {name}"


def test_seed_zero_is_not_the_absence_of_a_seed() -> None:
    """``seed=0`` is a stream; ``seed=None`` is no recoverable stream at all.

    Collapsing the two would let an unreproducible run inherit the address of a
    reproducible one.
    """
    assert _provenance(seed=SeedInfo(entropy=0)).root != _provenance(seed=None).root
    assert _provenance(seed=SeedInfo(entropy=0)).root != _provenance(seed=SeedInfo(entropy=1)).root
    assert (
        _provenance(seed=SeedInfo(entropy=1)).root != _provenance(seed=SeedInfo(entropy=(1,))).root
    )


def test_the_record_does_not_alias_the_caller_params_dict() -> None:
    """A record whose root can be changed after the fact is not a record.

    `Provenance` is documented as immutable, but the params mapping was stored by
    reference: mutating the caller's dict retroactively rewrote the root of a
    record that may already have been written to a manifest under the old one.
    """
    params: dict[str, Any] = {"order": 1}
    prov = _provenance(params=params)
    before = prov.root
    params["order"] = 2
    params["injected"] = True
    assert prov.root == before
    assert dict(prov.params) == {"order": 1}


def test_a_one_ulp_change_at_the_root_of_the_dag_reaches_every_leaf() -> None:
    """Perturb the series in the last bit; series -> exponents -> spectrum ->
    calibration must all re-address.

    A chain that absorbs a change anywhere along it lets two different analyses
    be filed under one address, which is exactly how a stale result gets quoted.
    """
    x = _series()
    x2 = x.copy()
    x2[0] = np.nextafter(x2[0], np.inf)

    def chain(series: FloatArray) -> tuple[str, str, str]:
        exps = mfdfa(series, q=_Q)
        spec = spectrum_from_exponents(exps)
        cal = calibrate_from_spectrum(exps, spec)
        assert exps.provenance is not None
        assert spec.provenance is not None
        assert cal.provenance is not None
        return exps.provenance.root, spec.provenance.root, cal.provenance.root

    a, b = chain(x), chain(x2)
    assert chain(x) == a, "the chain must be reproducible before it can be discriminating"
    for depth, (ra, rb) in enumerate(zip(a, b, strict=True)):
        assert ra != rb, f"depth {depth} absorbed a one-ulp change in the input series"


def test_a_mid_chain_substitution_reaches_the_leaf() -> None:
    """Calibrate the same exponents against two *different* spectra.

    The spectrum supplies alpha_0 and the observed width, so it is a second
    input and not a function of the first. Chaining only the exponents gave both
    calibrations one content address while their numbers differed.
    """
    exps = mfdfa(_series(), q=_Q)
    fenchel = spectrum_from_exponents(exps, method="fenchel")
    parametric = spectrum_from_exponents(exps, method="parametric")
    a = calibrate_from_spectrum(exps, fenchel)
    b = calibrate_from_spectrum(exps, parametric)

    assert a.alpha_0 != b.alpha_0, "the two calibrations must actually differ"
    assert a.provenance is not None
    assert b.provenance is not None
    assert a.provenance.root != b.provenance.root


def test_verify_rejects_a_tampered_manifest(tmp_path: Path) -> None:
    """Rewrite each identifying field of a stored manifest in turn.

    The store names files by digest but never re-derived it, so a hand-edited
    ``params`` (or estimator, or input digest, or seed) survived ``verify()``
    intact: the manifest claimed an address it no longer had.
    """
    store = ArtifactStore(tmp_path / "cas")
    root = _put(store)
    assert store.verify(root)
    pristine = (store.manifests / f"{root}.json").read_text()

    edits: dict[str, dict[str, Any]] = {
        "params": {"params": {"a": 99, "b": "x"}},
        "estimator": {"estimator": "something_else"},
        "estimator_version": {"estimator_version": "2"},
        "seed": {"seed": {"bit_generator": "PCG64", "entropy": 99, "spawn_key": []}},
        "env": {"env": {**json.loads(pristine)["env"], "numpy": "0.0.0"}},
    }
    for name, patch in edits.items():
        forged = {**json.loads(pristine), **patch}
        (store.manifests / f"{root}.json").write_text(json.dumps(forged, indent=2, sort_keys=True))
        assert not store.verify(root), f"a rewritten {name} passed verification"
    (store.manifests / f"{root}.json").write_text(pristine)
    assert store.verify(root)


def test_verify_rejects_a_manifest_filed_under_a_stolen_address(tmp_path: Path) -> None:
    """Put computation B's manifest under computation A's digest.

    Content addressing is only worth something if the address is derived from
    the content on read as well as on write.
    """
    store = ArtifactStore(tmp_path / "cas")
    a = _put(store)
    b = _put(store, params={"a": 2, "b": "x"})
    assert a != b
    (store.manifests / f"{a}.json").write_text((store.manifests / f"{b}.json").read_text())
    assert not store.verify(a)


def test_verify_returns_false_for_a_corrupted_npz(tmp_path: Path) -> None:
    """Flip one byte inside the compressed payload.

    Detecting exactly this is the method's purpose, yet the zip CRC failure
    escaped as a ``BadZipFile`` traceback out of a ``-> bool`` audit routine, so
    ``Run.verify()`` aborted the whole session instead of naming the bad node.
    """
    store = ArtifactStore(tmp_path / "cas")
    root = _put(store)
    path = store.arrays / f"{root}.npz"
    blob = bytearray(path.read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    path.write_bytes(bytes(blob))

    with pytest.raises(zipfile.BadZipFile):
        store.get_arrays(root)
    assert store.verify(root) is False


def test_verify_detects_a_deleted_or_extended_npz(tmp_path: Path) -> None:
    """Remove the payload, then add an array nobody recorded.

    A missing payload must not read as "nothing to check", and an extra array
    smuggled into the archive must not read as "everything recorded matches".
    """
    store = ArtifactStore(tmp_path / "cas")
    root = _put(store)
    path = store.arrays / f"{root}.npz"
    payload = path.read_bytes()
    path.unlink()
    assert not store.verify(root)

    path.write_bytes(payload)
    assert store.verify(root)
    np.savez_compressed(path, zeta=np.arange(32.0), smuggled=np.ones(3))
    assert not store.verify(root)


@pytest.mark.xfail(
    strict=True,
    reason="the manifest summary is outside every digest, so a reported number can be rewritten",
)
def test_the_stored_summary_cannot_be_rewritten(tmp_path: Path) -> None:
    """Edit the reported ``h2`` in a stored manifest and re-verify.

    ``root`` is a pure function of the *inputs*, by design, so nothing in the
    store binds the *outputs* it files next to them: ``verify()`` returns True
    while the manifest reports h2 = 0.99 for a computation that produced 0.5.
    Fix: record a document digest over the whole manifest (summary and array
    digests included) and check it in ``verify``.
    """
    store = ArtifactStore(tmp_path / "cas")
    root = _put(store)
    path = store.manifests / f"{root}.json"
    forged = json.loads(path.read_text())
    forged["summary"]["h2"] = "0.99"
    path.write_text(json.dumps(forged, indent=2, sort_keys=True))

    assert store.get_manifest(root)["summary"]["h2"] == "0.99"
    assert not store.verify(root)


def test_re_hashing_a_stored_document_is_stable(tmp_path: Path) -> None:
    """Serialise, reload, re-hash: the address must not move.

    Canonicalisation is injective but deliberately NOT idempotent -- a string
    that spells a sentinel is escaped on every pass, which is exactly what stops
    it colliding with the sentinel. So the invariant every verification path
    relies on is not "canonicalising twice is a no-op", it is "an already
    canonical document re-hashes to the same address", which is what
    ``digest_document`` is for. Getting this wrong makes honest manifests
    containing an array, a NaN or a ``__``-prefixed string fail verification on
    the first round trip.
    """
    store = ArtifactStore(tmp_path / "cas")
    prov = _provenance(params={"a": 1.5, "b": float("nan"), "c": (1, 2), "d": np.arange(3.0)})
    root = store.put(prov, arrays={"zeta": np.arange(4.0)}, summary={"h2": 0.5})

    document = store.get_manifest(root)
    reloaded = json.loads(json.dumps(document))
    assert digest_document(document) == digest_document(reloaded)
    assert store.verify(root)

    # And the non-idempotence is a deliberate property, not an accident.
    once = canonical_json({"v": "__nan__"})
    assert canonical_json(json.loads(once)) != once


def test_run_manifest_rejects_a_forged_reported_number(tmp_path: Path) -> None:
    """Rewrite a node's summary in a run manifest and reload it.

    ``TraceNode.to_canonical`` left ``summary`` out of the session Merkle root,
    so the one thing a human reads off the manifest — the reported h(2) — could
    be edited freely: ``load_manifest`` accepted the file and ``describe()``
    printed the forged value under an unchanged root.
    """
    exps = ScalingExponents(q=_Q, zeta=_Q * 0.5, provenance=_provenance())
    with Run("forge", store=tmp_path / "artifacts") as run:
        run.record(exps, name="exponents", summary={"h2": 0.5})
    path = tmp_path / "artifacts" / "run-forge.json"

    honest = load_manifest(path)
    assert honest.root == run.manifest_root

    raw = json.loads(path.read_text())
    raw["nodes"][0]["summary"]["h2"] = "0.99"
    path.write_text(json.dumps(raw, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="manifest root mismatch"):
        load_manifest(path)


def test_a_run_manifest_survives_a_round_trip(tmp_path: Path) -> None:
    """The tamper check must not fire on an untampered file.

    A digest that covers the summary is only useful if canonicalisation of the
    in-memory summary and of its JSON image agree; otherwise every honest
    reload would raise.
    """
    exps = ScalingExponents(q=_Q, zeta=_Q * 0.5, provenance=_provenance())
    with Run("clean", store=tmp_path / "artifacts") as run:
        run.record(exps, name="exponents")
    path = tmp_path / "artifacts" / "run-clean.json"
    assert load_manifest(path).root == run.manifest_root
    assert load_manifest(path).root == load_manifest(path).root


@pytest.mark.xfail(
    strict=True,
    reason="with seed=None the recipe is recorded but the realisation is not, so two "
    "different paths share one address",
)
def test_two_unseeded_paths_do_not_share_one_content_address() -> None:
    """Generate two unseeded fBm paths and compare their roots.

    ``fbm_with_provenance`` promises that "the path can be regenerated from the
    manifest alone". With ``seed=None`` nothing recoverable is recorded, so two
    genuinely different realisations are filed under one address: store both and
    the second silently overwrites the first, and ``verify()`` is content.
    """
    p1, prov1 = fbm_with_provenance(256, 0.7, seed=None)
    p2, prov2 = fbm_with_provenance(256, 0.7, seed=None)
    assert not np.array_equal(p1, p2), "unseeded draws must differ for the attack to be meaningful"
    assert prov1.root != prov2.root


def test_a_stochastic_estimator_records_a_seed_that_moves_the_root() -> None:
    """Two seeds must not produce one address.

    Guards the other direction of the unseeded hole: when a seed *is* supplied
    it has to reach the digest, or a Monte Carlo result could be quoted under
    the address of a different draw.
    """
    prov = new_provenance("stochastic", params={"n": 4}, seed=None)
    seeded = [
        new_provenance("stochastic", params={"n": 4}, seed=SeedInfo(entropy=s)) for s in (0, 1)
    ]
    roots = {prov.root, seeded[0].root, seeded[1].root}
    assert len(roots) == 3
