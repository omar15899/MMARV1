"""Audit-trail attacks: a root that misses a real change, or condemns an honest one.

Companion to ``test_provenance_attacks.py``, which attacks the canonical
encoding directly. These four attacks come in from the other side: the array
digest primitive, the mutability of a record after it is written, and the
parameters :func:`mmar.analysis.analyze` chooses to record at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from mmar.analysis import analyze
from mmar.core.provenance import ArtifactStore, InputRef, array_digest, new_provenance
from mmar.core.rng import SeedLike, make_rng
from mmar.core.types import FloatArray, ScalingExponents
from mmar.estimators.structure_function import mfc_exponents


def _constant_estimator(
    series: NDArray[Any], *, seed: SeedLike = None, **kwargs: Any
) -> ScalingExponents:
    """A minimal conforming estimator whose exponents depend only on ``q``.

    ``analyze`` documents that any callable with the standard signature is
    acceptable, and such a callable need not attach its own provenance. That
    strips away the estimator's own record and leaves ``analyze``'s parameter
    capture as the only thing standing between two different computations and
    one content address.
    """
    q: FloatArray = np.asarray(kwargs.get("q", [1.0, 2.0, 3.0]), dtype=np.float64)
    return ScalingExponents(q=q, zeta=0.5 * q, d_support=1.0, provenance=None)


def test_analyze_root_binds_a_moment_grid_passed_as_an_array() -> None:
    """Change ``q``, get different exponents, demand a different root.

    ``analyze`` filtered every ``np.ndarray`` out of the parameters it recorded.
    The moment grid decides which exponents come back, so two analyses returning
    different zeta were filed under one content address whenever ``q`` was passed
    as an array rather than a list -- and in an ``ArtifactStore`` the second
    silently deduplicates onto the first.
    """
    x: FloatArray = np.asarray(make_rng(11).standard_normal(2048).cumsum(), dtype=np.float64)
    qa, qb = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0])

    ra = analyze(x, estimator=_constant_estimator, q=qa, cumulants=False)
    rb = analyze(x, estimator=_constant_estimator, q=qb, cumulants=False)
    assert ra.provenance is not None
    assert rb.provenance is not None
    assert not np.allclose(ra.exponents.zeta, rb.exponents.zeta), "the two results must differ"
    assert ra.provenance.root != rb.provenance.root, "different exponents, one content address"

    # The other direction, so the grid is bound by content and not by identity:
    # an equal grid in a different buffer must still be the same computation.
    again = analyze(x, estimator=_constant_estimator, q=qa.copy(), cumulants=False)
    assert again.provenance is not None
    assert again.provenance.root == ra.provenance.root


def test_params_do_not_alias_nested_caller_state(tmp_path: Path) -> None:
    """Mutate the caller's *nested* parameter values after the record is built.

    ``Provenance`` copied the parameter mapping so a record could not be moved
    after the fact, but the copy was shallow: the list of scales, the nested
    option dict and the ``q`` array stayed shared with the caller. Appending one
    scale then moved the root of a manifest already written to disk under the old
    address, and ``ArtifactStore.verify`` had nothing to say about it because the
    file on disk was never the thing that changed.
    """
    scales = [8, 16, 32]
    grid = np.array([-2.0, 0.0, 2.0])
    opts: dict[str, Any] = {"weights": [1.0, 1.0]}
    prov = new_provenance("attack", params={"scales": scales, "q": grid, "opts": opts})

    store = ArtifactStore(tmp_path / "cas")
    root = store.put(prov)

    scales.append(64)
    grid[0] = 99.0
    opts["weights"].append(7.0)

    assert prov.root == root, "mutating caller state moved an already-written root"
    assert dict(prov.params)["scales"] == [8, 16, 32]
    assert dict(prov.params)["opts"] == {"weights": [1.0, 1.0]}
    assert np.array_equal(dict(prov.params)["q"], np.array([-2.0, 0.0, 2.0]))
    assert store.verify(root) is True


def test_array_digest_separates_a_scalar_from_a_length_one_vector() -> None:
    """A 0-d array and a one-element array must not share a content address.

    ``array_digest`` densified before reading the shape, and
    ``np.ascontiguousarray`` promotes 0-d to ``(1,)``, so both hashed as shape
    ``(1,)``. ``ArtifactStore.put`` files array payloads under this digest with
    no separate shape field, so a stored scalar and a stored one-element vector
    were indistinguishable to ``verify``.
    """
    assert array_digest(np.array(1.0)) != array_digest(np.array([1.0]))
    assert array_digest(np.array([1.0])) == array_digest([1.0])
    assert array_digest(np.array(1.0)) == array_digest(np.array(1.0))


def test_array_digest_refuses_arrays_of_python_objects() -> None:
    """An object array hashes its pointer table, not its content.

    ``tobytes`` on ``dtype=object`` emits addresses, so two arrays holding equal
    content received different digests inside a single process and a fresh digest
    on every run -- silent non-reproducibility in the one primitive the whole
    audit trail rests on. It has to be an error, not a plausible-looking number.
    """
    with pytest.raises(TypeError, match="numeric array"):
        array_digest(np.array([{"a": 1}], dtype=object))
    with pytest.raises(TypeError, match="numeric array"):
        InputRef.from_array("x", np.array([{"a": 1}], dtype=object))
    assert array_digest(np.array(["ab", "cd"])) == array_digest(np.array(["ab", "cd"]))


def test_verify_accepts_an_untouched_artifact_from_a_shipped_estimator(tmp_path: Path) -> None:
    """Write a real estimator's record, change nothing, and demand ``verify`` agree.

    ``mfc_exponents`` records ``q`` and ``scales`` as ndarrays, which canonicalise
    to the ``{"__array__": ...}`` sentinel. ``verify`` re-canonicalised the
    manifest it had just read back, escaping that key a second time, so every
    structure-function artifact ever stored failed its own integrity check. A
    false alarm from an audit routine is as corrosive as a missed one: it trains
    the reader to ignore the routine.
    """
    x: FloatArray = np.asarray(make_rng(3).standard_normal(4096).cumsum(), dtype=np.float64)
    exponents = mfc_exponents(x)
    assert exponents.provenance is not None
    assert any(isinstance(v, np.ndarray) for v in exponents.provenance.params.values())

    store = ArtifactStore(tmp_path / "cas")
    root = store.put(exponents.provenance, arrays={"zeta": exponents.zeta})
    assert store.verify(root) is True

    manifest = store.get_manifest(root)
    manifest["params"]["min_windows"] = 999
    (store.manifests / f"{root}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    assert store.verify(root) is False, "a rewritten params block must still be caught"
