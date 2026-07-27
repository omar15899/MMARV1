"""The content address must be an identity, and it must bind the numbers.

Three attacks on the traceability layer, each of which succeeded before the
fixes these tests pin. They matter more than an estimator bug: a wrong exponent
is visible to a sceptical reader, whereas a forged or non-binding audit record
is what a sceptical reader would have checked it with.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmar.core.provenance import array_digest, canonical_json, new_provenance
from mmar.core.types import HolderSpectrum, PointwiseHolder, ScalingExponents, StructureFunctions


def _exponents() -> ScalingExponents:
    q = np.linspace(-3.0, 3.0, 21)
    return ScalingExponents(q=q.copy(), zeta=q * 0.7, provenance=new_provenance("demo"))


def test_stringified_keys_that_collide_are_refused_not_merged() -> None:
    """Observed: {1: 'x', '1': 'y'} canonicalised to {"1":"y"} -- one entry gone.

    A parameter vanishing from the audit record is bad; two computations that
    differ only in it sharing a content address is worse.
    """
    with pytest.raises(ValueError, match="canonicalise to"):
        canonical_json({1: "x", "1": "y"})
    assert canonical_json({"a": 1, "b": 2})


def test_a_parameter_cannot_spell_the_array_sentinel() -> None:
    """Observed: a dict param shaped like the array sentinel gave an identical root."""
    arr = np.arange(4.0)
    real = new_provenance("d", params={"w": arr})
    forged = new_provenance(
        "d", params={"w": {"__array__": array_digest(arr), "shape": [4], "dtype": "<f8"}}
    )
    assert real.root != forged.root


@pytest.mark.parametrize("sentinel", ["__nan__", "__inf__", "__-inf__"])
def test_a_string_cannot_spell_a_float_sentinel(sentinel: str) -> None:
    special = {"__nan__": float("nan"), "__inf__": float("inf"), "__-inf__": float("-inf")}
    assert (
        new_provenance("d", params={"v": special[sentinel]}).root
        != new_provenance("d", params={"v": sentinel}).root
    )


def test_escaping_is_reversible_enough_to_stay_injective() -> None:
    """Distinct inputs must stay distinct however many underscores they carry."""
    roots = {
        new_provenance("d", params={"v": name}).root
        for name in ("__nan__", "___nan__", "____nan__", "nan", "_nan_")
    }
    assert len(roots) == 5


def test_mutating_a_result_in_place_is_refused() -> None:
    """Observed: result.zeta[:] *= 2 moved h2 from 0.70 to 1.40, root unchanged.

    Every reported quantity is derived from these arrays on demand, while the
    root digests only inputs and settings. Without this, the audit record stops
    describing the object it is attached to.
    """
    exps = _exponents()
    assert exps.h2 == pytest.approx(0.7)
    with pytest.raises(ValueError, match="read-only"):
        exps.zeta[:] = exps.zeta * 2.0
    assert exps.h2 == pytest.approx(0.7)


def test_every_result_type_freezes_the_arrays_it_reports_from() -> None:
    q = np.linspace(-2.0, 2.0, 9)
    exps = _exponents()
    spec = HolderSpectrum(alpha=q.copy(), f_alpha=np.ones_like(q))
    point = PointwiseHolder(t=q.copy(), h=np.full_like(q, 0.5))
    struct = StructureFunctions(q=q.copy(), scales=np.arange(2.0, 6.0), values=np.ones((9, 4)))

    for obj, names in (
        (exps, ("q", "zeta")),
        (spec, ("alpha", "f_alpha")),
        (point, ("t", "h")),
        (struct, ("q", "scales", "values")),
    ):
        for name in names:
            arr = getattr(obj, name)
            assert not arr.flags.writeable, f"{type(obj).__name__}.{name} is writeable"


def test_freezing_does_not_disturb_the_values() -> None:
    q = np.linspace(-3.0, 3.0, 21)
    exps = ScalingExponents(q=q.copy(), zeta=q * 0.7)
    assert np.array_equal(exps.zeta, q * 0.7)
    assert exps.tau[10] == pytest.approx(-1.0)
