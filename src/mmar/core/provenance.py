"""Content-addressed provenance.

Every quantity this library produces carries a :class:`Provenance` record whose
``root`` is a SHA-256 content address over the *causally identifying* inputs:
the digests of the input arrays, the estimator identity and version, the fully
canonicalised parameter set, the RNG seed, and the environment fingerprint.

Two runs share a root if and only if they were the same computation. Wall-clock
time is recorded but deliberately excluded from the digest, so reproducibility
is testable: re-running a deterministic estimator must reproduce the root.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ArtifactStore",
    "EnvFingerprint",
    "InputRef",
    "Provenance",
    "SeedInfo",
    "array_digest",
    "canonical_json",
    "digest_mapping",
    "new_provenance",
]

_HASH: Final = "sha256"
_DIGEST_CHARS: Final = 64


def _canonicalise(obj: Any) -> Any:  # noqa: PLR0911 - one branch per canonical form
    """Map an arbitrary parameter value onto a JSON-serialisable canonical form.

    Floats are emitted as their shortest round-tripping repr so that the digest
    is stable across platforms; NaN/inf are tagged rather than silently coerced.
    """
    if obj is None or isinstance(obj, bool | int | str):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "__nan__"
        if obj == float("inf"):
            return "__inf__"
        if obj == float("-inf"):
            return "__-inf__"
        return repr(obj)
    if isinstance(obj, np.generic):
        return _canonicalise(obj.item())
    if isinstance(obj, np.ndarray):
        return {"__array__": array_digest(obj), "shape": list(obj.shape), "dtype": obj.dtype.str}
    if isinstance(obj, Mapping):
        return {str(k): _canonicalise(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, Sequence | set | frozenset):
        items = sorted(obj, key=repr) if isinstance(obj, set | frozenset) else list(obj)
        return [_canonicalise(v) for v in items]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_canonical"):
        return _canonicalise(obj.to_canonical())
    return {"__repr__": repr(obj), "__type__": type(obj).__qualname__}


def canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding: sorted keys, no whitespace, canonical scalars."""
    return json.dumps(_canonicalise(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def array_digest(arr: NDArray[Any] | Sequence[float]) -> str:
    """Content address of a numeric array, invariant to memory layout.

    The digest binds dtype, shape and the C-contiguous byte image in a fixed
    (little-endian) byte order, so it is stable across machines.
    """
    a = np.ascontiguousarray(arr)
    if a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"))
    h = hashlib.new(_HASH)
    h.update(a.dtype.newbyteorder("<").str.encode())
    h.update(repr(a.shape).encode())
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def digest_mapping(obj: Any) -> str:
    """Content address of any canonicalisable object."""
    return hashlib.new(_HASH, canonical_json(obj).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class InputRef:
    """A reference to one input of a computation, by content address."""

    name: str
    digest: str
    shape: tuple[int, ...]
    dtype: str
    n_finite: int | None = None

    @classmethod
    def from_array(cls, name: str, arr: NDArray[Any]) -> InputRef:
        a = np.asarray(arr)
        finite = int(np.isfinite(a).sum()) if a.dtype.kind in "fc" else int(a.size)
        return cls(
            name=name,
            digest=array_digest(a),
            shape=tuple(int(s) for s in a.shape),
            dtype=a.dtype.str,
            n_finite=finite,
        )

    @classmethod
    def from_provenance(cls, name: str, prov: Provenance) -> InputRef:
        """Chain an upstream result into a downstream computation (Merkle edge)."""
        return cls(name=name, digest=prov.root, shape=(), dtype="provenance")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "digest": self.digest,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }


@dataclass(frozen=True, slots=True)
class SeedInfo:
    """Everything needed to regenerate the exact random stream."""

    entropy: int | tuple[int, ...] | None
    spawn_key: tuple[int, ...] = ()
    bit_generator: str = "PCG64"

    def to_canonical(self) -> dict[str, Any]:
        return {
            "entropy": list(self.entropy) if isinstance(self.entropy, tuple) else self.entropy,
            "spawn_key": list(self.spawn_key),
            "bit_generator": self.bit_generator,
        }


@dataclass(frozen=True, slots=True)
class EnvFingerprint:
    """Software environment. Recorded and hashed: numerics are version-sensitive."""

    python: str
    numpy: str
    scipy: str | None
    platform: str
    library: str

    @classmethod
    def capture(cls) -> EnvFingerprint:
        from mmar import __version__  # noqa: PLC0415 - deferred to break a circular import

        try:
            import scipy  # noqa: PLC0415 - optional at import time, probed lazily

            scipy_v: str | None = scipy.__version__
        except ImportError:  # pragma: no cover - scipy is a hard dependency
            scipy_v = None
        return cls(
            python=".".join(str(v) for v in sys.version_info[:3]),
            numpy=np.__version__,
            scipy=scipy_v,
            platform=f"{platform.system()}-{platform.machine()}",
            library=__version__,
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "numpy": self.numpy,
            "scipy": self.scipy,
            "platform": self.platform,
            "library": self.library,
        }


@dataclass(frozen=True, slots=True)
class Provenance:
    """Immutable audit record attached to every result object.

    ``root`` is the content address. It is a pure function of
    ``(estimator, estimator_version, params, inputs, seed, env)`` — never of
    the wall clock — so identical computations are recognisably identical.
    """

    estimator: str
    estimator_version: str
    params: Mapping[str, Any]
    inputs: tuple[InputRef, ...]
    env: EnvFingerprint
    seed: SeedInfo | None = None
    created_utc: str = ""
    notes: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def root(self) -> str:
        return digest_mapping(self._identifying())

    @property
    def short_root(self) -> str:
        return self.root[:12]

    def _identifying(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator,
            "estimator_version": self.estimator_version,
            "params": dict(self.params),
            "inputs": [i.to_canonical() for i in self.inputs],
            "env": self.env.to_canonical(),
            "seed": self.seed.to_canonical() if self.seed is not None else None,
        }

    def derive(
        self,
        estimator: str,
        *,
        estimator_version: str = "1",
        params: Mapping[str, Any] | None = None,
        extra_inputs: Iterable[InputRef] = (),
        seed: SeedInfo | None = None,
        notes: Iterable[str] = (),
    ) -> Provenance:
        """Create a child record that chains this one as an input (Merkle edge)."""
        return Provenance(
            estimator=estimator,
            estimator_version=estimator_version,
            params=dict(params or {}),
            inputs=(InputRef.from_provenance("parent", self), *extra_inputs),
            env=self.env,
            seed=seed if seed is not None else self.seed,
            created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            notes=tuple(notes),
        )

    def with_diagnostics(self, **kwargs: Any) -> Provenance:
        """Attach non-identifying diagnostics (they do not change ``root``)."""
        return replace(self, diagnostics={**dict(self.diagnostics), **kwargs})

    def to_manifest(self) -> dict[str, Any]:
        """Full serialisable record, digest included."""
        return {
            "root": self.root,
            "created_utc": self.created_utc,
            "notes": list(self.notes),
            "diagnostics": json.loads(canonical_json(dict(self.diagnostics))),
            **json.loads(canonical_json(self._identifying())),
        }

    def __str__(self) -> str:
        return f"{self.estimator}@{self.estimator_version} root={self.short_root}"


def new_provenance(
    estimator: str,
    *,
    estimator_version: str = "1",
    params: Mapping[str, Any] | None = None,
    inputs: Iterable[InputRef] = (),
    seed: SeedInfo | None = None,
    notes: Iterable[str] = (),
) -> Provenance:
    """Construct a root-level provenance record with the environment captured."""
    return Provenance(
        estimator=estimator,
        estimator_version=estimator_version,
        params=dict(params or {}),
        inputs=tuple(inputs),
        env=EnvFingerprint.capture(),
        seed=seed,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        notes=tuple(notes),
    )


class ArtifactStore:
    """Content-addressed on-disk store for results and their manifests.

    Layout::

        <root_dir>/manifests/<digest>.json     provenance + scalar summary
        <root_dir>/arrays/<digest>.npz         the numeric payload

    Writing the same result twice is a no-op: the digest already exists.
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.manifests = self.root_dir / "manifests"
        self.arrays = self.root_dir / "arrays"

    def _ensure(self) -> None:
        self.manifests.mkdir(parents=True, exist_ok=True)
        self.arrays.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        prov: Provenance,
        arrays: Mapping[str, NDArray[Any]] | None = None,
        summary: Mapping[str, Any] | None = None,
    ) -> str:
        """Persist a result. Returns its content address."""
        self._ensure()
        root = prov.root
        manifest = prov.to_manifest()
        if summary:
            manifest["summary"] = json.loads(canonical_json(dict(summary)))
        if arrays:
            manifest["arrays"] = {k: array_digest(v) for k, v in sorted(arrays.items())}
            payload = {k: np.asarray(v) for k, v in arrays.items()}
            # mypy resolves the ** into savez_compressed's `allow_pickle` parameter;
            # the keys here are array names, never that flag.
            np.savez_compressed(self.arrays / f"{root}.npz", **payload)  # type: ignore[arg-type]
        (self.manifests / f"{root}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return root

    def has(self, root: str) -> bool:
        return (self.manifests / f"{root}.json").exists()

    def get_manifest(self, root: str) -> dict[str, Any]:
        text = (self.manifests / f"{root}.json").read_text()
        obj: dict[str, Any] = json.loads(text)
        return obj

    def get_arrays(self, root: str) -> dict[str, NDArray[Any]]:
        path = self.arrays / f"{root}.npz"
        if not path.exists():
            return {}
        with np.load(path) as data:
            return {k: np.asarray(data[k]) for k in data.files}

    def verify(self, root: str) -> bool:
        """Re-hash the stored arrays and confirm they match the recorded digests."""
        manifest = self.get_manifest(root)
        recorded: Mapping[str, str] = manifest.get("arrays", {})
        if not recorded:
            return len(root) == _DIGEST_CHARS
        stored = self.get_arrays(root)
        if set(stored) != set(recorded):
            return False
        return all(array_digest(stored[k]) == d for k, d in recorded.items())

    def list_roots(self) -> list[str]:
        if not self.manifests.exists():
            return []
        return sorted(p.stem for p in self.manifests.glob("*.json"))
