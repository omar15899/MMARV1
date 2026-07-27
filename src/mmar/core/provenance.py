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

import copy
import hashlib
import json
import math
import platform
import sys
import zipfile
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
    "digest_document",
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
    if obj is None or isinstance(obj, bool | int):
        return obj
    if isinstance(obj, str):
        # A user string that spells a sentinel would otherwise be indistinguishable
        # from the sentinel itself, so NaN and the string "__nan__" would collide.
        return _escape(obj)
    if isinstance(obj, float):
        if math.isnan(obj):
            return _NAN
        if obj == float("inf"):
            return _INF
        if obj == float("-inf"):
            return _NEG_INF
        return repr(obj)
    if isinstance(obj, np.generic):
        return _canonicalise(obj.item())
    if isinstance(obj, np.ndarray):
        return {_ARRAY: array_digest(obj), "shape": list(obj.shape), "dtype": obj.dtype.str}
    if isinstance(obj, Mapping):
        return _canonicalise_mapping(obj)
    if isinstance(obj, Sequence | set | frozenset):
        items = sorted(obj, key=repr) if isinstance(obj, set | frozenset) else list(obj)
        return [_canonicalise(v) for v in items]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_canonical"):
        return _canonicalise(obj.to_canonical())
    return {"__repr__": repr(obj), "__type__": type(obj).__qualname__}


_NAN: Final = "__nan__"
_INF: Final = "__inf__"
_NEG_INF: Final = "__-inf__"
_ARRAY: Final = "__array__"
_SENTINELS: Final = frozenset({_NAN, _INF, _NEG_INF, _ARRAY, "__repr__", "__type__"})


def _looks_numeric(text: str) -> bool:
    """Whether a string would be indistinguishable from an emitted float."""
    try:
        float(text)
    except ValueError:
        return False
    return True


def _escape(text: str) -> str:
    """Prefix an underscore onto any string that could be read as an emitted value.

    The canonicaliser emits ``"__nan__"`` for NaN, ``{"__array__": ...}`` for an
    array, and ``repr(v)`` for a finite float. Without escaping, a caller whose
    parameter is literally the string ``"__nan__"``, or ``"1.5"``, or a dict with
    an ``"__array__"`` key, canonicalises to the same bytes as the value and
    therefore to the same content address. The encoding must be injective or the
    digest is not an identity.

    Injectivity: strings beginning ``_`` are escaped too, so the map is
    ``t -> "_" + t`` on the escaped class and the identity off it. Two distinct
    strings can only collide if one is ``"_" + other``, and then the second
    begins with ``_`` and would itself have been escaped. Note the map is
    deliberately *not* idempotent -- see :func:`digest_document`.
    """
    return "_" + text if text.startswith("_") or _looks_numeric(text) else text


def _canonicalise_mapping(obj: Mapping[Any, Any]) -> dict[str, Any]:
    """Canonicalise a mapping, refusing key collisions rather than losing entries.

    Keys are stringified so that the encoding is JSON, which means ``1`` and
    ``"1"`` map to the same key. Silently keeping one of them would drop a
    parameter from the audit record *and* make two different computations share
    a root, so the collision is an error.
    """
    out: dict[str, Any] = {}
    seen: dict[str, Any] = {}
    for key, value in sorted(obj.items(), key=lambda kv: str(kv[0])):
        name = _escape(str(key))
        if name in out:
            raise ValueError(
                f"parameter keys {seen[name]!r} and {key!r} both canonicalise to {name!r}; "
                "one would be dropped from the provenance record and two different "
                "computations would share a content address. Use distinct string keys."
            )
        seen[name] = key
        out[name] = _canonicalise(value)
    return out


def canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding: sorted keys, no whitespace, canonical scalars."""
    return json.dumps(_canonicalise(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def array_digest(arr: NDArray[Any] | Sequence[float]) -> str:
    """Content address of a numeric array, invariant to memory layout.

    The digest binds dtype, shape and the C-contiguous byte image in a fixed
    (little-endian) byte order, so it is stable across machines.

    An object-dtype array is refused: its buffer holds pointers, so hashing it
    would mint a *different* address for the same content on every process.
    """
    a = np.asarray(arr)
    if a.dtype.hasobject:
        raise TypeError(
            f"array_digest needs a numeric array; dtype={a.dtype!r} stores Python objects "
            "whose in-memory bytes are addresses, so the digest would differ between two "
            "runs on identical data and a manifest would never verify twice"
        )
    # Read the shape before densifying: ascontiguousarray promotes a 0-d array to
    # shape (1,), which would give a scalar and a length-one vector one address.
    shape = a.shape
    a = np.ascontiguousarray(a)
    if a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"))
    h = hashlib.new(_HASH)
    h.update(a.dtype.newbyteorder("<").str.encode())
    h.update(repr(shape).encode())
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def digest_document(doc: Any) -> str:
    """Content address of an **already canonical** document.

    :func:`_canonicalise` is deliberately *not* idempotent: a string that spells
    a sentinel is escaped on every pass, which is what makes the encoding
    injective. So anything that re-hashes the output of :func:`canonical_json`
    --- a manifest read back from disk, a :class:`~mmar.tracing.TraceNode` whose
    params were canonicalised at construction --- must re-encode with the same
    JSON settings rather than canonicalise a second time. Otherwise a document
    containing an array, a NaN or a ``__``-prefixed string hashes differently on
    every round trip and honest manifests start failing verification.
    """
    encoded = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.new(_HASH, encoded.encode()).hexdigest()


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

    def __post_init__(self) -> None:
        """Deep-copy the mappings in.

        Without this the record aliases the caller's dict: mutating it after
        construction retroactively changes ``root``, so an audit record already
        written to a manifest stops matching its own address. A shallow copy is
        not enough, because the values that carry the numbers -- a list of
        scales, a nested option dict, a ``q`` array -- stay shared, and mutating
        one of those moves the root just as effectively as replacing a key.
        """
        object.__setattr__(self, "params", copy.deepcopy(dict(self.params)))
        object.__setattr__(self, "diagnostics", copy.deepcopy(dict(self.diagnostics)))

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

    @staticmethod
    def _manifest_root_matches(root: str, manifest: Mapping[str, Any]) -> bool:
        """Whether the manifest's identifying content still hashes to ``root``.

        Without this the store is content-*named* rather than content-addressed:
        the digest is only ever the filename, so editing ``params`` (or writing a
        wholly fabricated manifest under a 64-hex name) leaves nothing to detect.

        The manifest already holds the *output* of :func:`canonical_json`, so it
        is re-encoded with the same JSON settings rather than pushed through
        :func:`_canonicalise` a second time. Canonicalisation is deliberately not
        idempotent -- a string that spells a sentinel is escaped on every pass --
        so re-canonicalising here would fail every artifact whose parameters
        contain an array, a NaN, or a ``__``-prefixed string.
        """
        identifying = ("estimator", "estimator_version", "params", "inputs", "env", "seed")
        if any(k not in manifest for k in identifying):
            return False
        return digest_document({k: manifest[k] for k in identifying}) == root

    def verify(self, root: str) -> bool:
        """Re-hash the manifest and the stored arrays against the recorded digests.

        Returns ``False`` for a manifest whose identifying fields no longer hash
        to its own address, as well as for arrays that no longer match.
        """
        manifest = self.get_manifest(root)
        if len(root) != _DIGEST_CHARS or not self._manifest_root_matches(root, manifest):
            return False
        recorded: Mapping[str, str] = manifest.get("arrays", {})
        if not recorded:
            return True
        try:
            stored = self.get_arrays(root)
        except (OSError, ValueError, zipfile.BadZipFile):
            # A truncated or bit-flipped .npz is exactly the corruption this
            # method exists to report; it must be a False, never a traceback.
            return False
        if set(stored) != set(recorded):
            return False
        return all(array_digest(stored[k]) == d for k, d in recorded.items())

    def list_roots(self) -> list[str]:
        if not self.manifests.exists():
            return []
        return sorted(p.stem for p in self.manifests.glob("*.json"))
