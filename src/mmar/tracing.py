"""Session-level traceability: a Merkle DAG over everything an analysis computed.

:class:`~mmar.core.provenance.Provenance` makes a *single* result auditable. This
module makes a *session* auditable. A :class:`Run` records every result produced
inside it, links each to the results it consumed, and closes with one root hash
covering the whole computation.

That buys three things that matter for research code:

**Reproduction.** The manifest carries every parameter, every seed and the
environment. :func:`replay_plan` prints the exact call sequence needed to
regenerate the run.

**Verification.** :meth:`Run.verify` re-hashes the stored arrays against their
recorded digests, so silent corruption or a hand-edited artifact is detectable.

**Attribution.** :func:`diff_manifests` answers the question that actually comes
up six months later: *these two runs disagree — what changed?* It reports the
differing parameters, not just that the roots differ.

Example
-------
>>> from mmar.tracing import Run
>>> with Run("lockheed-h63", store="artifacts") as run:            # doctest: +SKIP
...     exps = run.record(mfdfa(x, q=q), name="mfdfa")
...     spec = run.record(spectrum_from_exponents(exps), name="spectrum")
>>> run.manifest_root                                              # doctest: +SKIP
'3f9a…'
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from mmar.core.provenance import (
    ArtifactStore,
    EnvFingerprint,
    Provenance,
    canonical_json,
    digest_mapping,
)

__all__ = [
    "HasProvenance",
    "Run",
    "RunManifest",
    "TraceNode",
    "diff_manifests",
    "load_manifest",
    "replay_plan",
]


@runtime_checkable
class HasProvenance(Protocol):
    """Anything this library produces: a result object carrying an audit record."""

    provenance: Provenance | None


@dataclass(frozen=True, slots=True)
class TraceNode:
    """One computation in the session DAG."""

    name: str
    root: str
    estimator: str
    estimator_version: str
    params: Mapping[str, Any]
    parents: tuple[str, ...]
    """Content addresses of the inputs — the DAG edges."""
    summary: Mapping[str, Any] = field(default_factory=dict)
    array_digests: Mapping[str, str] = field(default_factory=dict)
    created_utc: str = ""

    def to_canonical(self) -> dict[str, Any]:
        """Identifying content of the node.

        ``summary`` is included: it holds the numbers a reader of the manifest
        actually acts on, so leaving it out of the Merkle root would let anyone
        rewrite a reported ``h2`` in the JSON and still pass
        :func:`load_manifest`. It is deterministic given the node's root, so it
        does not disturb session-root reproducibility. ``created_utc`` stays out
        — wall-clock time must never enter a digest.
        """
        return {
            "name": self.name,
            "root": self.root,
            "estimator": self.estimator,
            "estimator_version": self.estimator_version,
            "params": dict(self.params),
            "parents": list(self.parents),
            "summary": dict(self.summary),
        }


@dataclass(frozen=True, slots=True)
class RunManifest:
    """The complete, serialisable record of one analysis session."""

    label: str
    nodes: tuple[TraceNode, ...]
    env: EnvFingerprint
    started_utc: str
    finished_utc: str
    notes: tuple[str, ...] = ()

    @property
    def root(self) -> str:
        """Merkle root over the session: changes if any node's identity changes.

        Computed from the *identifying* part of each node only, so re-running the
        same pipeline on the same inputs yields the same session root even though
        the timestamps differ.
        """
        return digest_mapping(
            {
                "label": self.label,
                "nodes": [n.to_canonical() for n in self.nodes],
                "env": self.env.to_canonical(),
            }
        )

    @property
    def short_root(self) -> str:
        return self.root[:12]

    def by_name(self, name: str) -> TraceNode:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(f"no node named {name!r}; have {[n.name for n in self.nodes]}")

    def leaves(self) -> tuple[TraceNode, ...]:
        """Nodes nothing else consumed — the outputs of the session."""
        consumed = {p for n in self.nodes for p in n.parents}
        return tuple(n for n in self.nodes if n.root not in consumed)

    def ancestors(self, name: str) -> tuple[TraceNode, ...]:
        """Every node the named result transitively depends on, deepest last."""
        by_root = {n.root: n for n in self.nodes}
        seen: dict[str, TraceNode] = {}
        stack = list(self.by_name(name).parents)
        while stack:
            r = stack.pop()
            node = by_root.get(r)
            if node is None or r in seen:
                continue
            seen[r] = node
            stack.extend(node.parents)
        return tuple(seen.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_root": self.root,
            "label": self.label,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "env": json.loads(canonical_json(self.env.to_canonical())),
            "notes": list(self.notes),
            "nodes": [
                {
                    **json.loads(canonical_json(n.to_canonical())),
                    "summary": json.loads(canonical_json(dict(n.summary))),
                    "array_digests": dict(n.array_digests),
                    "created_utc": n.created_utc,
                }
                for n in self.nodes
            ],
        }

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return p

    def describe(self) -> str:
        """Human-readable rendering of the DAG, for a terminal or a log."""
        lines = [
            f"run {self.label}  root={self.short_root}",
            f"  env: python {self.env.python}, numpy {self.env.numpy}, "
            f"scipy {self.env.scipy}, mmar {self.env.library}",
            f"  {len(self.nodes)} node(s), {self.started_utc} -> {self.finished_utc}",
        ]
        by_root = {n.root: n for n in self.nodes}
        for n in self.nodes:
            parents = ", ".join(by_root[p].name if p in by_root else p[:8] for p in n.parents)
            lines.append(f"  - {n.name:<22} {n.estimator}@{n.estimator_version}  [{n.root[:8]}]")
            if parents:
                lines.append(f"      <- {parents}")
            for k, v in sorted(n.summary.items()):
                lines.append(f"      {k} = {v}")
        return "\n".join(lines)


def load_manifest(path: str | Path) -> RunManifest:
    """Read a manifest back. The reconstructed root must match the recorded one."""
    raw = json.loads(Path(path).read_text())
    env = EnvFingerprint(**raw["env"])
    nodes = tuple(
        TraceNode(
            name=n["name"],
            root=n["root"],
            estimator=n["estimator"],
            estimator_version=n["estimator_version"],
            params=n["params"],
            parents=tuple(n["parents"]),
            summary=n.get("summary", {}),
            array_digests=n.get("array_digests", {}),
            created_utc=n.get("created_utc", ""),
        )
        for n in raw["nodes"]
    )
    manifest = RunManifest(
        label=raw["label"],
        nodes=nodes,
        env=env,
        started_utc=raw["started_utc"],
        finished_utc=raw["finished_utc"],
        notes=tuple(raw.get("notes", ())),
    )
    if manifest.root != raw["manifest_root"]:
        raise ValueError(
            f"manifest root mismatch: file records {raw['manifest_root'][:12]}, "
            f"recomputes to {manifest.short_root}. The file has been altered."
        )
    return manifest


class Run:
    """Context manager that records every result computed inside it.

    Parameters
    ----------
    label
        Name of the session; appears in the manifest and in the artifact path.
    store
        Directory for the content-addressed artifact store, or an existing
        :class:`~mmar.core.provenance.ArtifactStore`. ``None`` records the DAG in
        memory only — useful in tests, where the digests still matter but the
        bytes do not.
    persist_arrays
        Whether numeric payloads are written alongside the manifests. Off makes
        the run cheap while keeping it fully attributable.
    """

    def __init__(
        self,
        label: str,
        *,
        store: str | Path | ArtifactStore | None = None,
        persist_arrays: bool = True,
        notes: Sequence[str] = (),
    ) -> None:
        self.label = label
        self.store = ArtifactStore(store) if isinstance(store, str | Path) else store
        self.persist_arrays = persist_arrays and self.store is not None
        self.notes = tuple(notes)
        self._nodes: list[TraceNode] = []
        self._started = ""
        self._finished = ""
        self._env = EnvFingerprint.capture()

    def __enter__(self) -> Run:
        self._started = datetime.now(UTC).isoformat(timespec="seconds")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._finished = datetime.now(UTC).isoformat(timespec="seconds")
        if exc_type is None and self.store is not None:
            self.manifest.write(Path(self.store.root_dir) / f"run-{self.label}.json")

    def record(
        self,
        result: Any,
        *,
        name: str,
        arrays: Mapping[str, NDArray[Any]] | None = None,
        summary: Mapping[str, Any] | None = None,
    ) -> Any:
        """Register a result in the session DAG and return it unchanged.

        Designed to wrap a call site without disturbing it::

            exps = run.record(mfdfa(x, q=q), name="mfdfa")

        Raises
        ------
        TypeError
            If the object carries no provenance. An untraceable result has no
            place in a traced run; that is the point of the module.
        ValueError
            If ``name`` is already used in this run. Manifest queries resolve a
            name to the first matching node, so a duplicate is an ambiguous
            record rather than a second entry.
        """
        if any(n.name == name for n in self._nodes):
            raise ValueError(
                f"this run already contains a node named {name!r}. Node names are how a "
                "manifest is queried (RunManifest.by_name, ancestors, diff_manifests), all "
                "of which return the first match, so a duplicate makes the record ambiguous "
                "and hides the second result. Give it a distinct name."
            )
        prov = getattr(result, "provenance", None)
        if not isinstance(prov, Provenance):
            raise TypeError(
                f"{type(result).__name__} carries no Provenance, so it cannot be recorded. "
                "Every estimator in this library attaches one; if you built the object by "
                "hand, construct it with new_provenance(...)."
            )
        payload = dict(arrays or {}) or _default_arrays(result)
        digests: Mapping[str, str] = {}
        if self.store is not None:
            self.store.put(
                prov,
                arrays=payload if self.persist_arrays else None,
                summary=summary or _default_summary(result),
            )
            manifest = self.store.get_manifest(prov.root)
            digests = manifest.get("arrays", {})

        self._nodes.append(
            TraceNode(
                name=name,
                root=prov.root,
                estimator=prov.estimator,
                estimator_version=prov.estimator_version,
                # Canonicalise on the way in, so an in-memory manifest and one
                # round-tripped through JSON compare equal parameter by parameter.
                params=json.loads(canonical_json(dict(prov.params))),
                parents=tuple(i.digest for i in prov.inputs),
                summary=dict(summary or _default_summary(result)),
                array_digests=dict(digests),
                created_utc=prov.created_utc,
            )
        )
        return result

    @property
    def manifest(self) -> RunManifest:
        return RunManifest(
            label=self.label,
            nodes=tuple(self._nodes),
            env=self._env,
            started_utc=self._started,
            finished_utc=self._finished or datetime.now(UTC).isoformat(timespec="seconds"),
            notes=self.notes,
        )

    @property
    def manifest_root(self) -> str:
        return self.manifest.root

    def verify(self) -> dict[str, bool]:
        """Re-hash every stored artifact against its recorded digest."""
        if self.store is None:
            return {}
        return {n.name: self.store.verify(n.root) for n in self._nodes}

    def __iter__(self) -> Iterator[TraceNode]:
        return iter(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)


def _default_summary(result: Any) -> dict[str, Any]:
    """Scalar highlights worth putting in the manifest, per result type."""
    out: dict[str, Any] = {"type": type(result).__name__}
    if hasattr(result, "summary") and callable(result.summary):
        try:
            out.update(result.summary())
            return out
        except (TypeError, ValueError, AttributeError):  # pragma: no cover
            pass
    for attr in ("h2", "hurst_tau_root", "alpha_0", "c1", "c2", "estimator"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            if isinstance(value, float | int | str):
                out[attr] = value
    if hasattr(result, "descriptors") and callable(result.descriptors):
        with contextlib.suppress(TypeError, ValueError, AttributeError):
            out.update(result.descriptors().to_canonical())
    return out


def _default_arrays(result: Any) -> dict[str, NDArray[Any]]:
    """The numeric payload worth persisting, per result type."""
    out: dict[str, NDArray[Any]] = {}
    for attr in ("q", "zeta", "alpha", "f_alpha", "h", "t", "c"):
        value = getattr(result, attr, None)
        if isinstance(value, np.ndarray):
            out[attr] = value
    return out


def diff_manifests(a: RunManifest, b: RunManifest) -> dict[str, Any]:
    """Explain why two runs differ, at the level of individual parameters.

    Returns the nodes present in only one run, and for nodes sharing a name but
    not a root, exactly which parameters changed. This is the question that
    actually gets asked when two analyses disagree.
    """
    names_a = {n.name for n in a.nodes}
    names_b = {n.name for n in b.nodes}
    report: dict[str, Any] = {
        "root_a": a.root,
        "root_b": b.root,
        "identical": a.root == b.root,
        "only_in_a": sorted(names_a - names_b),
        "only_in_b": sorted(names_b - names_a),
        "changed": {},
        "env_changed": {},
    }
    env_a, env_b = a.env.to_canonical(), b.env.to_canonical()
    for k in env_a:
        if env_a[k] != env_b.get(k):
            report["env_changed"][k] = {"a": env_a[k], "b": env_b.get(k)}

    for name in sorted(names_a & names_b):
        na, nb = a.by_name(name), b.by_name(name)
        if na.root == nb.root:
            continue
        pa, pb = dict(na.params), dict(nb.params)
        changed = {
            k: {"a": pa.get(k), "b": pb.get(k)}
            for k in sorted(set(pa) | set(pb))
            if pa.get(k) != pb.get(k)
        }
        report["changed"][name] = {
            "root_a": na.root[:12],
            "root_b": nb.root[:12],
            "params": changed,
            "estimator_version": (
                None
                if na.estimator_version == nb.estimator_version
                else {"a": na.estimator_version, "b": nb.estimator_version}
            ),
            "parents_differ": na.parents != nb.parents,
        }
    return report


def replay_plan(manifest: RunManifest) -> str:
    """The call sequence needed to regenerate the run, in dependency order.

    Not executable code — parameters are canonicalised, not eval-able — but a
    complete and unambiguous specification of what to run and with what.
    """
    lines = [
        f"# replay plan for run {manifest.label}  (root {manifest.short_root})",
        f"# requires: python {manifest.env.python}, numpy {manifest.env.numpy}, "
        f"scipy {manifest.env.scipy}, mmar {manifest.env.library}",
        "",
    ]
    by_root = {n.root: n for n in manifest.nodes}
    emitted: set[str] = set()

    def emit(node: TraceNode) -> None:
        if node.root in emitted:
            return
        emitted.add(node.root)
        for p in node.parents:
            parent = by_root.get(p)
            if parent is not None:
                emit(parent)
        inputs = ", ".join(by_root[p].name if p in by_root else f"<{p[:8]}>" for p in node.parents)
        params = ", ".join(f"{k}={v!r}" for k, v in sorted(node.params.items()))
        lines.append(
            f"{node.name} = {node.estimator}({inputs}{', ' if inputs and params else ''}{params})"
        )
        lines.append(f"#   expect root {node.root}")

    for node in manifest.nodes:
        emit(node)
    return "\n".join(lines)
