"""Seeded randomness with a recoverable seed record.

Nothing in this library calls the global :mod:`numpy.random` state. Every
stochastic routine takes a ``seed`` and threads a :class:`numpy.random.Generator`
through, so that a :class:`~mmar.core.provenance.SeedInfo` in a manifest is
sufficient to reproduce the exact stream. Parallel replicas are derived by
:meth:`numpy.random.SeedSequence.spawn`, never by incrementing an integer seed
(sequential integer seeds produce correlated PCG64 streams).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mmar.core.provenance import SeedInfo

__all__ = ["SeedLike", "make_rng", "seed_info", "spawn_rngs"]

SeedLike = int | Sequence[int] | np.random.SeedSequence | np.random.Generator | None


def _to_seed_sequence(seed: SeedLike) -> np.random.SeedSequence:
    if isinstance(seed, np.random.SeedSequence):
        return seed
    if isinstance(seed, np.random.Generator):
        raise TypeError("a Generator carries no recoverable seed; pass the seed instead")
    if seed is None:
        return np.random.SeedSequence()
    if isinstance(seed, int):
        return np.random.SeedSequence(seed)
    return np.random.SeedSequence(list(seed))


def make_rng(seed: SeedLike) -> np.random.Generator:
    """Build a generator. Passing an existing ``Generator`` returns it unchanged.

    Accepting a live generator lets callers thread one stream through a whole
    pipeline; accepting an int gives bit-for-bit reproducibility.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.Generator(np.random.PCG64(_to_seed_sequence(seed)))


def seed_info(seed: SeedLike) -> SeedInfo | None:
    """The recoverable seed record, or ``None`` when the stream is not recoverable.

    ``None`` is returned for a live :class:`numpy.random.Generator` *and* for
    ``seed=None``. In the latter case every consumer builds its own
    :class:`~numpy.random.SeedSequence` from operating-system entropy, so a
    :class:`SeedInfo` minted here would be an *unrelated* draw: writing it into
    a manifest would claim a reproducibility that regenerates a different
    stream. An absent seed is recorded as absent.
    """
    if seed is None or isinstance(seed, np.random.Generator):
        return None
    ss = _to_seed_sequence(seed)
    entropy = ss.entropy
    if isinstance(entropy, Sequence) and not isinstance(entropy, str | bytes):
        recorded: int | tuple[int, ...] | None = tuple(int(e) for e in entropy)
    elif entropy is None:
        recorded = None
    else:
        recorded = int(entropy)
    return SeedInfo(
        entropy=recorded,
        spawn_key=tuple(int(k) for k in ss.spawn_key),
        bit_generator="PCG64",
    )


def spawn_rngs(seed: SeedLike, n: int) -> list[np.random.Generator]:
    """``n`` statistically independent generators derived from one seed.

    Use for Monte Carlo replicas, bootstrap resamples and surrogate ensembles:
    the streams are guaranteed non-overlapping and the whole ensemble is
    reproducible from the single parent seed.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if isinstance(seed, np.random.Generator):
        parent = seed.bit_generator.seed_seq
        if not isinstance(parent, np.random.SeedSequence):
            raise TypeError(
                f"cannot spawn from a {type(parent).__name__}; the generator was built from a "
                "custom ISeedSequence that does not support spawning. Pass an integer seed."
            )
        children = parent.spawn(n)
    else:
        children = _to_seed_sequence(seed).spawn(n)
    return [np.random.Generator(np.random.PCG64(child)) for child in children]
