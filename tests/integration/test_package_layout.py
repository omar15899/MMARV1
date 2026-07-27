"""Import-time hazards. These break at runtime, far from their cause.

The package namespace is a shared surface: exporting a function under the same
name as a submodule silently rebinds the attribute, and every `from package
import submodule` in the codebase then yields a callable. Nothing warns.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

import pytest


def test_the_fbm_name_is_deliberately_the_function() -> None:
    """`mmar.models.fbm` resolves to the generator, not the module.

    That is the intended API -- but it means internal code must import from the
    submodule (`from mmar.models.fbm import fbm`) and never by attribute
    (`from mmar.models import fbm as module`). Regression: mmar/models/mmar.py
    did the latter and every simulation raised
    AttributeError: 'function' object has no attribute 'fbm'.
    """
    from mmar import models

    assert callable(models.fbm)
    from mmar.models.fbm import fbm as fbm_fn

    assert models.fbm is fbm_fn
    assert isinstance(sys.modules["mmar.models.fbm"], type(sys))


# Deliberate: `from .mfdfa import mfdfa` is the ordinary Python idiom, and the
# function is the name callers want. Each entry is a place where internal code
# must import from the submodule rather than by attribute.
INTENDED_SHADOWING = {
    ("mmar.models", "fbm"),
    ("mmar.estimators", "chhabra_jensen"),
    ("mmar.estimators", "mfdfa"),
    ("mmar.estimators", "mfdma"),
    ("mmar.estimators", "wavelet_leaders"),
    ("mmar.estimators", "wtmm"),
}


def test_submodule_shadowing_is_exactly_the_intended_set() -> None:
    """A new collision must be added here on purpose, having checked its callers."""
    found = set()
    for pkg_name in ("mmar", "mmar.core", "mmar.models", "mmar.estimators", "mmar.inference"):
        pkg = importlib.import_module(pkg_name)
        submodules = {m.name for m in pkgutil.iter_modules(pkg.__path__)}
        for name in getattr(pkg, "__all__", []):
            if name in submodules and not isinstance(getattr(pkg, name, None), type(sys)):
                found.add((pkg_name, name))
    assert found == INTENDED_SHADOWING, (
        f"new shadowing {found - INTENDED_SHADOWING}, "
        f"resolved shadowing {INTENDED_SHADOWING - found}"
    )


@pytest.mark.parametrize(
    ("pkg_name", "name"), [("mmar.models", "cascade"), ("mmar.models", "theory")]
)
def test_modules_imported_by_attribute_must_stay_modules(pkg_name: str, name: str) -> None:
    """Internal code does `from mmar.models import cascade as _cascade_mod`.

    Exporting a function under either name would rebind the attribute and break
    every such call site at runtime, with an AttributeError far from the cause.
    """
    pkg = importlib.import_module(pkg_name)
    assert isinstance(getattr(pkg, name), type(sys)), (
        f"{pkg_name}.{name} is no longer a module; internal call sites import it by attribute"
    )


def test_every_shadowed_submodule_is_still_reachable() -> None:
    """Shadowing hides the attribute, not the module: the import form still works."""
    for pkg_name, name in sorted(INTENDED_SHADOWING):
        module = importlib.import_module(f"{pkg_name}.{name}")
        assert isinstance(module, type(sys))
        assert hasattr(module, name), f"{pkg_name}.{name}.{name} should be the callable"


@pytest.mark.parametrize(
    "module",
    [
        "mmar",
        "mmar.analysis",
        "mmar.calibration",
        "mmar.cli",
        "mmar.core",
        "mmar.estimators",
        "mmar.inference",
        "mmar.io",
        "mmar.models",
        "mmar.tracing",
        "mmar.viz",
    ],
)
def test_every_public_subpackage_imports(module: str) -> None:
    importlib.import_module(module)


@pytest.mark.parametrize("module", ["mmar", "mmar.core", "mmar.models", "mmar.estimators"])
def test_dunder_all_entries_all_resolve(module: str) -> None:
    """A name in __all__ that does not exist breaks `from module import *`."""
    mod = importlib.import_module(module)
    missing = [n for n in getattr(mod, "__all__", []) if not hasattr(mod, n)]
    assert not missing, f"{module}.__all__ names that do not resolve: {missing}"
