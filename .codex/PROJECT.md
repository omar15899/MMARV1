# Contexto técnico de MMARV1

Referencia de dominio preservada al migrar el perfil el 2026-09-17.
Las versiones, estados de ramas, resultados de tests y rutas de esta referencia
son observaciones del inventario anterior, no comprobaciones actuales. Contrasta
lo relevante con el código, scripts y CI antes de usarlo.
El contrato operativo vigente es [AGENTS.md](../AGENTS.md). Las menciones a
agentes o comandos Claude describen capacidades y comprobaciones de dominio;
Codex las realiza directamente o delega cuando aporte valor, sin importar
cadenas, permisos ni publicación automática de otro cliente.

## Identidad

MMARV1 is a standalone scientific Python library — package name `mmar`, version 2.0.0 — implementing the multifractal formalism: Hölder/singularity spectra, multiplicative cascade models, and the Mandelbrot–Fisher–Calvet Multifractal Model of Asset Returns. It ships nine estimator families behind one interface (wavelet leaders, MF-DFA, MF-DMA, WTMM, structure functions, Chhabra–Jensen, box counting, generalised Hurst, and pointwise h(t)), a closed-form theory oracle every estimator is tested against, a surrogate-based inference layer whose job is to say "no evidence of multifractality" when that is the truth, and content-addressed SHA-256 provenance on every number it produces. In the constellation it sits in the `adjacent` perimeter with `tier: null` and `status: external` — explicitly NOT bound by QuantOS tier discipline, with no .importlinter and no dependency on any QuantOS lab. It is a ground-up v2 rebuild of Omar's thesis code, which is preserved verbatim and importable under `legacy/` so the thesis results stay auditable.

## Fuentes del dominio

- README.md

- docs/ESTIMATOR_GUIDE.md

- docs/KNOWN_LIMITATIONS.md

- docs/LEGACY_AUDIT.md

- docs/MIGRATION.md

- docs/BRANCHES.md

## Invariantes

- One convention carrier, enforced by the types. README: "`ScalingExponents` stores `ζ(q)` and derives `τ = ζ − d` and `h(q) = ζ(q)/q`. A dropped `±1` between the measure and signal conventions is the most common error in this field, and it cannot happen here because there is only one carrier." Do not introduce a second representation of the scaling exponents.

- Provenance roots must be reproducible and must never depend on wall-clock time. README: the `root` is "a SHA-256 over the input digests, every parameter, the RNG seed and the environment — never the wall clock, so reproducibility is *testable*." CI enforces it: two identical `mmar analyze` invocations must yield byte-identical `root`, and `mmar verify` must re-hash a stored run against its manifest.

- Theory anchors must hold exactly, not approximately. CI runs a dedicated job step "theory anchors must hold exactly" (`uv run pytest -m theory -q`); README pins them as "exact to machine precision on `τ(0) = −1`, `τ(1) = 0`, `f(α₀) = 1` and `τ_X(1/H) = 0`."

- mypy runs in strict mode (`[tool.mypy] strict = true`, `warn_unreachable = true`, `disallow_any_generics = true`) and CI runs `uv run mypy` as a blocking gate. Do not add `# type: ignore` to pass; `ignore_missing_imports` is granted only to pywt/matplotlib/pandas/scipy.

- Every measured claim must be produced by running the code, not inferred from reading it. docs/ESTIMATOR_GUIDE.md: "Every number below was **measured**, not quoted from a paper." docs/LEGACY_AUDIT.md: "Every claim below that carries a number was **observed by running the code**, not inferred from reading it." The probes that produced them are preserved at docs/probes/.

- Known defects are recorded, not silently patched or silently tolerated. docs/KNOWN_LIMITATIONS.md exists for "findings reported and deliberately **not** fixed, with the reason" — "written down so they are decisions, not oversights." Its eight entries each name the reproducing test. Fixing one requires updating both the doc and the corresponding strict xfail.

- `legacy/` is frozen v1, preserved verbatim so thesis results remain reproducible. Do not refactor, lint or modernise it.

- Constellation git policy applies (inherited from ~/GitHub/.claude/CLAUDE.md): atomic commits per repo, never bundled across repos; no `--no-verify`; no force-push to main.

## Comandos de referencia

- uv sync --all-extras

- uv run ruff check src tests examples

- uv run ruff format --check src tests examples

- uv run mypy

- uv run pytest -m "not slow" --cov=mmar --cov-report=term-missing --cov-report=xml

- uv run pytest -m theory -q

- uv run pytest -m slow -q

- uv run mmar analyze --estimator mfdfa --q=-3:3:9 --json-out a.json

- uv run mmar analyze --estimator mfdfa --q=-3:3:9 --store artifacts --label ci

- uv run mmar verify artifacts/run-ci.json

- pip install -e ".[full]"

## Observaciones técnicas

- `[tool.pytest.ini_options] filterwarnings = ["error::RuntimeWarning"]` — any RuntimeWarning (overflow, divide-by-zero, invalid value in numpy) fails the test rather than printing. Numerical code must guard these explicitly.

- `tests/adversarial/test_degenerate_inputs.py` carries four `@pytest.mark.xfail(strict=True)` tests that document OPEN DEFECTS (e.g. wavelet_leaders fitting float64 round-off on a straight line, reported in docs/KNOWN_LIMITATIONS.md §2). Because they are strict, actually fixing the defect makes the suite FAIL with XPASS — the xfail must be removed in the same change as the fix.

- CI (.github/workflows/ci.yml) only triggers on push to `main`, `feat/**` or `fix/**` (plus pull_request and workflow_dispatch). A branch named anything else gets no CI on push. The `slow` job additionally runs only on `workflow_dispatch` or `refs/heads/main`.

- CI sets `env: UV_FROZEN: "1"`, so uv.lock must be in sync with pyproject.toml or `uv sync` fails in CI.

- `uv run mypy` takes no path arguments — it works because `[tool.mypy]` sets `packages = ["mmar"]` and `mypy_path = "src"`. Passing paths explicitly changes what is checked.

- Version skew is deliberate but confusing: `.python-version` is 3.13 (local .venv is Python 3.13.3), `[tool.mypy] python_version` is 3.12, `requires-python >= 3.12`, and CI tests a 3.12/3.13 matrix.

- `artifacts/` is in .gitignore — provenance run manifests written by `mmar analyze --store artifacts` are never committed. The CI reproducibility job regenerates them each run.

- No build command is configured anywhere (no Makefile, no build step in CI). The build backend is hatchling with `[tool.hatch.build.targets.wheel] packages = ["src/mmar"]`; nothing in the repo invokes it.

- `--q` must use the `=` form when the lower bound is negative: `--q=-5:5:41`, not `--q -5:5:41`, or argparse reads it as a flag. This is commented in src/mmar/cli.py.

- The CLI defines non-standard exit codes (declared in the parser epilog): 0 success, 1 error, 2 argparse usage, 3 missing optional dependency, 4 calibration inadmissible, 5 stored artifacts failed verification. Do not treat any non-zero code as a generic failure.

- v1 code is duplicated on disk: `MMAR/` at the repo root and `legacy/MMAR/` are BOTH tracked in git. `legacy/` is the canonical preserved copy referenced by docs/MIGRATION.md; the root `MMAR/` is the un-removed original.

- `tests/property/` and `tests/regression/` directories exist but are empty (no test files).

- The tree churns build artefacts badly enough that Spotlight indexing pinned mds_stores at ~200% CPU; `.metadata_never_index` and a .gitignore note exist as mitigation. Do not remove them.

- Verified-green baseline as of this survey (branch feat/holder-engine, one modified file tests/adversarial/test_mutation_coverage.py): ruff check "All checks passed!", ruff format --check "76 files already formatted", mypy "Success: no issues found in 38 source files", pytest collects 1432 tests across 37 test files.
