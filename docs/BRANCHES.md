# Branch topology of MMARV1

Reconstructed 2026-07-27 from `git log --all --graph` and pairwise diffs against
`main`. Five branches exist; none is a superset of the others, and the branch you
would naturally check out (`main`) is the one with the least content.

## The shape

```
  ● 04e2de1  main ← origin/HEAD          .gitignore restored
  ● 7fec3ce                              .gitignore added
  ● c2744b6  Readme.md rewritten
  ● 325e151 · 38a52ef · 79e2eda · 7b29776
  ● 5949461                              "minor changes for public presentation"   ← THE STRIP
  ● 22ce03d · e2aaf75 · 000f226 · 59fca55 · 036a400 · e27965c · f21e56b
  │
  ● 2824af8  o15899mod                   "mejoras a las tau y a la medida multifractalbinomial"
  ● 19f3241
  ●   b97b940  merge
  ├─● 4f737bf  code_suggestions          "cambios menores, ha de seguir siendo mejorado"
  │ ● f0e9df3  reestructuracion
  │ ●   843e999  merge
  │ ├─● 462ae49 · d0c26b4 · e5493da · 872dce8 · 124c743
  ● c8a60da                              git-graph extension
  │ ● 22b32af  wip-multifractal-pre-refactor-2026-05-12
  │ ● 991b804                            WIP constructor changes
  │
  ● e899ffd  changes                     "ponerlo todo al día para ya empezar bien"
```

`origin/changes` is detached from the others — it shares no recent ancestor with
`main`'s tip and is best read as an independent snapshot.

## What each branch actually holds

| Branch | Tip | `multifractalcharacteristics.py` | Thesis assets | Verdict |
|---|---|---|---|---|
| `main` | 04e2de1 | 469 L | **deleted** | public, stripped, cleanest prose |
| `o15899mod` | 2824af8 | 590 L | present | **most mathematically advanced** |
| `code_suggestions` | 4f737bf | 626 L | present | largest, most commentary |
| `wip-multifractal-pre-refactor-…` | 22b32af | 593 L | present | abandoned refactor baseline |
| `changes` | e899ffd | 359 L | present | oldest / smallest |

## The strip

Commit `5949461`, *"some minor changes for public presentation"*, is not minor.
It removed from the lineage that became `main`:

- `Scripts/Lockheed.txt` — **the dataset**. Lockheed Martin daily OHLCV,
  7 049 rows, 1995-03-17 → 2023-03-17. Without it none of the thesis results can
  be reproduced.
- `Scripts/TFG_simulacion_completa.ipynb` — 9.0 MB, 1 841 cells. The complete
  thesis notebook.
- `Scripts/pruebaMMAR1.py`, `Scripts/prueba_det1.py` — the only working examples
  of how to call the library.
- `MMAR/README.txt` — the bibliography (Mandelbrot, Calvet & Fisher, Evertsz,
  Meneveau & Sreenivasan, Falconer, Øksendal …). This is the most valuable prose
  in the repository and `main`'s replacement `Readme.md` does not contain it.

`main`'s `Readme.md` was written *after* the strip and documents a command-line
interface that has never existed in any branch (see `LEGACY_AUDIT.md` D14).

Net effect: `main` is the branch a reader lands on, and it is the branch from
which the data, the examples and the references were removed.

## Recovery

Nothing was lost — the objects live on the other branches. This rebuild restores:

| Restored to | From |
|---|---|
| `data/lockheed_1995_2023.tsv` | `origin/o15899mod:Scripts/Lockheed.txt` |
| `docs/BIBLIOGRAPHY_legacy.md` | `origin/o15899mod:MMAR/README.txt` |
| `legacy/MMAR/**` | `main` (the last coherent v1 tree) |

## Divergence between the four non-`main` branches

The four are variations on the same intent, differing mainly in how much
explanatory commentary was pasted in:

- **`code_suggestions`** (626 L) is the most heavily annotated. Its extra 36
  lines over `o15899mod` are commentary, not mathematics.
- **`o15899mod`** inlined `multifractal_measure_rand2` directly into
  `MultifractalCharacteristics.multifractal_measure_rand`, removing the
  cross-module call. It also added the NaN guard
  `if np.isnan(y).all(): raise ValueError` in the plotting path and reordered
  `masas1` to force `masas2 = False`, fixing part of D10 — but only along the
  `masas1=True` path, so `masas1=False, masas2=False` still raises
  `UnboundLocalError`. This is the branch whose commit message advertises
  "mejoras a las tau"; the improvements are real but do not touch D2–D6.
- **`wip-multifractal-pre-refactor-2026-05-12`** is a checkpoint taken before a
  refactor that was never carried out. `991b804` is explicitly labelled WIP.
- **`changes`** predates the other three.

**No branch fixes any of the four critical defects.** D1 (the model is never
composed), D2 (`q ≥ 0` only), D3 and D4 (the confidence intervals) are present,
identically, in all five.

## Hygiene

Every branch except `main` tracks compiled artefacts and OS metadata:

```
.DS_Store  ×5     __pycache__/*.pyc  ×9 per branch     .vscode/  MMARV1.code-workspace
```

`origin/changes` and `origin/o15899mod` additionally *delete* `.gitignore`, which
is why those artefacts are tracked there. `main` restored it in `04e2de1` — the
most recent commit in the repository is a `.gitignore` repair.

## Recommended disposition

1. **Keep** `main` as the published history. This rebuild branches from it.
2. **Keep** `o15899mod` indefinitely as the archival tip: it holds the dataset,
   the notebook and the bibliography, and it is the reference for what the thesis
   actually ran.
3. **Retire** `code_suggestions`, `wip-multifractal-pre-refactor-2026-05-12` and
   `changes` once this rebuild merges — they contain no content absent from
   `o15899mod` beyond commentary. Tag before deleting:
   `git tag archive/<branch> <sha>`.
4. **Never** delete a branch before confirming `git diff` against `o15899mod` is
   commentary-only. The strip at `5949461` is the cautionary precedent.

Recovery of anything mentioned here:

```sh
git show origin/o15899mod:Scripts/Lockheed.txt > lockheed.tsv
git show origin/o15899mod:Scripts/TFG_simulacion_completa.ipynb > thesis.ipynb
git show origin/code_suggestions:MMAR/MMAR_Simulation/multifractalcharacteristics.py
```
