# Known limitations

Findings from the adversarial pass that were **reported and not fixed**, each
because the remedy is a design decision rather than a mechanical repair. They
are written down so they are decisions, not oversights. Every one is
reproducible from the test named beside it.

Measured biases that are properties of the *methods* rather than of this
implementation live in [`ESTIMATOR_GUIDE.md`](ESTIMATOR_GUIDE.md).

---

## 1. A large offset silently halves the Hurst exponent

**`tests/adversarial/test_numerical_extremes.py`**

A level series whose increments sit near the ulp of a large offset — the
classic float64 log-price trap — is censored scale-dependently and the censoring
lands directly in the slope.

```
x = fbm(4096, 0.7, seed=3) / std(diff(...))
mfc_exponents(1e-9 * x).h2         = 0.724
mfc_exponents(1e8 + 1e-9 * x).h2   = 0.386      # same process, offset added
```

At the finest lag 77 % of increments are exactly zero and dropped; at the
coarsest, none are. Only `mfdfa` refuses outright. The secondary diagnostics do
fire (`concavity_defect_in_stderr = 11.0`, `f_max = 0.558 < d = 1`) but the
headline `h2`/`alpha_0`/`width` are reported as ordinary numbers.

*Why not fixed:* the natural remedy is to warn above a drop-fraction threshold,
and choosing that threshold is a policy call. **Mitigation today:** the count is
in `provenance.diagnostics["zeros_dropped"]`. Centre and scale a level series
before analysing it.

## 2. Wavelet estimators fabricate a spectrum from round-off on a straight line

**`tests/adversarial/test_degenerate_inputs.py`** (strict xfail)

The sharpest finding of the adversarial pass, and it hits the **default**
estimator. Where the analysing wavelet annihilates the signal exactly — any
polynomial of degree below its vanishing moments — the wavelet families fit
float64 round-off and return a confident, fully formed multifractal spectrum
for a perfectly smooth function.

```python
ramp = np.arange(4096, dtype=float) + 1.0     # alpha(t) = 1 everywhere, width 0

analyze(ramp, estimator="wavelet_leaders")    # alpha_0 = 0.281,  width = 1.178
analyze(ramp, estimator="wtmm")               # alpha_0 = 0.106,  width = 0.849
leader_holder(ramp)                           # median h = 0.047
oscillation_holder(ramp)                      # median h = 1.040   <- correct
```

`db3` has three vanishing moments, so the details of a ramp are *algebraically*
zero. The measured interior coefficients run 4.3e-14 to 2.1e-12 against a signal
of size 4.1e3 — relative 1e-16, pure round-off. Round-off in a linear filter
grows with the analysing support, so `log|d|` against `j` is a clean straight
line and the estimator is confident: `leader_holder` reports a median `R² =
0.93`.

Three things make it worse than a merely wrong number:

- **It is not scale invariant.** `alpha_0` moves 0.233 → 0.281 → 0.300 as the
  signal is multiplied by 1e-6, 1, 1e6 — the library's own stated invariant is
  that every exponent is unchanged under `x → cx`.
- **Noise of *relative* size 1e-15 flips the sign** of `leader_holder`'s median,
  from +0.034 to −0.304.
- **`support_truncated` fires but points the wrong way**: it is documented to
  mean the width is a *lower* bound, and here the width should be zero.
  `alpha_0` carries no qualification at all.

A cubic — which `db3` does *not* annihilate — is answered correctly and
saturates at exactly +3.000, so the machinery is sound; only the round-off floor
is missing. `oscillation_holder`, which never divides by a vanishing quantity,
returns +1.040 uniformly.

**MF-DFA already refuses this exact input by name**, from an explicit
`(s·eps)²·⟨w²⟩` floor in `_window_variances`:

```
scale s=20 leaves only 0 windows with a detrended variance above the round-off floor
```

*Why not fixed:* transporting that floor to a multiresolution coefficient needs
a per-cell local-energy bound rather than a per-window one, and the file was
under concurrent edit when the finding landed. It is a real fix and it is worth
doing. **Mitigation today:** the xfail pins the behaviour, `consensus()` against
MF-DFA exposes it immediately (MF-DFA raises where the wavelets answer), and
`oscillation_holder` is trustworthy on smooth signals.

## 3. The chi-square monofractality test is not amplitude-invariant

**`tests/adversarial/test_mutation_coverage.py`**

`monofractal_test` rejects `H0` for an exactly monofractal series, and which
verdict you get depends on the amplitude of the input.

*Why not fixed:* the module already documents that the `ζ(q)` estimates across
`q` are strongly correlated — same data, same scales — so a naive chi-square is
anti-conservative, and directs the reader to the surrogate test as the primary.
The amplitude dependence sharpens that warning rather than changing it.
**Mitigation:** use `multifractality_report`, which is the documented primary.

## 4. A fit can span 0.30 decades over four increments

**`tests/adversarial/test_mutation_coverage.py`**

`structure_function`'s `min_length` of 64 admits a fit over as few as four
increments; the resulting `h(2)` is negative on 10 % of seeds, unqualified in
the summary.

*Why not fixed:* raising the floor would reject short series that other
estimators handle. **Mitigation:** `ScalingRange.decades` is on every fit —
below ~1 decade a power-law claim is weak, and the guide says so.

## 5. `Provenance` is immutable, its `params` mapping is a plain dict

**`tests/adversarial/test_provenance_attacks.py`**

`params` and `diagnostics` are typed `Mapping` but constructed as `dict`, and
`root` is a property recomputed on access rather than frozen at construction.
The copy taken is one level deep, so mutating a *nested* container inside a
parameter can still move the address of a record already written to a manifest.

*Why not fixed:* a deep freeze means recursively wrapping user objects, which
would reject legitimate parameter types. **Mitigation:** the store's
`verify()` re-derives the manifest's own address, so a record whose content
stopped matching its filename is detected on read.

## 6. `fbm_with_provenance(..., seed=None)` files two realisations under one address

**`tests/adversarial/test_provenance_attacks.py`**

With no seed there is nothing to record, so two different paths share a content
address while the docstring promises regeneration from the manifest alone.

*Why not fixed:* raising on `seed=None` would break the exploratory path.
**Mitigation:** `seed_info(None)` correctly returns `None` rather than
inventing entropy — an absent seed is recorded as absent — so the manifest does
not *claim* reproducibility. Pass a seed if you want the promise.

## 7. The stored `summary` is outside every digest

**`tests/adversarial/test_provenance_attacks.py`** (recorded as a strict xfail)

`root` is a pure function of the *inputs*, by design, so nothing binds the
*outputs* filed next to them: a rewritten `h2` in a stored manifest still
verifies. The session-level `RunManifest` **does** cover node summaries in its
Merkle root, so a forged number in a run manifest is caught; the per-artifact
store is the gap.

*Why not fixed:* the fix is a document digest over the whole manifest, which
changes the store format. It is the right change and it is deliberately not
being made in the same pass as everything else.

## 8. Minor

- **MF-DMA discards the tail.** It tiles the residual from one end only, always
  the same end. MF-DFA takes a second backward sweep precisely to avoid this.
- **GHE's reported exponent and its stored fit disagree.** With the default
  `average_tau_max=True`, `zeta[i]` is the mean of the nested-cutoff slopes
  while `fits[i]` is only the full-window regression — up to 0.209 apart on
  clean fBm.
- **Non-integer scales are truncated, not rounded**, while the library's own
  `log_spaced_scales` rounds. The fit stays self-consistent; the request is
  silently reinterpreted. `structure.scales` shows what was actually used.
- **Duplicate node names in a `Run`.** `by_name` resolves to the first match,
  so a second result recorded under the same name is unreachable while still
  contributing to the session root.
