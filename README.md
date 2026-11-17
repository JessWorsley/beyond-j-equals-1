# Closure-model cosmology pipeline (Cobaya)

Constrains the cosmographic closure models j = j(q) (Chakraborty, Louw et al.) with DESI DR2 BAO, one SN sample at a time
(Pantheon+ or DES-SN5YR with the Dovekie recalibration) and compressed Planck distance priors (Chen, Huang & Wang 2018,
wCDM column).  A flat CPL (w0wa) model and the LCDM limit are run as benchmarks.

## Models and conventions

| name | definition | sampled parameters |
|---|---|---|
| `model_1` | j = 1 + 3 eps (q + 1) | H0, Om0, ombh2, q0, j0 |
| `model_2` | j = 1 + eps  (j = j0) | H0, Om0, ombh2, q0, j0 |
| `model_3` | j = 1 + 3 eps (q - 1/2) | H0, Om0, ombh2, q0, j0 |
| `cpl` | flat w0wa dark energy | H0, Om0, ombh2, w0, wa |
| `lcdm` | LCDM limit (j0 = 1, q0 = 1.5 Om0 - 1) | H0, Om0, ombh2 |

- `theta = [H0, p1, p2, Om0]` everywhere; (p1, p2) = (q0, j0) for closure models, (w0, wa) for CPL.  `r_d` is derived
  (calibrated to Planck); in `--no-planck` runs it is sampled instead (no `ombh2`).
- Om0 is an independent matter fraction, so the effective dark density rho_DE = h^2(z) - Om0 (1+z)^3 can be negative
  (Omega_m(z) > 1, w_DE undefined where it crosses zero).  Positivity is an optional assumption (see "Variants").
- Radiation: SN and BAO distances are radiation-free (`INCLUDE_RADIATION_LATE`); all CMB quantities include it
  (`INCLUDE_RADIATION_EARLY`, needed for r_s(z*) and l_A).
- The compressed Planck priors are valid only if f_X(z*) ~ 0 (dark fraction at recombination); `fX_star` is a derived
  parameter in every Planck run.
- SN offset M is marginalised analytically (counts as 1 extra parameter in AIC/BIC).

### Run names (`chain_utils.run_prefix`)
`desi_<sn>_planck_<model>[tag]` for closure models (`<sn>` = `pantheon` | `desy5`), `cpl_run` / `cpl_<sn>_run`,
`lcdm_limit_baseline` / `lcdm_limit_<sn>`.  `tag` is appended as given, so include the underscore (`_posdark`, `_zm3`).
Minimiser runs add `_min` (`chains/<prefix>_min.minimum.txt`).  `make_runs.py` adds `_noplanck` and `_rdfix` itself.
Analysis scripts infer the model and SN dataset from the prefix.

## Files

### Core (import these, do not run them)
| file | role |
|---|---|
| `constants.py` | `MODEL`, constants, radiation switches, fiducial point, `Z_MIN_PANTHEON`, optional `Z_MATCH`, `Z_MATCH_ALLOW_NEGATIVE`, plot style |
| `closure_cosmo.py` | the ONLY definition of H(z): closure models + CPL, `background_model`, distances, sound horizon, `cmb_quantities`, `calibrate`, `dark_fraction` |
| `calculate_distances.py`, `bao.py` | SN luminosity distance (1+z_hel) chi(z_hd) and BAO theory vector from `closure_cosmo` |
| `load_data.py` | DESI DR2, Pantheon+ (zHD > 0.01), DES-SN5YR/Dovekie, M marginalisation, Planck distance priors |
| `likelihoods.py` | Cobaya likelihoods: `DESILikelihood`, `DESIFreeRdLikelihood`, `PantheonLikelihood`, `DESY5Likelihood`, `PlanckLikelihood` (+ `...CPL` variants).  Options: `model`, `rd_fixed`, `use_omb`, `fx_max`, `fx_sigma`, `positive_dark` |
| `chain_utils.py` | read chains, IAT / R-hat, `run_prefix`, `infer_model`, `chi2_columns` |
| `inferred.py` | eps, c1, c2, w_DE(0) from (q0, j0, Om0) |

### Making and running chains
| file | usage |
|---|---|
| `make_runs.py` | writes the yaml files to `runs/`: `--sn`, `--models`, `--tag`, `--positive-dark`, `--fx-sigma`, `--fx-max`, `--minimize`, `--best-of N`, `--rd-fixed 147.2`, `--no-planck`, `--z-match <z\|none> [--z-match-negative]`, `--outdir` |
| `stamp_zmatch.py <prefix> <z_match\|none> [true\|false]` | writes `z_match` / `z_match_allow_negative` into an existing `runs/<prefix>.yaml` (and `_min`) written before 2026-11-16, so the analysis scripts use the setting the chain was run with (changes no chain) |
| `pipeline.yaml` | the manifest: datasets, models, benchmarks, and every variant (tag, make_runs arguments, `z_match`, label). Edit this to add or change runs |
| `pipeline.py <status\|make\|run\|analyse\|compare\|all\|adopt>` | runs the whole chain from the manifest (`--only`, `--jobs`, `--force`, `--dry-run`); skips finished runs; logs in `logs/`; status in `chains/pipeline_status.json` |
| `variants_config.py` | reads `pipeline.yaml` (used by `pipeline.py` and `compare_variants.py`) |
| `start_from_chain.py <prefix>` | writes `runs/<prefix>_min_start.yaml`: a minimiser started at the best chain sample (use when the minimiser stalls, e.g. hard cuts) |

Run with `cobaya-run -f runs/<name>.yaml` (`-f` deletes existing output with that prefix).

### Analysis
| file | usage |
|---|---|
| `analyse_chains.py <prefix>` | weighted results table (CSV), best-sample chi2 per likelihood, reduced chi2, corner plot |
| `plot_chains.py <prefix> [yaml] [parts]` | traces, ACF/IAT, marginals, degeneracies, chi2, w_DE/Omega_m functions, residuals, corner -> `chains/plots/`.  Parts: `traces acf iat marginals degeneracy chi2 functions residuals corner`.  Prior ranges are read from `runs/<prefix>.yaml` (or the given yaml) |
| `compare_w.py [sn]` | w_DE(z), rho_DE(z) bands of the closure models vs CPL (and DESI chains via `EXTERNAL`) |
| `model_comparison.py` | chi2_min, AIC, AICc, BIC, Akaike weights per SN dataset (uses `<prefix>_min` minimiser output if present) |
| `discontinuity_check.py <prefix>` | size of the kink at `Z_MATCH` on a z-match chain: Omega_DE(z_m), jump in rho'_DE and in q, w just below z_m |
| `high_z_check.py [prefix ...]` | Omega_m(z) at high z and asymptotic exponent of E^2 for models 1-3 (illustrative point or chain medians); writes `high_z_check.png` |
| `compare_variants.py [sn] [model]` | one model under different assumptions (base, no Planck, positive dark, z-match): chi2 per dataset, same-data Delta chi2/AIC/BIC, posterior medians, negative-dark-density probabilities, plot.  Takes the lower of the minimiser and chain-sample chi2 (`+` mark).  Each variant carries its own `Z_MATCH` in `VARIANTS` |
| `positivity_check.py <prefix>` | on an UNCONSTRAINED chain: posterior vs prior probability of rho_DE >= 0, Bayes factor of the positivity cut, where rho_DE < 0 occurs (inside / beyond the data range), chi2 per likelihood inside / outside the allowed region |
| `bound_check.py <prefix>` | model_3 only: asymptotic positivity bound j0 >= 3q0 - 1/2 + (1-2q0)^2 / (2(1-Om0)); margin and redshift where rho_DE first turns negative |

### Tests and diagnostics (run from the project folder)
| file | purpose |
|---|---|
| `test_closure_cosmo.py`, `test_likelihood_plumbing.py`, `test_load_desy5.py`, `test_make_runs.py` | run all four after any change (they assume default `Z_MATCH = None`) |
| `debug_planck.py` | evaluate the CMB quantities and Planck chi2 at test points |
| `check_point.py` | evaluate every likelihood at one point through Cobaya (reads the legacy `likelihood.yaml`) |
| `check_paper_equations.py` | appendix equations of the paper vs the code |
| `mcmc_compare_mock.py` | mock-data check that sampling q0 or dq gives the same posterior |

### Obsolete (safe to delete)
`cosmology.py`, `likelihoods_cmb.py`, `test_cosmology.py`, `data_planck_template.py`, and the old hand-written yaml files
(`likelihood.yaml`, `likelihood_desy5.yaml`, `cpl.yaml`, `cpl_desy5.yaml`, `lcdm_limit.yaml`, `minimize.yaml`) once
`check_point.py` is pointed at a file in `runs/`.  (`cosmology.py` shadows nothing now, but an old copy once shadowed `closure_cosmo`.)

## Recipes

### Automated (recommended)
Describe the runs once in `pipeline.yaml`, then:
```
python pipeline.py status
python pipeline.py all --jobs 2        # make -> run -> analyse -> compare;  or step by step: make, run, analyse, compare
python pipeline.py run --only zm3pos   # just one variant
python pipeline.py adopt               # chains you ran by hand are finished: mark them so `run` skips them
```
`Z_MATCH` is set per variant in the manifest (`z_match`, `z_match_negative`) and written into the yaml: `constants.py` is not edited.
The manual steps below do the same one command at a time.

Baseline chains and minimisers:
```
python make_runs.py --minimize --best-of 8
cobaya-run -f runs/desi_pantheon_planck_model_3.yaml
cobaya-run -f runs/desi_pantheon_planck_model_3_min.yaml
python analyse_chains.py desi_pantheon_planck_model_3
python plot_chains.py   desi_pantheon_planck_model_3
```

### Variants (give each its own tag)
| variant | make_runs | note |
|---|---|---|
| positive dark density | `--models model_1 model_2 model_3 --tag _posdark --positive-dark` | hard cut on rho_DE >= 0 for z <= z*; minimiser can stall on the wall -> `start_from_chain.py` |
| no Planck | `--no-planck` | DESI + SN, r_d sampled; H0 is unconstrained (BAO fixes only H0 r_d); quote Om0, q0, j0 |
| DESI-style fixed r_d | `--rd-fixed 147.2` | r_d fixed, 2D (R, lA) prior, omega_b fixed; tag `_rdfix` |
| late-time closure | set `Z_MATCH = 3.0` (and `Z_MATCH_ALLOW_NEGATIVE`) in `constants.py`, then `--tag _zm3` | closure only for z <= Z_MATCH, then Om0(1+z)^3 plus a frozen dark density |
| soft f_X(z*) control | `--fx-sigma 0.02` or `--fx-max 0.05` with a tag | alternative to the positivity cut |

`make_runs.py` writes the `Z_MATCH` / `Z_MATCH_ALLOW_NEGATIVE` values of `constants.py` (as of the moment it is called, also
`None`) into every likelihood block of the yaml, and the likelihoods apply them when Cobaya starts, so the yaml alone defines the
run: editing `constants.py` afterwards does not change an existing yaml.  Analysis scripts (`plot_chains.py`, `compare_w.py`,
`positivity_check.py`, `bound_check.py`) read the setting back from `runs/<prefix>.yaml`; `compare_variants.py` uses the value in
`VARIANTS`.  Yaml files written BEFORE this change (2026-11-16) have no such options and follow `constants.py`.  To make z-match
yaml files: set the two constants, run `make_runs.py --tag _zm3` (or `_zm3pos` with `Z_MATCH_ALLOW_NEGATIVE = False`), then comment
the constants out again before generating normal runs.

### Comparing variants
```
python compare_variants.py pantheon model_3
python positivity_check.py desi_pantheon_planck_model_3      # unconstrained chain
python bound_check.py desi_pantheon_planck_model_3
```
Total chi2, AIC and BIC are only comparable between runs with the SAME data (the no-Planck run has no Planck data).
Run the minimiser for every variant; the lowest chain sample is only an upper bound on chi2_min.

## Known limitations / open items
- Planck prior numbers in `load_data.planck_distance_priors` were typed by hand: verify against Chen et al. (2018).
- Union3 is not in the pipeline.  The fixed-r_d cross-check, `--no-planck` and `Z_MATCH` runs are untested on real data.
- A hard cut (`positive_dark`) makes the likelihood a wall; Cobaya's minimiser may stall there.
- `analyse_chains.py` and `plot_chains.py` handle `--no-planck` chains (no omega_b, no Planck panel); `_zm` chains need
  `Z_MATCH` set when analysing.
- Cobaya's `best_of` and `method` minimiser options are used as written in `make_runs.py`; check the names against your
  Cobaya version if a minimiser yaml is rejected.

## Change log
- 2026-11-13: README created.  Added `compare_variants.py`, `positivity_check.py`, `bound_check.py`, `start_from_chain.py`;
  `make_runs.py` options `--rd-fixed`, `--no-planck`, `--best-of`; `DESIFreeRdLikelihood`; Planck options `use_omb`;
  `plot_chains.py` reads the yaml from `runs/<prefix>.yaml`; `chi2_columns` maps `DESIFreeRd` to `DESI`.
- 2026-11-16: `plot_chains.py` z_t histogram now interpolates the q = 0 crossing (was snapped to the z grid).
- 2026-11-16: `Z_MATCH` is now stored in the yaml (`z_match`, `z_match_allow_negative` likelihood options) and read back by the analysis scripts (`chain_utils.apply_run_settings`); `make_runs.py` warns when the tag and `Z_MATCH` disagree.  Reason: a base minimiser run started while `constants.py` still had `Z_MATCH = 3.0` silently became a z-match fit.
- 2026-11-16: added `stamp_zmatch.py` for yaml files written before `z_match` was stored in them (z-match chains were plotted with the full closure).
- 2026-11-16: added `pipeline.yaml` / `pipeline.py` / `variants_config.py` (manifest-driven make -> run -> analyse -> compare, with status tracking); `make_runs.py --z-match`; `compare_variants.py` takes its variants from the manifest when present.
- 2026-11-16: manifest variants `zm25pos`, `zm4pos`, `zm5pos` (Z_MATCH robustness, positive frozen density); `discontinuity_check.py` added.
- 2026-11-17: `high_z_check.py` added (models 1 and 2 only approach matter domination at j0 = 1).  Z_MATCH profile points `zm35pos`, `zm7pos` added to the manifest.
