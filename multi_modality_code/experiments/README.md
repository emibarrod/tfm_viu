# Sepsis-3 Experiment Layer

This folder contains the baseline/ablation experiment stage for the frozen
Sepsis-3 multimodal task.

It consumes Stage 3 exports and writes Stage 5 results.

## Inputs

- `data/04_multimodal/` (primary, antibiotics excluded)
- `data/04_multimodal_abx/` (sensitivity export with antibiotics)

Required files in each export:

- `cohort.csv`
- `static.csv`
- `vitals_timeseries.csv`
- `labs_timeseries.csv`
- `treatments_timeseries.csv`
- `clinical_scores.csv` — SOFA/qSOFA at t0, written by
  `multi_modality_code/6_build_clinical_scores.py`. Only the
  `clinical_scores_only` ablation reads it, and that ablation fails loudly if it
  is missing rather than scoring zeros.

## Main Entry Point

Run from repo root:

```bash
uv run python -m multi_modality_code.experiments.run_experiments \
  --multimodal_dir data/04_multimodal \
  --abx_dir data/04_multimodal_abx \
  --output_dir data/05_results \
  --seeds 42,43,44 \
  --threshold_method youden \
  --mimic_version "MIMIC-IV v3.1"
```

## What It Does

1. Loads train/val/test split from `cohort.csv` (`split` column).
2. Verifies no `subject_id` overlap across splits.
3. Aggregates long modalities (vitals/labs/treatments) to one row per `stay_id`.
4. Builds baselines:
   - Logistic Regression
   - HistGradientBoosting
   - XGBoost (when available in environment)
5. Runs modality ablations.
6. Tunes the decision threshold on validation with Youden's J (`--threshold_method`,
   default `youden`; F1 is the secondary analysis — see `threshold_report.py`), evaluates
   on test, saves calibration plots, and persists val/test probabilities under
   `<output_dir>/predictions/` so any threshold criterion, calibration metric or
   prevalence rescaling can be recomputed without retraining.

## Ablations

An ablation is an experiment where one subset of modalities is used in isolation
or in combination, so that each modality's contribution to predictive performance
can be measured independently. The progression from single-modality runs to the
full model shows how much each data source adds.

The four available modalities are:

- **Static** — patient context known at or before ICU admission: age, gender,
  ICU unit, `diagnosis_count` (the admission's ICD row count, a crude coding-burden
  proxy), the **Charlson comorbidity index and its 17 category flags**, and hours
  from hospital/ICU admission to the prediction point. `gender` and `unit` are
  numeric codes for nominal categories, so they enter only as one-hot dummies; a
  train-fitted zero-variance filter then drops `adm_order` and `re_admission`,
  which are structurally constant under the first-ICU-stay filter. Both
  comorbidity representations come from codes assigned for billing at discharge,
  so neither is strictly available at t0 — see the `primary_no_billing_features`
  ablation below and `../difference_charlson_vs_count.md`.
- **Vitals** — ICU monitoring events: heart rate, blood pressure, MAP,
  respiratory rate, SpO2, temperature, GCS, FiO2, PEEP, and other chart events.
  Aggregated over the 24-hour lookback window.
- **Labs** — laboratory and blood-gas results: creatinine, lactate, bilirubin,
  platelets, WBC, haemoglobin, PaO2, pH, and others from `labu.csv`. Aggregated
  over the 24-hour lookback window.
- **Treatments** — intervention events: IV fluid volumes, vasopressor rates
  (norepinephrine-equivalent), and urine output. Aggregated over the 24-hour
  lookback window. Antibiotics are excluded by default because they participate
  in the Sepsis-3 label definition.

There is also a fifth, non-architectural block used by exactly one ablation:

- **Clinical scores** — `sofa_total_at_t0` and `qsofa_at_t0`, the two bedside
  severity scores evaluated at the prediction time (Stage 6). Not a modality of
  the multimodal model; the clinical comparator.

The eleven ablation configurations are:

| Configuration | Modalities used | Purpose |
|---|---|---|
| `static_only` | static | Lower bound: how much can demographics and comorbidities predict on their own? |
| `vitals_only` | vitals | How much do physiological monitoring signals contribute? |
| `labs_only` | labs | How much do lab and blood-gas results contribute? |
| `treatments_no_antibiotics` | treatments | How much do fluid/vasopressor/urine patterns contribute? |
| `vitals_plus_labs` | vitals + labs | Do the two main clinical signal types together outperform each alone? |
| `static_plus_vitals_plus_labs` | static + vitals + labs | Does adding patient context improve on top of clinical signals? |
| `static_plus_vitals_plus_labs_plus_treatments` | static + vitals + labs + treatments | Primary full model: best non-leaking configuration. |
| `sensitivity_with_antibiotics` | static + vitals + labs + treatments + antibiotics | **Sensitivity/leakage analysis only.** Antibiotics partially define the label, so any performance gain here reflects label leakage. Report separately with an explicit caveat. |
| `clinical_scores_only` | clinical scores | **Clinical comparator, classical arm only.** Logistic regression on SOFA and qSOFA at t0 — the bedside baseline the ML models have to beat. Logistic regression only: a boosted forest over two integers would add noise, not information. |
| `primary_no_temporal_position` | static + vitals + labs + treatments, minus `hours_from_icu_intime_to_prediction` and `hours_from_admission_to_prediction` | **Classical arm only.** How much of the primary result rests on where the window sits inside the stay rather than on physiology. Same split, same features, same library versions, two columns fewer — the clean version of the question the control-sampling bug raised. |
| `primary_no_billing_features` | static + vitals + labs + treatments, minus `diagnosis_count`, the Charlson index and its 17 flags | **Classical arm only.** How much rests on ICD codes assigned at discharge, which are not strictly knowable at t0. |

The last three are flagged `classical_only`: `dl/run_dl_experiments.py` resolves
branch widths through `GridBundle.branch_input_dim()`, which has no clinical-score
branch and does not know about `drop_features`, so it skips them (and says so in
its log) rather than emitting a row that does not mean what it says.

## Outputs

Written under `data/05_results/`:

- `metrics.csv` (one row per ablation x model)
- `manifest.json` (run metadata and paths)
- `calibration/*.png` (calibration curves, from the reported seed)
- `features/*.csv` (cached aggregated feature matrices)
- `predictions/<ablation>__<model>_seed<seed>.npz` (val/test probabilities plus
  stay_ids, one file per seed)
- `paired_tests.csv`, `clinical_rules.csv` (written by `posthoc_analysis.py`)
- `threshold_comparison.csv` (written by `threshold_report.py`)
- `antibiotic_exposure.csv`, `antibiotic_exposure_summary.csv` (written by
  `antibiotic_exposure.py`): per-arm share of stays with an antibiotic interval
  overlapping the 24 h window, and where the positives' suspicion antibiotic falls
  relative to t0 and onset — the mechanism behind `sensitivity_with_antibiotics`

## Notes

- XGBoost on macOS requires OpenMP (`libomp`). If unavailable, XGBoost rows are
  marked `skipped` in `metrics.csv` while sklearn baselines still run.
- The pipeline keeps `event_time <= prediction_time` guarantees from Stage 3.

## Deep learning baselines

`dl/` implements a GRU/LSTM per-modality intermediate-fusion model family
that runs the identical 8-ablation matrix, splits, and evaluation protocol as
the classical baselines above, so results are directly comparable. See
[`dl/README.md`](dl/README.md) for the full input-representation spec,
architecture, hyperparameters, and exact commands.

**Status:** executed. `data/05_results/deep_learning/dl_metrics.csv` holds the
8 shared ablations x 3 seeds (42/43/44). Headline: the GRU trails the best tree
model on seven of eight ablations (primary ablation 0.697 [0.672, 0.724] vs
HistGBT 0.721 [0.695, 0.745]); the paired test makes that gap significant only on
the antibiotics ablation (p < 0.001), not on the primary one (p = 0.077), while the
GRU *does* beat logistic regression significantly (+0.041, p = 0.001). It
reproduces the same modality ordering and shows small seed sensitivity (AUROC std
<= 0.007). The three `classical_only` ablations are not mirrored there. See
[`dl/README.md`](dl/README.md#results) for the full table. Both arms place their
threshold with Youden's J.

## Missing values, feature filtering and caching

- **Missing values reach the models as NaN.** The design matrix is no longer
  `nan_to_num`-ed to zero, so the logistic-regression pipeline's median imputer does real
  work and HistGradientBoosting/XGBoost use their native missing-value handling. Only
  infinities are neutralised.
- **Counts, presence flags and event sums are filled with 0, not NaN**
  (`aggregate.fill_structural_zeros`). Absence of observations is a defined zero for these,
  and leaving them NaN would let median imputation collapse every `__present` indicator to a
  constant 1.0 — silently destroying the missingness signal for the linear model while the
  tree models carried on unaffected. Value statistics (`__min/max/mean/std/last/slope`) stay
  NaN, since they are genuinely undefined without observations.
- **A zero-variance filter, fitted on the training split**, removes columns with no
  variation (`aggregate.drop_zero_variance`). It is applied identically on the classical and
  DL sides so both model families see the same static feature set.
- **Cached feature matrices carry a source fingerprint** (`<name>.csv.meta.json`) and rebuild
  automatically when the Stage-4 exports change. Pass `--force_rebuild` to rebuild
  unconditionally. The DL grid cache additionally stores its concept lists, so changing
  `VITALS_CONCEPTS`/`LABS_CONCEPTS` invalidates it instead of silently mislabelling channels.

## Statistical rigor and testing

- **Three seeds per (ablation, model)** in the classical arm, matching the DL
  arm's protocol: the reported row is the median-AUPRC seed (a real run, not an
  average of metrics that are not linear in the predictions), and the spread is
  reported as `auroc_mean`/`auroc_std`/`auprc_mean`/`auprc_std` with the seed used
  in `median_seed`. Logistic regression (liblinear) and HistGradientBoosting are
  deterministic given the data, so their std is 0 by construction; XGBoost varies
  because it subsamples rows and columns. That asymmetry is the finding, not a
  defect — it is why the DL arm's seed spread cannot be compared naively against
  "the classical arm".
- `metrics.csv` includes `auroc_ci_low/high` and `auprc_ci_low/high`:
  1000-resample percentile bootstrap 95% CIs (`evaluate.bootstrap_ci()`),
  computed post-hoc on the already-predicted test-set probabilities. The bootstrap
  RNG is seeded from the *first* seed regardless of which model seed produced the
  row, so the interval describes test-set sampling variability only.
- **Paired tests instead of overlapping CIs** (`posthoc_analysis.py` →
  `paired_tests.csv`). Two models scored on the same patients have correlated
  errors, so "the marginal CIs overlap" is a conservative heuristic, not a test.
  The paired bootstrap resamples the patients once and recomputes both metrics on
  the same resample, giving a CI and a two-sided p-value on the *difference*. Two
  families: every model pair within an ablation, and the primary feature set
  against every other one holding the model fixed. Nominal p-values, no
  multiplicity correction — quote only pre-specified comparisons.
- **Calibration is quantified, not just plotted**: `evaluate.expected_calibration_error()`
  (10 equal-width bins) adds an `ece` column. Like every calibration statistic
  here it describes calibration *to this cohort's 50 % prevalence*.
- **Prevalence-corrected operating point.** The cohort is 1:1 case-control, so
  PPV, F1 and the AUPRC baseline are artefacts of the design. `metrics.csv`
  therefore reports `ppv` alongside `ppv_at_5pct_prevalence` and
  `ppv_at_10pct_prevalence` (`evaluate.ppv_at_prevalence()`, Bayes' rule on
  sensitivity/specificity), which bracket reported ICU sepsis incidence. The gap
  between them is what an operational reading has to use.
- **Clinical cut-offs** (`posthoc_analysis.py` → `clinical_rules.csv`): SOFA >= 2
  and qSOFA >= 2 as published, plus neighbouring cut-offs, evaluated on the test
  split with the same prevalence rescaling.
- `feature_importance.csv` holds the top-15 permutation-importance features
  (`sklearn.inspection.permutation_importance`, `scoring="average_precision"`,
  `n_repeats=20`) for each of the 3 classical models, scoped to the primary
  ablation (`static_plus_vitals_plus_labs_plus_treatments`) only. The DL arm has
  its own counterpart, `deep_learning/dl_feature_importance.csv` — same score,
  repeats and ablation, but one row per *concept* rather than per column (see
  `dl/README.md`), so only the static rows are directly comparable between the
  two files.
- `tests/` (repo root) is a pytest suite covering subject-grouped split
  leakage, `event_time <= prediction_time` guarantees, antibiotics isolation
  between the primary/sensitivity exports, aggregation-schema regressions
  against `metrics.csv`'s recorded `n_features`, the SOFA/qSOFA tier boundaries
  and the Stage 6 export's contract, and the control-matching
  contract (no prediction time clipped to discharge, 6h tail margin in both
  arms, balanced time-since-ICU-admission, controls without `onset_time`).
  Run with `uv run pytest tests/ -v` from the repo root — note the schema test
  compares against `metrics.csv`, so run the experiment stage before the tests
  after any change that shifts feature widths.

## Deferred in this cycle

Explicitly out of scope for now, tracked here so they aren't lost:

- **Text / ClinicalBERT** — the text modality (MIMIC-IV-Note) is not used
  anywhere in this pipeline.
- **GRU-D / TFT / cross-attention (late) fusion** — `dl/models.py`'s
  `SequenceEncoder` is designed so these can be dropped in later without
  redesigning the pipeline, but none are implemented yet.
- **Per-timestep attribution for the DL models** — `dl_feature_importance.csv`
  answers *which concept* the recurrent model leans on, but not *which hour of
  the window*; permuting a concept's whole trajectory is deliberately blind to
  that.
