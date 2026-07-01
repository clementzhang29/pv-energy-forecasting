# Loop Engineering and Harness Guide

## 1. Purpose

This document defines how to run this project as a controlled research loop rather than a one-off modeling script.

The project goal comes from the task document:

```text
Use device location and weather features to predict pvGenTotal.
The final score is daily aggregated MAPE over future one-day predictions.
```

The loop exists to make every modeling change measurable, reversible, and easy for another operator to continue.

## 2. Design References

This harness adapts several open-source engineering patterns:

| Reference | Pattern Adopted |
|---|---|
| `karpathy/autoresearch` | narrow editable surface, fixed experiment budget, one primary metric, keep-or-revert loop, Markdown program file |
| Optuna | explicit search space, trial records, objective-driven optimization |
| MLflow | run metadata, metrics, artifacts, and result comparison |
| Kedro | modular data science pipeline with clear data, feature, model, and reporting stages |
| RESCAST-100K | cross-domain residential forecasting benchmark with geography, climate, and equipment-domain splits |
| DropPatch | time-series representation learning for cross-domain, few-shot, and cold-start scenarios |
| Building thermal foundation model work | physics-informed temporal modeling and zero-shot transfer across buildings and climates |

Useful reference links:

- https://github.com/karpathy/autoresearch
- https://optuna.readthedocs.io/
- https://mlflow.org/docs/latest/ml/tracking/
- https://kedro.org/
- https://arxiv.org/html/2605.01364v1
- https://arxiv.org/html/2412.15315v1
- https://www.nature.com/articles/s41598-024-70336-3

Project reference notes:

```text
references/literature_review.md
references/papers/
```

## 3. Loop Engineering Principles

### 3.1 One Loop, One Score

Every experiment must optimize one primary selection score:

```text
combined_validation_daily_mape =
  0.5 * valid_seen_daily_mape
  + 0.5 * valid_unseen_daily_mape
```

Secondary metrics are allowed, but they cannot override the primary score unless the run violates a hard gate.

Secondary metrics:

- `test1_daily_mape`
- `test2_daily_mape`
- `deployment_test2_daily_mape`
- `low_load_share_lt_0_05`
- `daily_mae`
- point-level `mae`
- runtime
- artifact size

Hard gates:

- script must run without exception
- no data leakage from test labels into model selection
- no committed credentials or local environment files
- generated report must match the latest metrics

### 3.2 Narrow Editable Surface

Autonomous or semi-autonomous experiments should edit only the files needed for the hypothesis.

Safe-to-edit files:

```text
src/features.py
src/modeling.py
src/split.py
src/metrics.py
scripts/run_experiments.py
scripts/analyze_results.py
requirements.txt
requirements-optional.txt
README.md
AGENT_HARNESS.md
LOOP_ENGINEERING.md
```

Edit with caution:

```text
src/data.py
src/reporting.py
src/visualize.py
```

Do not edit during ordinary experiments:

```text
outputs/predictions/final_predictions.csv
outputs/predictions/daily_predictions.csv
outputs/metrics.json
outputs/final_report.md
outputs/figures/*
```

Generated outputs should be replaced only by running the documented scripts.

### 3.3 Fixed Evaluation Harness

Every experiment must run the same command unless the experiment explicitly changes the harness:

```bash
python scripts/setup_local_libomp.py
scripts/run_with_local_libomp.sh python scripts/run_experiments.py \
  --data-dir /Users/apple/Downloads/pv_data \
  --output-dir outputs

scripts/run_with_local_libomp.sh python scripts/analyze_results.py \
  --predictions outputs/predictions/final_predictions.csv \
  --output-dir outputs

scripts/run_with_local_libomp.sh python scripts/diagnose_distribution_shift.py \
  --data-dir /Users/apple/Downloads/pv_data \
  --output-dir outputs
```

Compile check:

```bash
python -m py_compile scripts/run_experiments.py scripts/analyze_results.py src/*.py
```

The loop must not compare a run evaluated with different validation logic against an older run without clearly marking the validation change.

## 4. Harness Contract

### 4.1 Inputs

Required local dataset:

```text
/Users/apple/Downloads/pv_data
```

Expected layout:

```text
pv_data/
├── train/<deviceSn>/<date>.csv
├── test1/<deviceSn>/<date>.csv
└── test2/<deviceSn>/<date>.csv
```

Required label:

```text
pvGenTotal
```

Primary covariates:

- `deviceSn`
- `latitude`
- `longitude`
- `us_timestamp`
- solar geometry columns
- 15-minute weather columns
- hourly `_H` weather columns

### 4.2 Outputs

Each valid run must produce:

```text
outputs/metrics.json
outputs/model_comparison.csv
outputs/final_metrics_by_split.csv
outputs/deployment_metrics_by_split.csv
outputs/blend_weight_summary.csv
outputs/final_report.md
outputs/predictions/final_predictions.csv
outputs/predictions/daily_predictions.csv
outputs/figures/*.png
```

Optional multi-fold unseen validation outputs:

```text
outputs/unseen_cv_summary.csv
outputs/unseen_cv_aggregate.csv
outputs/unseen_cv_summary.md
outputs/unseen_cv_candidate_summary.csv
outputs/cold_start_stress_summary.csv
outputs/cold_start_stress_summary.md
```

The validation-selected prediction and cold-start deployment prediction are intentionally tracked separately:

```text
final_prediction = default validation-selected model
deployment_prediction = stress-supported cold-start risk-control policy
```

### 4.3 Required Metric JSON Fields

`outputs/metrics.json` must contain:

```text
best_prediction_column
candidate_scores
final_by_split.train.daily_mape
final_by_split.valid_seen.daily_mape
final_by_split.valid_unseen.daily_mape
final_by_split.test1.daily_mape
final_by_split.test2.daily_mape
experiments
```

### 4.4 Baseline to Beat

Current validation-selected model:

```text
pred_validated_blend
```

Current score:

```text
combined_validation_daily_mape = 0.118314
valid_seen_daily_mape = 0.072630
valid_unseen_daily_mape = 0.163997
multi_fold_combined_validation_mean = 0.146453
multi_fold_valid_unseen_mean = 0.210086
multi_fold_valid_unseen_std = 0.145725
```

An experiment is an improvement only if:

```text
new_combined_validation_daily_mape < 0.118314
or new_multi_fold_combined_validation_mean < 0.146453
```

Preferred minimum improvement threshold:

```text
relative improvement >= 1%
```

Small improvements below 1% may be kept only if they reduce risk, simplify the pipeline, or materially improve `valid_unseen`.

## 5. The Research Loop

### 5.1 Loop Steps

Each cycle should follow:

```text
1. Observe
2. Hypothesize
3. Mutate
4. Run
5. Score
6. Decide
7. Log
8. Commit or revert
```

### 5.2 Observe

Read the latest:

```text
AGENT_HARNESS.md
LOOP_ENGINEERING.md
outputs/metrics.json
outputs/model_comparison.csv
outputs/blend_analysis.md
```

Identify the bottleneck:

- high `valid_seen` means known-device modeling is weak
- high `valid_unseen` means cold-start generalization is weak
- high `test2` in retrospective analysis means capacity proxy is weak
- poor point-level MAE but good daily MAPE means curve shape needs work

### 5.3 Hypothesize

Every experiment must start with a short hypothesis:

```text
Hypothesis:
Adding region-level capacity proxy from nearest comparable devices will reduce valid_unseen daily MAPE because unseen devices lack device-specific history.
```

Good hypothesis properties:

- targets one metric
- identifies expected mechanism
- names affected files
- predicts expected direction

### 5.4 Mutate

Change one logical unit at a time.

Examples:

- add one new feature family
- change one validation strategy
- add one model family
- add one calibration rule
- change one blending method

Avoid:

- changing features, split logic, objective, and reporting in the same trial
- editing generated outputs manually
- optimizing directly on `test1` or `test2`

### 5.5 Run

Use the standard commands:

```bash
source .venv/bin/activate
python -m py_compile scripts/run_experiments.py scripts/analyze_results.py src/*.py
python scripts/setup_local_libomp.py
scripts/run_with_local_libomp.sh python scripts/run_experiments.py --data-dir /Users/apple/Downloads/pv_data --output-dir outputs
scripts/run_with_local_libomp.sh python scripts/analyze_results.py --predictions outputs/predictions/final_predictions.csv --output-dir outputs
```

If runtime becomes too high, use a separate smoke run only for debugging:

```bash
python scripts/run_experiments.py \
  --data-dir /Users/apple/Downloads/pv_data \
  --output-dir outputs_smoke \
  --sample-frac 0.2
```

Smoke results must not be compared to full-run metrics.

### 5.6 Score

Extract:

```text
valid_seen_daily_mape
valid_unseen_daily_mape
combined_validation_daily_mape
test1_daily_mape
test2_daily_mape
```

Use validation for model selection. Test metrics may be reported as retrospective diagnostics only.

### 5.7 Decide

Keep if:

- primary validation score improves
- no hard gate fails
- output files are regenerated
- report is consistent with metrics

Revert if:

- primary validation score worsens materially
- training breaks
- outputs are missing
- validation logic changes without a clear reason
- code complexity increases without measurable gain

### 5.8 Log

Each accepted or rejected experiment should append one entry to:

```text
outputs/loop_log.md
```

Suggested entry format:

```markdown
## YYYY-MM-DD HH:MM - E13 Region Capacity Proxy

Hypothesis:
...

Files changed:
- src/features.py
- scripts/run_experiments.py

Command:
...

Result:
- valid_seen_daily_mape:
- valid_unseen_daily_mape:
- combined_validation_daily_mape:

Decision:
Keep / Revert

Notes:
...
```

## 6. Experiment Backlog

### 6.1 Highest Priority

#### E13 Region Capacity Proxy

Problem:

```text
test2 and valid_unseen are much harder than test1 and valid_seen.
```

Plan:

- build region-level daily capacity statistics
- use latitude/longitude plus weather regime
- add nearest-neighbor weighted average by distance
- add fallback when device history is missing

Expected improvement:

```text
valid_unseen daily MAPE decreases
```

Files:

```text
src/features.py
src/modeling.py
```

#### E14 Multi-Fold Unseen Validation

Problem:

```text
Current unseen validation uses one deterministic held-out device set.
```

Plan:

- add GroupKFold by `deviceSn`
- report mean/std across folds
- keep time-based seen validation unchanged

Expected improvement:

```text
more reliable validation selection
```

Files:

```text
src/split.py
scripts/run_experiments.py
src/reporting.py
```

Current status:

```text
Implemented as scripts/run_unseen_cv.py.
Latest 3-fold valid_unseen Daily MAPE mean = 0.2101, std = 0.1457.
Latest 3-fold combined validation Daily MAPE mean = 0.1465, std = 0.0685.
```

Interpretation: single-fold unseen validation is unstable. Future cold-start model selection should use multi-fold mean/std or out-of-fold predictions.

Low-capacity stress validation:

```text
Implemented as scripts/run_cold_start_stress.py.
Latest stress-selected candidate = pred_low_output_guard.
Stress combined validation Daily MAPE = 0.1696.
```

Interpretation: when validation intentionally simulates low-output unseen devices, the broader low-output guard becomes the best candidate. This is supporting evidence for dynamic cold-start policies.

Current deployment policy:

```text
deployment_prediction_column = pred_precision_low_output_guard
deployment_test2_daily_mape = 0.1868
```

#### E15 Non-Negative Daily Blend

Problem:

```text
daily_rescaled and norm_calibrated trade off test1 and test2 behavior.
```

Plan:

- fit non-negative blend weights on validation daily predictions
- use only validation or out-of-fold predictions
- keep test labels out of weight selection

Expected improvement:

```text
combined validation daily MAPE decreases
```

Files:

```text
src/modeling.py
scripts/analyze_results.py
```

### 6.2 Medium Priority

#### E16 All-Empty Feature Pruning

Remove columns with no observed train values before imputation.

#### E17 Robust Daily Objective

Try Huber/quantile-style daily regression or target clipping for high-error days.

#### E18 Error Case Report

Generate a table and plots for top 10 highest-MAPE days by split.

#### E19 CatBoost Optional Model

Install optional dependency and compare CatBoost against the current GBDT path.

## 7. Branching and Commit Policy

Use a branch per experiment:

```bash
git checkout -b exp/E13-region-capacity-proxy
```

After the run:

```bash
git status --short
git diff --stat
```

If accepted:

```bash
git add src scripts README.md AGENT_HARNESS.md LOOP_ENGINEERING.md outputs
git commit -m "Improve cold-start capacity features"
git checkout main
git merge --ff-only exp/E13-region-capacity-proxy
git push
```

If rejected:

```bash
git reset --hard HEAD
git checkout main
git branch -D exp/E13-region-capacity-proxy
```

Never use `git reset --hard` on a branch with unreviewed user edits.

## 8. Human Review Gates

Ask for human review before:

- changing the task metric
- changing the train/test interpretation
- deleting output artifacts
- adding large dependencies
- pushing files over 50 MB
- making the repository private/public setting changes
- rewriting git history

No review needed for:

- feature additions within existing files
- local validation experiments
- regenerating metrics and figures
- updating loop logs

## 9. Data Leakage Rules

Allowed:

- use `train` labels for training
- use held-out train dates/devices for validation
- report `test1` and `test2` metrics because labels are present locally

Not allowed:

- choose model weights by minimizing `test1` or `test2`
- use test-day `pvGenTotal` as a feature
- compute device history from validation or test dates
- manually edit predictions after seeing test errors

If retrospective test analysis suggests a better blend, record it as a finding and convert it into a validation-selected method in the next experiment.

## 10. Runbook for the Next Operator

Start here:

```bash
cd /path/to/pv-energy-forecasting
git pull
git status --short --branch
source .venv/bin/activate || true
```

If `.venv` is missing:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run baseline reproduction:

```bash
python -m py_compile scripts/run_experiments.py scripts/analyze_results.py src/*.py
python scripts/setup_local_libomp.py
scripts/run_with_local_libomp.sh python scripts/run_experiments.py --data-dir /Users/apple/Downloads/pv_data --output-dir outputs
scripts/run_with_local_libomp.sh python scripts/analyze_results.py --predictions outputs/predictions/final_predictions.csv --output-dir outputs
```

Then inspect:

```bash
cat outputs/blend_analysis.md
cat outputs/metrics_summary.md
```

Pick the next backlog item from Section 6.

## 11. Success Definition

Short-term success:

```text
valid_seen_daily_mape <= 0.07
valid_unseen_daily_mape <= 0.13
combined_validation_daily_mape < 0.10
```

Medium-term success:

```text
valid_unseen_daily_mape <= 0.12
test2 retrospective daily_mape materially improves without selecting on test labels
```

Long-term success:

```text
multiple-fold unseen validation is stable
model selection no longer depends on a single holdout split
final report explains both predictive performance and failure modes
```

## 12. Minimal Loop Prompt

Use this instruction for a new operator:

```text
Read AGENT_HARNESS.md and LOOP_ENGINEERING.md.
Reproduce the current baseline.
Choose one backlog experiment.
Make one logical change.
Run the full harness.
Compare combined validation Daily MAPE against 0.118314 and multi-fold mean against 0.146453.
Keep only if validation improves and hard gates pass.
Append outputs/loop_log.md.
Commit and push accepted changes.
```
