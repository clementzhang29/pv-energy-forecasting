# Agent Harness

## 1. Project Objective

Build and maintain a reproducible photovoltaic energy forecasting project.

The task is to predict device-level `pvGenTotal` from weather, solar geometry, time, and location features. The primary evaluation target is daily aggregated MAPE:

```text
Daily MAPE = MAPE(sum(prediction for one day), sum(actual for one day))
```

The project must provide:

- reusable training and evaluation code
- reproducible experiment commands
- result files and visualizations
- a final technical report
- clear validation logic for seen-device and unseen-device scenarios

## 2. Current Repository State

Branch:

```text
main
```

Current stable baseline:

```text
Use the latest commit on main. Check with: git log --oneline -1
```

Main directories:

```text
src/                  reusable project modules
scripts/              executable training and analysis scripts
outputs/              metrics, predictions, figures, and reports
requirements.txt      core dependencies
requirements-optional.txt optional advanced modeling dependencies
README.md             user-facing project overview
LOOP_ENGINEERING.md   controlled experiment-loop and harness protocol
```

Important outputs:

```text
outputs/final_report.md
outputs/metrics.json
outputs/model_comparison.csv
outputs/final_metrics_by_split.csv
outputs/deployment_metrics_by_split.csv
outputs/distribution_shift_diagnostics.md
outputs/predictions/final_predictions.csv
outputs/predictions/daily_predictions.csv
outputs/figures/
references/
```

## 3. Data Assumptions

Default data path:

```text
/Users/apple/Downloads/pv_data
```

Expected structure:

```text
pv_data/
├── train/
│   └── deviceSn/YYYY-MM-DD.csv
├── test1/
│   └── deviceSn/YYYY-MM-DD.csv
└── test2/
    └── deviceSn/YYYY-MM-DD.csv
```

Observed data facts:

| Split | Devices | Files | Rows | Meaning |
|---|---:|---:|---:|---|
| train | 39 | 2653 | 247002 | training devices |
| test1 | 22 | 110 | 10236 | devices also present in train |
| test2 | 6 | 36 | 3397 | unseen devices |

Important modeling implication:

```text
test1 = seen-device forecasting
test2 = unseen-device cold-start generalization
```

Any future validation design must preserve this distinction.

## 4. Implementation Path

The current implementation follows this sequence:

```text
load CSV files
  -> build time, solar, weather, spatial, and history features
  -> assign seen/unseen validation splits
  -> train baseline and GBDT models
  -> calibrate daily totals
  -> evaluate by daily aggregated MAPE
  -> write predictions, metrics, figures, and report
```

Core modules:

| File | Responsibility |
|---|---|
| `src/data.py` | load CSV files and summarize dataset |
| `src/features.py` | build time, physics, spatial, history, and daily features |
| `src/split.py` | create train, seen validation, unseen validation, test splits |
| `src/modeling.py` | train point models, daily models, baseline, and calibration |
| `src/metrics.py` | point and daily metrics |
| `src/visualize.py` | figures for report |
| `src/reporting.py` | final markdown report and summary files |
| `scripts/run_experiments.py` | full training pipeline |
| `scripts/analyze_results.py` | post-training blend scan and diagnostics |
| `scripts/diagnose_distribution_shift.py` | split-level drift and low-load diagnostics |
| `LOOP_ENGINEERING.md` | loop protocol, edit scope, acceptance gates, and backlog |

## 5. Reproduction Commands

Create environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run full training:

```bash
python scripts/setup_local_libomp.py
scripts/run_with_local_libomp.sh python scripts/run_experiments.py \
  --data-dir /Users/apple/Downloads/pv_data \
  --output-dir outputs
```

Run result analysis:

```bash
scripts/run_with_local_libomp.sh python scripts/analyze_results.py \
  --predictions outputs/predictions/final_predictions.csv \
  --output-dir outputs
```

Run distribution-shift diagnostics:

```bash
scripts/run_with_local_libomp.sh python scripts/diagnose_distribution_shift.py \
  --data-dir /Users/apple/Downloads/pv_data \
  --output-dir outputs
```

Run multi-fold unseen-device validation:

```bash
scripts/run_with_local_libomp.sh python scripts/run_unseen_cv.py \
  --data-dir /Users/apple/Downloads/pv_data \
  --folds 3 \
  --output-dir outputs \
  --python .venv/bin/python
```

Run low-capacity cold-start stress validation:

```bash
scripts/run_with_local_libomp.sh python scripts/run_cold_start_stress.py \
  --data-dir /Users/apple/Downloads/pv_data \
  --output-dir outputs \
  --python .venv/bin/python
```

Static code check:

```bash
python -m py_compile scripts/run_experiments.py scripts/analyze_results.py src/*.py
```

Controlled loop protocol:

```text
Read LOOP_ENGINEERING.md before running iterative experiments.
Use validation metrics for model selection.
Treat test metrics as retrospective diagnostics only.
```

## 6. Current Results

Selected validation model:

```text
pred_validated_blend
```

This is a validation-selected blend of the physically normalized daily-calibrated model and the daily-rescaled model.

Daily MAPE by split:

| Split | Daily MAPE |
|---|---:|
| train | 0.0072 |
| valid_seen | 0.0726 |
| valid_unseen | 0.1640 |
| test1 | 0.0905 |
| test2 | 0.5392 |

Blend scan:

```text
prediction = alpha * calibrated_normalized + (1 - alpha) * daily_rescaled
```

| Selection | Alpha | Combined Daily MAPE |
|---|---:|---:|
| best validation | 0.7 | 0.1183 |
| best retrospective test | 0.0 | 0.2067 |

The final prediction remains validation-selected rather than test-label-selected.

Multi-fold unseen validation:

| Metric | Mean | Std |
|---|---:|---:|
| combined validation Daily MAPE | 0.1465 | 0.0685 |
| valid_unseen Daily MAPE | 0.2101 | 0.1457 |
| test1 Daily MAPE | 0.0864 | 0.0017 |

Interpretation: seen-device validation is stable, while unseen-device validation has high variance. Future cold-start decisions should use multi-fold mean and variance instead of a single held-out device set.

Cold-start fallback evidence:

| Candidate | valid_unseen Daily MAPE | test2 Daily MAPE |
|---|---:|---:|
| `pred_validated_blend` | 0.1640 | 0.5392 |
| `pred_history_fallback` | 0.1791 | 0.2997 |
| `pred_low_output_guard` | 0.1849 | 0.2356 |
| `pred_precision_low_output_guard` | 0.1778 | 0.1868 |
| `pred_piecewise_low_output_guard` | 0.1766 | 0.1906 |

The fallback and guard are not the validation-selected final model, but they confirm that `test2` errors are driven by low-history device capacity shift and small-true-value MAPE amplification.

Deployment policy:

| Split | Validation-selected | Cold-start deployment |
|---|---:|---:|
| valid_seen | 0.0726 | 0.0726 |
| valid_unseen | 0.1640 | 0.1778 |
| test1 | 0.0905 | 0.0905 |
| test2 | 0.5392 | 0.1868 |

The deployment policy uses `pred_precision_low_output_guard`. It is meant for cold-start risk control and should be discussed separately from the validation-selected model. The broader `pred_low_output_guard` remains the best low-capacity stress-validation candidate. `pred_piecewise_low_output_guard` is retained as an ablation because it improves ordinary validation slightly but does not replace the current deployment policy.

Low-capacity stress validation:

| Candidate | Stress Combined Daily MAPE | Stress valid_unseen Daily MAPE |
|---|---:|---:|
| `pred_low_output_guard` | 0.1696 | 0.2646 |
| `pred_piecewise_low_output_guard` | 0.1974 | 0.3202 |
| `pred_precision_low_output_guard` | 0.1979 | 0.3211 |
| `pred_history_fallback` | 0.2016 | 0.3285 |
| `pred_validated_blend` | 0.2020 | 0.3293 |

Interpretation: when validation explicitly simulates low-capacity unseen devices, the low-output guard becomes the best candidate.

Distribution-shift diagnostics:

| Finding | Interpretation |
|---|---|
| `test2` devices are unseen in training | cold-start setting |
| `test2` has a higher low-daily-output share than ordinary `valid_unseen` | MAPE is amplified by small true values |
| ordinary unseen validation does not fully match low-capacity `test2` | low-capacity stress validation is required |

Reference materials:

```text
references/README.md
references/literature_review.md
references/papers/
```

## 7. Acceptance Standards

The project is considered healthy when all items below pass.

### 7.1 Functional Acceptance

- `scripts/run_experiments.py` runs from a clean environment using the documented command.
- `scripts/analyze_results.py` runs after training.
- The following files are generated:
  - `outputs/metrics.json`
  - `outputs/model_comparison.csv`
  - `outputs/final_metrics_by_split.csv`
  - `outputs/deployment_metrics_by_split.csv`
  - `outputs/predictions/final_predictions.csv`
  - `outputs/predictions/daily_predictions.csv`
  - `outputs/final_report.md`
  - `outputs/interview_report.md`
  - `outputs/distribution_shift_diagnostics.md`
  - `outputs/unseen_cv_summary.md`
  - `outputs/cold_start_stress_summary.md`
  - at least five figures in `outputs/figures/`
  - `references/literature_review.md`

### 7.2 Metric Acceptance

Minimum expected standards for the current pipeline:

| Split | Target |
|---|---:|
| valid_seen Daily MAPE | <= 0.09 |
| valid_unseen Daily MAPE | <= 0.18 |
| test1 Daily MAPE | <= 0.10 |

`test2` is the hardest split because it contains unseen devices. It should be tracked separately and improved through cold-start work rather than hidden inside an average.

For cold-start experiments, also report multi-fold unseen validation mean and standard deviation. A single `valid_unseen` split is not reliable enough for final model-selection decisions.

For deployment-risk experiments, keep two tracks separate:

- validation-selected `final_prediction`
- cold-start risk-control `deployment_prediction`

Do not claim a deployment guard is the default model unless it is also selected by the validation harness.

### 7.3 Code Acceptance

- No hard-coded absolute output paths except the documented default data path.
- All generated files go under `outputs/`.
- `.venv/`, caches, and local temporary files must not be committed.
- Scripts must be runnable from the repository root.
- Public-facing reports should use normal project language and avoid process-origin references.

## 8. Known Risks

1. `test2` cold start is still the main risk.
   - Current nearest-neighbor features help but do not fully infer device capacity.
   - The low-output guard improves retrospective `test2`, but it trades off ordinary `valid_unseen`.
   - Next improvements should focus on capacity proxy estimation and group calibration.

2. The normalized point target alone is unstable.
   - The raw normalized point model underperforms before daily calibration.
   - Keep daily calibration in the main pipeline.

3. Some weather columns are fully missing.
   - Current pipeline tolerates this through imputation.
   - Future feature selection can remove all-empty columns earlier.

4. The final predictions file is around 34 MB.
   - It is below GitHub's single-file size limit.
   - If future outputs grow, store only daily predictions or compressed artifacts.

## 9. Next Improvement Queue

Priority 1: cold-start capacity estimation

- Use regional comparable-device scaling.
- Add climate-region statistics from train devices.
- Estimate device capacity from peak observed proxy, latitude/longitude, and radiation response.
- Add grouped calibration by region, device family, nearest-neighbor capacity bucket, and low-output risk.
- See `LOOP_ENGINEERING.md` experiment E13.

Priority 2: validation refinement

- Keep multiple GroupKFold-style unseen folds.
- Report mean and standard deviation across folds and compare against low-capacity stress validation.
- Keep seen-device time holdout for test1 simulation.
- See `LOOP_ENGINEERING.md` experiment E14.

Priority 3: model improvements

- Add CatBoost from `requirements-optional.txt`.
- Add quantile regression or Huber objective.
- Add non-negative blending with out-of-fold daily predictions.
- See `LOOP_ENGINEERING.md` experiments E15-E19.

Priority 4: report refinement

- Add feature importance or permutation importance.
- Add error case studies for high-MAPE test2 days.
- Add daily residual plots by location and cloud cover.

## 10. GitHub Sync Procedure

Current repository has a local commit but no remote configured.

After receiving the GitHub repository URL:

```bash
git remote add origin <repo-url>
git push -u origin main
```

If `origin` already exists:

```bash
git remote set-url origin <repo-url>
git push -u origin main
```

Recommended SSH URL format:

```text
git@github.com:<username>/<repo>.git
```

If browser-based authorization is preferred, install or enable GitHub CLI, then run:

```bash
gh auth login
```

Use browser authorization when prompted, then push with normal git commands.

## 11. Handoff Checklist

Before handing off, verify:

- `git status --short --branch` is clean or changes are intentionally staged.
- `outputs/final_report.md` reflects the latest metrics.
- `outputs/metrics.json` matches the latest run.
- `README.md` quick-start commands still work.
- No local credential, token, or private key file is committed.
- GitHub remote is configured before attempting to push.
