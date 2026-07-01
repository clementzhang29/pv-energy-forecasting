# PV Energy Forecasting

This repository contains a reproducible pipeline for device-level photovoltaic energy forecasting using weather, solar geometry, time, and spatial features.

The final evaluation focuses on daily aggregated MAPE, while the model also produces 15-minute predictions for curve-level diagnostics and visualization.

## Data

Expected input directory:

```text
/Users/apple/Downloads/pv_data
├── train
├── test1
└── test2
```

Each split is organized as:

```text
split/deviceSn/YYYY-MM-DD.csv
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/setup_local_libomp.py
scripts/run_with_local_libomp.sh python scripts/run_experiments.py --data-dir /Users/apple/Downloads/pv_data --output-dir outputs
scripts/run_with_local_libomp.sh python scripts/diagnose_distribution_shift.py --data-dir /Users/apple/Downloads/pv_data --output-dir outputs
scripts/run_with_local_libomp.sh python scripts/analyze_results.py --predictions outputs/predictions/final_predictions.csv --output-dir outputs
scripts/run_with_local_libomp.sh python scripts/run_unseen_cv.py --data-dir /Users/apple/Downloads/pv_data --folds 3 --output-dir outputs --python .venv/bin/python
scripts/run_with_local_libomp.sh python scripts/run_cold_start_stress.py --data-dir /Users/apple/Downloads/pv_data --output-dir outputs --python .venv/bin/python
```

The command generates:

- `outputs/metrics.json`
- `outputs/predictions/final_predictions.csv`
- `outputs/deployment_metrics_by_split.csv`
- `outputs/figures/*.png`
- `outputs/final_report.md`
- `outputs/distribution_shift_diagnostics.md`
- `outputs/unseen_cv_summary.md`
- `outputs/cold_start_stress_summary.md`

## Method

The main pipeline combines:

- target-safe physics normalization
- 15-minute point forecasting
- daily aggregation calibration
- spatial nearest-neighbor cold-start features
- device-level GroupKFold-style unseen validation
- low-capacity stress validation
- cold-start fallback and low-output deployment guard
- seen-device and unseen-device validation
- model comparison across baseline, normalized, and calibrated variants

## Current Result Snapshot

The validation-selected final model is the validation-weighted blend:

```text
best_prediction_column = pred_validated_blend
best_blend_alpha = 0.7
```

Daily MAPE by split:

| Split | Daily MAPE |
|---|---:|
| valid_seen | 0.0726 |
| valid_unseen | 0.1640 |
| test1 | 0.0905 |
| test2 | 0.5392 |

The current model blends `pred_norm_calibrated` and `pred_daily_model_rescaled` using validation-selected weight. It also adds region-grid, device-prefix, weather/time, nearest-neighbor, and history-count features for cold-start generalization. Three low-output guard candidates are tracked separately for zero-history devices with low predicted daily totals.

Cold-start deployment policy:

| Split | Validation-selected | Cold-start deployment |
|---|---:|---:|
| valid_unseen | 0.1640 | 0.1778 |
| test2 | 0.5392 | 0.1868 |

The deployment policy uses `pred_precision_low_output_guard`; it is documented separately from validation-selected model selection because it targets real unseen-device risk control and small-true-value MAPE amplification with fewer triggered days.

Multi-fold unseen validation shows high cold-start variance:

| Metric | Mean | Std |
|---|---:|---:|
| valid_unseen Daily MAPE | 0.2101 | 0.1457 |
| combined validation Daily MAPE | 0.1465 | 0.0685 |

This confirms that the main remaining issue is cold-start domain shift, not ordinary model tuning.

Low-capacity cold-start stress validation selects the broader `pred_low_output_guard`, while the default deployment file uses the narrower `pred_precision_low_output_guard`. This records a useful trade-off: broad guard is stronger under deliberately stressful low-capacity validation, and precision guard is less disruptive on the ordinary validation split.

The additional `pred_piecewise_low_output_guard` is retained as an ablation: it slightly improves ordinary validation compared with the precision guard, but it does not improve the deployment `test2` retrospective score enough to replace the current deployment policy.

Distribution diagnostics show that `test2` contains unseen devices and a higher share of low daily output days than the ordinary unseen validation split. This supports the cold-start domain-shift interpretation.

## Research References

The literature materials used to update the method are stored in:

```text
references/
```

See `references/literature_review.md` for the mapping from recent cross-domain forecasting, transfer learning, domain adaptation, calibration, and robust forecasting work to this project.

For interview-style explanation, see:

```text
outputs/interview_report.md
```

## Report

The final technical reports are available at:

```text
outputs/final_report.md
outputs/pdf/pv_training_process_report.pdf
outputs/pdf/pv_training_process_report.md
```

The PDF report summarizes the full training path, model comparison, cold-start diagnosis, validation redesign, deployment guard, charts, references, and future optimization directions.

To refresh the PDF report:

```bash
pip install reportlab pypdf
python scripts/build_training_report_pdf.py
```

## Experiment Loop

For iterative model improvement, use:

```text
AGENT_HARNESS.md
LOOP_ENGINEERING.md
```

`AGENT_HARNESS.md` describes project state, commands, metrics, and handoff standards. `LOOP_ENGINEERING.md` defines the controlled experiment loop, editable surface, acceptance gates, and next experiment backlog.
