from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import describe_dataset, load_point_data
from src.features import build_features, make_daily_frame
from src.metrics import evaluate_predictions, summarize_daily_by_split
from src.modeling import (
    fit_daily_calibrator,
    fit_predict,
    irradiance_baseline,
    make_daily_model,
    rescale_point_predictions,
)
from src.reporting import save_json, write_final_report, write_model_comparison_csv
from src.split import assign_eval_split, train_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/Users/apple/Downloads/pv_data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--holdout-days", type=int, default=7)
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--unseen-fold", type=int, default=None)
    parser.add_argument("--n-unseen-folds", type=int, default=None)
    parser.add_argument("--unseen-strategy", default="default")
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "figures": output_dir / "figures",
        "predictions": output_dir / "predictions",
        "models": output_dir / "models",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def experiment_metrics(frame: pd.DataFrame, pred_col: str) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for split, part in frame.groupby("eval_split", observed=True):
        result[str(split)] = evaluate_predictions(part, pred_col=pred_col).as_dict()
    seen = result.get("valid_seen", {}).get("daily_mape")
    unseen = result.get("valid_unseen", {}).get("daily_mape")
    if seen is not None and unseen is not None:
        result["combined_validation"] = {
            "daily_mape": float(0.5 * seen + 0.5 * unseen),
            "mae": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "daily_mae": float("nan"),
            "daily_rmse": float("nan"),
        }
    return result


def sanitize_predictions(frame: pd.DataFrame, col: str) -> pd.Series:
    pred = frame[col].fillna(0).clip(lower=0)
    off_mask = (frame["is_solar_available"] <= 0) | (frame["solar_proxy"] <= 1e-8)
    pred.loc[off_mask] = 0.0
    return pred


def apply_low_output_guard(
    frame: pd.DataFrame,
    pred_col: str,
    output_col: str,
    threshold: float = 0.15,
    multiplier: float = 0.7,
) -> tuple[pd.DataFrame, dict[str, float]]:
    result = frame.copy()
    day_key = ["split", "eval_split", "deviceSn", "date"]
    daily_pred = (
        result.groupby(day_key, observed=True)[pred_col]
        .sum()
        .reset_index(name="_guard_daily_pred")
    )
    result = result.merge(daily_pred, on=day_key, how="left")
    cold_start_alpha = result.get("cold_start_alpha", pd.Series(1.0, index=result.index)).fillna(1)
    guard_mask = (cold_start_alpha <= 1e-6) & (result["_guard_daily_pred"] <= threshold)
    result[output_col] = result[pred_col]
    result.loc[guard_mask, output_col] = result.loc[guard_mask, output_col] * multiplier
    result[output_col] = sanitize_predictions(result, output_col)

    guarded_days = (
        result.loc[guard_mask, day_key]
        .drop_duplicates()
        .groupby("eval_split", observed=True)
        .size()
        .to_dict()
    )
    summary = {
        "daily_prediction_threshold": float(threshold),
        "multiplier": float(multiplier),
        "guarded_point_rows": int(guard_mask.sum()),
    }
    for split, count in guarded_days.items():
        summary[f"guarded_days_{split}"] = int(count)
    return result.drop(columns=["_guard_daily_pred"]), summary


def apply_piecewise_low_output_guard(
    frame: pd.DataFrame,
    pred_col: str,
    output_col: str,
    low_threshold: float = 0.04,
    high_threshold: float = 0.06,
    low_multiplier: float = 0.50,
    mid_multiplier: float = 0.55,
) -> tuple[pd.DataFrame, dict[str, float]]:
    result = frame.copy()
    day_key = ["split", "eval_split", "deviceSn", "date"]
    daily_pred = (
        result.groupby(day_key, observed=True)[pred_col]
        .sum()
        .reset_index(name="_piecewise_daily_pred")
    )
    result = result.merge(daily_pred, on=day_key, how="left")
    cold_start_alpha = result.get("cold_start_alpha", pd.Series(1.0, index=result.index)).fillna(1)
    cold_mask = cold_start_alpha <= 1e-6
    low_mask = cold_mask & (result["_piecewise_daily_pred"] <= low_threshold)
    mid_mask = (
        cold_mask
        & (result["_piecewise_daily_pred"] > low_threshold)
        & (result["_piecewise_daily_pred"] <= high_threshold)
    )
    result[output_col] = result[pred_col]
    result.loc[low_mask, output_col] = result.loc[low_mask, output_col] * low_multiplier
    result.loc[mid_mask, output_col] = result.loc[mid_mask, output_col] * mid_multiplier
    result[output_col] = sanitize_predictions(result, output_col)

    summary = {
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold),
        "low_multiplier": float(low_multiplier),
        "mid_multiplier": float(mid_multiplier),
        "guarded_point_rows": int((low_mask | mid_mask).sum()),
        "low_point_rows": int(low_mask.sum()),
        "mid_point_rows": int(mid_mask.sum()),
    }
    for label, mask in [("low", low_mask), ("mid", mid_mask), ("all", low_mask | mid_mask)]:
        guarded_days = (
            result.loc[mask, day_key]
            .drop_duplicates()
            .groupby("eval_split", observed=True)
            .size()
            .to_dict()
        )
        for split, count in guarded_days.items():
            summary[f"{label}_guarded_days_{split}"] = int(count)
    return result.drop(columns=["_piecewise_daily_pred"]), summary


def run() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    dirs = ensure_dirs(output_dir)

    point = load_point_data(args.data_dir)
    if args.sample_frac < 1.0:
        point = (
            point.groupby(["split", "deviceSn", "date"], observed=True)
            .filter(lambda _: np.random.rand() <= args.sample_frac)
            .reset_index(drop=True)
        )
    dataset_summary = describe_dataset(point)

    point = assign_eval_split(
        point,
        holdout_days=args.holdout_days,
        unseen_fold=args.unseen_fold,
        n_unseen_folds=args.n_unseen_folds,
        unseen_strategy=args.unseen_strategy,
    )
    point, feature_sets = build_features(point)
    daily = make_daily_frame(point)
    daily_features = [
        c
        for c in daily.columns
        if c
        not in {
            "deviceSn",
            "date",
            "split",
            "eval_split",
            "daily_target",
            "daily_target_log1p",
        }
        and pd.api.types.is_numeric_dtype(daily[c])
    ]
    train = point[train_mask(point)].copy()

    metrics: dict[str, object] = {"dataset": dataset_summary, "experiments": {}}
    models: dict[str, object] = {}

    point["pred_irradiance_baseline"] = irradiance_baseline(point)
    point["pred_irradiance_baseline"] = sanitize_predictions(point, "pred_irradiance_baseline")
    metrics["experiments"]["E00_irradiance_baseline"] = experiment_metrics(
        point, "pred_irradiance_baseline"
    )

    raw_model, raw_pred = fit_predict(
        train,
        point,
        feature_sets.point_features,
        target="pvGenTotal",
    )
    models["E01_lgbm_raw_point"] = raw_model
    point["pred_raw_point"] = raw_pred
    point["pred_raw_point"] = sanitize_predictions(point, "pred_raw_point")
    metrics["experiments"]["E01_lgbm_raw_point"] = experiment_metrics(point, "pred_raw_point")

    norm_train = train[train["solar_proxy"] > 1e-5].copy()
    norm_model, norm_pred = fit_predict(
        norm_train,
        point,
        feature_sets.point_features,
        target="y_norm_clipped",
    )
    models["E03_lgbm_norm_target"] = norm_model
    point["pred_norm_point"] = norm_pred * point["solar_proxy"]
    point["pred_norm_point"] = sanitize_predictions(point, "pred_norm_point")
    metrics["experiments"]["E03_lgbm_norm_target"] = experiment_metrics(point, "pred_norm_point")

    daily_model = make_daily_model()
    daily_train = daily[daily["eval_split"] == "train"].copy()
    daily_model.fit(daily_train[daily_features], daily_train["daily_target_log1p"])
    daily["pred_daily_model"] = np.expm1(daily_model.predict(daily[daily_features])).clip(min=0)
    models["E02_lgbm_daily"] = daily_model
    daily_pred = daily[["split", "deviceSn", "date", "pred_daily_model"]].rename(
        columns={"pred_daily_model": "pred_day_calibrated"}
    )
    daily_rescaled = rescale_point_predictions(
        point,
        raw_pred_col="pred_norm_point",
        daily_pred=daily_pred,
        output_col="pred_daily_model_rescaled",
    )
    point["pred_daily_model_rescaled"] = daily_rescaled["pred_daily_model_rescaled"]
    metrics["experiments"]["E02_lgbm_daily_rescaled"] = experiment_metrics(
        point, "pred_daily_model_rescaled"
    )

    calib_model, calibrated_daily = fit_daily_calibrator(
        point,
        raw_pred_col="pred_norm_point",
        daily_df=daily,
        daily_features=daily_features,
    )
    models["E05_daily_calibrator"] = calib_model
    calibrated = rescale_point_predictions(
        point,
        raw_pred_col="pred_norm_point",
        daily_pred=calibrated_daily,
        output_col="pred_norm_calibrated",
    )
    point["pred_norm_calibrated"] = calibrated["pred_norm_calibrated"]
    metrics["experiments"]["E05_norm_plus_daily_calibration"] = experiment_metrics(
        point, "pred_norm_calibrated"
    )

    blend_scores: dict[float, float] = {}
    for alpha in np.linspace(0, 1, 11):
        blend_col = f"_blend_{alpha:.1f}"
        point[blend_col] = (
            alpha * point["pred_norm_calibrated"]
            + (1 - alpha) * point["pred_daily_model_rescaled"]
        )
        point[blend_col] = sanitize_predictions(point, blend_col)
        blend_metric = experiment_metrics(point, blend_col)
        combined = blend_metric.get("combined_validation", {}).get("daily_mape")
        if combined is not None and np.isfinite(combined):
            blend_scores[float(alpha)] = float(combined)
    best_blend_alpha = min(blend_scores, key=blend_scores.get)
    point["pred_validated_blend"] = (
        best_blend_alpha * point["pred_norm_calibrated"]
        + (1 - best_blend_alpha) * point["pred_daily_model_rescaled"]
    )
    point["pred_validated_blend"] = sanitize_predictions(point, "pred_validated_blend")
    metrics["best_blend_alpha"] = float(best_blend_alpha)
    metrics["blend_validation_scores"] = {
        f"{alpha:.1f}": score for alpha, score in sorted(blend_scores.items())
    }
    metrics["experiments"]["E07_validated_blend"] = experiment_metrics(
        point, "pred_validated_blend"
    )
    point.drop(columns=[c for c in point.columns if c.startswith("_blend_")], inplace=True)

    history_days = point.get("device_history_days", pd.Series(0.0, index=point.index)).fillna(0)
    history_confidence = (history_days / 14.0).clip(lower=0, upper=1)
    point["cold_start_alpha"] = history_confidence
    point["pred_history_fallback"] = (
        point["cold_start_alpha"] * point["pred_validated_blend"]
        + (1 - point["cold_start_alpha"]) * point["pred_daily_model_rescaled"]
    )
    point["pred_history_fallback"] = sanitize_predictions(point, "pred_history_fallback")
    metrics["experiments"]["E08_history_dynamic_fallback"] = experiment_metrics(
        point, "pred_history_fallback"
    )

    point, guard_summary = apply_low_output_guard(
        point,
        pred_col="pred_history_fallback",
        output_col="pred_low_output_guard",
    )
    metrics["low_output_guard"] = guard_summary
    metrics["experiments"]["E09_low_output_guard"] = experiment_metrics(
        point, "pred_low_output_guard"
    )

    point, precision_guard_summary = apply_low_output_guard(
        point,
        pred_col="pred_history_fallback",
        output_col="pred_precision_low_output_guard",
        threshold=0.06,
        multiplier=0.4,
    )
    metrics["precision_low_output_guard"] = precision_guard_summary
    metrics["experiments"]["E10_precision_low_output_guard"] = experiment_metrics(
        point, "pred_precision_low_output_guard"
    )

    point, piecewise_guard_summary = apply_piecewise_low_output_guard(
        point,
        pred_col="pred_history_fallback",
        output_col="pred_piecewise_low_output_guard",
    )
    metrics["piecewise_low_output_guard"] = piecewise_guard_summary
    metrics["experiments"]["E11_piecewise_low_output_guard"] = experiment_metrics(
        point, "pred_piecewise_low_output_guard"
    )

    seen_devices = set(point.loc[train_mask(point), "deviceSn"].unique())
    is_seen_device = point["deviceSn"].isin(seen_devices)
    point["pred_seen_new_branch"] = np.where(
        is_seen_device,
        point["pred_norm_calibrated"],
        point["pred_daily_model_rescaled"],
    )
    point["pred_seen_new_branch"] = sanitize_predictions(point, "pred_seen_new_branch")
    metrics["experiments"]["E06_seen_new_branch"] = experiment_metrics(
        point, "pred_seen_new_branch"
    )

    candidate_cols = [
        "pred_irradiance_baseline",
        "pred_raw_point",
        "pred_norm_point",
        "pred_daily_model_rescaled",
        "pred_norm_calibrated",
        "pred_validated_blend",
        "pred_history_fallback",
        "pred_low_output_guard",
        "pred_precision_low_output_guard",
        "pred_piecewise_low_output_guard",
        "pred_seen_new_branch",
    ]
    candidate_scores: dict[str, float] = {}
    for col in candidate_cols:
        split_metrics = metrics["experiments"][
            {
                "pred_irradiance_baseline": "E00_irradiance_baseline",
                "pred_raw_point": "E01_lgbm_raw_point",
                "pred_norm_point": "E03_lgbm_norm_target",
                "pred_daily_model_rescaled": "E02_lgbm_daily_rescaled",
                "pred_norm_calibrated": "E05_norm_plus_daily_calibration",
                "pred_validated_blend": "E07_validated_blend",
                "pred_history_fallback": "E08_history_dynamic_fallback",
                "pred_low_output_guard": "E09_low_output_guard",
                "pred_precision_low_output_guard": "E10_precision_low_output_guard",
                "pred_piecewise_low_output_guard": "E11_piecewise_low_output_guard",
                "pred_seen_new_branch": "E06_seen_new_branch",
            }[col]
        ]
        combined = split_metrics.get("combined_validation", {}).get("daily_mape")
        if combined is not None and np.isfinite(combined):
            candidate_scores[col] = float(combined)
    best_pred_col = min(candidate_scores, key=candidate_scores.get)
    point["final_prediction"] = point[best_pred_col]
    metrics["best_prediction_column"] = best_pred_col
    metrics["candidate_scores"] = candidate_scores
    deployment_pred_col = "pred_precision_low_output_guard"
    point["deployment_prediction"] = point[deployment_pred_col]
    metrics["deployment_prediction_column"] = deployment_pred_col

    split_summary = summarize_daily_by_split(point, pred_col="final_prediction")
    split_summary.to_csv(output_dir / "final_metrics_by_split.csv", index=False)
    metrics["final_by_split"] = {
        row["eval_split"]: {
            k: float(row[k])
            for k in ["mae", "rmse", "mape", "daily_mae", "daily_rmse", "daily_mape"]
        }
        for row in split_summary.to_dict(orient="records")
    }
    deployment_summary = summarize_daily_by_split(point, pred_col="deployment_prediction")
    deployment_summary.to_csv(output_dir / "deployment_metrics_by_split.csv", index=False)
    metrics["deployment_by_split"] = {
        row["eval_split"]: {
            k: float(row[k])
            for k in ["mae", "rmse", "mape", "daily_mae", "daily_rmse", "daily_mape"]
        }
        for row in deployment_summary.to_dict(orient="records")
    }

    prediction_cols = [
        "split",
        "eval_split",
        "deviceSn",
        "date",
        "us_timestamp",
        "pvGenTotal",
        *candidate_cols,
        "final_prediction",
        "deployment_prediction",
    ]
    point[prediction_cols].to_csv(dirs["predictions"] / "final_predictions.csv", index=False)

    daily_preds = (
        point.groupby(["split", "eval_split", "deviceSn", "date"], observed=True)
        .agg(
            actual_daily=("pvGenTotal", "sum"),
            final_pred_daily=("final_prediction", "sum"),
            baseline_daily=("pred_irradiance_baseline", "sum"),
            raw_point_daily=("pred_raw_point", "sum"),
            norm_point_daily=("pred_norm_point", "sum"),
            calibrated_daily=("pred_norm_calibrated", "sum"),
            validated_blend_daily=("pred_validated_blend", "sum"),
            history_fallback_daily=("pred_history_fallback", "sum"),
            low_output_guard_daily=("pred_low_output_guard", "sum"),
            precision_low_output_guard_daily=("pred_precision_low_output_guard", "sum"),
            piecewise_low_output_guard_daily=("pred_piecewise_low_output_guard", "sum"),
            seen_new_branch_daily=("pred_seen_new_branch", "sum"),
            deployment_daily=("deployment_prediction", "sum"),
        )
        .reset_index()
    )
    daily_preds.to_csv(dirs["predictions"] / "daily_predictions.csv", index=False)

    from src.visualize import (
        plot_daily_distribution,
        plot_daily_scatter,
        plot_example_curve,
        plot_feature_importance,
        plot_geo_distribution,
        plot_split_mape,
    )

    figures: list[Path] = []
    figures.append(plot_daily_distribution(point, dirs["figures"]))
    figures.append(plot_geo_distribution(point, dirs["figures"]))
    figures.append(plot_daily_scatter(point, "final_prediction", dirs["figures"]))
    figures.append(plot_split_mape(split_summary, dirs["figures"]))
    figures.append(plot_example_curve(point, "final_prediction", dirs["figures"]))
    importance = plot_feature_importance(norm_model, feature_sets.point_features, dirs["figures"])
    if importance is not None:
        figures.append(importance)

    save_json(output_dir / "metrics.json", metrics)
    write_model_comparison_csv(metrics, output_dir)
    report_path = write_final_report(
        output_dir,
        dataset_summary=dataset_summary,
        metrics=metrics,
        figures=figures,
        best_model=best_pred_col,
    )
    print(f"Best prediction column: {best_pred_col}")
    print(f"Report written to: {report_path}")
    print(f"Metrics written to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    run()
