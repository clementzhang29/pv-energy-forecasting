from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/Users/apple/Downloads/pv_data")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def run_fold(args: argparse.Namespace, fold: int, cache_dir: Path) -> dict[str, object]:
    fold_dir = cache_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(ROOT / "scripts" / "run_with_local_libomp.sh"),
        args.python,
        str(ROOT / "scripts" / "run_experiments.py"),
        "--data-dir",
        args.data_dir,
        "--output-dir",
        str(fold_dir),
        "--unseen-fold",
        str(fold),
        "--n-unseen-folds",
        str(args.folds),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    metrics = json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8"))
    final = metrics["final_by_split"]
    row = {
        "fold": fold,
        "best_prediction_column": metrics["best_prediction_column"],
        "best_blend_alpha": metrics.get("best_blend_alpha"),
        "combined_validation_daily_mape": metrics["candidate_scores"][
            metrics["best_prediction_column"]
        ],
        "valid_seen_daily_mape": final["valid_seen"]["daily_mape"],
        "valid_unseen_daily_mape": final["valid_unseen"]["daily_mape"],
        "test1_daily_mape": final["test1"]["daily_mape"],
        "test2_daily_mape": final["test2"]["daily_mape"],
    }
    exp_map = {
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
    }
    for name, score in metrics.get("candidate_scores", {}).items():
        row[f"candidate_{name}_combined"] = score
        exp_name = exp_map.get(name)
        split_metrics = metrics.get("experiments", {}).get(exp_name, {}) if exp_name else {}
        unseen = split_metrics.get("valid_unseen", {}).get("daily_mape")
        if unseen is not None:
            row[f"candidate_{name}_valid_unseen"] = unseen
    return row


def markdown_table(frame: pd.DataFrame) -> str:
    rows = [
        "| " + " | ".join(str(col) for col in frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for record in frame.to_dict(orient="records"):
        values = []
        for col in frame.columns:
            value = record[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    summary = pd.DataFrame(rows)
    csv_path = output_dir / "unseen_cv_summary.csv"
    summary.to_csv(csv_path, index=False)

    numeric_cols = [
        "combined_validation_daily_mape",
        "valid_seen_daily_mape",
        "valid_unseen_daily_mape",
        "test1_daily_mape",
        "test2_daily_mape",
    ]
    aggregate = summary[numeric_cols].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate_path = output_dir / "unseen_cv_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)

    candidate_cols = [
        c
        for c in summary.columns
        if c.startswith("candidate_") and (c.endswith("_combined") or c.endswith("_valid_unseen"))
    ]
    candidate_rows = []
    for col in candidate_cols:
        name = col.removeprefix("candidate_")
        if name.endswith("_combined"):
            candidate = name.removesuffix("_combined")
            metric = "combined_validation_daily_mape"
        else:
            candidate = name.removesuffix("_valid_unseen")
            metric = "valid_unseen_daily_mape"
        values = summary[col].dropna()
        candidate_rows.append(
            {
                "candidate": candidate,
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    candidate_summary = pd.DataFrame(candidate_rows).sort_values(["metric", "mean"])
    candidate_path = output_dir / "unseen_cv_candidate_summary.csv"
    candidate_summary.to_csv(candidate_path, index=False)

    lines = [
        "# 多折未见设备验证",
        "",
        "该实验按 `deviceSn` 对训练设备做多折留出，每一折都模拟新设备冷启动。折内仍保留 seen-device 时间留出，用于同时观察 test1 类问题和 test2 类问题。",
        "",
        "## 每折结果",
        "",
        markdown_table(summary),
        "",
        "## 汇总统计",
        "",
        markdown_table(aggregate),
        "",
        "## 候选模型多折稳定性",
        "",
        markdown_table(candidate_summary),
        "",
        "## 解释",
        "",
        "如果 `valid_unseen_daily_mape` 的标准差较大，说明单一 unseen holdout 不稳定，后续模型选择应优先依赖多折均值，而不是某一折的局部最优。",
    ]
    (output_dir / "unseen_cv_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache" / "unseen_cv"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_fold(args, fold, cache_dir) for fold in range(args.folds)]
    write_summary(rows, output_dir)
    print(f"Wrote {output_dir / 'unseen_cv_summary.csv'}")
    print(f"Wrote {output_dir / 'unseen_cv_summary.md'}")


if __name__ == "__main__":
    main()
