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
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stress_dir = output_dir / "cache" / "cold_start_stress" / "low_capacity"
    stress_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(ROOT / "scripts" / "run_with_local_libomp.sh"),
        args.python,
        str(ROOT / "scripts" / "run_experiments.py"),
        "--data-dir",
        args.data_dir,
        "--output-dir",
        str(stress_dir),
        "--unseen-strategy",
        "low_capacity",
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    metrics = json.loads((stress_dir / "metrics.json").read_text(encoding="utf-8"))
    exp_map = {
        "pred_norm_calibrated": "E05_norm_plus_daily_calibration",
        "pred_validated_blend": "E07_validated_blend",
        "pred_history_fallback": "E08_history_dynamic_fallback",
        "pred_low_output_guard": "E09_low_output_guard",
        "pred_precision_low_output_guard": "E10_precision_low_output_guard",
        "pred_piecewise_low_output_guard": "E11_piecewise_low_output_guard",
        "pred_daily_model_rescaled": "E02_lgbm_daily_rescaled",
        "pred_seen_new_branch": "E06_seen_new_branch",
    }
    rows = []
    for candidate, exp in exp_map.items():
        values = metrics["experiments"][exp]
        rows.append(
            {
                "candidate": candidate,
                "combined_validation_daily_mape": values["combined_validation"]["daily_mape"],
                "valid_seen_daily_mape": values["valid_seen"]["daily_mape"],
                "valid_unseen_daily_mape": values["valid_unseen"]["daily_mape"],
                "test1_daily_mape": values["test1"]["daily_mape"],
                "test2_daily_mape": values["test2"]["daily_mape"],
            }
        )
    summary = pd.DataFrame(rows).sort_values("combined_validation_daily_mape")
    summary_path = output_dir / "cold_start_stress_summary.csv"
    summary.to_csv(summary_path, index=False)

    lines = [
        "# 冷启动压力验证",
        "",
        "该实验从训练设备中选择低日发电量设备作为 `valid_unseen`，模拟真实冷启动中小容量设备导致 MAPE 放大的场景。",
        "",
        f"最佳验证列：`{metrics['best_prediction_column']}`",
        "",
        markdown_table(summary),
        "",
        "解释：如果低输出 guard 或 fallback 在该压力验证下优于普通校准模型，说明低容量新设备需要更保守的容量先验；如果主融合模型仍最好，则说明 fallback 应作为风险诊断或分段策略，而不是全局替代。",
    ]
    report_path = output_dir / "cold_start_stress_summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
