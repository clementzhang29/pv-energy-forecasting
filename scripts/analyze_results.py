from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.metrics import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/predictions/final_predictions.csv")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    prediction_path = Path(args.predictions)
    data = pd.read_csv(prediction_path, parse_dates=["us_timestamp"])

    rows: list[dict[str, float | str]] = []
    for alpha in np.linspace(0, 1, 11):
        pred_col = f"blend_{alpha:.1f}"
        data[pred_col] = (
            alpha * data["pred_norm_calibrated"]
            + (1 - alpha) * data["pred_daily_model_rescaled"]
        ).clip(lower=0)
        for split in ["valid_seen", "valid_unseen", "test1", "test2"]:
            part = data[data["eval_split"] == split]
            metrics = evaluate_predictions(part, pred_col=pred_col).as_dict()
            rows.append(
                {
                    "alpha_norm_calibrated": float(alpha),
                    "eval_split": split,
                    **metrics,
                }
            )

    blend = pd.DataFrame(rows)
    blend.to_csv(output_dir / "blend_weight_scan.csv", index=False)

    summary = (
        blend.pivot_table(
            index="alpha_norm_calibrated",
            columns="eval_split",
            values="daily_mape",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    summary["combined_valid"] = 0.5 * (
        summary["valid_seen"] + summary["valid_unseen"]
    )
    summary["combined_test"] = 0.5 * (summary["test1"] + summary["test2"])
    summary.to_csv(output_dir / "blend_weight_summary.csv", index=False)

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_data = summary.melt(
            id_vars=["alpha_norm_calibrated"],
            value_vars=["combined_valid", "combined_test", "test1", "test2"],
            var_name="metric_group",
            value_name="daily_mape",
        )
        sns.lineplot(
            data=plot_data,
            x="alpha_norm_calibrated",
            y="daily_mape",
            hue="metric_group",
            marker="o",
            ax=ax,
        )
        ax.set_title("Blend Weight Scan")
        ax.set_xlabel("Weight on calibrated normalized model")
        ax.set_ylabel("Daily MAPE")
        fig.tight_layout()
        fig.savefig(output_dir / "figures" / "blend_weight_scan.png", dpi=160)
        plt.close(fig)
    except Exception:
        pass

    best_valid = summary.loc[summary["combined_valid"].idxmin()].to_dict()
    best_test = summary.loc[summary["combined_test"].idxmin()].to_dict()
    lines = [
        "# 融合权重扫描",
        "",
        "本分析扫描日级模型与归一化校准模型的线性融合权重：",
        "",
        "```text",
        "prediction = alpha * calibrated_normalized + (1 - alpha) * daily_rescaled",
        "```",
        "",
        f"验证集最优 alpha: {best_valid['alpha_norm_calibrated']:.1f}，综合验证 Daily MAPE: {best_valid['combined_valid']:.4f}。",
        f"测试集回看最优 alpha: {best_test['alpha_norm_calibrated']:.1f}，测试集综合 Daily MAPE: {best_test['combined_test']:.4f}。",
        "",
        "最终模型选择仍以验证集为准；测试集回看只用于诊断模型在真实已见/未见设备上的差异，不用于调参选型。",
        "",
        "当前现象说明：新设备的主要误差来自容量或规模估计。日级模型在测试回看中更稳，但验证集尚未可靠支持直接切换，因此下一步应优先做多折未见设备验证和冷启动容量代理。",
    ]
    (output_dir / "blend_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
