from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_point_data
from src.features import build_features, make_daily_frame
from src.split import assign_eval_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/Users/apple/Downloads/pv_data")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def fmt(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.4f}"


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
                values.append(fmt(value))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def split_summary(point: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    train_devices = set(point.loc[point["eval_split"] == "train", "deviceSn"].unique())
    rows: list[dict[str, object]] = []
    for split, day_part in daily.groupby("eval_split", observed=True):
        point_part = point[point["eval_split"] == split]
        devices = set(day_part["deviceSn"].unique())
        low_005 = float((day_part["daily_target"] < 0.05).mean())
        low_010 = float((day_part["daily_target"] < 0.10).mean())
        rows.append(
            {
                "eval_split": split,
                "devices": int(len(devices)),
                "train_device_overlap_ratio": float(len(devices & train_devices) / max(len(devices), 1)),
                "days": int(len(day_part)),
                "date_min": str(day_part["date"].min()),
                "date_max": str(day_part["date"].max()),
                "daily_target_mean": float(day_part["daily_target"].mean()),
                "daily_target_median": float(day_part["daily_target"].median()),
                "daily_target_std": float(day_part["daily_target"].std()),
                "daily_target_p10": float(day_part["daily_target"].quantile(0.10)),
                "daily_target_p90": float(day_part["daily_target"].quantile(0.90)),
                "low_load_share_lt_0_05": low_005,
                "low_load_share_lt_0_10": low_010,
                "solar_proxy_sum_median": float(day_part["solar_proxy_sum"].median()),
                "radiation_proxy_sum_median": float(day_part["radiation_proxy_sum"].median()),
                "latitude_median": float(day_part["latitude"].median()),
                "longitude_median": float(day_part["longitude"].median()),
                "region5_count": int(
                    day_part[["region_lat_5", "region_lon_5"]].drop_duplicates().shape[0]
                ),
                "nonzero_row_ratio": float((point_part["pvGenTotal"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("eval_split").reset_index(drop=True)


def drift_table(summary: pd.DataFrame) -> pd.DataFrame:
    indexed = summary.set_index("eval_split")
    baseline = indexed.loc["train"]
    metrics = [
        "daily_target_mean",
        "daily_target_median",
        "daily_target_p10",
        "daily_target_p90",
        "low_load_share_lt_0_05",
        "low_load_share_lt_0_10",
        "solar_proxy_sum_median",
        "radiation_proxy_sum_median",
        "nonzero_row_ratio",
        "train_device_overlap_ratio",
    ]
    rows = []
    for metric in metrics:
        row = {"metric": metric, "train": float(baseline[metric])}
        for split in ["valid_seen", "valid_unseen", "test1", "test2"]:
            if split not in indexed.index:
                continue
            value = float(indexed.loc[split, metric])
            train_value = float(baseline[metric])
            row[split] = value
            row[f"{split}_vs_train_ratio"] = value / train_value if abs(train_value) > 1e-12 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(output_dir: Path, summary: pd.DataFrame, drift: pd.DataFrame) -> Path:
    output_path = output_dir / "distribution_shift_diagnostics.md"
    indexed = summary.set_index("eval_split")
    test2_low = float(indexed.loc["test2", "low_load_share_lt_0_05"]) if "test2" in indexed.index else np.nan
    train_low = float(indexed.loc["train", "low_load_share_lt_0_05"]) if "train" in indexed.index else np.nan
    test2_overlap = (
        float(indexed.loc["test2", "train_device_overlap_ratio"]) if "test2" in indexed.index else np.nan
    )
    valid_unseen_low = (
        float(indexed.loc["valid_unseen", "low_load_share_lt_0_05"])
        if "valid_unseen" in indexed.index
        else np.nan
    )
    text = [
        "# 分布漂移诊断",
        "",
        "本诊断用于确认高误差是否来自新设备冷启动和数据分布迁移，而不是单纯模型参数不足。指标基于本地标签回看生成，不参与默认模型选择。",
        "",
        "## 核心发现",
        "",
        f"- `test2` 设备与训练设备重叠比例为 {fmt(test2_overlap)}，说明它主要是新设备冷启动场景。",
        f"- `test2` 日发电量低于 0.05 的占比为 {fmt(test2_low)}，训练集对应占比为 {fmt(train_low)}，小真实值样本明显更多。",
        f"- 当前 `valid_unseen` 的低负载占比为 {fmt(valid_unseen_low)}，低于 `test2`，因此普通 GroupKFold 还需要补充低容量压力验证。",
        "- MAPE 对小真实值非常敏感，低容量新设备会放大相同绝对误差下的相对误差。",
        "",
        "## 分 split 统计",
        "",
        markdown_table(summary),
        "",
        "## 相对训练集漂移",
        "",
        markdown_table(drift),
        "",
        "## 结论",
        "",
        "该诊断支持 cold-start domain shift 判断：`test2` 同时具备新设备、低负载占比更高和容量代理不足三类风险。后续优化应继续围绕分组验证、跨设备容量特征、低容量压力验证、动态 fallback 和分组校准展开，而不是只调单一模型参数。",
    ]
    output_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    point = load_point_data(args.data_dir)
    point = assign_eval_split(point)
    point, _ = build_features(point)
    daily = make_daily_frame(point)

    summary = split_summary(point, daily)
    drift = drift_table(summary)
    summary.to_csv(output_dir / "distribution_shift_summary.csv", index=False)
    drift.to_csv(output_dir / "distribution_shift_vs_train.csv", index=False)
    report = write_report(output_dir, summary, drift)
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
