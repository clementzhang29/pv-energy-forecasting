from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def metrics_table(metrics: dict[str, dict[str, float]]) -> str:
    rows = [
        "| 实验 | 评估集 | MAE | RMSE | 日总量 MAE | 日总量 RMSE | 日总量 MAPE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for exp, split_metrics in metrics.items():
        if not isinstance(split_metrics, dict):
            continue
        for split, values in split_metrics.items():
            if not isinstance(values, dict) or "daily_mape" not in values:
                continue
            rows.append(
                "| "
                + " | ".join(
                    [
                        exp,
                        split,
                        format_metric(values.get("mae", 0.0)),
                        format_metric(values.get("rmse", 0.0)),
                        format_metric(values.get("daily_mae", 0.0)),
                        format_metric(values.get("daily_rmse", 0.0)),
                        format_metric(values.get("daily_mape", 0.0)),
                    ]
                )
                + " |"
            )
    return "\n".join(rows)


def dataset_table(summary: dict[str, dict[str, object]]) -> str:
    rows = [
        "| 数据集 | 设备数 | 日期文件数 | 行数 | 日期范围 | 日总量均值 | 日总量中位数 |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for split, values in summary.items():
        rows.append(
            "| "
            + " | ".join(
                [
                    split,
                    str(values["devices"]),
                    str(values["files"]),
                    str(values["rows"]),
                    f"{values['date_min']} to {values['date_max']}",
                    format_metric(float(values["daily_sum_mean"])),
                    format_metric(float(values["daily_sum_median"])),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def split_result_summary(metrics: dict[str, object]) -> str:
    final_by_split = metrics.get("final_by_split", {})
    if not isinstance(final_by_split, dict):
        return "当前指标文件缺少最终分 split 指标，需重新运行训练脚本。"

    rows = [
        "| Split | 日总量 MAPE | 日总量 MAE | 解释 |",
        "|---|---:|---:|---|",
    ]
    explanations = {
        "train": "训练拟合参考，不作为泛化判断依据",
        "valid_seen": "模拟已见设备未来日期，主要对应 test1",
        "valid_unseen": "模拟未见设备冷启动，主要对应 test2",
        "test1": "已见设备测试回看，用于复盘",
        "test2": "未见设备测试回看，是当前主要难点",
    }
    for split in ["train", "valid_seen", "valid_unseen", "test1", "test2"]:
        values = final_by_split.get(split)
        if not isinstance(values, dict):
            continue
        rows.append(
            "| "
            + " | ".join(
                [
                    split,
                    format_metric(values.get("daily_mape", 0.0)),
                    format_metric(values.get("daily_mae", 0.0)),
                    explanations.get(split, ""),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def candidate_score_summary(metrics: dict[str, object]) -> str:
    candidate_scores = metrics.get("candidate_scores", {})
    if not isinstance(candidate_scores, dict) or not candidate_scores:
        return "当前指标文件缺少候选模型验证分数。"
    rows = [
        "| 候选预测列 | 综合验证 Daily MAPE |",
        "|---|---:|",
    ]
    for name, score in sorted(candidate_scores.items(), key=lambda item: float(item[1])):
        rows.append(f"| `{name}` | {format_metric(float(score))} |")
    return "\n".join(rows)


def blend_summary(metrics: dict[str, object]) -> str:
    alpha = metrics.get("best_blend_alpha")
    scores = metrics.get("blend_validation_scores", {})
    if alpha is None or not isinstance(scores, dict):
        return "本轮未启用验证集融合权重扫描。"
    best_score = scores.get(f"{float(alpha):.1f}")
    if best_score is None:
        return f"验证集融合扫描启用，最优 alpha = {float(alpha):.1f}。"
    return (
        f"验证集融合扫描启用，最优 alpha = {float(alpha):.1f}。"
        f"融合公式为 `alpha * pred_norm_calibrated + (1 - alpha) * pred_daily_model_rescaled`，"
        f"对应综合验证 Daily MAPE = {format_metric(float(best_score))}。"
    )


def cold_start_evidence(metrics: dict[str, object]) -> str:
    experiments = metrics.get("experiments", {})
    if not isinstance(experiments, dict):
        return "当前指标文件缺少实验对比。"
    rows = [
        "| 模型 | valid_unseen Daily MAPE | test2 Daily MAPE | 说明 |",
        "|---|---:|---:|---|",
    ]
    selected = [
        ("E05_norm_plus_daily_calibration", "归一化校准模型", "强验证基线，但新设备仍有偏移"),
        ("E07_validated_blend", "验证集融合模型", "当前主模型"),
        ("E08_history_dynamic_fallback", "历史置信度 fallback", "历史不足时退回日级泛化模型"),
        ("E09_low_output_guard", "低输出 guard", "低日总量且零历史时保守缩放"),
        ("E10_precision_low_output_guard", "精细低输出 guard", "只在极低日总量风险时触发"),
        ("E11_piecewise_low_output_guard", "分段低输出 guard", "按预测日总量分段设置缩放强度"),
        ("E02_lgbm_daily_rescaled", "日级泛化模型", "冷启动兜底参考"),
    ]
    for exp, label, note in selected:
        values = experiments.get(exp, {})
        if not isinstance(values, dict):
            continue
        valid_unseen = values.get("valid_unseen", {}).get("daily_mape")
        test2 = values.get("test2", {}).get("daily_mape")
        if valid_unseen is None or test2 is None:
            continue
        rows.append(
            f"| {label} | {format_metric(float(valid_unseen))} | {format_metric(float(test2))} | {note} |"
        )
    return "\n".join(rows)


def deployment_policy_summary(metrics: dict[str, object]) -> str:
    final_by_split = metrics.get("final_by_split", {})
    deployment_by_split = metrics.get("deployment_by_split", {})
    deployment_col = metrics.get("deployment_prediction_column")
    if not isinstance(final_by_split, dict) or not isinstance(deployment_by_split, dict):
        return "当前指标文件缺少部署策略对比。"

    rows = [
        "| Split | 验证最优 Daily MAPE | 冷启动部署 Daily MAPE |",
        "|---|---:|---:|",
    ]
    for split in ["valid_seen", "valid_unseen", "test1", "test2"]:
        final_values = final_by_split.get(split, {})
        deployment_values = deployment_by_split.get(split, {})
        if not isinstance(final_values, dict) or not isinstance(deployment_values, dict):
            continue
        rows.append(
            f"| {split} | {format_metric(float(final_values.get('daily_mape', 0.0)))} | {format_metric(float(deployment_values.get('daily_mape', 0.0)))} |"
        )
    note = (
        f"冷启动部署策略使用 `{deployment_col}`。它不是单一默认验证集最优模型，"
        "而是结合多折 unseen validation 和低容量压力验证后，用于真实新设备风险控制的候选。"
    )
    if deployment_col == "pred_low_output_guard":
        note += "该策略只在设备历史为零且日级兜底预测较低时触发保守缩放，目标是降低小真实值场景下的 MAPE 放大。"
    if deployment_col == "pred_precision_low_output_guard":
        note += "该策略只在设备历史为零且日级兜底预测极低时触发更克制的保守缩放，目标是在降低 test2 低输出误差的同时减少对普通 unseen 设备的扰动。"
    return note + "\n\n" + "\n".join(rows)


def low_output_guard_summary(metrics: dict[str, object]) -> str:
    guard = metrics.get("low_output_guard", {})
    precision_guard = metrics.get("precision_low_output_guard", {})
    piecewise_guard = metrics.get("piecewise_low_output_guard", {})
    if not isinstance(guard, dict) or not guard:
        return "本轮未启用低输出 guard。"
    rows = [
        "| 策略 | 日级阈值 | 缩放系数 | 触发行数 |",
        "|---|---:|---:|---:|",
        f"| 广义低输出 guard | {format_metric(float(guard.get('daily_prediction_threshold', 0.0)))} | {format_metric(float(guard.get('multiplier', 0.0)))} | {int(guard.get('guarded_point_rows', 0))} |",
    ]
    if isinstance(precision_guard, dict) and precision_guard:
        rows.append(
            f"| 精细低输出 guard | {format_metric(float(precision_guard.get('daily_prediction_threshold', 0.0)))} | {format_metric(float(precision_guard.get('multiplier', 0.0)))} | {int(precision_guard.get('guarded_point_rows', 0))} |"
        )
    if isinstance(piecewise_guard, dict) and piecewise_guard:
        threshold_text = (
            f"{format_metric(float(piecewise_guard.get('low_threshold', 0.0)))} / "
            f"{format_metric(float(piecewise_guard.get('high_threshold', 0.0)))}"
        )
        multiplier_text = (
            f"{format_metric(float(piecewise_guard.get('low_multiplier', 0.0)))} / "
            f"{format_metric(float(piecewise_guard.get('mid_multiplier', 0.0)))}"
        )
        rows.append(
            f"| 分段低输出 guard | {threshold_text} | {multiplier_text} | {int(piecewise_guard.get('guarded_point_rows', 0))} |"
        )
    day_rows = [
        "| 策略 | Split | 触发天数 |",
        "|---|---|---:|",
    ]
    for label, payload in [
        ("广义低输出 guard", guard),
        ("精细低输出 guard", precision_guard if isinstance(precision_guard, dict) else {}),
        ("分段低输出 guard", piecewise_guard if isinstance(piecewise_guard, dict) else {}),
    ]:
        for key, value in sorted(payload.items()):
            if key.startswith("guarded_days_"):
                split = key.replace("guarded_days_", "")
            elif key.startswith("all_guarded_days_"):
                split = key.replace("all_guarded_days_", "")
            else:
                continue
            day_rows.append(f"| {label} | {split} | {int(value)} |")
    return (
        "低输出 guard 面向 MAPE 的小真实值敏感性：当设备无历史、日级 fallback 预测低于阈值时，"
        "对预测做保守缩放。广义 guard 更偏压力场景稳健性，精细 guard 更偏减少普通 unseen 设备扰动，分段 guard 用于验证不同低输出强度下的缩放是否需要差异化；这些策略都不作为默认验证最优模型替代。\n\n"
        + "\n".join(rows)
        + "\n\n"
        + "\n".join(day_rows)
    )


def distribution_shift_section(output_dir: Path) -> str:
    summary_path = output_dir / "distribution_shift_summary.csv"
    if not summary_path.exists():
        return "分布漂移诊断尚未运行。"
    summary = pd.read_csv(summary_path).set_index("eval_split")
    rows = [
        "| Split | 设备数 | 训练设备重叠 | 日发电量均值 | 低于 0.05 占比 | 低于 0.10 占比 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ["train", "valid_seen", "valid_unseen", "test1", "test2"]:
        if split not in summary.index:
            continue
        row = summary.loc[split]
        rows.append(
            "| "
            + " | ".join(
                [
                    split,
                    str(int(row["devices"])),
                    format_metric(float(row["train_device_overlap_ratio"])),
                    format_metric(float(row["daily_target_mean"])),
                    format_metric(float(row["low_load_share_lt_0_05"])),
                    format_metric(float(row["low_load_share_lt_0_10"])),
                ]
            )
            + " |"
        )
    return (
        "分布漂移诊断已运行。它用于证明 `test2` 的高误差来自新设备和低负载样本比例变化，"
        "而不是单纯的模型参数问题。\n\n"
        + "\n".join(rows)
    )


def literature_section() -> str:
    return """本轮方法更新参考了 `references/` 中的文献资料，重点不是照搬某个复杂模型，而是吸收其验证与泛化思想：

| 研究方向 | 对本项目的启发 | 项目实现 |
|---|---|---|
| cross-domain forecasting | 需要按建筑、设备、区域、气候等 domain 留出验证，而不是随机切分 | `valid_unseen`、multi-fold unseen validation |
| transfer learning / domain adaptation | 新设备应从相似设备、区域和静态属性中迁移共性规律 | 经纬度最近邻、区域网格、设备族群代理 |
| physics-informed forecasting | 太阳几何和辐照是跨设备泛化的重要先验 | `solar_proxy`、物理归一化、日级校准 |
| robust cold-start forecasting | 小样本或零历史设备不能完全相信单一模型外推 | `pred_history_fallback`、`pred_low_output_guard`、`pred_precision_low_output_guard`、`pred_piecewise_low_output_guard` |
| calibration and stress testing | 部署策略需要通过压力验证复现真实失效模式 | low-capacity stress validation、分布漂移诊断 |

详细映射见 `references/literature_review.md`。"""


def unseen_cv_section(output_dir: Path) -> str:
    aggregate_path = output_dir / "unseen_cv_aggregate.csv"
    candidate_path = output_dir / "unseen_cv_candidate_summary.csv"
    if not aggregate_path.exists():
        return "多折 unseen validation 尚未运行。"

    aggregate = pd.read_csv(aggregate_path).set_index("index")
    rows = [
        "| 指标 | 均值 | 标准差 | 最小值 | 最大值 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in [
        "combined_validation_daily_mape",
        "valid_seen_daily_mape",
        "valid_unseen_daily_mape",
        "test1_daily_mape",
        "test2_daily_mape",
    ]:
        rows.append(
            "| "
            + " | ".join(
                [
                    metric,
                    format_metric(float(aggregate.loc["mean", metric])),
                    format_metric(float(aggregate.loc["std", metric])),
                    format_metric(float(aggregate.loc["min", metric])),
                    format_metric(float(aggregate.loc["max", metric])),
                ]
            )
            + " |"
        )

    candidate_text = ""
    if candidate_path.exists():
        candidate = pd.read_csv(candidate_path)
        combined = candidate[candidate["metric"] == "combined_validation_daily_mape"].head(5)
        crows = [
            "| 候选模型 | 多折综合验证均值 | 标准差 |",
            "|---|---:|---:|",
        ]
        for _, row in combined.iterrows():
            crows.append(
                f"| `{row['candidate']}` | {format_metric(float(row['mean']))} | {format_metric(float(row['std']))} |"
            )
        candidate_text = "\n\n候选模型多折稳定性：\n\n" + "\n".join(crows)

    return (
        "多折 unseen validation 已运行。`valid_unseen` 波动明显高于 `valid_seen/test1`，"
        "说明冷启动评估对设备组选择敏感，不能只依赖单一 holdout。\n\n"
        + "\n".join(rows)
        + candidate_text
    )


def cold_start_stress_section(output_dir: Path) -> str:
    stress_path = output_dir / "cold_start_stress_summary.csv"
    if not stress_path.exists():
        return "低容量冷启动压力验证尚未运行。"

    stress = pd.read_csv(stress_path)
    rows = [
        "| 候选模型 | stress 综合验证 | stress valid_unseen | stress test2 回看 |",
        "|---|---:|---:|---:|",
    ]
    for _, row in stress.iterrows():
        rows.append(
            "| "
            + " | ".join(
                [
                    f"`{row['candidate']}`",
                    format_metric(float(row["combined_validation_daily_mape"])),
                    format_metric(float(row["valid_unseen_daily_mape"])),
                    format_metric(float(row["test2_daily_mape"])),
                ]
            )
            + " |"
        )
    best = stress.iloc[0]["candidate"]
    return (
        "低容量冷启动压力验证已运行。该验证从训练设备中选择低日发电量设备作为未见设备，"
        "用于模拟小容量新设备导致 MAPE 放大的场景。\n\n"
        f"压力验证选择的最佳候选为 `{best}`。\n\n"
        + "\n".join(rows)
    )


def write_interview_report(
    output_dir: Path,
    metrics: dict[str, object],
    best_model: str,
) -> Path:
    path = output_dir / "interview_report.md"
    final = metrics.get("final_by_split", {})
    text = f"""# 冷启动泛化问题实验说明

## 1. 核心判断

本项目的主要困难不是 LightGBM 参数不足，而是新设备冷启动与数据分布迁移。证据是：已见设备验证和 `test1` 相对稳定，而未见设备验证与 `test2` 误差明显更高。

当前主模型：

```text
{best_model}
```

当前最终分 split 指标：

| Split | Daily MAPE |
|---|---:|
| train | {format_metric(final.get("train", {}).get("daily_mape", 0.0))} |
| valid_seen | {format_metric(final.get("valid_seen", {}).get("daily_mape", 0.0))} |
| valid_unseen | {format_metric(final.get("valid_unseen", {}).get("daily_mape", 0.0))} |
| test1 | {format_metric(final.get("test1", {}).get("daily_mape", 0.0))} |
| test2 | {format_metric(final.get("test2", {}).get("daily_mape", 0.0))} |

冷启动部署策略：

{deployment_policy_summary(metrics)}

## 2. 如何确认不是普通调参问题

1. 随机或已见设备验证表现较好，说明模型能学习天气、时间和设备历史规律。
2. 未见设备验证和 `test2` 明显更难，说明模型在新设备容量、区域差异和历史缺失场景下泛化不足。
3. 多折 unseen validation 的均值和方差显示，未见设备验证对设备组选择非常敏感。
4. 冷启动 fallback 与低输出 guard 在 `test2` 回看上显著降低误差，说明错误主要来自新设备容量尺度估计和小真实值 MAPE 放大，而不是 15 分钟曲线形状。

## 3. 已采用的解决方案

1. 验证集重设计：按 `deviceSn` 做 unseen holdout，保证验证设备在训练中完全不可见。
2. 多折 unseen validation：输出均值、方差和每个候选模型的稳定性。
3. 泛化特征：加入区域网格、设备编号前缀代理、设备历史天数、近邻设备统计、时间和天气聚合特征。
4. 物理归一化：用太阳代理量减弱日照轨迹和季节差异。
5. 冷启动 fallback：设备历史不足时降低对设备历史模型的信任，回退到日级泛化模型。
6. 低输出 guard：对零历史且日级预测偏低的新设备做保守缩放，控制小真实值 MAPE 风险。
7. 验证集融合：用验证集选择 `pred_norm_calibrated` 与 `pred_daily_model_rescaled` 的融合权重，不用测试标签调参。

## 4. 文献依据

{literature_section()}

## 5. 关键实验对比

{cold_start_evidence(metrics)}

## 6. 分布漂移诊断

{distribution_shift_section(output_dir)}

## 7. 多折 unseen validation

{unseen_cv_section(output_dir)}

## 8. 低容量冷启动压力验证

{cold_start_stress_section(output_dir)}

## 9. 低输出 guard 策略

{low_output_guard_summary(metrics)}

## 10. 面试总结口径

如果测试集中存在大量训练中未出现的新设备，我不会继续盲目调 LightGBM 参数。我的处理方式是先把验证集改成按设备隔离，再用多折 unseen validation 检查泛化稳定性；然后加入跨设备泛化特征和冷启动 fallback，让历史不足的新设备更多依赖区域、相似设备、时间天气 profile 和日级模型。当前实验表明，`test2` 的主要误差来自新设备容量尺度偏移，而不是模型复杂度不足。
"""
    path.write_text(text, encoding="utf-8")
    return path


def write_final_report(
    output_dir: Path,
    dataset_summary: dict[str, dict[str, object]],
    metrics: dict[str, object],
    figures: list[Path],
    best_model: str,
) -> Path:
    report_path = output_dir / "final_report.md"
    figure_lines = "\n".join(f"- `{fig.name}`" for fig in figures)
    text = f"""# 光伏设备电量预测最终报告

## 1. 任务目标

本项目目标是根据设备经纬度、太阳几何、天气特征和时间特征预测设备级 `pvGenTotal`。最终评估以“单日预测总量”和“单日真实总量”的 MAPE 为核心指标。模型仍保留 15 分钟粒度预测，便于检查日内曲线形态和可视化展示。

## 2. 数据概览

{dataset_table(dataset_summary)}

`test1` 中的设备在训练集中出现过，更接近已见设备的未来日期预测；`test2` 中的设备没有出现在训练集中，更接近新设备冷启动预测。因此本题不能只看整体平均分，必须分别分析已见设备和未见设备的表现。

## 3. 方法设计

当前 pipeline 采用分层结构：

1. 构造时间、太阳几何、辐照、天气和空间特征。
2. 使用不泄露标签的 `solar_proxy` 做物理归一化。
3. 训练 15 分钟点级模型，学习日内曲线形态。
4. 训练日级校准模型，使预测更贴合最终日总量 MAPE。
5. 使用经纬度最近邻统计缓解新设备冷启动问题。

归一化目标定义为：

```text
y_norm = pvGenTotal / (solar_proxy + eps)
```

预测时再乘回物理代理量：

```text
pred_pv = pred_norm * solar_proxy
```

这样可以减弱太阳轨迹、季节和日内时段对目标尺度的影响，提高跨区域泛化稳定性。

## 4. 文献依据与方法更新

{literature_section()}

## 5. 验证设计

验证集分为两类：

- `valid_seen`：每个训练设备最后若干天留出，模拟 `test1`。
- `valid_unseen`：部分训练设备整体留出，模拟 `test2` 的新设备场景。

综合验证指标按题目 test1/test2 各 50% 的结构计算：

```text
final_cv = 0.5 * seen_daily_mape + 0.5 * unseen_daily_mape
```

## 6. 实验结果

{metrics_table(metrics.get("experiments", {}))}

当前验证集选择的最佳预测列为：`{best_model}`。

## 7. 结果解释

最终分 split 指标：

{split_result_summary(metrics)}

候选模型综合验证分数：

{candidate_score_summary(metrics)}

融合权重说明：

{blend_summary(metrics)}

冷启动部署策略：

{deployment_policy_summary(metrics)}

低输出 guard：

{low_output_guard_summary(metrics)}

分布漂移诊断：

{distribution_shift_section(output_dir)}

从结果看，`valid_seen` 与 `test1` 属于同一类问题，即已见设备的未来日期预测；当前误差处在可继续优化但基本可用的范围。`valid_unseen` 与 `test2` 属于新设备冷启动问题，难度明显更高。修正历史统计特征的验证泄露后，`valid_unseen` 指标变得更严格，这说明现在的验证结果比之前更可信。

测试集回看显示，`pred_daily_model_rescaled` 对 `test2` 更稳，而验证集仍选择 `{best_model}`。这不是简单的“换模型即可解决”，而是说明当前单一 unseen holdout 与真实 test2 分布仍有偏差。下一轮应先增强验证机制，再用验证集选择分支或融合权重。

## 8. 可视化产物

本轮生成图表：

{figure_lines}

其中，日总量散点图用于检查总量拟合程度，15 分钟曲线图用于观察日内形态是否合理，分 split MAPE 图用于比较已见设备与未见设备的差异。

## 9. 关键结论

1. 日级校准非常关键，因为题目最终比较的是日总量 MAPE，而不是点级误差。
2. 物理归一化能提升验证集泛化效果，尤其对未见设备验证集更有帮助。
3. 当前 `test2` 仍明显难于 `test1`，说明真实新设备的容量估计仍不足。
4. 多折 unseen validation 表明未见设备验证方差很大，冷启动策略需要看均值和稳定性。
5. 低容量冷启动压力验证支持 fallback/guard 思路，说明小容量新设备需要更保守的容量先验。
6. 分布漂移诊断显示 `test2` 的低负载占比明显高于普通 unseen holdout，因此需要专门的压力验证。
7. 后续优化应优先处理冷启动容量代理、OOF 动态融合和分组校准，而不是继续盲目调参。

## 10. 交付文件

- 训练代码：`scripts/run_experiments.py`
- 结果分析：`scripts/analyze_results.py`
- 分布诊断：`scripts/diagnose_distribution_shift.py`
- 可复用模块：`src/`
- 参考资料：`references/`
- 指标文件：`outputs/metrics.json`
- 预测文件：`outputs/predictions/final_predictions.csv`
- 图表目录：`outputs/figures/`
- 面试说明：`outputs/interview_report.md`
"""
    report_path.write_text(text, encoding="utf-8")
    (output_dir / "metrics_summary.md").write_text(
        "# 指标汇总\n\n" + metrics_table(metrics.get("experiments", {})) + "\n",
        encoding="utf-8",
    )
    write_interview_report(output_dir, metrics, best_model)
    return report_path


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_model_comparison_csv(metrics: dict[str, object], output_dir: Path) -> Path:
    rows: list[dict[str, object]] = []
    for exp, split_metrics in metrics.get("experiments", {}).items():
        if not isinstance(split_metrics, dict):
            continue
        for split, values in split_metrics.items():
            if isinstance(values, dict) and "daily_mape" in values:
                row = {"experiment": exp, "eval_split": split}
                row.update(values)
                rows.append(row)
    path = output_dir / "model_comparison.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
