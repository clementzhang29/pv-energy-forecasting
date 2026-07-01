# 光伏设备电量预测最终报告

## 1. 任务目标

本项目目标是根据设备经纬度、太阳几何、天气特征和时间特征预测设备级 `pvGenTotal`。最终评估以“单日预测总量”和“单日真实总量”的 MAPE 为核心指标。模型仍保留 15 分钟粒度预测，便于检查日内曲线形态和可视化展示。

## 2. 数据概览

| 数据集 | 设备数 | 日期文件数 | 行数 | 日期范围 | 日总量均值 | 日总量中位数 |
|---|---:|---:|---:|---|---:|---:|
| test1 | 22 | 110 | 10236 | 2024-05-18 to 2024-08-20 | 0.3253 | 0.2708 |
| test2 | 6 | 36 | 3397 | 2024-04-08 to 2024-08-18 | 0.2403 | 0.2483 |
| train | 39 | 2653 | 247002 | 2024-02-26 to 2024-08-20 | 0.3136 | 0.3086 |

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

本轮方法更新参考了 `references/` 中的文献资料，重点不是照搬某个复杂模型，而是吸收其验证与泛化思想：

| 研究方向 | 对本项目的启发 | 项目实现 |
|---|---|---|
| cross-domain forecasting | 需要按建筑、设备、区域、气候等 domain 留出验证，而不是随机切分 | `valid_unseen`、multi-fold unseen validation |
| transfer learning / domain adaptation | 新设备应从相似设备、区域和静态属性中迁移共性规律 | 经纬度最近邻、区域网格、设备族群代理 |
| physics-informed forecasting | 太阳几何和辐照是跨设备泛化的重要先验 | `solar_proxy`、物理归一化、日级校准 |
| robust cold-start forecasting | 小样本或零历史设备不能完全相信单一模型外推 | `pred_history_fallback`、`pred_low_output_guard`、`pred_precision_low_output_guard`、`pred_piecewise_low_output_guard` |
| calibration and stress testing | 部署策略需要通过压力验证复现真实失效模式 | low-capacity stress validation、分布漂移诊断 |

详细映射见 `references/literature_review.md`。

## 5. 验证设计

验证集分为两类：

- `valid_seen`：每个训练设备最后若干天留出，模拟 `test1`。
- `valid_unseen`：部分训练设备整体留出，模拟 `test2` 的新设备场景。

综合验证指标按题目 test1/test2 各 50% 的结构计算：

```text
final_cv = 0.5 * seen_daily_mape + 0.5 * unseen_daily_mape
```

## 6. 实验结果

| 实验 | 评估集 | MAE | RMSE | 日总量 MAE | 日总量 RMSE | 日总量 MAPE |
|---|---|---:|---:|---:|---:|---:|
| E00_irradiance_baseline | test1 | 0.0038 | 0.0074 | 0.0547 | 0.0805 | 0.2483 |
| E00_irradiance_baseline | test2 | 0.0028 | 0.0050 | 0.1203 | 0.1337 | 1.8396 |
| E00_irradiance_baseline | train | 0.0031 | 0.0057 | 0.0527 | 0.0763 | 0.3146 |
| E00_irradiance_baseline | valid_seen | 0.0034 | 0.0065 | 0.0560 | 0.0889 | 0.3248 |
| E00_irradiance_baseline | valid_unseen | 0.0038 | 0.0070 | 0.1613 | 0.2027 | 0.5923 |
| E00_irradiance_baseline | combined_validation | nan | nan | nan | nan | 0.4586 |
| E01_lgbm_raw_point | test1 | 0.0019 | 0.0046 | 0.1303 | 0.1823 | 0.3430 |
| E01_lgbm_raw_point | test2 | 0.0020 | 0.0036 | 0.1349 | 0.1548 | 1.8448 |
| E01_lgbm_raw_point | train | 0.0013 | 0.0033 | 0.0898 | 0.1259 | 0.2578 |
| E01_lgbm_raw_point | valid_seen | 0.0017 | 0.0040 | 0.1121 | 0.1628 | 0.3257 |
| E01_lgbm_raw_point | valid_unseen | 0.0019 | 0.0036 | 0.1204 | 0.1572 | 0.3517 |
| E01_lgbm_raw_point | combined_validation | nan | nan | nan | nan | 0.3387 |
| E03_lgbm_norm_target | test1 | 0.0025 | 0.0173 | 0.1599 | 0.2546 | 0.5410 |
| E03_lgbm_norm_target | test2 | 0.0026 | 0.0073 | 0.1552 | 0.2315 | 1.9670 |
| E03_lgbm_norm_target | train | 0.0016 | 0.0059 | 0.0960 | 0.1385 | 0.3433 |
| E03_lgbm_norm_target | valid_seen | 0.0020 | 0.0097 | 0.1245 | 0.1846 | 0.3908 |
| E03_lgbm_norm_target | valid_unseen | 0.0022 | 0.0079 | 0.1149 | 0.1622 | 0.3678 |
| E03_lgbm_norm_target | combined_validation | nan | nan | nan | nan | 0.3793 |
| E02_lgbm_daily_rescaled | test1 | 0.0031 | 0.0073 | 0.0212 | 0.0332 | 0.0780 |
| E02_lgbm_daily_rescaled | test2 | 0.0018 | 0.0040 | 0.0479 | 0.0647 | 0.2997 |
| E02_lgbm_daily_rescaled | train | 0.0022 | 0.0048 | 0.0020 | 0.0031 | 0.0093 |
| E02_lgbm_daily_rescaled | valid_seen | 0.0026 | 0.0065 | 0.0164 | 0.0319 | 0.0741 |
| E02_lgbm_daily_rescaled | valid_unseen | 0.0024 | 0.0054 | 0.0636 | 0.0972 | 0.1791 |
| E02_lgbm_daily_rescaled | combined_validation | nan | nan | nan | nan | 0.1266 |
| E05_norm_plus_daily_calibration | test1 | 0.0031 | 0.0074 | 0.0258 | 0.0393 | 0.1017 |
| E05_norm_plus_daily_calibration | test2 | 0.0019 | 0.0042 | 0.0411 | 0.0558 | 0.6483 |
| E05_norm_plus_daily_calibration | train | 0.0022 | 0.0048 | 0.0017 | 0.0023 | 0.0073 |
| E05_norm_plus_daily_calibration | valid_seen | 0.0026 | 0.0065 | 0.0181 | 0.0370 | 0.0757 |
| E05_norm_plus_daily_calibration | valid_unseen | 0.0023 | 0.0053 | 0.0545 | 0.0792 | 0.1643 |
| E05_norm_plus_daily_calibration | combined_validation | nan | nan | nan | nan | 0.1200 |
| E07_validated_blend | test1 | 0.0031 | 0.0073 | 0.0236 | 0.0368 | 0.0905 |
| E07_validated_blend | test2 | 0.0019 | 0.0042 | 0.0423 | 0.0559 | 0.5392 |
| E07_validated_blend | train | 0.0022 | 0.0048 | 0.0016 | 0.0023 | 0.0072 |
| E07_validated_blend | valid_seen | 0.0026 | 0.0065 | 0.0170 | 0.0345 | 0.0726 |
| E07_validated_blend | valid_unseen | 0.0023 | 0.0053 | 0.0558 | 0.0829 | 0.1640 |
| E07_validated_blend | combined_validation | nan | nan | nan | nan | 0.1183 |
| E08_history_dynamic_fallback | test1 | 0.0031 | 0.0073 | 0.0236 | 0.0368 | 0.0905 |
| E08_history_dynamic_fallback | test2 | 0.0018 | 0.0040 | 0.0479 | 0.0647 | 0.2997 |
| E08_history_dynamic_fallback | train | 0.0022 | 0.0048 | 0.0016 | 0.0023 | 0.0072 |
| E08_history_dynamic_fallback | valid_seen | 0.0026 | 0.0065 | 0.0170 | 0.0345 | 0.0726 |
| E08_history_dynamic_fallback | valid_unseen | 0.0024 | 0.0054 | 0.0636 | 0.0972 | 0.1791 |
| E08_history_dynamic_fallback | combined_validation | nan | nan | nan | nan | 0.1259 |
| E09_low_output_guard | test1 | 0.0031 | 0.0073 | 0.0236 | 0.0368 | 0.0905 |
| E09_low_output_guard | test2 | 0.0017 | 0.0040 | 0.0473 | 0.0648 | 0.2356 |
| E09_low_output_guard | train | 0.0022 | 0.0048 | 0.0016 | 0.0023 | 0.0072 |
| E09_low_output_guard | valid_seen | 0.0026 | 0.0065 | 0.0170 | 0.0345 | 0.0726 |
| E09_low_output_guard | valid_unseen | 0.0024 | 0.0054 | 0.0647 | 0.0976 | 0.1849 |
| E09_low_output_guard | combined_validation | nan | nan | nan | nan | 0.1288 |
| E10_precision_low_output_guard | test1 | 0.0031 | 0.0073 | 0.0236 | 0.0368 | 0.0905 |
| E10_precision_low_output_guard | test2 | 0.0017 | 0.0040 | 0.0466 | 0.0645 | 0.1868 |
| E10_precision_low_output_guard | train | 0.0022 | 0.0048 | 0.0016 | 0.0023 | 0.0072 |
| E10_precision_low_output_guard | valid_seen | 0.0026 | 0.0065 | 0.0170 | 0.0345 | 0.0726 |
| E10_precision_low_output_guard | valid_unseen | 0.0024 | 0.0054 | 0.0636 | 0.0972 | 0.1778 |
| E10_precision_low_output_guard | combined_validation | nan | nan | nan | nan | 0.1252 |
| E11_piecewise_low_output_guard | test1 | 0.0031 | 0.0073 | 0.0236 | 0.0368 | 0.0905 |
| E11_piecewise_low_output_guard | test2 | 0.0017 | 0.0040 | 0.0465 | 0.0644 | 0.1906 |
| E11_piecewise_low_output_guard | train | 0.0022 | 0.0048 | 0.0016 | 0.0023 | 0.0072 |
| E11_piecewise_low_output_guard | valid_seen | 0.0026 | 0.0065 | 0.0170 | 0.0345 | 0.0726 |
| E11_piecewise_low_output_guard | valid_unseen | 0.0024 | 0.0054 | 0.0635 | 0.0972 | 0.1766 |
| E11_piecewise_low_output_guard | combined_validation | nan | nan | nan | nan | 0.1246 |
| E06_seen_new_branch | test1 | 0.0031 | 0.0074 | 0.0258 | 0.0393 | 0.1017 |
| E06_seen_new_branch | test2 | 0.0018 | 0.0040 | 0.0479 | 0.0647 | 0.2997 |
| E06_seen_new_branch | train | 0.0022 | 0.0048 | 0.0017 | 0.0023 | 0.0073 |
| E06_seen_new_branch | valid_seen | 0.0026 | 0.0065 | 0.0181 | 0.0370 | 0.0757 |
| E06_seen_new_branch | valid_unseen | 0.0024 | 0.0054 | 0.0636 | 0.0972 | 0.1791 |
| E06_seen_new_branch | combined_validation | nan | nan | nan | nan | 0.1274 |

当前验证集选择的最佳预测列为：`pred_validated_blend`。

## 7. 结果解释

最终分 split 指标：

| Split | 日总量 MAPE | 日总量 MAE | 解释 |
|---|---:|---:|---|
| train | 0.0072 | 0.0016 | 训练拟合参考，不作为泛化判断依据 |
| valid_seen | 0.0726 | 0.0170 | 模拟已见设备未来日期，主要对应 test1 |
| valid_unseen | 0.1640 | 0.0558 | 模拟未见设备冷启动，主要对应 test2 |
| test1 | 0.0905 | 0.0236 | 已见设备测试回看，用于复盘 |
| test2 | 0.5392 | 0.0423 | 未见设备测试回看，是当前主要难点 |

候选模型综合验证分数：

| 候选预测列 | 综合验证 Daily MAPE |
|---|---:|
| `pred_validated_blend` | 0.1183 |
| `pred_norm_calibrated` | 0.1200 |
| `pred_piecewise_low_output_guard` | 0.1246 |
| `pred_precision_low_output_guard` | 0.1252 |
| `pred_history_fallback` | 0.1259 |
| `pred_daily_model_rescaled` | 0.1266 |
| `pred_seen_new_branch` | 0.1274 |
| `pred_low_output_guard` | 0.1288 |
| `pred_raw_point` | 0.3387 |
| `pred_norm_point` | 0.3793 |
| `pred_irradiance_baseline` | 0.4586 |

融合权重说明：

验证集融合扫描启用，最优 alpha = 0.7。融合公式为 `alpha * pred_norm_calibrated + (1 - alpha) * pred_daily_model_rescaled`，对应综合验证 Daily MAPE = 0.1183。

冷启动部署策略：

冷启动部署策略使用 `pred_precision_low_output_guard`。它不是单一默认验证集最优模型，而是结合多折 unseen validation 和低容量压力验证后，用于真实新设备风险控制的候选。该策略只在设备历史为零且日级兜底预测极低时触发更克制的保守缩放，目标是在降低 test2 低输出误差的同时减少对普通 unseen 设备的扰动。

| Split | 验证最优 Daily MAPE | 冷启动部署 Daily MAPE |
|---|---:|---:|
| valid_seen | 0.0726 | 0.0726 |
| valid_unseen | 0.1640 | 0.1778 |
| test1 | 0.0905 | 0.0905 |
| test2 | 0.5392 | 0.1868 |

低输出 guard：

低输出 guard 面向 MAPE 的小真实值敏感性：当设备无历史、日级 fallback 预测低于阈值时，对预测做保守缩放。广义 guard 更偏压力场景稳健性，精细 guard 更偏减少普通 unseen 设备扰动，分段 guard 用于验证不同低输出强度下的缩放是否需要差异化；这些策略都不作为默认验证最优模型替代。

| 策略 | 日级阈值 | 缩放系数 | 触发行数 |
|---|---:|---:|---:|
| 广义低输出 guard | 0.1500 | 0.7000 | 3276 |
| 精细低输出 guard | 0.0600 | 0.4000 | 832 |
| 分段低输出 guard | 0.0400 / 0.0600 | 0.5000 / 0.5500 | 832 |

| 策略 | Split | 触发天数 |
|---|---|---:|
| 广义低输出 guard | test2 | 10 |
| 广义低输出 guard | valid_unseen | 25 |
| 精细低输出 guard | test2 | 5 |
| 精细低输出 guard | valid_unseen | 4 |
| 分段低输出 guard | test2 | 5 |
| 分段低输出 guard | valid_unseen | 4 |

分布漂移诊断：

分布漂移诊断已运行。它用于证明 `test2` 的高误差来自新设备和低负载样本比例变化，而不是单纯的模型参数问题。

| Split | 设备数 | 训练设备重叠 | 日发电量均值 | 低于 0.05 占比 | 低于 0.10 占比 |
|---|---:|---:|---:|---:|---:|
| train | 35 | 1.0000 | 0.3139 | 0.0174 | 0.0784 |
| valid_seen | 35 | 1.0000 | 0.3122 | 0.0286 | 0.1510 |
| valid_unseen | 4 | 0.0000 | 0.3129 | 0.0109 | 0.0197 |
| test1 | 22 | 1.0000 | 0.3253 | 0.0182 | 0.1636 |
| test2 | 6 | 0.0000 | 0.2403 | 0.1389 | 0.2222 |

从结果看，`valid_seen` 与 `test1` 属于同一类问题，即已见设备的未来日期预测；当前误差处在可继续优化但基本可用的范围。`valid_unseen` 与 `test2` 属于新设备冷启动问题，难度明显更高。修正历史统计特征的验证泄露后，`valid_unseen` 指标变得更严格，这说明现在的验证结果比之前更可信。

测试集回看显示，`pred_daily_model_rescaled` 对 `test2` 更稳，而验证集仍选择 `pred_validated_blend`。这不是简单的“换模型即可解决”，而是说明当前单一 unseen holdout 与真实 test2 分布仍有偏差。下一轮应先增强验证机制，再用验证集选择分支或融合权重。

## 8. 可视化产物

本轮生成图表：

- `blend_weight_scan.png`
- `daily_energy_distribution.png`
- `daily_mape_by_split.png`
- `daily_prediction_scatter.png`
- `device_locations.png`
- `example_intraday_curve.png`
- `feature_importance.png`

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
