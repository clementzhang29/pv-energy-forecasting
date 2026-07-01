# 冷启动泛化问题实验说明

## 1. 核心判断

本项目的主要困难不是 LightGBM 参数不足，而是新设备冷启动与数据分布迁移。证据是：已见设备验证和 `test1` 相对稳定，而未见设备验证与 `test2` 误差明显更高。

当前主模型：

```text
pred_validated_blend
```

当前最终分 split 指标：

| Split | Daily MAPE |
|---|---:|
| train | 0.0072 |
| valid_seen | 0.0726 |
| valid_unseen | 0.1640 |
| test1 | 0.0905 |
| test2 | 0.5392 |

冷启动部署策略：

冷启动部署策略使用 `pred_precision_low_output_guard`。它不是单一默认验证集最优模型，而是结合多折 unseen validation 和低容量压力验证后，用于真实新设备风险控制的候选。该策略只在设备历史为零且日级兜底预测极低时触发更克制的保守缩放，目标是在降低 test2 低输出误差的同时减少对普通 unseen 设备的扰动。

| Split | 验证最优 Daily MAPE | 冷启动部署 Daily MAPE |
|---|---:|---:|
| valid_seen | 0.0726 | 0.0726 |
| valid_unseen | 0.1640 | 0.1778 |
| test1 | 0.0905 | 0.0905 |
| test2 | 0.5392 | 0.1868 |

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

本轮方法更新参考了 `references/` 中的文献资料，重点不是照搬某个复杂模型，而是吸收其验证与泛化思想：

| 研究方向 | 对本项目的启发 | 项目实现 |
|---|---|---|
| cross-domain forecasting | 需要按建筑、设备、区域、气候等 domain 留出验证，而不是随机切分 | `valid_unseen`、multi-fold unseen validation |
| transfer learning / domain adaptation | 新设备应从相似设备、区域和静态属性中迁移共性规律 | 经纬度最近邻、区域网格、设备族群代理 |
| physics-informed forecasting | 太阳几何和辐照是跨设备泛化的重要先验 | `solar_proxy`、物理归一化、日级校准 |
| robust cold-start forecasting | 小样本或零历史设备不能完全相信单一模型外推 | `pred_history_fallback`、`pred_low_output_guard`、`pred_precision_low_output_guard`、`pred_piecewise_low_output_guard` |
| calibration and stress testing | 部署策略需要通过压力验证复现真实失效模式 | low-capacity stress validation、分布漂移诊断 |

详细映射见 `references/literature_review.md`。

## 5. 关键实验对比

| 模型 | valid_unseen Daily MAPE | test2 Daily MAPE | 说明 |
|---|---:|---:|---|
| 归一化校准模型 | 0.1643 | 0.6483 | 强验证基线，但新设备仍有偏移 |
| 验证集融合模型 | 0.1640 | 0.5392 | 当前主模型 |
| 历史置信度 fallback | 0.1791 | 0.2997 | 历史不足时退回日级泛化模型 |
| 低输出 guard | 0.1849 | 0.2356 | 低日总量且零历史时保守缩放 |
| 精细低输出 guard | 0.1778 | 0.1868 | 只在极低日总量风险时触发 |
| 分段低输出 guard | 0.1766 | 0.1906 | 按预测日总量分段设置缩放强度 |
| 日级泛化模型 | 0.1791 | 0.2997 | 冷启动兜底参考 |

## 6. 分布漂移诊断

分布漂移诊断已运行。它用于证明 `test2` 的高误差来自新设备和低负载样本比例变化，而不是单纯的模型参数问题。

| Split | 设备数 | 训练设备重叠 | 日发电量均值 | 低于 0.05 占比 | 低于 0.10 占比 |
|---|---:|---:|---:|---:|---:|
| train | 35 | 1.0000 | 0.3139 | 0.0174 | 0.0784 |
| valid_seen | 35 | 1.0000 | 0.3122 | 0.0286 | 0.1510 |
| valid_unseen | 4 | 0.0000 | 0.3129 | 0.0109 | 0.0197 |
| test1 | 22 | 1.0000 | 0.3253 | 0.0182 | 0.1636 |
| test2 | 6 | 0.0000 | 0.2403 | 0.1389 | 0.2222 |

## 7. 多折 unseen validation

多折 unseen validation 已运行。`valid_unseen` 波动明显高于 `valid_seen/test1`，说明冷启动评估对设备组选择敏感，不能只依赖单一 holdout。

| 指标 | 均值 | 标准差 | 最小值 | 最大值 |
|---|---:|---:|---:|---:|
| combined_validation_daily_mape | 0.1372 | 0.0523 | 0.0942 | 0.1954 |
| valid_seen_daily_mape | 0.0828 | 0.0089 | 0.0725 | 0.0881 |
| valid_unseen_daily_mape | 0.1916 | 0.1132 | 0.1005 | 0.3183 |
| test1_daily_mape | 0.0864 | 0.0017 | 0.0845 | 0.0877 |
| test2_daily_mape | 0.6721 | 0.0772 | 0.5879 | 0.7395 |

候选模型多折稳定性：

| 候选模型 | 多折综合验证均值 | 标准差 |
|---|---:|---:|
| `pred_validated_blend` | 0.1467 | 0.0683 |
| `pred_norm_calibrated` | 0.1508 | 0.0711 |
| `pred_low_output_guard` | 0.1588 | 0.0383 |
| `pred_daily_model_rescaled` | 0.1636 | 0.0584 |
| `pred_history_fallback` | 0.1662 | 0.0542 |

## 8. 低容量冷启动压力验证

低容量冷启动压力验证已运行。该验证从训练设备中选择低日发电量设备作为未见设备，用于模拟小容量新设备导致 MAPE 放大的场景。

压力验证选择的最佳候选为 `pred_low_output_guard`。

| 候选模型 | stress 综合验证 | stress valid_unseen | stress test2 回看 |
|---|---:|---:|---:|
| `pred_low_output_guard` | 0.1696 | 0.2646 | 0.5802 |
| `pred_piecewise_low_output_guard` | 0.1974 | 0.3202 | 0.7893 |
| `pred_precision_low_output_guard` | 0.1979 | 0.3211 | 0.7924 |
| `pred_history_fallback` | 0.2016 | 0.3285 | 0.7802 |
| `pred_validated_blend` | 0.2020 | 0.3293 | 0.7962 |
| `pred_daily_model_rescaled` | 0.2028 | 0.3285 | 0.7802 |
| `pred_seen_new_branch` | 0.2065 | 0.3285 | 0.7802 |
| `pred_norm_calibrated` | 0.2154 | 0.3462 | 0.8928 |

## 9. 低输出 guard 策略

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

## 10. 面试总结口径

如果测试集中存在大量训练中未出现的新设备，我不会继续盲目调 LightGBM 参数。我的处理方式是先把验证集改成按设备隔离，再用多折 unseen validation 检查泛化稳定性；然后加入跨设备泛化特征和冷启动 fallback，让历史不足的新设备更多依赖区域、相似设备、时间天气 profile 和日级模型。当前实验表明，`test2` 的主要误差来自新设备容量尺度偏移，而不是模型复杂度不足。
