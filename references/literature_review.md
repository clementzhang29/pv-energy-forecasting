# 文献依据与方法映射

## 1. 研究结论摘要

近期能源预测和建筑负荷预测研究基本集中在三条主线：

1. **Cross-domain / cold-start forecasting**：不再只做随机切分，而是按建筑、住宅、区域、气候区、设备类型或设备编号做跨域验证，真实评估模型在新对象上的泛化能力。
2. **Transfer learning / domain adaptation**：利用多源建筑或设备的历史经验，把共性时间规律、气象响应和静态属性迁移到新设备，缓解冷启动样本不足。
3. **Fallback + calibration + robust forecasting**：当目标设备缺少历史数据时，不让复杂模型单独外推，而是结合区域、相似设备、容量代理、物理约束和分组校准做风险控制。

本项目 `test2` 的高 MAPE 与这些研究中的 cold-start domain shift 高度一致。因此优化方向不是继续盲目调 LightGBM 参数，而是重构验证集、增强跨设备特征、引入相似设备迁移、做冷启动 fallback 和低输出场景的鲁棒校准。

## 2. 文献与项目决策映射

| 文献/资料 | 主要观点 | 本项目落地 |
|---|---|---|
| RESCAST-100K: Cross-Domain Residential Load and Indoor Temperature Forecasting (`2606.02852v1.pdf`) | 通过地理、气候、建筑结构、设备类型等可解释轴构造 source/target domain，评估 transfer learning、domain adaptation 和 zero-shot 泛化。 | 将验证集从随机切分改为按 `deviceSn` 留出；增加多折 unseen validation；报告中分开 `valid_seen/test1` 与 `valid_unseen/test2`。 |
| Toward a Foundational Thermal Model for Residential Buildings (`2605.01364v1`) | 跨建筑、跨气候区泛化需要物理先验、静态属性和时间依赖共同建模，避免记住单个建筑模式。 | 使用太阳几何、辐照代理、经纬度、区域网格和设备编号族群作为跨设备特征；避免把设备历史统计泄露到 unseen 验证。 |
| Enhancing Masked Time-Series Modeling via Dropping Patches (`2412.15315v1`) | 时间序列表征预训练和 patch-level dropout 有助于 cross-domain、few-shot 和 cold-start 场景。 | 当前阶段暂不引入深度预训练，但在后续计划中保留自监督时序表征、few-shot fine-tuning 和 domain adaptation 实验。 |
| BuilDa synthetic building data generation and transfer learning (`2512.00483v2.pdf`) | 大规模合成建筑数据可用于 transfer learning，并通过 fine-tuning 提升小样本建筑建模。 | 将低容量压力验证作为一种训练内模拟分布漂移的方法；后续可构造合成低容量/极端天气样本做鲁棒训练。 |
| Machine learning-based energy management and power forecasting in grid-connected microgrids (`s41598-024-70336-3.pdf`) | 可再生能源预测需要历史发电、天气、动态电网/环境条件联合建模，并比较传统基线和机器学习模型。 | 保留 irradiance baseline、raw model、physics-normalized model、daily calibration 等多模型对比，避免只有单一模型结果。 |
| HyperLoad (`2512.19114v1.pdf`) | 小样本、冷启动、多源数据碎片和分布漂移是绿色能源/负荷预测的核心难点。 | 在面试报告中明确：`test2` 不是普通精度问题，而是新设备历史缺失、容量代理不足和小真实值 MAPE 放大的综合问题。 |

## 3. 对当前项目方法的更新

### 3.1 验证方式

原始随机或时间切分容易让同一设备历史同时出现在训练和验证中，导致冷启动泛化被高估。更新后的 harness 使用：

- `valid_seen`：训练设备的未来日期，用于模拟 `test1`。
- `valid_unseen`：训练设备整体留出，用于模拟 `test2`。
- multi-fold unseen validation：按 `deviceSn` 做多折留出，观察均值和方差。
- low-capacity stress validation：专门留出低日发电量设备，模拟小容量新设备和 MAPE 放大。

### 3.2 特征方式

模型不只记忆设备 ID，而是尽量描述设备和环境：

- 太阳几何与辐照代理：`solar_proxy`、`elevation_weight`、`radiation_proxy`。
- 地理与区域：经纬度、5 度/10 度区域网格。
- 相似设备迁移：经纬度最近邻日均值、P90、距离。
- 设备族群代理：设备编号前缀。
- 历史置信度：`device_history_days`，用于判断是否应该相信设备历史模型。

### 3.3 部署策略

默认模型仍通过验证集选择，避免用测试标签调权；但真实部署可增加风险控制分支：

```text
deployment_prediction = pred_precision_low_output_guard
```

该策略在设备无历史且日级 fallback 预测极低时触发保守缩放，用于降低低真实值下 MAPE 被放大的风险，同时减少对普通 unseen 设备的扰动。更宽松的 `pred_low_output_guard` 在低容量压力验证中表现更强，因此作为极端低容量场景的压力候选保留。`pred_piecewise_low_output_guard` 进一步验证了分段缩放思路，但当前仅作为消融候选保留。三者都不是替代验证最优模型的普通调参，而是由分布诊断、低容量压力验证和多折 unseen validation 支持的部署风险控制策略。

## 4. 下一步研究路线

1. **分组校准**：按区域、设备族群、最近邻容量桶、低输出风险分组做 bias correction。
2. **迁移学习**：使用多设备训练共享表示，再对少量新设备样本做快速校准。
3. **自监督时序表征**：参考 DropPatch 类方法，学习对缺失片段、冷启动和少样本更稳健的日内曲线表示。
4. **概率预测**：引入 quantile regression，输出置信区间；对高不确定新设备触发更保守 fallback。
5. **合成压力样本**：参考 synthetic building data 思路，构造低容量、极端天气、低辐照等分布压力测试集。

## 5. 面试表达

可以这样概括：

```text
我没有把 test2 高误差当成普通 LightGBM 调参问题，而是先做分布诊断。结果显示 test2 是未见设备，并且低日发电量样本比例明显更高；MAPE 对小真实值敏感，所以同样绝对误差会被放大。根据 cross-domain forecasting 和 transfer learning 文献，我把验证集改为 device-level GroupKFold，引入相似设备迁移特征，并增加低容量压力验证。最后把默认验证最优模型和冷启动部署策略分开：默认模型保证验证选择合规，部署策略用 low-output guard 控制新设备低容量风险。
```
