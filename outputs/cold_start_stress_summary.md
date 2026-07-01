# 冷启动压力验证

该实验从训练设备中选择低日发电量设备作为 `valid_unseen`，模拟真实冷启动中小容量设备导致 MAPE 放大的场景。

最佳验证列：`pred_low_output_guard`

| candidate | combined_validation_daily_mape | valid_seen_daily_mape | valid_unseen_daily_mape | test1_daily_mape | test2_daily_mape |
| --- | --- | --- | --- | --- | --- |
| pred_low_output_guard | 0.1696 | 0.0746 | 0.2646 | 0.0768 | 0.5802 |
| pred_piecewise_low_output_guard | 0.1974 | 0.0746 | 0.3202 | 0.0768 | 0.7893 |
| pred_precision_low_output_guard | 0.1979 | 0.0746 | 0.3211 | 0.0768 | 0.7924 |
| pred_history_fallback | 0.2016 | 0.0746 | 0.3285 | 0.0768 | 0.7802 |
| pred_validated_blend | 0.2020 | 0.0746 | 0.3293 | 0.0768 | 0.7962 |
| pred_daily_model_rescaled | 0.2028 | 0.0772 | 0.3285 | 0.0777 | 0.7802 |
| pred_seen_new_branch | 0.2065 | 0.0845 | 0.3285 | 0.0887 | 0.7802 |
| pred_norm_calibrated | 0.2154 | 0.0845 | 0.3462 | 0.0887 | 0.8928 |

解释：如果低输出 guard 或 fallback 在该压力验证下优于普通校准模型，说明低容量新设备需要更保守的容量先验；如果主融合模型仍最好，则说明 fallback 应作为风险诊断或分段策略，而不是全局替代。
