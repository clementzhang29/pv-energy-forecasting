# 多折未见设备验证

该实验按 `deviceSn` 对训练设备做多折留出，每一折都模拟新设备冷启动。折内仍保留 seen-device 时间留出，用于同时观察 test1 类问题和 test2 类问题。

## 每折结果

| fold | best_prediction_column | best_blend_alpha | combined_validation_daily_mape | valid_seen_daily_mape | valid_unseen_daily_mape | test1_daily_mape | test2_daily_mape | candidate_pred_irradiance_baseline_combined | candidate_pred_irradiance_baseline_valid_unseen | candidate_pred_raw_point_combined | candidate_pred_raw_point_valid_unseen | candidate_pred_norm_point_combined | candidate_pred_norm_point_valid_unseen | candidate_pred_daily_model_rescaled_combined | candidate_pred_daily_model_rescaled_valid_unseen | candidate_pred_norm_calibrated_combined | candidate_pred_norm_calibrated_valid_unseen | candidate_pred_validated_blend_combined | candidate_pred_validated_blend_valid_unseen | candidate_pred_history_fallback_combined | candidate_pred_history_fallback_valid_unseen | candidate_pred_low_output_guard_combined | candidate_pred_low_output_guard_valid_unseen | candidate_pred_precision_low_output_guard_combined | candidate_pred_precision_low_output_guard_valid_unseen | candidate_pred_piecewise_low_output_guard_combined | candidate_pred_piecewise_low_output_guard_valid_unseen | candidate_pred_seen_new_branch_combined | candidate_pred_seen_new_branch_valid_unseen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | pred_validated_blend | 0.7000 | 0.0942 | 0.0878 | 0.1005 | 0.0845 | 0.6888 | 0.3878 | 0.4358 | 0.3401 | 0.3239 | 0.4124 | 0.3777 | 0.1098 | 0.1449 | 0.0973 | 0.0944 | 0.0942 | 0.1005 | 0.1164 | 0.1449 | 0.1190 | 0.1501 | 0.1170 | 0.1462 | 0.1165 | 0.1452 | 0.1225 | 0.1449 |
| 1 | pred_validated_blend | 0.8000 | 0.1220 | 0.0881 | 0.1559 | 0.0870 | 0.7395 | 0.4400 | 0.5480 | 0.3287 | 0.3319 | 0.4786 | 0.4872 | 0.1552 | 0.2286 | 0.1237 | 0.1545 | 0.1220 | 0.1559 | 0.1584 | 0.2286 | 0.1619 | 0.2356 | 0.1591 | 0.2301 | 0.1589 | 0.2297 | 0.1608 | 0.2286 |
| 2 | pred_low_output_guard | 0.4000 | 0.1954 | 0.0725 | 0.3183 | 0.0877 | 0.5879 | 0.6701 | 1.0668 | 0.5464 | 0.7711 | 0.6514 | 0.9433 | 0.2258 | 0.3752 | 0.2315 | 0.3850 | 0.2238 | 0.3751 | 0.2238 | 0.3752 | 0.1954 | 0.3183 | 0.2234 | 0.3742 | 0.2233 | 0.3740 | 0.2265 | 0.3752 |

## 汇总统计

| index | combined_validation_daily_mape | valid_seen_daily_mape | valid_unseen_daily_mape | test1_daily_mape | test2_daily_mape |
| --- | --- | --- | --- | --- | --- |
| mean | 0.1372 | 0.0828 | 0.1916 | 0.0864 | 0.6721 |
| std | 0.0523 | 0.0089 | 0.1132 | 0.0017 | 0.0772 |
| min | 0.0942 | 0.0725 | 0.1005 | 0.0845 | 0.5879 |
| max | 0.1954 | 0.0881 | 0.3183 | 0.0877 | 0.7395 |

## 候选模型多折稳定性

| candidate | metric | mean | std | min | max |
| --- | --- | --- | --- | --- | --- |
| pred_validated_blend | combined_validation_daily_mape | 0.1467 | 0.0683 | 0.0942 | 0.2238 |
| pred_norm_calibrated | combined_validation_daily_mape | 0.1508 | 0.0711 | 0.0973 | 0.2315 |
| pred_low_output_guard | combined_validation_daily_mape | 0.1588 | 0.0383 | 0.1190 | 0.1954 |
| pred_daily_model_rescaled | combined_validation_daily_mape | 0.1636 | 0.0584 | 0.1098 | 0.2258 |
| pred_history_fallback | combined_validation_daily_mape | 0.1662 | 0.0542 | 0.1164 | 0.2238 |
| pred_piecewise_low_output_guard | combined_validation_daily_mape | 0.1662 | 0.0537 | 0.1165 | 0.2233 |
| pred_precision_low_output_guard | combined_validation_daily_mape | 0.1665 | 0.0535 | 0.1170 | 0.2234 |
| pred_seen_new_branch | combined_validation_daily_mape | 0.1699 | 0.0526 | 0.1225 | 0.2265 |
| pred_raw_point | combined_validation_daily_mape | 0.4050 | 0.1225 | 0.3287 | 0.5464 |
| pred_irradiance_baseline | combined_validation_daily_mape | 0.4993 | 0.1502 | 0.3878 | 0.6701 |
| pred_norm_point | combined_validation_daily_mape | 0.5141 | 0.1234 | 0.4124 | 0.6514 |
| pred_validated_blend | valid_unseen_daily_mape | 0.2105 | 0.1453 | 0.1005 | 0.3751 |
| pred_norm_calibrated | valid_unseen_daily_mape | 0.2113 | 0.1534 | 0.0944 | 0.3850 |
| pred_low_output_guard | valid_unseen_daily_mape | 0.2347 | 0.0841 | 0.1501 | 0.3183 |
| pred_daily_model_rescaled | valid_unseen_daily_mape | 0.2496 | 0.1165 | 0.1449 | 0.3752 |
| pred_history_fallback | valid_unseen_daily_mape | 0.2496 | 0.1165 | 0.1449 | 0.3752 |
| pred_seen_new_branch | valid_unseen_daily_mape | 0.2496 | 0.1165 | 0.1449 | 0.3752 |
| pred_piecewise_low_output_guard | valid_unseen_daily_mape | 0.2497 | 0.1157 | 0.1452 | 0.3740 |
| pred_precision_low_output_guard | valid_unseen_daily_mape | 0.2502 | 0.1153 | 0.1462 | 0.3742 |
| pred_raw_point | valid_unseen_daily_mape | 0.4756 | 0.2559 | 0.3239 | 0.7711 |
| pred_norm_point | valid_unseen_daily_mape | 0.6027 | 0.3000 | 0.3777 | 0.9433 |
| pred_irradiance_baseline | valid_unseen_daily_mape | 0.6835 | 0.3366 | 0.4358 | 1.0668 |

## 解释

如果 `valid_unseen_daily_mape` 的标准差较大，说明单一 unseen holdout 不稳定，后续模型选择应优先依赖多折均值，而不是某一折的局部最优。
