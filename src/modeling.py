from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ModelResult:
    name: str
    predictions: np.ndarray
    model: object


def _try_lightgbm(params: dict | None = None):
    try:
        from lightgbm import LGBMRegressor

        base = {
            "n_estimators": 700,
            "learning_rate": 0.035,
            "num_leaves": 48,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "min_child_samples": 30,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
        if params:
            base.update(params)
        return LGBMRegressor(**base)
    except Exception:
        return None


def make_gbdt_model(name: str = "gbdt") -> object:
    model = _try_lightgbm()
    if model is not None:
        return model
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=450,
                    learning_rate=0.045,
                    max_leaf_nodes=45,
                    l2_regularization=0.02,
                    random_state=42,
                ),
            ),
        ]
    )


def make_daily_model() -> object:
    model = _try_lightgbm(
        {
            "n_estimators": 500,
            "learning_rate": 0.04,
            "num_leaves": 32,
            "min_child_samples": 15,
        }
    )
    if model is not None:
        return model
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=320,
                    learning_rate=0.04,
                    max_leaf_nodes=31,
                    l2_regularization=0.04,
                    random_state=42,
                ),
            ),
        ]
    )


def fit_predict(
    train_df: pd.DataFrame,
    all_df: pd.DataFrame,
    features: list[str],
    target: str,
    model: object | None = None,
) -> tuple[object, np.ndarray]:
    if model is None:
        model = make_gbdt_model()
    x_train = train_df[features]
    y_train = train_df[target]
    x_all = all_df[features]
    model.fit(x_train, y_train)
    pred = np.asarray(model.predict(x_all), dtype=float)
    return model, pred


def irradiance_baseline(point_df: pd.DataFrame) -> np.ndarray:
    train = point_df[point_df["eval_split"] == "train"]
    device_daily = (
        train.groupby(["deviceSn", "date"], observed=True)["pvGenTotal"]
        .sum()
        .groupby("deviceSn", observed=True)
        .mean()
    )
    global_daily = float(
        train.groupby(["deviceSn", "date"], observed=True)["pvGenTotal"].sum().mean()
    )
    nn_daily = (
        point_df.groupby("deviceSn", observed=True)["nn_3_daily_mean"].median().fillna(global_daily)
    )
    pred = np.zeros(len(point_df), dtype=float)
    for (_, _), idx in point_df.groupby(["deviceSn", "date"], observed=True).groups.items():
        part = point_df.loc[idx]
        device = part["deviceSn"].iloc[0]
        daily_total = float(device_daily.get(device, nn_daily.get(device, global_daily)))
        weights = part["solar_proxy"].to_numpy(dtype=float)
        if weights.sum() <= 0:
            weights = part["elevation_weight"].to_numpy(dtype=float)
        if weights.sum() <= 0:
            pred[idx] = daily_total / len(part)
        else:
            pred[idx] = daily_total * weights / weights.sum()
    return pred


def fit_daily_calibrator(
    point_df: pd.DataFrame,
    raw_pred_col: str,
    daily_df: pd.DataFrame,
    daily_features: list[str],
) -> tuple[object, pd.DataFrame]:
    raw_daily = (
        point_df.groupby(["split", "deviceSn", "date"], observed=True)[raw_pred_col]
        .sum()
        .reset_index(name=f"{raw_pred_col}_sum")
    )
    daily = daily_df.merge(raw_daily, on=["split", "deviceSn", "date"], how="left")
    feature_cols = [f"{raw_pred_col}_sum"] + daily_features
    train = daily[daily["eval_split"] == "train"].copy()
    model = make_daily_model()
    model.fit(train[feature_cols], np.log1p(train["daily_target"]))
    daily["pred_day_calibrated"] = np.expm1(model.predict(daily[feature_cols]))
    daily["pred_day_calibrated"] = daily["pred_day_calibrated"].clip(lower=0)
    return model, daily[["split", "deviceSn", "date", "pred_day_calibrated"]]


def rescale_point_predictions(
    point_df: pd.DataFrame,
    raw_pred_col: str,
    daily_pred: pd.DataFrame,
    output_col: str,
) -> pd.DataFrame:
    frame = point_df.merge(daily_pred, on=["split", "deviceSn", "date"], how="left")
    raw_sum = frame.groupby(["split", "deviceSn", "date"], observed=True)[raw_pred_col].transform(
        "sum"
    )
    scale = frame["pred_day_calibrated"] / np.maximum(raw_sum, 1e-9)
    frame[output_col] = (frame[raw_pred_col] * scale).fillna(0).clip(lower=0)
    solar_off = (frame["is_solar_available"] <= 0) | (frame["solar_proxy"] <= 1e-8)
    frame.loc[solar_off, output_col] = 0.0
    return frame.drop(columns=["pred_day_calibrated"])


def fit_stacking_daily(
    daily_predictions: pd.DataFrame,
    pred_cols: list[str],
) -> tuple[object, pd.DataFrame]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    train = daily_predictions[daily_predictions["eval_split"] == "train"].copy()
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=0.1, positive=True)),
        ]
    )
    model.fit(train[pred_cols], train["daily_target"])
    daily_predictions["stacked_daily_pred"] = model.predict(daily_predictions[pred_cols]).clip(
        min=0
    )
    return model, daily_predictions
