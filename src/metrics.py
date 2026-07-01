from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EPS = 1e-9


@dataclass(frozen=True)
class MetricBundle:
    mae: float
    rmse: float
    mape: float
    daily_mae: float
    daily_rmse: float
    daily_mape: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "daily_mae": self.daily_mae,
            "daily_rmse": self.daily_rmse,
            "daily_mape": self.daily_mape,
        }


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = EPS) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def daily_frame(
    frame: pd.DataFrame,
    pred_col: str,
    target_col: str = "pvGenTotal",
) -> pd.DataFrame:
    cols = ["deviceSn", "date"]
    grouped = (
        frame.groupby(cols, observed=True)[[target_col, pred_col]]
        .sum()
        .reset_index()
        .rename(columns={target_col: "y_true_day", pred_col: "y_pred_day"})
    )
    grouped["abs_error"] = (grouped["y_true_day"] - grouped["y_pred_day"]).abs()
    grouped["ape"] = grouped["abs_error"] / np.maximum(grouped["y_true_day"].abs(), EPS)
    return grouped


def evaluate_predictions(
    frame: pd.DataFrame,
    pred_col: str,
    target_col: str = "pvGenTotal",
) -> MetricBundle:
    y_true = frame[target_col].to_numpy(dtype=float)
    y_pred = frame[pred_col].to_numpy(dtype=float)
    day = daily_frame(frame, pred_col=pred_col, target_col=target_col)
    y_true_day = day["y_true_day"].to_numpy(dtype=float)
    y_pred_day = day["y_pred_day"].to_numpy(dtype=float)
    return MetricBundle(
        mae=mae(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        mape=mape(y_true, y_pred),
        daily_mae=mae(y_true_day, y_pred_day),
        daily_rmse=rmse(y_true_day, y_pred_day),
        daily_mape=mape(y_true_day, y_pred_day),
    )


def summarize_daily_by_split(frame: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for name, part in frame.groupby("eval_split", observed=True):
        metrics = evaluate_predictions(part, pred_col=pred_col).as_dict()
        metrics["eval_split"] = str(name)
        metrics["n_days"] = int(part.groupby(["deviceSn", "date"], observed=True).ngroups)
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("eval_split").reset_index(drop=True)
