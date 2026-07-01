from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.metrics import daily_frame


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 160
    plt.rcParams["font.family"] = "DejaVu Sans"


def plot_daily_distribution(point_df: pd.DataFrame, output_dir: Path) -> Path:
    configure_style()
    daily = (
        point_df.groupby(["split", "deviceSn", "date"], observed=True)["pvGenTotal"]
        .sum()
        .reset_index(name="daily_sum")
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data=daily, x="daily_sum", hue="split", bins=32, kde=True, ax=ax)
    ax.set_title("Daily Energy Distribution")
    ax.set_xlabel("Daily sum of pvGenTotal")
    ax.set_ylabel("Count")
    path = output_dir / "daily_energy_distribution.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_geo_distribution(point_df: pd.DataFrame, output_dir: Path) -> Path:
    configure_style()
    devices = (
        point_df.groupby(["split", "deviceSn"], observed=True)
        .agg(latitude=("latitude", "median"), longitude=("longitude", "median"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=devices,
        x="longitude",
        y="latitude",
        hue="split",
        style="split",
        s=72,
        ax=ax,
    )
    ax.set_title("Device Locations by Split")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    path = output_dir / "device_locations.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_daily_scatter(point_df: pd.DataFrame, pred_col: str, output_dir: Path) -> Path:
    configure_style()
    daily = daily_frame(point_df, pred_col=pred_col)
    daily = daily.merge(
        point_df[["deviceSn", "date", "eval_split"]].drop_duplicates(),
        on=["deviceSn", "date"],
        how="left",
    )
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    sns.scatterplot(
        data=daily,
        x="y_true_day",
        y="y_pred_day",
        hue="eval_split",
        alpha=0.75,
        ax=ax,
    )
    lim = max(daily["y_true_day"].max(), daily["y_pred_day"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="black", linewidth=1, linestyle="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title("Daily Prediction vs Actual")
    ax.set_xlabel("Actual daily energy")
    ax.set_ylabel("Predicted daily energy")
    path = output_dir / "daily_prediction_scatter.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_split_mape(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    configure_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = metrics_df.sort_values("daily_mape")["eval_split"].tolist()
    sns.barplot(data=metrics_df, x="eval_split", y="daily_mape", order=order, ax=ax)
    ax.set_title("Daily MAPE by Evaluation Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Daily MAPE")
    ax.tick_params(axis="x", rotation=20)
    path = output_dir / "daily_mape_by_split.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_example_curve(point_df: pd.DataFrame, pred_col: str, output_dir: Path) -> Path:
    configure_style()
    candidates = (
        point_df[point_df["eval_split"].isin(["test1", "test2", "valid_seen", "valid_unseen"])]
        .groupby(["eval_split", "deviceSn", "date"], observed=True)["pvGenTotal"]
        .sum()
        .reset_index()
        .sort_values("pvGenTotal", ascending=False)
    )
    if candidates.empty:
        candidates = (
            point_df.groupby(["eval_split", "deviceSn", "date"], observed=True)["pvGenTotal"]
            .sum()
            .reset_index()
        )
    chosen = candidates.iloc[0]
    part = point_df[
        (point_df["deviceSn"] == chosen["deviceSn"]) & (point_df["date"] == chosen["date"])
    ].sort_values("us_timestamp")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(part["us_timestamp"], part["pvGenTotal"], label="Actual", linewidth=2)
    ax.plot(part["us_timestamp"], part[pred_col], label="Predicted", linewidth=2)
    ax.set_title(f"15-minute Curve Example: {chosen['eval_split']} / {chosen['date']}")
    ax.set_xlabel("Time")
    ax.set_ylabel("pvGenTotal")
    ax.legend()
    fig.autofmt_xdate()
    path = output_dir / "example_intraday_curve.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_feature_importance(model: object, features: list[str], output_dir: Path) -> Path | None:
    configure_style()
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = getattr(model, "feature_importances_")
    elif hasattr(model, "named_steps") and "model" in model.named_steps:
        inner = model.named_steps["model"]
        if hasattr(inner, "feature_importances_"):
            importance = inner.feature_importances_
    if importance is None:
        return None
    data = (
        pd.DataFrame({"feature": features, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(25)
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.barplot(data=data, y="feature", x="importance", ax=ax)
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    path = output_dir / "feature_importance.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
