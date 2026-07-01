from __future__ import annotations

import pandas as pd


def assign_eval_split(
    point_df: pd.DataFrame,
    holdout_days: int = 7,
    unseen_fold: int | None = None,
    n_unseen_folds: int | None = None,
    unseen_strategy: str = "default",
) -> pd.DataFrame:
    frame = point_df.copy()
    frame["eval_split"] = "train"
    train_devices = set(frame.loc[frame["split"] == "train", "deviceSn"].unique())

    train_dates = (
        frame.loc[frame["split"] == "train", ["deviceSn", "date"]]
        .drop_duplicates()
        .sort_values(["deviceSn", "date"])
    )
    seen_holdout = (
        train_dates.groupby("deviceSn", observed=True)
        .tail(holdout_days)
        .assign(eval_split="valid_seen")
    )
    key = set(zip(seen_holdout["deviceSn"], seen_holdout["date"]))
    mask_seen = frame.apply(
        lambda r: r["split"] == "train" and (r["deviceSn"], r["date"]) in key,
        axis=1,
    )
    frame.loc[mask_seen, "eval_split"] = "valid_seen"

    test1_mask = frame["split"] == "test1"
    test2_mask = frame["split"] == "test2"
    frame.loc[test1_mask, "eval_split"] = "test1"
    frame.loc[test2_mask, "eval_split"] = "test2"

    unseen_candidates = sorted(
        device for device in train_devices if device not in set(frame.loc[test1_mask, "deviceSn"])
    )
    if unseen_strategy == "low_capacity":
        daily = (
            frame.loc[
                (frame["split"] == "train") & frame["deviceSn"].isin(unseen_candidates),
                ["deviceSn", "date", "pvGenTotal"],
            ]
            .groupby(["deviceSn", "date"], observed=True)["pvGenTotal"]
            .sum()
            .reset_index(name="daily_sum")
        )
        capacity_order = (
            daily.groupby("deviceSn", observed=True)["daily_sum"].mean().sort_values()
        )
        n_unseen = max(4, min(8, len(unseen_candidates) // 4 or 1))
        unseen_devices = set(capacity_order.head(n_unseen).index)
    elif n_unseen_folds and n_unseen_folds > 1:
        fold = int(unseen_fold or 0) % int(n_unseen_folds)
        unseen_devices = {
            device
            for idx, device in enumerate(unseen_candidates)
            if idx % int(n_unseen_folds) == fold
        }
    else:
        n_unseen = max(4, min(8, len(unseen_candidates) // 4 or 1))
        unseen_devices = set(unseen_candidates[:n_unseen])
    unseen_mask = (frame["split"] == "train") & frame["deviceSn"].isin(unseen_devices)
    frame.loc[unseen_mask, "eval_split"] = "valid_unseen"
    return frame


def train_mask(frame: pd.DataFrame) -> pd.Series:
    return (frame["split"] == "train") & (frame["eval_split"] == "train")


def validation_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["eval_split"].isin(["valid_seen", "valid_unseen"])
