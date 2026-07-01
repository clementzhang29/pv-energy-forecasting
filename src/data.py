from __future__ import annotations

from pathlib import Path

import pandas as pd


def list_csv_files(data_dir: Path) -> list[Path]:
    files: list[Path] = []
    for split in ("train", "test1", "test2"):
        split_dir = data_dir / split
        if split_dir.exists():
            files.extend(sorted(split_dir.glob("*/*.csv")))
    return files


def read_one_csv(path: Path, data_dir: Path) -> pd.DataFrame:
    split = path.relative_to(data_dir).parts[0]
    device = path.parent.name
    date = path.stem
    frame = pd.read_csv(path)
    frame["split"] = split
    frame["deviceSn"] = frame.get("deviceSn", device)
    frame["date"] = date
    return frame


def load_point_data(data_dir: str | Path) -> pd.DataFrame:
    data_path = Path(data_dir).expanduser().resolve()
    files = list_csv_files(data_path)
    if not files:
        raise FileNotFoundError(f"No csv files found under {data_path}")
    frames = [read_one_csv(path, data_path) for path in files]
    data = pd.concat(frames, ignore_index=True)
    data["us_timestamp"] = pd.to_datetime(data["us_timestamp"], errors="coerce")
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date.astype(str)
    return data


def describe_dataset(point_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for split, part in point_df.groupby("split", observed=True):
        daily = part.groupby(["deviceSn", "date"], observed=True)["pvGenTotal"].sum()
        summary[str(split)] = {
            "devices": int(part["deviceSn"].nunique()),
            "files": int(part.groupby(["deviceSn", "date"], observed=True).ngroups),
            "rows": int(len(part)),
            "date_min": str(part["date"].min()),
            "date_max": str(part["date"].max()),
            "daily_sum_mean": float(daily.mean()),
            "daily_sum_median": float(daily.median()),
            "daily_sum_min": float(daily.min()),
            "daily_sum_max": float(daily.max()),
            "nonzero_row_ratio": float((part["pvGenTotal"] > 0).mean()),
        }
    return summary
