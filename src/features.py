from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EPS = 1e-6


@dataclass(frozen=True)
class FeatureSets:
    point_features: list[str]
    daily_features: list[str]
    categorical_features: list[str]


def _to_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(frame["us_timestamp"], errors="coerce")
    frame["hour"] = ts.dt.hour.astype("float32")
    frame["minute"] = ts.dt.minute.astype("float32")
    frame["slot_id"] = (frame["hour"] * 4 + (frame["minute"] // 15)).astype("float32")
    frame["day_of_year"] = ts.dt.dayofyear.astype("float32")
    frame["month"] = ts.dt.month.astype("float32")
    frame["weekday"] = ts.dt.weekday.astype("float32")
    frame["is_weekend"] = (frame["weekday"] >= 5).astype("float32")
    frame["sin_hour"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["cos_hour"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["sin_slot"] = np.sin(2 * np.pi * frame["slot_id"] / 96)
    frame["cos_slot"] = np.cos(2 * np.pi * frame["slot_id"] / 96)
    frame["sin_doy"] = np.sin(2 * np.pi * frame["day_of_year"] / 365)
    frame["cos_doy"] = np.cos(2 * np.pi * frame["day_of_year"] / 365)
    return frame


def add_physics_features(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "altitude",
        "zenith_angle",
        "azimuth",
        "is_day",
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
        "direct_normal_irradiance",
        "global_tilted_irradiance",
        "terrestrial_radiation",
        "shortwave_radiation_instant",
        "direct_radiation_instant",
        "diffuse_radiation_instant",
        "direct_normal_irradiance_instant",
        "global_tilted_irradiance_instant",
        "terrestrial_radiation_instant",
        "cloud_cover_H",
    ]
    _to_numeric(frame, [c for c in numeric_cols if c in frame.columns])

    altitude = frame.get("altitude", 0).fillna(0)
    zenith = frame.get("zenith_angle", 90).fillna(90)
    azimuth = frame.get("azimuth", 0).fillna(0)
    frame["altitude_pos"] = np.maximum(altitude, 0)
    frame["elevation_weight"] = np.maximum(np.sin(np.deg2rad(altitude)), 0)
    frame["zenith_cos"] = np.maximum(np.cos(np.deg2rad(zenith)), 0)
    frame["azimuth_sin"] = np.sin(np.deg2rad(azimuth))
    frame["azimuth_cos"] = np.cos(np.deg2rad(azimuth))

    radiation_candidates = [
        "global_tilted_irradiance_instant",
        "global_tilted_irradiance",
        "shortwave_radiation_instant",
        "shortwave_radiation",
    ]
    available = [c for c in radiation_candidates if c in frame.columns]
    if available:
        radiation = frame[available].max(axis=1).fillna(0)
    else:
        radiation = pd.Series(0.0, index=frame.index)
    frame["radiation_proxy"] = radiation.clip(lower=0)
    frame["solar_proxy"] = frame["radiation_proxy"] * frame["elevation_weight"]
    if "cloud_cover_H" in frame.columns:
        cloud = frame["cloud_cover_H"].fillna(frame["cloud_cover_H"].median())
        frame["radiation_cloud_adjusted"] = frame["radiation_proxy"] * (1 - cloud.clip(0, 100) / 100)
    else:
        frame["radiation_cloud_adjusted"] = frame["radiation_proxy"]
    frame["radiation_x_elevation"] = frame["radiation_proxy"] * frame["elevation_weight"]
    frame["is_solar_available"] = (
        (frame["solar_proxy"] > 1e-4) & (frame.get("is_day", 1).fillna(1) > 0)
    ).astype("float32")
    return frame



def add_radiation_profile_features(frame: pd.DataFrame) -> pd.DataFrame:
    """为每个设备添加辐射曲线相似度特征帮助冷启动容量估计。
    
    对设备的每日 15 分钟辐射曲线做归一化后求均值，得到该设备的典型辐射 profile。
    这个 profile 独立于发电量，只依赖天气和位置，可帮助未见设备估计容量尺度。
    """
    if "solar_proxy" not in frame.columns or "deviceSn" not in frame.columns:
        return frame
    if frame["solar_proxy"].nunique() <= 1:
        return frame
    # 每个设备的典型归一化辐射曲线
    device_profile = (
        frame.groupby(["deviceSn", "slot_id"], observed=True)["solar_proxy"]
        .mean()
        .groupby("deviceSn", observed=True)
        .apply(lambda x: (x / (x.max() + 1e-9)).to_list())
        .reset_index(name="radiation_profile")
    )
    # 为每个设备计算与其他所有设备的 profile 余弦相似度均值
    from sklearn.metrics.pairwise import cosine_similarity
    profiles = np.array(device_profile["radiation_profile"].to_list())
    sim = cosine_similarity(profiles)
    # mask 自己
    np.fill_diagonal(sim, 0)
    n = sim.shape[1]
    mean_sim = sim.sum(axis=1) / np.maximum(n - 1, 1)
    device_profile["radiation_similarity_mean"] = mean_sim
    frame = frame.merge(
        device_profile[["deviceSn", "radiation_similarity_mean"]],
        on="deviceSn",
        how="left"
    )
    frame["radiation_similarity_mean"] = frame["radiation_similarity_mean"].fillna(0)
    return frame


def add_weather_cluster_features(frame: pd.DataFrame) -> pd.DataFrame:
    """基于日级天气条件（辐照总量、温度、湿度）做简单聚类，
    为冷启动设备提供天气类型先验。"""
    required = ["solar_proxy", "deviceSn", "date", "split"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        return frame
    # 构建日级天气概要
    day_weather = (
        frame.groupby(["deviceSn", "date"], observed=True)
        .agg(
            solar_mean=("solar_proxy", "mean"),
            solar_max=("solar_proxy", "max"),
        )
        .reset_index()
    )
    if "temperature_2m" in frame.columns:
        temp = frame.groupby(["deviceSn", "date"], observed=True)["temperature_2m"].mean().reset_index()
        day_weather = day_weather.merge(temp, on=["deviceSn", "date"], how="left")
    # 按天气条件分桶：多云/晴/阴
    day_weather["weather_bin"] = 0  # default
    if "solar_max" in day_weather.columns:
        day_weather.loc[day_weather["solar_max"] > 700, "weather_bin"] = 2  # 晴
        day_weather.loc[
            (day_weather["solar_max"] > 300) & (day_weather["solar_max"] <= 700), "weather_bin"
        ] = 1  # 多云
    # 将天气桶编码合并回原始 frame
    weather_map = dict(zip(
        zip(day_weather["deviceSn"], day_weather["date"]),
        day_weather["weather_bin"]
    ))
    frame["weather_bin"] = frame.apply(
        lambda r: weather_map.get((r["deviceSn"], r["date"]), 0),
        axis=1
    ).astype("int32")
    return frame

def add_spatial_features(frame: pd.DataFrame) -> pd.DataFrame:
    _to_numeric(frame, [c for c in ["latitude", "longitude"] if c in frame.columns])
    frame["lat_round_1"] = frame["latitude"].round(1)
    frame["lon_round_1"] = frame["longitude"].round(1)
    frame["region_lat_5"] = np.floor(frame["latitude"] / 5) * 5
    frame["region_lon_5"] = np.floor(frame["longitude"] / 5) * 5
    frame["region_lat_10"] = np.floor(frame["latitude"] / 10) * 10
    frame["region_lon_10"] = np.floor(frame["longitude"] / 10) * 10
    frame["lat_lon_bin"] = (
        frame["lat_round_1"].astype(str) + "_" + frame["lon_round_1"].astype(str)
    )
    device_family = frame["deviceSn"].astype(str).str.extract(r"EP(\d{7})", expand=False)
    frame["device_family_code"] = pd.to_numeric(device_family, errors="coerce").fillna(-1)
    return frame


def _history_stats(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = (
        train_df.groupby(["deviceSn", "date"], observed=True)["pvGenTotal"]
        .sum()
        .reset_index(name="daily_sum")
    )
    device_stats = (
        daily.groupby("deviceSn", observed=True)["daily_sum"]
        .agg(
            device_daily_mean="mean",
            device_daily_median="median",
            device_daily_p90=lambda x: x.quantile(0.9),
            device_daily_max="max",
            device_history_days="size",
        )
        .reset_index()
    )
    slot_stats = (
        train_df.groupby(["deviceSn", "slot_id"], observed=True)["pvGenTotal"]
        .agg(device_slot_mean="mean", device_slot_p90=lambda x: x.quantile(0.9))
        .reset_index()
    )
    global_slot = (
        train_df.groupby("slot_id", observed=True)["pvGenTotal"]
        .agg(global_slot_mean="mean", global_slot_p90=lambda x: x.quantile(0.9))
        .reset_index()
    )
    return device_stats, slot_stats, global_slot


def _nearest_neighbor_features(train_df: pd.DataFrame, all_devices: pd.DataFrame) -> pd.DataFrame:
    daily = (
        train_df.groupby(["deviceSn", "date"], observed=True)["pvGenTotal"]
        .sum()
        .reset_index(name="daily_sum")
    )
    train_devices = (
        train_df.groupby("deviceSn", observed=True)
        .agg(
            train_lat=("latitude", "median"),
            train_lon=("longitude", "median"),
        )
        .reset_index()
        .merge(
            daily.groupby("deviceSn", observed=True)["daily_sum"]
            .agg(train_daily_mean="mean", train_daily_p90=lambda x: x.quantile(0.9))
            .reset_index(),
            on="deviceSn",
            how="left",
        )
    )
    rows: list[dict[str, float | str]] = []
    for row in all_devices.itertuples(index=False):
        lat = float(row.latitude)
        lon = float(row.longitude)
        distances = haversine_km(
            lat,
            lon,
            train_devices["train_lat"].to_numpy(dtype=float),
            train_devices["train_lon"].to_numpy(dtype=float),
        )
        candidates = train_devices.copy()
        candidates["distance"] = distances
        candidates = candidates[candidates["deviceSn"] != row.deviceSn].sort_values("distance")
        if candidates.empty:
            candidates = train_devices.assign(distance=distances).sort_values("distance")
        top3 = candidates.head(3)
        top5 = candidates.head(5)
        rows.append(
            {
                "deviceSn": row.deviceSn,
                "nn_1_distance_km": float(top3["distance"].iloc[0]),
                "nn_3_daily_mean": float(top3["train_daily_mean"].mean()),
                "nn_3_daily_p90": float(top3["train_daily_p90"].mean()),
                "nn_5_daily_mean": float(top5["train_daily_mean"].mean()),
                "nn_5_daily_p90": float(top5["train_daily_p90"].mean()),
            }
        )
    return pd.DataFrame(rows)


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius = 6371.0
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def add_history_and_spatial_stats(frame: pd.DataFrame) -> pd.DataFrame:
    if "eval_split" in frame.columns:
        train_df = frame[(frame["split"] == "train") & (frame["eval_split"] == "train")].copy()
    else:
        train_df = frame[frame["split"] == "train"].copy()
    device_stats, slot_stats, global_slot = _history_stats(train_df)
    all_devices = (
        frame.groupby("deviceSn", observed=True)
        .agg(latitude=("latitude", "median"), longitude=("longitude", "median"))
        .reset_index()
    )
    nn_stats = _nearest_neighbor_features(train_df, all_devices)

    frame = frame.merge(device_stats, on="deviceSn", how="left")
    frame = frame.merge(slot_stats, on=["deviceSn", "slot_id"], how="left")
    frame = frame.merge(global_slot, on="slot_id", how="left")
    frame = frame.merge(nn_stats, on="deviceSn", how="left")
    for col in [
        "device_daily_mean",
        "device_daily_median",
        "device_daily_p90",
        "device_daily_max",
        "device_history_days",
        "device_slot_mean",
        "device_slot_p90",
    ]:
        if col in frame.columns:
            fallback = frame.get("nn_3_daily_mean", pd.Series(np.nan, index=frame.index))
            if "slot" in col:
                fallback = frame.get("global_slot_mean", fallback)
            if col == "device_history_days":
                fallback = pd.Series(0.0, index=frame.index)
            frame[col] = frame[col].fillna(fallback).fillna(frame[col].median())
    return frame


def build_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, FeatureSets]:
    frame = frame.copy()
    frame = add_time_features(frame)
    frame = add_physics_features(frame)
    frame = add_spatial_features(frame)
    frame = add_history_and_spatial_stats(frame)
    frame["y_norm"] = frame["pvGenTotal"] / (frame["solar_proxy"] + EPS)
    cap = frame.loc[frame["split"] == "train", "y_norm"].quantile(0.995)
    frame["y_norm_clipped"] = frame["y_norm"].clip(lower=0, upper=cap)

    excluded = {
        "pvGenTotal",
        "y_norm",
        "y_norm_clipped",
        "us_timestamp",
        "date_H",
        "date",
        "split",
        "eval_split",
    }
    feature_cols: list[str] = []
    categorical_cols = ["deviceSn", "lat_lon_bin"]
    for col in frame.columns:
        if col in excluded:
            continue
        if col in categorical_cols:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            feature_cols.append(col)

    daily = make_daily_frame(frame)
    daily_features = [
        c
        for c in daily.columns
        if c
        not in {
            "deviceSn",
            "date",
            "split",
            "daily_target",
            "eval_split",
        }
        and pd.api.types.is_numeric_dtype(daily[c])
    ]
    return frame, FeatureSets(
        point_features=feature_cols,
        daily_features=daily_features,
        categorical_features=categorical_cols,
    )


def make_daily_frame(point_df: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "daily_target": ("pvGenTotal", "sum"),
        "rows_in_day": ("pvGenTotal", "size"),
        "solar_proxy_sum": ("solar_proxy", "sum"),
        "solar_proxy_max": ("solar_proxy", "max"),
        "radiation_proxy_sum": ("radiation_proxy", "sum"),
        "radiation_proxy_max": ("radiation_proxy", "max"),
        "elevation_weight_sum": ("elevation_weight", "sum"),
        "elevation_weight_max": ("elevation_weight", "max"),
        "is_solar_available_mean": ("is_solar_available", "mean"),
        "latitude": ("latitude", "median"),
        "longitude": ("longitude", "median"),
        "region_lat_5": ("region_lat_5", "median"),
        "region_lon_5": ("region_lon_5", "median"),
        "region_lat_10": ("region_lat_10", "median"),
        "region_lon_10": ("region_lon_10", "median"),
        "device_family_code": ("device_family_code", "median"),
        "device_daily_mean": ("device_daily_mean", "median"),
        "device_daily_p90": ("device_daily_p90", "median"),
        "device_history_days": ("device_history_days", "median"),
        "nn_3_daily_mean": ("nn_3_daily_mean", "median"),
        "nn_3_daily_p90": ("nn_3_daily_p90", "median"),
        "nn_5_daily_mean": ("nn_5_daily_mean", "median"),
        "nn_1_distance_km": ("nn_1_distance_km", "median"),
    }
    optional = [
        "temperature_2m",
        "relative_humidity_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "wind_speed_10m",
        "wind_speed_80m",
        "visibility",
        "cloud_cover_H",
        "cloud_cover_low_H",
        "cloud_cover_mid_H",
        "cloud_cover_high_H",
        "shortwave_radiation_H",
        "global_tilted_irradiance_H",
        "uv_index_H",
    ]
    for col in optional:
        if col in point_df.columns:
            aggregations[f"{col}_mean"] = (col, "mean")
            aggregations[f"{col}_max"] = (col, "max")
            if col in {"precipitation", "rain", "shortwave_radiation_H", "global_tilted_irradiance_H"}:
                aggregations[f"{col}_sum"] = (col, "sum")

    daily = (
        point_df.groupby(["split", "deviceSn", "date"], observed=True)
        .agg(**aggregations)
        .reset_index()
    )
    if "eval_split" in point_df.columns:
        eval_split = (
            point_df.groupby(["split", "deviceSn", "date"], observed=True)["eval_split"]
            .first()
            .reset_index()
        )
        daily = daily.merge(eval_split, on=["split", "deviceSn", "date"], how="left")
    daily["daily_target_log1p"] = np.log1p(daily["daily_target"])
    daily["target_per_solar"] = daily["daily_target"] / (daily["solar_proxy_sum"] + EPS)
    return daily


