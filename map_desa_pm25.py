from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray:
    """Great-circle distance in kilometers."""
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(np.asarray(lat2, dtype=float))
    lon2_r = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(a))


def find_nearest_grid(
    desa_lat: float,
    desa_lon: float,
    grid_points: pd.DataFrame,
) -> dict[str, float]:
    if grid_points.empty:
        raise ValueError("Grid CAMS kosong.")

    required = {"lat", "lon"}
    missing = required - set(grid_points.columns)
    if missing:
        raise ValueError(f"Kolom grid tidak lengkap: {sorted(missing)}")

    unique_grid = grid_points[["lat", "lon"]].drop_duplicates().reset_index(drop=True)

    distances = haversine_distance(
        float(desa_lat),
        float(desa_lon),
        unique_grid["lat"].to_numpy(),
        unique_grid["lon"].to_numpy(),
    )
    idx = int(np.argmin(distances))
    row = unique_grid.iloc[idx]

    return {
        "grid_lat": float(row["lat"]),
        "grid_lon": float(row["lon"]),
        "distance_km": float(distances[idx]),
    }
