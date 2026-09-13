from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from map_desa_pm25 import find_nearest_grid

APP_ROOT = Path(__file__).resolve().parent
RAMALAN_PATH = APP_ROOT / "ramalan_asap.parquet"
DESA_PATH = APP_ROOT / "desa_kota.csv"

RAMALAN_REQUIRED = {
    "lat",
    "lon",
    "tanggal_data",
    "tanggal_ramal",
    "horizon_hari",
    "pm25_duga",
}
DESA_REQUIRED = {"nama", "jenis", "lat", "lon", "provinsi"}


def _validate_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} kehilangan kolom: {sorted(missing)}")


@lru_cache(maxsize=1)
def load_ramalan(path: str | None = None) -> pd.DataFrame:
    file_path = Path(path) if path else RAMALAN_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    df = pd.read_parquet(file_path)
    _validate_columns(df, RAMALAN_REQUIRED, "ramalan_asap.parquet")

    df = df.copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="raise")
    df["lon"] = pd.to_numeric(df["lon"], errors="raise")
    df["horizon_hari"] = pd.to_numeric(df["horizon_hari"], errors="raise").astype(int)
    df["pm25_duga"] = pd.to_numeric(df["pm25_duga"], errors="coerce")
    df["tanggal_data"] = pd.to_datetime(df["tanggal_data"], errors="coerce")
    df["tanggal_ramal"] = pd.to_datetime(df["tanggal_ramal"], errors="coerce")

    return df


@lru_cache(maxsize=1)
def load_desa(path: str | None = None) -> pd.DataFrame:
    file_path = Path(path) if path else DESA_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    df = pd.read_csv(file_path)
    _validate_columns(df, DESA_REQUIRED, "desa_kota.csv")

    df = df.copy()
    df["nama"] = df["nama"].astype("string").fillna("")
    df["jenis"] = df["jenis"].astype("string").fillna("")
    df["provinsi"] = df["provinsi"].astype("string").fillna("")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return df


def search_location(
    query: str,
    desa_df: pd.DataFrame,
    max_results: int = 20,
) -> pd.DataFrame:
    query = str(query or "").strip().lower()
    if not query:
        return desa_df.head(0).copy()

    name = desa_df["nama"].str.lower()
    province = desa_df["provinsi"].str.lower()
    jenis = desa_df["jenis"].str.lower()

    exact = name.eq(query)
    starts = name.str.startswith(query, na=False)
    contains = name.str.contains(query, na=False, regex=False)
    broader = (
        name.str.contains(query, na=False, regex=False)
        | province.str.contains(query, na=False, regex=False)
        | jenis.str.contains(query, na=False, regex=False)
    )

    scored = desa_df.loc[broader].copy()
    if scored.empty:
        return scored

    scored["_score"] = np.select(
        [exact.loc[scored.index], starts.loc[scored.index], contains.loc[scored.index]],
        [3, 2, 1],
        default=0,
    )
    scored = scored.sort_values(
        by=["_score", "nama"],
        ascending=[False, True],
        kind="stable",
    ).drop(columns="_score")

    return scored.head(max_results)


def _latest_batch(grid_data: pd.DataFrame) -> pd.DataFrame:
    valid = grid_data.dropna(subset=["tanggal_data"])
    if valid.empty:
        return grid_data.head(0).copy()

    latest_date = valid["tanggal_data"].max()
    batch = valid.loc[valid["tanggal_data"].eq(latest_date)].copy()
    return batch.sort_values("horizon_hari")


def get_forecast(
    desa_row: pd.Series,
    ramalan_df: pd.DataFrame,
) -> dict:
    unique_grids = ramalan_df[["lat", "lon"]].drop_duplicates()
    nearest = find_nearest_grid(
        float(desa_row["lat"]),
        float(desa_row["lon"]),
        unique_grids,
    )

    # Avoid fragile exact floating point comparisons by comparing against the
    # nearest grid coordinates with a tiny tolerance.
    lat_mask = np.isclose(ramalan_df["lat"].to_numpy(), nearest["grid_lat"])
    lon_mask = np.isclose(ramalan_df["lon"].to_numpy(), nearest["grid_lon"])
    grid_data = ramalan_df.loc[lat_mask & lon_mask].copy()

    batch = _latest_batch(grid_data)
    if batch.empty:
        return {
            "found": False,
            "distance_km": nearest["distance_km"],
        }

    horizon_rows = []
    for horizon in (1, 2, 3):
        rows = batch.loc[batch["horizon_hari"].eq(horizon)]
        if rows.empty:
            continue

        row = rows.iloc[-1]
        horizon_rows.append(
            {
                "horizon_hari": int(horizon),
                "tanggal_ramal": row["tanggal_ramal"],
                "pm25_duga": (
                    float(row["pm25_duga"]) if pd.notna(row["pm25_duga"]) else None
                ),
                "ispu": row["ispu"] if "ispu" in row.index else None,
                "kategori_ispu": (
                    row["kategori_ispu"] if "kategori_ispu" in row.index else None
                ),
                "siaga_tidak_sehat": bool(
                    row["siaga_tidak_sehat"]
                    if "siaga_tidak_sehat" in row.index
                    and pd.notna(row["siaga_tidak_sehat"])
                    else False
                ),
                "siaga_bahaya": bool(
                    row["siaga_bahaya"]
                    if "siaga_bahaya" in row.index and pd.notna(row["siaga_bahaya"])
                    else False
                ),
                "catatan_wilayah": (
                    row["catatan_wilayah"] if "catatan_wilayah" in row.index else None
                ),
            }
        )

    if not horizon_rows:
        return {
            "found": False,
            "distance_km": nearest["distance_km"],
        }

    # pm25_hari_ini is expected to be identical within a batch for the same
    # grid; take the first non-null value available.
    baseline = (
        batch["pm25_hari_ini"]
        if "pm25_hari_ini" in batch.columns
        else pd.Series(dtype=float)
    )
    baseline_value = None
    if not baseline.empty:
        non_null = baseline.dropna()
        if not non_null.empty:
            baseline_value = float(non_null.iloc[0])

    return {
        "found": True,
        "grid_lat": nearest["grid_lat"],
        "grid_lon": nearest["grid_lon"],
        "distance_km": nearest["distance_km"],
        "tanggal_data": batch["tanggal_data"].iloc[0],
        "pm25_hari_ini": baseline_value,
        "horizons": horizon_rows,
    }
