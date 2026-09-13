from pathlib import Path

import pandas as pd

from inference import predict_asap


BASE_DIR = Path(__file__).resolve().parent


def map_to_nearest_village(
    predictions: pd.DataFrame,
    villages: pd.DataFrame,
) -> pd.DataFrame:

    required_village_cols = [
        "nama",
        "jenis",
        "lat",
        "lon",
        "provinsi",
    ]

    missing = [
        c for c in required_village_cols
        if c not in villages.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom desa_kota.csv tidak lengkap: {missing}"
        )

    rows = []

    for _, village in villages.iterrows():

        distance_sq = (
            (predictions["lat"] - village["lat"]) ** 2
            + (predictions["lon"] - village["lon"]) ** 2
        )

        nearest_idx = distance_sq.idxmin()

        nearest = predictions.loc[nearest_idx]

        rows.append({
            "nama": village["nama"],
            "jenis": village["jenis"],
            "lat": village["lat"],
            "lon": village["lon"],
            "provinsi": village["provinsi"],
            "pm2_5_h1": nearest["pm2_5_h1"],
            "pm2_5_h2": nearest["pm2_5_h2"],
            "pm2_5_h3": nearest["pm2_5_h3"],
        })

    return pd.DataFrame(rows)


def main():

    village_path = BASE_DIR / "desa_kota.csv"
    feature_path = BASE_DIR / "features_asap.csv"

    if not village_path.exists():
        raise FileNotFoundError(
            f"Tidak ditemukan: {village_path}"
        )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Tidak ditemukan: {feature_path}"
        )

    villages = pd.read_csv(village_path)
    features = pd.read_csv(feature_path)

    # Prediksi H1/H2/H3 menggunakan inference.py
    predictions = predict_asap(features)

    required_prediction_cols = [
        "lat",
        "lon",
        "pm2_5_h1",
        "pm2_5_h2",
        "pm2_5_h3",
    ]

    missing = [
        c for c in required_prediction_cols
        if c not in predictions.columns
    ]

    if missing:
        raise ValueError(
            f"Output inference.py kurang kolom: {missing}"
        )

    result = map_to_nearest_village(
        predictions,
        villages,
    )

    output_path = BASE_DIR / "prediksi_pm25_desa.csv"

    result.to_csv(
        output_path,
        index=False,
    )

    print("\n=== HASIL ===")
    print(result.head(10).to_string(index=False))

    print(
        f"\nTotal desa/kota: {len(result):,}"
    )

    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()
