from dataclasses import dataclass, field
from typing import List, Optional
import os
import pandas as pd
from sklearn.cluster import KMeans
from src.ports import AssetTelemetryDTO


@dataclass
class CmapssDataIngestionAdapter:
    base_dir: Optional[str] = None
    column_names: List[str] = field(default_factory=lambda: [
        "unit_number", "time_in_cycles", "setting_1", "setting_2", "setting_3",
        "s_1", "s_2", "s_3", "s_4", "s_5", "s_6", "s_7", "s_8", "s_9", "s_10",
        "s_11", "s_12", "s_13", "s_14", "s_15", "s_16", "s_17", "s_18", "s_19",
        "s_20", "s_21"
    ])

    def load_and_preprocess_all(self, dataset_names: List[str] = None) -> pd.DataFrame:
        if dataset_names is None:
            dataset_names = ["FD001", "FD002", "FD003", "FD004"]

        all_dfs = []

        for ds_name in dataset_names:
            file_path = os.path.join(self.base_dir, f"train_{ds_name}")
            if not os.path.exists(file_path):
                print(f"⚠️ Uyarı: {file_path} bulunamadı, atlanıyor.")
                continue

            df = pd.read_csv(file_path, sep=r"\s+", header=None, names=self.column_names)
            df["dataset"] = ds_name

            # 1. RUL Hesabı
            max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
            df["RUL"] = max_cycles - df["time_in_cycles"]
            df["failure_in_window"] = (df["RUL"] <= 30).astype(int)

            # 2. Termodinamik Metrik Türetimi
            df["SFC"] = df["s_16"] / (df["s_11"] + 1e-6)
            df["EGT_Margin"] = 650.0 - df["s_4"]
            df["TPR"] = df["s_8"] / (df["s_2"] + 1e-6)

            all_dfs.append(df)

        if not all_dfs:
            raise ValueError("Hiçbir veri seti yüklenemedi!")

        combined_df = pd.concat(all_dfs, ignore_index=True)

        # 3. Tüm Veri Üzerinde Rejim Bazlı Normalizasyon (Operating Condition Normalization)
        setting_cols = ["setting_1", "setting_2", "setting_3"]
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        combined_df["op_cluster"] = kmeans.fit_predict(combined_df[setting_cols])

        sensor_cols = ["SFC", "EGT_Margin", "TPR", "s_2", "s_3", "s_4", "s_11", "s_12"]
        for col in sensor_cols:
            group_mean = combined_df.groupby("op_cluster")[col].transform("mean")
            group_std = combined_df.groupby("op_cluster")[col].transform("std").replace(0, 1e-6)
            combined_df[col] = (combined_df[col] - group_mean) / group_std

        return combined_df

    def extract_single_asset_dto(self, df: pd.DataFrame, dataset_name: str, unit_id: int, cycle: int) -> AssetTelemetryDTO:
        asset_rows = df[(df["dataset"] == dataset_name) & (df["unit_number"] == unit_id) & (df["time_in_cycles"] == cycle)]
        if asset_rows.empty:
            raise ValueError(f"Set: {dataset_name}, Motor ID: {unit_id}, Döngü: {cycle} bulunamadı!")

        row = asset_rows.iloc[0]

        features = {
            "SFC": float(row["SFC"]),
            "EGT_Margin": float(row["EGT_Margin"]),
            "TPR": float(row["TPR"]),
            "s_2": float(row["s_2"]),
            "s_3": float(row["s_3"]),
            "s_4": float(row["s_4"]),
            "s_11": float(row["s_11"]),
            "s_12": float(row["s_12"])
        }

        return AssetTelemetryDTO(
            asset_id=f"CMAPSS_{dataset_name}_UNIT_{unit_id}",
            timestamp=float(row["time_in_cycles"]),
            features=features,
            raw_payload={
                "true_rul": float(row["RUL"]),
                "SFC_raw": float(row["SFC"]),
                "EGT_Margin_raw": float(row["EGT_Margin"])
            }
        )