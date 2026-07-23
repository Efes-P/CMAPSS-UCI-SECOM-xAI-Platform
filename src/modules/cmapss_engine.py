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
    kmeans_model: Optional[KMeans] = None

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Termodinamik Metrikler
        df["SFC"] = df["s_16"] / (df["s_11"] + 1e-6)
        df["EGT_Margin"] = 650.0 - df["s_4"]
        df["TPR"] = df["s_8"] / (df["s_2"] + 1e-6)

        base_sensors = ["SFC", "EGT_Margin", "TPR", "s_2", "s_3", "s_4", "s_11", "s_12"]

        # Kayan Pencere (Rolling Features - 10 Uçuşluk Trend)
        for sensor in base_sensors:
            df[f"{sensor}_mean_10"] = df.groupby(["dataset", "unit_number"] if "dataset" in df else "unit_number")[sensor].transform(lambda x: x.rolling(10, min_periods=1).mean())
            df[f"{sensor}_std_10"] = df.groupby(["dataset", "unit_number"] if "dataset" in df else "unit_number")[sensor].transform(lambda x: x.rolling(10, min_periods=1).std()).fillna(0)

        return df

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

            # 1. Parçalı (Piecewise) RUL Hesabı (Literatür Standardı Max: 125)
            max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
            raw_rul = max_cycles - df["time_in_cycles"]
            df["RUL"] = raw_rul.clip(upper=125)
            df["failure_in_window"] = (df["RUL"] <= 30).astype(int)

            df = self._engineer_features(df)
            all_dfs.append(df)

        if not all_dfs:
            raise ValueError("Hiçbir train veri seti yüklenemedi!")

        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Rejim Normalizasyonu
        setting_cols = ["setting_1", "setting_2", "setting_3"]
        self.kmeans_model = KMeans(n_clusters=6, random_state=42, n_init=10)
        combined_df["op_cluster"] = self.kmeans_model.fit_predict(combined_df[setting_cols])

        feature_cols_to_norm = ["SFC", "EGT_Margin", "TPR", "s_2", "s_3", "s_4", "s_11", "s_12",
                                "SFC_mean_10", "EGT_Margin_mean_10", "TPR_mean_10", "s_2_mean_10",
                                "s_3_mean_10", "s_4_mean_10", "s_11_mean_10", "s_12_mean_10"]

        for col in feature_cols_to_norm:
            group_mean = combined_df.groupby("op_cluster")[col].transform("mean")
            group_std = combined_df.groupby("op_cluster")[col].transform("std").replace(0, 1e-6)
            combined_df[col] = (combined_df[col] - group_mean) / group_std

        return combined_df

    def load_and_preprocess_test(self, dataset_names: List[str] = None) -> pd.DataFrame:
        if dataset_names is None:
            dataset_names = ["FD001", "FD002", "FD003", "FD004"]

        all_test_dfs = []

        for ds_name in dataset_names:
            test_file = os.path.join(self.base_dir, f"test_{ds_name}")
            rul_file = os.path.join(self.base_dir, f"RUL_{ds_name}")

            if not os.path.exists(test_file) or not os.path.exists(rul_file):
                print(f"⚠️ Uyarı: {ds_name} test veya RUL dosyası bulunamadı, atlanıyor.")
                continue

            test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=self.column_names)
            test_df["dataset"] = ds_name
            test_df = self._engineer_features(test_df)

            rul_df = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["true_rul"])
            rul_df["true_rul"] = rul_df["true_rul"].clip(upper=125)
            rul_df["unit_number"] = rul_df.index + 1
            rul_df["dataset"] = ds_name

            last_cycles = test_df.groupby("unit_number").last().reset_index()
            merged = pd.merge(last_cycles, rul_df, on=["dataset", "unit_number"])
            all_test_dfs.append(merged)

        if not all_test_dfs:
            raise ValueError("Hiçbir test veri seti yüklenemedi!")

        combined_test = pd.concat(all_test_dfs, ignore_index=True)

        if self.kmeans_model is not None:
            setting_cols = ["setting_1", "setting_2", "setting_3"]
            combined_test["op_cluster"] = self.kmeans_model.predict(combined_test[setting_cols])

            feature_cols_to_norm = ["SFC", "EGT_Margin", "TPR", "s_2", "s_3", "s_4", "s_11", "s_12",
                                    "SFC_mean_10", "EGT_Margin_mean_10", "TPR_mean_10", "s_2_mean_10",
                                    "s_3_mean_10", "s_4_mean_10", "s_11_mean_10", "s_12_mean_10"]

            for col in feature_cols_to_norm:
                group_mean = combined_test.groupby("op_cluster")[col].transform("mean")
                group_std = combined_test.groupby("op_cluster")[col].transform("std").replace(0, 1e-6)
                combined_test[col] = (combined_test[col] - group_mean) / group_std

        return combined_test

    def extract_single_asset_dto(self, df: pd.DataFrame, dataset_name: str, unit_id: int, cycle: int) -> AssetTelemetryDTO:
        asset_rows = df[(df["dataset"] == dataset_name) & (df["unit_number"] == unit_id) & (df["time_in_cycles"] == cycle)]
        if asset_rows.empty:
            raise ValueError(f"Set: {dataset_name}, Motor ID: {unit_id}, Döngü: {cycle} bulunamadı!")

        row = asset_rows.iloc[0]

        feature_cols = [
            "SFC", "EGT_Margin", "TPR", "s_2", "s_3", "s_4", "s_11", "s_12",
            "SFC_mean_10", "EGT_Margin_mean_10", "TPR_mean_10", "s_2_mean_10",
            "s_3_mean_10", "s_4_mean_10", "s_11_mean_10", "s_12_mean_10",
            "SFC_std_10", "EGT_Margin_std_10", "TPR_std_10"
        ]

        features = {col: float(row[col]) for col in feature_cols if col in row}
        true_rul_val = float(row["true_rul"]) if "true_rul" in row else float(row["RUL"])

        return AssetTelemetryDTO(
            asset_id=f"CMAPSS_{dataset_name}_UNIT_{unit_id}",
            timestamp=float(row["time_in_cycles"]),
            features=features,
            raw_payload={
                "true_rul": true_rul_val,
                "SFC_raw": float(row["SFC"]),
                "EGT_Margin_raw": float(row["EGT_Margin"])
            }
        )