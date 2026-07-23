from dataclasses import dataclass, field
from typing import List, Optional
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from src.ports import AssetTelemetryDTO


@dataclass
class CmapssDataIngestionAdapter:
    base_dir: Optional[str] = None
    window_size: int = 30
    column_names: List[str] = field(default_factory=lambda: [
        "unit_number", "time_in_cycles", "setting_1", "setting_2", "setting_3",
        "s_1", "s_2", "s_3", "s_4", "s_5", "s_6", "s_7", "s_8", "s_9", "s_10",
        "s_11", "s_12", "s_13", "s_14", "s_15", "s_16", "s_17", "s_18", "s_19",
        "s_20", "s_21"
    ])
    kmeans_model: Optional[KMeans] = None
    cluster_means: dict = field(default_factory=dict)
    cluster_stds: dict = field(default_factory=dict)

    def _engineer_features(self, df: pd.DataFrame, window_sz: int) -> pd.DataFrame:
        df["SFC"] = df["s_16"] / (df["s_11"] + 1e-6)
        df["EGT_Margin"] = 650.0 - df["s_4"]
        df["TPR"] = df["s_8"] / (df["s_2"] + 1e-6)

        base_sensors = ["SFC", "EGT_Margin", "TPR", "s_2", "s_3", "s_4", "s_11", "s_12", "s_15", "s_20", "s_21"]
        group_col = ["dataset", "unit_number"] if "dataset" in df else "unit_number"

        for sensor in base_sensors:
            grp = df.groupby(group_col)[sensor]
            df[f"{sensor}_mean_{window_sz}"] = grp.transform(lambda x: x.rolling(window_sz, min_periods=1).mean())
            df[f"{sensor}_std_{window_sz}"] = grp.transform(lambda x: x.rolling(window_sz, min_periods=1).std()).fillna(
                0)
            df[f"{sensor}_diff_1"] = grp.diff().fillna(0)
            df[f"{sensor}_diff_5"] = grp.diff(5).fillna(0)
            df[f"{sensor}_trend"] = df[sensor] - df[f"{sensor}_mean_{window_sz}"]

        return df

    def load_and_preprocess_single_set(self, dataset_name: str) -> pd.DataFrame:
        file_path = os.path.join(self.base_dir, f"train_{dataset_name}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} bulunamadı!")

        df = pd.read_csv(file_path, sep=r"\s+", header=None, names=self.column_names)
        df["dataset"] = dataset_name

        max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
        raw_rul = max_cycles - df["time_in_cycles"]

        # Sadece FD004 için clipping limitini ve kayan pencereyi özelleştiriyoruz
        clip_limit = 115 if dataset_name == "FD004" else 125
        w_size = 45 if dataset_name == "FD004" else self.window_size

        df["RUL"] = raw_rul.clip(upper=clip_limit)
        df["failure_in_window"] = (df["RUL"] <= 30).astype(int)

        # FD004 için kritik son uçuşlara verilen ağırlığı arttırıyoruz
        weight_factor = 5.0 if dataset_name == "FD004" else 3.0
        df["sample_weight"] = np.where(df["RUL"] <= 30, weight_factor, 1.0)

        df = self._engineer_features(df, window_sz=w_size)

        if dataset_name in ["FD002", "FD004"]:
            setting_cols = ["setting_1", "setting_2", "setting_3"]
            self.kmeans_model = KMeans(n_clusters=6, random_state=42, n_init=10)
            df["op_cluster"] = self.kmeans_model.fit_predict(df[setting_cols])

            feature_cols_to_norm = [c for c in df.columns if
                                    c not in ["unit_number", "time_in_cycles", "setting_1", "setting_2", "setting_3",
                                              "dataset", "RUL", "failure_in_window", "sample_weight", "op_cluster"]]

            self.cluster_means = {}
            self.cluster_stds = {}
            for col in feature_cols_to_norm:
                self.cluster_means[col] = df.groupby("op_cluster")[col].mean().to_dict()
                self.cluster_stds[col] = df.groupby("op_cluster")[col].std().replace(0, 1e-6).to_dict()

                group_mean = df["op_cluster"].map(self.cluster_means[col])
                group_std = df["op_cluster"].map(self.cluster_stds[col])
                df[col] = (df[col] - group_mean) / group_std

        return df

    def load_and_preprocess_test_single_set(self, dataset_name: str) -> pd.DataFrame:
        test_file = os.path.join(self.base_dir, f"test_{dataset_name}")
        rul_file = os.path.join(self.base_dir, f"RUL_{dataset_name}")

        if not os.path.exists(test_file) or not os.path.exists(rul_file):
            raise FileNotFoundError(f"{dataset_name} test veya RUL dosyası bulunamadı!")

        test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=self.column_names)
        test_df["dataset"] = dataset_name

        w_size = 45 if dataset_name == "FD004" else self.window_size
        clip_limit = 115 if dataset_name == "FD004" else 125

        test_df = self._engineer_features(test_df, window_sz=w_size)

        if dataset_name in ["FD002", "FD004"] and self.kmeans_model is not None:
            setting_cols = ["setting_1", "setting_2", "setting_3"]
            test_df["op_cluster"] = self.kmeans_model.predict(test_df[setting_cols])

            feature_cols_to_norm = [c for c in test_df.columns if
                                    c not in ["unit_number", "time_in_cycles", "setting_1", "setting_2", "setting_3",
                                              "dataset", "op_cluster"]]
            for col in feature_cols_to_norm:
                if col in self.cluster_means and col in self.cluster_stds:
                    group_mean = test_df["op_cluster"].map(self.cluster_means[col])
                    group_std = test_df["op_cluster"].map(self.cluster_stds[col])
                    test_df[col] = (test_df[col] - group_mean) / group_std

        rul_df = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["true_rul"])
        rul_df["true_rul"] = rul_df["true_rul"].clip(upper=clip_limit)

        unique_units = np.sort(test_df["unit_number"].unique())
        rul_map = {}
        for idx, u_id in enumerate(unique_units):
            if idx < len(rul_df):
                rul_map[u_id] = float(rul_df.iloc[idx]["true_rul"])
            else:
                rul_map[u_id] = 0.0

        test_df["end_rul"] = test_df["unit_number"].map(rul_map)
        max_cycles = test_df.groupby("unit_number")["time_in_cycles"].transform("max")
        test_df["true_rul"] = (max_cycles - test_df["time_in_cycles"] + test_df["end_rul"]).clip(upper=clip_limit)

        return test_df

    def extract_single_asset_dto(self, df: pd.DataFrame, dataset_name: str, unit_id: int,
                                 cycle: int) -> AssetTelemetryDTO:
        asset_rows = df[
            (df["dataset"] == dataset_name) & (df["unit_number"] == unit_id) & (df["time_in_cycles"] == cycle)]
        if asset_rows.empty:
            raise ValueError(f"Set: {dataset_name}, Motor ID: {unit_id}, Döngü: {cycle} bulunamadı!")

        row = asset_rows.iloc[0]
        exclude_cols = ["unit_number", "time_in_cycles", "setting_1", "setting_2", "setting_3", "dataset", "RUL",
                        "failure_in_window", "sample_weight", "true_rul", "end_rul", "op_cluster"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

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