from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.ports import AssetTelemetryDTO


@dataclass
class CmapssDataIngestionAdapter:
    base_dir: Optional[str] = None
    window_size: int = 30
    n_regimes: int = 6
    column_names: List[str] = field(default_factory=lambda: [
        "unit_number", "time_in_cycles", "setting_1", "setting_2", "setting_3",
        "s_1", "s_2", "s_3", "s_4", "s_5", "s_6", "s_7", "s_8", "s_9", "s_10",
        "s_11", "s_12", "s_13", "s_14", "s_15", "s_16", "s_17", "s_18", "s_19",
        "s_20", "s_21"
    ])
    # {regime_id: {sensor_name: (baseline_mean, baseline_std)}}
    regressors: dict = field(default_factory=dict)
    active_sensors: list = field(default_factory=list)
    scaler: Optional[StandardScaler] = None
    regime_scaler: Optional[StandardScaler] = None
    regime_model: Optional[KMeans] = None

    def _assign_regimes(self, df: pd.DataFrame, is_train: bool) -> np.ndarray:
        setting_cols = ["setting_1", "setting_2", "setting_3"]
        if is_train:
            self.regime_scaler = StandardScaler()
            settings_scaled = self.regime_scaler.fit_transform(df[setting_cols])
            self.regime_model = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            regime_labels = self.regime_model.fit_predict(settings_scaled)
        else:
            settings_scaled = self.regime_scaler.transform(df[setting_cols])
            regime_labels = self.regime_model.predict(settings_scaled)
        return regime_labels

    def _apply_baseline_residual_normalization(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        sensor_cols = [c for c in self.column_names if c.startswith("s_")]
        df = df.copy()
        df["regime"] = self._assign_regimes(df, is_train)

        if is_train:
            # 🔒 active_sensors SADECE train'de belirlenir, test'te yeniden hesaplanmaz (Bug #2 fix)
            self.active_sensors = [s for s in sensor_cols if df[s].std() > 1e-4]
            self.regressors = {}
            healthy_df = df[df["RUL"] >= 110] if "RUL" in df else df
            if healthy_df.empty:
                healthy_df = df

            for regime_id in range(self.n_regimes):
                regime_healthy = healthy_df[healthy_df["regime"] == regime_id]
                if regime_healthy.empty:
                    regime_healthy = healthy_df
                self.regressors[regime_id] = {}
                for s in self.active_sensors:
                    mean_val = regime_healthy[s].mean()
                    std_val = regime_healthy[s].std()
                    if not np.isfinite(std_val) or std_val < 1e-6:
                        std_val = 1.0
                    self.regressors[regime_id][s] = (float(mean_val), float(std_val))

        residual_dict = {}
        for s in self.active_sensors:
            expected = np.zeros(len(df))
            scale = np.ones(len(df))
            for regime_id in range(self.n_regimes):
                mask = (df["regime"] == regime_id).values
                if not mask.any():
                    continue
                stats = self.regressors.get(regime_id, {}).get(s)
                if stats is None:
                    all_means = [v[s][0] for v in self.regressors.values() if s in v]
                    all_stds = [v[s][1] for v in self.regressors.values() if s in v]
                    mean_val = float(np.mean(all_means)) if all_means else 0.0
                    std_val = float(np.mean(all_stds)) if all_stds else 1.0
                else:
                    mean_val, std_val = stats
                expected[mask] = mean_val
                scale[mask] = std_val
            residual_dict[f"{s}_residual"] = (df[s].values - expected) / scale

        residual_df = pd.DataFrame(residual_dict, index=df.index)
        return pd.concat([df, residual_df], axis=1)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        residual_cols = [c for c in df.columns if c.endswith("_residual")]
        group_col = ["dataset", "unit_number"] if "dataset" in df else "unit_number"

        new_features = {}
        for col in residual_cols:
            grp = df.groupby(group_col)[col]
            new_features[f"{col}_mean_{self.window_size}"] = grp.transform(lambda x: x.rolling(self.window_size, min_periods=1).mean())
            new_features[f"{col}_std_{self.window_size}"] = grp.transform(lambda x: x.rolling(self.window_size, min_periods=1).std()).fillna(0)
            new_features[f"{col}_diff_1"] = grp.diff().fillna(0)

        new_feats_df = pd.DataFrame(new_features, index=df.index)
        return pd.concat([df, new_feats_df], axis=1)

    def load_and_preprocess_single_set(self, dataset_name: str) -> pd.DataFrame:
        file_path = os.path.join(self.base_dir, f"train_{dataset_name}")
        if not os.path.exists(file_path):
            file_path = f"{file_path}.txt"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{dataset_name} train dosyası bulunamadı!")

        df = pd.read_csv(file_path, sep=r"\s+", header=None, names=self.column_names)
        df["dataset"] = dataset_name

        max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
        raw_rul = max_cycles - df["time_in_cycles"]
        df["RUL"] = raw_rul.clip(upper=125)
        df["failure_in_window"] = (df["RUL"] <= 30).astype(int)

        df = self._apply_baseline_residual_normalization(df, is_train=True)
        df = self._engineer_features(df)

        exclude_cols = ["unit_number", "time_in_cycles", "setting_1", "setting_2", "setting_3",
                        "dataset", "RUL", "failure_in_window", "regime"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        self.scaler = StandardScaler()
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])

        return df

    def load_and_preprocess_test_single_set(self, dataset_name: str) -> pd.DataFrame:
        test_file = os.path.join(self.base_dir, f"test_{dataset_name}")
        if not os.path.exists(test_file):
            test_file = f"{test_file}.txt"

        rul_file = os.path.join(self.base_dir, f"RUL_{dataset_name}")
        if not os.path.exists(rul_file):
            rul_file = f"{rul_file}.txt"

        if not os.path.exists(test_file) or not os.path.exists(rul_file):
            raise FileNotFoundError(f"{dataset_name} test veya RUL dosyası bulunamadı!")

        test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=self.column_names)
        test_df["dataset"] = dataset_name

        test_df = self._apply_baseline_residual_normalization(test_df, is_train=False)
        test_df = self._engineer_features(test_df)

        rul_raw = pd.read_csv(rul_file, sep=r"\s+", header=None)
        valid_rul_values = rul_raw.dropna(how='all').iloc[:, 0].astype(float).values

        unique_units = np.sort(test_df["unit_number"].unique())
        rul_map = {u_id: float(valid_rul_values[idx]) for idx, u_id in enumerate(unique_units) if idx < len(valid_rul_values)}

        test_df["end_rul"] = test_df["unit_number"].map(rul_map)
        max_cycles = test_df.groupby("unit_number")["time_in_cycles"].transform("max")
        test_df["true_rul"] = (max_cycles - test_df["time_in_cycles"] + test_df["end_rul"]).clip(upper=125)

        exclude_cols = ["unit_number", "time_in_cycles", "setting_1", "setting_2", "setting_3", "dataset",
                        "RUL", "failure_in_window", "end_rul", "true_rul", "regime"]
        feature_cols = [c for c in test_df.columns if c not in exclude_cols]

        if self.scaler is not None:
            test_df[feature_cols] = self.scaler.transform(test_df[feature_cols])

        return test_df

    def create_lstm_sequences(self, df: pd.DataFrame, feature_cols: List[str],
                               sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_seq, y_rul_seq, y_fail_seq, groups_seq = [], [], [], []

        for unit_id, group in df.groupby("unit_number"):
            data = group[feature_cols].values
            rul = group["RUL"].values if "RUL" in group else group["true_rul"].values
            fail = group["failure_in_window"].values if "failure_in_window" in group else (rul <= 30).astype(int)

            num_rows = len(data)
            for i in range(num_rows):
                if i < sequence_length - 1:
                    pad_len = sequence_length - (i + 1)
                    pad = np.tile(data[0], (pad_len, 1))
                    seq = np.vstack([pad, data[:i+1]])
                else:
                    seq = data[i - sequence_length + 1 : i + 1]

                X_seq.append(seq)
                y_rul_seq.append(rul[i])
                y_fail_seq.append(fail[i])
                groups_seq.append(unit_id)  # 🔑 GroupShuffleSplit için gerekli (Bug #1 fix)

        return (np.array(X_seq, dtype=np.float32), np.array(y_rul_seq, dtype=np.float32),
                np.array(y_fail_seq, dtype=np.float32), np.array(groups_seq))