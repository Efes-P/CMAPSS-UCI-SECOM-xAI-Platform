from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
from src.ports import AssetTelemetryDTO


@dataclass
class CmapssDataIngestionAdapter:
    raw_data_path: Optional[str] = None
    column_names: List[str] = field(default_factory=lambda: [
        "unit_number", "time_in_cycles", "setting_1", "setting_2", "setting_3",
        "s_1", "s_2", "s_3", "s_4", "s_5", "s_6", "s_7", "s_8", "s_9", "s_10",
        "s_11", "s_12", "s_13", "s_14", "s_15", "s_16", "s_17", "s_18", "s_19",
        "s_20", "s_21"
    ])

    def load_and_preprocess(self, raw_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if raw_data is not None:
            df = raw_data.copy()
        elif self.raw_data_path:
            df = pd.read_csv(self.raw_data_path, sep=r"\s+", header=None, names=self.column_names)
        else:
            raise ValueError("Ne DataFrame ne de dosya yolu sağlandı!")

        # 1. RUL Hesabı
        max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
        df["RUL"] = max_cycles - df["time_in_cycles"]
        df["failure_in_window"] = (df["RUL"] <= 30).astype(int)

        # 2. Termodinamik Metrik Türetimi
        df["SFC"] = df["s_16"] / (df["s_11"] + 1e-6)
        df["EGT_Margin"] = 650.0 - df["s_4"]
        df["TPR"] = df["s_8"] / (df["s_2"] + 1e-6)

        return df

    def extract_single_asset_dto(self, df: pd.DataFrame, unit_id: int, cycle: int) -> AssetTelemetryDTO:
        asset_rows = df[(df["unit_number"] == unit_id) & (df["time_in_cycles"] == cycle)]
        if asset_rows.empty:
            raise ValueError(f"Motor ID: {unit_id}, Döngü: {cycle} bulunamadı!")

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
            asset_id=f"CMAPSS_UNIT_{unit_id}",
            timestamp=float(row["time_in_cycles"]),
            features=features,
            raw_payload={
                "true_rul": float(row["RUL"]),
                "SFC_raw": float(row["SFC"]),
                "EGT_Margin_raw": float(row["EGT_Margin"])
            }
        )