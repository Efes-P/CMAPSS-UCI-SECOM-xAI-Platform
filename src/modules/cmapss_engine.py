import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from src.ports import AssetType, AssetStandardInputDTO


class CmapssDataIngestionAdapter:
    """
    Inbound Adapter: NASA C-MAPSS Ham Veri Yükleme ve Fiziksel Feature Engineering.
    Ham sensör sütunlarını işleyerek SFC, EGT Margin ve TPR metriklerini türetir.
    """

    def __init__(self, raw_data_path: str = "data/raw_cmapss/train_FD001.txt"):
        self.raw_data_path = raw_data_path
        self.column_names = (
            ["unit_number", "time_in_cycles", "setting_1", "setting_2", "setting_3"]
            + [f"s_{i}" for i in range(1, 22)]
        )

    def load_and_preprocess(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Ham CMAPSS verisini okur, RUL ve Termodinamik Metrikleri hesaplar.
        """
        if df is None:
            # Gerçek veri dosyasından okuma
            df = pd.read_csv(self.raw_data_path, sep=r"\s+", header=None, names=self.column_names)

        # 1. RUL (Remaining Useful Life) Hesabı
        max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
        df["RUL"] = max_cycles - df["time_in_cycles"]

        # 2. TERMODİNAMİK / FİZİKSEL FEATURE ENGINEERING
        # A. SFC (Specific Fuel Consumption)
        df["SFC"] = df["s_12"] * df["s_11"] / (df["s_8"] + 1e-5)

        # B. EGT Margin (Exhaust Gas Temp Margin - Limit ~660K varsayımı)
        MAX_EGT_LIMIT = 660.0
        df["EGT_Margin"] = MAX_EGT_LIMIT - df["s_4"]

        # C. TPR (Total Pressure Ratio)
        df["TPR"] = df["s_11"] / (df["s_2"] + 1e-5)

        # 3. Pencereli Sınıflandırma Etiketi (Örn: Önümüzdeki 20 uçuşta arıza var mı?)
        LABEL_WINDOW = 20
        df["failure_in_window"] = (df["RUL"] <= LABEL_WINDOW).astype(int)

        return df

    def extract_single_asset_dto(self, df: pd.DataFrame, unit_id: int, cycle: int) -> AssetStandardInputDTO:
        """
        Tek bir motor ve uçuş döngüsü için veriyi çekip Core Engine DTO formatına paketler.
        """
        asset_row = df[(df["unit_number"] == unit_id) & (df["time_in_cycles"] == cycle)]

        if asset_row.empty:
            raise ValueError(f"Motor {unit_id} için cycle {cycle} bulunamadı!")

        row = asset_row.iloc[0]

        # Core Engine'e gidecek türetilmiş metrikler
        features = {
            "SFC": float(row["SFC"]),
            "EGT_Margin": float(row["EGT_Margin"]),
            "TPR": float(row["TPR"]),
            "s_2": float(row["s_2"]),
            "s_3": float(row["s_3"]),
            "s_4": float(row["s_4"]),
            "s_11": float(row["s_11"]),
            "s_12": float(row["s_12"]),
        }

        return AssetStandardInputDTO(
            asset_id=f"JetEngine_Unit_{unit_id}_Cycle_{cycle}",
            asset_type=AssetType.TIME_SERIES_JET_ENGINE,
            processed_features=features,
            raw_payload={"true_rul": float(row["RUL"]), "true_failure": int(row["failure_in_window"])},
        )


# --- BAĞIMSIZ TEST BLOĞU (SINIFIN TAMAMEN DIŞINDA) ---
if __name__ == "__main__":
    # Test için sahte CMAPSS verisi
    mock_raw_data = pd.DataFrame({
        "unit_number": [1, 1, 1],
        "time_in_cycles": [10, 20, 30],
        "setting_1": [0.0, 0.0, 0.0], "setting_2": [0.0, 0.0, 0.0], "setting_3": [100, 100, 100],
        "s_1": [518.67]*3, "s_2": [642.0]*3, "s_3": [1580.0]*3, "s_4": [1400.0, 1420.0, 1450.0],
        "s_5": [14.62]*3, "s_6": [21.61]*3, "s_7": [553.0]*3, "s_8": [2388.0]*3,
        "s_9": [9046.0]*3, "s_10": [1.3]*3, "s_11": [47.2]*3, "s_12": [521.0]*3,
        "s_13": [2388.0]*3, "s_14": [8138.0]*3, "s_15": [8.4]*3, "s_16": [0.03]*3,
        "s_17": [392]*3, "s_18": [2388]*3, "s_19": [100]*3, "s_20": [38.8]*3, "s_21": [23.3]*3
    })

    adapter = CmapssDataIngestionAdapter()
    processed_df = adapter.load_and_preprocess(mock_raw_data)

    dto = adapter.extract_single_asset_dto(processed_df, unit_id=1, cycle=20)

    print("✅ CMAPSS ADAPTER TEST BAŞARILI!")
    print(f"Türetilen Metrikler (DTO): {dto.processed_features}")