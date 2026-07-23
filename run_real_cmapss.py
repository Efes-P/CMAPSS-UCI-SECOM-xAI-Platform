import os
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error

from src.core_engine import PureHexagonalMaintenanceEngine
from src.modules.cmapss_engine import CmapssDataIngestionAdapter
from src.modules.cmapss_model_adapter import CmapssModelAdapter
from src.ports import LLMTranslatorPort, MaintenanceReportPayload


class SimpleLLMAdapter(LLMTranslatorPort):
    def translate_to_report(self, payload: MaintenanceReportPayload) -> MaintenanceReportPayload:
        payload.generated_report_tr = f"[{payload.consensus_status.value}] Tahmini RUL: {payload.predicted_rul}"
        payload.generated_report_en = f"[{payload.consensus_status.name}] Estimated RUL: {payload.predicted_rul}"
        return payload


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(BASE_DIR, "data", "raw_cmapss")

    print("🔥 C-MAPSS TEST SETLERİ (GÜÇLENDİRİLMİŞ FEATURE ENGINE & RUL CLIPPING)\n")

    ingestion_adapter = CmapssDataIngestionAdapter(base_dir=raw_data_dir)
    model_adapter = CmapssModelAdapter()
    llm_adapter = SimpleLLMAdapter()

    engine = PureHexagonalMaintenanceEngine(
        model_port=model_adapter,
        llm_port=llm_adapter
    )

    datasets = ["FD001", "FD002", "FD003", "FD004"]
    print("📊 1. Train Veri Setleri Okunuyor ve Model Eğitiliyor...")
    train_df = ingestion_adapter.load_and_preprocess_all(dataset_names=datasets)

    feature_cols = [
        "SFC", "EGT_Margin", "TPR", "s_2", "s_3", "s_4", "s_11", "s_12",
        "SFC_mean_10", "EGT_Margin_mean_10", "TPR_mean_10", "s_2_mean_10",
        "s_3_mean_10", "s_4_mean_10", "s_11_mean_10", "s_12_mean_10",
        "SFC_std_10", "EGT_Margin_std_10", "TPR_std_10"
    ]

    X_train = train_df[feature_cols]
    y_rul_train = train_df["RUL"]
    y_fail_train = train_df["failure_in_window"]

    model_adapter.train_mock_models(X_train, y_rul_train, y_fail_train)
    print("✅ Model Eğitimi Tamamlandı!\n")

    print("🧪 2. Görmediği Test Setleri (test_FD00x & RUL_FD00x) Yükleniyor...")
    test_df = ingestion_adapter.load_and_preprocess_test(dataset_names=datasets)

    print(f"📈 Toplam Değerlendirilecek Test Motor Sayısı: {len(test_df)}\n")

    print(f"{'Dataset':<10} | {'Test Motor Sayısı':<20} | {'RMSE (Hata Payı - Döngü)':<25}")
    print("-" * 60)

    overall_y_true = []
    overall_y_pred = []

    for ds in datasets:
        ds_test_df = test_df[test_df["dataset"] == ds]
        if ds_test_df.empty:
            continue

        y_true_ds = ds_test_df["true_rul"].values
        y_pred_ds = []

        for idx, row in ds_test_df.iterrows():
            features = {col: float(row[col]) for col in feature_cols if col in row}
            pred_rul = model_adapter.predict_rul(features)
            y_pred_ds.append(pred_rul)

        rmse_ds = root_mean_squared_error(y_true_ds, y_pred_ds)

        overall_y_true.extend(y_true_ds)
        overall_y_pred.extend(y_pred_ds)

        print(f"{ds:<10} | {len(ds_test_df):<20} | {rmse_ds:<25.2f}")

    print("-" * 60)
    total_rmse = root_mean_squared_error(overall_y_true, overall_y_pred)
    print(f"🎯 TÜM SETLER GENEL HİBRİT RMSE SKORU: {total_rmse:.2f} Döngü\n")

    print("✅ İYİLEŞTİRİLMİŞ AKADEMİK TEST TAMAMLANDI!")