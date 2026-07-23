import os
import pandas as pd
import numpy as np

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

    print("🔥 TÜM C-MAPSS SETLERİ (FD001, FD002, FD003, FD004) BİRLEŞTİRİLİYOR VE EĞİTİLİYOR\n")

    ingestion_adapter = CmapssDataIngestionAdapter(base_dir=raw_data_dir)
    model_adapter = CmapssModelAdapter()
    llm_adapter = SimpleLLMAdapter()

    engine = PureHexagonalMaintenanceEngine(
        model_port=model_adapter,
        llm_port=llm_adapter
    )

    datasets = ["FD001", "FD002", "FD003", "FD004"]
    print("📊 Tüm veri setleri okunuyor, birleştiriliyor ve Rejim Bazında normalize ediliyor...")
    full_df = ingestion_adapter.load_and_preprocess_all(dataset_names=datasets)

    print(f"📈 Toplam İşlenen Satır Sayısı: {len(full_df):,}")

    feature_cols = ["SFC", "EGT_Margin", "TPR", "s_2", "s_3", "s_4", "s_11", "s_12"]
    X_train = full_df[feature_cols]
    y_rul = full_df["RUL"]
    y_fail = full_df["failure_in_window"]

    print("🤖 XGBoost Modelleri Hibrit Genel Veri Kümesi Üzerinde Eğitiliyor...")
    model_adapter.train_mock_models(X_train, y_rul, y_fail)
    print("✅ Genel Model Eğitimi Tamamlandı!\n")

    # Her setten Örnek Motor #1'i Test Edelim
    for ds in datasets:
        unit_df = full_df[(full_df["dataset"] == ds) & (full_df["unit_number"] == 1)]
        if unit_df.empty:
            continue

        total_cycles = unit_df["time_in_cycles"].max()
        print(f"\n✈️  {ds} - Motor Unit #1 (Toplam {total_cycles} Döngü):")
        print(f"{'Cycle':<8} | {'Gerçek RUL':<10} | {'Tahmin RUL':<10} | {'Risk (%)':<10} | {'Consensus Durumu':<25} | {'Trafik Işığı'}")
        print("-" * 90)

        sample_cycles = [1, 50, 100, 150, total_cycles - 15, total_cycles]
        for c in sample_cycles:
            if c in unit_df["time_in_cycles"].values:
                input_dto = ingestion_adapter.extract_single_asset_dto(full_df, dataset_name=ds, unit_id=1, cycle=c)
                report = engine.process_asset(input_dto)

                true_rul = input_dto.raw_payload["true_rul"]
                risk_pct = report.failure_probability * 100

                print(
                    f"{c:<8} | "
                    f"{true_rul:<10.0f} | "
                    f"{report.predicted_rul:<10.1f} | "
                    f"%{risk_pct:<9.1f} | "
                    f"{report.consensus_status.name:<25} | "
                    f"{report.traffic_light}"
                )

    print("\n✅ TÜM SETLER ÜZERİNDE HİBRİT TEST TAMAMLANDI!")