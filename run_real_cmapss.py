import os
import random
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, accuracy_score

from src.core_engine import PureHexagonalMaintenanceEngine
from src.modules.cmapss_engine import CmapssDataIngestionAdapter
from src.ports import LLMTranslatorPort, MaintenanceReportPayload, MaintenanceModelPort, AssetTelemetryDTO


# Sadece FD004 için Custom Asymmetric Loss Function
# XGBoost sample_weight kullandığımız için imzaya (y_true, y_pred, sample_weight=None) ekliyoruz!
def asymmetric_mse_objective(y_true, y_pred, sample_weight=None):
    grad = y_pred - y_true
    hess = np.ones_like(y_true)

    # İyimser tahminlerin (Geç uyarı: Tahmin > Gerçek) türev cezasını 5 katına çıkar
    late_mask = y_pred > y_true
    grad[late_mask] = grad[late_mask] * 5.0
    hess[late_mask] = 5.0

    if sample_weight is not None:
        grad *= sample_weight
        hess *= sample_weight

    return grad, hess


class EnsembleRegressorAdapter(MaintenanceModelPort):
    def __init__(self, is_fd004: bool = False):
        self.xgb_model = None
        self.hgb_model = None
        self.classifier = None
        self.is_fd004 = is_fd004

    def train_models(self, X_train: pd.DataFrame, y_rul: pd.Series, y_fail: pd.Series, sample_weights: pd.Series):
        if self.is_fd004:
            # FD004 ÖZEL ASİMETRİK XGBOOST REGRESSOR
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=450,
                max_depth=5,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8,
                objective=asymmetric_mse_objective,
                random_state=42
            )
        else:
            # DİĞER SETLER İÇİN STANDART XGBOOST REGRESSOR
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=350,
                max_depth=6,
                learning_rate=0.025,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )

        self.xgb_model.fit(X_train, y_rul, sample_weight=sample_weights)

        # HistGradientBoosting
        self.hgb_model = HistGradientBoostingRegressor(
            max_iter=350 if self.is_fd004 else 300,
            max_depth=6,
            learning_rate=0.025 if self.is_fd004 else 0.03,
            random_state=42
        )
        self.hgb_model.fit(X_train, y_rul, sample_weight=sample_weights)

        # XGBoost Classifier
        self.classifier = xgb.XGBClassifier(
            n_estimators=200 if self.is_fd004 else 150,
            max_depth=5,
            learning_rate=0.03 if self.is_fd004 else 0.05,
            random_state=42
        )
        self.classifier.fit(X_train, y_fail)

    def predict_rul(self, features: dict) -> float:
        df_feat = pd.DataFrame([features])
        pred_xgb = self.xgb_model.predict(df_feat)[0]
        pred_hgb = self.hgb_model.predict(df_feat)[0]

        # FD004 için XGBoost'un asimetrik tahminine daha fazla ağırlık veriyoruz (%70 XGB / %30 HGB)
        if self.is_fd004:
            ensemble_pred = 0.7 * pred_xgb + 0.3 * pred_hgb
        else:
            ensemble_pred = 0.5 * pred_xgb + 0.5 * pred_hgb

        return max(0.0, float(ensemble_pred))

    def predict_failure_risk(self, features: dict) -> float:
        df_feat = pd.DataFrame([features])
        prob = self.classifier.predict_proba(df_feat)[0][1]
        return float(prob)


class SimpleLLMAdapter(LLMTranslatorPort):
    def translate_to_report(self, payload: MaintenanceReportPayload) -> MaintenanceReportPayload:
        payload.generated_report_tr = f"[{payload.consensus_status.value}] Tahmini RUL: {payload.predicted_rul}"
        payload.generated_report_en = f"[{payload.consensus_status.name}] Estimated RUL: {payload.predicted_rul}"
        return payload


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(BASE_DIR, "data", "raw_cmapss")

    print("🚀 C-MAPSS CANLI SİMÜLASYON VE AKADEMİK METRİK TESTİ (FD004 ASİMETRİK KALKANLI)\n")

    ingestion_adapter = CmapssDataIngestionAdapter(base_dir=raw_data_dir, window_size=30)
    llm_adapter = SimpleLLMAdapter()

    datasets = ["FD001", "FD002", "FD003", "FD004"]

    overall_y_true_rul = []
    overall_y_pred_rul = []
    overall_y_true_class = []
    overall_y_pred_class = []

    for ds in datasets:
        is_fd004 = (ds == "FD004")

        # 1. Model Eğitimi
        train_df = ingestion_adapter.load_and_preprocess_single_set(ds)

        exclude_cols = ["unit_number", "time_in_cycles", "setting_1", "setting_2", "setting_3", "dataset", "RUL",
                        "failure_in_window", "sample_weight", "end_rul", "op_cluster"]
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]

        X_train = train_df[feature_cols]
        y_rul_train = train_df["RUL"]
        y_fail_train = train_df["failure_in_window"]
        weights = train_df["sample_weight"]

        # FD004 için özel asimetrik modlu adapter örneklenecek
        model_adapter = EnsembleRegressorAdapter(is_fd004=is_fd004)
        model_adapter.train_models(X_train, y_rul_train, y_fail_train, sample_weights=weights)

        engine = PureHexagonalMaintenanceEngine(
            model_port=model_adapter,
            llm_port=llm_adapter
        )

        # 2. Test Verisi
        test_df = ingestion_adapter.load_and_preprocess_test_single_set(ds)

        # ----------------------------------------------------
        # BÖLÜM A: RASTGELE SEÇİLEN MOTORUN SİMÜLASYONU
        # ----------------------------------------------------
        available_units = test_df["unit_number"].unique()
        selected_unit = random.choice(available_units)

        unit_df = test_df[test_df["unit_number"] == selected_unit].copy()
        max_cycle = unit_df["time_in_cycles"].max()

        print(f"\n✈️  DATASET: {ds} | SİMÜLASYON MOTORU: Unit #{selected_unit} (Toplam {max_cycle} Uçuş)")
        print(
            f"{'Cycle':<8} | {'Gerçek RUL':<10} | {'Tahmin RUL':<10} | {'Risk (%)':<10} | {'Consensus Durumu':<24} | {'Işık'}")
        print("-" * 85)

        for idx, row in unit_df.iterrows():
            c = int(row["time_in_cycles"])
            features = {col: float(row[col]) for col in feature_cols if col in row}

            pred_rul = model_adapter.predict_rul(features)
            risk_pct = model_adapter.predict_failure_risk(features) * 100

            dto = AssetTelemetryDTO(
                asset_id=f"CMAPSS_{ds}_UNIT_{selected_unit}",
                timestamp=float(c),
                features=features,
                raw_payload={"true_rul": float(row["true_rul"]) if pd.notna(row["true_rul"]) else 0.0}
            )
            report = engine.process_asset(dto)

            is_critical_period = (max_cycle - c) <= 5
            is_periodic = (c % 10 == 0 or c == 1)

            if is_periodic or is_critical_period:
                val = row["true_rul"]
                true_rul_str = f"{val:<10.0f}" if pd.notna(val) else "N/A       "
                print(
                    f"{c:<8} | "
                    f"{true_rul_str} | "
                    f"{pred_rul:<10.1f} | "
                    f"%{risk_pct:<9.1f} | "
                    f"{report.consensus_status.name:<24} | "
                    f"{report.traffic_light}"
                )

        print("-" * 85)

        # ----------------------------------------------------
        # BÖLÜM B: TÜM TEST MOTORLARININ GENEL METRİKLERİ
        # ----------------------------------------------------
        last_cycles_df = test_df.groupby("unit_number").last().reset_index()

        y_true_ds = []
        y_pred_ds = []
        y_true_fail_ds = []
        y_pred_fail_ds = []

        for idx, row in last_cycles_df.iterrows():
            if pd.isna(row["true_rul"]):
                continue

            features = {col: float(row[col]) for col in feature_cols if col in row}
            p_rul = model_adapter.predict_rul(features)
            p_risk = model_adapter.predict_failure_risk(features)

            t_rul = float(row["true_rul"])
            y_true_ds.append(t_rul)
            y_pred_ds.append(p_rul)

            y_true_fail_ds.append(1 if t_rul <= 30 else 0)
            y_pred_fail_ds.append(1 if p_risk >= 0.5 else 0)

        rmse_ds = root_mean_squared_error(y_true_ds, y_pred_ds)
        acc_ds = accuracy_score(y_true_fail_ds, y_pred_fail_ds) * 100

        overall_y_true_rul.extend(y_true_ds)
        overall_y_pred_rul.extend(y_pred_ds)
        overall_y_true_class.extend(y_true_fail_ds)
        overall_y_pred_class.extend(y_pred_fail_ds)

        print(f"📊 {ds} GENEL TEST SONUÇLARI (Toplam {len(y_true_ds)} Test Motoru):")
        print(f"   • RUL Tahmin Hata Payı (RMSE) : {rmse_ds:.2f} Uçuş")
        print(f"   • Kritik Arıza Yakalama (Accuracy): %{acc_ds:.1f}")
        print("=" * 85)

    # ----------------------------------------------------
    # BÖLÜM C: NİHAİ GENEL ÖZET
    # ----------------------------------------------------
    total_rmse = root_mean_squared_error(overall_y_true_rul, overall_y_pred_rul)
    total_acc = accuracy_score(overall_y_true_class, overall_y_pred_class) * 100

    print("\n🎯 TÜM SETLER GENEL HİBRİT ÖZETİ:")
    print(f"   • Genel RUL RMSE Skor : {total_rmse:.2f} Uçuş")
    print(f"   • Genel Arıza Accuracy: %{total_acc:.1f}")
    print("=" * 85)

    print("\n✅ SİMÜLASYON VE AKADEMİK METRİK TESTİ BAŞARIYLA TAMAMLANDI!")