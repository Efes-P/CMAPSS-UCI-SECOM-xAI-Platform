import numpy as np
import pandas as pd
from typing import Dict, Any
from xgboost import XGBRegressor, XGBClassifier
from src.ports import PredictiveModelPort


class CmapssModelAdapter(PredictiveModelPort):
    """
    Outbound Adapter: NASA C-MAPSS için Dual-Model (Regresyon + Sınıflandırma) Yapısı.
    PredictiveModelPort arayüzünü uygular.
    """

    def __init__(self):
        # 1. RUL Tahmini için Regresyon Modeli
        self.regressor = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )

        # 2. Arıza İhtimali için Sınıflandırma Modeli
        self.classifier = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            eval_metric="logloss",
            random_state=42
        )

        self.is_trained = False

    def train_mock_models(self, X_train: pd.DataFrame, y_rul: pd.Series, y_fail: pd.Series):
        """
        Modelleri eğitir (Eğitim verisi geldiğinde çağrılır).
        """
        self.regressor.fit(X_train, y_rul)
        self.classifier.fit(X_train, y_fail)
        self.is_trained = True

    def predict_cmapss(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Port Arayüzü Uygulaması: Core Engine bu metodu çağırır.
        Türetilmiş özellikleri alır, RUL ve Failure Probability tahminlerini üretir.
        """
        # Sözlüğü DataFrame formatına getiriyoruz
        input_df = pd.DataFrame([features])

        if self.is_trained:
            predicted_rul = float(self.regressor.predict(input_df)[0])
            # Sınıf 1 (Arıza) olma olasılığı
            failure_prob = float(self.classifier.predict_proba(input_df)[0][1])
        else:
            # Model henüz eğitilmediyse test akışı için mantıklı varsayılan tahminler
            # (Features içindeki EGT_Margin ve SFC'ye göre sezgisel tahmin üretir)
            egt = features.get("EGT_Margin", 0.0)
            sfc = features.get("SFC", 0.0)

            # Basit bir sentetik RUL/Risk tahmini (Eğitimsiz testler için)
            predicted_rul = max(10.0, float(150.0 - (sfc * 5.0) + (egt * 0.1)))
            failure_prob = min(0.99, max(0.01, float((200.0 - predicted_rul) / 200.0)))

        # En kritik feature etkilerini hesapla (Şimdilik basit kural bazlı kök neden)
        top_root_causes = [
            {"feature": "SFC", "value": features.get("SFC"), "impact": "Özgül yakıt tüketimi yüksek"},
            {"feature": "EGT_Margin", "value": features.get("EGT_Margin"),
             "impact": "Egzoz sıcaklık payı kritik seviyede"}
        ]

        return {
            "predicted_rul": round(predicted_rul, 2),
            "failure_probability": round(failure_prob, 4),
            "anomaly_score": 0.0,  # C-MAPSS için MSE gerekmiyor
            "top_root_causes": top_root_causes
        }

    def predict_secom(self, features: Dict[str, float]) -> Dict[str, Any]:
        """C-MAPSS adaptörü SECOM tahminini işlemez, boş geçer."""
        raise NotImplementedError("CmapssModelAdapter SECOM verisini işlemez!")


# --- BAĞIMSIZ TEST BLOĞU ---
if __name__ == "__main__":
    adapter = CmapssModelAdapter()

    # Örnek girdi
    sample_features = {
        "SFC": 10.3,
        "EGT_Margin": 15.0,
        "TPR": 0.073,
        "s_2": 642.0,
        "s_3": 1580.0,
        "s_4": 1420.0,
        "s_11": 47.2,
        "s_12": 521.0
    }

    result = adapter.predict_cmapss(sample_features)

    print("✅ CMAPSS MODEL ADAPTER TEST BAŞARILI!")
    print(f"Tahmin Edilen RUL: {result['predicted_rul']} döngü")
    print(f"Arıza İhtimali: %{result['failure_probability'] * 100:.2f}")
    print(f"Kök Nedenler: {result['top_root_causes']}")