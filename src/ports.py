from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional


# --- 1. ENUMLAR (Sistem Durumları) ---
class AssetType(Enum):
    TIME_SERIES_JET_ENGINE = "CMAPSS"
    HIGH_DIM_SEMICONDUCTOR = "SECOM"


class ConsensusStatus(Enum):
    SAFE = "GÜVENLİ / NORMAL"
    SCHEDULED_MAINTENANCE = "PLANLI BAKIM GEREKLİ"
    TRANSIENT_SENSOR_SHOCK = "ERKEN ANOMALİ / SENSÖR ŞOKU"
    ANOMALOUS_BATCH = "ISKARTA / ANOMALİ TESPİT EDİLDİ"


# --- 2. INBOUND DTO (Core Engine'e Giren Veri Kontratı) ---
@dataclass
class AssetStandardInputDTO:
    asset_id: str                          # Örn: "Engine_Unit_12" veya "Chip_Batch_88"
    asset_type: AssetType                  # C-MAPSS mi SECOM mu?
    processed_features: Dict[str, float]   # Türetilmiş temiz metrikler (SFC, EGT_Margin, TPR)
    raw_payload: Optional[Any] = None      # Ek ham veriler


# --- 3. OUTBOUND DTO (Core Engine'den Çıkan Veri Kontratı) ---
@dataclass
class MaintenanceReportPayload:
    module_name: str
    asset_id: str
    asset_type: AssetType
    predicted_rul: Optional[float]         # Kalan Ömür (SECOM için None)
    failure_probability: float            # Arıza İhtimali (0.0 - 1.0)
    anomaly_score: float                  # Reconstruction MSE skoru
    consensus_status: ConsensusStatus      # Logic Gate Çıktısı
    traffic_light: str                    # "GREEN", "YELLOW", "RED"
    top_root_causes: List[Dict[str, Any]] = field(default_factory=list)
    generated_report_tr: str = ""
    generated_report_en: str = ""


# --- 4. OUTBOUND PORTS (Dış Adaptör Arayüzleri) ---
class PredictiveModelPort(ABC):
    @abstractmethod
    def predict_cmapss(self, features: Dict[str, float]) -> Dict[str, Any]:
        """C-MAPSS için RUL ve Sınıflandırma tahminlerini döndürür."""
        pass

    @abstractmethod
    def predict_secom(self, features: Dict[str, float]) -> Dict[str, Any]:
        """SECOM için Autoencoder MSE Reconstruction skorunu döndürür."""
        pass


class LLMTranslatorPort(ABC):
    @abstractmethod
    def translate_to_report(self, payload: MaintenanceReportPayload) -> MaintenanceReportPayload:
        """Sayısal veriyi alır, LLM üzerinden TR/EN rapora dönüştürür."""
        pass
