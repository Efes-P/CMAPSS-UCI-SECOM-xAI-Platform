from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum
from abc import ABC, abstractmethod


class ConsensusStatus(Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    SCHEDULED_MAINTENANCE = "SCHEDULED_MAINTENANCE"
    CRITICAL_HALT = "CRITICAL_HALT"


@dataclass
class AssetTelemetryDTO:
    asset_id: str
    timestamp: float
    features: Dict[str, float]
    raw_payload: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class MaintenanceReportPayload:
    asset_id: str
    predicted_rul: float
    failure_probability: float
    consensus_status: ConsensusStatus
    traffic_light: str
    generated_report_tr: Optional[str] = None
    generated_report_en: Optional[str] = None


class MaintenanceModelPort(ABC):
    @abstractmethod
    def predict_rul(self, features: Dict[str, float]) -> float:
        pass

    @abstractmethod
    def predict_failure_risk(self, features: Dict[str, float]) -> float:
        pass


class LLMTranslatorPort(ABC):
    @abstractmethod
    def translate_to_report(self, payload: MaintenanceReportPayload) -> MaintenanceReportPayload:
        pass