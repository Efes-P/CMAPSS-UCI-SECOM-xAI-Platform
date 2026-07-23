from typing import Tuple
from src.ports import (
    AssetType,
    ConsensusStatus,
    AssetStandardInputDTO,
    MaintenanceReportPayload,
    PredictiveModelPort,
    LLMTranslatorPort,
)


class PureHexagonalMaintenanceEngine:
    """
    Core Domain Engine (Hexagon Çekirdeği).
    ML kütüphanelerinden veya LLM API'lerinden izoledir.
    Sadece Portlar (Interfaces) üzerinden haberleşir ve İş Mantığını (Logic Gate) yürütür.
    """

    def __init__(self, model_port: PredictiveModelPort, llm_port: LLMTranslatorPort):
        """Dependency Injection (Bağımlılık Enjeksiyonu)"""
        self.model_port = model_port
        self.llm_port = llm_port

    def process_asset(self, input_dto: AssetStandardInputDTO) -> MaintenanceReportPayload:
        """Gelen DTO paketini türüne göre yönlendirir, Logic Gate'ten geçirir ve LLM Portuna iletir."""
        if input_dto.asset_type == AssetType.TIME_SERIES_JET_ENGINE:
            payload = self._process_cmapss(input_dto)
        elif input_dto.asset_type == AssetType.HIGH_DIM_SEMICONDUCTOR:
            payload = self._process_secom(input_dto)
        else:
            raise ValueError(f"Desteklenmeyen Varlık Tipi: {input_dto.asset_type}")

        # Her iki veriseti için ortak çıkış portu (LLM Raporlama)
        return self.llm_port.translate_to_report(payload)

    def _process_cmapss(self, input_dto: AssetStandardInputDTO) -> MaintenanceReportPayload:
        """C-MAPSS Veriseti İşleme Akışı ve Dual-Model Consensus"""
        predictions = self.model_port.predict_cmapss(input_dto.processed_features)

        rul = predictions.get("predicted_rul", 0.0)
        fail_prob = predictions.get("failure_probability", 0.0)

        # ⚡ LOGIC GATE: DUAL-MODEL CONSENSUS KONTROLÜ
        consensus_status, traffic_light = self._evaluate_cmapss_consensus(rul, fail_prob)

        return MaintenanceReportPayload(
            module_name="C-MAPSS Engine",
            asset_id=input_dto.asset_id,
            asset_type=input_dto.asset_type,
            predicted_rul=rul,
            failure_probability=fail_prob,
            anomaly_score=predictions.get("anomaly_score", 0.0),
            consensus_status=consensus_status,
            traffic_light=traffic_light,
            top_root_causes=predictions.get("top_root_causes", []),
        )

    def _process_secom(self, input_dto: AssetStandardInputDTO) -> MaintenanceReportPayload:
        """SECOM Veriseti İşleme Akışı (Autoencoder MSE Loss)"""
        predictions = self.model_port.predict_secom(input_dto.processed_features)

        mse_score = predictions.get("reconstruction_mse", 0.0)
        is_anomaly = predictions.get("is_anomaly", False)

        consensus_status = (
            ConsensusStatus.ANOMALOUS_BATCH
            if is_anomaly
            else ConsensusStatus.SAFE
        )
        traffic_light = "RED" if is_anomaly else "GREEN"

        return MaintenanceReportPayload(
            module_name="SECOM Engine",
            asset_id=input_dto.asset_id,
            asset_type=input_dto.asset_type,
            predicted_rul=None,  # SECOM'da RUL yok
            failure_probability=predictions.get("failure_probability", 0.0),
            anomaly_score=mse_score,
            consensus_status=consensus_status,
            traffic_light=traffic_light,
            top_root_causes=predictions.get("top_root_causes", []),
        )

    def _evaluate_cmapss_consensus(
        self, rul: float, fail_prob: float
    ) -> Tuple[ConsensusStatus, str]:
        """
        ⚡ CONSENSUS LOGIC GATE
        RUL Regresyonu ile Arıza Sınıflandırması arasındaki çelişkileri çözer.
        """
        RUL_SAFE_THRESHOLD = 30.0    # 30 cycle üstü mekanik olarak sağlıklı
        RISK_HIGH_THRESHOLD = 0.70   # %70 üstü risk alarm demektir

        if rul > RUL_SAFE_THRESHOLD and fail_prob < RISK_HIGH_THRESHOLD:
            # Durum A: Normal Operasyon
            return ConsensusStatus.SAFE, "GREEN"

        elif rul <= RUL_SAFE_THRESHOLD and fail_prob >= RISK_HIGH_THRESHOLD:
            # Durum B: Planlı Bakım Zamanı
            return ConsensusStatus.SCHEDULED_MAINTENANCE, "RED"

        elif rul > RUL_SAFE_THRESHOLD and fail_prob >= RISK_HIGH_THRESHOLD:
            # Durum C: ÇAKIŞMA! (Erken Anomali / Sensör Şoku)
            return ConsensusStatus.TRANSIENT_SENSOR_SHOCK, "YELLOW"

        else:
            return ConsensusStatus.SCHEDULED_MAINTENANCE, "YELLOW"