from dataclasses import dataclass
from src.ports import (
    MaintenanceModelPort,
    LLMTranslatorPort,
    AssetTelemetryDTO,
    MaintenanceReportPayload,
    ConsensusStatus
)


@dataclass
class PureHexagonalMaintenanceEngine:
    model_port: MaintenanceModelPort
    llm_port: LLMTranslatorPort

    def process_asset(self, telemetry: AssetTelemetryDTO) -> MaintenanceReportPayload:
        predicted_rul = self.model_port.predict_rul(telemetry.features)
        failure_prob = self.model_port.predict_failure_risk(telemetry.features)

        # Logic Gate / Threshold Kararı
        if failure_prob >= 0.70 or predicted_rul <= 20:
            consensus = ConsensusStatus.SCHEDULED_MAINTENANCE
            traffic_light = "RED"
        elif failure_prob >= 0.30 or predicted_rul <= 50:
            consensus = ConsensusStatus.WARNING
            traffic_light = "YELLOW"
        else:
            consensus = ConsensusStatus.SAFE
            traffic_light = "GREEN"

        base_payload = MaintenanceReportPayload(
            asset_id=telemetry.asset_id,
            predicted_rul=predicted_rul,
            failure_probability=failure_prob,
            consensus_status=consensus,
            traffic_light=traffic_light
        )

        return self.llm_port.translate_to_report(base_payload)