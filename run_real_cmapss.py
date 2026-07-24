import os
import random
import time
import copy
from typing import Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import root_mean_squared_error, accuracy_score
from sklearn.model_selection import GroupShuffleSplit

from src.core_engine import PureHexagonalMaintenanceEngine
from src.modules.cmapss_engine import CmapssDataIngestionAdapter
from src.ports import LLMTranslatorPort, MaintenanceReportPayload, MaintenanceModelPort, AssetTelemetryDTO


# ----------------------------------------------------
# ⚖️ ASİMETRİK C-MAPSS HAVACILIK LOSS FONKSİYONU
# ----------------------------------------------------
class AsymmetricCMAPSSLoss(nn.Module):
    def __init__(self):
        super(AsymmetricCMAPSSLoss, self).__init__()

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        loss = torch.where(
            diff < 0,
            torch.exp(-diff / 13.0) - 1.0,
            torch.exp(diff / 10.0) - 1.0
        )
        return torch.mean(loss)


# ----------------------------------------------------
# 🧠 SOTA MİMARİ: 1D-CNN + LSTM HYBRID NETWORK
# ----------------------------------------------------
class CmapssCNNLSTMNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super(CmapssCNNLSTMNetwork, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc_rul = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.fc_risk = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_conv = x.transpose(1, 2)
        x_conv = self.relu(self.conv1d(x_conv))
        x_conv = x_conv.transpose(1, 2)

        out, _ = self.lstm(x_conv)
        last_hidden = out[:, -1, :]

        pred_rul = self.fc_rul(last_hidden)
        pred_risk = self.fc_risk(last_hidden)
        return pred_rul.squeeze(-1), pred_risk.squeeze(-1)


# ----------------------------------------------------
# 🔌 OPTİMİZE DEEP TEMPORAL ADAPTER (GROUP-AWARE SPLIT + EARLY STOPPING)
# ----------------------------------------------------
class DeepTemporalModelAdapter(MaintenanceModelPort):
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.feature_history = []
        self.feature_cols = []

    def fit_model(self, X_seq: np.ndarray, y_rul: np.ndarray, y_fail: np.ndarray, groups: np.ndarray,
                  max_epochs: int = 100, batch_size: int = 64, patience: int = 8,
                  hidden_dim: int = 64, num_layers: int = 2, pos_weight_val: float = 4.0):
        input_dim = X_seq.shape[2]
        self.model = CmapssCNNLSTMNetwork(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(self.device)

        # 🔒 GroupShuffleSplit: aynı motorun sequence'ları train VE val'e bölünmüyor (Bug #1 fix)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        train_idx, val_idx = next(gss.split(X_seq, y_rul, groups=groups))

        X_tr, X_val = X_seq[train_idx], X_seq[val_idx]
        y_rul_tr, y_rul_val = y_rul[train_idx], y_rul[val_idx]
        y_fail_tr, y_fail_val = y_fail[train_idx], y_fail[val_idx]

        train_dataset = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_rul_tr, dtype=torch.float32),
            torch.tensor(y_fail_tr, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_rul_val, dtype=torch.float32),
            torch.tensor(y_fail_val, dtype=torch.float32)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)

        criterion_rul = AsymmetricCMAPSSLoss()
        pos_weight = torch.tensor([pos_weight_val]).to(self.device)
        criterion_risk = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        print(
            f"   ⚡ Deep CNN-LSTM Eğitimi Başladı ({self.device} üzerinde | Max Epochs: {max_epochs} | "
            f"Batch: {batch_size} | Patience: {patience} | Hidden: {hidden_dim})...")

        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            train_loss = 0.0
            for bx, by_rul, by_fail in train_loader:
                bx, by_rul, by_fail = bx.to(self.device), by_rul.to(self.device), by_fail.to(self.device)

                optimizer.zero_grad()
                pred_rul, pred_risk = self.model(bx)

                loss_rul = criterion_rul(pred_rul, by_rul)
                loss_risk = criterion_risk(pred_risk, by_fail)
                loss = (loss_rul / 50.0) + 50.0 * loss_risk

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by_rul, by_fail in val_loader:
                    bx, by_rul, by_fail = bx.to(self.device), by_rul.to(self.device), by_fail.to(self.device)
                    pred_rul, pred_risk = self.model(bx)

                    loss_rul = criterion_rul(pred_rul, by_rul)
                    loss_risk = criterion_risk(pred_risk, by_fail)
                    loss = (loss_rul / 50.0) + 50.0 * loss_risk
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"   🛑 Early Stopping Tetiklendi! Epoch {epoch}'da eğitim durduruldu. En iyi model ağırlıkları yüklendi.")
                break

        self.model.load_state_dict(best_model_wts)
        self.model.eval()

    def predict_rul(self, features: dict) -> float:
        if not self.feature_cols:
            self.feature_cols = list(features.keys())

        current_feat_vector = [features[c] for c in self.feature_cols]
        self.feature_history.append(current_feat_vector)

        if len(self.feature_history) > self.sequence_length:
            self.feature_history.pop(0)

        feat_arr = np.array(self.feature_history, dtype=np.float32)
        if len(feat_arr) < self.sequence_length:
            pad_len = self.sequence_length - len(feat_arr)
            pad = np.tile(feat_arr[0], (pad_len, 1))
            seq = np.vstack([pad, feat_arr])
        else:
            seq = feat_arr

        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_rul, _ = self.model(seq_tensor)

        return max(0.0, float(pred_rul.item()))

    def predict_failure_risk(self, features: dict) -> float:
        feat_arr = np.array(self.feature_history, dtype=np.float32)
        if len(feat_arr) < self.sequence_length:
            pad_len = self.sequence_length - len(feat_arr)
            pad = np.tile(feat_arr[0], (pad_len, 1))
            seq = np.vstack([pad, feat_arr])
        else:
            seq = feat_arr

        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, pred_risk = self.model(seq_tensor)

        return float(pred_risk.item())

    def predict_batch_unit(self, unit_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        data = unit_df[self.feature_cols].values
        num_rows = len(data)
        sequences = []

        for i in range(num_rows):
            if i < self.sequence_length - 1:
                pad_len = self.sequence_length - (i + 1)
                pad = np.tile(data[0], (pad_len, 1))
                seq = np.vstack([pad, data[:i + 1]])
            else:
                seq = data[i - self.sequence_length + 1: i + 1]
            sequences.append(seq)

        batch_tensor = torch.tensor(np.array(sequences, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            preds_rul, preds_risk = self.model(batch_tensor)

        return torch.clamp(preds_rul, min=0.0).cpu().numpy(), preds_risk.cpu().numpy()


class SimpleLLMAdapter(LLMTranslatorPort):
    def translate_to_report(self, payload: MaintenanceReportPayload) -> MaintenanceReportPayload:
        payload.generated_report_tr = f"[{payload.consensus_status.value}] Tahmini RUL: {payload.predicted_rul}"
        payload.generated_report_en = f"[{payload.consensus_status.name}] Estimated RUL: {payload.predicted_rul}"
        return payload


# ----------------------------------------------------
# 🎛️ DATASET'E ÖZGÜ HİPERPARAMETRELER (Bug #4 fix)
# FD002/FD004: 6 operating condition -> daha büyük kapasite + daha sabırlı early stopping
# ----------------------------------------------------
DATASET_CONFIGS = {
    "FD001": dict(hidden_dim=64,  num_layers=2, sequence_length=30, patience=8,  pos_weight=4.0),
    "FD003": dict(hidden_dim=64,  num_layers=2, sequence_length=30, patience=8,  pos_weight=4.0),
    "FD002": dict(hidden_dim=96,  num_layers=2, sequence_length=35, patience=12, pos_weight=3.0),
    "FD004": dict(hidden_dim=128, num_layers=2, sequence_length=40, patience=15, pos_weight=3.0),
}


if __name__ == "__main__":
    start_time = time.time()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(BASE_DIR, "data", "raw_cmapss")

    print("🚀 C-MAPSS CANLI SİMÜLASYON VE AKADEMİK METRİK TESTİ (SOTA CNN-LSTM + GROUP-AWARE SPLIT + REGIME CLUSTERING)\n")

    llm_adapter = SimpleLLMAdapter()

    datasets = ["FD001", "FD002", "FD003", "FD004"]

    overall_y_true_rul = []
    overall_y_pred_rul = []
    overall_y_true_class = []
    overall_y_pred_class = []

    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]

        # ⚠️ Her dataset kendi ingestion adapter'ını alır: regime_model, active_sensors, scaler
        #    dataset'ler arasında karışmasın diye.
        ingestion_adapter = CmapssDataIngestionAdapter(base_dir=raw_data_dir, window_size=30)

        train_df = ingestion_adapter.load_and_preprocess_single_set(ds)
        test_df = ingestion_adapter.load_and_preprocess_test_single_set(ds)

        exclude_cols = ["unit_number", "time_in_cycles", "setting_1", "setting_2", "setting_3", "dataset", "RUL",
                        "failure_in_window", "end_rul", "true_rul", "regime"]
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]

        print(f"\n🔥 {ds} Seti İçin Jenerik Deep Temporal Model Eğitiliyor (seq_len={cfg['sequence_length']}, "
              f"hidden={cfg['hidden_dim']}, patience={cfg['patience']})...")

        X_seq_train, y_rul_train, y_fail_train, groups_train = ingestion_adapter.create_lstm_sequences(
            train_df, feature_cols, sequence_length=cfg["sequence_length"]
        )

        model_adapter = DeepTemporalModelAdapter(sequence_length=cfg["sequence_length"])
        model_adapter.feature_cols = feature_cols

        model_adapter.fit_model(
            X_seq_train, y_rul_train, y_fail_train, groups_train,
            max_epochs=100, batch_size=64, patience=cfg["patience"],
            hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], pos_weight_val=cfg["pos_weight"]
        )

        engine = PureHexagonalMaintenanceEngine(
            model_port=model_adapter,
            llm_port=llm_adapter
        )

        # ----------------------------------------------------
        # BÖLÜM A: RASTGELE SEÇİLEN MOTORUN SİMÜLASYONU
        # ----------------------------------------------------
        model_adapter.feature_history.clear()  # 🔒 Bug #5 fix: dataset/unit geçişinde state sızıntısı olmasın

        available_units = test_df["unit_number"].unique()
        selected_unit = random.choice(available_units)

        unit_df = test_df[test_df["unit_number"] == selected_unit].copy()
        max_cycle = unit_df["time_in_cycles"].max()

        print(f"\n✈️  DATASET: {ds} | SİMÜLASYON MOTORU: Unit #{selected_unit} (Toplam {max_cycle} Uçuş)")
        print(
            f"{'Cycle':<8} | {'Gerçek RUL':<10} | {'Tahmin RUL':<10} | {'Risk (%)':<10} | {'Consensus Durumu':<24} | {'Işık'}")
        print("-" * 85)

        sim_preds_rul, sim_preds_risk = model_adapter.predict_batch_unit(unit_df)

        for idx, row in unit_df.reset_index(drop=True).iterrows():
            c = int(row["time_in_cycles"])
            features = {col: float(row[col]) for col in feature_cols if col in row}

            pred_rul = float(sim_preds_rul[idx])
            risk_pct = float(sim_preds_risk[idx]) * 100

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
        y_true_ds, y_pred_ds, y_true_fail_ds, y_pred_fail_ds = [], [], [], []

        for unit_id, unit_test_df in test_df.groupby("unit_number"):
            preds_rul, preds_risk = model_adapter.predict_batch_unit(unit_test_df)

            last_row = unit_test_df.iloc[-1]
            if pd.isna(last_row["true_rul"]):
                continue

            t_rul = float(last_row["true_rul"])
            final_pred_rul = float(preds_rul[-1])
            final_pred_risk = float(preds_risk[-1])

            y_true_ds.append(t_rul)
            y_pred_ds.append(final_pred_rul)
            y_true_fail_ds.append(1 if t_rul <= 30 else 0)
            y_pred_fail_ds.append(1 if final_pred_risk >= 0.5 else 0)

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

    total_rmse = root_mean_squared_error(overall_y_true_rul, overall_y_pred_rul)
    total_acc = accuracy_score(overall_y_true_class, overall_y_pred_class) * 100

    end_time = time.time()
    total_seconds = int(end_time - start_time)
    minutes = total_seconds // 60
    seconds = total_seconds % 60

    print("\n🎯 TÜM SETLER GENEL HİBRİT ÖZETİ:")
    print(f"   • Genel RUL RMSE Skor : {total_rmse:.2f} Uçuş")
    print(f"   • Genel Arıza Accuracy: %{total_acc:.1f}")
    print(f"   ⏱️ Toplam Çalışma Süresi: {minutes} Dakika {seconds} Saniye")
    print("=" * 85)

    print("\n✅ SİMÜLASYON VE AKADEMİK METRİK TESTİ BAŞARIYLA TAMAMLANDI!")