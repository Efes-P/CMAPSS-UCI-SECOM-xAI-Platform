from dataclasses import dataclass, field
from typing import Dict, Optional
import xgboost as xgb
import pandas as pd
from src.ports import MaintenanceModelPort


@dataclass
class CmapssModelAdapter(MaintenanceModelPort):
    regressor: Optional[xgb.XGBRegressor] = field(default=None)
    classifier: Optional[xgb.XGBClassifier] = field(default=None)

    def train_mock_models(self, X_train: pd.DataFrame, y_rul: pd.Series, y_fail: pd.Series):
        self.regressor = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
        self.regressor.fit(X_train, y_rul)

        self.classifier = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
        self.classifier.fit(X_train, y_fail)

    def predict_rul(self, features: Dict[str, float]) -> float:
        if self.regressor is None:
            raise ValueError("Model henüz eğitilmedi!")
        df_feat = pd.DataFrame([features])
        pred = self.regressor.predict(df_feat)[0]
        return max(0.0, float(pred))

    def predict_failure_risk(self, features: Dict[str, float]) -> float:
        if self.classifier is None:
            raise ValueError("Model henüz eğitilmedi!")
        df_feat = pd.DataFrame([features])
        prob = self.classifier.predict_proba(df_feat)[0][1]
        return float(prob)