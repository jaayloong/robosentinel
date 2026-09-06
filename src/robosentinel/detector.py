from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from .config import DetectorConfig
from .features import engineer_features, validate_signals

@dataclass(slots=True)
class DetectionResult:
    frame: pd.DataFrame
    threshold: float
    feature_columns: list[str]

class RoboSentinelDetector:
    """Hybrid unsupervised detector with interpretable sensor attribution."""
    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self.scaler = RobustScaler()
        self.model = IsolationForest(n_estimators=220, contamination=self.config.contamination, random_state=self.config.random_state, n_jobs=-1)
        self.feature_columns = []
        self.signal_median_ = None; self.signal_mad_ = None; self.threshold_ = None

    def fit(self, baseline: pd.DataFrame):
        signals = validate_signals(baseline, self.config.signals)
        features = engineer_features(baseline, signals); self.feature_columns = list(features.columns)
        self.model.fit(self.scaler.fit_transform(features))
        raw = baseline[signals].astype(float); self.signal_median_ = raw.median()
        self.signal_mad_ = (raw - self.signal_median_).abs().median().replace(0, 1e-6)
        self.threshold_ = float(np.quantile(self._ensemble_score(baseline), 1 - self.config.contamination))
        return self

    def _ensure_fitted(self):
        if not self.feature_columns or self.signal_median_ is None or self.signal_mad_ is None: raise RuntimeError("Detector must be fitted before scoring.")

    def _robust_signal_z(self, df):
        self._ensure_fitted(); raw = df[self.config.signals].astype(float)
        return 0.6745 * (raw - self.signal_median_) / self.signal_mad_

    @staticmethod
    def _minmax(values):
        lo, hi = float(np.min(values)), float(np.max(values))
        return np.zeros_like(values, dtype=float) if hi - lo < 1e-12 else (values - lo) / (hi - lo)

    def _ensemble_score(self, df):
        features = engineer_features(df, self.config.signals)
        scaled = self.scaler.transform(features[self.feature_columns])
        iforest = self._minmax(-self.model.decision_function(scaled))
        robust_raw = self._robust_signal_z(df).abs().max(axis=1).to_numpy()
        robust = 1.0 - np.exp(-robust_raw / max(self.config.robust_z_threshold, 1e-6))
        w1, w2 = self.config.ensemble_iforest_weight, self.config.ensemble_robust_weight
        return (w1 * iforest + w2 * robust) / (w1 + w2)

    def score(self, df):
        self._ensure_fitted(); validate_signals(df, self.config.signals)
        scores = self._ensemble_score(df); threshold = float(self.threshold_ if self.threshold_ is not None else 0.75)
        robust_z = self._robust_signal_z(df).abs(); out = df.copy()
        out["anomaly_score"] = scores; out["is_anomaly"] = (scores >= threshold).astype(int)
        names = np.array(self.config.signals); z = robust_z.to_numpy(); top = np.argsort(-z, axis=1)[:, :3]
        out["root_cause_1"] = names[top[:, 0]]; out["root_cause_2"] = names[top[:, 1]]; out["root_cause_3"] = names[top[:, 2]]
        out["root_cause_strength"] = z[np.arange(len(z)), top[:, 0]]
        severity = pd.cut(scores, bins=[-np.inf, threshold * .75, threshold, min(1.0, threshold + .14), np.inf], labels=["normal", "watch", "warning", "critical"], include_lowest=True)
        out["severity"] = severity.astype(str)
        return DetectionResult(out, threshold, self.feature_columns.copy())

    def fit_score_split(self, df, baseline_fraction: float = .35):
        if not .1 <= baseline_fraction <= .8: raise ValueError("baseline_fraction must be between 0.1 and 0.8")
        cut = max(25, int(len(df) * baseline_fraction)); baseline = df.iloc[:cut].copy()
        if "is_fault" in baseline.columns:
            normals = baseline[baseline["is_fault"] == 0]
            if len(normals) >= 25: baseline = normals
        self.fit(baseline); return self.score(df)
