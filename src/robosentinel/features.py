from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_SIGNALS


def validate_signals(df: pd.DataFrame, signals: list[str] | None = None) -> list[str]:
    signals = signals or DEFAULT_SIGNALS
    missing = [c for c in signals if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required telemetry columns: {', '.join(missing)}")
    return signals


def engineer_features(df: pd.DataFrame, signals: list[str] | None = None) -> pd.DataFrame:
    """Build model features from raw telemetry without leaking future samples."""
    signals = validate_signals(df, signals)
    x = df[signals].astype(float).copy()

    for col in signals:
        s = x[col]
        x[f"{col}__delta"] = s.diff().fillna(0.0)
        x[f"{col}__ema_gap"] = (s - s.ewm(span=20, adjust=False).mean()).fillna(0.0)
        roll = s.rolling(window=25, min_periods=4)
        mean = roll.mean().bfill().fillna(s.mean())
        std = roll.std(ddof=0).replace(0, np.nan).bfill().fillna(s.std(ddof=0) or 1.0)
        x[f"{col}__local_z"] = ((s - mean) / std).replace([np.inf, -np.inf], 0).fillna(0.0)

    eps = 1e-6
    x["current_per_volt"] = x["motor_current_a"] / (x["battery_voltage_v"] + eps)
    x["tracking_stress"] = x["velocity_error"].abs() * x["motor_current_a"]
    x["thermal_load"] = x["motor_temp_c"] * x["motor_current_a"] / 100.0
    x["vibration_current"] = x["vibration_rms"] * x["motor_current_a"]
    return x.replace([np.inf, -np.inf], 0).fillna(0.0)
