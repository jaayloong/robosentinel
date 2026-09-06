from __future__ import annotations

import numpy as np
import pandas as pd


def generate_telemetry(n: int = 1500, anomaly_fraction: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """Generate robot drivetrain telemetry with labeled correlated fault events."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    load = 0.55 + 0.18 * np.sin(t / 55.0) + 0.07 * np.sin(t / 13.0)
    load += rng.normal(0, 0.035, n)
    load = np.clip(load, 0.15, 1.0)
    motor_current = 7.0 + 24.0 * load + rng.normal(0, 1.1, n)
    motor_temp = 37.0 + 31.0 * pd.Series(load).rolling(35, min_periods=1).mean().to_numpy() + rng.normal(0, 0.75, n)
    vibration = 0.12 + 0.42 * load + rng.normal(0, 0.025, n)
    battery = 12.65 - 0.72 * load - 0.012 * motor_current + rng.normal(0, 0.035, n)
    velocity_error = rng.normal(0, 0.055, n) + 0.018 * np.sin(t / 17.0)
    latency = 7.5 + 2.3 * load + rng.normal(0, 0.55, n)
    df = pd.DataFrame({"timestamp_ms": t * 20, "motor_current_a": motor_current, "motor_temp_c": motor_temp, "vibration_rms": vibration, "battery_voltage_v": battery, "velocity_error": velocity_error, "loop_latency_ms": latency, "fault_type": "normal", "is_fault": 0})
    target = max(1, int(n * anomaly_fraction))
    specs = [("drivetrain_drag", 1.35), ("thermal_runaway", 1.25), ("brownout", 1.0), ("mechanical_vibration", 1.15), ("control_loop_lag", 1.0)]
    injected = attempts = 0
    while injected < target and attempts < 200:
        attempts += 1
        width = int(rng.integers(10, 31)); start = int(rng.integers(60, max(61, n - width - 5)))
        idx = np.arange(start, min(start + width, n))
        if df.loc[idx, "is_fault"].any(): continue
        fault, strength = specs[int(rng.integers(0, len(specs)))]
        ramp = np.sin(np.linspace(0.15, np.pi - 0.15, len(idx)))
        if fault == "drivetrain_drag":
            df.loc[idx, "motor_current_a"] += strength * 9.5 * ramp; df.loc[idx, "velocity_error"] += strength * 0.22 * ramp; df.loc[idx, "battery_voltage_v"] -= strength * 0.28 * ramp
        elif fault == "thermal_runaway":
            df.loc[idx, "motor_temp_c"] += strength * np.linspace(3, 17, len(idx)); df.loc[idx, "motor_current_a"] += strength * 4.0 * ramp
        elif fault == "brownout":
            df.loc[idx, "battery_voltage_v"] -= strength * 1.45 * ramp; df.loc[idx, "loop_latency_ms"] += strength * 3.6 * ramp
        elif fault == "mechanical_vibration":
            df.loc[idx, "vibration_rms"] += strength * 0.78 * ramp; df.loc[idx, "motor_current_a"] += strength * 3.2 * ramp
        else:
            df.loc[idx, "loop_latency_ms"] += strength * 11.0 * ramp; df.loc[idx, "velocity_error"] += strength * 0.28 * ramp
        df.loc[idx, "fault_type"] = fault; df.loc[idx, "is_fault"] = 1; injected += len(idx)
    return df
