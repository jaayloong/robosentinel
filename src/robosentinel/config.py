from dataclasses import dataclass, field


DEFAULT_SIGNALS = [
    "motor_current_a",
    "motor_temp_c",
    "vibration_rms",
    "battery_voltage_v",
    "velocity_error",
    "loop_latency_ms",
]


@dataclass(slots=True)
class DetectorConfig:
    contamination: float = 0.04
    random_state: int = 42
    robust_z_threshold: float = 4.0
    ensemble_iforest_weight: float = 0.65
    ensemble_robust_weight: float = 0.35
    signals: list[str] = field(default_factory=lambda: DEFAULT_SIGNALS.copy())
