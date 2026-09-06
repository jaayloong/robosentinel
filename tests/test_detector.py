from robosentinel.detector import RoboSentinelDetector
from robosentinel.evaluation import evaluate_labeled
from robosentinel.simulator import generate_telemetry

def test_simulator_schema_and_faults():
    df=generate_telemetry(n=700,anomaly_fraction=.06,seed=3); assert len(df)==700; assert df["is_fault"].sum()>0; assert df["fault_type"].nunique()>=2

def test_detector_outputs_explanations():
    df=generate_telemetry(n=900,anomaly_fraction=.06,seed=7); result=RoboSentinelDetector().fit_score_split(df); scored=result.frame
    assert 0<result.threshold<1.5; assert scored["anomaly_score"].between(0,1).all(); assert scored["root_cause_1"].notna().all(); assert scored["is_anomaly"].sum()>0

def test_detector_has_signal_on_synthetic_faults():
    df=generate_telemetry(n=1300,anomaly_fraction=.07,seed=12); scored=RoboSentinelDetector().fit_score_split(df).frame; metrics=evaluate_labeled(scored)
    assert metrics["recall"]>=.35; assert metrics["precision"]>=.20
