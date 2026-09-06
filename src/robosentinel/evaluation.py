from __future__ import annotations
import pandas as pd

def evaluate_labeled(result: pd.DataFrame) -> dict[str, float]:
    if "is_fault" not in result.columns: return {}
    y = result["is_fault"].astype(int).to_numpy(); pred = result["is_anomaly"].astype(int).to_numpy()
    tp = int(((y == 1) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision,4), "recall": round(recall,4), "f1": round(f1,4), "accuracy": round((tp+tn)/max(1,len(y)),4), "tp":tp, "tn":tn, "fp":fp, "fn":fn}
