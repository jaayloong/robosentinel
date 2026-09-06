from __future__ import annotations
import argparse
import pandas as pd
from .detector import RoboSentinelDetector
from .evaluation import evaluate_labeled
from .simulator import generate_telemetry

def main() -> None:
    parser = argparse.ArgumentParser(prog="robosentinel"); sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("simulate", help="Generate synthetic robot telemetry"); gen.add_argument("--rows", type=int, default=1500); gen.add_argument("--seed", type=int, default=42); gen.add_argument("--output", default="telemetry.csv")
    analyze = sub.add_parser("analyze", help="Analyze a telemetry CSV"); analyze.add_argument("input"); analyze.add_argument("--output", default="scored_telemetry.csv"); analyze.add_argument("--baseline-fraction", type=float, default=.35)
    args = parser.parse_args()
    if args.command == "simulate":
        df = generate_telemetry(n=args.rows, seed=args.seed); df.to_csv(args.output, index=False); print(f"Wrote {len(df)} samples to {args.output}"); return
    df = pd.read_csv(args.input); result = RoboSentinelDetector().fit_score_split(df, baseline_fraction=args.baseline_fraction); result.frame.to_csv(args.output, index=False)
    metrics = evaluate_labeled(result.frame); print(f"Threshold: {result.threshold:.4f}"); print(f"Anomalies: {int(result.frame['is_anomaly'].sum())}/{len(result.frame)}")
    if metrics: print("Metrics:", metrics)
    print(f"Wrote scored telemetry to {args.output}")

if __name__ == "__main__": main()
