import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize metrics from a YOLO training run.")
    parser.add_argument("--csv", default="runs/train/yolo11n_v0/results.csv")
    parser.add_argument(
        "--output",
        default="reports/metrics/yolo11n_v0_summary.csv",
        help="Where to write the one-row summary.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"Training metrics CSV not found: {csv_path}")

    results = pd.read_csv(csv_path)
    results.columns = results.columns.str.strip()
    map_column = "metrics/mAP50-95(B)"
    if map_column not in results:
        raise ValueError(f"Expected '{map_column}' in the training results CSV.")

    best = results.loc[results[map_column].idxmax()]
    summary = {
        "best_epoch": int(best["epoch"]) if "epoch" in results else int(best.name),
        "precision": best.get("metrics/precision(B)"),
        "recall": best.get("metrics/recall(B)"),
        "mAP50": best.get("metrics/mAP50(B)"),
        "mAP50-95": best[map_column],
    }
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(output_path, index=False)

    print(pd.Series(summary).to_string())
    print(f"\nSummary saved to {output_path}")


if __name__ == "__main__":
    main()
