import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze.py"


def test_analyze_saves_metrics_for_best_map_epoch(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    output_path = tmp_path / "summary.csv"
    pd.DataFrame(
        {
            "epoch": [0, 1],
            "metrics/precision(B)": [0.7, 0.8],
            "metrics/recall(B)": [0.6, 0.75],
            "metrics/mAP50(B)": [0.65, 0.78],
            "metrics/mAP50-95(B)": [0.42, 0.56],
        }
    ).to_csv(results_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--csv",
            str(results_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = pd.read_csv(output_path)
    assert summary.loc[0, "best_epoch"] == 1
    assert summary.loc[0, "mAP50-95"] == 0.56
    assert "Summary saved to" in result.stdout
