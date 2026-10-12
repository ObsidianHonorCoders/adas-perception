import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run object detection on an image or video.")
    parser.add_argument("--source", required=True, help="Image, video, or directory to process.")
    parser.add_argument(
        "--weights",
        default="runs/train/yolo11n_v0/weights/best.pt",
        help="Path to the trained checkpoint.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    args = parser.parse_args()

    weights = resolve_path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {weights}")

    source_path = Path(args.source)
    source = str(source_path if source_path.is_absolute() else ROOT / source_path)
    model = YOLO(str(weights))
    model.predict(
        source=source,
        conf=args.conf,
        save=True,
        project=str(ROOT / "runs" / "predict"),
        name=weights.stem,
    )


if __name__ == "__main__":
    main()
