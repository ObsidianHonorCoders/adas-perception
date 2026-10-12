import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained detector.")
    parser.add_argument(
        "--weights",
        default="runs/train/yolo11n_v0/weights/best.pt",
        help="Path to the trained checkpoint.",
    )
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()

    weights = resolve_path(args.weights)
    data_path = resolve_path(args.data)
    if not weights.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {weights}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {data_path}")

    dataset = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    dataset_root = Path(dataset.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    split_path = dataset.get(args.split)
    if not split_path:
        raise ValueError(f"The dataset config does not define a '{args.split}' split.")
    split_dir = Path(split_path)
    if not split_dir.is_absolute():
        split_dir = dataset_root / split_dir
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset {args.split} split not found: {split_dir}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        project=str(ROOT / "runs" / "eval"),
        name=f"{weights.stem}_{args.split}",
    )
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
