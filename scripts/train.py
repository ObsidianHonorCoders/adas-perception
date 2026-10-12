import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the YOLO11n v0 baseline.")
    parser.add_argument("--config", default="configs/yolo_v0.yaml")
    parser.add_argument("--data", help="Override the dataset YAML path.")
    parser.add_argument("--model", help="Override the model/checkpoint path.")
    parser.add_argument("--epochs", type=int, help="Override the number of epochs.")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    model_value = args.model or config["model"]
    data_value = args.data or config["data"]
    model_path = resolve_path(model_value)
    data_path = resolve_path(data_value)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}. Run "
            "`python scripts/download_weights.py` first."
        )
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {data_path}")

    project = resolve_path(config.get("project", "runs/train"))
    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        epochs=args.epochs if args.epochs is not None else config["epochs"],
        imgsz=config.get("imgsz", 640),
        batch=config.get("batch", 16),
        patience=config.get("patience", 20),
        workers=config.get("workers", 4),
        seed=config.get("seed", 0),
        project=str(project),
        name=config.get("name", "yolo11n_v0"),
    )


if __name__ == "__main__":
    main()
