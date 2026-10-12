from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "models" / "pretrained"
WEIGHTS_PATH = WEIGHTS_DIR / "yolo11n.pt"


def main() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if WEIGHTS_PATH.exists():
        print(f"YOLO11n weights already exist: {WEIGHTS_PATH}")
        return

    YOLO(str(WEIGHTS_PATH))
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Ultralytics did not download the checkpoint to {WEIGHTS_PATH}")
    print(f"Downloaded YOLO11n weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
