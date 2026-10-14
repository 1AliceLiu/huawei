# train_merged12.py
# dataset : auto_check_dataset/clean_data  (12类, train:val:test=7:2:1, group 无泄漏, 仅用已审查数据)
# results : runs/detect/yolo26m_merged12_<YYYYMMDD_HHMM>
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

ROOT = Path(__file__).resolve().parents[1]


def main():
    SETTINGS["tensorboard"] = True

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"yolo26m_merged12_{current_time}"

    model = YOLO(str(ROOT / "yolo26m.pt"))
    model.train(
        data=str(ROOT / "auto_check_dataset/clean_data/data.yaml"),
        epochs=120,
        patience=25,
        imgsz=960,
        batch=66,
        optimizer="MuSGD",
        momentum=0.9,
        warmup_bias_lr=0.0,
        lr0=0.0005,
        cos_lr=True,
        device="-1,-1",
        name=run_name,
        workers=4,
    )


if __name__ == "__main__":
    main()
