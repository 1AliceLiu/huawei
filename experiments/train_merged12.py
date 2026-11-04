# train_merged12.py
# dataset : dataset/clean_part_batch_12.v8i.merged12.yolov11/mix_batch_12.v8i.merged1  (12类, 纯随机划分)
# results : runs/detect/yolo26s_merged12_<YYYYMMDD_HHMM>
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

ROOT = Path(__file__).resolve().parents[1]


def main():
    SETTINGS["tensorboard"] = True

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"yolo26m_merged12_{current_time}"

    # model = YOLO(str(ROOT / "yolo26m.pt"))
    model = YOLO(str(ROOT / "yolo26m.pt"))
    model.train(
        data=str(ROOT / "dataset/clean_part_batch_12.v8i.merged12.yolov11/data.yaml"),
        epochs=120,
        patience=15,
        imgsz=1280,
        batch=32,
        seed=0,
        optimizer="MuSGD",
        momentum=0.9,
        weight_decay=0.001,
        warmup_bias_lr=0.0,
        lr0=0.0005,
        cos_lr=True,
        device="2,1",
        name=run_name,
        workers=4,
    )


if __name__ == "__main__":
    main()