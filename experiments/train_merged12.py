# train_merged12.py
# dataset : dataset/batch_12.v8i.merged12.yolov11/splits  (12类, 逐类 train:val:test=7:2:1, group 无泄漏)
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
        data=str(ROOT / "dataset/batch_12.v8i.merged12.yolov11/splits/data_train_pool.yaml"),
        epochs=500,
        patience=50,
        imgsz=640,
        batch=66,
        optimizer="MuSGD",
        momentum=0.9,
        warmup_bias_lr=0.0,
        lr0=0.001,
        cos_lr=True,
        device="0,1,2",
        name=run_name,
        workers=4,
    )


if __name__ == "__main__":
    main()
