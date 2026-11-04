# train_merged12_1280.py
# 目的 : 验证提高输入分辨率 (imgsz 960 -> 1280) 对稀有类/小目标 (尤其 Container、Motorcycle) 的影响。
# dataset : dataset/clean_part_batch_12.v8i.merged12.yolov11  (12类, group 无泄漏; 标签已于 2026-10-16 人工清理)
# baseline: runs/detect/yolo26s_merged12_20261016_0118  (yolo26s, imgsz=960, batch=128)
# 本次改动 : 模型 yolo26m ; imgsz=1280 ; batch=32 (DDP 总 batch, 均分 16/卡) ; device=1,2
#            (原计划 batch=64 在 2xL20 上 OOM, 见 experiments/train_merged12_1280.md)
# results : runs/detect/yolo26m_merged12_1280_<YYYYMMDD_HHMM>
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

ROOT = Path(__file__).resolve().parents[1]


def main():
    SETTINGS["tensorboard"] = True

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"yolo26m_merged12_1280_{current_time}"

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
        device="1,2",
        name=run_name,
        workers=4,
    )


if __name__ == "__main__":
    main()
