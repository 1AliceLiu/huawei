# 夜间实验 step4: 用 tuning 最佳超参做正式训练（单卡，全量数据，完整 epoch）。
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset/batch_12.v14i.merged12.yolov11/data.yaml"
WEIGHTS = ROOT / "yolo26m.pt"
HYP = ROOT / "logs/best_hyperparameters.yaml"
NAME = "yolo26m_merged12_tuned_final"


def main():
    SETTINGS["tensorboard"] = True
    doc = yaml.safe_load(HYP.read_text())
    hyp = {k: float(v) for k, v in doc["hyperparameters"].items()}
    print("FINAL_TRAIN_HYP " + str(hyp))

    model = YOLO(str(WEIGHTS))
    model.train(
        data=str(DATA),
        epochs=400,
        patience=100,
        imgsz=640,
        batch=64,
        optimizer="MuSGD",  # 固定为 baseline 实际 optimizer，禁用 auto
        momentum=0.9,       # baseline auto 分支实际写入的值
        warmup_bias_lr=0.0, # baseline auto 分支实际写入的值
        cos_lr=True,
        close_mosaic=10,
        device=0,
        name=NAME,
        workers=8,
        exist_ok=True,
        **hyp,
    )


if __name__ == "__main__":
    main()
