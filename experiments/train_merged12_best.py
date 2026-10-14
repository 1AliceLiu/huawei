# 正式训练（使用 tuning 最佳超参）
# dataset : dataset/batch_12.v14i.merged12.yolov11  (12类)
# tuning  : Ray+ASHA, 最佳 trial = yolo26m_merged12_nighttune-2/_tune_74eac_00000
# 最佳参数: lr0=0.011384, lrf=0.153545, weight_decay=0.000712,
#           box=4.624008, cls=0.321418, dfl=1.995964, mosaic=0.438084, scale=0.546631
#
# 运行（只用物理卡 1，卡 0 有服务，勿动）：
#   CUDA_VISIBLE_DEVICES=1 python3 experiments/train_merged12_best.py
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset/batch_12.v14i.merged12.yolov11/data.yaml"
WEIGHTS = ROOT / "yolo26m.pt"
HYP_FILE = ROOT / "logs/best_hyperparameters.yaml"

# tuning 最佳参数（与 logs/best_hyperparameters.yaml 一致，作为兜底写死）
BEST = {
    "lr0": 0.011384356043089535,
    "lrf": 0.15354537604105567,
    "weight_decay": 0.0007120138023833988,
    "box": 4.624008482667295,
    "cls": 0.32141806670197043,
    "dfl": 1.9959644782892578,
    "mosaic": 0.43808360184679473,
    "scale": 0.5466308088210211,
}


def main():
    SETTINGS["tensorboard"] = True

    # 优先读 tuning 落盘的 yaml；读不到就用上面写死的 BEST
    hyp = dict(BEST)
    if HYP_FILE.exists():
        try:
            doc = yaml.safe_load(HYP_FILE.read_text())
            hyp = {k: float(v) for k, v in doc["hyperparameters"].items()}
            print(f"[hyp] loaded from {HYP_FILE}")
        except Exception as e:
            print(f"[hyp] yaml load failed ({e}), using built-in BEST")

    print("[hyp]", hyp)

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"yolo26m_merged12_best_{current_time}"

    model = YOLO(str(WEIGHTS))
    model.train(
        data=str(DATA),
        epochs=400,
        patience=100,
        imgsz=640,
        batch=64,
        optimizer="MuSGD",   # 与 baseline 实际 optimizer 一致，禁用 auto
        momentum=0.9,        # baseline auto 分支实际写入的值
        warmup_bias_lr=0.0,  # baseline auto 分支实际写入的值
        cos_lr=True,
        close_mosaic=10,
        device=0,
        name=run_name,
        workers=8,
        **hyp,
    )


if __name__ == "__main__":
    main()
