# 夜间实验 step2: Ray Tune + ASHA 自动搜索 8 个训练超参。
# 单卡（由外部 CUDA_VISIBLE_DEVICES 指定）。optimizer 固定为 baseline 实际使用的 MuSGD。
import signal
import sys

from ray import tune as ray_tune
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset/batch_12.v14i.merged12.yolov11/data.yaml"
WEIGHTS = ROOT / "yolo26m.pt"
TUNE_NAME = "yolo26m_merged12_nighttune"

SEARCH_SPACE = {
    "lr0": ray_tune.loguniform(5e-4, 3e-2),
    "lrf": ray_tune.uniform(0.01, 0.2),
    "weight_decay": ray_tune.uniform(0.0, 0.001),
    "box": ray_tune.uniform(3.0, 12.0),
    "cls": ray_tune.uniform(0.2, 2.0),
    "dfl": ray_tune.uniform(0.4, 4.0),
    "mosaic": ray_tune.uniform(0.3, 1.0),
    "scale": ray_tune.uniform(0.2, 0.9),
}


def _term(signum, frame):
    try:
        import ray

        ray.shutdown()
    except Exception:
        pass
    sys.exit(143)


def main():
    signal.signal(signal.SIGTERM, _term)
    SETTINGS["tensorboard"] = True

    model = YOLO(str(WEIGHTS))
    model.tune(
        data=str(DATA),
        space=SEARCH_SPACE,
        use_ray=True,
        gpu_per_trial=1,
        epochs=15,          # trial 短训，快速判方向
        iterations=10,      # 上限；实际由 90min 时间封顶决定跑几个
        grace_period=5,     # ASHA 第一档
        imgsz=640,
        batch=64,
        optimizer="MuSGD",  # 固定为 baseline 实际 optimizer，禁用 auto
        momentum=0.9,       # baseline auto 分支实际写入的值
        warmup_bias_lr=0.0, # baseline auto 分支实际写入的值
        cos_lr=True,
        close_mosaic=0,     # trial 只有 15 epoch，保持 mosaic 全程开启，使 mosaic 搜索有意义
        device=0,
        name=TUNE_NAME,
        workers=8,
    )


if __name__ == "__main__":
    main()
