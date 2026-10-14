# train_merged12_tune.py
# dataset : dataset/batch_12.v14i.merged12.yolov11  (12类: v9 11类 + Container)
# results : runs/detect/yolo26m_merged12_tune_<YYYYMMDD_HHMM>
#
# 运行前必须先隔离显卡：卡 0 有服务占用，用 CUDA_VISIBLE_DEVICES 把卡 0 藏起来，
# 让 Ray 和所有 trial 子进程都只能看到卡 1/2/3：
#   export CUDA_VISIBLE_DEVICES=1,2,3
#   python3 experiments/train_merged12_tune.py
#
# 注意：Ray trial 的工作目录不是项目根目录，因此 data 和权重必须用绝对路径。
from datetime import datetime
from pathlib import Path

from ray import tune as ray_tune
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset/batch_12.v14i.merged12.yolov11/data.yaml"
WEIGHTS = ROOT / "yolo26m.pt"


def main():
    SETTINGS["tensorboard"] = True

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"yolo26m_merged12_tune_{current_time}"

    model = YOLO(str(WEIGHTS))

    # 用 Ray Tune 的分布对象定义搜索空间，不能再用 (min, max) 元组
    search_space = {
        "lr0":    ray_tune.loguniform(1e-4, 5e-3),   # 跨数量级，用 loguniform
        "lrf":    ray_tune.uniform(0.01, 0.9),
        "box":    ray_tune.uniform(5.0, 12.0),
        "cls":    ray_tune.uniform(0.3, 1.0),
        "mosaic": ray_tune.uniform(0.5, 1.0),
        "scale":  ray_tune.uniform(0.3, 0.9),
    }

    model.tune(
        data=str(DATA),
        space=search_space,
        use_ray=True,            # 启用 Ray Tune，才能快速筛选
        gpu_per_trial=1,         # 每个 trial 只占 1 张卡，避免和其他服务抢卡
        epochs=40,               # 好组合跑满的上限
        iterations=10,           # 先试 10 组，样本太少难有结论
        grace_period=10,         # 前 10 epoch 不杀，避免误杀
        imgsz=640,
        batch=64,
        cos_lr=True,
        close_mosaic=10,
        device=-1,
        name=run_name,
        workers=8,
    )


if __name__ == "__main__":
    main()