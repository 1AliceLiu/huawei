# 夜间实验 step5: 正式训练 best.pt 在当前 val 上重评，与 baseline 同口径比较。
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset/batch_12.v14i.merged12.yolov11/data.yaml"
CKPT = ROOT / "runs/detect/yolo26m_merged12_tuned_final/weights/best.pt"
OUT = ROOT / "logs/night_final_metrics.json"


def main():
    model = YOLO(str(CKPT))
    m = model.val(
        data=str(DATA),
        split="val",
        imgsz=640,
        batch=64,
        device=0,
        project=str(ROOT / "runs/detect"),
        name="night_final_val",
        exist_ok=True,
    )
    res = {
        "checkpoint": str(CKPT),
        "mAP50-95": float(m.box.map),
        "mAP50": float(m.box.map50),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print("FINAL_METRICS " + json.dumps(res))


if __name__ == "__main__":
    main()
