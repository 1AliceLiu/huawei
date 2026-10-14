# 夜间实验 step3: 从 tuning 结果里解析最佳 trial，保存 best_hyperparameters.yaml。
# 解析失败则 fallback 到 baseline 默认参数，并记录原因。
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TUNE_GLOB = "yolo26m_merged12_nighttune*"
OUT = ROOT / "logs/best_hyperparameters.yaml"

BASELINE = {
    "lr0": 0.01,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "mosaic": 1.0,
    "scale": 0.5,
}
KEYS = list(BASELINE)


def best_trial():
    best = None  # (trial_name, metric, config)
    trial_dirs = []
    for base in sorted((ROOT / "runs/detect").glob(TUNE_GLOB)):
        if base.name.startswith("_") or not base.is_dir():
            continue
        trial_dirs += sorted(base.glob("_tune_*"))
    for d in trial_dirs:
        rj = d / "result.json"
        if not rj.exists():
            continue
        bestmap, cfg = -1.0, None
        for line in rj.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            v = rec.get("metrics/mAP50-95(B)")
            if v is not None and float(v) > bestmap:
                bestmap, cfg = float(v), rec.get("config")
        if cfg is not None and (best is None or bestmap > best[1]):
            best = (f"{d.parent.name}/{d.name}", bestmap, cfg)
    return best


def main():
    b = best_trial()
    if b is None:
        hyp = dict(BASELINE)
        meta = {"fallback": True, "reason": "no valid trial found in tune dir"}
    else:
        name, metric, cfg = b
        hyp = {k: float(cfg[k]) for k in KEYS if k in cfg}
        for k in KEYS:
            hyp.setdefault(k, BASELINE[k])
        meta = {"fallback": False, "best_trial": name, "tuning_metric_mAP50-95": metric}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump({"hyperparameters": hyp, "meta": meta}, sort_keys=False))
    print("BEST_HYPERPARAMETERS " + json.dumps({"hyperparameters": hyp, "meta": meta}))


if __name__ == "__main__":
    main()
