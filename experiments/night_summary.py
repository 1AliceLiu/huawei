# 夜间实验 step6: 汇总 baseline vs final，输出对比摘要。
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "logs/night_baseline_metrics.json"
FINAL = ROOT / "logs/night_final_metrics.json"
HYP = ROOT / "logs/best_hyperparameters.yaml"
OUT = ROOT / "logs/night_summary.json"


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    base = load(BASE)
    final = load(FINAL)
    hyp = HYP.read_text() if HYP.exists() else None
    summary = {"baseline": base, "final": final, "best_hyperparameters": hyp}
    if base and final:
        for k in ("mAP50-95", "mAP50", "precision", "recall"):
            d = final[k] - base[k]
            summary.setdefault("delta", {})[k] = d
            summary["delta"][k + "_rel"] = (d / base[k]) if base[k] else None
    OUT.write_text(json.dumps(summary, indent=2))
    print("SUMMARY " + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
