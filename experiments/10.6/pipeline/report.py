"""Build human-readable quality report from atlas/*/stats.json.
Run: uv run python experiments/10.6/pipeline/report.py
Output: experiments/10.6/atlas/REPORT.md
"""
import json
from pathlib import Path

ATLAS = Path("experiments/10.6/atlas")


def main():
    summary = json.load(open(ATLAS / "summary.json"))
    total_boxes = sum(s["boxes"] for s in summary)
    lines = ["# Dataset Visual Survey — Quality Report (train split)", "",
             f"Total boxes: {total_boxes}. Embedding: DINOv2 ViT-S/14, area filter >= 0.01 "
             "(fallback 0.005/0.0 if class too small). Flags: artifact = single-source dominated, "
             "candidate = cross-video visual subtype.", "",
             "| Class | Boxes | Qual% | Emb | K | Artifacts | Notes |",
             "|---|---|---|---|---|---|---|"]
    for s in summary:
        st = {}
        try:
            st = json.load(open(ATLAS / s["name"] / "stats.json"))
        except FileNotFoundError:
            pass
        qp = (s["qualified"] / max(1, s["boxes"]) * 100) if s["boxes"] else 0
        notes = []
        for c in st.get("clusters", []):
            if c["flag"].startswith("artifact"):
                notes.append(f"C{c['s']}:n={c['n']}/v={c['videos']}")
        lines.append(f"| {s['name']} | {s['boxes']} | {qp:.1f}% | {s['embedded']} | {s['k']} | "
                     f"{s.get('artifacts', 0)} | {'; '.join(notes)} |")
    lines += ["", "## Per-class cluster detail", ""]
    for s in summary:
        p = ATLAS / s["name"] / "stats.json"
        if not p.exists():
            lines += [f"### {s['name']}", "no data (empty class)", ""]
            continue
        st = json.load(open(p))
        lines.append(f"### {s['name']} — boxes={st['boxes_train']}, qualified={st['qualified']}, "
                     f"videos={st['videos']}, med_area={st['median_area']:.4f}")
        for c in st["clusters"]:
            lines.append(f"- C{c['s']}: n={c['n']}, videos={c['videos']}, "
                         f"top1={c['top1_share']*100:.0f}%, {c['flag']}, top={c['top_videos']}")
        lines.append("")
    (ATLAS / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {ATLAS/'REPORT.md'}")


if __name__ == "__main__":
    main()
