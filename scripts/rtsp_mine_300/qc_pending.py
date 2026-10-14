"""标注前 QC：对 pending 目录（images + annotations_xany + manifest.csv）做质量初筛。

只做保守标记，不删文件：
- delete 建议：只给明显废图（跨 resume 边界的 phash 近重、大面积灰屏/坏帧、极糊）
- review 建议：贴边框、全小框、低置信弱类框（Motorcycle 工人幻觉已知）→ 标的时候重点看
- keep：其余

输出：<dir>/qc_report.csv + 控制台汇总。删除沿用旧流程：人工在 X-AnyLabeling 里把废图移走，
append_*.py 按幸存文件 reconcile manifest（deletions 留在 _delete_/）。

用法：
  python scripts/rtsp_mine_300/qc_pending.py --dir tmp/r7_pending [--preview-n 12]
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from collect import is_gray_screen, phash  # noqa: E402

EDGE = 2.0
DUP_HAMMING = 6


def lap_var(img) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="tmp/r7_pending")
    ap.add_argument("--preview-n", type=int, default=12)
    args = ap.parse_args()

    d = ROOT / args.dir
    img_dir, js_dir = d / "images", d / "annotations_xany"
    manifest = {}
    mp = d / "manifest.csv"
    if mp.is_file():
        for r in csv.DictReader(open(mp, encoding="utf-8")):
            manifest[r["file"]] = r

    rows = []
    for p in sorted(img_dir.glob("*.jpg")):
        stem = p.stem
        cam = stem.rsplit("_", 1)[-1] if "_" in stem else ""
        img = cv2.imread(str(p))
        if img is None:
            rows.append(dict(file=p.name, cam=cam, err="unreadable", action="delete"))
            continue
        H, W = img.shape[:2]
        gray_std = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std())
        lv = lap_var(img)
        h = phash(img)
        bad, metric = is_gray_screen(img)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corrupt = float((g > 250).mean()) > 0.35 or float((g < 5).mean()) > 0.50

        jp = js_dir / (stem + ".json")
        shapes = json.loads(jp.read_text(encoding="utf-8")).get("shapes", []) if jp.is_file() else []
        boxes = []
        for s in shapes:
            (x1, y1), (x2, y2) = s["points"]
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(W), x2), min(float(H), y2)
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (W * H)
            edge = x1 <= EDGE or y1 <= EDGE or x2 >= W - EDGE or y2 >= H - EDGE
            conf = (s.get("attributes") or {}).get("conf")
            boxes.append((s.get("label"), area, edge, conf))
        n = len(boxes)
        min_area = min([b[1] for b in boxes], default=0.0)
        max_area = max([b[1] for b in boxes], default=0.0)
        n_edge = sum(1 for b in boxes if b[2])
        low_conf = [f"{b[0]}:{b[3]}" for b in boxes
                    if b[3] is not None and b[3] < 0.25 and b[0] in
                    ("Motorcycle", "LGV", "Container", "HGV", "Van")]
        m = manifest.get(p.name, {})
        rows.append(dict(
            file=p.name, cam=cam, score=m.get("score", ""), n_box=n,
            min_area=f"{min_area:.5f}", max_area=f"{max_area:.5f}", n_edge=n_edge,
            low_conf=";".join(low_conf), gray_std=f"{gray_std:.1f}", lap=f"{lv:.0f}",
            gray_flag=int(bad), corrupt=int(corrupt), phash=h, action="keep", reason=""))

    # 近重：同 cam 内两两 phash 汉明 < 6，只留 manifest score 最高者
    by_cam = defaultdict(list)
    for r in rows:
        if "err" not in r:
            by_cam[r["cam"]].append(r)
    n_dup = 0
    for cam, lst in by_cam.items():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if bin(lst[i]["phash"] ^ lst[j]["phash"]).count("1") < DUP_HAMMING:
                    si = float(lst[i]["score"] or 0)
                    sj = float(lst[j]["score"] or 0)
                    drop = lst[i] if si < sj else lst[j]
                    if drop["action"] == "keep":
                        drop["action"] = "delete"
                        drop["reason"] = f"near-dup of {(lst[j] if drop is lst[i] else lst[i])['file']}"
                        n_dup += 1
    for r in rows:
        if "err" in r:
            continue
        if r["action"] == "delete":
            continue
        if r["gray_flag"] or r["corrupt"]:
            r["action"] = "delete"
            r["reason"] = "gray/corrupt"
        elif float(r["lap"]) < 50:
            r["action"] = "delete"
            r["reason"] = "heavy-blur"
        elif r["n_box"] == 0:
            r["action"] = "review"
            r["reason"] = "zero-box"
        elif int(r["n_edge"]) == r["n_box"] and r["n_box"] > 0:
            r["action"] = "review"
            r["reason"] = "all-edge"
        elif float(r["max_area"]) < 0.003:
            r["action"] = "review"
            r["reason"] = "all-tiny"
        elif r["low_conf"]:
            r["action"] = "review"
            r["reason"] = "low-conf-weak:" + r["low_conf"]

    out = d / "qc_report.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "cam", "score", "n_box", "min_area",
                                          "max_area", "n_edge", "low_conf", "gray_std",
                                          "lap", "gray_flag", "corrupt", "action", "reason"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    from collections import Counter
    c = Counter(r["action"] for r in rows)
    print(f"{len(rows)} imgs -> {dict(c)} (dup={n_dup}) -> {out}")
    for r in rows:
        if r["action"] == "delete":
            print(f"  DEL {r['file']} [{r['reason']}]")
    rv = [r for r in rows if r["action"] == "review"]
    print(f"review {len(rv)} (top15):")
    for r in rv[:15]:
        print(f"  REV {r['file']} [{r['reason']}] n={r['n_box']}")


if __name__ == "__main__":
    main()
