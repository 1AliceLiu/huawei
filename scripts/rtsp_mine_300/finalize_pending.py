"""采集收尾一键脚本：prelabel 去重 -> JSON 合并到 images/ -> manifest 刷新。

解决 R7 的两个痛点：
1. 同目标叠框：同类 IoU>=--same-iou 删低 conf；异类 IoU>=--cross-iou 留高 conf
   （R7 实测：同类 0.6-0.85 全是真重复；异类只敢留高 conf，类错留给人工改）。
2. 标哪份：把 annotations_xany/*.json 拷贝到 images/*.json，打开即标。

只动 --dir 目录；先整库备份 annotations_xany_bak_<ts>；manifest 的 n_box/n_cls
按最终 JSON 重算。--dry-run 只打印不写。

用法：
  python scripts/rtsp_mine_300/finalize_pending.py --dir tmp/r8_pending
  python scripts/rtsp_mine_300/finalize_pending.py --dir tmp/r8_pending --dry-run
"""
import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from collect import box_iou  # noqa: E402


def box_of(s):
    (x1, y1), (x2, y2) = s["points"]
    return [float(x1), float(y1), float(x2), float(y2)]


def conf_of(s):
    try:
        return float((s.get("attributes") or {}).get("conf", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="tmp/r8_pending")
    ap.add_argument("--same-iou", type=float, default=0.6)
    ap.add_argument("--cross-iou", type=float, default=0.9)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = ROOT / args.dir
    js_dir, img_dir = d / "annotations_xany", d / "images"
    jps = sorted(js_dir.glob("*.json"))
    assert jps, f"无 JSON：{js_dir}"
    print(f"{len(jps)} jsons, same-iou={args.same_iou} cross-iou={args.cross_iou} "
          f"dry_run={args.dry_run}")

    if not args.dry_run:
        bak = d / f"annotations_xany_bak_{datetime.now():%Y%m%d_%H%M%S}"
        shutil.copytree(js_dir, bak)
        print("backup ->", bak.name)

    stat = Counter()
    per_file = {}
    for jp in jps:
        payload = json.loads(jp.read_text(encoding="utf-8"))
        shapes = payload.get("shapes", [])
        drop = set()
        idx = list(range(len(shapes)))
        # pass 1: 同类
        by_cls = {}
        for i in idx:
            by_cls.setdefault(shapes[i].get("label"), []).append(i)
        for items in by_cls.values():
            items.sort(key=lambda i: -conf_of(shapes[i]))
            for a in range(len(items)):
                if items[a] in drop:
                    continue
                for b in range(a + 1, len(items)):
                    if items[b] in drop:
                        continue
                    if box_iou(box_of(shapes[items[a]]), box_of(shapes[items[b]])) >= args.same_iou:
                        drop.add(items[b])
                        stat["same"] += 1
        # pass 2: 异类（只看存活的）
        alive = [i for i in idx if i not in drop]
        for a, b in combinations(alive, 2):
            if shapes[a].get("label") == shapes[b].get("label"):
                continue
            if box_iou(box_of(shapes[a]), box_of(shapes[b])) >= args.cross_iou:
                loser = b if conf_of(shapes[a]) >= conf_of(shapes[b]) else a
                drop.add(loser)
                stat["cross"] += 1
        if drop:
            per_file[jp.stem] = sorted(drop)
            stat["files"] += 1
        if not args.dry_run and drop:
            payload["shapes"] = [s for i, s in enumerate(shapes) if i not in drop]
            jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"dup removed: same={stat['same']} cross={stat['cross']} in {stat['files']} files")
    if args.dry_run:
        return

    # 合并到 images/（打开即标）
    n_merge = 0
    for jp in jps:
        dst = img_dir / jp.name
        if not dst.is_file() or dst.read_text(encoding="utf-8") != jp.read_text(encoding="utf-8"):
            shutil.copy2(jp, dst)
            n_merge += 1
    print(f"merged to images/: {n_merge}")

    # manifest n_box/n_cls 按最终 JSON 重算（file/cam 行保留，只收敛到现存文件）
    mp = d / "manifest.csv"
    if mp.is_file():
        rows = list(csv.DictReader(open(mp, encoding="utf-8")))
        keep = [r for r in rows if (img_dir / r["file"]).is_file()]
        for r in keep:
            shp = json.loads((img_dir / (Path(r["file"]).stem + ".json")).read_text(
                encoding="utf-8")).get("shapes", []) if (img_dir / (Path(r["file"]).stem + ".json")).is_file() else []
            r["n_box"] = str(len(shp))
            r["n_cls"] = str(len({s.get("label") for s in shp}))
        with open(mp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(keep)
        print(f"manifest: {len(rows)} -> {len(keep)} (n_box/n_cls refreshed)")


if __name__ == "__main__":
    main()
