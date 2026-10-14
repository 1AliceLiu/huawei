#!/usr/bin/env python3
"""group-disjoint + 逐类框数比例的 train/val 划分（用于 YOLOv26 迁移微调）。

相比按图片数贴比例，这里保证：
* 每个 group（相机/视频段）原子，不跨 split（防相近帧泄漏）；
* val 中**每个类别的框数**尽量贴合目标比例（默认 10%）；
* 对稀有类额外施加**最小框数约束**（默认 20），否则这些类的 mAP 是噪声。

输出::

    <out>/
      data.yaml                 # train: train/images, val: valid/images
      train/images train/labels
      valid/images valid/labels
      report.json               # 逐类 train/val 框数与比例、约束是否满足
      split_manifest.csv

用法::

    python scripts/build_clean_split.py --val-ratio 0.1 --min-boxes 20
    python scripts/build_clean_split.py --iters 400000 --restarts 8
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def group_key(filename: str) -> str:
    s = Path(filename).stem
    s = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", s, flags=re.I)
    s = re.sub(r"\.rf\.[0-9a-f]+$", "", s, flags=re.I)
    m = re.search(r"_(dahua|cam)(\d+)", s)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    m = re.search(r"_frame_", s)
    if m:
        return "stream_" + s[: m.start()]
    m = re.match(r"^(.*?)_f\d+$", s)
    if m:
        return "seg_" + m.group(1)
    m = re.match(r"^(.*?)_\d+$", s)
    if m and re.search(r"[A-Za-z]", m.group(1)):
        return "pre_" + m.group(1)
    if re.match(r"^\d+$", s):
        return "NUM"
    return "other_" + re.sub(r"\d+", "#", s)


def load(images_dir: str, labels_dir: str, nclasses: int):
    items = []
    for p in sorted(glob.glob(os.path.join(images_dir, "*"))):
        if not p.lower().endswith(IMG_EXT):
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        lp = os.path.join(labels_dir, stem + ".txt")
        if not os.path.isfile(lp):
            continue
        boxes = [0] * nclasses
        for ln in open(lp, encoding="utf-8"):
            t = ln.split()
            if len(t) >= 5:
                c = int(float(t[0]))
                if 0 <= c < nclasses:
                    boxes[c] += 1
        items.append({"img": p, "label": lp, "stem": stem,
                      "group": group_key(os.path.basename(p)), "boxes": boxes})
    return items


def optimise(groups, nclasses, val_ratio, min_boxes, iters, restarts, seed):
    keys = sorted(groups)
    gboxes = [[sum(it["boxes"][c] for it in groups[k]) for c in range(nclasses)] for k in keys]
    gsize = [len(groups[k]) for k in keys]
    total = [sum(gb[c] for gb in gboxes) for c in range(nclasses)]
    total_img = sum(gsize)
    n = len(keys)
    TRAIN, VAL = 0, 1

    def err(vb, vimg):
        e = abs(vimg / total_img - val_ratio)
        for c in range(nclasses):
            if total[c]:
                e += abs(vb[c] / total[c] - val_ratio)
                e += 3.0 * max(0, min_boxes - vb[c]) / max(min_boxes, 1)
        return e

    best = None
    for r in range(restarts):
        rng = random.Random(seed + r)
        assign = [TRAIN] * n
        vb = [0] * nclasses
        vimg = 0
        cur = err(vb, vimg)
        lb, la = cur, assign[:]
        t0, t1 = 0.03, 0.0005
        for it in range(iters):
            temp = t0 * (t1 / t0) ** (it / iters)
            i = rng.randrange(n)
            new = 1 - assign[i]
            # 应用
            if new == VAL:
                vimg += gsize[i]
                for c in range(nclasses):
                    vb[c] += gboxes[i][c]
            else:
                vimg -= gsize[i]
                for c in range(nclasses):
                    vb[c] -= gboxes[i][c]
            ne = err(vb, vimg)
            if ne <= cur or rng.random() < math.exp((cur - ne) / max(temp, 1e-9)):
                assign[i] = new
                cur = ne
                if ne < lb:
                    lb, la = ne, assign[:]
            else:
                # 回滚
                if new == VAL:
                    vimg -= gsize[i]
                    for c in range(nclasses):
                        vb[c] -= gboxes[i][c]
                else:
                    vimg += gsize[i]
                    for c in range(nclasses):
                        vb[c] += gboxes[i][c]
        if best is None or lb < best[0]:
            best = (lb, la)
        print(f"[restart {r}] err={lb:.4f} (best={best[0]:.4f})", flush=True)
    return keys, best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-data", default="auto_check_dataset/clean_data")
    ap.add_argument("--classes", default="dataset/Dahua2/classes.txt")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--min-boxes", type=int, default=20, help="val 中每类的最小框数")
    ap.add_argument("--iters", type=int, default=300000)
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = [l.strip() for l in open(args.classes, encoding="utf-8") if l.strip()]
    nc = len(names)
    cd = Path(args.clean_data)
    items = load(str(cd / "images"), str(cd / "labels"), nc)
    groups = {}
    for it in items:
        groups.setdefault(it["group"], []).append(it)
    print(f"images={len(items)} groups={len(groups)} nc={nc}")

    keys, assign = optimise(groups, nc, args.val_ratio, args.min_boxes,
                            args.iters, args.restarts, args.seed)
    split_of = {keys[i]: assign[i] for i in range(len(keys))}
    val_keys = {k for k, v in split_of.items() if v == 1}

    tr = [it for it in items if split_of[it["group"]] == 0]
    va = [it for it in items if split_of[it["group"]] == 1]
    def boxcounts(rows):
        c = [0] * nc
        for it in rows:
            for k in range(nc):
                c[k] += it["boxes"][k]
        return c
    tbc, vbc = boxcounts(tr), boxcounts(va)
    print(f"\nimages: train={len(tr)} val={len(va)}  ({len(va)/(len(tr)+len(va)):.3f} val)")
    print(f"{'class':16s}{'train':>9}{'val':>8}{'val_ratio':>11}{'ok(>=min)':>11}")
    rows = []
    for c in range(nc):
        tot = tbc[c] + vbc[c]
        r = vbc[c] / tot if tot else 0.0
        ok = vbc[c] >= args.min_boxes
        rows.append({"class": names[c], "train": tbc[c], "val": vbc[c],
                     "val_ratio": round(r, 4), "ok_min": ok})
        print(f"{names[c]:16s}{tbc[c]:>9}{vbc[c]:>8}{r:>11.3f}{str(ok):>11}")

    if args.dry_run:
        print("\ndry-run，未写文件")
        return

    # 写出
    for sp in ("train", "valid"):
        shutil.rmtree(cd / sp, ignore_errors=True)
        (cd / sp / "images").mkdir(parents=True, exist_ok=True)
        (cd / sp / "labels").mkdir(parents=True, exist_ok=True)
    shutil.rmtree(cd / "test", ignore_errors=True)  # 9:1 无 test
    for it, sp in [(it, "train") for it in tr] + [(it, "valid") for it in va]:
        shutil.copy2(it["img"], cd / sp / "images" / os.path.basename(it["img"]))
        shutil.copy2(it["label"], cd / sp / "labels" / (it["stem"] + ".txt"))

    (cd / "data.yaml").write_text(
        f"path: {cd.resolve()}\ntrain: train/images\nval: valid/images\n\n"
        f"nc: {nc}\nnames: {names}\n", encoding="utf-8")
    report = {"images": {"train": len(tr), "val": len(va)},
              "val_ratio_target": args.val_ratio, "min_boxes": args.min_boxes,
              "per_class": rows,
              "all_classes_meet_min": all(r["ok_min"] for r in rows)}
    (cd / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    with open(cd / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "group", "split"])
        for it in tr:
            w.writerow([it["stem"], it["group"], "train"])
        for it in va:
            w.writerow([it["stem"], it["group"], "valid"])
    print(f"\n已写出 train/valid + data.yaml + report.json -> {cd.resolve()}")


if __name__ == "__main__":
    main()
