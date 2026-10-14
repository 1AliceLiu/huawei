#!/usr/bin/env python3
"""稀有类过采样 + copy-paste 增强（写入训练集）。

针对 HGV / LGV / Motorcycle / Container 等稀有类：
1. **过采样**：含稀有类的训练图，按频率复制若干份（文件名加 `__osN` 后缀）；
2. **copy-paste**：从训练集里裁剪稀有类目标（bbox 紧裁剪），随机贴到其它训练图上，
   生成新的 image+label（文件名加 `__cpN` 后缀），提升稀有类出现密度。

只改 train，不动 valid；生成后 train 的 images 与 labels 仍一一对应。

用法::

    python scripts/augment_rare.py --dry-run
    python scripts/augment_rare.py --cp-num 800
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import shutil

import cv2
import numpy as np


def load_names(path):
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]


def read_label(p):
    out = []
    for ln in open(p, encoding="utf-8"):
        t = ln.split()
        if len(t) >= 5:
            out.append([float(x) for x in t[:5]])
    return out


def to_xyxy(box, W, H):
    c, xc, yc, w, h = box
    return int((xc - w / 2) * W), int((yc - h / 2) * H), int((xc + w / 2) * W), int((yc + h / 2) * H)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="auto_check_dataset/clean_data")
    ap.add_argument("--classes", default="dataset/Dahua2/classes.txt")
    ap.add_argument("--rare", default="HGV,LGV,Motorcycle,Container")
    ap.add_argument("--os-factors", default="Container:3,Motorcycle:2,HGV:2,LGV:2",
                    help="每类过采样倍数，或 omitted")
    ap.add_argument("--cp-num", type=int, default=800, help="copy-paste 生成图片数")
    ap.add_argument("--cp-per", type=int, default=3, help="每张最多贴几个稀有目标")
    ap.add_argument("--min-crop", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = load_names(args.classes)
    rare = {names.index(x): x for x in args.rare.split(",")}
    factors = {}
    for kv in args.os_factors.split(","):
        if ":" in kv:
            k, v = kv.split(":")
            if k in names:
                factors[names.index(k)] = int(v)
    rng = random.Random(args.seed)
    D = args.data
    trI, trL = f"{D}/train/images", f"{D}/train/labels"
    imgs = sorted(glob.glob(trI + "/*.jpg"))
    # 排除已生成的增强图，避免二次增强
    base = [p for p in imgs if "__os" not in os.path.basename(p) and "__cp" not in os.path.basename(p)]
    print(f"train 原始图 {len(base)}，rare={list(rare.values())}, os_factors={factors}")

    # ---------- 1) 过采样 ----------
    os_add = 0
    for p in base:
        stem = os.path.splitext(os.path.basename(p))[0]
        boxes = read_label(f"{trL}/{stem}.txt")
        present = {int(b[0]) for b in boxes} & set(rare)
        if not present:
            continue
        f = max(factors.get(c, 1) for c in present)
        if f <= 1:
            continue
        for k in range(1, f):
            if args.dry_run:
                os_add += 1
                continue
            shutil.copy2(p, f"{trI}/{stem}__os{k}.jpg")
            shutil.copy2(f"{trL}/{stem}.txt", f"{trL}/{stem}__os{k}.txt")
            os_add += 1
    print(f"[过采样] 新增 {os_add} 张")

    # ---------- 2) copy-paste ----------
    # 收集稀有类 crop
    crops = []  # (patch, class)
    for p in base:
        stem = os.path.splitext(os.path.basename(p))[0]
        boxes = read_label(f"{trL}/{stem}.txt")
        if not any(int(b[0]) in rare for b in boxes):
            continue
        im = cv2.imread(p)
        if im is None:
            continue
        H, W = im.shape[:2]
        for b in boxes:
            c = int(b[0])
            if c not in rare:
                continue
            x1, y1, x2, y2 = to_xyxy(b, W, H)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < args.min_crop or y2 - y1 < args.min_crop:
                continue
            crops.append((im[y1:y2, x1:x2].copy(), c))
    print(f"[copy-paste] 可用稀有 crop: {len(crops)}")
    if not crops:
        print("无稀有 crop，跳过 copy-paste")
        return

    from collections import Counter
    cnt = Counter(c for _, c in crops)
    weights = [1.0 / cnt[c] for _, c in crops]
    cp_add = 0
    for i in range(args.cp_num):
        bg = rng.choice(base)
        stem = os.path.splitext(os.path.basename(bg))[0]
        bgboxes = read_label(f"{trL}/{stem}.txt")
        im = cv2.imread(bg)
        if im is None:
            continue
        H, W = im.shape[:2]
        occ = [to_xyxy(b, W, H) for b in bgboxes]
        new = []
        for _ in range(rng.randint(1, args.cp_per)):
            patch, c = rng.choices(crops, weights=weights, k=1)[0]
            ph, pw = patch.shape[:2]
            s = rng.uniform(0.8, 1.2)
            nw, nh = max(args.min_crop, int(pw * s)), max(args.min_crop, int(ph * s))
            lo_y, hi_y = int(0.25 * H), H - nh
            if nw >= W or nh >= H or hi_y <= lo_y:
                continue
            px = rng.randint(0, W - nw)
            py = rng.randint(lo_y, hi_y)  # 偏下半部（路面）
            box = (px, py, px + nw, py + nh)
            if any(iou(box, o) > 0.15 for o in occ + [n[2] for n in new]):
                continue
            patch_r = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
            im[py:py + nh, px:px + nw] = patch_r
            xc = (px + nw / 2) / W
            yc = (py + nh / 2) / H
            new.append((c, [c, xc, yc, nw / W, nh / H], box))
        if not new:
            continue
        if args.dry_run:
            cp_add += 1
            continue
        lines = [" ".join(str(v) for v in b) for b in bgboxes] + \
                [" ".join(str(v) for v in n[1]) for n in new]
        out_stem = f"{stem}__cp{i:04d}"
        cv2.imwrite(f"{trI}/{out_stem}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        open(f"{trL}/{out_stem}.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
        cp_add += 1
    print(f"[copy-paste] 新增 {cp_add} 张")

    if not args.dry_run:
        i = len(glob.glob(trI + "/*.jpg"))
        l = len(glob.glob(trL + "/*.txt"))
        print(f"train 现有 images={i} labels={l} one-to-one={i == l}")


if __name__ == "__main__":
    main()
