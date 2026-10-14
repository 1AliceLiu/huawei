#!/usr/bin/env python3
"""批量去除 clean_data 图片左上角时间戳 / 左下角编号（就地覆盖）。

难点：白色文字常压在明亮背景上，单图绝对阈值不可靠。这里用
**同一相机多帧投票**：同一路视频里 OSD 位置/字体固定，把多帧的
"白色带描边"文字掩膜做并集，就能稳定框出整行 OSD，再对整行矩形
做 inpaint（保证无残留）。帧数不足的相机退回单图检测。

用法::

    python scripts/remove_osd_clean.py --dry-run
    python scripts/remove_osd_clean.py --root auto_check_dataset/clean_data
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REGIONS = {"TL": (0.0, 0.0, 0.34, 0.085), "BL": (0.0, 0.885, 0.34, 1.0)}


def group_key(filename: str) -> str:
    s = Path(filename).stem
    s = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", s, flags=re.I)
    s = re.sub(r"\.rf\.[0-9a-f]+$", "", s, flags=re.I)
    s = re.sub(r"__(os|cp)\d+$", "", s)  # 增强图与底图同组
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


def text_mask(im: np.ndarray, thr: int, dark: int) -> np.ndarray:
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    darkm = cv2.dilate((g < dark).astype(np.uint8), np.ones((3, 3), np.uint8), 1)
    return (((g > thr) & (darkm > 0)).astype(np.uint8) * 255)


def rects_from_union(union: np.ndarray, W: int, H: int, region, pad_ratio=0.6):
    x0, y0, x1, y1 = (int(region[0] * W), int(region[1] * H),
                      int(region[2] * W), int(region[3] * H))
    sub = union[y0:y1, x0:x1]
    j = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((1, 31), np.uint8))
    cnts, _ = cv2.findContours(j, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw > 0.05 * W and 0.006 * H < bh < 0.09 * H:
            py = max(2, int(pad_ratio * bh))
            out.append((max(0, x0 + bx - 5), max(0, y0 + by - py),
                        min(W, x0 + bx + bw + 5), min(H, y0 + by + bh + py)))
    return out


def single_rects(im, thr, dark):
    H, W = im.shape[:2]
    tm = text_mask(im, thr, dark)
    out = []
    for reg in REGIONS.values():
        out += rects_from_union(tm, W, H, reg)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="auto_check_dataset/clean_data")
    ap.add_argument("--corners", nargs="+", default=["TL", "BL"], choices=["TL", "TR", "BL", "BR"])
    ap.add_argument("--thr", type=int, default=165)
    ap.add_argument("--dark", type=int, default=115)
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--max-sample", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    regions = {k: REGIONS[k] for k in args.corners}
    imgs = [p for p in glob.glob(f"{args.root}/**/*.jpg", recursive=True)]
    print(f"发现图片 {len(imgs)}")

    # 按 (group, 分辨率) 分组
    groups = defaultdict(list)
    for p in imgs:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        groups[(group_key(os.path.basename(p)), im.shape)].append(p)
    print(f"组数 {len(groups)}")

    # 每组算 OSD 矩形
    rects_by_group = {}
    multi = single = 0
    for key, paths in groups.items():
        gid, shape = key
        sample = paths if len(paths) <= args.max_sample else \
            [paths[i] for i in np.linspace(0, len(paths) - 1, args.max_sample).astype(int)]
        if len(paths) >= args.min_frames:
            H, W = shape
            union = np.zeros((H, W), np.uint8)
            for p in sample:
                im = cv2.imread(p)
                if im is None:
                    continue
                union = cv2.bitwise_or(union, text_mask(im, args.thr, args.dark))
            rects = []
            for reg in regions.values():
                rects += rects_from_union(union, W, H, reg)
            multi += 1
        else:
            im = cv2.imread(paths[0])
            rects = []
            for reg in regions.values():
                tm = text_mask(im, args.thr, args.dark)
                rects += rects_from_union(tm, im.shape[1], im.shape[0], reg)
            single += 1
        rects_by_group[key] = rects
    print(f"多帧投票组 {multi}，单图回退组 {single}")

    if args.dry_run:
        hit = sum(1 for r in rects_by_group.values() if r)
        print(f"命中 OSD 的组 {hit}/{len(rects_by_group)}（dry-run 未写）")
        return

    done = removed = 0
    for key, paths in groups.items():
        rects = rects_by_group.get(key, [])
        for p in paths:
            im = cv2.imread(p)
            if im is None:
                continue
            if rects:
                m = np.zeros(im.shape[:2], np.uint8)
                for (a, b, c, d) in rects:
                    m[b:d, a:c] = 255
                im = cv2.inpaint(im, m, 6, cv2.INPAINT_NS)
                removed += 1
            cv2.imwrite(p, im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            done += 1
    print(f"处理 {done} 张，其中做了去 OSD {removed} 张")


if __name__ == "__main__":
    main()
