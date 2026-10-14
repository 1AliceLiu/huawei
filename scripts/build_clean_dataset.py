#!/usr/bin/env python3
"""从 clean_data + high_anhor 构建 group-disjoint 的 train/valid/test 数据集。

用途
----
把审查后的 ``auto_check_dataset/clean_data``（已修正）与 ``temp/high_anhor``
（全 keep 的好图）合并，按 **相机/视频组（group-disjoint）** 划分，避免同一路
视频的相近帧跨 split 造成泄漏，同时尽量命中给定的图片数量比例（默认 7:2:1）。

输出::

    <out>/
      data.yaml
      train/images  train/labels
      valid/images  valid/labels
      test/images   test/labels
      split_manifest.csv        # 每张图来自哪个来源、属于哪个 group、落到哪个 split

用法::

    # 审查结束后重跑（会清空 out 后重建）
    python scripts/build_clean_dataset.py

    # 自定义比例 / 只划分 clean_data / 先干跑看比例
    python scripts/build_clean_dataset.py --train 0.8 --val 0.1 --test 0.1
    python scripts/build_clean_dataset.py --no-high-anhor
    python scripts/build_clean_dataset.py --dry-run
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import re
import shutil
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def group_key(filename: str) -> str:
    """Derive a stable source group (camera / video segment) from an image filename."""
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


def discover(source: str, img_dir: str, label_dir: str) -> list[dict]:
    items = []
    for p in sorted(glob.glob(os.path.join(img_dir, "*"))):
        if not p.lower().endswith(IMG_EXT):
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        lp = os.path.join(label_dir, stem + ".txt")
        items.append({"source": source, "img": p, "label": lp, "stem": stem,
                      "group": group_key(os.path.basename(p))})
    return items


def assign(groups: dict[str, list], ratio: dict[str, float], seed: int = 0) -> dict[str, str]:
    """group-disjoint 贪心：每个 group 整体落入一个 split，尽量贴合比例。"""
    total = sum(len(g["items"]) for g in groups.values())
    targets = {k: ratio[k] * total for k in ratio}
    cur = {k: 0 for k in ratio}
    out = {}
    # 大的 group 先放，减少比例漂移；名称做 tie-break 保证确定性
    for gid in sorted(groups, key=lambda g: (-len(groups[g]["items"]), g)):
        size = len(groups[gid]["items"])
        best = max(cur, key=lambda k: (targets[k] - cur[k], -cur[k], k))
        out[gid] = best
        cur[best] += size
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-data", default="auto_check_dataset/clean_data")
    ap.add_argument("--high-anhor", default="temp/high_anhor")
    ap.add_argument("--no-high-anhor", action="store_true")
    ap.add_argument("--out", default="dataset/clean_batch_12.v8i.merged12.yolov11")
    ap.add_argument("--classes", default="dataset/Dahua2/classes.txt")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--no-group", action="store_true", help="不做 group-disjoint，按图片随机")
    ap.add_argument("--dry-run", action="store_true", help="只打印比例，不写文件")
    args = ap.parse_args()

    ratio = {"train": args.train, "valid": args.val, "test": args.test}
    assert abs(sum(ratio.values()) - 1.0) < 1e-6, "比例之和必须为 1"

    items = discover("clean_data", f"{args.clean_data}/images", f"{args.clean_data}/labels")
    if not args.no_high_anhor and os.path.isdir(f"{args.high_anhor}/images"):
        items += discover("high_anhor", f"{args.high_anhor}/images", f"{args.high_anhor}/labels")
    if not items:
        raise SystemExit("未发现任何图片，请检查 --clean-data / --high-anhor")

    groups = collections.defaultdict(lambda: {"items": []})
    for it in items:
        gid = it["stem"] if args.no_group else it["group"]
        groups[gid]["items"].append(it)

    split_of = assign(groups, ratio)
    for it in items:
        gid = it["stem"] if args.no_group else it["group"]
        it["split"] = split_of[gid]

    cnt = collections.Counter(it["split"] for it in items)
    per_src = collections.Counter((it["source"], it["split"]) for it in items)
    total = len(items)
    print(f"总图数 {total}，组数 {len(groups)}"
          + ("" if args.no_group else "（group-disjoint）"))
    print("  图片数:", dict(cnt),
          " 比例:", {k: round(cnt[k] / total, 3) for k in ratio})
    for s in sorted({it["source"] for it in items}):
        print(f"  {s}: " + " ".join(f"{k}={per_src[(s, k)]}" for k in ratio))
    missing = [it["stem"] for it in items if not os.path.isfile(it["label"])]
    print(f"  缺失 label: {len(missing)}")

    if args.dry_run:
        print("dry-run，未写文件")
        return

    out = Path(args.out)
    if out.exists():
        for sp in ratio:
            shutil.rmtree(out / sp, ignore_errors=True)
    for sp in ratio:
        (out / sp / "images").mkdir(parents=True, exist_ok=True)
        (out / sp / "labels").mkdir(parents=True, exist_ok=True)

    copied = collections.Counter()
    for it in items:
        sp = it["split"]
        if not (os.path.isfile(it["img"]) and os.path.isfile(it["label"])):
            continue
        shutil.copy2(it["img"], out / sp / "images" / os.path.basename(it["img"]))
        shutil.copy2(it["label"], out / sp / "labels" / (it["stem"] + ".txt"))
        copied[sp] += 1

    names = [l.strip() for l in open(args.classes, encoding="utf-8") if l.strip()]
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
        f"train: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(names)}\nnames: {names}\n", encoding="utf-8")

    with open(out / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "source", "group", "split"])
        w.writeheader()
        for it in sorted(items, key=lambda x: (x["split"], x["source"], x["stem"])):
            w.writerow({"stem": it["stem"], "source": it["source"],
                        "group": it["group"], "split": it["split"]})

    print("已写出:", dict(copied), "->", out.resolve())


if __name__ == "__main__":
    main()
