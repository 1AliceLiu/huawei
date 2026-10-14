#!/usr/bin/env python3
"""为「框级审查工具」(review_tool/server.py) 生成待审框条目。

每个待审框 = 一个 <item_id>.json，落在 ``suspect_select/<tag>/items/`` 下。
与旧的按「整图」导出不同，这里以 *框* 为单位，浏览器聚焦单个框，不被同图其它框干扰。

item_id 取 crop 文件名（形如 ``train_<stem>_<box_idx>_c<cls_id>``），
其中 ``box_idx`` 就是源 label 文件中的行号，回写时据此精确定位该框。

用法::

    python auto_check_dataset/review_tool/build_items.py            # 默认 4 个类别
    python auto_check_dataset/review_tool/build_items.py --tags mislabel
    python auto_check_dataset/review_tool/build_items.py --force    # 覆盖已生成条目
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FNAME_RE = re.compile(r"_(\d+)_c(\d+)$")


def abspath(p: str) -> str:
    p = str(p).replace("\\", "/")
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def label_for_image(src: str) -> str | None:
    d = os.path.dirname(src)
    if os.path.basename(d) != "images":
        return None
    stem = os.path.splitext(os.path.basename(src))[0]
    return os.path.join(os.path.dirname(d), "labels", stem + ".txt")


def clean(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return v


def load_decided(clean_dir: str) -> set[str]:
    path = os.path.join(clean_dir, "decisions.csv")
    if not os.path.isfile(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {r["item_id"] for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suspects", default="auto_check_dataset/clean_output/suspects/suspects.csv")
    ap.add_argument("--suspect-root", default="auto_check_dataset/suspect_select")
    ap.add_argument("--clean-dir", default="auto_check_dataset/clean_data")
    ap.add_argument("--tags", nargs="+",
                    default=["mislabel", "duplicate", "outlier_centroid", "outlier_density"])
    ap.add_argument("--force", action="store_true", help="覆盖已存在条目（不影响已判定记录）")
    args = ap.parse_args()

    suspects = abspath(args.suspects)
    root = abspath(args.suspect_root)
    clean_dir = abspath(args.clean_dir)
    decided = load_decided(clean_dir)

    df = pd.read_csv(suspects, low_memory=False)
    print(f"suspects.csv: {len(df)} 行；已判定 {len(decided)} 个框")

    made = skipped = exists = 0
    for tag in args.tags:
        sub = df[df["reasons"].str.contains(tag, regex=False)]
        idir = os.path.join(root, tag, "items")
        os.makedirs(idir, exist_ok=True)
        n_made = 0
        for _, row in sub.iterrows():
            crop = str(row["path"]).replace("\\", "/")
            stem = os.path.splitext(os.path.basename(crop))[0]
            out = os.path.join(idir, stem + ".json")
            if stem in decided:
                skipped += 1
                continue
            if os.path.isfile(out) and not args.force:
                exists += 1
                continue
            m = FNAME_RE.search(stem)
            src = str(row["source_image"]).replace("\\", "/")
            item = {
                "item_id": stem,
                "tag": tag,
                "source_image": abspath(src),
                "source_stem": os.path.splitext(os.path.basename(src))[0],
                "label_path": label_for_image(abspath(src)),
                "box_idx": int(m.group(1)) if m else -1,
                "cls_id": int(row["cls_id"]),
                "cls_name": clean(row["cls_name"]),
                "x1": float(row["x1"]), "y1": float(row["y1"]),
                "x2": float(row["x2"]), "y2": float(row["y2"]),
                "crop": abspath(crop),
                "reasons": clean(row["reasons"]),
                "severity": float(row["severity"]),
                "suggested_action": clean(row["suggested_action"]),
                "relabel_target": clean(row.get("relabel_target")),
                "confidence": clean(row.get("confidence")),
                "nn_consistency": None if pd.isna(row.get("nn_consistency")) else float(row["nn_consistency"]),
                "nearest_class": clean(row.get("nearest_class")),
                "nearest_class_frac": None if pd.isna(row.get("nearest_class_frac")) else float(row["nearest_class_frac"]),
            }
            with open(out, "w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=1)
            made += 1
            n_made += 1
        print(f"[{tag}] 待审框 {len(sub)}，新生成 {n_made} -> {os.path.relpath(idir, REPO)}")

    print(f"\n完成：新生成 {made}，已存在 {exists}，已判定跳过 {skipped}")
    print(f"启动审查工具: python auto_check_dataset/review_tool/server.py")


if __name__ == "__main__":
    main()
