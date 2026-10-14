"""Build a group-disjoint, frozen validation/test split + training pool for the merged12 dataset.

Motivation
----------
The legacy train/valid/test split of ``batch_12.v14i.merged12.yolov11`` is NOT
group-disjoint: 74 groups are shared between train and valid, 74 between train and
test and 59 between valid and test. Consecutive frames / the same camera therefore
leak across splits, which makes the reported mAP optimistic and unusable as a
"measuring stick".

This script recombines all existing images (train + valid + test), derives a source
group (camera id / video segment) from every filename and re-splits **by group**:

* every group is atomic -> it lands entirely in train, valid or test;
* the ``NUM`` group (images whose filename carries no camera metadata) is forced
  into train, so it can never leak into the frozen evaluation sets;
* the frozen val/test are stratified so that all classes (incl. rare ones) appear.

Outputs (under <dataset>/splits by default)
-------------------------------------------
* ``train_pool.txt`` / ``frozen_val.txt`` / ``frozen_test.txt``  - one image path per line
* ``manifest.json`` - per-image group + split + per-class box counts
* ``data_train_pool.yaml`` - YOLO data config: train=train_pool, val=frozen_val, test=frozen_test

Usage
-----
    python scripts/build_unified_val.py
    python scripts/build_unified_val.py --val-size 1250 --test-size 1250 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DS = ROOT / "dataset" / "batch_12.v14i.merged12.yolov11"
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "valid", "test")
NAMES = [
    "Coach", "Franchised Bus", "HGV", "LGV", "Light Bus", "MGV",
    "Motorcycle", "PLB GMB", "Private Car", "Taxi", "Van", "Container",
]


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


def load_items(ds: Path) -> list[dict]:
    items: list[dict] = []
    for split in SPLITS:
        idir = ds / split / "images"
        if not idir.is_dir():
            continue
        for img in sorted(idir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXT:
                continue
            lbl = ds / split / "labels" / (img.stem + ".txt")
            boxes: Counter[int] = Counter()
            if lbl.is_file():
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    p = line.split()
                    if len(p) >= 5:
                        try:
                            boxes[int(float(p[0]))] += 1
                        except ValueError:
                            pass
            items.append({
                "path": str(img.resolve()),
                "group": group_key(img.name),
                "orig_split": split,
                "boxes": dict(boxes),
            })
    return items


def profile_groups(items: list[dict]) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for it in items:
        g = groups.setdefault(it["group"], {"items": [], "boxes": Counter()})
        g["items"].append(it)
        g["boxes"].update(it["boxes"])
    for g in groups.values():
        g["size"] = len(g["items"])
    return groups


def build_frozen_pair(groups: dict[str, dict], nc: int, target_size: int, max_size: int,
                      total_boxes: Counter, seed: int, min_class_boxes: int,
                      forbidden: set[str]) -> tuple[set[str], set[str]]:
    """Build val and test simultaneously so both span many scene groups while leaving
    the training pool with the majority of scenes.

    Groups are atomic. Two stages:

    1. class-coverage pre-pass: rarest classes first, guarantee each frozen split gets
       at least ``min_class_boxes`` boxes of every class by reserving a small group.
    2. randomized round-robin fill: shuffle the remaining groups and assign each to the
       frozen split that is furthest below its size target, skipping groups that would
       overflow. Because the order is random, scene diversity is spread over train /
       val / test instead of val+test hoovering up every small group.

    ``forbidden`` groups (the metadata-less ``NUM`` bucket) are never eligible.
    """
    val: set[str] = set()
    test: set[str] = set()
    have_v: Counter = Counter()
    have_t: Counter = Counter()
    size_v = size_t = 0

    def add(k: str, side: str) -> None:
        nonlocal size_v, size_t
        if side == "v":
            val.add(k)
            have_v.update(groups[k]["boxes"])
            size_v += groups[k]["size"]
        else:
            test.add(k)
            have_t.update(groups[k]["boxes"])
            size_t += groups[k]["size"]

    candidates = [k for k in groups if k not in forbidden and groups[k]["size"] <= max_size]

    # ---- stage 1: guarantee class presence in both frozen splits ----
    for c in sorted(range(nc), key=lambda x: total_boxes[x]):
        for side in ("v", "t"):
            while (have_v if side == "v" else have_t)[c] < min_class_boxes:
                have = have_v if side == "v" else have_t
                size = size_v if side == "v" else size_t
                pool = sorted((k for k in candidates if k not in val | test
                               and groups[k]["boxes"].get(c, 0) > 0
                               and size + groups[k]["size"] <= max_size),
                              key=lambda k: groups[k]["size"])
                if not pool:
                    break
                add(pool[0], side)

    # ---- stage 2: randomized round-robin fill to the size target ----
    rng = random.Random(seed)
    order = candidates[:]
    rng.shuffle(order)
    for k in order:
        if k in val or k in test:
            continue
        s = groups[k]["size"]
        left_v, left_t = target_size - size_v, target_size - size_t
        sides = []
        if left_v > 0 and size_v + s <= max_size:
            sides.append("v")
        if left_t > 0 and size_t + s <= max_size:
            sides.append("t")
        if not sides:
            continue
        if len(sides) == 2:
            side = "v" if left_v >= left_t else "t"
        else:
            side = sides[0]
        add(k, side)
    return val, test


def write_list(path: Path, keys: set[str], groups: dict[str, dict]) -> int:
    imgs = [it for k in sorted(keys) for it in groups[k]["items"]]
    imgs.sort(key=lambda it: it["path"])
    path.write_text("\n".join(it["path"] for it in imgs) + "\n", encoding="utf-8")
    return len(imgs)


def class_report(title: str, groups: dict[str, dict], keys: set[str]) -> Counter:
    boxes = Counter()
    n = 0
    for k in keys:
        n += groups[k]["size"]
        boxes.update(groups[k]["boxes"])
    print(f"\n[{title}] images={n} groups={len(keys)} boxes={sum(boxes.values())}")
    for i, name in enumerate(NAMES):
        print(f"    {i:>2} {name:<15} {boxes[i]:>7}")
    return boxes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DS)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--val-size", type=int, default=1250, help="target size for each frozen split")
    ap.add_argument("--max-size", type=int, default=1500, help="hard cap per frozen split / max group size")
    ap.add_argument("--min-class-boxes", type=int, default=20, help="minimum boxes per class in each frozen split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = args.dataset.resolve()
    out = (args.out_dir or (ds / "splits")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    items = load_items(ds)
    groups = profile_groups(items)
    total_boxes = Counter()
    for g in groups.values():
        total_boxes.update(g["boxes"])

    if "NUM" not in groups:
        print("[warn] no NUM group found")

    forbidden = {"NUM"}
    val_keys, test_keys = build_frozen_pair(
        groups, len(NAMES), args.val_size, args.max_size, total_boxes,
        args.seed, args.min_class_boxes, forbidden)
    train_keys = set(groups) - val_keys - test_keys

    overlap = (val_keys & test_keys) | (val_keys & train_keys) | (test_keys & train_keys)
    if overlap:
        raise SystemExit(f"group leakage detected: {sorted(overlap)[:10]} (n={len(overlap)})")

    n_val = write_list(out / "frozen_val.txt", val_keys, groups)
    n_test = write_list(out / "frozen_test.txt", test_keys, groups)
    n_train = write_list(out / "train_pool.txt", train_keys, groups)

    manifest = {
        "dataset": str(ds),
        "seed": args.seed,
        "num_groups": len(groups),
        "images": {
            "train_pool": n_train, "frozen_val": n_val, "frozen_test": n_test,
            "total": n_train + n_val + n_test,
        },
        "items": items,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    names = "[" + ", ".join(f"'{n}'" for n in NAMES) + "]"
    (out / "data_train_pool.yaml").write_text(
        "# train = full group-disjoint training pool, val/test = frozen evaluation sets\n"
        "# regenerate with scripts/build_unified_val.py\n"
        f"path: {ds}\n"
        f"train: {out / 'train_pool.txt'}\n"
        f"val: {out / 'frozen_val.txt'}\n"
        f"test: {out / 'frozen_test.txt'}\n\n"
        f"nc: {len(NAMES)}\nnames: {names}\n",
        encoding="utf-8",
    )

    print(f"dataset   : {ds}")
    print(f"out dir   : {out}")
    print(f"groups    : {len(groups)} (NUM forced to train)")
    print(f"images    : train_pool={n_train}  frozen_val={n_val}  frozen_test={n_test}")
    class_report("frozen_val", groups, val_keys)
    class_report("frozen_test", groups, test_keys)
    class_report("train_pool", groups, train_keys)
    missing_val = [NAMES[c] for c in range(len(NAMES)) if not any(groups[k]["boxes"][c] for k in val_keys)]
    missing_test = [NAMES[c] for c in range(len(NAMES)) if not any(groups[k]["boxes"][c] for k in test_keys)]
    print(f"\nclass missing in val : {missing_val or 'none'}")
    print(f"class missing in test: {missing_test or 'none'}")
    print(f"group leakage        : none")


if __name__ == "__main__":
    main()
