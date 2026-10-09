"""Append a reviewed (or explicitly unreviewed) pending batch into v14i in place.

Rules (same as APPEND_r2..r9 / build_merged12.py):
- X-AnyLabeling JSON -> YOLO txt, taxonomy = merged12 12-class order
- export dedup: same-class IoU>=0.85 keep higher conf (attributes.conf)
- split: by-cam chronological contiguous 80/10/10 (n<10 -> all train)
- collision: abort if target stem already exists in dataset
- writes APPEND_<tag>.md + <tag>_files.txt (per-split stems, for wholesale revert)

Usage:
  uv run --project . --no-sync python scripts/append_pending.py --pending tmp/r10_pending --tag r10_1005
"""
import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NAMES = ["Coach", "Franchised Bus", "HGV", "LGV", "Light Bus", "MGV",
         "Motorcycle", "PLB GMB", "Private Car", "Taxi", "Van", "Container"]
NAME2ID = {n: i for i, n in enumerate(NAMES)}
SPLITS = ["train", "valid", "test"]


def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-9)


def parse_xany(jp: Path):
    """-> (boxes, w, h); boxes = [(cls_id, xc, yc, bw, bh, conf)] normalized."""
    d = json.loads(jp.read_text(encoding="utf-8"))
    w, h = int(d["imageWidth"]), int(d["imageHeight"])
    boxes = []
    for s in d.get("shapes", []):
        if s.get("shape_type", "rectangle") != "rectangle":
            raise ValueError(f"{jp.name}: unsupported shape_type")
        label = s["label"].strip()
        if label not in NAME2ID:
            raise ValueError(f"{jp.name}: unknown label={label!r}")
        (x1, y1), (x2, y2) = ((float(v) for v in p) for p in s["points"])
        x1, x2 = sorted((max(0.0, min(float(w), x1)), max(0.0, min(float(w), x2))))
        y1, y2 = sorted((max(0.0, min(float(h), y1)), max(0.0, min(float(h), y2))))
        if x2 - x1 <= 1e-6 or y2 - y1 <= 1e-6:
            continue
        conf = float((s.get("attributes") or {}).get("conf", -1.0))
        boxes.append((NAME2ID[label],
                      (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h,
                      conf, (x1, y1, x2, y2)))
    return boxes, w, h


def dedup(boxes, iou_thresh=0.85):
    by_cls = defaultdict(list)
    for b in boxes:
        by_cls[b[0]].append(b)
    out, removed = [], 0
    for items in by_cls.values():
        items.sort(key=lambda b: -b[5])
        kept = []
        for b in items:
            if all(box_iou(b[6], k[6]) < iou_thresh for k in kept):
                kept.append(b)
            else:
                removed += 1
        out.extend(kept)
    return out, removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pending", required=True)
    ap.add_argument("--dataset", default="dataset/batch_12.v14i.merged12.yolov11")
    ap.add_argument("--tag", required=True, help="e.g. r10_1005")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    pend = ROOT / args.pending
    ds = ROOT / args.dataset
    assert (pend / "images").is_dir() and (pend / "annotations_xany").is_dir(), pend
    assert (ds / "train" / "images").is_dir(), ds

    # existing stems (collision guard)
    existing = set()
    for split in SPLITS:
        existing.update(p.stem for p in (ds / split / "images").glob("*") if p.is_file())

    # parse + dedup
    items = {}  # stem -> (split-cam, boxes_yolo, img_src)
    total_boxes_raw, dedup_removed = 0, 0
    for jp in sorted((pend / "annotations_xany").glob("*.json")):
        stem = jp.stem
        img_src = pend / "images" / (stem + ".jpg")
        if not img_src.is_file():
            raise SystemExit(f"missing image for {jp.name}")
        if stem in existing:
            raise SystemExit(f"collision: {stem} already in dataset, abort")
        boxes, _, _ = parse_xany(jp)
        total_boxes_raw += len(boxes)
        boxes, n_rm = dedup(boxes)
        dedup_removed += n_rm
        cam = stem.rsplit("_", 1)[1] if "_" in stem else "unknown"
        items[stem] = (cam, [(c, x, y, w, h) for c, x, y, w, h, _, _ in boxes], img_src)

    # by-cam chronological 80/10/10
    by_cam = defaultdict(list)
    for stem, (cam, _, _) in items.items():
        by_cam[cam].append(stem)
    assign = {}
    for cam, lst in by_cam.items():
        lst.sort()
        n = len(lst)
        n_val = n_test = max(1, round(n * 0.1)) if n >= 10 else 0
        i_test = n - n_test
        i_val = i_test - n_val
        for i, s in enumerate(lst):
            assign[s] = "train" if i < i_val else ("valid" if i < i_test else "test")

    # write in place
    counts = {s: Counter() for s in SPLITS}
    nimgs = Counter()
    appended = []
    for stem, split in sorted(assign.items()):
        _, boxes, img_src = items[stem]
        shutil.copy2(img_src, ds / split / "images" / (stem + ".jpg"))
        lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes]
        (ds / split / "labels" / (stem + ".txt")).write_text(
            ("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        nimgs[split] += 1
        for c, *_ in boxes:
            counts[split][c] += 1
        appended.append(f"{split}\t{stem}")

    total_boxes = {c: sum(counts[s][c] for s in SPLITS) for c in range(len(NAMES))}
    (ds / f"{args.tag}_files.txt").write_text("\n".join(appended) + "\n", encoding="utf-8")
    detail = ", ".join(f"{NAMES[c]}={total_boxes[c]}" for c in range(len(NAMES)))
    (ds / f"APPEND_{args.tag}.md").write_text(
        f"# Append {args.pending} ({args.note or 'pending batch'}) -> v14i (in place)\n\n"
        f"- date: {date.today().isoformat()}, collected {len(items)} (0956 prelabel, UNREVIEWED), "
        f"merged {len(items)}, raw boxes {total_boxes_raw}\n"
        f"- split added: train={nimgs['train']} valid={nimgs['valid']} test={nimgs['test']} "
        f"(by-cam chronological 80/10/10, same rule as build_merged12.py)\n"
        f"- class boxes: {{{', '.join(f'{c}: {total_boxes[c]}' for c in range(len(NAMES)))}}}\n"
        f"  ({detail})\n"
        f"- export dedup(IoU>=0.85) removed: {dedup_removed}\n"
        f"- WARNING: prelabel boxes merged WITHOUT manual review (user decision 2026-10-05); "
        f"revert via {args.tag}_files.txt\n",
        encoding="utf-8")

    print(f"merged {len(items)} images -> {ds}")
    print(f"split: " + " ".join(f"{s}={nimgs[s]}" for s in SPLITS))
    print(f"raw boxes={total_boxes_raw} dedup_removed={dedup_removed} kept={sum(total_boxes.values())}")
    print(f"{'id':>2} {'name':<15} {'train':>6} {'valid':>6} {'test':>6}")
    for i, n in enumerate(NAMES):
        print(f"{i:>2} {n:<15} {counts['train'][i]:>6} {counts['valid'][i]:>6} {counts['test'][i]:>6}")


if __name__ == "__main__":
    main()
