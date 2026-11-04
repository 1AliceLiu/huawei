"""Audit YOLO label files for duplicate / conflicting annotations.

Scans every `labels/` directory under a dataset root (train/valid/test and any
top-level images+labels layout), and flags:

  duplicate_exact   same class AND identical box appears twice            (definite bug)
  box_multilabel    identical box carries two DIFFERENT classes           (definite bug)
  class_conflict    different classes, box IoU >= --conflict-iou          (suspicious)
  near_dup_same     same class, box IoU >= --dup-iou but not identical    (suspicious)
  invalid_format    line is not `cls x y w h`
  invalid_class     class id not an int in [0, nc)
  invalid_range     coords outside [0,1] or w/h <= 0

Usage:
    python scripts/audit_labels.py                       # report only, all datasets
    python scripts/audit_labels.py --copy                # also copy offending files
    python scripts/audit_labels.py --datasets clean_part_batch_12.v8i.merged12.yolov11
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
EPS = 1e-6


def dataset_names(ds: Path) -> list[str] | None:
    y = ds / "data.yaml"
    if not y.is_file():
        return None
    try:
        doc = yaml.safe_load(y.read_text())
        n = doc.get("names")
        if isinstance(n, dict):
            return [n[k] for k in sorted(n)]
        return list(n) if n else None
    except Exception:
        return None


def parse_label(path: Path):
    """Return (boxes, issues). boxes: list of (cls, cx, cy, w, h)."""
    boxes, issues = [], []
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        p = s.split()
        if len(p) != 5:
            issues.append(("invalid_format", ln, s))
            continue
        try:
            cls_f = float(p[0])
            vals = [float(v) for v in p[1:5]]
        except ValueError:
            issues.append(("invalid_format", ln, s))
            continue
        if not float(cls_f).is_integer():
            issues.append(("invalid_class", ln, s))
            continue
        cls = int(cls_f)
        cx, cy, w, h = vals
        if any(v < -EPS or v > 1 + EPS for v in (cx, cy)) or w <= 0 or h <= 0 or w > 1 + EPS or h > 1 + EPS:
            issues.append(("invalid_range", ln, s))
        boxes.append((cls, cx, cy, w, h))
    return boxes, issues


def xyxy(b):
    _, cx, cy, w, h = b
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a, b):
    ax1, ay1, ax2, ay2 = xyxy(a)
    bx1, by1, bx2, by2 = xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def same_box(a, b) -> bool:
    return all(abs(x - y) <= EPS for x, y in zip(a[1:], b[1:]))


def find_labels_dirs(ds: Path) -> list[Path]:
    return sorted(p for p in ds.rglob("labels") if p.is_dir())


def scan_file(path: Path, nc: int | None, conflict_iou: float, dup_iou: float):
    boxes, issues = parse_label(path)
    found: dict[str, list] = defaultdict(list)
    for typ, ln, s in issues:
        if typ == "invalid_class" and nc is not None:
            try:
                c = int(float(s.split()[0]))
                if 0 <= c < nc:
                    continue
            except Exception:
                pass
        found[typ].append({"line": ln, "text": s})
    for i in range(len(boxes)):
        if nc is not None and not (0 <= boxes[i][0] < nc):
            continue
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if nc is not None and not (0 <= b[0] < nc):
                continue
            if same_box(a, b):
                if a[0] == b[0]:
                    found["duplicate_exact"].append({"lines": [i + 1, j + 1], "box": list(a)})
                else:
                    found["box_multilabel"].append({"lines": [i + 1, j + 1], "a": list(a), "b": list(b)})
                continue
            iou_v = iou(a, b)
            if a[0] != b[0] and iou_v >= conflict_iou:
                found["class_conflict"].append({"lines": [i + 1, j + 1], "iou": round(iou_v, 4), "a": list(a), "b": list(b)})
            elif a[0] == b[0] and iou_v >= dup_iou:
                found["near_dup_same"].append({"lines": [i + 1, j + 1], "iou": round(iou_v, 4), "box": list(a)})
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "dataset"))
    ap.add_argument("--out", default=str(ROOT / "temp" / "error-label"))
    ap.add_argument("--datasets", nargs="*", default=None, help="subset of dataset dir names")
    ap.add_argument("--conflict-iou", type=float, default=0.9)
    ap.add_argument("--dup-iou", type=float, default=0.95)
    ap.add_argument("--copy", action="store_true", help="copy offending label (+ image) files")
    ap.add_argument("--copy-images", action="store_true", help="also copy the matching image")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    ds_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if args.datasets:
        ds_dirs = [d for d in ds_dirs if d.name in set(args.datasets)]

    report: dict = {"settings": vars(args), "datasets": {}}
    total = defaultdict(int)
    for ds in ds_dirs:
        names = dataset_names(ds)
        nc = len(names) if names else None
        lbl_dirs = find_labels_dirs(ds)
        if not lbl_dirs:
            continue
        ds_rep = {"nc": nc, "names": names, "label_dirs": [str(d.relative_to(root)) for d in lbl_dirs], "files": {}, "counts": defaultdict(int)}
        for ldir in lbl_dirs:
            split = ldir.parent.relative_to(ds).as_posix() or "." if ldir.parent != ds else "."
            for lf in sorted(ldir.glob("*.txt")):
                found = scan_file(lf, nc, args.conflict_iou, args.dup_iou)
                if not found:
                    continue
                rel = lf.relative_to(ds).as_posix()
                ds_rep["files"][rel] = {k: v for k, v in found.items()}
                for k, v in found.items():
                    ds_rep["counts"][k] += len(v)
                    total[k] += len(v)
                if args.copy:
                    for k in found:
                        dest_dir = out / ds.name / k / split
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(lf, dest_dir / f"{split}_{lf.name}")
                        if args.copy_images:
                            img_dir = ldir.parent / "images"
                            for ext in IMAGE_EXT:
                                img = img_dir / f"{lf.stem}{ext}"
                                if img.is_file():
                                    shutil.copy2(img, dest_dir / f"{split}_{img.name}")
                                    break
        ds_rep["counts"] = dict(ds_rep["counts"])
        if ds_rep["files"]:
            report["datasets"][ds.name] = ds_rep

    report["totals"] = dict(total)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = ["LABEL ANOMALY REPORT", "=" * 60]
    for ds_name, ds_rep in report["datasets"].items():
        lines.append(f"\n[{ds_name}] files_with_issues={len(ds_rep['files'])}  {ds_rep['counts']}")
        for rel, found in list(ds_rep["files"].items()):
            lines.append(f"  {rel}: { {k: len(v) for k, v in found.items()} }")
    lines.append("\nTOTALS: " + str(report["totals"]))
    (out / "report.txt").write_text("\n".join(lines))

    print(f"[done] datasets with issues: {len(report['datasets'])}")
    print("[totals]", json.dumps(report["totals"], ensure_ascii=False))
    for ds_name, ds_rep in report["datasets"].items():
        print(f"  {ds_name}: files={len(ds_rep['files'])} counts={ds_rep['counts']}")
    print(f"[out] {out / 'report.json'}\n[out] {out / 'report.txt'}")


if __name__ == "__main__":
    main()
