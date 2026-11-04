"""Apply hand-edited X-AnyLabeling JSONs (from temp/error-label.tar) back into a YOLO dataset.

The tar holds  error-label/<dataset>/<type>/<split>/<split>_<stem>.{jpg,json,txt}.
Only the .json files carry the human edits (the .txt are untouched). For every
unique (split, stem) we pick the JSON with the FEWEST shapes among its copies
(resolves images that appear under several anomaly types), convert shapes -> YOLO
txt, back up the original label, and overwrite dataset/<split>/labels/<stem>.txt.

Usage:
    python scripts/apply_error_label_jsons.py
    python scripts/apply_error_label_jsons.py --tar temp/error-label.tar --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAR = ROOT / "temp/error-label.tar"
DEFAULT_DATASET = ROOT / "dataset/clean_part_batch_12.v8i.merged12.yolov11"
DEFAULT_BACKUP = ROOT / "temp/error-label-backup"
SPLITS = {"train", "valid", "test"}


def load_names(dataset: Path) -> list[str]:
    doc = yaml.safe_load((dataset / "data.yaml").read_text())
    names = doc["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(n) for n in names]


def shapes_to_txt(payload: dict, name2id: dict[str, int], W: int, H: int) -> str:
    lines = []
    for s in payload.get("shapes", []):
        pts = s.get("points", [])
        if len(pts) < 2:
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        cx = ((x1 + x2) / 2) / W
        cy = ((y1 + y2) / 2) / H
        w = (x2 - x1) / W
        h = (y2 - y1) / H
        cid = name2id.get(str(s.get("label")), -1)
        if cid < 0:
            raise ValueError(f"unknown label {s.get('label')!r}")
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", default=str(DEFAULT_TAR))
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--backup", default=str(DEFAULT_BACKUP))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    backup = Path(args.backup)
    names = load_names(dataset)
    name2id = {n: i for i, n in enumerate(names)}
    print(f"[info] classes = {names}")

    tmp = Path(tempfile.mkdtemp(prefix="el_apply_"))
    with tarfile.open(args.tar) as tf:
        tf.extractall(tmp)
    jsons = sorted(tmp.rglob("*.json"))

    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for jp in jsons:
        split = jp.parent.name
        if split not in SPLITS:
            continue
        stem = jp.stem[len(split) + 1:] if jp.stem.startswith(split + "_") else jp.stem
        groups[(split, stem)].append(jp)

    written = skipped = conflicts = 0
    missing_img = []
    rows = []
    for (split, stem), paths in sorted(groups.items()):
        if len(paths) > 1:
            conflicts += 1
        # pick the copy with the fewest shapes (tie -> lexicographically by type dir)
        chosen = min(paths, key=lambda p: (len(json.loads(p.read_text()).get("shapes", [])), p.as_posix()))
        payload = json.loads(chosen.read_text())
        img = dataset / split / "images" / f"{stem}.jpg"
        if not img.is_file():
            missing_img.append((split, stem, chosen.as_posix()))
            skipped += 1
            continue
        W = int(payload.get("imageWidth") or 0)
        H = int(payload.get("imageHeight") or 0)
        if W <= 0 or H <= 0:
            import cv2
            im = cv2.imread(str(img))
            H, W = im.shape[:2]
        txt = shapes_to_txt(payload, name2id, W, H)
        dst = dataset / split / "labels" / f"{stem}.txt"
        if not args.dry_run:
            if dst.is_file():
                bdst = backup / split / dst.name
                bdst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst, bdst)
            dst.write_text(txt, encoding="utf-8")
        written += 1
        rows.append((split, stem, len(payload.get("shapes", [])), len(paths), chosen.parent.parent.name))

    if missing_img:
        print(f"[warn] {len(missing_img)} stems had no image in dataset, skipped:")
        for r in missing_img[:10]:
            print("   ", r)

    print(f"\n[done] unique images={len(groups)} written={written} skipped={skipped} multi-copy={conflicts}")
    print(f"[backup] originals -> {backup}")

    # drop stale caches so ultralytics rebuilds them
    if not args.dry_run:
        for split in SPLITS:
            c = dataset / split / "labels.cache"
            if c.exists():
                c.unlink()
                print(f"[cache] removed {c.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
