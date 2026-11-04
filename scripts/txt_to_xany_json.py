"""Convert YOLO label txt -> X-AnyLabeling (LabelMe superset) JSON, for manual fixing.

Mirrors auto_check_dataset/build_suspect_sets.py:convert_txt so the output loads in
the same annotation software. For every `<stem>.txt` that has a sibling image, writes
`<stem>.json` next to the image.

Usage:
    python scripts/txt_to_xany_json.py
    python scripts/txt_to_xany_json.py --root temp/error-label/clean_part_batch_12.v8i.merged12.yolov11
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "temp/error-label/clean_part_batch_12.v8i.merged12.yolov11"
DEFAULT_YAML = ROOT / "dataset/clean_part_batch_12.v8i.merged12.yolov11/data.yaml"
IMAGE_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
VERSION = "4.0.0-beta.7"


def load_classes(path: Path) -> list[str]:
    doc = yaml.safe_load(path.read_text())
    names = doc["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(n) for n in names]


def find_image(folder: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXT:
        p = folder / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def txt_to_xany(txt_path: Path, img_path: Path, classes: list[str]) -> dict:
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    shapes = []
    for line in txt_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cid = int(float(p[0]))
        xc, yc, w, h = (float(v) for v in p[1:5])
        x1, y1 = (xc - w / 2) * W, (yc - h / 2) * H
        x2, y2 = (xc + w / 2) * W, (yc + h / 2) * H
        shapes.append({
            "label": classes[cid] if 0 <= cid < len(classes) else str(cid),
            "score": None,
            "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "rectangle",
            "flags": {},
            "kie_linking": [],
        })
    return {
        "version": VERSION,
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": img_path.name,
        "imageData": None,
        "imageHeight": H,
        "imageWidth": W,
        "description": "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--yaml", default=str(DEFAULT_YAML))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    classes = load_classes(Path(args.yaml))
    txts = sorted(root.rglob("*.txt"))
    n_ok = n_skip = n_box = 0
    for txt in txts:
        img = find_image(txt.parent, txt.stem)
        if img is None:
            print(f"[skip] no image for {txt}")
            n_skip += 1
            continue
        payload = txt_to_xany(txt, img, classes)
        n_box += len(payload["shapes"])
        out = txt.with_suffix(".json")
        if not args.dry_run:
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        n_ok += 1
    print(f"[done] json written={n_ok} skipped={n_skip} total_shapes={n_box}")
    print(f"[root] {root}")


if __name__ == "__main__":
    main()
