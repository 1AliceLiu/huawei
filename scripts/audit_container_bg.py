"""Mine Container false-negatives (true Container predicted as background) on a split.

Runs a trained checkpoint over one split, matches predictions to Container GT boxes
with one-to-one greedy IoU matching. Every Container GT box that has NO matched
prediction (IoU >= --iou AND conf >= --match-conf) is a "predicted as background"
miss. Its crop (GT box + margin, GT box drawn, best candidate pred drawn in red) is
saved for human triage, together with a manifest JSON.

Usage:
    python scripts/audit_container_bg.py
    python scripts/audit_container_bg.py --split train --conf 0.05 --match-conf 0.25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "runs/detect/yolo26s_merged12_20261016_0118/weights/best.pt"
DEFAULT_DATA = ROOT / "dataset/clean_part_batch_12.v8i.merged12.yolov11"
IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def load_names(data_dir: Path) -> list[str]:
    doc = yaml.safe_load((data_dir / "data.yaml").read_text())
    names = doc["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return list(names)


def load_boxes(label_path: Path, cls_id: int) -> list[tuple[float, float, float, float]]:
    """Return xyxy-normalized boxes of a given class from a YOLO label file."""
    out = []
    if not label_path.is_file():
        return out
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) >= 5 and int(float(p[0])) == cls_id:
            x, y, w, h = (float(v) for v in p[1:5])
            out.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of xyxy boxes (a: Nx4, b: Mx4) -> NxM."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def greedy_match(iou: np.ndarray, thresh: float) -> set[int]:
    """Greedy one-to-one matching; return set of GT indices that got a match."""
    pairs = [
        (float(iou[i, j]), i, j)
        for i in range(iou.shape[0])
        for j in range(iou.shape[1])
        if iou[i, j] >= thresh
    ]
    pairs.sort(reverse=True)
    pred_used: set[int] = set()
    gt_used: set[int] = set()
    for _, i, j in pairs:
        if i in pred_used or j in gt_used:
            continue
        pred_used.add(i)
        gt_used.add(j)
    return gt_used


def draw_crop(img, gt_xyxy, pred, margin, cls_name):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = gt_xyxy
    bw, bh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - margin * bw))
    cy1 = max(0, int(y1 - margin * bh))
    cx2 = min(w, int(x2 + margin * bw))
    cy2 = min(h, int(y2 + margin * bh))
    if cx2 - cx1 < 16 or cy2 - cy1 < 16:
        return None
    crop = img[cy1:cy2, cx1:cx2].copy()
    ox, oy = cx1, cy1
    cv2.rectangle(
        crop,
        (int(x1 - ox), int(y1 - oy)),
        (int(x2 - ox), int(y2 - oy)),
        (0, 200, 0),
        2,
    )
    if pred is not None:
        px1, py1, px2, py2, pconf, pname = pred
        cv2.rectangle(
            crop,
            (int(px1 - ox), int(py1 - oy)),
            (int(px2 - ox), int(py2 - oy)),
            (0, 0, 255),
            1,
        )
        label = f"best_pred={pname} {pconf:.2f}"
    else:
        label = "best_pred=NONE"
    cv2.putText(crop, f"GT {cls_name}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(crop, label, (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return crop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=None, help="default <data>/hard_samples/container_as_bg")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.05, help="inference conf for candidate gathering")
    ap.add_argument("--match-conf", type=float, default=0.25, help="conf needed for a pred to 'cover' a GT")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU needed to 'cover' a GT")
    ap.add_argument("--margin", type=float, default=0.4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--save-full", action="store_true", help="also save the whole image")
    ap.add_argument("--limit", type=int, default=0, help="limit images (debug)")
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out) if args.out else data_dir / "hard_samples" / "container_as_bg"
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    full_dir = out_dir / "full"
    if args.save_full:
        full_dir.mkdir(parents=True, exist_ok=True)

    names = load_names(data_dir)
    cls_id = names.index("Container")
    names_map = {i: n for i, n in enumerate(names)}
    print(f"[info] Container class id = {cls_id}, weights = {args.weights}")

    img_dir = data_dir / args.split / "images"
    lbl_dir = data_dir / args.split / "labels"
    all_imgs = {p.stem: p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMAGE_EXT}
    paths = [
        all_imgs[lb.stem]
        for lb in sorted(lbl_dir.glob("*.txt"))
        if lb.stem in all_imgs and load_boxes(lb, cls_id)
    ]
    if args.limit:
        paths = paths[: args.limit]

    model = YOLO(args.weights)
    records: list[dict] = []
    n_scanned = n_with = n_gt = n_miss = 0
    checked = 0

    for img_path in paths:
        checked += 1
        if checked % 200 == 0:
            print(f"  ... {checked}/{len(paths)} scanned, misses so far = {n_miss}")
        res = model.predict(
            str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]
        n_scanned += 1
        gt = load_boxes(lbl_dir / f"{img_path.stem}.txt", cls_id)
        if not gt:
            continue
        n_with += 1
        n_gt += len(gt)

        boxes = res.boxes
        if boxes is not None and len(boxes):
            pred_xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            pred_conf = boxes.conf.cpu().numpy().astype(np.float32)
            pred_cls = boxes.cls.cpu().numpy().astype(int)
        else:
            pred_xyxy = np.zeros((0, 4), np.float32)
            pred_conf = np.zeros((0,), np.float32)
            pred_cls = np.zeros((0,), int)

        gt_arr = np.array(gt, dtype=np.float32)
        h, w = res.orig_shape
        gt_px = gt_arr * np.array([w, h, w, h], np.float32)
        pred_px = pred_xyxy

        iou_all = iou_matrix(pred_px, gt_px)
        strong = pred_conf >= args.match_conf
        iou_strong = iou_all.copy()
        if len(strong):
            iou_strong[~strong] = 0.0
        else:
            iou_strong = np.zeros_like(iou_all)
        matched = greedy_match(iou_strong, args.iou)

        img = None
        for gi, gbox in enumerate(gt_px):
            if gi in matched:
                continue
            n_miss += 1
            if img is None:
                img = cv2.imread(str(img_path))
            if img is None:
                break
            best = None
            if len(pred_px):
                bi = int(np.argmax(iou_all[:, gi]))
                if iou_all[bi, gi] > 0:
                    best = (
                        pred_px[bi, 0],
                        pred_px[bi, 1],
                        pred_px[bi, 2],
                        pred_px[bi, 3],
                        float(pred_conf[bi]),
                        names_map.get(int(pred_cls[bi]), str(int(pred_cls[bi]))),
                    )
            crop = draw_crop(img, gbox, best, args.margin, names_map[cls_id])
            rel = None
            if crop is not None:
                fn = f"{args.split}_{img_path.stem}_gt{gi}.jpg"
                cv2.imwrite(str(crop_dir / fn), crop)
                rel = f"crops/{fn}"
                if args.save_full:
                    cv2.imwrite(str(full_dir / f"{args.split}_{img_path.stem}.jpg"), img)
            records.append(
                {
                    "image": str(img_path),
                    "label": str(lbl_dir / f"{img_path.stem}.txt"),
                    "gt_index": gi,
                    "gt_xyxy_px": [round(float(v), 1) for v in gbox],
                    "best_pred_class": best[5] if best else None,
                    "best_pred_conf": round(best[4], 4) if best else None,
                    "best_pred_iou": round(float(iou_all[int(np.argmax(iou_all[:, gi])), gi]), 4) if len(pred_px) else 0.0,
                    "crop": rel,
                }
            )

    manifest = {
        "weights": args.weights,
        "data": str(data_dir),
        "split": args.split,
        "container_class_id": cls_id,
        "settings": {"imgsz": args.imgsz, "conf": args.conf, "match_conf": args.match_conf, "iou": args.iou},
        "stats": {
            "images_scanned": n_scanned,
            "images_with_container": n_with,
            "container_gt_boxes": n_gt,
            "container_missed": n_miss,
            "miss_rate": round(n_miss / n_gt, 4) if n_gt else 0.0,
        },
        "records": records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("\n[done]", json.dumps(manifest["stats"], ensure_ascii=False))
    print(f"[out] crops  : {crop_dir}")
    print(f"[out] manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
