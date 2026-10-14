"""Extract a high-quality, high-density, multi-class subset from the training pool.

Reads the group-disjoint training pool produced by ``build_unified_val.py`` and scores
every image on four dimensions, then de-duplicates near-identical frames and performs
a stratified selection:

* density        - number of annotated boxes (busy scenes carry more context)
* diversity      - rarity-weighted count of classes present (rewards rare/weak classes)
* small-object   - number of small boxes (normalized area proxy for hard, distant targets)
* clarity        - Laplacian variance on a resolution-normalized grayscale crop
* exposure       - downweights severely over/under-exposed frames

All dimensions are converted to percentile ranks (robust to the mixed 640x640 /
1920x1080 resolutions), combined into one score, then near-duplicates are removed with a
64-bit perceptual hash. Finally a per-class floor guarantees every class (incl. rare
Container) is represented before the remaining budget is filled by score.

Outputs (under <dataset>/splits):
* ``train_high_quality.txt`` - one image path per line
* ``high_quality_report.json`` - selected count + per-class box distribution

Usage
-----
    python scripts/extract_quality_subset.py
    python scripts/extract_quality_subset.py --target-size 1800 --max-size 2000 --dedup-hamming 6
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DS = ROOT / "dataset" / "batch_12.v14i.merged12.yolov11"
NAMES = [
    "Coach", "Franchised Bus", "HGV", "LGV", "Light Bus", "MGV",
    "Motorcycle", "PLB GMB", "Private Car", "Taxi", "Van", "Container",
]

WEIGHTS = {"density": 0.30, "diversity": 0.25, "clarity": 0.20, "small": 0.15, "exposure": 0.10}
SMALL_THRESH = 0.04  # boxes whose sqrt(area) < 4% of the frame count as "small"


def image_path_from_line(ds: Path, line: str) -> Path:
    p = Path(line.strip())
    return p if p.is_absolute() else (ds / p)


def label_path_for(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def parse_boxes(lbl: Path) -> list[tuple[int, float, float, float, float]]:
    if not lbl.is_file():
        return []
    out = []
    for line in lbl.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 5:
            try:
                out.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
            except ValueError:
                pass
    return out


def phash64(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    bits = dct > np.median(dct)
    val = 0
    for b in bits.flatten():
        val = (val << 1) | int(b)
    return val


def analyze(job: tuple[str, str, int]) -> dict:
    """Worker: returns per-image features. Kept top-level for pickling."""
    path, lbl, flag = job
    boxes = parse_boxes(Path(lbl))
    classes = {b[0] for b in boxes}
    small = sum(1 for b in boxes if math.sqrt(b[3] * b[4]) < SMALL_THRESH)
    cls_ids = [b[0] for b in boxes]
    gray = cv2.imread(path, flag)
    if gray is None:
        return {"path": path, "ok": False, "nbox": len(boxes), "box_ids": cls_ids,
                "classes": sorted(classes), "small": small, "lap": 0.0, "mean": 0.0, "hash": 0}
    h, w = gray.shape[:2]
    s = 256.0 / max(1, min(h, w))
    norm = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    lap = float(cv2.Laplacian(norm, cv2.CV_64F).var())
    return {"path": path, "ok": True, "nbox": len(boxes), "box_ids": cls_ids,
            "classes": sorted(classes), "small": small, "lap": lap,
            "mean": float(norm.mean()), "hash": phash64(norm)}


def rank_pct(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values))
    return ranks / max(1, len(values) - 1)


def popcount_min(xor: np.ndarray) -> np.ndarray:
    """Minimum? no - returns popcount per row; caller takes min."""
    return np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DS)
    ap.add_argument("--train-list", type=Path, default=None, help="default <dataset>/splits/train_pool.txt")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--target-size", type=int, default=1800)
    ap.add_argument("--max-size", type=int, default=2000)
    ap.add_argument("--min-class-boxes", type=int, default=30)
    ap.add_argument("--dedup-hamming", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--reduce", type=int, default=2, choices=[1, 2, 4], help="decode downscale factor for clarity")
    args = ap.parse_args()

    ds = args.dataset.resolve()
    out = (args.out_dir or (ds / "splits")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    train_list = args.train_list or (out / "train_pool.txt")
    lines = [l.strip() for l in train_list.read_text(encoding="utf-8").splitlines() if l.strip()]
    flag = {1: cv2.IMREAD_GRAYSCALE, 2: cv2.IMREAD_REDUCED_GRAYSCALE_2, 4: cv2.IMREAD_REDUCED_GRAYSCALE_4}[args.reduce]

    jobs = []
    for line in lines:
        img = image_path_from_line(ds, line)
        jobs.append((str(img), str(label_path_for(img)), flag))

    cv2.setNumThreads(1)
    feats: list[dict] = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for i, r in enumerate(ex.map(analyze, jobs, chunksize=32)):
                feats.append(r)
                if (i + 1) % 1000 == 0:
                    print(f"  analyzed {i + 1}/{len(jobs)}", flush=True)
    else:
        for i, j in enumerate(jobs):
            feats.append(analyze(j))
            if (i + 1) % 1000 == 0:
                print(f"  analyzed {i + 1}/{len(jobs)}", flush=True)

    ok = [f for f in feats if f["ok"]]
    if not ok:
        raise SystemExit("no readable images")

    total_boxes = Counter()
    for f in ok:
        total_boxes.update(f["box_ids"])
    rarity = {c: 1.0 / math.sqrt(total_boxes[c] + 1) for c in range(len(NAMES))}

    n = len(ok)
    dens = np.array([f["nbox"] for f in ok], dtype=np.float64)
    div = np.array([sum(rarity[c] for c in f["classes"]) for f in ok], dtype=np.float64)
    sml = np.array([f["small"] for f in ok], dtype=np.float64)
    clr = np.array([math.log1p(f["lap"]) for f in ok], dtype=np.float64)
    exp_ok = np.array([1.0 if 20.0 <= f["mean"] <= 235.0 else 0.0 for f in ok], dtype=np.float64)

    score = (WEIGHTS["density"] * rank_pct(dens)
             + WEIGHTS["diversity"] * rank_pct(div)
             + WEIGHTS["small"] * rank_pct(sml)
             + WEIGHTS["clarity"] * rank_pct(clr)
             + WEIGHTS["exposure"] * exp_ok)
    for f, s in zip(ok, score):
        f["score"] = float(s)

    ranked = sorted(ok, key=lambda f: f["score"], reverse=True)

    # ---- near-duplicate removal (greedy by score, 64-bit pHash hamming) ----
    kept: list[dict] = []
    kept_hashes = np.empty(0, dtype=np.uint64)
    th = args.dedup_hamming
    for f in ranked:
        h = np.uint64(f["hash"])
        if kept_hashes.size:
            pc = popcount_min(kept_hashes ^ h)
            if pc.min() <= th:
                continue
        kept.append(f)
        kept_hashes = np.append(kept_hashes, h)

    # ---- stratified selection: per-class floor, then fill by score ----
    selected: list[dict] = []
    chosen: set[str] = set()
    have = Counter()

    def take(f: dict) -> None:
        selected.append(f)
        chosen.add(f["path"])
        have.update(f["box_ids"])

    for c in sorted(range(len(NAMES)), key=lambda x: total_boxes[x]):
        if have[c] >= args.min_class_boxes:
            continue
        for f in kept:
            if f["path"] in chosen:
                continue
            if c in f["classes"]:
                take(f)
                if have[c] >= args.min_class_boxes:
                    break

    for f in kept:
        if len(selected) >= args.target_size:
            break
        if f["path"] not in chosen:
            take(f)

    if len(selected) > args.max_size:
        selected = selected[: args.max_size]

    have = Counter()
    for f in selected:
        have.update(f["box_ids"])
    selected.sort(key=lambda f: f["score"], reverse=True)
    (out / "train_high_quality.txt").write_text(
        "\n".join(f["path"] for f in selected) + "\n", encoding="utf-8")

    report = {
        "source_list": str(train_list),
        "source_images": len(lines),
        "readable_images": len(ok),
        "duplicates_removed": len(ok) - len(kept),
        "selected": len(selected),
        "target_size": args.target_size,
        "dedup_hamming": args.dedup_hamming,
        "weights": WEIGHTS,
        "per_class_boxes": {NAMES[c]: have[c] for c in range(len(NAMES))},
        "source_per_class_boxes": {NAMES[c]: total_boxes[c] for c in range(len(NAMES))},
    }
    (out / "high_quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nsource pool      : {len(lines)} images")
    print(f"readable / dupes : {len(ok)} / {len(ok) - len(kept)} removed")
    print(f"selected         : {len(selected)} -> {out / 'train_high_quality.txt'}")
    print(f"avg boxes/img    : {sum(f['nbox'] for f in selected) / max(1, len(selected)):.2f}"
          f"  (pool avg {sum(f['nbox'] for f in ok) / max(1, len(ok)):.2f})")
    print(f"{'id':>2} {'class':<15} {'source':>8} {'subset':>8}")
    for c, name in enumerate(NAMES):
        print(f"{c:>2} {name:<15} {total_boxes[c]:>8} {have[c]:>8}")


if __name__ == "__main__":
    main()
