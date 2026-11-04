#!/usr/bin/env python3
"""把人工复核后的 JSON 转成 YOLO txt，并按 group-disjoint + 类别均衡重建 7:2:1 划分。

用法::

    python scripts/build_balanced_split.py --dry-run
    python scripts/build_balanced_split.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path

import yaml

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset/clean_part_batch_12.v8i.merged12.yolov11"
DEFAULT_REVIEWED = ROOT / "merged-reviewed"
DEFAULT_TXT_OUT = ROOT / "temp/merged-reviewed-labels"
SPLITS = ("train", "valid", "test")
TARGET = (0.7, 0.2, 0.1)


def group_key(filename: str) -> str:
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


def load_names(dataset: Path) -> list[str]:
    doc = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    names = doc["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(n) for n in names]


def image_size(payload: dict, img: Path) -> tuple[int, int]:
    w = int(payload.get("imageWidth") or 0)
    h = int(payload.get("imageHeight") or 0)
    if w > 0 and h > 0:
        return w, h
    import cv2

    im = cv2.imread(str(img))
    if im is None:
        raise ValueError(f"cannot read image size: {img}")
    return int(im.shape[1]), int(im.shape[0])


def json_to_lines(payload: dict, name2id: dict[str, int], W: int, H: int) -> list[str]:
    lines = []
    for s in payload.get("shapes", []):
        if s.get("shape_type", "rectangle") != "rectangle":
            continue
        label = str(s.get("label", "")).strip()
        if label not in name2id and label == "M":
            label = "Motorcycle"
        if label not in name2id:
            raise ValueError(f"unknown label {s.get('label')!r}")
        pts = s.get("points") or []
        if len(pts) < 2:
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        x1 = max(0.0, min(float(W), min(xs)))
        x2 = max(0.0, min(float(W), max(xs)))
        y1 = max(0.0, min(float(H), min(ys)))
        y2 = max(0.0, min(float(H), max(ys)))
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1e-6 or bh <= 1e-6:
            continue
        lines.append(
            f"{name2id[label]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
            f"{bw / W:.6f} {bh / H:.6f}"
        )
    return lines


def count_boxes(text: str, nclasses: int) -> list[int]:
    c = [0] * nclasses
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 5:
            k = int(float(p[0]))
            if 0 <= k < nclasses:
                c[k] += 1
    return c


def convert_reviewed(reviewed: Path, txt_out: Path, names: list[str], dry_run: bool) -> dict[str, str]:
    name2id = {n: i for i, n in enumerate(names)}
    out: dict[str, str] = {}
    unknown = collections.Counter()
    jsons = sorted(reviewed.glob("*.json"))
    if not dry_run:
        txt_out.mkdir(parents=True, exist_ok=True)
    for jp in jsons:
        payload = json.loads(jp.read_text(encoding="utf-8"))
        img = reviewed / f"{jp.stem}.jpg"
        if not img.is_file():
            for ext in IMG_EXT[1:]:
                cand = reviewed / f"{jp.stem}{ext}"
                if cand.is_file():
                    img = cand
                    break
        W, H = image_size(payload, img)
        for s in payload.get("shapes", []):
            label = str(s.get("label", "")).strip()
            if label not in name2id and label != "M":
                unknown[label] += 1
        lines = json_to_lines(payload, name2id, W, H)
        text = ("\n".join(lines) + "\n") if lines else ""
        out[jp.stem] = text
        if not dry_run:
            (txt_out / f"{jp.stem}.txt").write_text(text, encoding="utf-8")
    if unknown:
        print(f"[warn] unknown labels ignored: {dict(unknown)}")
    print(f"[convert] reviewed json={len(jsons)} -> txt={len(out)} -> {txt_out}")
    return out


def load_items(dataset: Path, reviewed_txt: dict[str, str], nclasses: int):
    items = []
    for split in SPLITS:
        img_dir = dataset / split / "images"
        lbl_dir = dataset / split / "labels"
        for p in sorted(img_dir.iterdir()):
            if not p.is_file() or not p.name.lower().endswith(IMG_EXT):
                continue
            stem = p.stem
            lp = lbl_dir / f"{stem}.txt"
            if not lp.is_file():
                raise SystemExit(f"missing label for image: {p}")
            text = reviewed_txt.get(stem)
            if text is None:
                text = lp.read_text(encoding="utf-8")
            items.append(
                {
                    "stem": stem,
                    "img": p,
                    "text": text,
                    "boxes": count_boxes(text, nclasses),
                    "group": group_key(p.name),
                    "old_split": split,
                }
            )
    if len({it["stem"] for it in items}) != len(items):
        raise SystemExit("duplicate stems found in dataset")
    return items


def optimise(groups: dict[str, list], nclasses: int, restarts: int, iters: int, seed: int):
    keys = sorted(groups)
    G = [[sum(it["boxes"][c] for it in groups[k]) for c in range(nclasses)] for k in keys]
    S = [len(groups[k]) for k in keys]
    total = [sum(g[c] for g in G) for c in range(nclasses)]
    total_img = sum(S)
    n = len(keys)
    weight_img = 2.0

    def err(ib, vb):
        e = weight_img * sum(abs(ib[s] / total_img - TARGET[s]) for s in range(3))
        for c in range(nclasses):
            if total[c]:
                e += sum(abs(vb[s][c] / total[c] - TARGET[s]) for s in range(3))
        return e

    def run(rs: int):
        rng = random.Random(rs)
        assign = [0] * n
        ib = [total_img, 0, 0]
        vb = [[0] * nclasses for _ in range(3)]
        vb[0] = total[:]
        cur = err(ib, vb)
        lb, la = cur, assign[:]
        t0, t1 = 0.05, 0.0002
        for it in range(iters):
            temp = t0 * (t1 / t0) ** (it / iters)
            i = rng.randrange(n)
            old = assign[i]
            new = rng.choice([s for s in range(3) if s != old])
            ib[old] -= S[i]
            ib[new] += S[i]
            for c in range(nclasses):
                vb[old][c] -= G[i][c]
                vb[new][c] += G[i][c]
            ne = err(ib, vb)
            if ne <= cur or rng.random() < math.exp((cur - ne) / max(temp, 1e-9)):
                assign[i] = new
                cur = ne
                if ne < lb:
                    lb, la = ne, assign[:]
            else:
                ib[new] -= S[i]
                ib[old] += S[i]
                for c in range(nclasses):
                    vb[new][c] -= G[i][c]
                    vb[old][c] += G[i][c]
        return lb, la

    best = None
    for r in range(restarts):
        lb, la = run(seed + r)
        if best is None or lb < best[0]:
            best = (lb, la)
        print(f"[anneal {r:02d}] err={lb:.6f} best={best[0]:.6f}", flush=True)
    assign = best[1]

    def full(a):
        ib = [0, 0, 0]
        vb = [[0] * nclasses for _ in range(3)]
        for i, s in enumerate(a):
            ib[s] += S[i]
            for c in range(nclasses):
                vb[s][c] += G[i][c]
        return err(ib, vb), ib, vb

    cur, ib, vb = full(assign)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            old = assign[i]
            for new in range(3):
                if new == old:
                    continue
                ib[old] -= S[i]
                ib[new] += S[i]
                for c in range(nclasses):
                    vb[old][c] -= G[i][c]
                    vb[new][c] += G[i][c]
                ne = err(ib, vb)
                if ne < cur - 1e-12:
                    assign[i] = new
                    cur = ne
                    changed = True
                    break
                ib[new] -= S[i]
                ib[old] += S[i]
                for c in range(nclasses):
                    vb[new][c] -= G[i][c]
                    vb[old][c] += G[i][c]
        print(f"[local move] err={cur:.6f}", flush=True)

    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                ia, ja = assign[i], assign[j]
                if ia == ja:
                    continue
                ib[ia] -= S[i]
                ib[ja] += S[i]
                ib[ja] -= S[j]
                ib[ia] += S[j]
                for c in range(nclasses):
                    vb[ia][c] -= G[i][c]
                    vb[ja][c] += G[i][c]
                    vb[ja][c] -= G[j][c]
                    vb[ia][c] += G[j][c]
                ne = err(ib, vb)
                if ne < cur - 1e-12:
                    assign[i], assign[j] = ja, ia
                    cur = ne
                    changed = True
                else:
                    ib[ia] += S[i]
                    ib[ja] -= S[i]
                    ib[ja] += S[j]
                    ib[ia] -= S[j]
                    for c in range(nclasses):
                        vb[ia][c] += G[i][c]
                        vb[ja][c] -= G[i][c]
                        vb[ja][c] += G[j][c]
                        vb[ia][c] -= G[j][c]
        print(f"[local swap] err={cur:.6f}", flush=True)

    return keys, assign, G, S


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--reviewed", default=str(DEFAULT_REVIEWED))
    ap.add_argument("--txt-out", default=str(DEFAULT_TXT_OUT))
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--iters", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    reviewed = Path(args.reviewed)
    names = load_names(dataset)
    nc = len(names)

    reviewed_txt = convert_reviewed(reviewed, Path(args.txt_out), names, args.dry_run)
    items = load_items(dataset, reviewed_txt, nc)
    groups = collections.defaultdict(list)
    for it in items:
        groups[it["group"]].append(it)
    print(f"[data] images={len(items)} groups={len(groups)} reviewed={len(reviewed_txt)} nc={nc}")

    keys, assign, G, S = optimise(groups, nc, args.restarts, args.iters, args.seed)
    split_of = {keys[i]: assign[i] for i in range(len(keys))}

    split_items = {s: [] for s in range(3)}
    for it in items:
        split_items[split_of[it["group"]]].append(it)

    def boxcounts(rows):
        c = [0] * nc
        for it in rows:
            for k in range(nc):
                c[k] += it["boxes"][k]
        return c

    ib = [len(split_items[s]) for s in range(3)]
    vb = [boxcounts(split_items[s]) for s in range(3)]
    print(f"\nimages: train={ib[0]} valid={ib[1]} test={ib[2]}  "
          f"ratios={[round(ib[s] / len(items), 4) for s in range(3)]}")
    print(f"{'class':16s}{'train':>8}{'valid':>8}{'test':>8}   test_ratio")
    for c in range(nc):
        tot = sum(vb[s][c] for s in range(3))
        print(f"{names[c]:16s}{vb[0][c]:>8}{vb[1][c]:>8}{vb[2][c]:>8}   "
              f"{(vb[2][c] / tot) if tot else 0:.3f}")

    if args.dry_run:
        print("\ndry-run: no files written")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = ROOT / f"temp/backup_split_{ts}"
    backup.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        shutil.copytree(dataset / split / "labels", backup / "labels" / split)
    for name in ("split_manifest.csv", "data.yaml", "report.json"):
        src = dataset / name
        if src.is_file():
            shutil.copy2(src, backup / name)
    print(f"[backup] {backup}")

    old_src = {}
    manifest = dataset / "split_manifest.csv"
    if manifest.is_file():
        for row in csv.DictReader(open(manifest, encoding="utf-8")):
            old_src[row["stem"]] = row.get("source", "unknown")

    staging = dataset / f".rebuild_{ts}"
    staging.mkdir(parents=True, exist_ok=True)
    for s in range(3):
        (staging / SPLITS[s] / "images").mkdir(parents=True)
        (staging / SPLITS[s] / "labels").mkdir(parents=True)
    for s in range(3):
        for it in split_items[s]:
            dst_img = staging / SPLITS[s] / "images" / it["img"].name
            try:
                os.link(it["img"], dst_img)
            except OSError:
                shutil.copy2(it["img"], dst_img)
            (staging / SPLITS[s] / "labels" / f"{it['stem']}.txt").write_text(
                it["text"], encoding="utf-8"
            )

    old_dir = dataset / f".old_{ts}"
    old_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        os.replace(dataset / split, old_dir / split)
    for split in SPLITS:
        os.replace(staging / split, dataset / split)
    shutil.rmtree(old_dir)
    shutil.rmtree(staging)

    doc = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    path = doc.get("path", str(dataset))
    (dataset / "data.yaml").write_text(
        f"path: {path}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {nc}\nnames: {names}\n",
        encoding="utf-8",
    )

    with open(dataset / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "source", "group", "split"])
        for s in range(3):
            for it in sorted(split_items[s], key=lambda x: x["stem"]):
                w.writerow([it["stem"], old_src.get(it["stem"], "unknown"), it["group"], SPLITS[s]])

    per_class = []
    for c in range(nc):
        tot = sum(vb[s][c] for s in range(3))
        per_class.append(
            {
                "class": names[c],
                "train": vb[0][c],
                "valid": vb[1][c],
                "test": vb[2][c],
                "test_ratio": round(vb[2][c] / tot, 4) if tot else 0.0,
            }
        )
    report = {
        "images": {"train": ib[0], "valid": ib[1], "test": ib[2], "total": len(items)},
        "image_ratios": [round(ib[s] / len(items), 4) for s in range(3)],
        "target": {"train": TARGET[0], "valid": TARGET[1], "test": TARGET[2]},
        "per_class": per_class,
        "reviewed_txt": len(reviewed_txt),
    }
    (dataset / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for split in SPLITS:
        cache = dataset / split / "labels.cache"
        if cache.exists():
            cache.unlink()
    print(f"[done] rebuilt {dataset}")


if __name__ == "__main__":
    main()
