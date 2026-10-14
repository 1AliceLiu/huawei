"""标注后入库：将 pending 目录的幸存文件并入 merged12 数据集（in place）。

流程（固化 R2-R8 的手工步骤）：
1. 幸存文件 = images/*.jpg 且 labels/*.txt 同时存在（废图用户已移到 _delete_/）。
2. 校验 txt：5 列、类别 0-11、坐标归一化；空文件视为背景保留。
3. 同类 IoU>=0.85 去重（txt 无 conf，保留首个），计数上报。
4. 按机位 + 文件名 chronological 连续切分 80/10/10（n<10 全进 train，
   与 build_merged12.split_922 同口径；valid/test 各 max(1, round(n*0.1))，取尾段）。
5. 拷贝 jpg+txt 到 dataset train/valid/test；manifest.csv reconcile 到幸存文件。
6. 只打印统计，不写 APPEND_*.md（人工按打印结果补）。

用法：
  python scripts/rtsp_mine_300/append_pending.py --dir tmp/r9_pending [--dry-run]
"""
import argparse
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = ["Coach", "Franchised Bus", "HGV", "LGV", "Light Bus", "MGV",
         "Motorcycle", "PLB GMB", "Private Car", "Taxi", "Van", "Container"]


def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="tmp/r9_pending")
    ap.add_argument("--dataset", default="dataset/batch_12.v14i.merged12.yolov11")
    ap.add_argument("--dup-iou", type=float, default=0.85)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = ROOT / args.dir
    ds = ROOT / args.dataset
    img_dir, lbl_dir = d / "images", d / "labels"
    assert img_dir.is_dir() and lbl_dir.is_dir(), f"目录缺失：{img_dir} / {lbl_dir}"

    stems = sorted(p.stem for p in img_dir.glob("*.jpg")
                   if (lbl_dir / (p.stem + ".txt")).is_file())
    assert stems, "无幸存文件（jpg+txt 配对为空）"
    print(f"survived: {len(stems)} (jpg+txt paired)")

    # ---- 校验 + 同类去重 ----
    boxes_of, n_dup, n_box = {}, 0, 0
    for stem in stems:
        rows = [l.split() for l in (lbl_dir / (stem + ".txt")).read_text(
            encoding="utf-8").splitlines() if l.strip()]
        parsed = []
        for t in rows:
            assert len(t) == 5, f"{stem}: 列数={len(t)}"
            c = int(t[0])
            assert 0 <= c <= 11, f"{stem}: 非法类别 {c}"
            x, y, w, h = map(float, t[1:])
            assert all(0.0 <= v <= 1.0 for v in (x, y, w, h)) and w > 0 and h > 0, \
                f"{stem}: 坐标越界"
            parsed.append([c, x, y, w, h])
        kept, drop = [], set()
        by_cls = defaultdict(list)
        for i, b in enumerate(parsed):
            by_cls[b[0]].append(i)
        for idxs in by_cls.values():
            for a in range(len(idxs)):
                if idxs[a] in drop:
                    continue
                ba = parsed[idxs[a]]
                xa = [ba[1] - ba[3] / 2, ba[2] - ba[4] / 2,
                      ba[1] + ba[3] / 2, ba[2] + ba[4] / 2]
                for b in range(a + 1, len(idxs)):
                    if idxs[b] in drop:
                        continue
                    bb = parsed[idxs[b]]
                    xb = [bb[1] - bb[3] / 2, bb[2] - bb[4] / 2,
                          bb[1] + bb[3] / 2, bb[2] + bb[4] / 2]
                    if box_iou(xa, xb) >= args.dup_iou:
                        drop.add(idxs[b])
                        n_dup += 1
        kept = [b for i, b in enumerate(parsed) if i not in drop]
        boxes_of[stem] = kept
        n_box += len(kept)
        if drop and not args.dry_run:
            (lbl_dir / (stem + ".txt")).write_text(
                "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                          for c, x, y, w, h in kept) + ("\n" if kept else ""),
                encoding="utf-8")
    print(f"export dedup(IoU>={args.dup_iou}) removed: {n_dup}, manual boxes: {n_box}")

    # ---- 按机位 chronological 切分（同 build_merged12.split_922 口径） ----
    by_cam = defaultdict(list)
    for s in stems:
        by_cam[s.rsplit("_", 1)[1]].append(s)
    assign = {}
    for cam, lst in by_cam.items():
        lst.sort()
        n = len(lst)
        if n >= 10:
            n_val = max(1, round(n * 0.1))
            n_test = max(1, round(n * 0.1))
            for i, s in enumerate(lst):
                assign[s] = ("train" if i < n - n_test - n_val
                             else ("valid" if i < n - n_test else "test"))
        else:
            for s in lst:
                assign[s] = "train"

    # ---- 拷贝 ----
    cnt, cls_cnt = Counter(), Counter()
    for stem in stems:
        split = assign[stem]
        for src, sub in ((img_dir / (stem + ".jpg"), "images"),
                         (lbl_dir / (stem + ".txt"), "labels")):
            dst = ds / split / sub / src.name
            assert not dst.exists(), f"目标已存在（重名冲突）：{dst}"
            if not args.dry_run:
                shutil.copy2(src, dst)
        cnt[split] += 1
        for c, *_ in boxes_of[stem]:
            cls_cnt[c] += 1
    print(f"split added: train={cnt['train']} valid={cnt['valid']} test={cnt['test']}")
    print(f"class boxes: {dict(sorted(cls_cnt.items()))}")
    print("  (" + ", ".join(f"{NAMES[i]}={cls_cnt[i]}" for i in sorted(cls_cnt)) + ")")

    # ---- manifest reconcile ----
    mp = d / "manifest.csv"
    if mp.is_file() and not args.dry_run:
        rows = list(csv.DictReader(open(mp, encoding="utf-8")))
        keep = [r for r in rows if r["file"][:-4] in set(stems)]
        with open(mp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(keep)
        print(f"manifest: {len(rows)} -> {len(keep)}")
    if args.dry_run:
        print("dry-run: no files written")


if __name__ == "__main__":
    main()
