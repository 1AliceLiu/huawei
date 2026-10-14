"""Build a group-disjoint train/val/test split with a per-class box ratio target.

Sibling of ``build_unified_val.py``. That script splits by *image count* (with a
small per-class floor) and is not proportional on a per-class basis. This script
keeps the same leakage-safe property -- source groups (camera / video segment) are
atomic, so no camera appears in two splits -- but additionally tries to match a
per-class box-count ratio.

Default target (matching the requested ``train_pool:frozen_test:frozen_val = 7:1:2``):

    train 70%   test 10%   val 20%

of every class's boxes. Groups are assigned to a single split; the metadata-less
``NUM`` group is forced to train so it can never leak into the frozen sets.

Because groups are atomic and have irregular per-class contents, an exact ratio is
not always reachable. The assignment is found with simulated annealing (multiple
restarts) minimising the sum of absolute per-class ratio deviations; the achieved
ratios and max deviation are written to ``report.json`` and printed.

Usage
-----
    python scripts/build_splits11.py
    python scripts/build_splits11.py --test-ratio 0.1 --val-ratio 0.2 --iters 600000 --restarts 6
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_unified_val import NAMES, load_items, profile_groups  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DS = ROOT / "dataset" / "batch_12.v14i.merged12.yolov11"
TRAIN, TEST, VAL = 0, 1, 2


def optimise(groups: dict, test_ratio: float, val_ratio: float, iters: int,
             restarts: int, seed: int, forbidden: set[str]) -> tuple[list[int], list[str]]:
    keys = sorted(groups)
    boxes = [[groups[k]["boxes"].get(c, 0) for c in range(len(NAMES))] for k in keys]
    sizes = [groups[k]["size"] for k in keys]
    total = [sum(b[c] for b in boxes) for c in range(len(NAMES))]
    n = len(keys)
    fixed = {keys.index(f) for f in forbidden if f in groups}

    movable = [i for i in range(n) if i not in fixed]
    best_assign: list[int] = [TRAIN] * n
    best_err = float("inf")

    for r in range(restarts):
        rng = random.Random(seed + r)
        assign = [TRAIN] * n
        ct = [0] * len(NAMES)
        cv = [0] * len(NAMES)

        def add(i: int, s: int) -> None:
            if s == TEST:
                for c, v in enumerate(boxes[i]):
                    ct[c] += v
            elif s == VAL:
                for c, v in enumerate(boxes[i]):
                    cv[c] += v

        def rem(i: int, s: int) -> None:
            if s == TEST:
                for c, v in enumerate(boxes[i]):
                    ct[c] -= v
            elif s == VAL:
                for c, v in enumerate(boxes[i]):
                    cv[c] -= v

        def err() -> float:
            e = 0.0
            for c in range(len(NAMES)):
                if total[c]:
                    e += abs(ct[c] / total[c] - test_ratio)
                    e += abs(cv[c] / total[c] - val_ratio)
            return e

        cur = err()
        local_best, local_assign = cur, assign[:]
        t0, t1 = 0.02, 0.0003
        for it in range(iters):
            temp = t0 * (t1 / t0) ** (it / iters)
            i = rng.choice(movable)
            if rng.random() < 0.8:
                old = assign[i]
                new = rng.randrange(3)
                if new == old:
                    continue
                rem(i, old)
                add(i, new)
                assign[i] = new
                ne = err()
                if ne <= local_best:
                    local_best, local_assign = ne, assign[:]
                if ne <= cur or rng.random() < math.exp((cur - ne) / temp):
                    cur = ne
                else:
                    rem(i, new)
                    add(i, old)
                    assign[i] = old
            else:
                j = rng.choice(movable)
                if assign[i] == assign[j] or {assign[i], assign[j]} != {TEST, VAL}:
                    continue
                si, sj = assign[i], assign[j]
                rem(i, si)
                rem(j, sj)
                add(i, sj)
                add(j, si)
                assign[i], assign[j] = sj, si
                ne = err()
                if ne <= local_best:
                    local_best, local_assign = ne, assign[:]
                if ne <= cur or rng.random() < math.exp((cur - ne) / temp):
                    cur = ne
                else:
                    rem(i, sj)
                    rem(j, si)
                    add(i, si)
                    add(j, sj)
                    assign[i], assign[j] = si, sj

        if local_best < best_err:
            best_err, best_assign = local_best, local_assign
        print(f"[restart {r}] err={local_best:.4f}  (best={best_err:.4f})", flush=True)

    return best_assign, keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DS)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <dataset>/splits1.1")
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--iters", type=int, default=600_000)
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = args.dataset.resolve()
    out = (args.out_dir or (ds / "splits1.1")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    items = load_items(ds)
    groups = profile_groups(items)
    print(f"images={len(items)} groups={len(groups)}")

    assign, keys = optimise(groups, args.test_ratio, args.val_ratio, args.iters,
                            args.restarts, args.seed, forbidden={"NUM"})

    split_of = {keys[i]: assign[i] for i in range(len(keys))}
    lists = {TRAIN: [], TEST: [], VAL: []}
    for it in items:
        lists[split_of[it["group"]]].append(it["path"])
    for v in lists.values():
        v.sort()

    names = {TRAIN: "train_pool", TEST: "frozen_test", VAL: "frozen_val"}
    for k, name in names.items():
        (out / f"{name}.txt").write_text("\n".join(lists[k]) + "\n", encoding="utf-8")

    # per-class achieved counts / ratios
    total = Counter()
    counts = {TRAIN: Counter(), TEST: Counter(), VAL: Counter()}
    for it in items:
        total.update(it["boxes"])
        counts[split_of[it["group"]]].update(it["boxes"])

    rows = []
    max_dev = 0.0
    for c, name in enumerate(NAMES):
        t = total[c] or 1
        tr, te, va = counts[TRAIN][c], counts[TEST][c], counts[VAL][c]
        rtr, rte, rva = tr / t, te / t, va / t
        max_dev = max(max_dev, abs(rte - args.test_ratio), abs(rva - args.val_ratio))
        rows.append({"class": name, "train": tr, "test": te, "val": va,
                     "train_ratio": round(rtr, 4), "test_ratio": round(rte, 4),
                     "val_ratio": round(rva, 4)})

    img_counts = {names[k]: len(lists[k]) for k in names}
    target_train = 1.0 - args.test_ratio - args.val_ratio
    report = {
        "dataset": str(ds),
        "out_dir": str(out),
        "target": {"train": target_train, "test": args.test_ratio, "val": args.val_ratio},
        "group_disjoint": True,
        "num_forced_to_train": "NUM" in groups,
        "images": img_counts,
        "max_class_ratio_deviation": round(max_dev, 4),
        "per_class": rows,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {"dataset": str(ds), "seed": args.seed, "num_groups": len(groups),
                "items": [{"path": it["path"], "group": it["group"], "boxes": it["boxes"],
                           "split": names[split_of[it["group"]]]} for it in items]}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    names_lit = "[" + ", ".join(f"'{n}'" for n in NAMES) + "]"
    (out / "data_train_pool.yaml").write_text(
        "# train = group-disjoint pool, val/test = frozen sets\n"
        "# per-class box ratio train:test:val = "
        f"{target_train:g}:{args.test_ratio:g}:{args.val_ratio:g}\n"
        "# regenerate with scripts/build_splits11.py\n"
        f"path: {ds}\n"
        f"train: {out / 'train_pool.txt'}\n"
        f"val: {out / 'frozen_val.txt'}\n"
        f"test: {out / 'frozen_test.txt'}\n\n"
        f"nc: {len(NAMES)}\nnames: {names_lit}\n",
        encoding="utf-8",
    )

    print(f"\nout dir     : {out}")
    print(f"images      : {img_counts}")
    print(f"max class ratio deviation : {max_dev:.4f}")
    print(f"\n{'class':<15}{'train':>8}{'test':>7}{'val':>7} | {'train':>6}{'test':>6}{'val':>6}")
    for row in rows:
        print(f"{row['class']:<15}{row['train']:>8}{row['test']:>7}{row['val']:>7} | "
              f"{row['train_ratio']:>6.2f}{row['test_ratio']:>6.2f}{row['val_ratio']:>6.2f}")


if __name__ == "__main__":
    main()
