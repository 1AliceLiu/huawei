"""R7 审核工具链 v0：audit 报告 -> 可复核的自动修框 -> 写回。

设计（audit 结论驱动，人只看 residual）：
1. 子 agent 逐图审计，产出报告 output/rtsp_mine_300/audit_r7/chunk_NN.md
   （每图 `IMG <file> VERDICT: OK|REVIEW` + `- [类型] 详情` 条目）
2. 人工把确认要修的条目转成 ops.jsonl（一行一条）：
     {"file": "xxx.jpg", "op": "del", "idx": 2, "why": "DUP-SAME Van#1"}
     {"file": "xxx.jpg", "op": "relabel", "idx": 0, "to": "Private Car", "why": "worker->Moto"}
   本脚本只做：备份 -> 应用 ops -> 写回 JSON -> 打印前后统计。
   安全规则：del 只删 idx 存在的框；relabel 的 to 必须在 classes.txt 内；
   同车异类二选一、漏框补框一律不自动做，留给人工。
3. dup 自动检（解决“同车叠多个同等框”）：same-class IoU>=T 的低 conf 者自动生成 ops
   （collect 落盘前已做 IoU>=0.85 去重，这里默认 T=0.85 只做复核；trial 若证明
   0.5~0.85 区间多为真重复，再放宽 --auto-iou 到 0.6，人工抽查）。

用法：
  python scripts/rtsp_mine_300/audit_fix.py --dir tmp/r7_pending --auto-iou 0.85 --emit-ops output/ops_auto.jsonl
  python scripts/rtsp_mine_300/audit_fix.py --dir tmp/r7_pending --apply output/ops_confirmed.jsonl
"""
import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from collect import box_iou  # noqa: E402


def load_shapes(js_path: Path):
    d = json.loads(js_path.read_text(encoding="utf-8"))
    return d, d.get("shapes", [])


def box_of(s):
    (x1, y1), (x2, y2) = s["points"]
    return [float(x1), float(y1), float(x2), float(y2)]


def auto_dup_ops(js_dir: Path, iou_thresh: float):
    """同类 IoU>=T 只保留高 conf，生成 del ops。返回 list[dict]。"""
    ops = []
    for jp in sorted(js_dir.glob("*.json")):
        _, shapes = load_shapes(jp)
        boxes = [(i, s.get("label"), box_of(s),
                  float((s.get("attributes") or {}).get("conf", 0) or 0))
                 for i, s in enumerate(shapes)]
        by_cls = {}
        for b in boxes:
            by_cls.setdefault(b[1], []).append(b)
        for cls, items in by_cls.items():
            items.sort(key=lambda t: -t[3])
            gone = set()
            for a in range(len(items)):
                if items[a][0] in gone:
                    continue
                for c in range(a + 1, len(items)):
                    if items[c][0] in gone:
                        continue
                    if box_iou(items[a][2], items[c][2]) >= iou_thresh:
                        gone.add(items[c][0])
                        ops.append({"file": jp.stem + ".jpg", "op": "del",
                                    "idx": items[c][0],
                                    "why": f"auto-dup {cls} IoU>= {iou_thresh} "
                                           f"(keep idx {items[a][0]} conf {items[a][3]})"})
    return ops


def apply_ops(d: Path, ops):
    names = (d / "classes.txt").read_text(encoding="utf-8").split()
    js_dir = d / "annotations_xany"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = d / f"annotations_xany_bak_{ts}"
    shutil.copytree(js_dir, bak)
    # index-space semantics: group per file, dels = set of ORIGINAL idx,
    # relabels keyed by ORIGINAL idx; rebuild once so order never matters.
    by_file = {}
    for o in ops:
        by_file.setdefault(o["file"], []).append(o)
    n_del = n_rel = n_skip = 0
    for fname, items in sorted(by_file.items()):
        jp = js_dir / (Path(fname).stem + ".json")
        if not jp.is_file():
            print(f"[skip] 无 JSON: {fname}")
            n_skip += len(items)
            continue
        payload, shapes = load_shapes(jp)
        dels, rels = set(), {}
        for o in items:
            idx = int(o["idx"])
            if not (0 <= idx < len(shapes)):
                print(f"[skip] idx 越界 {fname}#{idx}")
                n_skip += 1
                continue
            if o["op"] == "del":
                dels.add(idx)
            elif o["op"] == "relabel":
                if o["to"] not in names:
                    print(f"[skip] 非法类 {fname}#{idx} -> {o['to']}")
                    n_skip += 1
                    continue
                if idx in rels and rels[idx] != o["to"]:
                    print(f"[skip] 冲突 relabel {fname}#{idx}: {rels[idx]} vs {o['to']}")
                    n_skip += 1
                    continue
                rels[idx] = o["to"]
        # del wins over relabel on same idx (dup-keep note moot if box is wrong)
        for idx in sorted(dels & set(rels)):
            print(f"[note] {fname}#{idx} relabel->{rels.pop(idx)} 被 del 覆盖")
            n_skip += 1
        new_shapes = []
        for i, s in enumerate(shapes):
            if i in dels:
                continue
            if i in rels:
                s = dict(s)
                s["label"] = rels[i]
                n_rel += 1
            new_shapes.append(s)
        n_del += len(dels)
        payload["shapes"] = new_shapes
        jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"applied files={len(by_file)} del={n_del} relabel={n_rel} skip={n_skip}, backup -> {bak.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="tmp/r7_pending")
    ap.add_argument("--auto-iou", type=float, default=0.85)
    ap.add_argument("--emit-ops", default=None)
    ap.add_argument("--apply", default=None)
    args = ap.parse_args()
    d = ROOT / args.dir
    if args.emit_ops:
        ops = auto_dup_ops(d / "annotations_xany", args.auto_iou)
        with open(args.emit_ops, "w", encoding="utf-8") as f:
            for o in ops:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        print(f"auto-dup IoU>={args.auto_iou}: {len(ops)} ops -> {args.emit_ops}")
        by_cls = {}
        for o in ops:
            by_cls[o["why"].split()[1]] = by_cls.get(o["why"].split()[1], 0) + 1
        print("by-class:", by_cls)
    if args.apply:
        ops = [json.loads(l) for l in open(args.apply, encoding="utf-8") if l.strip()]
        apply_ops(d, ops)


if __name__ == "__main__":
    main()
