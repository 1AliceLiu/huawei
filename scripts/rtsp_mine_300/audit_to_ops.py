"""audit 报告 -> ops：只转高置信修框，其余进 residual 留人工。

输入：output/rtsp_mine_300/audit_r7/chunk_*.md + trial_12.md
输出：
  output/rtsp_mine_300/ops_audit_auto.jsonl  （audit_fix.py --apply 可吃）
  output/rtsp_mine_300/residual_audit.md      （人工在 X-AnyLabeling 里过）
规则（保守：宁可进 residual，不错删错改）：
  DUP（SAME/CROSS）：loser 必须与 keep 同图 IoU>0.25 否则 residual；
    明确“留一删一”但不知留谁 -> residual；“两者皆错+正确为X” -> residual。
  WRONG-del：须有无车依据（行人/工人/水马/护栏/空路面/工程机械/树/背景/噪声/
    栏杆/自行车/路人/撑伞/打伞/骑车/影子/挖机/指示牌）或 Motorcycle+人；
    纯 TINY 复核不动。
  WRONG-relabel：只转 Taxi / PLB GMB / Van / Private Car / HGV 五类，且
    HGV 须带重型依据（泥头/翻斗/大平板/牵引/吊机/自卸/渣土/大货/无集装箱/非集装箱）
    或源为 Container；LGV/MGV/Light Bus/Coach/Container 等互判全进 residual。
  同 file+idx 又 del 又 relabel -> del 胜（apply 侧同样规则，双保险）。
  MISSING -> residual（人工补框，脚本不自动加框）。

用法：python scripts/rtsp_mine_300/audit_to_ops.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from collect import box_iou  # noqa: E402

AUDIT_DIR = ROOT / "output/rtsp_mine_300/audit_r7"
JS_DIR = ROOT / "tmp/r7_pending/annotations_xany"
OUT_OPS = ROOT / "output/rtsp_mine_300/ops_audit_auto.jsonl"
OUT_RES = ROOT / "output/rtsp_mine_300/residual_audit.md"

NO_CAR = ("行人", "工人", "水马", "护栏", "空路面", "无车", "无车辆", "工程机械",
          "铲车", "树叶", "树木", "背景", "噪声", "栏杆", "自行车", "单车", "路人",
          "撑伞", "打伞", "骑车", "影子", "挖机", "指示牌", "吊机车", "路缘", "斑马线")
HEAVY = ("泥头", "翻斗", "大平板", "平板", "牵引", "吊机", "自卸", "渣土", "大货",
         "无集装箱", "非集装箱", "不是集装箱", "空板", "骨架", "大厢", "大型", "重型", "重卡")
AUTO_REL = {"Taxi", "PLB GMB", "Van", "Private Car", "HGV"}

IMG_RE = re.compile(r"^IMG\s+(\S+)\s+VERDICT")
TAG_RE = re.compile(r"-\s*\[([A-Z\-]+)\]")


SEP_RUN = r"(?:\s*[\/，,、]\s*\[?#?(?:box|框)?(\d+)\]?)"


def idxs_in(text: str) -> list:
    out = []
    for m in re.finditer(r"(?:#|box|框)\s*\[?#?(\d+)\]?", text):
        out.append(int(m.group(1)))
        tail = text[m.end():]
        m2 = re.match(SEP_RUN, tail)
        while m2:
            out.append(int(m2.group(1)))
            tail = tail[m2.end():]
            m2 = re.match(SEP_RUN, tail)
    for m in re.finditer(r"\[(\d+)\]", text):
        out.append(int(m.group(1)))
    seen, uniq = set(), []
    for i in out:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


def keeps_in(text: str) -> list:
    out = []
    for m in re.finditer(r"(?:留|保留)\s*(?:box|框)?\s*\[?#?(\d+)\]?", text):
        out.append(int(m.group(1)))
    return out


def dels_in(text: str) -> list:
    out = []
    for m in re.finditer(r"删\s*(?:box|框)?\s*\[?#?(\d+)\]?", text):
        out.append(int(m.group(1)))
        tail = text[m.end():]
        m2 = re.match(SEP_RUN, tail)
        while m2:
            out.append(int(m2.group(1)))
            tail = tail[m2.end():]
            m2 = re.match(SEP_RUN, tail)
    return out


def relabel_target(text: str):
    m = re.search(r"改\s*([A-Za-z ]+?)(?:\s|[，,。]|$)", text)
    if m:
        t = m.group(1).strip()
        if t and t not in ("标", "为"):
            return t
    m = re.search(r"(?:应为|正确为|判为)\s*([A-Za-z /]+?)(?:\s|[，,。]|$|\()", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"→\s*([A-Za-z /]+?)(?:\s|[，,。]|$|\()", text)
    if m:
        return m.group(1).strip()
    # “A被判成B”倒装：真值是 A（如 Van/小巴被判成大货车 -> Van）
    m = re.search(r"([A-Za-z]+)(?:/[A-Za-z\u4e00-\u9fff ]+?)?被判成", text)
    if m:
        return m.group(1).strip()
    return None


def nominated_label(text: str):
    """DUP 判X / 留X（只给类名不给号）：紧邻判/留/保留的 12 类名。"""
    for name in ("Franchised Bus", "Private Car", "PLB GMB", "Light Bus",
                 "Coach", "Container", "Motorcycle", "HGV", "LGV", "MGV",
                 "Van", "Taxi"):
        if re.search(r"(?:判|留|保留)\s*" + re.escape(name), text):
            return name
    return None


AMB_TGT = ("应为", "改", "→", "正确为")


def load_shapes(stem: str):
    jp = JS_DIR / (stem + ".json")
    d = json.loads(jp.read_text(encoding="utf-8"))
    return d.get("shapes", [])


def box_of(s):
    (x1, y1), (x2, y2) = s["points"]
    return [float(x1), float(y1), float(x2), float(y2)]


def main():
    ops, residual = [], []
    stat = defaultdict(int)
    files = sorted(AUDIT_DIR.glob("*.md"))
    for fp in files:
        cur = None
        for line in fp.read_text(encoding="utf-8").splitlines():
            m = IMG_RE.match(line.strip())
            if m:
                cur = m.group(1)
                continue
            m = TAG_RE.match(line.strip())
            if not m or not cur:
                continue
            tag, text = m.group(1), re.sub(
                r"\[[^\[\]]*,[^\[\]]*\]", "",
                line.strip().replace("同框", "同车"))
            stem = Path(cur).stem
            # 注：同框->同车、剥含逗号坐标括号，避免框号解析误食坐标数
            t = text
            try:
                shapes = load_shapes(stem)
            except FileNotFoundError:
                residual.append(f"- {cur}: 无 JSON，整行待人工 ({text})")
                stat["no-json"] += 1
                continue
            labels = [s.get("label") for s in shapes]

            def ok_idx(i):
                return 0 <= i < len(shapes)

            if tag == "MISSING":
                residual.append(f"- {cur}: MISSING 人工补框 ({text})")
                stat["missing"] += 1
                continue
            if tag == "TINY" and "删除" not in text and "删" not in text:
                stat["tiny-skip"] += 1
                continue

            if tag.startswith("DUP"):
                idxs = idxs_in(text)
                keeps = keeps_in(text)
                dels = dels_in(text)
                both_wrong = any(k in text for k in ("皆错", "都错", "两者皆错", "都删"))
                tgt0 = relabel_target(text)
                if both_wrong and tgt0:
                    residual.append(f"- {cur}: DUP 两者皆错需改类，人工 ({text})")
                    stat["dup-bothwrong"] += 1
                    continue
                if both_wrong and not dels:
                    dels = [i for i in idxs if ok_idx(i)]  # 建议都删：全删
                    keeps = []
                if not keeps and not dels:
                    # 判X正确 / 留X：按 JSON 实际标签对号入座
                    X = nominated_label(text)
                    if X and X in ("Coach", "Franchised Bus", "HGV", "LGV", "Light Bus",
                                   "MGV", "Motorcycle", "PLB GMB", "Private Car",
                                   "Taxi", "Van", "Container"):
                        cands = [i for i in idxs if ok_idx(i) and labels[i] == X]
                        if len(cands) == 1:
                            keeps = cands
                if not keeps and not dels:
                    residual.append(f"- {cur}: DUP 留删不明，人工 ({text})")
                    stat["dup-unclear"] += 1
                    continue
                if dels and not keeps:
                    keeps = [i for i in idxs if i not in dels and ok_idx(i)]
                if keeps and not dels:
                    dels = [i for i in idxs if i not in keeps]
                ref = keeps[0] if keeps else None
                group = [i for i in idxs if ok_idx(i)]
                good, bad = [], []
                for d_ in dels:
                    if not ok_idx(d_):
                        bad.append(d_)
                        continue
                    if ref is None:
                        good.append(d_)  # 都删 verdict：逐框皆错，无需重叠校验
                        continue
                    hit = False
                    for g in ([ref] if ref is not None else []) + [x for x in group if x != d_]:
                        if g is None or not ok_idx(g):
                            continue
                        try:
                            iou = box_iou(box_of(shapes[g]), box_of(shapes[d_]))
                        except Exception:  # noqa: BLE001
                            iou = 0.0
                        if iou > 0.25:
                            hit = True
                            break
                    if not hit and "同坐标" not in text and "完全相同" not in text:
                        bad.append(d_)
                        continue
                    good.append(d_)
                for d_ in good:
                    ops.append({"file": cur, "op": "del", "idx": d_,
                                "why": f"audit-dup keep={ref} ({fp.name})"})
                    stat["dup-del"] += 1
                for d_ in bad:
                    residual.append(f"- {cur}: DUP #{d_} 与 keep IoU 低/越界，人工 ({text})")
                    stat["dup-iou-low"] += 1
                # keep 框若被点名改类 -> 走 relabel 规则复核
                tgt = relabel_target(text)
                if tgt and keeps and tgt != labels[keeps[0]]:
                    residual.append(f"- {cur}: DUP keep #{keeps[0]} 疑需改 {tgt}，人工 ({text})")
                    stat["dup-keep-relabel"] += 1
                continue

            if tag == "WRONG":
                idxs = idxs_in(text)
                tgt = relabel_target(text)
                dels = dels_in(text)
                if tgt:
                    if tgt not in AUTO_REL:
                        residual.append(f"- {cur}: 改 {tgt} 非自动类，人工 ({text})")
                        stat["relabel-nonauto"] += 1
                        continue
                    for i in idxs:
                        if not ok_idx(i):
                            residual.append(f"- {cur}: 改类 idx 越界，人工 ({text})")
                            stat["relabel-badidx"] += 1
                            continue
                        if tgt == "HGV" and labels[i] in ("LGV", "MGV") and \
                                not any(k in text for k in HEAVY) and labels[i] != "Container":
                            residual.append(f"- {cur}: #{i} {labels[i]}->{tgt} 缺重型依据，人工 ({text})")
                            stat["relabel-noevidence"] += 1
                            continue
                        ops.append({"file": cur, "op": "relabel", "idx": i,
                                    "to": tgt, "why": f"audit ({fp.name})"})
                        stat["relabel"] += 1
                    continue
                if any(k in text for k in AMB_TGT) and not re.search(r"→\s*删除", text):
                    residual.append(f"- {cur}: WRONG 改类不明，人工 ({text})")
                    stat["wrong-unclear"] += 1
                    continue
                # WRONG 标签本身即 verdict：无替代类 = 删除
                targets = dels or idxs
                if not targets:
                    residual.append(f"- {cur}: WRONG 无框号，人工 ({text})")
                    stat["wrong-noidx"] += 1
                    continue
                for i in targets:
                    if not ok_idx(i):
                        residual.append(f"- {cur}: 删框 idx 越界，人工 ({text})")
                        stat["del-badidx"] += 1
                        continue
                    ops.append({"file": cur, "op": "del", "idx": i,
                                "why": f"audit-wrong ({fp.name})"})
                    stat["wrong-del"] += 1
                continue

            if tag == "TINY":  # 明确说删的 tiny
                for i in dels_in(text) or idxs_in(text):
                    if ok_idx(i):
                        ops.append({"file": cur, "op": "del", "idx": i,
                                    "why": f"audit-tiny ({fp.name})"})
                        stat["tiny-del"] += 1
                continue
            residual.append(f"- {cur}: 未知 TAG，人工 ({text})")
            stat["unknown"] += 1

    # 去重：同 file+idx+op 合并；同 file+idx del+relabel -> del 胜
    seen, uniq = set(), []
    for o in ops:
        k = (o["file"], o["idx"], o["op"])
        if k in seen:
            stat["dedup"] += 1
            continue
        seen.add(k)
        uniq.append(o)
    dels = {(o["file"], o["idx"]) for o in uniq if o["op"] == "del"}
    final = [o for o in uniq if not (o["op"] == "relabel" and (o["file"], o["idx"]) in dels)]
    stat["relabel-overridden"] = len(uniq) - len(final)

    with open(OUT_OPS, "w", encoding="utf-8") as f:
        for o in sorted(final, key=lambda x: (x["file"], x["idx"])):
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    with open(OUT_RES, "w", encoding="utf-8") as f:
        f.write(f"# R7 residual（人工在 X-AnyLabeling 里过，共 {len(residual)} 条）\n\n")
        for r in residual:
            f.write(r + "\n")
    print(f"ops={len(final)} (del={sum(1 for o in final if o['op']=='del')} "
          f"relabel={sum(1 for o in final if o['op']=='relabel')}) -> {OUT_OPS.name}")
    print(f"residual={len(residual)} -> {OUT_RES.name}")
    print("stat:", dict(sorted(stat.items())))


if __name__ == "__main__":
    main()
