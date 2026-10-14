#!/usr/bin/env python3
"""框级审查工具后端（方法 C）。

扫描 ``suspect_select/<tag>/items/*.json``（由 ``build_items.py`` 生成，也可手动添加），
提供：

* ``GET  /``                       审查页面
* ``GET  /api/state[?refresh=1]``  类别与待审队列
* ``GET  /api/image?id=&view=``    三联视图 tight / context / full
* ``POST /api/verdict``            提交判定，删除该框条目并写入 clean_data
* ``POST /api/undo``               撤销上一步

判定语义
--------
* ``keep``      : 保留原类，clean_data 保持原样
* ``delete``    : 从 clean_data 的该图 label 中删除这一行
* ``relabel``   : 把该行的类别改为 new_cls

clean_data 结构::

    clean_data/
      labels/<source_stem>.txt   # 原图 label 经判定修正后
      images/<source_image>      # 首次触达时复制一份
      decisions.csv              # 审查流水

用法::

    python auto_check_dataset/review_tool/server.py --port 8765
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))

TARGET = (0, 0, 255)     # BGR 红
OTHER = (150, 150, 150)  # BGR 灰
LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# 索引
# --------------------------------------------------------------------------- #
class Store:
    def __init__(self, suspect_root: str, clean_dir: str, classes: list[str]):
        self.suspect_root = suspect_root
        self.clean_dir = clean_dir
        self.classes = classes
        self.items: dict[str, list[tuple[str, str, dict]]] = {}   # id -> [(tag, path, data)]
        self._labels: dict[str, list[str]] = {}
        self.undo_stack: list[dict] = []
        os.makedirs(os.path.join(clean_dir, "labels"), exist_ok=True)
        os.makedirs(os.path.join(clean_dir, "images"), exist_ok=True)
        self.load()

    # ---- 扫描 ----
    def load(self) -> None:
        self.items = {}
        if not os.path.isdir(self.suspect_root):
            return
        for tag in sorted(os.listdir(self.suspect_root)):
            idir = os.path.join(self.suspect_root, tag, "items")
            if not os.path.isdir(idir):
                continue
            for name in sorted(os.listdir(idir)):
                if not name.endswith(".json"):
                    continue
                path = os.path.join(idir, name)
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue
                iid = data.get("item_id") or os.path.splitext(name)[0]
                self.items.setdefault(iid, []).append((tag, path, data))

    def decided_ids(self) -> set[str]:
        path = os.path.join(self.clean_dir, "decisions.csv")
        if not os.path.isfile(path):
            return set()
        with open(path, newline="", encoding="utf-8") as f:
            return {r["item_id"] for r in csv.DictReader(f)}

    def state(self) -> dict:
        decided = self.decided_ids()
        cats = {}
        queue = []
        for iid, copies in self.items.items():
            if iid in decided:
                continue
            tag, _path, data = copies[0]
            cats.setdefault(tag, 0)
            cats[tag] += 1
            queue.append(self._brief(iid, copies))
        queue.sort(key=lambda x: (-(x["severity"] or 0), x["id"]))
        tag_list = [{"name": t, "count": c} for t, c in sorted(cats.items(), key=lambda kv: -kv[1])]
        return {
            "classes": self.classes,
            "tags": tag_list,
            "total": len(queue),
            "queue": queue,
        }

    @staticmethod
    def _brief(iid: str, copies: list) -> dict:
        tags = [c[0] for c in copies]
        d = copies[0][2]
        return {
            "id": iid,
            "tags": tags,
            "tag": tags[0],
            "cls_id": d.get("cls_id"),
            "cls_name": d.get("cls_name"),
            "reasons": d.get("reasons"),
            "severity": d.get("severity"),
            "confidence": d.get("confidence"),
            "suggested_action": d.get("suggested_action"),
            "relabel_target": d.get("relabel_target"),
            "nearest_class": d.get("nearest_class"),
            "nearest_class_frac": d.get("nearest_class_frac"),
            "nn_consistency": d.get("nn_consistency"),
            "source_stem": d.get("source_stem"),
            "box_idx": d.get("box_idx"),
        }

    # ---- label 读写 ----
    def _read_label(self, path: str) -> list[str]:
        if path in self._labels:
            return list(self._labels[path])
        lines: list[str] = []
        if path and os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if len(ln.split()) >= 5]
        self._labels[path] = lines
        return list(lines)

    def _clean_label_path(self, source_stem: str) -> str:
        return os.path.join(self.clean_dir, "labels", source_stem + ".txt")

    @staticmethod
    def _img_size(src: str):
        img = cv2.imread(src)
        if img is None:
            return None, None, None
        h, w = img.shape[:2]
        return img, w, h

    def _target_index(self, lines: list[str], data: dict, W: int, H: int) -> int | None:
        if not lines:
            return None
        xc = (data["x1"] + data["x2"]) / 2.0 / max(W, 1)
        yc = (data["y1"] + data["y2"]) / 2.0 / max(H, 1)
        best, best_d = None, 1e9
        for i, ln in enumerate(lines):
            p = ln.split()
            try:
                cx, cy = float(p[1]), float(p[2])
            except (IndexError, ValueError):
                continue
            d = (cx - xc) ** 2 + (cy - yc) ** 2
            if d < best_d:
                best, best_d = i, d
        if best is None:
            return None
        if best_d > 0.02 and 0 <= data.get("box_idx", -1) < len(lines):
            return data["box_idx"]
        return best

    # ---- 判定 ----
    def apply(self, iid: str, action: str, new_cls: int | None) -> dict:
        with LOCK:
            copies = self.items.get(iid)
            if not copies:
                return {"ok": False, "error": "item not found"}
            tag, path, data = copies[0]
            src = data["source_image"]
            stem = data["source_stem"]
            orig_label = data["label_path"]
            clean_label = self._clean_label_path(stem)

            undo = {"item_files": [], "clean_label": clean_label,
                    "clean_before": None, "image": None, "decision": iid}

            if not os.path.isfile(clean_label):
                if orig_label and os.path.isfile(orig_label):
                    shutil.copy2(orig_label, clean_label)
                else:
                    with open(clean_label, "w", encoding="utf-8"):
                        pass
            with open(clean_label, encoding="utf-8") as f:
                undo["clean_before"] = f.read()

            # 复制源图（首次）
            dst_img = os.path.join(self.clean_dir, "images", os.path.basename(src))
            if os.path.isfile(src) and not os.path.isfile(dst_img):
                shutil.copy2(src, dst_img)
                undo["image"] = dst_img

            if action in ("delete", "relabel"):
                img, W, H = self._img_size(src)
                lines = self._read_label(clean_label)
                idx = self._target_index(lines, data, W or 1, H or 1)
                if idx is not None:
                    if action == "delete":
                        lines.pop(idx)
                    else:
                        p = lines[idx].split()
                        p[0] = str(int(new_cls))
                        lines[idx] = " ".join(p)
                    with open(clean_label, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines) + ("\n" if lines else ""))
                    self._labels.pop(clean_label, None)

            # 记录判定
            dec_path = os.path.join(self.clean_dir, "decisions.csv")
            new_file = not os.path.isfile(dec_path)
            with open(dec_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["item_id", "tag", "source_stem", "box_idx",
                                "old_cls", "action", "new_cls", "time"])
                w.writerow([iid, tag, stem, data.get("box_idx"),
                            data.get("cls_id"), action,
                            "" if new_cls is None else new_cls,
                            time.strftime("%Y-%m-%d %H:%M:%S")])

            # 删除该框条目（含其它类别下的同名副本）
            for _t, p, _d in copies:
                if os.path.isfile(p):
                    with open(p, encoding="utf-8") as f:
                        undo["item_files"].append((p, f.read()))
                    os.remove(p)
            self.items.pop(iid, None)
            self.undo_stack.append(undo)
            return {"ok": True, "remaining": sum(len(v) for v in self.items.values())}

    def undo(self) -> dict:
        with LOCK:
            if not self.undo_stack:
                return {"ok": False, "error": "nothing to undo"}
            u = self.undo_stack.pop()
            for p, content in u["item_files"]:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    f.write(content)
                try:
                    data = json.loads(content)
                    iid = data.get("item_id")
                    tag = os.path.basename(os.path.dirname(os.path.dirname(p)))
                    self.items.setdefault(iid, []).append((tag, p, data))
                except Exception:
                    pass
            if u["clean_before"] is not None:
                with open(u["clean_label"], "w", encoding="utf-8") as f:
                    f.write(u["clean_before"])
                self._labels.pop(u["clean_label"], None)
            if u["image"] and os.path.isfile(u["image"]):
                os.remove(u["image"])
            # 删除 decisions 最后一行
            dec_path = os.path.join(self.clean_dir, "decisions.csv")
            if os.path.isfile(dec_path):
                with open(dec_path, encoding="utf-8") as f:
                    rows = f.readlines()
                if len(rows) > 1:
                    with open(dec_path, "w", encoding="utf-8") as f:
                        f.writelines(rows[:-1])
            return {"ok": True}

    # ---- 图像渲染 ----
    def render(self, iid: str, view: str) -> bytes | None:
        copies = self.items.get(iid)
        if not copies:
            return None
        data = copies[0][2]
        src = data["source_image"]
        img = cv2.imread(src)
        if img is None:
            return None
        H, W = img.shape[:2]
        x1 = int(max(0, min(W - 1, data["x1"])))
        y1 = int(max(0, min(H - 1, data["y1"])))
        x2 = int(max(0, min(W, data["x2"])))
        y2 = int(max(0, min(H, data["y2"])))

        if view == "tight":
            pad_x = int(0.06 * (x2 - x1)) + 2
            pad_y = int(0.06 * (y2 - y1)) + 2
            c = img[max(0, y1 - pad_y):min(H, y2 + pad_y),
                    max(0, x1 - pad_x):min(W, x2 + pad_x)].copy()
            return self._encode(self._fit(c, 420))
        if view == "context":
            m = int(max(x2 - x1, y2 - y1) * 1.0) + 12
            a, b = max(0, x1 - m), max(0, y1 - m)
            cc, dd = min(W, x2 + m), min(H, y2 + m)
            c = img[b:dd, a:cc].copy()
            cv2.rectangle(c, (x1 - a, y1 - b), (x2 - a, y2 - b), TARGET, 2)
            return self._encode(self._fit(c, 420))
        # full
        c = img.copy()
        for ln in self._read_label(data.get("label_path")):
            p = ln.split()
            try:
                cid = int(float(p[0])); cx, cy, bw, bh = map(float, p[1:5])
            except ValueError:
                continue
            a = int((cx - bw / 2) * W); b = int((cy - bh / 2) * H)
            e = int((cx + bw / 2) * W); d = int((cy + bh / 2) * H)
            cv2.rectangle(c, (a, b), (e, d), OTHER, 1)
        cv2.rectangle(c, (x1, y1), (x2, y2), TARGET, 3)
        label = f"{data.get('cls_name', '')}  [{x2 - x1}x{y2 - y1}]"
        ty = max(18, y1 - 8)
        cv2.putText(c, label, (max(2, x1), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(c, label, (max(2, x1), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    TARGET, 2, cv2.LINE_AA)
        return self._encode(self._fit(c, 1100))

    @staticmethod
    def _fit(img, max_side):
        h, w = img.shape[:2]
        s = min(1.0, max_side / max(h, w, 1))
        if s < 1.0:
            img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                             interpolation=cv2.INTER_AREA)
        return img

    @staticmethod
    def _encode(img) -> bytes:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buf.tobytes() if ok else b""


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
STORE: Store = None
INDEX_HTML = os.path.join(HERE, "index.html")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, "application/json; charset=utf-8",
                   json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            with open(INDEX_HTML, "rb") as f:
                return self._send(200, "text/html; charset=utf-8", f.read())
        if u.path == "/api/state":
            if q.get("refresh"):
                STORE.load()
            return self._json(STORE.state())
        if u.path == "/api/image":
            iid = q.get("id", [""])[0]
            view = q.get("view", ["full"])[0]
            data = STORE.render(iid, view)
            if data is None:
                return self._send(404, "text/plain", b"not found")
            return self._send(200, "image/jpeg", data)
        return self._send(404, "text/plain", b"not found")

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        if u.path == "/api/verdict":
            iid = body.get("id", "")
            action = body.get("action", "keep")
            new_cls = body.get("new_cls")
            res = STORE.apply(iid, action, new_cls)
            return self._json(res, 200 if res.get("ok") else 400)
        if u.path == "/api/undo":
            return self._json(STORE.undo())
        return self._send(404, "text/plain", b"not found")


def load_classes(path: str) -> list[str]:
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--suspect-root", default="auto_check_dataset/suspect_select")
    ap.add_argument("--clean-dir", default="auto_check_dataset/clean_data")
    ap.add_argument("--classes", default="dataset/Dahua2/classes.txt")
    ap.add_argument("--refresh", action="store_true", help="启动时重新扫描（默认即扫描）")
    args = ap.parse_args()

    def ap_join(p):
        return p if os.path.isabs(p) else os.path.join(REPO, p)

    global STORE
    STORE = Store(ap_join(args.suspect_root), ap_join(args.clean_dir),
                  load_classes(ap_join(args.classes)))
    print(f"待审框条目 {len(STORE.items)} 个（类别 {len({c[0] for v in STORE.items.values() for c in v})}）")
    print(f"clean_data -> {ap_join(args.clean_dir)}")
    print(f"打开 http://{args.host}:{args.port}/  （Ctrl+C 退出）")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
