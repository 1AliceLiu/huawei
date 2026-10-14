#!/usr/bin/env python3
"""去除视频帧/抓拍图上烧录的时间戳 OSD（on-screen display）。

思路：OSD 对同一路视频是**固定位置**的白色（低饱和、高亮）文字。
本脚本在候选角区里自动检测这行文字，生成掩膜后用 inpaint 抹除，
再写回同尺寸图片（坐标不变，因此原有标注框不受影响）。

两种用法
--------
1) 角区自动检测（适合整目录批量）::

    python remove_osd.py --input dataset/batch_12.v8i.yolov11/test/images \
        --out dataset/_no_osd --corner TL

2) 精确指定归一化区域 x0,y0,x1,y1（适合某相机固定 OSD）::

    python remove_osd.py --input img.jpg --box 0,0,0.17,0.055 --out out/

可选三选一的方法：ns（默认，效果较好）/ telea / inpaint_ns。单图默认把结果存到
仓库根目录，方便快速验证。
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 各角区候选范围（归一化 x0,y0,x1,y1）
CORNERS = {
    "TL": (0.0, 0.0, 0.45, 0.10),
    "TR": (0.55, 0.0, 1.0, 0.10),
    "BL": (0.0, 0.90, 0.45, 1.0),
    "BR": (0.55, 0.90, 1.0, 1.0),
}


def detect_mask(im: np.ndarray, box: tuple[float, float, float, float],
                thr: int = 160, sat_thr: int = 100, pad: int = 3) -> np.ndarray:
    """在 box（归一化）内检测白色文字，返回全图 uint8 掩膜。"""
    H, W = im.shape[:2]
    x0, y0, x1, y1 = (int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H))
    x1, y1 = max(x0 + 1, x1), max(y0 + 1, y1)
    roi = im[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    white = ((gray > thr) & (hsv[:, :, 1] < sat_thr)).astype(np.uint8) * 255
    joined = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 17), np.uint8))

    # 只保留像「一行文字」的连通块，取其内的白色笔画（而非整块矩形，避免大面积涂抹）
    glyphs = np.zeros_like(white)
    cnts, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = 0
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw > 0.03 * W and bw < 0.75 * W and 0.006 * H < bh < 0.10 * H:
            glyphs[by:by + bh, bx:bx + bw] = white[by:by + bh, bx:bx + bw]
            keep += 1
    if keep == 0:
        return np.zeros((H, W), np.uint8)
    # 适度膨胀以覆盖文字的抗锯齿边缘与描边
    k = 2 * max(1, pad) + 1
    glyphs = cv2.dilate(glyphs, np.ones((k, k), np.uint8), iterations=1)
    mask = np.zeros((H, W), np.uint8)
    mask[y0:y1, x0:x1] = glyphs
    return mask


def remove_osd(im: np.ndarray, mask: np.ndarray, method: str = "ns", radius: int = 4) -> np.ndarray:
    if mask is None or not mask.any():
        return im
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(im, mask, radius, flag)


def iter_inputs(paths: list[str], recursive: bool) -> list[str]:
    out = []
    for p in paths:
        if os.path.isdir(p):
            pat = "**/*" if recursive else "*"
            for f in glob.glob(os.path.join(p, pat), recursive=recursive):
                if f.lower().endswith(IMG_EXTS):
                    out.append(f)
        elif os.path.isfile(p):
            out.append(p)
        else:
            out.extend(f for f in glob.glob(p) if f.lower().endswith(IMG_EXTS))
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True, help="图片/目录/通配符")
    ap.add_argument("--out", default=None,
                    help="输出目录；单文件且未指定时存到仓库根目录")
    ap.add_argument("--corner", default="TL", choices=list(CORNERS), help="候选角区")
    ap.add_argument("--box", default=None, help="归一化 x0,y0,x1,y1，覆盖 --corner")
    ap.add_argument("--method", default="ns", choices=["ns", "telea"])
    ap.add_argument("--radius", type=int, default=4, help="inpaint 半径")
    ap.add_argument("--thr", type=int, default=160, help="白色亮度阈值")
    ap.add_argument("--sat-thr", type=int, default=100, help="白色饱和度上限")
    ap.add_argument("--pad", type=int, default=3, help="笔画掩膜膨胀半径（覆盖描边）")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--suffix", default="_noosd", help="输出文件名后缀")
    ap.add_argument("--name", default=None, help="单图输出文件名（默认存到仓库根目录）")
    ap.add_argument("--preview", action="store_true", help="额外保存 *_preview.jpg（原图/结果对比）")
    args = ap.parse_args()

    box = tuple(float(v) for v in args.box.split(",")) if args.box else CORNERS[args.corner]
    files = iter_inputs(args.input, args.recursive)
    if not files:
        raise SystemExit("未找到输入图片")

    repo_root = os.path.dirname(os.path.abspath(__file__))
    single = len(files) == 1 and args.out is None
    if args.name:
        out_dir = args.out or repo_root
    elif single:
        out_dir = repo_root
    else:
        out_dir = args.out or os.path.join(repo_root, "_no_osd")
    os.makedirs(out_dir, exist_ok=True)

    done = hit = 0
    for p in files:
        im = cv2.imread(p)
        if im is None:
            print("读取失败:", p)
            continue
        mask = detect_mask(im, box, args.thr, args.sat_thr, args.pad)
        out = remove_osd(im, mask, args.method, args.radius)
        name = args.name or (os.path.splitext(os.path.basename(p))[0] + args.suffix + ".jpg")
        dst = os.path.join(out_dir, name)
        cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if mask.any():
            hit += 1
        done += 1
        if args.preview:
            band = np.vstack([im[:120, :520], np.full((6, 520, 3), 200, np.uint8), out[:120, :520]])
            cv2.imwrite(os.path.join(out_dir, os.path.splitext(name)[0] + "_preview.jpg"),
                        cv2.resize(band, (1040, band.shape[0] * 2)))
        print(f"[{done}/{len(files)}] {'命中' if mask.any() else '未检出'} -> {dst}")

    print(f"\n完成 {done} 张（其中检出 OSD {hit} 张），输出目录: {out_dir}")
    if single:
        print(f"验证样例已保存到仓库根目录: {os.path.join(out_dir, os.path.splitext(os.path.basename(files[0]))[0] + args.suffix + '.jpg')}")


if __name__ == "__main__":
    main()
