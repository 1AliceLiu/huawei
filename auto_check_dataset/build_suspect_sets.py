import argparse
import glob
import json
import os
import shutil

import pandas as pd
from PIL import Image


def load_classes(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def convert_txt(txt_path, img_path, classes):
    W, H = Image.open(img_path).size
    shapes = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            cid = int(float(p[0]))
            xc, yc, w, h = map(float, p[1:5])
            x1 = (xc - w / 2) * W
            y1 = (yc - h / 2) * H
            x2 = (xc + w / 2) * W
            y2 = (yc + h / 2) * H
            shapes.append({
                "label": classes[cid], "score": None,
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "group_id": None, "description": "", "difficult": False,
                "shape_type": "rectangle", "flags": {}, "kie_linking": [],
            })
    return {
        "version": "4.0.0-beta.7", "flags": {}, "checked": False,
        "shapes": shapes, "imagePath": os.path.basename(img_path),
        "imageData": None, "imageHeight": H, "imageWidth": W, "description": "",
    }


def find_image(images_dir, stem):
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        p = os.path.join(images_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def process_tag(df, tag, out_root, classes):
    sub = df.drop_duplicates("source_image")
    di = os.path.join(out_root, tag, "images")
    dl = os.path.join(out_root, tag, "labels")
    os.makedirs(di, exist_ok=True)
    os.makedirs(dl, exist_ok=True)

    n_img = n_json = miss = 0
    for img in sub["source_image"]:
        stem = os.path.splitext(os.path.basename(img))[0]
        label = os.path.splitext(img.replace("/images/", "/labels/"))[0] + ".txt"
        if not (os.path.isfile(img) and os.path.isfile(label)):
            miss += 1
            continue
        dst_img = os.path.join(di, os.path.basename(img))
        shutil.copy2(img, dst_img)
        shutil.copy2(label, os.path.join(dl, stem + ".txt"))
        data = convert_txt(label, dst_img, classes)
        with open(os.path.join(di, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        n_img += 1
        n_json += 1

    print(f"[{tag}] 框 {len(df)} -> 唯一图 {len(sub)}，"
          f"复制 {n_img}，json {n_json}，缺失 {miss}")
    return n_img, n_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suspects", default="auto_check_dataset/clean_output/suspects/suspects.csv")
    ap.add_argument("--out-root", default="auto_check_dataset/suspect_select")
    ap.add_argument("--classes", default="dataset/Dahua2/classes.txt")
    ap.add_argument("--top-n", type=int, default=60, help="top_suspects 取 severity 前 N")
    args = ap.parse_args()

    classes = load_classes(args.classes)
    df = pd.read_csv(args.suspects, low_memory=False)

    reason_tags = ["tiny", "duplicate", "mislabel",
                   "outlier_centroid", "outlier_density"]
    for tag in reason_tags:
        process_tag(df[df["reasons"].str.contains(tag)], tag, args.out_root, classes)

    top = df.sort_values("severity", ascending=False).head(args.top_n)
    process_tag(top, "top_suspects", args.out_root, classes)


if __name__ == "__main__":
    main()
