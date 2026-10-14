import argparse
import csv
import os

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import umap
import yaml
from PIL import Image
from sklearn.cluster import KMeans
from transformers import AutoImageProcessor, AutoModel

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


# ================= 0. 环境与模型 =================
def resolve_device(device=None):
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_dino(model_name='facebook/dinov2-small', device=None):
    device = resolve_device(device)
    print(f"加载模型 {model_name} -> {device}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model, device


# ================= 1. ROI 提取与预处理 =================
def _index_images(split_dir):
    images_dir = os.path.join(split_dir, "images")
    if not os.path.isdir(images_dir):
        return {}
    index = {}
    for name in os.listdir(images_dir):
        if name.lower().endswith(IMG_EXTS):
            index[os.path.splitext(name)[0]] = os.path.join(images_dir, name)
    return index


def crop_dataset(dataset_dir, out_dir, splits=("train", "valid", "test"),
                 min_size=16, pad_ratio=0.0, limit=None):
    """阶段一：读取 YOLO 数据集，按标注裁剪 ROI 并落盘。"""
    os.makedirs(out_dir, exist_ok=True)

    names = []
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if os.path.isfile(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            names = (yaml.safe_load(f) or {}).get("names", []) or []

    manifest = []
    total_boxes = 0
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            continue
        img_index = _index_images(split_dir)
        labels_dir = os.path.join(split_dir, "labels")
        if not os.path.isdir(labels_dir):
            continue

        for lbl_name in sorted(os.listdir(labels_dir)):
            if not lbl_name.lower().endswith(".txt"):
                continue
            stem = os.path.splitext(lbl_name)[0]
            img_path = img_index.get(stem)
            if img_path is None:
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            H, W = img.shape[:2]

            with open(os.path.join(labels_dir, lbl_name), "r", encoding="utf-8") as f:
                lines = [ln.split() for ln in f if len(ln.split()) >= 5]

            for idx, parts in enumerate(lines):
                total_boxes += 1
                cls_id = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = int((xc - bw / 2) * W)
                y1 = int((yc - bh / 2) * H)
                x2 = int((xc + bw / 2) * W)
                y2 = int((yc + bh / 2) * H)

                pad = int(pad_ratio * max(x2 - x1, y2 - y1))
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(W, x2 + pad), min(H, y2 + pad)

                cw, ch = x2 - x1, y2 - y1
                status = "valid"
                if cw < min_size or ch < min_size:
                    status = "too_small"

                cls_name = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
                out_name = f"{split}_{stem}_{idx}_c{cls_id}.jpg"
                if status == "valid":
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    cv2.imwrite(os.path.join(out_dir, out_name), crop,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])

                manifest.append({
                    "file": out_name, "status": status, "split": split,
                    "source_image": img_path, "cls_id": cls_id, "cls_name": cls_name,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": cw, "h": ch,
                })

                if limit is not None and len(manifest) >= limit:
                    break
            if limit is not None and len(manifest) >= limit:
                break
        if limit is not None and len(manifest) >= limit:
            break

    manifest_path = os.path.join(out_dir, "crop_manifest.csv")
    fields = ["file", "status", "split", "source_image", "cls_id", "cls_name",
              "x1", "y1", "x2", "y2", "w", "h"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in manifest:
            w.writerow(row)

    kept = sum(1 for m in manifest if m["status"] == "valid")
    print(f"[阶段一] 标注框总数 {total_boxes}，落盘 {kept}，过滤(tiny_noise) {len(manifest) - kept}")
    print(f"[阶段一] crop 目录: {out_dir}  清单: {manifest_path}")
    return manifest


def _load_manifest(crop_dir):
    path = os.path.join(crop_dir, "crop_manifest.csv")
    if os.path.isfile(path):
        df = pd.read_csv(path)
        cols = ["file", "cls_id", "cls_name", "split", "source_image",
                "x1", "y1", "x2", "y2"]
        return df[df["status"] == "valid"][cols]
    return None


# ================= 2. 增强型特征提取（批处理） =================
def extract_features_enhanced(crop_dir, processor, model, device, min_size=16,
                              batch_size=64, manifest_df=None):
    files = [f for f in os.listdir(crop_dir) if f.lower().endswith(IMG_EXTS)]
    print(f"开始处理 {len(files)} 张切图...")

    embeddings = []
    metas = []

    def flush(imgs, mrows):
        if not imgs:
            return
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        pooler = getattr(outputs, "pooler_output", None)
        feat = pooler if pooler is not None else outputs.last_hidden_state[:, 0]
        feat = feat.cpu().numpy().astype(np.float32)
        feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
        embeddings.extend(feat)
        metas.extend(mrows)

    batch_imgs, batch_meta = [], []
    for fname in files:
        path = os.path.join(crop_dir, fname)
        try:
            img = Image.open(path).convert('RGB')
            w, h = img.size
            if w < min_size or h < min_size:
                continue
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            batch_imgs.append(img)
            batch_meta.append({'path': path, 'file': fname, 'status': 'valid',
                               'width': w, 'height': h, 'laplacian_var': lap_var})
            if len(batch_imgs) >= batch_size:
                flush(batch_imgs, batch_meta)
                batch_imgs, batch_meta = [], []
        except Exception as e:
            print(f"读取图片失败 {path}: {e}")
    flush(batch_imgs, batch_meta)

    meta_df = pd.DataFrame(metas)
    if manifest_df is not None and not meta_df.empty:
        meta_df = meta_df.merge(manifest_df, on="file", how="left")
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32), meta_df
    print(f"特征提取完成: {len(embeddings)} 个向量, 维度 {len(embeddings[0])}")
    return np.array(embeddings, dtype=np.float32), meta_df


# ================= 3. 聚类、原型采样与 UMAP 可视化 =================
def cluster_and_visualize(embeddings, meta_df, k=30, output_dir="./clean_output"):
    os.makedirs(output_dir, exist_ok=True)

    n = len(embeddings)
    if n == 0:
        print("没有有效样本，跳过聚类。")
        return meta_df, pd.DataFrame()

    k = max(1, min(k, n))
    if k < 2:
        print(f"有效样本仅 {n} 个，不足以聚类。")
        return meta_df, pd.DataFrame()

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    meta_df['cluster'] = km.fit_predict(embeddings)

    centers = km.cluster_centers_
    meta_df['dist_to_center'] = np.linalg.norm(
        embeddings - centers[meta_df['cluster'].values], axis=1)

    stats = meta_df['cluster'].value_counts().rename_axis('cluster').reset_index(name='count')
    stats['proportion'] = stats['count'] / len(meta_df)
    stats = stats.sort_values('count', ascending=False)
    stats.to_csv(os.path.join(output_dir, "cluster_stats.csv"), index=False)

    if n >= 4:
        print("正在生成 UMAP 降维交互散点图...")
        reducer = umap.UMAP(n_neighbors=min(15, n - 1), min_dist=0.1,
                             n_components=2, random_state=42)
        umap_embedding = reducer.fit_transform(embeddings)
        meta_df['umap_x'] = umap_embedding[:, 0]
        meta_df['umap_y'] = umap_embedding[:, 1]

        hover = [c for c in ['file', 'cls_name', 'dist_to_center',
                             'width', 'height', 'laplacian_var'] if c in meta_df.columns]
        fig = px.scatter(
            meta_df, x='umap_x', y='umap_y',
            color=meta_df['cluster'].astype(str),
            hover_data=hover,
            title="DINOv2 Embeddings UMAP Projection & Cluster Distribution"
        )
        fig.write_html(os.path.join(output_dir, "umap_clusters.html"))
        print(f"交互式散点图: {output_dir}/umap_clusters.html")
    else:
        print(f"样本过少({n})，跳过 UMAP 可视化。")

    meta_df['action'] = 'keep'
    export_cols = ['file', 'cluster', 'dist_to_center', 'action', 'width', 'height', 'path']
    if 'cls_name' in meta_df.columns:
        export_cols.insert(1, 'cls_name')
    meta_df[export_cols].to_csv(os.path.join(output_dir, "verdict_template.csv"), index=False)
    print(f"回写模板文件: {output_dir}/verdict_template.csv")

    return meta_df, stats


# ================= 4. 采样拼图（Top / 尾部脏数据） =================
def _tile(im, cell):
    h, w = im.shape[:2]
    s = min(cell / w, cell / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    r = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((cell, cell, 3), 245, np.uint8)
    y = (cell - nh) // 2
    x = (cell - nw) // 2
    canvas[y:y + nh, x:x + nw] = r
    return canvas


def build_mosaic(rows, out_path, per_cluster=5, cell=140, header=24):
    """rows: list of (label, [image_path,...])"""
    rows = [(lab, [p for p in paths if os.path.isfile(p)]) for lab, paths in rows]
    rows = [(lab, paths) for lab, paths in rows if paths]
    if not rows:
        return None
    cols = per_cluster
    H = header + len(rows) * cell
    W = cols * cell
    canvas = np.full((H, W, 3), 255, np.uint8)
    for ri, (label, paths) in enumerate(rows):
        y0 = header + ri * cell
        for ci, p in enumerate(paths[:cols]):
            im = cv2.imread(p)
            if im is None:
                continue
            canvas[y0:y0 + cell, ci * cell:(ci + 1) * cell] = _tile(im, cell)
        cv2.putText(canvas, label[:60], (4, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out_path


def render_cluster_mosaics(meta_df, cluster_col, cluster_ids, out_dir, prefix,
                           per_cluster=5, cell=140, name_fn=None):
    rows = []
    for cid in cluster_ids:
        sub = meta_df[meta_df[cluster_col] == cid].sort_values('dist_to_center')
        reps = sub.head(per_cluster)['path'].tolist()
        label = name_fn(cid, len(sub)) if name_fn else f"{cluster_col} {cid} (n={len(sub)})"
        rows.append((label, reps))
    return build_mosaic(rows, os.path.join(out_dir, f"{prefix}.jpg"),
                        per_cluster=per_cluster, cell=cell)


def global_mosaics(meta_df, output_dir, top_n=10, per_cluster=5):
    """全局：Top N 主流簇 与 尾部小簇/离群脏数据"""
    cluster_ids = meta_df['cluster'].value_counts().index.tolist()
    top = cluster_ids[:top_n]
    tail = cluster_ids[-top_n:]

    def lab(cid, n):
        pct = 100.0 * n / len(meta_df)
        return f"cluster {cid}  n={n} ({pct:.2f}%)"

    render_cluster_mosaics(meta_df, 'cluster', top, output_dir,
                           "global_top_clusters", per_cluster, name_fn=lab)
    render_cluster_mosaics(meta_df, 'cluster', tail, output_dir,
                           "global_tail_clusters", per_cluster, name_fn=lab)

    outliers = meta_df.sort_values('dist_to_center', ascending=False).head(top_n * per_cluster)
    rows = [("global outliers (far from center, 疑似脏数据)", outliers['path'].tolist())]
    build_mosaic(rows, os.path.join(output_dir, "global_outliers.jpg"),
                 per_cluster=top_n * per_cluster)
    print(f"全局采样拼图: {output_dir}/global_top_clusters.jpg, "
          f"global_tail_clusters.jpg, global_outliers.jpg")


# ================= 5. 按类别：Top10 子类 + 尾部脏数据 =================
def per_class_analysis(embeddings, meta_df, output_dir, top_n=10,
                       k_per_class=10, per_cluster=5):
    if 'cls_name' not in meta_df.columns or meta_df['cls_name'].isna().all():
        print("缺少类别信息(cls_name)，跳过按类别分析。")
        return
    out_dir = os.path.join(output_dir, "per_class")
    os.makedirs(out_dir, exist_ok=True)

    meta_df = meta_df.reset_index(drop=True)
    print(f"按类别分析：每类最多 {k_per_class} 个子簇，输出 Top{top_n} 与尾部脏数据...")

    for cls_name, idx in meta_df.groupby('cls_name').groups.items():
        idx = np.asarray(idx)
        sub = meta_df.loc[idx].copy()
        X = embeddings[idx]
        n = len(sub)
        safe = str(cls_name).replace("/", "_").replace(" ", "_")

        if n < 4:
            rows = [(f"{cls_name} 仅 {n} 个样本", sub['path'].tolist())]
            build_mosaic(rows, os.path.join(out_dir, f"{safe}_all.jpg"),
                         per_cluster=max(1, n))
            continue

        k = max(2, min(k_per_class, n))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        sub['sub'] = km.fit_predict(X)
        centers = km.cluster_centers_
        sub['sub_dist'] = np.linalg.norm(X - centers[sub['sub'].values], axis=1)

        counts = sub['sub'].value_counts()
        counts.to_csv(os.path.join(out_dir, f"{safe}_subclusters.csv"))
        order = counts.index.tolist()
        top = order[:top_n]
        tail = order[-top_n:] if len(order) > top_n else order

        def lab(cid, nn):
            return f"{cls_name} sub{cid}  n={nn} ({100.0 * nn / n:.1f}%)"

        render_cluster_mosaics(sub, 'sub', top, out_dir, f"{safe}_top{len(top)}",
                               per_cluster, name_fn=lab)
        render_cluster_mosaics(sub, 'sub', tail, out_dir, f"{safe}_tail",
                               per_cluster, name_fn=lab)

        dirty = sub.sort_values('sub_dist', ascending=False).head(per_cluster * 4)
        build_mosaic([(f"{cls_name} 尾部离群(疑似脏数据) n={n}", dirty['path'].tolist())],
                     os.path.join(out_dir, f"{safe}_outliers.jpg"), per_cluster=per_cluster * 4)

    print(f"按类别拼图与子簇统计已输出到: {out_dir}/")


# ================= 入口 =================
def main():
    ap = argparse.ArgumentParser(description="目标检测数据集自动清洗流水线")
    ap.add_argument("--dataset", default="dataset/batch_12.v8i.merged12.yolov11")
    ap.add_argument("--crop-dir", default="auto_check_dataset/crop")
    ap.add_argument("--output-dir", default="auto_check_dataset/clean_output")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--min-size", type=int, default=16)
    ap.add_argument("--pad", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--per-class-k", type=int, default=10)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--crop-only", action="store_true")
    ap.add_argument("--skip-crop", action="store_true")
    ap.add_argument("--no-per-class", action="store_true")
    args = ap.parse_args()

    if not args.skip_crop:
        crop_dataset(args.dataset, args.crop_dir, splits=args.splits,
                     min_size=args.min_size, pad_ratio=args.pad)
    if args.crop_only:
        return

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_df = _load_manifest(args.crop_dir)
    processor, model, device = load_dino(args.model, args.device)
    embeddings, meta_df = extract_features_enhanced(
        args.crop_dir, processor, model, device,
        min_size=args.min_size, batch_size=args.batch_size, manifest_df=manifest_df)

    if len(embeddings):
        np.save(os.path.join(args.output_dir, "embeddings.npy"), embeddings)
        meta_df.to_csv(os.path.join(args.output_dir, "meta.csv"), index=False)
        print(f"已保存 embeddings.npy / meta.csv 到 {args.output_dir}")

    meta_df, _ = cluster_and_visualize(embeddings, meta_df, k=args.k,
                                       output_dir=args.output_dir)
    if len(embeddings):
        global_mosaics(meta_df, args.output_dir, top_n=args.top_n)
        if not args.no_per_class:
            per_class_analysis(embeddings, meta_df, args.output_dir,
                               top_n=args.top_n, k_per_class=args.per_class_k)


if __name__ == "__main__":
    main()
