import argparse
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from auto_check_dataset.main import build_mosaic


# ---------- 工具 ----------
class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _chunk_rows(paths, per_row):
    return [paths[i:i + per_row] for i in range(0, len(paths), per_row)]


def _mosaic_by_reason(df, reason, out_dir, per_row=6, max_rows=30):
    sub = df[df['reasons'].str.contains(reason)]
    if sub.empty:
        return
    rows = []
    for chunk in _chunk_rows(sub['path'].tolist(), per_row)[:max_rows]:
        rows.append((f"{reason}  (n={len(sub)})", chunk))
    build_mosaic(rows, os.path.join(out_dir, f"{reason}.jpg"),
                 per_cluster=per_row, cell=130)


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser(description="无监督脏数据排查 (A/B/C/D)")
    ap.add_argument("--clean-dir", default="auto_check_dataset/clean_output")
    ap.add_argument("--out-dir", default="auto_check_dataset/clean_output/suspects")
    ap.add_argument("--knn", type=int, default=20)
    ap.add_argument("--dup-sim", type=float, default=0.97)
    ap.add_argument("--pct", type=float, default=1.0,
                    help="质量/离群分位数阈值(%)")
    ap.add_argument("--consistency-thr", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    E = np.load(os.path.join(args.clean_dir, "embeddings.npy")).astype(np.float32)
    meta = pd.read_csv(os.path.join(args.clean_dir, "meta.csv"))
    assert len(E) == len(meta), f"向量数 {len(E)} 与 meta 行数 {len(meta)} 不一致"
    N = len(E)
    print(f"载入 {N} 个向量, {E.shape[1]} 维")

    # 归一化
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    labels = meta['cls_name'].fillna("NA").to_numpy()
    cid = meta['cls_id'].fillna(-1).astype(int).values
    id2name = (meta.dropna(subset=['cls_name']).drop_duplicates('cls_id')
               .set_index('cls_id')['cls_name'].to_dict())

    # ---------- 一次全局 kNN（服务 A / B / D） ----------
    print(f"计算全局 {args.knn} 近邻...")
    nn = NearestNeighbors(n_neighbors=args.knn + 1, metric='cosine',
                          algorithm='brute', n_jobs=-1).fit(E)
    dist, idx = nn.kneighbors(E)
    dist, idx = dist[:, 1:], idx[:, 1:]           # 去掉自身
    sim = 1.0 - dist

    same = labels[idx] == labels[:, None]
    consistency = same.mean(axis=1)
    masked = np.where(same, dist, np.inf)
    nearest_same = masked.min(axis=1)             # inf = top-k 内无同类

    # 近邻多数类
    cid_to_col = {c: i for i, c in enumerate(sorted(set(cid)))}
    n_cls = len(cid_to_col)
    nbr_cid = np.vectorize(cid_to_col.get)(cid[idx], -1)
    counts = np.zeros((N, n_cls), dtype=np.int32)
    for j in range(args.knn):
        np.add.at(counts, (np.arange(N), nbr_cid[:, j]), 1)
    maj_col = counts.argmax(axis=1)
    col_to_cid = {v: k for k, v in cid_to_col.items()}
    maj_cid = np.vectorize(col_to_cid.get)(maj_col, -1)
    maj_frac = counts.max(axis=1) / args.knn
    relabel_target = np.array([id2name.get(int(c), str(c)) for c in maj_cid])

    # ---------- A: 跨类错标 ----------
    mislabel = (consistency < args.consistency_thr) & (maj_frac >= 0.5) & (maj_cid != cid)
    print(f"[A] 跨类错标: {mislabel.sum()}")

    # ---------- B: 类内离群 ----------
    # 类质心距离
    centroid_dist = np.zeros(N, dtype=np.float32)
    for c in np.unique(cid):
        m = cid == c
        cen = E[m].mean(axis=0)
        cen /= (np.linalg.norm(cen) + 1e-8)
        centroid_dist[m] = 1.0 - E[m] @ cen
    # 类内分位阈值
    out_centroid = np.zeros(N, dtype=bool)
    out_density = np.zeros(N, dtype=bool)
    for c in np.unique(cid):
        m = cid == c
        thr_c = np.percentile(centroid_dist[m], 100 - args.pct)
        out_centroid[m] = centroid_dist[m] > thr_c
        valid = np.isfinite(nearest_same[m])
        if valid.any():
            thr_d = np.percentile(nearest_same[m][valid], 100 - args.pct)
            out_density[m] = (~np.isfinite(nearest_same[m])) | (nearest_same[m] > thr_d)
        else:
            out_density[m] = True
    outlier = out_centroid | out_density
    print(f"[B] 类内离群: {outlier.sum()} (质心 {out_centroid.sum()}, 密度 {out_density.sum()})")

    # ---------- C: 图像质量 ----------
    min_side = np.minimum(meta['width'].values, meta['height'].values)
    aspect = meta['width'].values / np.maximum(meta['height'].values, 1)
    lap = meta['laplacian_var'].values
    thr_side = np.percentile(min_side, args.pct)
    thr_lap = np.percentile(lap, args.pct)
    thr_asp_hi = np.percentile(aspect, 100 - args.pct)
    thr_asp_lo = np.percentile(aspect, args.pct)
    q_small = min_side < thr_side
    q_blur = lap < thr_lap
    q_aspect = (aspect > thr_asp_hi) | (aspect < thr_asp_lo)
    quality = q_small | q_blur | q_aspect
    print(f"[C] 质量问题: {quality.sum()} (小 {q_small.sum()}, 糊 {q_blur.sum()}, 长宽比 {q_aspect.sum()})")

    # ---------- D: 重复/近重复 ----------
    uf = UnionFind(N)
    dup_neighbor = sim[:, 0] > args.dup_sim
    for i in np.nonzero(dup_neighbor)[0]:
        uf.union(i, int(idx[i, 0]))
    roots = np.array([uf.find(i) for i in range(N)])
    _, inv, cnt = np.unique(roots, return_inverse=True, return_counts=True)
    dup_group = np.where(cnt[inv] > 1, inv, -1)
    # 每组保留第一个出现的
    dup = dup_group >= 0
    rep = np.zeros(N, dtype=bool)
    seen = set()
    for i in range(N):
        g = dup_group[i]
        if g >= 0 and g not in seen:
            rep[i] = True
            seen.add(g)
    dup_delete = dup & ~rep
    print(f"[D] 重复组内样本: {dup.sum()}, 其中可删 {dup_delete.sum()}")

    # ---------- 汇总 ----------
    reasons = [[] for _ in range(N)]
    score = np.zeros(N, dtype=np.float32)
    for i in range(N):
        if mislabel[i]:
            reasons[i].append("mislabel")
            score[i] += 2.0
        if out_centroid[i]:
            reasons[i].append("outlier_centroid")
            score[i] += 1.0
        if out_density[i]:
            reasons[i].append("outlier_density")
            score[i] += 1.5
        if q_small[i]:
            reasons[i].append("tiny")
            score[i] += 1.0
        if q_blur[i]:
            reasons[i].append("blur")
            score[i] += 1.0
        if q_aspect[i]:
            reasons[i].append("aspect")
            score[i] += 1.0
        if dup_delete[i]:
            reasons[i].append("duplicate")
            score[i] += 1.0

    reason_str = [";".join(r) for r in reasons]
    flagged = np.array([len(r) > 0 for r in reasons])

    actions = []
    for i in range(N):
        r = reasons[i]
        if not r:
            actions.append("keep")
        elif "mislabel" in r:
            actions.append("relabel")
        elif "duplicate" in r:
            actions.append("delete")
        elif ("outlier_centroid" in r or "outlier_density" in r) and \
                ("blur" in r or "tiny" in r or "aspect" in r):
            actions.append("delete")
        else:
            actions.append("review")

    out = meta.copy()
    out['nn_consistency'] = consistency.round(4)
    out['nearest_class'] = relabel_target
    out['nearest_class_frac'] = maj_frac.round(4)
    out['centroid_dist'] = centroid_dist.round(4)
    out['nearest_same_dist'] = np.where(np.isfinite(nearest_same), nearest_same, -1).round(4)
    out['dup_group'] = dup_group
    out['reasons'] = reason_str
    out['severity'] = score
    out['suggested_action'] = actions
    out['relabel_target'] = np.where(mislabel, relabel_target, "")
    conf = np.full(N, "", dtype=object)
    high = mislabel & (consistency < 0.2) & (maj_frac >= 0.6)
    medium = mislabel & ~high & (consistency < 0.35)
    low = mislabel & ~high & ~medium
    conf[high] = "high"
    conf[medium] = "medium"
    conf[low] = "low"
    out['confidence'] = conf

    out = out.sort_values('severity', ascending=False)
    sus_path = os.path.join(args.out_dir, "suspects.csv")
    out[out['suggested_action'] != 'keep'].to_csv(sus_path, index=False)
    out.to_csv(os.path.join(args.out_dir, "all_scored.csv"), index=False)

    # 错标对汇总（混淆方向）
    ml = out[out['suggested_action'] == 'relabel']
    if not ml.empty:
        pairs = (ml.groupby(['cls_name', 'relabel_target', 'confidence'])
                 .size().reset_index(name='count')
                 .sort_values('count', ascending=False))
        pairs.to_csv(os.path.join(args.out_dir, "relabel_pairs.csv"), index=False)

    # ---------- 复查拼图 ----------
    print("生成复查拼图...")
    sub = out[out['suggested_action'] != 'keep'].reset_index(drop=True)
    for reason in ["mislabel", "outlier_centroid", "outlier_density",
                   "blur", "tiny", "aspect", "duplicate"]:
        _mosaic_by_reason(sub, reason, args.out_dir)
    # 总览：最严重的 60 张
    top = sub.head(60)['path'].tolist()
    if top:
        build_mosaic([("top suspects (severity desc)", top)],
                     os.path.join(args.out_dir, "top_suspects.jpg"), per_cluster=6, cell=130)

    # ---------- 统计 ----------
    print("\n=== 汇总 ===")
    print(f"标记样本: {flagged.sum()} / {N} ({100*flagged.mean():.2f}%)")
    print(pd.Series(actions).value_counts().to_string())
    if (out['suggested_action'] == 'relabel').any():
        print("\n错标置信度:")
        print(out.loc[out['suggested_action'] == 'relabel', 'confidence']
              .value_counts().to_string())
    print(f"\nsuspects.csv: {sus_path}")
    print(f"拼图目录: {args.out_dir}/")


if __name__ == "__main__":
    main()
