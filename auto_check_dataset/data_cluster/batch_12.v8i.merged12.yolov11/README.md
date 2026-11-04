# clean_output 结果说明

本目录是目标检测数据集清洗流水线（`auto_check_dataset/main.py`）的输出。

- 输入：`dataset/batch_12.v8i.merged12.yolov11`
- 有效 ROI：**54,924** 个（tiny_noise 过滤 221 个），12 个类别
- 编码模型：`facebook/dinov2-small`，384 维，L2 归一化
- 全局 K-Means `K=30`，每类子簇 `K=10`

## 我想看「所有类别的 Top10 子类」

看 `per_class/<类别>_top10.jpg`（共 12 张，每类一张）。

- 每行 = 该类的一个**子簇**，行首标注 `类别 sub{id}  n=数量 (占比%)`
- 每行 5 张 = 离该子簇质心最近的**代表图**，最能代表这个子类
- 完整子簇大小表：`per_class/<类别>_subclusters.csv`

> 若某一行 5 张图是"异类"（例如 Private Car 下混入货车/巴士），说明该类存在不合规子类，可按行批量处理。

## 我想看「占比较小的脏数据」

- `per_class/<类别>_tail.jpg` —— 该类**占比最小的 10 个子簇**（稀有子类）
- `per_class/<类别>_outliers.jpg` —— 离质心最远的离群图（**最可疑的错标/脏数据**）
- 全局版本：`global_tail_clusters.jpg`、`global_outliers.jpg`

## 文件清单

| 文件 | 说明 |
|---|---|
| `per_class/<类别>_top10.jpg` | 每类 Top10 子簇代表图拼图 |
| `per_class/<类别>_tail.jpg` | 每类尾部（占比最小）子簇拼图 |
| `per_class/<类别>_outliers.jpg` | 每类离群/疑似脏数据拼图 |
| `per_class/<类别>_subclusters.csv` | 每类子簇大小表 |
| `global_top_clusters.jpg` | 全局最大 10 个簇的代表图拼图 |
| `global_tail_clusters.jpg` | 全局最小 10 个簇的拼图 |
| `global_outliers.jpg` | 全局离质心最远的样本拼图 |
| `umap_clusters.html` | 交互散点图，悬停显示 `file / cls_name / dist_to_center`，按簇着色 |
| `cluster_stats.csv` | 全局 30 个簇的数量/占比（降序，末尾即长尾脏数据） |
| `meta.csv` | 每个 crop 的全量元数据（`source_image`、`x1..y2`、`cls_name`、`cluster`、`dist_to_center` 等） |
| `embeddings.npy` | 54,924 × 384 的 DINOv2 向量，便于复算 |
| `verdict_template.csv` | 回写模板 |

## 下一步：闭环回写

编辑 `verdict_template.csv` 的 `action` 列（`keep` / `delete` / `relabel`）。该文件已包含 `source_image` 与 `x1..y2`，可直接定位回原始 `batch_12.v8i.merged12.yolov11` 的标注框，做批量修改/剔除。

## 脏数据排查（`suspects/`）

由 `auto_check_dataset/find_dirty.py` 生成，无监督多信号（A 跨类错标 / B 类内离群 / C 质量 / D 重复），阈值按分位数自动确定。

运行：
```bash
.venv/bin/python -m auto_check_dataset.find_dirty \
  --clean-dir auto_check_dataset/clean_output \
  --out-dir auto_check_dataset/clean_output/suspects
```

本次结果：标记 **10,532 / 54,924 (19.2%)** — `relabel` 6840、`review` 2441、`delete` 1251。

| 文件 | 说明 |
|---|---|
| `suspects.csv` | 主结果，按 `severity` 降序；含 `reasons`、`suggested_action`、`relabel_target`、`confidence` |
| `all_scored.csv` | 全部样本的打分（含未标记的 keep） |
| `relabel_pairs.csv` | 错标方向汇总（`原类别 -> 目标类别` + 置信度 + 数量） |
| `mislabel.jpg` | 跨类错标复查拼图 |
| `outlier_centroid.jpg` / `outlier_density.jpg` | 类内离群（远离质心 / 低密度）复查拼图 |
| `blur.jpg` / `tiny.jpg` / `aspect.jpg` | 模糊 / 过小 / 极端长宽比复查拼图 |
| `duplicate.jpg` | 近重复（余弦 > 0.97）复查拼图 |
| `top_suspects.jpg` | 最严重 60 张总览 |

`suspects.csv` 关键列：`reasons`（命中的信号）、`severity`（加权严重度）、`confidence`（错标置信度 high/medium/low）、`suggested_action`（`relabel`/`delete`/`review`）、`relabel_target`（建议目标类）。

错标高发方向：Van→Private Car、Taxi→Private Car、LGV→MGV、Private Car→Taxi/Van（相邻车型，需人工确认）。

阈值可调：`--pct`（质量/离群分位数，默认 1%）、`--consistency-thr`（近邻一致性，默认 0.5）、`--dup-sim`（重复阈值，默认 0.97）、`--knn`（近邻数，默认 20）。
