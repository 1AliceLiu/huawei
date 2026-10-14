# suspect_select 数据汇总

按脏数据标签分类导出的图片与标签（同一张图有多个问题框时只保留一份）。
每个标签目录结构：`<标签>/images`（.jpg + 同名 .json）、`<标签>/labels`（.txt）。

| 标签 | 问题框 | 唯一图 | jpg | json |
|---|---:|---:|---:|---:|
| blur | 550 | 511 | 511 | 511 |
| tiny | 417 | 351 | 351 | 351 |
| top_suspects | 60 | 59 | 59 | 59 |
| duplicate | 1262 | 1086 | 1086 | 1086 |
| mislabel | 6840 | 4671 | 4671 | 4671 |
| outlier_centroid | 556 | 510 | 510 | 510 |
| outlier_density | 1635 | 1428 | 1428 | 1428 |

## 说明

- **问题框**：`suspects.csv` 中命中该标签的标注框数。
- **唯一图**：去重后的源图数量（同图多框只算一份）。
- **jpg / json**：`images/` 中图片与转换后标注文件数量，二者相等。
- `top_suspects` 非 `suspects.csv` 的标签，而是按 `severity` 取前 60 的框。
- json 为 labelme 格式（`shapes[].label` + 四角 `points`），由 `labels/*.txt` 转换，未写 `attributes`。
