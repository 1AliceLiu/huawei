# Dataset Visual Survey Pipeline

调查 YOLO 数据集每个类别内部视觉亚型与分布，评估数据集质量。
探索期结论（V0→V3）已固化，不再调参。

##  frozen 参数
- crop：bbox 外扩 10% → 黑边正方形 → 224
- filter：normalized area ≥ 0.01（类太小则 fallback 0.005 / 0.0）
- embedding：DINOv2 ViT-S/14（torch.hub，384D），L2 后聚类
- L1：每类内 KMeans（kmeans++，50 轮，seed=0），K 自适应：n≥1500→5，≥600→4，≥200→3，≥60→2，否则 1
- 代表图：离中心最近 TOP9 medoid 邻域
- 质量旗：n≥20 且（videos≤3 或 top1≥60%）→ artifact:single-source；top1≥40% → artifact:source-dominated；否则 subtype-candidate

## 运行
```bash
uv run python experiments/10.6/pipeline/survey.py        # 全类别 L1 → atlas/<Class>/
uv run python experiments/10.6/pipeline/report.py        # → atlas/REPORT.md
uv run python experiments/10.6/pipeline/atlas_overview.py # → atlas/OVERVIEW.jpg（一页总览）
uv run python experiments/10.6/pipeline/spotcheck_moto.py # medoid 回原图画框核查（模板，按需改类）
```

## 产出
- `atlas/<Class>/{contact_sheet.jpg, medoids/, stats.json, embeddings.pt, manifest.json}`
- `embeddings.pt` 按 manifest 行顺序，可直接做 L2 视角内二次聚类，无需重跑模型
- `atlas/PrivateCar/side/`：已验证的 L2 范例（side 内 tall-van vs low-sedan，744 框 / 249 视频）

## 已知结论（2026-10-06，train split）
- Motorcycle C0（n=71）= 行人误标，原图验框确认
- Container 全类仅 12 个视频；PrivateCar C3（206/232 单视频）、PLB GMB C0（单机位 42%）为污染簇
- PrivateCar/Taxi 仅 18% 框、Motorcycle 仅 10% 框达到 area≥0.01（小目标现实）
