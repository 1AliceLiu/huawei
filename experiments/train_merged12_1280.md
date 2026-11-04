# yolo26m merged12 @ imgsz=1280 (batch=64)

## 目的
验证**提高输入分辨率**（imgsz 960 → 1280）对稀有类 / 小目标的影响，重点关注
`Container`（AP50 仅 0.34，判据是箱体条纹 + 后轮 6 个，对小尺度/压缩姿态极敏感）
与 `Motorcycle`（AP50-95 0.35，小目标）。

## 数据集来源
- 路径：`dataset/clean_part_batch_12.v8i.merged12.yolov11`
- 12 类：`Coach, Franchised Bus, HGV, LGV, Light Bus, MGV, Motorcycle, PLB GMB, Private Car, Taxi, Van, Container`
- 划分：train / valid / test，**按 group（camera/segment）无泄漏**
- 标签状态：**2026-10-16 已人工清理**
  - 通过 `scripts/audit_labels.py` 检出并修正重复框(29)、同框多标签(20)、类冲突(47)、近重复(128)
  - 人工修正的 197 张图由 `scripts/apply_error_label_jsons.py` 写回（原始 label 备份在 `temp/error-label-backup/`）
  - 修正后重扫异常：0

## 相对基线的改动
| 项 | baseline `runs/detect/yolo26s_merged12_20261016_0118` | 本次 |
|---|---|---|
| 模型 | yolo26s | **yolo26m** |
| imgsz | 960 | **1280** |
| batch（总） | 128 | **32**（DDP 16/卡；原计划 64 因 OOM 减半） |
| device | 3,2 | 1,2 |
| epochs / patience | 120 / 15 | 120 / 15 |
| optimizer / lr0 | MuSGD / 5e-4 | 同 |
| seed | 0 | 0 |

说明：
- 分辨率提升到 1280 后单样本显存约 ×1.8，`yolo26m batch=64 @1280`（32/卡）在 2×L20(46G) 上 **OOM**
  （GPU2 已占 44.25GiB 仍差 400MB）。DDP 模式 ultralytics 不会自动减 batch，故手动改为 **batch=32**。
- 即：本次相对基线的核心变量是 **分辨率 960→1280 + 模型 s→m**，batch 由 128 降到 32
  （有效 batch 变小，训练步数变多）。

## 运行
`experiments/train_merged12_1280.py`（tmux 会话，GPU 1、2）

```bash
tmux new-session -d -s train1280 \
  "cd /workspaces/huawei && source .venv/bin/activate && \
   python -u experiments/train_merged12_1280.py 2>&1 | tee logs/train_merged12_1280.log"
```

- 结果目录：`runs/detect/yolo26m_merged12_1280_<YYYYMMDD_HHMM>`
- 日志：`logs/train_merged12_1280.log`

## 评测方式
训练后用**同分辨率**（imgsz=1280）在 `valid` 上评测，与 baseline 逐类对比 AP：
```bash
yolo val model=runs/detect/yolo26m_merged12_1280_<TS>/weights/best.pt \
     data=dataset/clean_part_batch_12.v8i.merged12.yolov11/data.yaml imgsz=1280
```

## 备注
- DDP（多卡）下 ultralytics **不会**在 OOM 时自动减半 batch，会直接报错；
  若首轮 OOM，可将 `batch` 降到 32 或减少卡数。
- `cls_pw` 本次仍未开启（0.0）；若分辨率收益不足，下一步再叠加 `cls_pw≈0.3~0.5`。
