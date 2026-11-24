rtsp://118.140.234.166:8554/dahua1001619

## 目标类
Motorcycle
Private Car
Taxi
Light Van
LGV
MGV
HGV
Container
Public Light Bus/ GMB
Light Bus
Franchised Bus
Coach

## COCO 80 类

car
motorcycle
bus
truck

| COCO大类 | 细分子类 |
| :---: | :--- |
| **motorcycle** | Motorcycle |
| **car** | Private Car（私家车） |
| | Taxi（出租车） |
| | Light Van（轻型客货车） |
| **bus** | Public Light Bus / GMB（公共小巴） |
| | Light Bus（轻型巴士） |
| | Franchised Bus（专营巴士） |
| | Coach（长途/旅游巴士） |
| **truck** | LGV（轻型货车） |
| | MGV（中型货车） |
| | HGV（重型货车） |
| | Container（集装箱车） |


tar -zxvf data.tar.gz

## 类别合并:batch_12 v9 (11类) -> v10 merge8 (8类)

- 重建数据集:`dataset/batch_12.v10i.merge8.yolov11`(图/标签 7929 对,与 v9 逐文件对应,仅重映射 class id,未改动任何标注框)
- 重建脚本:`scripts/remap_batch12_merge8.py`(`python scripts/remap_batch12_merge8.py`)
- 依据:`runs/detect/yolo26m_traffic_van_focus_20260902_0201` 在 v9 上的混淆矩阵,细分类边界是载重/涂装等监控视角下不可判读的视觉特征,是类间误报的主要来源

### 映射表

| 新类别 (id) | 原类别 (id) | 合并理由 |
| :--- | :--- | :--- |
| Bus (0) | Coach (0) + Light Bus (4) | 10% 的 Coach 被误判为 Light Bus,均为巴士车身 |
| Franchised Bus (1) | Franchised Bus (1) | 双层车视觉独特,AP 0.927,保留 |
| Truck (2) | HGV (2) + MGV (5) + LGV (3) | 37% LGV、17% HGV 被误判为 MGV;载重吨位边界视觉不可分;救起最差类 LGV(AP 0.638 / R 0.533) |
| Motorcycle (3) | Motorcycle (6) | AP 0.93,无需变动 |
| PLB GMB (4) | PLB GMB (7) | 与 Light Bus 混淆仅 2–3%,暂保留,列为观察对象 |
| Private Car (5) | Private Car (8) | 主类,AP 0.919,保留 |
| Taxi (6) | Taxi (9) | 涂装独特,AP 0.919,保留 |
| Van (7) | Van (10) | 业务目标类,单独保留 |

### 合并后分布与遗留风险

| id | 类别 | train | valid | test |
| :--- | :--- | ---: | ---: | ---: |
| 0 | Bus | 4534 | 199 | 148 |
| 1 | Franchised Bus | 3046 | 104 | 133 |
| 2 | Truck | 4126 | 185 | 165 |
| 3 | Motorcycle | 1325 | 46 | 77 |
| 4 | PLB GMB | 1982 | 72 | 120 |
| 5 | Private Car | 18612 | 769 | 952 |
| 6 | Taxi | 6136 | 241 | 294 |
| 7 | Van | 2458 | 116 | 109 |

遗留风险(重训后重看混淆矩阵):
1. Van 是本方案未处理的唯一大混淆源:27% 的 Van 被误判为 Private Car,推理时可对 Van 单独提高 conf 阈值或补充难负样本
2. 新边界 Van <-> Truck(原 LGV 与 Van 语义同源)与 PLB GMB <-> Bus 需重点关注

预期:类间误报内化约四成,mAP50 自 0.856 提升;background 漏检(各类 4–12%)不受合并影响,属另一议题

camera_id	code	name	rtsp
1003	1000227	Mobile Camera1003	rtsp://118.140.234.166:8554/dahua1003
866	1000226	Mobile Camera0866	rtsp://118.140.234.166:8554/dahua866
548	1000172	Mobile Camera0548	rtsp://118.140.234.166:8554/dahua548