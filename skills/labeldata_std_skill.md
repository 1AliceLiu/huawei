
>Name: 标签数据的规范化标注
Description: 应用于ai和人工标注的label数据标准

## 任务
在截取的视频帧中标注人眼可分辨清的车种类，类别共有12种：Motorcycle
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

## 标注原则
- 首先明确:**目前急需Container/Van/LGV/Motorcycle四种数据,尽量收集包含这四类车的视频帧（三种数据不一定要在同一张图中）**
- 一张图中应包含多种类型的车，如：若图中多数为private car（这里特指数据过量的类别，数据稀少类型除外），则舍弃该帧
- 收集时,确保自己能够较为清晰的判断出图中车的类别,若不可明确,则舍弃该框
- 标注时需要将所有能人眼识别到的车进行类型标注
- 由于数据截取方式不同,可能有的连续图片是同一个场景几乎同一时间,此时图片中的车没变化,所以只需要标记一个图片即可,其他重复图片舍弃即可
- 标注软件使用X-AnyLabeling-v4.1.0-Windows-CUDA13.exe，使用该软件打开图片存放的文件夹即可操作。

## 类别说明

|类别|说明|示例图|
| :---: | :---: | :---: |
|Light Bus|私家巴⼠，如校⻋、养⽼院接送⻋等。主体颜⾊⼀般为⽩⾊或浅⻩⾊，⻋⾝有腰线。|![alt text](images/image-39.png)|
|Coach|长途巴士，主要为旅游巴⼠|![alt text](images/image-43.png)|
|Franchised Bus|有固定路线的公共巴⼠，涂装明显，⻋⾝有公司  logo （如九巴、城巴、港铁巴⼠等）。港铁巴⼠等）。 |![alt text](images/image-35.png)|
|PLB GMB|⾹港公共⼩巴，通常由  16 座⾯包⻋改装，涂装明显，⼀般为红⾊或绿⾊。|![alt text](images/image-28.png)|
|Van|典型特征：⻋⾝较⻓的⾯包⻋，后排⼀般不载客。⻋⾝有时印有  GoGoVan 、 Lalamove 等公司  logo 。|见上表|
|LGV|（ 4 轮）典型厢式货⻋、⽪卡多为两轴；部分轻型货⻋后轮可能为单轮或双轮。|![alt text](images/image-32.png) ![alt text](images/image-34.png)|
|MGV|部分3轴中型货⻋轴数甚⾄可能多于某些轻量化HGV|![alt text](images/image-36.png)|
|HGV|许多HGV为两轴（如7.5吨⾄18吨  rigid 货⻋），外观和轮数可能与重型MGV ⽆异；更⼤吨位  HGV 才普遍采⽤  3 轴、 4 轴甚⾄更多轴。**可以通过观察车轮数、车高和车头形状进行快速判断**|![alt text](images/image-38.png)|
|Container|较为稀有，载有集装箱的重型货车|![alt text](images/image-33.png)|
|Taxi|常见颜色为绿色和红色，香港的士外观独特很好分辨，**但需要注意一种suv的taxi，车身标有taxi logo** |![alt text](images/image-30.png) ![alt text](images/image-37.png) ![alt text](images/image-40.png) ![alt text](images/image-41.png)![alt text](images/image-42.png)|
|Motorcycle|摩托车，也较为稀少，**需重点关注**|![alt text](images/image-31.png)|
|Private Car|重点分辨保姆车与Van|![alt text](images/image-29.png)|

## 重点注意
### 1、LGV与MGV的区别:
根据香港运输署发布的LGV与MGV车辆驾驶证考试准则，LGV总重量不少于2.7吨，MGV总重量不少于9吨。**而最重要的区别在于香港较为常见的大卡车品牌后视镜的安装位置,如下图所示**

| 品牌 | LGV | MGV |
| :---: | :---: | :---: |
| 十五铃isuzu | ![alt text](images/image.png) | ![alt text](images/image-1.png) |
| HINO | ![alt text](images/image-2.png) | ![alt text](images/image-3.png) |
|UD Truck|-|![alt text](images/image-4.png)|
|三菱Fuso|![alt text](images/image-5.png)|![alt text](images/image-6.png)|

**综上，LGV的后视镜从车挡玻璃下方安装，MGV的后视镜从车挡玻璃上方安装。**

### 2、Van与Private Car（保姆车）的区别：
**A柱、车身流线型、车头与车尾 如下图所示**
|位置|Van|Private Car（保姆车）|
| :---: | :---: | :---: |
|A柱夹角| ![alt text](images/image-8.png) | ![alt text](images/image-17.png) ![alt text](images/image-27.png) |
|车身流线型| ![alt text](images/image-11.png) ![alt text](images/image-14.png)  ![alt text](images/image-19.png)| ![alt text](images/image-15.png) ![alt text](images/image-18.png) ![alt text](images/image-21.png) ![alt text](images/image-22.png)|
|车头|![alt text](images/image-9.png) ![alt text](images/image-12.png) ![alt text](images/image-13.png)![alt text](images/image-24.png) | ![alt text](images/image-16.png) ![alt text](images/image-26.png)|
|车尾| ![alt text](images/image-7.png) ![alt text](images/image-23.png)|  ![alt text](images/image-20.png) ![alt text](images/image-25.png)|

**除此之外，Van的后面车窗一般都非透明。**

### 3、**Container标注时需要显著看到集装箱才可标为Container**

## 验收步骤（Acceptance）

> 标注交付后，按以下 5 步验收。Step 1、2 为硬性门槛，必须全部通过才能进入人工抽查；Step 3 为质量判定；Step 4、5 为辅助与留档。

### Step 1｜交付物完整性核对
- 图片数量与 json 数量一致，且**同名一一对应**，无多余、无缺失
- 目录结构符合约定：图片与标注json存放于 `images/` 内（标注软件将自动保存修改后的json，无需人工操作）
- 已按**标注原则**要求剔除重复场景帧（同一场景几乎同一时间、画面无变化的帧只保留一张）
- 已剔除无法清晰判断车辆类别的框或帧

### Step 2｜格式自动校验（必须零错误）
逐个 json 检查以下规则，**任一不通过即整批退回**：
- 每个 shape 必须是 `rectangle`，禁止 polygon / line / circle
- 每个 rectangle 的 `label` 必须命中合法类别表：以**任务**的类别名称为准，共 12 类（Coach / Franchised Bus / HGV / LGV / Light Bus / MGV / Motorcycle / PLB GMB / Private Car / Taxi / Van / Container），大小写、拼写完全一致，禁止表外类别与拼写变体
- `points` 四角坐标均在图像宽、高范围内，不允许越界
- 无退化框（宽或高 ≈ 0）
- 无过小框

### Step 3｜内容人工抽查
- **抽样比例：随机抽取 10%~20%，且不少于 20 张图**
- **优先类别 Container / Van / LGV 抽样比例提高至 50%（当前急需数据）**
- 逐张核对以下高风险点：
  - **LGV vs MGV**：后视镜安装位置——LGV 从挡风玻璃**下方**安装，MGV 从挡风玻璃**上方**安装（对照品牌表）
  - **Van vs Private Car（保姆车）**：A柱夹角、车身流线型、车头、车尾；Van 后车窗一般非透明
  - **Container**：必须**显著看到集装箱**才可标为 Container
  - **Taxi**：注意 SUV 型 taxi，车身标有 taxi logo，勿标成 Private Car
  - **LGV/MGV/HGV**：结合车轮数、车高、车头形状判断，勿凭吨位想当然
  - **漏标**：图中所有能识别的车是否都已标注
  - **框质量**：框贴合车身、无明显背景残留；严重遮挡/模糊的目标所在帧是否已按要求舍弃
  - 一辆车不能同时被标为多个冲突类别，**若车身被遮挡但露出显著标志可以判断车型与位置则可将其框出；若被遮挡时猜不出其是什么车则放弃标注**


### Step 4｜结果判定与返工
- **通过标准：Step 2 零错误，且 Step 3 抽查框准确率 ≥ 90%**（如抽查 100 个框，错误 ≤ 10 个）；同时-归一化后坐标落在 [0, 1]，为后续 json → yolotxt 转换把关（这一步将由aliceliu编程检查）
- **不通过**：输出问题清单，格式为 `文件名 | 问题类型 | 类别 | 说明`，退回标注方返工；返工后仅对问题部分复验，再重新执行 Step 3 抽样



