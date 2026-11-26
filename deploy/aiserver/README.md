# aiserver 车辆事件服务

服务器使用本机 SSH 配置中的 `ESGAIserver`，地址为 `192.168.1.25`。部署目录为 `/opt/huawei-traffic`，Docker Compose 项目名为 `huawei-traffic`。推理使用物理 GPU 2，容器内编号为 0。

## 调用 API

```text
GET https://supervisor-cx5200-v5.tailb2db87.ts.net:10000/api/traffic/cameras
GET https://supervisor-cx5200-v5.tailb2db87.ts.net:10000/api/traffic/events?after=0&limit=100
GET https://supervisor-cx5200-v5.tailb2db87.ts.net:10000/api/traffic/counts
Authorization: Bearer <访问凭证>
```

公网入口由本机 Tailscale Funnel 提供，使用 Let's Encrypt 公共证书，仅暴露 TCP 10000，无需路由器端口映射。内网入口仍为 `https://192.168.1.25:18090`。

```sh
tailscale funnel status
tailscale funnel --https=10000 off   # 关闭公网入口
tailscale funnel --bg --https=10000 https+insecure://192.168.1.25:18090   # 重新开启
```

`output/aiserver/customer_delivery/access.json` 保存公网地址和凭证，通过私密渠道交付；本机 `output/aiserver/access.json` 继续使用内网地址。公网访问不需要 `aiserver.crt`；该证书仅用于内网自签名入口，签发时设置了 365 天有效期，续期后需向调用方更新信任证书。

在项目根目录运行以下 Python 示例即可验证证书并读取事件。

```python
import json
from pathlib import Path
import ssl
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener

folder = Path("output/aiserver")
access = json.loads((folder / "access.json").read_text(encoding="utf-8"))
certificate = access.get("certificate_file")
context = ssl.create_default_context(
    cafile=str(folder / certificate) if certificate else None)
client = build_opener(ProxyHandler({}), HTTPSHandler(context=context))
request = Request(
    access["base_url"] + "/api/traffic/events?after=0&limit=100",
    headers={"Authorization": "Bearer " + access["bearer_token"]},
)
with client.open(request, timeout=15) as response:
    print(json.dumps(json.load(response), ensure_ascii=False, indent=2))
```

事件字段与游标规则见 [车辆事件 API](../../docs/traffic_api.md)。服务器使用独立事件库，从本次部署开始累计。消费端应为这个服务器地址单独保存游标。

累计数量接口按摄像头、最终类别统计全部已入库过线事件，包含历史记录，重启不清零。完整字段及覆盖更新规则见上述 API 文档。

## 持续推送累计数量

在服务器 `/opt/huawei-traffic/.env` 中设置 `COUNTS_WEBHOOK_URL=<对方接收地址>`，有接收方鉴权时再设置 `COUNTS_WEBHOOK_TOKEN=<接收方 token>`。该文件仅允许管理员读取，不提交到代码库。修改后重建 worker 和 gateway 容器使环境变量生效；镜像无变化时不必重新构建。

发送端每秒检查一次，在数量变化后 POST 完整快照，失败自动重试，确认状态保存在事件库内。生产接收端须正确处理 Idempotency-Key 和 revision，并在可靠保存后返回 2xx。

本机 `counts-receiver` 验证服务已移除，`.env` 中的 `COUNTS_WEBHOOK_URL` 也已清空，worker 不再尝试推送。如需临时本地验证，可单独运行 `scripts/receive_traffic_counts.py --host 127.0.0.1 --folder <目录>`，不要再作为 Compose 服务常驻。原有 HTTPS 访问 token 与接收方 token 用途不同，不应混用。

公网查询入口为 `https://supervisor-cx5200-v5.tailb2db87.ts.net:10000`，由 Tailscale Funnel 转发到本机 `192.168.1.25:18090`。Funnel 配置保存在 tailscaled 状态中，机器重启后仍生效；公网证书由 Tailscale 自动续期。

## 文件与运行方式

| 服务器路径 | 用途 |
| --- | --- |
| `/opt/huawei-traffic/source/` | 构建镜像所用的应用代码、测试及 Dockerfile |
| `/opt/huawei-traffic/compose.yaml` | 推理容器及 HTTPS 网关配置 |
| `/opt/huawei-traffic/data/best.pt` | 当前部署的模型权重 |
| `/opt/huawei-traffic/data/huawei-rtsp.txt` | 所有摄像头的流地址 |
| `/opt/huawei-traffic/data/camera_regions.json` | 每路 ROI、终点线与方向 |
| `/opt/huawei-traffic/data/traffic_events.sqlite3` | 持久化的车辆事件数据库 |
| `/opt/huawei-traffic/proxy/` | Nginx 配置、访问凭证映射、证书与私钥 |
| `/opt/huawei-traffic/access.json` | 仅服务器管理员可读的调用凭证 |

代码镜像固定 Ultralytics 8.4.130、PyAV 19.0.1 和 lap 0.5.13，复用服务器已有 CUDA 镜像中的 PyTorch 2.15.0+cu132。模型权重沿用 `yolo26m_merged12_best_20261021_0957/weights/best.pt`。推理容器以普通用户运行，GPU 只分配第 2 号。两项服务均配置异常退出自动重启，Docker 已设置开机启动。

数据目录独立挂载，容器重建后保留标定和已入库事件。停止时发送 SIGINT，让现有退出流程保存已确认的缓冲事件。每个容器日志最多保留 3 个 10 MB 文件。

## 状态、日志与重启

```powershell
ssh ESGAIserver 'cd /opt/huawei-traffic && docker compose ps'
ssh ESGAIserver 'cd /opt/huawei-traffic && docker compose logs --tail 100 worker gateway'
ssh ESGAIserver 'cd /opt/huawei-traffic && docker compose restart worker'
```

健康检查分别验证内部 API 响应和 HTTPS 网关的未授权请求拒绝行为。摄像头是否持续出帧应查看 `/api/traffic/cameras` 中的状态和帧龄。

更新代码时，先将相应源文件复制到服务器 `source/` 下，再运行容器内测试、构建及启动命令。发布前应为当前镜像保留一个独立标签，以便回退。

```sh
cd /opt/huawei-traffic
docker image tag huawei-traffic:20261124 huawei-traffic:previous
docker compose build worker
docker compose run --rm --no-deps --entrypoint python worker -m unittest discover -s tests -q
docker compose up -d --force-recreate --wait --wait-timeout 180 worker gateway
```

如需回退代码，可把 `huawei-traffic:previous` 重新标记为 Compose 使用的 `huawei-traffic:20261124`，再使用 `docker compose up -d --no-build --force-recreate --wait worker gateway`。保留 `data/` 目录即可保留已入库事件。

## 预览与标定

在本机终端建立 SSH 隧道。

```powershell
ssh -N -L 18093:127.0.0.1:18093 -L 18094:127.0.0.1:18094 -L 18095:127.0.0.1:18095 ESGAIserver
```

然后打开 `http://127.0.0.1:18093/`，即可查看服务器推理画面并使用原来的区域标定页面。标定保存到服务器数据目录，下一帧自动加载。

源流中断和跟踪 ID 切换仍可能引起漏计或重复计数。尚未完成标定的摄像头会显示 `needs_calibration`，需要通过实际画面补齐 ROI、终点线和方向。
