"""实时视频流预测 + 浏览器 MJPEG 预览（多路）。

用法示例：
    python realtime_web.py \
        --model runs/detect/yolo26m_merged12_20261013_0935/weights/best.pt \
        --sources-file huawei-rtsp.txt \
        --imgsz 960 --half --device 0 --port 8000

huawei-rtsp.txt 每行一个 RTSP 地址（# 开头为注释）。单路也可用 --source。

然后浏览器打开： http://<本机IP>:8000  （网格页显示所有摄像头）
按 Ctrl+C 退出。
"""
import argparse
import html
import os
from pathlib import Path
import queue
import socket
import threading
import time
from urllib.parse import urlsplit

# RTSP 走 TCP，降低花屏/丢包概率（需在创建 VideoCapture 之前设置）
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np
import torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ultralytics import YOLO
from region_calibration import (
    CalibrationHandlerMixin, CalibrationStore, calibration_matches_frame, camera_key,
)
from traffic_api import EventStore, TrafficCamera, TrafficHandlerMixin

try:
    import av
    _HAS_AV = True
except Exception:
    _HAS_AV = False

STOP = threading.Event()


class FrameHub:
    """保存最新一帧 JPEG，供 HTTP 流读取。"""

    def __init__(self, quality=80):
        self.quality = quality
        self.lock = threading.Lock()
        self.jpeg = None
        self.seq = 0
        self.raw_frame = None

    def set_raw(self, frame):
        with self.lock:
            self.raw_frame = frame

    def get_raw(self):
        with self.lock:
            frame = self.raw_frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return buf.tobytes() if ok else None

    def set(self, img_bgr):
        # 预览缩图降低十路 JPEG 编码和浏览器开销；推理和标定底图保留原尺寸。
        h, w = img_bgr.shape[:2]
        if w > 960:
            img_bgr = cv2.resize(img_bgr, (960, max(1, round(h * 960 / w))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        with self.lock:
            self.jpeg = buf.tobytes()
            self.seq += 1

    def get(self):
        with self.lock:
            return self.jpeg

    def get_with_seq(self):
        with self.lock:
            return self.seq, self.jpeg


class AVStreamReader:
    """PyAV 抓流，可使用 CUDA/NVDEC 解码；后台只保留最新帧。"""

    def __init__(self, url, hwaccel=None):
        self.url = url
        self.hwaccel = hwaccel
        self.container = None
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.n = 0
        self.received_at = None
        self.generation = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _open(self):
        if self.container is not None:
            try:
                self.container.close()
            except Exception:
                pass
        try:
            self.container = av.open(self.url, options={"rtsp_transport": "tcp"}, timeout=(8.0, 8.0),
                                     hwaccel=self.hwaccel)
            decoder = "CUDA/NVDEC" if self.container.streams.video[0].codec_context.is_hwaccel else "CPU"
            print(f"[PyAV] RTSP 已连接: {self.url}，解码: {decoder}")
            return True
        except Exception as e:
            print(f"[PyAV 连接失败] {self.url}: {e}，2s 后重试...")
            self.container = None
            return False

    def _loop(self):
        while not STOP.is_set():
            if self.container is None:
                if not self._open():
                    time.sleep(2.0)
                    continue
            try:
                for f in self.container.decode(video=0):
                    if STOP.is_set():
                        break
                    img = f.to_ndarray(format="bgr24")
                    with self.lock:
                        self.frame = img
                        self.ok = True
                        self.n += 1
                        self.received_at = time.monotonic()
                if not STOP.is_set():
                    raise EOFError("stream ended")
            except Exception as e:
                print(f"[PyAV 解码中断] {self.url}: {e}，尝试重连...")
                with self.lock:
                    self.ok = False
                    self.generation += 1
                self._open()
                time.sleep(1.0)

    def read(self):
        with self.lock:
            if self.ok and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def stop(self):
        if self.container is not None:
            try:
                self.container.close()
            except Exception:
                pass
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


class StreamReader:
    """OpenCV 抓流（文件/摄像头/RTSP），后台线程只保留最新一帧。"""

    def __init__(self, source):
        self.source = source
        self.cap = None
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.n = 0
        self.received_at = None
        self.generation = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _open(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _loop(self):
        while not STOP.is_set():
            if self.cap is None:
                self.cap = self._open()
                if self.cap is None:
                    print(f"[连接失败] {self.source}，2s 后重试...")
                    time.sleep(2.0)
                    continue
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("[解码中断] 尝试重连...")
                with self.lock:
                    self.ok = False
                    self.generation += 1
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                time.sleep(1.0)
                continue
            with self.lock:
                self.frame = frame
                self.ok = True
                self.n += 1
                self.received_at = time.monotonic()

    def read(self):
        with self.lock:
            if self.ok and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def stop(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


def read_sample(reader):
    """Read frame, sequence and receive time under the same lock."""
    with reader.lock:
        frame = reader.frame.copy() if reader.ok and reader.frame is not None else None
        return frame, reader.n, reader.received_at, reader.generation


def grid_page(count, host_header="127.0.0.1:8000", names=None):
    host, _, port = host_header.partition(":")
    try:
        base_port = int(port or 8000)
    except ValueError:
        base_port = 8000
    host = html.escape(host, quote=True)
    cells = "".join(
        f'<section class="cam"><div class="cam-bar"><span>{html.escape(names[i] if names else f"Camera {i + 1}")}</span>'
        f'<a href="/calibrate/{i}" target="_blank" rel="noopener">区域标定</a></div>'
        f'<div class="video"><img alt="摄像头实时预览" src="//{host}:{base_port + 1 + i // 6}/stream/{i}"></div></section>'
        for i in range(count))
    return ("<!doctype html><html><head><meta charset='utf-8'><title>YOLO Multi-Cam</title><style>"
            "html,body{margin:0;background:#111}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:2px;padding:2px}"
            ".cam{position:relative;background:#000;min-width:0}.video{aspect-ratio:16/9}"
            ".cam-bar{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:8px 12px;"
            "color:#e5edf7;font:14px system-ui;background:#1b2533}.cam-bar span{overflow-wrap:anywhere}"
            ".cam-bar a{color:#bfdbfe;white-space:nowrap;padding:4px 10px;border:1px solid #52647e;border-radius:5px;text-decoration:none}"
            ".cam img{width:100%;height:100%;object-fit:contain;display:block}"
            "</style></head><body><div class='grid'>" + cells + "</div></body></html>")


def cam_name(url):
    source = urlsplit(url)
    return source.path.rstrip("/").replace("\\", "/").split("/")[-1] or source.hostname or "Camera"


def load_sources(args):
    if args.sources_file:
        urls = [ln.strip() for ln in open(args.sources_file, encoding="utf-8-sig")
                if ln.strip() and not ln.strip().startswith("#")]
        if not urls:
            raise SystemExit(f"{args.sources_file} 里没有有效的流地址")
    elif args.source:
        urls = [args.source]
    else:
        raise SystemExit("需要 --source 或 --sources-file")
    return [int(u) if str(u).isdigit() else u for u in urls]


def placeholder(text):
    img = np.full((720, 1280, 3), 25, dtype=np.uint8)
    cv2.putText(img, text, (40, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 200, 255), 3, cv2.LINE_AA)
    return img


def put_latest(pending, item):
    """单生产者预览队列：编码繁忙时替换旧帧，不阻塞推理。"""
    try:
        pending.put_nowait(item)
    except queue.Full:
        try:
            pending.get_nowait()
        except queue.Empty:
            pass
        pending.put_nowait(item)


def _encode_worker(st):
    """每路一个画框+JPEG编码线程；队列满丢旧帧（预览宁缺不滞）。"""
    while not STOP.is_set():
        try:
            item = st["q"].get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            break
        frame, result, info = item
        st["hub"].set_raw(result.orig_img)
        # 标签缩小一档，960 宽预览的框线仍保持约 2.5px。
        line_width = max(3, round(2.5 * result.orig_img.shape[1] / 960))
        annotated = result.plot(line_width=line_width, pil=False,
                                boxes=True, labels=True, conf=True, probs=False)
        calibration = st["calibration_store"].get(st["camera_key"])
        draw_calibration(annotated, calibration)
        cv2.rectangle(annotated, (0, 0), (330, 26), (0, 0, 0), -1)
        cv2.putText(annotated, info, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
        st["hub"].set(annotated)


def draw_calibration(frame, calibration):
    h, w = frame.shape[:2]
    if not calibration_matches_frame(calibration, [w, h]):
        return
    points = lambda values: np.rint(np.asarray(values) * [w - 1, h - 1]).astype(np.int32)
    roi = points(calibration["roi"])
    cv2.polylines(frame, [roi], True, (110, 231, 183), 2, cv2.LINE_AA)
    cv2.putText(frame, "ROI", tuple(roi[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110, 231, 183), 2, cv2.LINE_AA)
    if calibration["line"]:
        a, b = points(calibration["line"])
        cv2.line(frame, tuple(a), tuple(b), (36, 191, 251), 2, cv2.LINE_AA)
        for label, point in zip(("A", "B"), (a, b)):
            cv2.circle(frame, tuple(point), 4, (36, 191, 251), -1)
            cv2.putText(frame, label, tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 191, 251), 2, cv2.LINE_AA)
    if calibration["direction"]:
        a, b = points(calibration["direction"])
        cv2.arrowedLine(frame, tuple(a), tuple(b), (249, 232, 103), 2, cv2.LINE_AA, tipLength=0.2)


def parse_crop(s):
    s = (s or "none").strip().lower()
    if s in ("none", "full"):
        return None
    if s in ("left", "l"):
        return (0.0, 0.0, 0.5, 1.0)
    if s in ("right", "r"):
        return (0.5, 0.0, 1.0, 1.0)
    parts = s.replace(",", " ").split()
    if len(parts) == 4:
        return tuple(float(x) for x in parts)
    raise SystemExit(f"无法解析 --crop: {s!r}（用 none/left/right 或 4 个归一化数字）")


def local_ip():
    try:
        ips = socket.gethostbyname_ex(socket.gethostname())[2]
        lan = [i for i in ips if i.startswith("192.168.")] or ips
        return lan[-1]
    except Exception:
        return "127.0.0.1"


class MJPEGHandler(TrafficHandlerMixin, CalibrationHandlerMixin, BaseHTTPRequestHandler):
    hubs = []
    names = []
    camera_keys = []

    def log_message(self, *args):
        pass

    def _send_jpeg(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _idx(self, path, prefix):
        rest = path[len(prefix):]
        if rest in ("", "/"):
            return 0
        if rest.startswith("/"):
            try:
                return int(rest[1:])
            except ValueError:
                return None
        return None

    def do_GET(self):
        if self.traffic_get():
            return
        path = self.path.split("?")[0]
        if self.calibration_get(path):
            return
        if path in ("/", "/index.html"):
            body = grid_page(len(self.hubs), self.headers.get("Host", "127.0.0.1:8000"), self.names).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/snapshot" or path.startswith("/snapshot/"):
            idx = self._idx(path, "/snapshot")
            if idx is None or not (0 <= idx < len(self.hubs)):
                self.send_error(404)
                return
            data = self.hubs[idx].get()
            if data is None:
                self.send_error(503, "no frame yet")
            else:
                self._send_jpeg(data)
        elif path == "/stream" or path.startswith("/stream/"):
            idx = self._idx(path, "/stream")
            if idx is None or not (0 <= idx < len(self.hubs)):
                self.send_error(404)
                return
            hub = self.hubs[idx]
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                last_seq, last_sent = -1, 0.0
                while not STOP.is_set():
                    seq, data = hub.get_with_seq()
                    if data is None:
                        time.sleep(0.02)
                        continue
                    now = time.time()
                    # 只发新帧，并限制到 25fps，避免同一张 JPEG 被 3ms 循环刷爆带宽
                    if seq == last_seq or now - last_sent < 1 / 25:
                        time.sleep(0.005)
                        continue
                    last_seq, last_sent = seq, now
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(b"Content-Length: %d\r\n\r\n" % len(data))
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.003)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        else:
            self.send_error(404)


def serve(host, port):
    ThreadingHTTPServer.allow_reuse_address = False  # Windows 下 SO_REUSEADDR 会让两个实例静默抢占同一端口、互踢 RTSP
    httpd = ThreadingHTTPServer((host, port), MJPEGHandler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="best.pt 路径")
    ap.add_argument("--source", default=None, help="单路：RTSP URL / 视频文件 / 摄像头序号")
    ap.add_argument("--sources-file", default=None, help="多路：每行一个流地址的文本文件（# 为注释）")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="0", help="0 / 0,1 / cpu")
    ap.add_argument("--half", action="store_true", help="FP16（需 CUDA）")
    ap.add_argument("--track", action="store_true", help="预览显示 ByteTrack ID（事件 API 始终启用逐路追踪）")
    ap.add_argument("--classes", type=int, nargs="*", default=None, help="只保留的类别 id")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--calibration-file", type=Path,
                    default=Path(__file__).with_name("camera_regions.json"))
    ap.add_argument("--events-db", type=Path,
                    default=Path(__file__).parent / "output" / "traffic_events.sqlite3")
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--backend", choices=["auto", "av", "cv2"], default="auto",
                    help="抓流后端：auto=有 av 就优先用 PyAV（RTSP 更稳），否则 cv2")
    ap.add_argument("--crop", default="none",
                    help='裁剪后推理：none | left | right | "x1 y1 x2 y2"(归一化比例，如 "0 0 0.5 1")')
    args = ap.parse_args()

    urls = load_sources(args)

    crop = parse_crop(args.crop)
    calibration_path = args.calibration_file
    try:
        calibration_store = CalibrationStore(calibration_path)
    except (OSError, ValueError) as error:
        raise SystemExit(f"无法读取标定文件 {calibration_path}: {error}") from error

    use_cuda = torch.cuda.is_available() and str(args.device) != "cpu"
    use_half = bool(args.half) and use_cuda
    if args.half and not use_half:
        print("[警告] 当前无可用 CUDA 或 device=cpu，已自动关闭 FP16。")

    print(f"加载模型: {args.model}")
    model = YOLO(args.model, task="detect")

    # 新版 ultralytics 用 quantize=16 表示 FP16，旧版用 half=True；预热顺便探测参数名
    prec = {"quantize": 16} if use_half else {}
    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    try:
        model.predict(dummy, imgsz=args.imgsz, device=args.device, verbose=False, **prec)
    except TypeError:
        prec = {"half": True} if use_half else {}
        try:
            model.predict(dummy, imgsz=args.imgsz, device=args.device, verbose=False, **prec)
        except Exception as e:
            print(f"[警告] 预热失败（可忽略）: {e}")
    except Exception as e:
        print(f"[警告] 预热失败（可忽略）: {e}")

    is_rtsp = all(isinstance(u, str) and u.lower().startswith(("rtsp", "rtmp", "http")) for u in urls)
    backend = args.backend
    if backend == "auto":
        backend = "av" if (_HAS_AV and is_rtsp) else "cv2"
    if backend == "av" and not _HAS_AV:
        print("[提示] 未安装 av 模块，改用 cv2 后端。")
        backend = "cv2"
    print(f"[抓流后端] {backend}，共 {len(urls)} 路流")

    hwaccel = None
    if backend == "av" and use_cuda:
        hwaccel = av.codec.hwaccel.HWAccel("cuda", device=str(args.device).split(",")[0],
                                          allow_software_fallback=True)
        print("[视频解码] CUDA/NVDEC；不支持硬解的编码使用 CPU。")

    streams = []
    event_store = EventStore(args.events_db)
    for u in urls:
        reader = AVStreamReader(u, hwaccel=hwaccel) if backend == "av" else StreamReader(u)
        hub = FrameHub(quality=args.jpeg_quality)
        hub.set(placeholder("等待视频流..."))
        st = {"url": u, "reader": reader, "hub": hub, "seen": -1,
              "camera_key": camera_key(u, crop), "calibration_store": calibration_store,
              "fps": 0.0, "t_prev": time.time(), "q": queue.Queue(maxsize=1)}
        st["traffic"] = TrafficCamera(st["camera_key"], cam_name(str(u)), calibration_store, event_store)
        st["worker"] = threading.Thread(target=_encode_worker, args=(st,), daemon=True)
        st["worker"].start()
        streams.append(st)
    MJPEGHandler.hubs = [s["hub"] for s in streams]
    MJPEGHandler.names = [cam_name(str(u)) for u in urls]
    MJPEGHandler.camera_keys = [s["camera_key"] for s in streams]
    MJPEGHandler.calibration_store = calibration_store
    MJPEGHandler.traffic_cameras = [s["traffic"] for s in streams]
    MJPEGHandler.event_store = event_store
    try:
        httpd = serve(args.host, args.port)
    except OSError as e:
        print(f"[错误] 无法监听 {args.host}:{args.port} —— {e}\n"
              f"        端口可能被占用，请换 --port（如 --port 8080）。")
        return
    extra = []
    # 视频长连接独立分流，主端口保留给页面和标定请求，避免浏览器连接池被占满。
    for off in range(1, (len(urls) + 5) // 6 + 1):
        try:
            extra.append(serve(args.host, args.port + off))
        except OSError:
            print(f"[提示] 端口 {args.port + off} 被占用，该分组画面可能加载失败。")
    print("=" * 60)
    print(f"  本机浏览器打开:  http://127.0.0.1:{args.port}")
    print(f"  局域网其它设备:  http://{local_ip()}:{args.port}")
    print("=" * 60)
    for s in streams:
        print(f"连接视频流: {s['url']}")

    common = dict(conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                  device=args.device, verbose=False, classes=args.classes, **prec)
    last_warn_t = 0.0
    try:
        with torch.no_grad():
            while not STOP.is_set():
                # 攒齐各路新帧 → 一次批量推理；画框/编码交给各路工作线程并行做
                jobs = []
                for st in streams:
                    frame, sequence, received_at, generation = read_sample(st["reader"])
                    if frame is None or st["seen"] == sequence:
                        continue
                    st["seen"] = sequence
                    if crop is not None:
                        h, w = frame.shape[:2]
                        x1f, y1f, x2f, y2f = crop
                        frame = frame[int(y1f * h):int(y2f * h), int(x1f * w):int(x2f * w)]
                    jobs.append((st, frame, received_at, generation))

                if jobs:
                    now = time.time()
                    try:
                        # ByteTrack needs low-confidence detections to reconnect existing tracks.
                        results = model.predict([job[1] for job in jobs], **dict(common, conf=min(args.conf, .1)))
                    except Exception as e:
                        print(f"[推理异常] {e}")
                        results = None
                    if results is not None:
                        for (st, _, received_at, generation), res in zip(jobs, results):
                            tracks = st["traffic"].process(res, received_at, generation)
                            if args.track and len(tracks):
                                res = res[tracks[:, -1].astype(int)]
                                res.update(boxes=tracks[:, :-1])
                            res = res[res.boxes.conf >= args.conf]
                            dt = now - st["t_prev"]
                            st["t_prev"] = now
                            if dt > 0:
                                st["fps"] = 0.9 * st["fps"] + 0.1 * (1.0 / dt)
                            info = f"FPS {st['fps']:4.1f}  imgsz {args.imgsz}  conf {args.conf}"
                            # shortcut: fps 统计送检速率；显示速率需以实际收到的 JPEG 为准。
                            put_latest(st["q"], (None, res, info))
                    last_warn_t = time.time()
                else:
                    now = time.time()
                    if now - last_warn_t > 5:
                        last_warn_t = now
                        print("[等待中] 所有流都还没有新帧；浏览器显示占位图。")
                    time.sleep(0.01)
                for st in streams:
                    st["traffic"].flush(time.monotonic())
    except KeyboardInterrupt:
        pass
    finally:
        STOP.set()
        for st in streams:
            st["reader"].stop()
            put_latest(st["q"], None)
        httpd.shutdown()
        for h in extra:
            h.shutdown()
        for st in streams:
            st["traffic"].flush(float("inf"))
        event_store.close()
        print("已退出。")


if __name__ == "__main__":
    main()
