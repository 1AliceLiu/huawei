"""实时视频流预测 + 浏览器 MJPEG 预览。

用法示例：
    python realtime_web.py \
        --model runs/detect/yolo26m_merged12_20261013_0935/weights/best.pt \
        --source "rtsp://118.140.234.166:8554/dahua1001619" \
        --imgsz 960 --half --device 0 --port 8000 --crop left

然后浏览器打开： http://<本机IP>:8000
按 Ctrl+C 退出。
"""
import argparse
import os
import socket
import threading
import time

# RTSP 走 TCP，降低花屏/丢包概率（需在创建 VideoCapture 之前设置）
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np
import torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ultralytics import YOLO

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

    def set(self, img_bgr):
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        with self.lock:
            self.jpeg = buf.tobytes()
            self.seq += 1

    def get(self):
        with self.lock:
            return self.jpeg


class AVStreamReader:
    """PyAV 抓流（与 main.py 一致），RTSP 兼容性通常好于 OpenCV。只保留最新帧。"""

    def __init__(self, url):
        self.url = url
        self.container = None
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _open(self):
        if self.container is not None:
            try:
                self.container.close()
            except Exception:
                pass
        try:
            self.container = av.open(self.url, options={"rtsp_transport": "tcp", "stimeout": "8000000"})
            print("[PyAV] RTSP 已连接")
            return True
        except Exception as e:
            print(f"[PyAV 连接失败] {e}，2s 后重试...")
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
            except Exception as e:
                print(f"[PyAV 解码中断] {e}，尝试重连...")
                with self.lock:
                    self.ok = False
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


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>YOLO Realtime</title></head>
<body style="margin:0;background:#111;text-align:center">
<div style="color:#0f0;font-family:sans-serif;padding:6px">YOLO 实时预测</div>
<img src="/stream" style="max-width:100%;height:auto">
</body></html>"""


def placeholder(text):
    img = np.full((720, 1280, 3), 25, dtype=np.uint8)
    cv2.putText(img, text, (40, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 200, 255), 3, cv2.LINE_AA)
    return img


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
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class MJPEGHandler(BaseHTTPRequestHandler):
    hub = None

    def log_message(self, *args):
        pass

    def _send_jpeg(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/snapshot":
            data = self.hub.get()
            if data is None:
                self.send_error(503, "no frame yet")
            else:
                self._send_jpeg(data)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not STOP.is_set():
                    data = self.hub.get()
                    if data is None:
                        time.sleep(0.02)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(b"Content-Length: %d\r\n\r\n" % len(data))
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.003)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def serve(hub, host, port):
    MJPEGHandler.hub = hub
    httpd = ThreadingHTTPServer((host, port), MJPEGHandler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="best.pt 路径")
    ap.add_argument("--source", required=True, help="RTSP URL / 视频文件 / 摄像头序号")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="0", help="0 / 0,1 / cpu")
    ap.add_argument("--half", action="store_true", help="FP16（需 CUDA）")
    ap.add_argument("--track", action="store_true", help="启用 ByteTrack 追踪（带 ID）")
    ap.add_argument("--classes", type=int, nargs="*", default=None, help="只保留的类别 id")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--backend", choices=["auto", "av", "cv2"], default="auto",
                    help="抓流后端：auto=有 av 就优先用 PyAV（RTSP 更稳），否则 cv2")
    ap.add_argument("--crop", default="none",
                    help='裁剪后推理：none | left | right | "x1 y1 x2 y2"(归一化比例，如 "0 0 0.5 1")')
    args = ap.parse_args()

    if args.source.isdigit():
        args.source = int(args.source)

    crop = parse_crop(args.crop)

    use_half = bool(args.half) and torch.cuda.is_available() and str(args.device) != "cpu"
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

    hub = FrameHub(quality=args.jpeg_quality)
    hub.set(placeholder("等待视频流..."))
    try:
        httpd = serve(hub, args.host, args.port)
    except OSError as e:
        print(f"[错误] 无法监听 {args.host}:{args.port} —— {e}\n"
              f"        端口可能被占用，请换 --port（如 --port 8080）。")
        return
    print("=" * 60)
    print(f"  本机浏览器打开:  http://127.0.0.1:{args.port}")
    print(f"  局域网其它设备:  http://{local_ip()}:{args.port}")
    print("=" * 60)

    is_url = isinstance(args.source, str) and args.source.lower().startswith(("rtsp", "rtmp", "http"))
    backend = args.backend
    if backend == "auto":
        backend = "av" if (_HAS_AV and is_url) else "cv2"
    if backend == "av" and not _HAS_AV:
        print("[提示] 未安装 av 模块，改用 cv2 后端。")
        backend = "cv2"
    print(f"[抓流后端] {backend}")
    reader = AVStreamReader(args.source) if backend == "av" else StreamReader(args.source)
    print(f"连接视频流: {args.source}")

    fps, t_prev = 0.0, time.time()
    last_warn_t = 0.0
    try:
        with torch.no_grad():
            while not STOP.is_set():
                ok, frame = reader.read()
                if not ok or frame is None:
                    now = time.time()
                    if now - last_warn_t > 5:
                        last_warn_t = now
                        print(f"[等待中] 还收不到帧（source={args.source}）；"
                              f"浏览器此时应显示“等待视频流...”占位图。")
                    time.sleep(0.02)
                    continue

                if crop is not None:
                    h, w = frame.shape[:2]
                    x1f, y1f, x2f, y2f = crop
                    frame = frame[int(y1f * h):int(y2f * h), int(x1f * w):int(x2f * w)]

                common = dict(conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                              device=args.device, verbose=False, classes=args.classes, **prec)
                try:
                    if args.track:
                        results = model.track(frame, persist=True, tracker="bytetrack.yaml", **common)
                    else:
                        results = model.predict(frame, **common)
                except Exception as e:
                    print(f"[推理异常] {e}")
                    time.sleep(0.1)
                    continue

                annotated = results[0].plot(line_width=1, pil=False,
                                            boxes=True, labels=True, conf=True, probs=False)

                now = time.time()
                dt = now - t_prev
                t_prev = now
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps
                info = f"FPS {fps:4.1f}  imgsz {args.imgsz}  conf {args.conf}"
                cv2.rectangle(annotated, (0, 0), (330, 26), (0, 0, 0), -1)
                cv2.putText(annotated, info, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2, cv2.LINE_AA)

                hub.set(annotated)
    except KeyboardInterrupt:
        pass
    finally:
        STOP.set()
        reader.stop()
        httpd.shutdown()
        print("已退出。")


if __name__ == "__main__":
    main()
