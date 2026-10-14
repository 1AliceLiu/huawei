import av
import cv2
import threading
import time
import numpy as np
import torch
from ultralytics import YOLO

# ================= 配置参数 =================
RTSP_URL = "rtsp://118.140.234.166:8554/dahua1000393"
# RTSP_URL = "rtsp://118.140.234.166:8554/dahua1001620"
MODEL_PATH = r"runs/detect/yolo26m_merged12_20261013_0935/weights/best.pt"
CONF_THRES = 0.2
DEVICE = 0

IMGSZ = 1280
KEEP_CLASSES = None 

# ================= 鱼眼参数设置 =================
# k1 > 0 表示桶形膨胀（中间凸起），数值越大鱼眼效果越剧烈（建议范围：0.2 ~ 0.8）
FISHEYE_K1 = 0.5
FISHEYE_K2 = 0.1
# 画面放缩因子：膨胀后四周会有黑边，可以调大（如 1.2 ~ 1.5）来放大画面填满视野
SCALE = 1.0


class FisheyeTransform:
    """鱼眼（桶形膨胀）效果生成器"""
    def __init__(self, width, height, k1=0.5, k2=0.1, scale=1.0):
        self.w = width
        self.h = height
        
        cx, cy = width / 2.0, height / 2.0
        fx = fy = max(width, height) / 2.0
        
        # 原始相机内参
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float32)
        
        # 桶形畸变系数：k1 为正数会产生凸起鱼眼效果
        D = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # 新内参矩阵（调整 scale 可以放大/缩小鱼眼画面，隐藏边缘黑边）
        new_K = K.copy()
        new_K[0, 0] *= scale
        new_K[1, 1] *= scale
        
        # 使用 initUndistortRectifyMap 计算映射表
        # 注意：这里交换了 K 和 new_K 的位置，以实现正向的“鱼眼膨胀”效果
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            new_K, D, np.eye(3), K, (width, height), cv2.CV_32FC1
        )

    def apply(self, img):
        """应用鱼眼效果，边缘填充黑色"""
        return cv2.remap(
            img, 
            self.map1, 
            self.map2, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )


class AVStreamer:
    """基于 PyAV 的多线程 RTSP 读取类"""
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.container = None
        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.stopped = False

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def _open(self):
        if self.container is not None:
            try:
                self.container.close()
            except Exception:
                pass
        try:
            self.container = av.open(
                self.rtsp_url,
                options={"rtsp_transport": "tcp", "stimeout": "8000000"},
            )
            return True
        except Exception as e:
            print(f"[连接失败] {e}")
            self.container = None
            return False

    def update(self):
        while not self.stopped:
            if self.container is None:
                if not self._open():
                    time.sleep(2.0)
                    continue

            try:
                for frame in self.container.decode(video=0):
                    if self.stopped:
                        break
                    img = frame.to_ndarray(format="bgr24")
                    with self.lock:
                        self.frame = img
                        self.ret = True
            except Exception as e:
                print(f"[解码中断] {e}，尝试重连...")
                with self.lock:
                    self.ret = False
                self._open()
                time.sleep(1.0)

    def read(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.container is not None:
            try:
                self.container.close()
            except Exception:
                pass


def main():
    print("加载自定义交通模型中...")
    model = YOLO(MODEL_PATH, task="detect")

    print(f"连接 RTSP... {RTSP_URL}")
    streamer = AVStreamer(RTSP_URL)

    cv2.namedWindow("Traffic Preview", cv2.WINDOW_AUTOSIZE)

    fisheye = None

    print("开始实时预览（按 'q' 或 'ESC' 退出）...")

    try:
        with torch.no_grad():
            while True:
                success, frame = streamer.read()
                if not success or frame is None:
                    time.sleep(0.01)
                    continue

                # 截取左半区域
                left = frame[:, : frame.shape[1] // 2]

                # 动态初始化鱼眼变换类
                if fisheye is None:
                    h, w = left.shape[:2]
                    fisheye = FisheyeTransform(
                        width=w, 
                        height=h, 
                        k1=FISHEYE_K1, 
                        k2=FISHEYE_K2, 
                        scale=SCALE
                    )

                # 1. 增加鱼眼（凸起）效果
                left_fisheye = fisheye.apply(left)

                # 2. YOLO 追踪推理
                results = model.track(
                    left_fisheye,
                    conf=CONF_THRES,
                    imgsz=IMGSZ,
                    verbose=False,
                    device=DEVICE,
                    classes=KEEP_CLASSES,
                    persist=True,
                    tracker="bytetrack.yaml",
                )

                # 3. 绘制检测框
                annotated_frame = results[0].plot(
                    line_width=1,
                    pil=False,
                    boxes=True,
                    labels=True,
                    probs=False,
                )

                cv2.imshow("Traffic Preview", annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()
        cv2.destroyAllWindows()
        print("预览结束。")


if __name__ == "__main__":
    main()