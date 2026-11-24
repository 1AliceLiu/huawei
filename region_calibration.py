"""Per-camera geometry validation, atomic storage and preview HTTP endpoints."""
import copy
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
from urllib.parse import urlsplit


def camera_key(source, crop):
    return hashlib.sha256(json.dumps([str(source), crop], separators=(",", ":")).encode()).hexdigest()


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p):
    return (abs(_cross(a, b, p)) < 1e-9
            and min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9
            and min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9)


def _intersects(a, b, c, d):
    return ((_cross(a, b, c) * _cross(a, b, d) < 0
             and _cross(c, d, a) * _cross(c, d, b) < 0)
            or any((_on_segment(a, b, c), _on_segment(a, b, d),
                    _on_segment(c, d, a), _on_segment(c, d, b))))


def _inside(point, polygon):
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if _on_segment(a, b, point):
            return True
        if (a[1] > point[1]) != (b[1] > point[1]):
            if point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]:
                inside = not inside
    return inside


def validate_calibration(data):
    if not isinstance(data, dict) or set(data) != {"roi", "line", "direction", "frame_size"}:
        raise ValueError("标定数据需要包含 ROI、计数线、方向和画面尺寸")
    size = data["frame_size"]
    if (not isinstance(size, list) or len(size) != 2
            or any(type(n) is not int or not 2 <= n <= 32768 for n in size)):
        raise ValueError("画面尺寸无效，请重新获取底图")
    for name in ("roi", "line", "direction"):
        points = data[name]
        if not isinstance(points, list) or len(points) > 64:
            raise ValueError("每个图形最多允许 64 个点")
        for point in points:
            if (not isinstance(point, list) or len(point) != 2
                    or any(type(n) not in (int, float) or not math.isfinite(n) or not 0 <= n <= 1
                           for n in point)):
                raise ValueError("坐标必须是画面内的有效点")
        if len({tuple(p) for p in points}) != len(points):
            raise ValueError("图形中不能有重复点")
    roi, line, direction = data["roi"], data["line"], data["direction"]
    if not roi and not line and not direction:
        return copy.deepcopy(data)
    if len(roi) < 3:
        raise ValueError("ROI 至少需要 3 个点")
    edges = list(zip(roi, roi[1:] + roi[:1]))
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in edges)
    if abs(area) < 1e-6:
        raise ValueError("ROI 面积太小，请重新绘制")
    for i, (a, b) in enumerate(edges):
        for j in range(i + 1, len(edges)):
            if j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            if _intersects(a, b, *edges[j]):
                raise ValueError("ROI 边界不能交叉")
    if not line and not direction:
        return copy.deepcopy(data)
    if len(line) != 2 or len(direction) != 2:
        raise ValueError("计数线和方向箭头都需要各画 2 个点；也可仅保存 ROI")
    if any(not _inside(point, roi) for point in line):
        raise ValueError("计数线的两个端点必须位于 ROI 内")
    # Check the whole segment, including concave polygons and boundary endpoints.
    a, b = line
    cuts = [0.0, 1.0]
    dx, dy = b[0] - a[0], b[1] - a[1]
    for c, d in edges:
        ex, ey = d[0] - c[0], d[1] - c[1]
        denominator = dx * ey - dy * ex
        if abs(denominator) > 1e-9:
            t = ((c[0] - a[0]) * ey - (c[1] - a[1]) * ex) / denominator
            u = ((c[0] - a[0]) * dy - (c[1] - a[1]) * dx) / denominator
            if 0 < t < 1 and 0 <= u <= 1:
                cuts.append(t)
    cuts.sort()
    for left, right in zip(cuts, cuts[1:]):
        t = (left + right) / 2
        if not _inside([a[0] + t * dx, a[1] + t * dy], roi):
            raise ValueError("整条计数线必须位于 ROI 内")
    vector_end = [a[0] + direction[1][0] - direction[0][0],
                  a[1] + direction[1][1] - direction[0][1]]
    if abs(_cross(a, b, vector_end)) < 1e-9:
        raise ValueError("方向箭头需要穿越计数线方向，不能与计数线平行")
    return copy.deepcopy(data)


def calibration_matches_frame(calibration, frame_size):
    return bool(calibration and calibration["roi"] and math.isclose(
        calibration["frame_size"][0] / calibration["frame_size"][1],
        frame_size[0] / frame_size[1], rel_tol=0.002))


class CalibrationStore:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.stamp = self._stamp()
        self.data = self._read()

    def _read(self):
        data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        if not isinstance(data, dict):
            raise ValueError("标定文件必须是按摄像头保存的 JSON 对象")
        for value in data.values():
            validate_calibration(value)
        return data

    def _stamp(self):
        try:
            stat = self.path.stat()
            return stat.st_mtime_ns, stat.st_size, stat.st_ino
        except FileNotFoundError:
            return None

    @contextmanager
    def _file_lock(self):
        # OS locks release on process exit; a crashed preview cannot leave a stale lock.
        with self.path.with_suffix(self.path.suffix + ".lock").open("a+b") as file:
            if os.name == "nt":
                import msvcrt
                if file.tell() == 0:
                    file.write(b"\0")
                file.flush()
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    file.seek(0)
                    msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)

    def get(self, key):
        with self.lock:
            stamp = self._stamp()
            if stamp != self.stamp:
                self.data = self._read()
                self.stamp = stamp
            return copy.deepcopy(self.data.get(key))

    def save(self, key, value):
        value = validate_calibration(value)
        with self.lock, self._file_lock():
            updated = self._read()
            updated[key] = value
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                                 prefix=self.path.name + ".", suffix=".tmp", delete=False) as file:
                    temporary = file.name
                    json.dump(updated, file, ensure_ascii=False, indent=2, allow_nan=False)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, self.path)
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
            self.data = updated
            self.stamp = self._stamp()


class CalibrationHandlerMixin:
    def _calibration_response(self, status, body, content_type="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _calibration_index(self, path):
        match = re.fullmatch(r"/(?:api/calibration|calibrate|calibration-frame)/(\d+)", path)
        if not match or not 0 <= int(match[1]) < len(self.camera_keys):
            self._calibration_response(404, {"error": "摄像头不存在"})
            return None
        return int(match[1])

    def calibration_get(self, path):
        if path == "/calibration.js":
            asset, mime = "calibration.js", "text/javascript; charset=utf-8"
        elif path.startswith("/calibrate/"):
            if self._calibration_index(path) is None:
                return True
            asset, mime = "calibration.html", "text/html; charset=utf-8"
        elif path.startswith("/api/calibration/"):
            idx = self._calibration_index(path)
            if idx is not None:
                self._calibration_response(200, {"camera": self.names[idx],
                    "calibration": self.calibration_store.get(self.camera_keys[idx])})
            return True
        elif path.startswith("/calibration-frame/"):
            idx = self._calibration_index(path)
            if idx is not None:
                data = self.hubs[idx].get_raw()
                self._calibration_response(200 if data else 503,
                    data if data else {"error": "尚无有效视频帧，请稍后刷新底图"}, "image/jpeg" if data else "application/json; charset=utf-8")
            return True
        else:
            return False
        self._calibration_response(200, (Path(__file__).parent / "web" / asset).read_bytes(), mime)
        return True

    def do_POST(self):
        path = self.path.split("?")[0]
        if not path.startswith("/api/calibration/"):
            self._calibration_response(404, {"error": "接口不存在"})
            return
        idx = self._calibration_index(path)
        if idx is None:
            return
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self._calibration_response(403, {"error": "不允许跨站保存标定"})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            self._calibration_response(415, {"error": "请使用 JSON 数据保存"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 32768:
                raise ValueError("标定请求大小无效")
            data = json.loads(self.rfile.read(length))
            self.calibration_store.save(self.camera_keys[idx], data)
        except (ValueError, UnicodeDecodeError) as error:
            self._calibration_response(400, {"error": str(error)})
        except OSError:
            self._calibration_response(500, {"error": "保存失败，请检查标定文件的写入权限和磁盘空间"})
        else:
            self._calibration_response(200, {"saved": True})
