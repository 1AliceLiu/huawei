"""Per-camera tracking, ordered crossing events and a durable JSON event feed."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import ROOT, YAML

from region_calibration import calibration_matches_frame
from traffic_count import LineCounter


class CameraTracker(BYTETracker):
    @staticmethod
    def reset_id():
        # ByteTrack's counter is process-wide; resetting one camera must not reuse another camera's IDs.
        pass


class EventStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            payload TEXT NOT NULL,
            UNIQUE(camera_id, sequence))""")
        self.connection.commit()

    def append(self, camera_id, events):
        if not events:
            return
        with self.lock, self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            sequence = self.connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM events WHERE camera_id = ?",
                (camera_id,)).fetchone()[0]
            for event in events:
                sequence += 1
                payload = json.dumps(dict(event, camera_id=camera_id, sequence=sequence),
                                     ensure_ascii=False, allow_nan=False)
                self.connection.execute(
                    "INSERT INTO events(camera_id, sequence, payload) VALUES (?, ?, ?)",
                    (camera_id, sequence, payload))

    def read(self, after, limit, camera_id=None):
        sql = "SELECT id, payload FROM events WHERE id > ?"
        params = [after]
        if camera_id is not None:
            sql += " AND camera_id = ?"
            params.append(camera_id)
        with self.lock:
            rows = self.connection.execute(sql + " ORDER BY id LIMIT ?", params + [limit + 1]).fetchall()
        events = [dict(json.loads(payload), event_id=event_id) for event_id, payload in rows[:limit]]
        return {"events": events, "next_after": events[-1]["event_id"] if events else after,
                "has_more": len(rows) > limit}

    def close(self):
        with self.lock:
            self.connection.close()


class TrafficCamera:
    def __init__(self, camera_id, name, calibration_store, event_store):
        self.camera_id, self.name = camera_id, name
        self.calibration_store, self.event_store = calibration_store, event_store
        self.lock = threading.Lock()
        self.tracker = CameraTracker(SimpleNamespace(**YAML.load(ROOT / "cfg/trackers/bytetrack.yaml")))
        self.calibration = None
        self.counter = None
        self.frame_size = None
        self.last_frame = None
        self.generation = None
        self.resets = 0
        self.state = "waiting_for_frame"
        self.pending_events = []
        self.clock_offset = time.time() - time.monotonic()

    def _publish(self, watermark):
        ready = sorted((event for event in self.pending_events if event["time"] < watermark),
                       key=lambda event: (event["time"], event["track_id"]))
        self.event_store.append(self.camera_id, [event["payload"] for event in ready])
        self.pending_events = [event for event in self.pending_events if event["time"] >= watermark]

    def process(self, result, received_at, generation):
        with self.lock:
            return self._process(result, received_at, generation)

    def _process(self, result, received_at, generation):
        height, width = result.orig_shape
        try:
            calibration = self.calibration_store.get(self.camera_id)
        except (OSError, ValueError):
            calibration = None
            self.state = "calibration_error"
        else:
            if not calibration or not calibration["line"] or not calibration["direction"]:
                self.state = "needs_calibration"
            elif not calibration_matches_frame(calibration, [width, height]):
                self.state = "frame_size_mismatch"
            else:
                self.state = "active"
        reset = (calibration != self.calibration or self.frame_size != [width, height]
                 or generation != self.generation or self.last_frame is None
                 or not 0 < received_at - self.last_frame <= .5)
        if reset:
            self._publish(math.inf)
            self.tracker.reset()
            self.counter = None
            self.resets += self.last_frame is not None
        self.calibration = calibration
        self.frame_size = [width, height]
        self.last_frame, self.generation = received_at, generation
        tracks = self.tracker.update(result.boxes.cpu().numpy(), result.orig_img)
        if self.state == "active":
            if self.counter is None:
                self.counter = LineCounter(dict(calibration, frame_size=[width, height]),
                                           keep_history=False, min_roi_observations=2)
            self.counter.expire(received_at)
            boxes = result.boxes.cpu().numpy()
            for track in tracks:
                track_id, detection_index = int(track[4]), int(track[-1])
                x1, y1, x2, y2 = boxes.xyxy[detection_index].tolist()
                point = [min(1, max(0, (x1 + x2) / 2 / (width - 1))),
                         min(1, max(0, y2 / (height - 1)))]
                class_id = int(boxes.cls[detection_index])
                quality = math.sqrt(max(0, (x2 - x1) * (y2 - y1)) / (width * height))
                if x1 <= 2 or y1 <= 2 or x2 >= width - 2 or y2 >= height - 2:
                    quality *= .25
                event = self.counter.observe(track_id, point, received_at, class_id,
                                             float(boxes.conf[detection_index]), quality)
                if event:
                    chosen = event["label"]
                    crossed_at = datetime.fromtimestamp(event["crossing_time_s"] + self.clock_offset,
                                                        timezone.utc).isoformat(timespec="milliseconds")
                    self.pending_events.append({"time": event["crossing_time_s"], "track_id": track_id,
                        "payload": {"camera_name": self.name, "crossed_at": crossed_at,
                                    "class_id": chosen, "class_name": result.names[chosen]}})
            self._flush(received_at)
        return tracks

    def _flush(self, now):
        watermark = now - .5
        if self.counter:
            self.counter.expire(now)
            candidates = [state["pending"]["crossing_time_s"] for state in self.counter.tracks.values()
                          if state["pending"] is not None and not state["counted"]]
            if candidates:
                watermark = min(watermark, min(candidates))
        self._publish(watermark)

    def flush(self, now):
        with self.lock:
            self._flush(now)

    def status(self):
        with self.lock:
            age = None if self.last_frame is None else max(0, time.monotonic() - self.last_frame)
            state = self.state
            try:
                saved = self.calibration_store.get(self.camera_id)
                if not saved or not saved["line"] or not saved["direction"]:
                    state = "needs_calibration"
                elif age is not None and age > 2:
                    state = "stream_stale"
            except (OSError, ValueError):
                state = "calibration_error"
            return {"camera_id": self.camera_id, "camera_name": self.name, "status": state,
                    "last_frame_age_s": None if age is None else round(age, 3),
                    "tracking_resets": self.resets, "pending_events": len(self.pending_events)}


class TrafficHandlerMixin:
    traffic_cameras = []
    event_store = None

    def traffic_get(self):
        url = urlsplit(self.path)
        if not url.path.startswith("/api/traffic/"):
            return False
        status, body = 200, {}
        try:
            if self.event_store is None:
                status, body = 503, {"error": "车辆事件服务尚未启动"}
            elif url.path == "/api/traffic/cameras":
                body = {"cameras": [camera.status() for camera in self.traffic_cameras]}
            elif url.path == "/api/traffic/events":
                query = parse_qs(url.query, keep_blank_values=True, max_num_fields=3)
                if set(query) - {"after", "limit", "camera_id"} or any(len(v) != 1 for v in query.values()):
                    raise ValueError("查询参数无效")
                after = int(query.get("after", ["0"])[0])
                limit = int(query.get("limit", ["100"])[0])
                if not 0 <= after <= 9223372036854775807 or not 1 <= limit <= 1000:
                    raise ValueError("after 必须为非负整数，limit 必须为 1 至 1000")
                camera_id = query.get("camera_id", [None])[0]
                if camera_id is not None and camera_id not in {c.camera_id for c in self.traffic_cameras}:
                    status, body = 404, {"error": "摄像头不存在"}
                else:
                    body = self.event_store.read(after, limit, camera_id)
            else:
                status, body = 404, {"error": "接口不存在"}
        except ValueError:
            status, body = 400, {"error": "查询参数无效"}
        except sqlite3.Error:
            status, body = 503, {"error": "事件存储暂时不可用"}
        self._calibration_response(status, body)
        return True
