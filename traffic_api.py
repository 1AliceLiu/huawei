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

import torch
from torchvision.ops import nms
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import ROOT, YAML

from region_calibration import calibration_matches_frame
from stage_timing import METRICS
from traffic_count import LineCounter


class CameraTracker(BYTETracker):
    @staticmethod
    def reset_id():
        # ByteTrack's counter is process-wide; resetting one camera must not reuse another camera's IDs.
        pass


class EventStore:
    def __init__(self, path, class_names=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.class_names = dict(class_names or {})
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
        with self.connection:
            self.connection.execute("""CREATE TABLE IF NOT EXISTS vehicle_counts (
                camera_id TEXT NOT NULL, class_id INTEGER NOT NULL, class_name TEXT NOT NULL,
                total INTEGER NOT NULL, first_crossed_at TEXT NOT NULL, last_crossed_at TEXT NOT NULL,
                last_event_id INTEGER NOT NULL, PRIMARY KEY(camera_id, class_id))""")
            self.connection.execute("""CREATE TABLE IF NOT EXISTS counts_push_state (
                destination TEXT PRIMARY KEY, digest TEXT NOT NULL)""")
            # Rebuild the derived totals on startup, including events written by an older deployment.
            self.connection.execute("BEGIN IMMEDIATE")
            self.connection.execute("DELETE FROM vehicle_counts")
            self.connection.execute("""INSERT INTO vehicle_counts
                SELECT camera_id, json_extract(payload, '$.class_id'), json_extract(payload, '$.class_name'),
                       COUNT(*), MIN(json_extract(payload, '$.crossed_at')),
                       MAX(json_extract(payload, '$.crossed_at')), MAX(id)
                FROM events GROUP BY camera_id, json_extract(payload, '$.class_id')""")

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
                inserted = self.connection.execute(
                    "INSERT INTO events(camera_id, sequence, payload) VALUES (?, ?, ?)",
                    (camera_id, sequence, payload))
                self.connection.execute("""INSERT INTO vehicle_counts VALUES (?, ?, ?, 1, ?, ?, ?)
                    ON CONFLICT(camera_id, class_id) DO UPDATE SET
                        class_name=excluded.class_name, total=vehicle_counts.total+1,
                        first_crossed_at=MIN(vehicle_counts.first_crossed_at, excluded.first_crossed_at),
                        last_crossed_at=MAX(vehicle_counts.last_crossed_at, excluded.last_crossed_at),
                        last_event_id=excluded.last_event_id""",
                    (camera_id, event["class_id"], event["class_name"], event["crossed_at"],
                     event["crossed_at"], inserted.lastrowid))

    def counts(self, cameras):
        with self.lock:
            rows = self.connection.execute("SELECT * FROM vehicle_counts").fetchall()
        result = []
        for camera_id, camera_name in cameras:
            counts = {r[1]: r for r in rows if r[0] == camera_id}
            classes = sorted(set(self.class_names) | set(counts))
            result.append({"camera_id": camera_id, "camera_name": camera_name,
                "revision": max((r[6] for r in counts.values()), default=0),
                "counting_since": min((r[4] for r in counts.values()), default=None),
                "updated_at": max((r[5] for r in counts.values()), default=None),
                "total": sum(r[3] for r in counts.values()),
                "counts": [{"class_id": label,
                            "class_name": self.class_names[label] if label in self.class_names else counts[label][2],
                            "count": counts[label][3] if label in counts else 0} for label in classes]})
        return {"type": "vehicle_counts", "scope": "all_time",
                "revision": max((c["revision"] for c in result), default=0), "cameras": result}

    def counts_push_digest(self, destination):
        with self.lock:
            row = self.connection.execute(
                "SELECT digest FROM counts_push_state WHERE destination=?", (destination,)).fetchone()
        return row[0] if row else None

    def acknowledge_counts(self, destination, digest):
        with self.lock, self.connection:
            self.connection.execute("INSERT OR REPLACE INTO counts_push_state VALUES (?, ?)",
                                    (destination, digest))

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
        self.reset_reasons = {}
        self.last_reset_reason = None
        self.gaps = 0
        self.last_gap = None
        self.state = "waiting_for_frame"
        self.pending_events = []
        self.clock_offset = time.time() - time.monotonic()

    def _publish(self, watermark):
        ready = sorted((event for event in self.pending_events if event["time"] < watermark),
                       key=lambda event: (event["time"], event["track_id"]))
        if not ready:
            return
        started = time.perf_counter()
        self.event_store.append(self.camera_id, [event["payload"] for event in ready])
        METRICS.add("db", time.perf_counter() - started, len(ready))
        self.pending_events = [event for event in self.pending_events if event["time"] >= watermark]

    def process(self, result, received_at, generation):
        with self.lock:
            return self._process(result, received_at, generation)

    def _process(self, result, received_at, generation):
        height, width = result.orig_shape
        if len(result.boxes) > 1:
            # End-to-end detections can contain the same vehicle under different class labels.
            boxes = torch.as_tensor(result.boxes.cpu().data, dtype=torch.float32)
            keep = nms(boxes[:, :4], boxes[:, -2], .8)
            result.update(boxes=boxes[keep])
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
        gap = None if self.last_frame is None else received_at - self.last_frame
        if gap is not None and gap > .5:
            self.gaps += 1
            self.last_gap = round(gap, 3)
        reset_reason = None
        if self.last_frame is None:
            reset_reason = "initial_frame"
        elif calibration != self.calibration:
            reset_reason = "calibration_changed"
        elif self.frame_size != [width, height]:
            reset_reason = "frame_size_changed"
        elif generation != self.generation:
            reset_reason = "stream_reconnected"
        elif gap <= 0:
            reset_reason = "clock_discontinuity"
        elif gap > 2:
            reset_reason = "frame_gap"
        # Short stalls keep identities and counted flags; LineCounter still rejects crossings across gaps > 0.5 s.
        if reset_reason:
            self._publish(math.inf)
            self.tracker.reset()
            self.counter = None
            if self.last_frame is not None:
                self.resets += 1
                self.last_reset_reason = reset_reason
                self.reset_reasons[reset_reason] = self.reset_reasons.get(reset_reason, 0) + 1
                print(f"[tracking-reset] {self.name} reason={reset_reason} gap_s={gap:.3f}", flush=True)
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
                    "tracking_resets": self.resets, "tracking_reset_reasons": dict(self.reset_reasons),
                    "last_reset_reason": self.last_reset_reason,
                    "tracking_gaps": self.gaps, "last_tracking_gap_s": self.last_gap,
                    "pending_events": len(self.pending_events)}


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
            elif url.path == "/api/traffic/counts":
                query = parse_qs(url.query, keep_blank_values=True, max_num_fields=1)
                if set(query) - {"camera_id"} or any(len(v) != 1 for v in query.values()):
                    raise ValueError("查询参数无效")
                camera_id = query.get("camera_id", [None])[0]
                cameras = [(c.camera_id, c.name) for c in self.traffic_cameras
                           if camera_id is None or c.camera_id == camera_id]
                if camera_id is not None and not cameras:
                    status, body = 404, {"error": "摄像头不存在"}
                else:
                    body = self.event_store.counts(cameras)
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
