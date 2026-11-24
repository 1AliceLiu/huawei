"""Replay one recorded camera clip using its saved ROI and directed count line."""
import argparse
from collections import Counter, defaultdict, deque
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import time

from region_calibration import (
    CalibrationStore, _cross, _inside, calibration_matches_frame,
    camera_key, validate_calibration,
)
from record_traffic import check_counting_timing


class ClassVote:
    """One strongest observation per 200 ms, aggregated without retaining frames."""

    def __init__(self):
        self.scores = Counter()
        self.bin = None
        self.best = None

    def add(self, label, confidence, quality, time_s):
        weight = confidence ** 2 * quality
        if weight <= 0:
            return
        bucket = math.floor(time_s / .2)
        if bucket != self.bin:
            self.bin, self.best = bucket, None
        if self.best is None or weight > self.best[1]:
            if self.best is not None:
                self.scores[self.best[0]] -= self.best[1]
            self.scores[label] += weight
            self.best = (label, weight)

    def decision(self):
        return {"label": self.scores.most_common(1)[0][0] if self.scores else None}


class LineCounter:
    def __init__(self, calibration, keep_history=True, min_roi_observations=1):
        self.calibration = validate_calibration(calibration)
        if not calibration["line"] or not calibration["direction"]:
            raise ValueError("需要已保存的 ROI、计数线和通行方向")
        self.roi = calibration["roi"]
        self.a, self.b = calibration["line"]
        self.width, self.height = calibration["frame_size"]
        a, b = self._pixels(self.a), self._pixels(self.b)
        self.line_length = math.dist(a, b)
        start, end = calibration["direction"]
        self.forward = 1 if _cross([0, 0], [self.b[0] - self.a[0], self.b[1] - self.a[1]],
                                   [end[0] - start[0], end[1] - start[1]]) > 0 else -1
        self.tracks = {}
        self.events = [] if keep_history else deque(maxlen=0)
        self._total = 0
        self.min_roi_observations = min_roi_observations
        self.tracking_epoch = 0

    @property
    def total(self):
        return self._total

    def _pixels(self, point):
        return [point[0] * (self.width - 1), point[1] * (self.height - 1)]

    def _distance(self, point):
        return _cross(self._pixels(self.a), self._pixels(self.b), self._pixels(point)) / self.line_length

    def new_segment(self):
        self.tracking_epoch += 1
        self.tracks.clear()

    def expire(self, time_s):
        self.tracks = {key: state for key, state in self.tracks.items()
                       if time_s - state["last_seen"] <= 30}
        for state in self.tracks.values():
            if time_s - state["last_seen"] > .5:
                state.update(pending=None, side=0)

    def observe(self, track_id, point, time_s, label, confidence=1.0, quality=1.0):
        distance = self._distance(point)
        side = 1 if distance > 3 else -1 if distance < -3 else 0
        state = self.tracks.get(track_id)
        if state is None or not 0 <= time_s - state["last_seen"] <= .5:
            counted = bool(state and state["counted"])
            state = {"last_seen": time_s, "side": 0, "point": list(point), "time": time_s,
                     "pending": None, "votes": ClassVote(), "counted": counted, "roi_observations": 0}
            self.tracks[track_id] = state
        state["last_seen"] = time_s
        if _inside(point, self.roi) and distance * self.forward <= 0:
            state["roi_observations"] += 1
            state["votes"].add(label, confidence, quality, time_s)
        elif distance * self.forward < 0:
            state["roi_observations"] = 0
            state["votes"] = ClassVote()
        event = None
        previous = state["point"]
        previous_distance = self._distance(previous)
        # Geometry uses consecutive observations; the deadband only confirms the side change.
        if previous_distance * distance <= 0 and previous_distance != distance:
            state["pending"] = None
            ratio = previous_distance / (previous_distance - distance)
            crossing = [previous[i] + ratio * (point[i] - previous[i]) for i in (0, 1)]
            vx, vy = self.b[0] - self.a[0], self.b[1] - self.a[1]
            position = ((crossing[0] - self.a[0]) * vx + (crossing[1] - self.a[1]) * vy) / (vx * vx + vy * vy)
            start, end = self.calibration["direction"]
            movement = self._pixels([point[i] - previous[i] for i in (0, 1)])
            direction = self._pixels([end[i] - start[i] for i in (0, 1)])
            if ((distance - previous_distance) * self.forward > 0
                    and sum(a * b for a, b in zip(movement, direction)) > 0 and _inside(previous, self.roi)
                    and state["roi_observations"] >= self.min_roi_observations
                    and 0 <= position <= 1 and _inside(crossing, self.roi)):
                state["pending"] = {"crossing_time_s": round(state["time"] + ratio * (time_s - state["time"]), 3),
                                    "crossing_x": round(crossing[0], 6), "crossing_y": round(crossing[1], 6),
                                    **state["votes"].decision()}
        if side:
            if (not state["counted"] and side == self.forward
                    and state["side"] == -self.forward and state["pending"] is not None):
                state["counted"] = True
                event = {"number": self.total + 1, "track_id": track_id,
                         "tracking_epoch": self.tracking_epoch,
                         "video_time_s": round(time_s, 3), **state["pending"]}
                self._total += 1
                self.events.append(event)
            state.update(side=side, pending=None)
        state.update(point=list(point), time=time_s)
        return event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--camera-index", type=int, default=8)
    parser.add_argument("--camera-name", default="2763")
    parser.add_argument("--model", type=Path,
                        default=Path("runs/detect/yolo26m_merged12_best_20261021_0957/weights/best.pt"))
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    import av
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from realtime_web import draw_calibration

    root = Path(__file__).resolve().parent
    sources = [line.strip() for line in (root / "huawei-rtsp.txt").read_text(encoding="utf-8-sig").splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    if not 0 <= args.camera_index < len(sources):
        raise ValueError("摄像头序号不在源列表内")
    calibration = CalibrationStore(root / "camera_regions.json").get(camera_key(sources[args.camera_index], None))
    if calibration is None:
        raise ValueError("该摄像头尚未保存标定")
    counter = LineCounter(calibration)
    folder = args.video.resolve().parent
    final_video = folder / "counted.mp4"
    partial_video = folder / "counted.partial.mp4"
    artifacts = [final_video, partial_video] + [folder / name for name in
        ("events.csv", "summary.json", "calibration_used.json", "timing_quality.json")]
    if any(path.exists() for path in artifacts):
        raise FileExistsError("计数结果或诊断文件已存在，请使用新的试验目录")
    timing = check_counting_timing(args.video)
    with (folder / "timing_quality.json").open("x", encoding="utf-8") as file:
        json.dump(timing, file, indent=2)
    if not timing["usable_for_counting"]:
        raise ValueError("视频帧率或连续性不满足试算要求，请先检查 timing_quality.json 并重新录制")
    (folder / "calibration_used.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    model = YOLO(str(args.model), task="detect")
    use_half = args.device != "cpu" and torch.cuda.is_available()
    precision = {"quantize": 16} if use_half else {}
    common = dict(imgsz=960, conf=.25, iou=.7, device=args.device, verbose=False)
    try:
        model.predict(np.zeros((960, 960, 3), dtype=np.uint8), **common, **precision)
    except TypeError:
        precision = {"half": True} if use_half else {}
        model.predict(np.zeros((960, 960, 3), dtype=np.uint8), **common, **precision)

    trails = defaultdict(lambda: deque(maxlen=20))
    counted = {}
    started = time.monotonic()
    frames = 0
    last_time = -1.0
    first_pts = None
    with av.open(str(args.video)) as source, av.open(str(partial_video), "w", options={"movflags": "+faststart"}) as output:
        source_stream = source.streams.video[0]
        rate = source_stream.average_rate or Fraction(25, 1)
        width, height = source_stream.width, source_stream.height
        if not calibration_matches_frame(calibration, [width, height]):
            raise ValueError("视频宽高比与标定不一致")
        output_stream = output.add_stream("libx264", rate=rate)
        output_stream.width, output_stream.height = width, height
        output_stream.pix_fmt = "yuv420p"
        output_stream.options = {"crf": "20", "preset": "veryfast"}
        output_stream.time_base = source_stream.time_base
        output_stream.codec_context.time_base = source_stream.time_base
        with (folder / "events.csv").open("w", newline="", encoding="utf-8-sig") as event_file:
            columns = ["number", "track_id", "tracking_epoch", "label", "video_time_s", "crossing_time_s", "crossing_x", "crossing_y", "frame_index"]
            writer = csv.DictWriter(event_file, fieldnames=columns)
            writer.writeheader()
            for frame_index, decoded in enumerate(source.decode(source_stream)):
                if decoded.pts is None:
                    raise ValueError("视频缺少帧时间戳")
                if first_pts is None:
                    first_pts = decoded.pts
                elapsed = float((decoded.pts - first_pts) * decoded.time_base)
                if elapsed < last_time:
                    raise ValueError("视频时间戳倒退，停止以避免产生错误计数")
                if last_time >= 0 and elapsed - last_time > .5:
                    for tracker in getattr(model.predictor, "trackers", []):
                        tracker.reset()
                    counter.new_segment()
                    trails.clear()
                last_time = elapsed
                frame = decoded.to_ndarray(format="bgr24")
                result = model.track(frame, persist=True, tracker="bytetrack.yaml", **common, **precision)[0]
                annotated = frame.copy()
                draw_calibration(annotated, calibration)
                if result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.int().cpu().tolist()
                    classes = result.boxes.cls.int().cpu().tolist()
                    confidences = result.boxes.conf.cpu().tolist()
                    for box, track_id, class_id, confidence in zip(boxes, ids, classes, confidences):
                        identity = (counter.tracking_epoch, track_id)
                        x1, y1, x2, y2 = box.tolist()
                        point = [min(1, max(0, (x1 + x2) / 2 / (width - 1))),
                                 min(1, max(0, y2 / (height - 1)))]
                        label = model.names[class_id]
                        event = counter.observe(track_id, point, elapsed, label, confidence)
                        if event:
                            counted[identity] = dict(event)
                            writer.writerow(dict(event, frame_index=frame_index))
                            event_file.flush()
                            print(f"COUNT {event['number']} ID={track_id} {event['label']} at {elapsed:.2f}s", flush=True)
                        inside = _inside(point, calibration["roi"])
                        color = (80, 220, 80) if inside else (140, 140, 140)
                        if identity in counted:
                            color = (0, 210, 255)
                        anchor = (round(point[0] * (width - 1)), round(point[1] * (height - 1)))
                        trails[track_id].append(anchor)
                        cv2.rectangle(annotated, (round(x1), round(y1)), (round(x2), round(y2)), color, 2 if inside else 1)
                        cv2.circle(annotated, anchor, 5, color, -1)
                        tag = f"ID {counter.tracking_epoch}:{track_id} {label}"
                        if identity in counted:
                            tag += f" COUNT #{counted[identity]['number']}"
                        cv2.putText(annotated, tag, (max(0, round(x1)), max(20, round(y1) - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
                        if inside and len(trails[track_id]) > 1:
                            cv2.polylines(annotated, [np.asarray(trails[track_id], dtype=np.int32)], False, color, 2, cv2.LINE_AA)
                cv2.rectangle(annotated, (0, 0), (850, 105), (20, 20, 20), -1)
                cv2.putText(annotated, f"CAM {args.camera_name} L2 | COUNT {counter.total} | VIDEO {elapsed:06.2f}s",
                            (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated, "Green: ROI | Grey: excluded | Yellow: counted | Dot: bottom-center",
                            (18, 80), cv2.FONT_HERSHEY_SIMPLEX, .65, (220, 220, 220), 2, cv2.LINE_AA)
                encoded_frame = av.VideoFrame.from_ndarray(annotated, format="bgr24")
                encoded_frame.pts = decoded.pts - first_pts
                encoded_frame.time_base = decoded.time_base
                for packet in output_stream.encode(encoded_frame):
                    output.mux(packet)
                frames += 1
                if frames % 250 == 0:
                    print(f"PROGRESS frames={frames} video={elapsed:.1f}s count={counter.total} wall={time.monotonic()-started:.1f}s", flush=True)
            for packet in output_stream.encode():
                output.mux(packet)
        summary = {"camera": args.camera_name, "source_label": sources[args.camera_index].rsplit("/", 1)[-1],
                   "frames": frames, "duration_s": round(last_time + 1 / float(rate), 3),
                   "total": counter.total, "by_class": dict(Counter(event["label"] for event in counter.events)),
                   "model": str(args.model), "imgsz": 960, "confidence": .25, "tracker": "bytetrack.yaml",
                   "anchor": "bottom_center", "line_deadband_pixels": 3, "max_tracking_gap_s": .5,
                   "tracking_resets": counter.tracking_epoch,
                   "processing_seconds": round(time.monotonic() - started, 2)}
        summary["timing_quality"] = timing
    if not frames:
        raise ValueError("视频中没有可解码帧")
    partial_video.rename(final_video)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
