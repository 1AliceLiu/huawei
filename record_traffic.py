"""Record a camera with receive-clock timestamps and audit its timing quality."""
import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import time


class ReceiveTimeline:
    time_base = Fraction(1, 90000)

    def __init__(self):
        self.first_ns = None
        self.last_ns = None
        self.last_pts = -1

    def timestamp(self, received_ns):
        if self.first_ns is None:
            self.first_ns = received_ns
        if self.last_ns is not None and received_ns < self.last_ns:
            raise ValueError("Receive clock moved backwards")
        self.last_ns = received_ns
        pts = ((received_ns - self.first_ns) * 90000 + 500000000) // 1000000000
        self.last_pts = max(self.last_pts + 1, pts)
        return self.last_pts


def assess_frame_times(times):
    times = list(times)
    if len(times) < 2 or any(not math.isfinite(value) for value in times):
        raise ValueError("Video needs at least two frames with finite timestamps")
    gaps = [b - a for a, b in zip(times, times[1:])]
    if any(gap <= 0 for gap in gaps):
        raise ValueError("Frame timestamps must increase strictly")
    duration = times[-1] - times[0]
    fps = (len(times) - 1) / duration
    large_gaps = sum(gap > .5 for gap in gaps)
    reasons = []
    if fps < 10:
        reasons.append("Effective frame rate is below 10 FPS")
    if large_gaps:
        reasons.append("Frame gaps exceed 0.5 seconds")
    return {"frames": len(times), "duration_s": round(duration, 6),
            "effective_fps": round(fps, 3), "max_gap_s": round(max(gaps), 6),
            "gaps_over_0_5s": large_gaps, "usable_for_counting": not reasons, "reasons": reasons}


def inspect_video_timing(path):
    import av
    with av.open(str(path)) as container:
        times = []
        for frame in container.decode(video=0):
            if frame.time is None:
                raise ValueError("Video contains a frame without a timestamp")
            times.append(float(frame.time))
    return assess_frame_times(times)


def check_counting_timing(path):
    path = Path(path)
    report = inspect_video_timing(path)
    sidecar = path.with_suffix(".timing.json")
    if sidecar.exists():
        capture = json.loads(sidecar.read_text(encoding="utf-8-sig"))
        if not isinstance(capture, dict) or any(
                name in capture and type(capture[name]) is not bool
                for name in ("capture_complete", "usable_for_counting")):
            raise ValueError("Invalid recorder timing report")
        if capture.get("capture_complete") is False or capture.get("usable_for_counting") is False:
            report["usable_for_counting"] = False
            report["reasons"].append("Recorder marked the capture incomplete or unsuitable")
    return report


def record_clip(source_url, output_path, seconds=30):
    import av
    output_path = Path(output_path)
    partial = output_path.with_name(output_path.stem + ".partial.mp4")
    frame_log = output_path.with_suffix(".frames.csv")
    timing_log = output_path.with_suffix(".timing.json")
    if any(path.exists() for path in (output_path, partial, frame_log, timing_log)):
        raise FileExistsError("Use a new output filename; existing recordings are preserved")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial.touch(exist_ok=False)
    clock = ReceiveTimeline()
    rows = []
    encode_times = []
    previous_source = None
    skipped = 0
    with av.open(source_url, options={"rtsp_transport": "tcp"}, timeout=(10, 10)) as source:
        stream = source.streams.video[0]
        print(f"Recording {stream.width}x{stream.height}; receive-clock PTS", flush=True)
        with av.open(str(partial), "w", options={"movflags": "+faststart"}) as output:
            # Receive-clock PTS determines VFR timing; 25 is only an encoder rate hint.
            target = output.add_stream("libx264", rate=25)
            target.width, target.height = stream.width, stream.height
            target.pix_fmt = "yuv420p"
            target.options = {"preset": "ultrafast", "crf": "23", "tune": "zerolatency"}
            target.codec_context.max_b_frames = 0
            target.codec_context.thread_count = 2
            target.codec_context.time_base = clock.time_base
            target.time_base = clock.time_base
            with frame_log.open("x", newline="", encoding="utf-8") as file:
                columns = ["frame", "receive_s", "source_s", "source_gap_s", "output_pts"]
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                for decoded in source.decode(stream):
                    # Allow the initial cached/key frames to settle before starting the clip.
                    if skipped < 5:
                        skipped += 1
                        continue
                    received = time.perf_counter_ns()
                    pts = clock.timestamp(received)
                    receive_s = (received - clock.first_ns) / 1e9
                    source_s = float(decoded.time) if decoded.time is not None else None
                    source_gap = source_s - previous_source if source_s is not None and previous_source is not None else None
                    row = {"frame": len(rows), "receive_s": receive_s, "source_s": source_s,
                           "source_gap_s": source_gap, "output_pts": pts}
                    rows.append(row)
                    writer.writerow(row)
                    previous_source = source_s
                    if decoded.width != target.width or decoded.height != target.height:
                        raise ValueError("Camera resolution changed during recording")
                    frame = av.VideoFrame.from_ndarray(decoded.to_ndarray(format="bgr24"), format="bgr24")
                    frame.pts, frame.time_base = pts, clock.time_base
                    before = time.perf_counter()
                    for packet in target.encode(frame):
                        output.mux(packet)
                    encode_times.append(time.perf_counter() - before)
                    if len(rows) % 100 == 0:
                        file.flush()
                        print(f"RECORD frames={len(rows)} elapsed={receive_s:.2f}s", flush=True)
                    if receive_s >= seconds:
                        break
                for packet in target.encode():
                    output.mux(packet)
    if len(rows) < 2:
        raise ValueError("Camera did not deliver enough frames")
    partial.rename(output_path)
    quality = inspect_video_timing(output_path)
    complete = rows[-1]["receive_s"] >= seconds
    if not complete:
        quality["usable_for_counting"] = False
        quality["reasons"].append("Recording ended before the requested duration")
    source_gaps = [row["source_gap_s"] for row in rows if row["source_gap_s"] is not None]
    report = {**quality, "clock": "perf_counter_ns / decoded-frame delivery", "startup_frames_skipped": skipped,
              "requested_duration_s": seconds, "capture_complete": complete,
              "received_frames": len(rows), "source_backward_steps": sum(gap < 0 for gap in source_gaps),
              "source_forward_jumps_over_0_5s": sum(gap > .5 for gap in source_gaps),
              "mean_encode_seconds": round(sum(encode_times) / len(encode_times), 6),
              "max_encode_seconds": round(max(encode_times), 6)}
    if report["frames"] != len(rows):
        raise ValueError("Encoded frame count differs from received frame count")
    with timing_log.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    root = Path(__file__).resolve().parent
    sources = [line.strip() for line in (root / "huawei-rtsp.txt").read_text(encoding="utf-8-sig").splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    if not 0 <= args.camera_index < len(sources):
        parser.error("Camera index is outside the source list")
    record_clip(sources[args.camera_index], args.output, args.seconds)


if __name__ == "__main__":
    main()
