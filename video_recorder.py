"""Write annotated sampled frames to rotating MP4 segments."""
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import re
import time

import av


class VideoRecorder:
    def __init__(self, root, camera, fps=10, segment_seconds=600, keep_segments=6):
        safe = re.sub(r"[^\w.-]+", "_", str(camera)).strip("._") or "camera"
        self.folder = Path(root) / safe
        self.folder.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.time_base = Fraction(1, 1) / Fraction(self.fps).limit_denominator(1000)
        self.segment_seconds = float(segment_seconds)
        self.keep_segments = max(1, int(keep_segments))
        self.container = None
        self.stream = None
        self.path = None
        self.frames = 0
        self.segment_started = 0.0
        self._prune()

    def _open(self, width, height):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.folder / f"{stamp}.mp4"
        suffix = 1
        while path.exists():
            suffix += 1
            path = self.folder / f"{stamp}_{suffix}.mp4"
        self.container = av.open(str(path), mode="w", format="mp4",
                                 options={"movflags": "+faststart"})
        keyint = str(max(1, int(self.fps * 2)))
        try:
            stream = self.container.add_stream("h264_nvenc", rate=Fraction(self.fps).limit_denominator(1000))
            stream.options = {"preset": "p4", "cq": "23", "g": keyint}
        except Exception:
            stream = self.container.add_stream("libx264", rate=Fraction(self.fps).limit_denominator(1000))
            stream.options = {"preset": "veryfast", "crf": "26", "g": keyint}
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        self.stream = stream
        self.path = path
        self.frames = 0
        self.segment_started = time.monotonic()
        self._prune()

    def write(self, frame_bgr):
        height, width = frame_bgr.shape[:2]
        if self.container is None or time.monotonic() - self.segment_started >= self.segment_seconds:
            self.close()
            self._open(width, height)
        video = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        video.pts = self.frames
        video.time_base = self.time_base
        for packet in self.stream.encode(video):
            self.container.mux(packet)
        self.frames += 1

    def close(self):
        if self.container is None:
            return
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()
            self.container = None
            self.stream = None
            self.path = None
            self._prune()

    def _prune(self):
        files = sorted(self.folder.glob("*.mp4"))
        for old in files[:-self.keep_segments]:
            try:
                old.unlink()
            except OSError:
                pass
