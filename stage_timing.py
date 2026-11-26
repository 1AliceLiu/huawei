"""Lightweight per-stage timing counters for the live worker."""
import threading
import time


class StageMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {}

    def add(self, name, seconds, units=1):
        with self.lock:
            item = self.data.setdefault(name, [0.0, 0, 0, 0.0])
            item[0] += seconds
            item[1] += units
            item[2] += 1
            item[3] = max(item[3], seconds)

    def bump(self, name, n=1):
        with self.lock:
            item = self.data.setdefault(name, [0.0, 0, 0, 0.0])
            item[1] += n
            item[2] += n

    def report(self):
        with self.lock:
            data, self.data = self.data, {}
        parts = []
        for name in ("ndarray", "infer", "track", "annotate", "preview", "record", "preview_skip", "db"):
            if name not in data:
                continue
            seconds, units, calls, _ = data[name]
            if name == "infer":
                parts.append(f"infer={seconds / calls * 1000:.1f}ms/batch "
                             f"({seconds / units * 1000:.2f}ms/img, {calls} batch/{units} img)")
            elif name == "preview_skip":
                parts.append(f"preview_skip={calls}")
            elif name == "db":
                parts.append(f"db={seconds / units * 1000:.2f}ms/event "
                             f"({units} events/{calls} writes)")
            else:
                unit = "cam" if name == "track" else "frame"
                parts.append(f"{name}={seconds / calls * 1000:.2f}ms/{unit} ({calls})")
        if parts:
            print("[timing] " + " | ".join(parts), flush=True)


METRICS = StageMetrics()
