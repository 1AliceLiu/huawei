"""Loopback HTTP receiver for verifying cumulative-count delivery."""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path


def validate_snapshot(value):
    if (not isinstance(value, dict) or value.get("type") != "vehicle_counts"
            or value.get("scope") != "all_time" or type(value.get("revision")) is not int
            or value["revision"] < 0 or not isinstance(value.get("cameras"), list)):
        raise ValueError("Invalid snapshot")
    cameras = set()
    revisions = []
    for camera in value["cameras"]:
        if (not isinstance(camera, dict) or not isinstance(camera.get("camera_id"), str)
                or not camera["camera_id"] or camera["camera_id"] in cameras
                or not isinstance(camera.get("camera_name"), str)
                or type(camera.get("total")) is not int or camera["total"] < 0
                or type(camera.get("revision")) is not int or camera["revision"] < 0
                or not isinstance(camera.get("counts"), list)):
            raise ValueError("Invalid camera")
        cameras.add(camera["camera_id"])
        revisions.append(camera["revision"])
        labels, total = set(), 0
        for item in camera["counts"]:
            if (not isinstance(item, dict) or type(item.get("class_id")) is not int
                    or item["class_id"] < 0 or item["class_id"] in labels
                    or not isinstance(item.get("class_name"), str)
                    or type(item.get("count")) is not int or item["count"] < 0):
                raise ValueError("Invalid class count")
            labels.add(item["class_id"])
            total += item["count"]
        if total != camera["total"]:
            raise ValueError("Total does not match class counts")
    if max(revisions, default=0) != value["revision"]:
        raise ValueError("Invalid revision")


def create_receiver(folder, port=18106, host="127.0.0.1"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    latest_path = folder / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8")) if latest_path.exists() else None
    if latest is not None:
        validate_snapshot(latest)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            nonlocal latest
            if self.path != "/api/traffic/counts":
                self.send_error(404)
                return
            try:
                self.connection.settimeout(5)
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_000_000 or self.headers.get_content_type() != "application/json":
                    raise ValueError("Invalid request")
                body = self.rfile.read(size)
                key = "counts:" + hashlib.sha256(body).hexdigest()
                if self.headers.get("Idempotency-Key") != key:
                    raise ValueError("Invalid idempotency key")
                snapshot = json.loads(body)
                validate_snapshot(snapshot)
                duplicate = snapshot == latest
                stale = latest is not None and snapshot["revision"] < latest["revision"]
                if latest is not None and snapshot["revision"] == latest["revision"] and not duplicate:
                    self.send_error(409)
                    return
                if not duplicate and not stale:
                    temporary = folder / "latest.tmp"
                    with temporary.open("w", encoding="utf-8") as file:
                        json.dump(snapshot, file, ensure_ascii=False, indent=2)
                        file.flush()
                        os.fsync(file.fileno())
                    os.replace(temporary, latest_path)
                    latest = snapshot
                status = 200 if duplicate or stale else 201
                receipt = {"received_at": datetime.now(timezone.utc).isoformat(), "http_status": status,
                           "idempotency_key": key, "duplicate": duplicate, "stale": stale, "snapshot": snapshot}
                with (folder / "received.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(receipt, ensure_ascii=False) + "\n")
                    file.flush()
                    os.fsync(file.fileno())
                response = json.dumps({"accepted": True, "revision": snapshot["revision"],
                                       "duplicate": duplicate, "stale": stale}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
            except (ValueError, TypeError, KeyError):
                self.send_error(400)
            except OSError:
                self.send_error(503)

    return HTTPServer((host, port), Handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--port", type=int, default=18106)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    with create_receiver(args.folder, args.port, args.host) as server:
        print(f"COUNT_RECEIVER_READY port={server.server_port}", flush=True)
        server.serve_forever()
