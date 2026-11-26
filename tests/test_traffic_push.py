import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from traffic_api import EventStore
from traffic_push import CountsPublisher
from scripts.receive_traffic_counts import create_receiver


class CountsPushTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "events.sqlite3"
        self.store = EventStore(self.path, {0: "Car", 1: "Truck"})
        self.addCleanup(self.store.close)
        self.requests = []
        self.status = 201
        test = self

        class Receiver(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                test.requests.append((self.path, dict(self.headers), body))
                self.send_response(test.status)
                if test.status == 307:
                    self.send_header("Location", "/unexpected-redirect")
                self.send_header("Content-Length", "0")
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/counts"
        self.publisher = CountsPublisher(self.store, [("a", "cam-a"), ("b", "cam-b")],
                                         self.url, threading.Event(), "test-token")

    def append(self, label=0):
        self.store.append("a", [{"class_id": label, "class_name": ("Car", "Truck")[label],
                                 "crossed_at": "2026-11-24T08:00:00.000+00:00"}])

    def test_real_http_retry_restart_and_latest_snapshot_do_not_double_count(self):
        self.append()
        self.status = 503
        with self.assertRaises(HTTPError):
            self.publisher.publish_once()
        self.assertIsNone(self.store.counts_push_digest(self.publisher.destination))
        self.status = 201
        self.assertTrue(self.publisher.publish_once())
        self.assertEqual(self.requests[0], self.requests[1])
        path, headers, body = self.requests[-1]
        self.assertEqual(headers["Idempotency-Key"], "counts:" + hashlib.sha256(body).hexdigest())
        self.assertEqual(headers["Authorization"], "Bearer test-token")
        self.assertFalse(self.publisher.publish_once())
        reopened = EventStore(self.path, self.store.class_names)
        try:
            publisher = CountsPublisher(reopened, self.publisher.cameras, self.url, threading.Event())
            self.assertFalse(publisher.publish_once())
            self.append(1)
            self.append(1)
            self.assertTrue(publisher.publish_once())
            latest = json.loads(self.requests[-1][2])
            self.assertEqual([v["count"] for v in latest["cameras"][0]["counts"]], [1, 2])
            self.assertEqual(latest["cameras"][0]["total"], 3)
            self.assertEqual(latest["cameras"][1]["total"], 0)
        finally:
            reopened.close()

    def test_redirect_is_not_followed_and_not_acknowledged(self):
        self.status = 307
        with self.assertRaises(HTTPError):
            self.publisher.publish_once()
        self.assertEqual(len(self.requests), 1)
        self.assertIsNone(self.store.counts_push_digest(self.publisher.destination))

    def test_invalid_webhook_config_is_rejected(self):
        for url in ("file:///tmp/out", "http://user:secret@example.com/", "https://example.com/#fragment",
                    "http://example.com:70000", "http://example.com:0"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                CountsPublisher(self.store, [], url, threading.Event())

    def test_receiver_persists_snapshots_rejects_invalid_data_and_ignores_stale_retry(self):
        folder = Path(self.folder.name) / "receiver"
        server = create_receiver(folder, 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        client = build_opener(ProxyHandler({}))

        def post(snapshot, key=None):
            body = json.dumps(snapshot).encode()
            request = Request(f"http://127.0.0.1:{server.server_port}/api/traffic/counts", data=body,
                              headers={"Content-Type": "application/json",
                                       "Idempotency-Key": key or "counts:" + hashlib.sha256(body).hexdigest()})
            with client.open(request, timeout=2) as response:
                return response.status

        try:
            self.append()
            old = self.store.counts(self.publisher.cameras)
            self.assertEqual(post(old), 201)
            self.assertEqual(post(old), 200)
            self.append(1)
            new = self.store.counts(self.publisher.cameras)
            self.assertEqual(post(new), 201)
            self.assertEqual(post(old), 200)
            self.assertEqual(json.loads((folder / "latest.json").read_text()), new)
            invalid = json.loads(json.dumps(new))
            invalid["cameras"][0]["total"] += 1
            with self.assertRaises(HTTPError) as failure:
                post(invalid)
            self.assertEqual(failure.exception.code, 400)
            with self.assertRaises(HTTPError) as failure:
                post(new, "counts:wrong")
            self.assertEqual(failure.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
        server = create_receiver(folder, 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            self.assertEqual(post(new), 200)
        finally:
            server.shutdown()
            server.server_close()
