import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from region_calibration import (
    CalibrationHandlerMixin, CalibrationStore, calibration_matches_frame,
    camera_key, validate_calibration,
)


VALID = {
    "roi": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
    "line": [[0.2, 0.5], [0.8, 0.5]],
    "direction": [[0.5, 0.3], [0.5, 0.7]],
    "frame_size": [1280, 720],
}


class GeometryTests(unittest.TestCase):
    def test_valid_roi_and_optional_line(self):
        self.assertEqual(validate_calibration(VALID), VALID)
        roi_only = dict(VALID, line=[], direction=[])
        self.assertEqual(validate_calibration(roi_only), roi_only)

    def test_rejects_invalid_geometry_and_coordinate_types(self):
        cases = [
            {"roi": [[0, 0], [1, 1]]},
            {"roi": [[0, 0], [1, 1], [0, 1], [1, 0]]},
            {"roi": [[0, 0], [0.5, 0.5], [1, 1]]},
            {"roi": [[0, 0], [1, 0], [1, 0], [0, 1]]},
            {"roi": [[False, 0], [1, 0], [0, 1]]},
            {"roi": [[float("nan"), 0], [1, 0], [0, 1]]},
            {"roi": [[float("inf"), 0], [1, 0], [0, 1]]},
            {"roi": [[-0.1, 0], [1, 0], [0, 1]]},
            {"line": [[0.2, 0.5], [0.2, 0.5]]},
            {"line": [[0, 0], [1, 1]]},
            {"line": [[0.2, 0.5]]},
            {"direction": []},
            {"direction": [[0.3, 0.3], [0.7, 0.3]]},
            {"frame_size": [0, 720]},
            {"frame_size": [True, 720]},
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_calibration(dict(VALID, **changes))

    def test_aspect_ratio_changes_disable_overlay_but_scaling_does_not(self):
        self.assertTrue(calibration_matches_frame(VALID, [1920, 1080]))
        self.assertFalse(calibration_matches_frame(VALID, [1280, 960]))
        self.assertFalse(calibration_matches_frame(None, [1280, 720]))

    def test_concave_roi_rejects_line_that_leaves_the_polygon(self):
        data = dict(VALID, roi=[[0, 0], [1, 0], [1, 1], [0.6, 1], [0.6, 0.4],
                                [0.4, 0.4], [0.4, 1], [0, 1]],
                    line=[[0.2, 0.8], [0.8, 0.8]])
        with self.assertRaises(ValueError):
            validate_calibration(data)

    def test_clear_is_saved_as_empty_geometry(self):
        empty = dict(VALID, roi=[], line=[], direction=[])
        self.assertEqual(validate_calibration(empty), empty)

    def test_camera_keys_identify_source_and_crop_not_list_position(self):
        sources = ["rtsp://example/2763", "rtsp://example/2762"]
        keys = {source: camera_key(source, None) for source in sources}
        self.assertEqual(keys, {source: camera_key(source, None) for source in reversed(sources)})
        self.assertNotEqual(keys[sources[0]], keys[sources[1]])
        self.assertNotEqual(keys[sources[0]], camera_key(sources[0], [0, 0, 0.5, 1]))


class StoreTests(unittest.TestCase):
    def test_two_preview_instances_merge_saves_and_reload_external_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "regions.json"
            first, second = CalibrationStore(path), CalibrationStore(path)
            first.save("2763", VALID)
            second.save("2762", VALID)
            reloaded = CalibrationStore(path)
            self.assertEqual(reloaded.get("2763"), VALID)
            self.assertEqual(reloaded.get("2762"), VALID)
            self.assertEqual(first.get("2762"), VALID)

    def test_concurrent_processes_preserve_all_camera_saves(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "regions.json"
            store = CalibrationStore(path)
            program = ("import json,sys; from region_calibration import CalibrationStore; "
                       "store=CalibrationStore(sys.argv[1]); data=json.loads(sys.argv[2]); "
                       "print('ready',flush=True); sys.stdin.readline(); "
                       "[store.save('child-'+str(i),data) for i in range(10)]")
            child = subprocess.Popen([sys.executable, "-c", program, str(path), json.dumps(VALID)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                self.assertEqual(child.stdout.readline().strip(), b"ready")
                child.stdin.write(b"go\n")
                child.stdin.flush()
                for i in range(10):
                    store.save("parent-" + str(i), VALID)
                _, errors = child.communicate(timeout=15)
                self.assertEqual(child.returncode, 0, errors.decode(errors="replace"))
                reloaded = CalibrationStore(path)
                self.assertEqual(set(reloaded.data), {prefix + str(i) for prefix in ("parent-", "child-") for i in range(10)})
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()

    def test_roundtrip_isolates_cameras_and_returns_independent_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "regions.json"
            store = CalibrationStore(path)
            store.save("2763", VALID)
            second = dict(VALID, line=[], direction=[])
            store.save("2762", second)
            store.get("2763")["roi"].clear()
            reloaded = CalibrationStore(path)
            self.assertEqual(reloaded.get("2763"), VALID)
            self.assertEqual(reloaded.get("2762"), second)
            self.assertIsNone(reloaded.get("missing"))
            self.assertNotIn("rtsp://", path.read_text(encoding="utf-8"))

    def test_failed_replace_preserves_file_and_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "regions.json"
            store = CalibrationStore(path)
            store.save("2763", VALID)
            original = path.read_bytes()
            with patch("region_calibration.os.replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    store.save("2763", dict(VALID, line=[], direction=[]))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(store.get("2763"), VALID)

    def test_corrupt_file_is_reported_instead_of_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "regions.json"
            path.write_text("{bad", encoding="utf-8")
            with self.assertRaises(ValueError):
                CalibrationStore(path)
            self.assertEqual(path.read_text(encoding="utf-8"), "{bad")


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()

        class Handler(CalibrationHandlerMixin, BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if not self.calibration_get(self.path.split("?")[0]):
                    self.send_error(404)

        Handler.names = ["2763", "2762"]
        Handler.camera_keys = ["camera-2763", "camera-2762"]
        Handler.hubs = []
        Handler.calibration_store = CalibrationStore(Path(self.folder.name) / "regions.json")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.folder.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        result = response.status, response.read()
        connection.close()
        return result

    def test_save_reload_and_validation_preserve_other_camera(self):
        headers = {"Content-Type": "application/json"}
        status, _ = self.request("POST", "/api/calibration/0", json.dumps(VALID), headers)
        self.assertEqual(status, 200)
        status, body = self.request("GET", "/api/calibration/0")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["calibration"], VALID)
        _, body = self.request("GET", "/api/calibration/1")
        self.assertIsNone(json.loads(body)["calibration"])
        invalid = dict(VALID, roi=[[0, 0]])
        status, _ = self.request("POST", "/api/calibration/0", json.dumps(invalid), headers)
        self.assertEqual(status, 400)
        _, body = self.request("GET", "/api/calibration/0")
        self.assertEqual(json.loads(body)["calibration"], VALID)

    def test_write_rejects_cross_origin_and_non_json(self):
        status, _ = self.request("POST", "/api/calibration/0", json.dumps(VALID),
                                 {"Content-Type": "application/json", "Origin": "https://other.example"})
        self.assertEqual(status, 403)
        status, _ = self.request("POST", "/api/calibration/0", json.dumps(VALID),
                                 {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)

    def test_missing_camera_and_editor_assets(self):
        status, _ = self.request("GET", "/api/calibration/99")
        self.assertEqual(status, 404)
        for path in ("/calibrate/0", "/calibration.js"):
            status, body = self.request("GET", path)
            self.assertEqual(status, 200)
            self.assertGreater(len(body), 100)

    def test_invalid_json_and_oversized_body_do_not_create_calibration(self):
        headers = {"Content-Type": "application/json"}
        for body in ("{bad", " " * 32769):
            with self.subTest(size=len(body)):
                status, _ = self.request("POST", "/api/calibration/0", body, headers)
                self.assertEqual(status, 400)
        _, body = self.request("GET", "/api/calibration/0")
        self.assertIsNone(json.loads(body)["calibration"])


if __name__ == "__main__":
    unittest.main()
