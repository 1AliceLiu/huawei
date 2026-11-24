import http.client
import queue
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

import cv2
import numpy as np

from realtime_web import AVStreamReader, FrameHub, MJPEGHandler, cam_name, draw_calibration, grid_page, serve
import realtime_web
from region_calibration import CalibrationStore


class PreviewCalibrationTests(unittest.TestCase):
    def test_vehicle_boxes_and_labels_stay_bold_after_preview_downscaling(self):
        from ultralytics.engine.results import Results
        for width in (960, 1920):
            with self.subTest(source_width=width), tempfile.TemporaryDirectory() as folder:
                frame = np.zeros((width * 9 // 16, width, 3), dtype=np.uint8)
                scale = width / 960
                boxes = np.array([[100 * scale, 150 * scale, 300 * scale, 400 * scale, .92, 0]],
                                 dtype=np.float32)
                result = Results(frame, path='fixture.jpg', names={0: 'Private Car'}, boxes=boxes.copy())
                hub = FrameHub()
                pending = queue.Queue()
                pending.put((None, result, 'FPS 8.0'))
                pending.put(None)
                state = {'q': pending, 'hub': hub, 'camera_key': 'fixture',
                         'calibration_store': CalibrationStore(Path(folder) / 'regions.json')}
                with patch.object(realtime_web, 'STOP', threading.Event()):
                    realtime_web._encode_worker(state)
                preview = cv2.imdecode(np.frombuffer(hub.get(), dtype=np.uint8), cv2.IMREAD_COLOR)
                edge_columns = preview[200:230, 99:102].max(axis=2).mean(axis=0)
                self.assertTrue(np.all(edge_columns > 100), edge_columns)
                self.assertGreater(preview[130:135, 110:180].max(axis=2).mean(), 80)
                self.assertEqual(int(frame.sum()), 0)
                np.testing.assert_array_equal(result.boxes.data, boxes)

    def test_preview_limits_bandwidth_without_resizing_calibration_frame(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :960] = (0, 0, 255)
        hub = FrameHub()
        hub.set_raw(frame)
        hub.set(frame)
        preview = cv2.imdecode(np.frombuffer(hub.get(), dtype=np.uint8), cv2.IMREAD_COLOR)
        raw = cv2.imdecode(np.frombuffer(hub.get_raw(), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(preview.shape, (540, 960, 3))
        self.assertEqual(raw.shape, (1080, 1920, 3))
        self.assertGreater(int(preview[270, 240, 2]), 240)
        self.assertLess(int(preview[270, 720, 2]), 10)
        small = np.zeros((180, 320, 3), dtype=np.uint8)
        hub.set(small)
        self.assertEqual(cv2.imdecode(np.frombuffer(hub.get(), dtype=np.uint8), cv2.IMREAD_COLOR).shape,
                         small.shape)

    def test_busy_preview_keeps_newest_frame_and_accepts_shutdown(self):
        pending = queue.Queue(maxsize=1)
        realtime_web.put_latest(pending, 'old')
        realtime_web.put_latest(pending, 'new')
        self.assertEqual(pending.get_nowait(), 'new')
        realtime_web.put_latest(pending, 'pending')
        realtime_web.put_latest(pending, None)
        self.assertIsNone(pending.get_nowait())

    def test_rtsp_connection_and_reads_have_timeouts(self):
        reader = AVStreamReader.__new__(AVStreamReader)
        reader.url = 'rtsp://camera.test/video'
        reader.container = None
        reader.hwaccel = None
        with patch('realtime_web.av.open') as open_stream:
            self.assertTrue(reader._open())
        self.assertEqual(open_stream.call_args.kwargs.get('timeout'), (8.0, 8.0))

    def test_rtsp_reader_passes_gpu_decoder_to_open_and_reconnect(self):
        from av.codec.hwaccel import HWAccel
        accelerator = HWAccel('cuda', device=0)
        with patch('realtime_web.threading.Thread.start'):
            reader = AVStreamReader('rtsp://camera.test/video', hwaccel=accelerator)
        with patch('realtime_web.av.open') as open_stream:
            self.assertTrue(reader._open())
            self.assertTrue(reader._open())
        self.assertTrue(all(call.kwargs.get('hwaccel') is accelerator
                            for call in open_stream.call_args_list))

    def test_video_streams_leave_control_port_free_for_calibration_requests(self):
        class Images(HTMLParser):
            def __init__(self):
                super().__init__()
                self.sources = []

            def handle_starttag(self, tag, attrs):
                if tag == "img":
                    self.sources.append(dict(attrs)["src"])

        for count in (1, 10, 18):
            with self.subTest(cameras=count):
                page = Images()
                page.feed(grid_page(count, "127.0.0.1:8000"))
                ports = Counter(urlsplit(source).port for source in page.sources)
                self.assertEqual(len(page.sources), count)
                self.assertNotIn(8000, ports)
                self.assertLessEqual(max(ports.values()), 6)
                self.assertEqual(urlsplit(page.sources[-1]).port, 8001 + (count - 1) // 6)

    def test_camera_labels_do_not_include_credentials_or_query_tokens(self):
        self.assertEqual(cam_name("rtsp://user:password@example/2763?token=secret"), "2763")
        self.assertEqual(cam_name("rtsp://user:password@example/"), "example")

    def test_scaled_overlay_does_not_change_raw_snapshot(self):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        hub = FrameHub()
        self.assertIsNone(hub.get_raw())
        hub.set_raw(frame)
        calibration = {"roi": [[.1, .1], [.9, .1], [.9, .9], [.1, .9]],
                       "line": [[.2, .5], [.8, .5]], "direction": [[.5, .3], [.5, .7]],
                       "frame_size": [1280, 720]}
        annotated = frame.copy()
        draw_calibration(annotated, calibration)
        self.assertGreater(int(annotated.sum()), 0)
        self.assertEqual(int(frame.sum()), 0)
        decoded = cv2.imdecode(np.frombuffer(hub.get_raw(), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, frame.shape)
        self.assertEqual(int(decoded.sum()), 0)
        wrong_ratio = np.zeros((240, 320, 3), dtype=np.uint8)
        draw_calibration(wrong_ratio, calibration)
        self.assertEqual(int(wrong_ratio.sum()), 0)

    def test_real_preview_handler_serves_links_clean_frame_and_saved_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            attributes = {name: getattr(MJPEGHandler, name, None) for name in
                          ("hubs", "names", "camera_keys", "calibration_store")}
            server = None
            try:
                hub = FrameHub()
                frame = np.zeros((180, 320, 3), dtype=np.uint8)
                hub.set_raw(frame)
                hub.set(frame)
                MJPEGHandler.hubs = [hub]
                MJPEGHandler.names = ["2763"]
                MJPEGHandler.camera_keys = ["test-camera"]
                MJPEGHandler.calibration_store = CalibrationStore(Path(folder) / "regions.json")
                server = serve("127.0.0.1", 0)
                for path, expected in (("/", b"/calibrate/0"), ("/calibration-frame/0", b"\xff\xd8"),
                                       ("/api/calibration/0", b'"2763"')):
                    with self.subTest(path=path):
                        connection = http.client.HTTPConnection(*server.server_address, timeout=3)
                        connection.request("GET", path)
                        response = connection.getresponse()
                        self.assertEqual(response.status, 200)
                        self.assertIn(expected, response.read())
                        connection.close()
            finally:
                if server:
                    server.shutdown()
                    server.server_close()
                for name, value in attributes.items():
                    setattr(MJPEGHandler, name, value)


if __name__ == "__main__":
    unittest.main()
