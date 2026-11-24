import http.client
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ultralytics.engine.results import Results

from region_calibration import CalibrationStore
from realtime_web import MJPEGHandler, serve
from traffic_api import EventStore, TrafficCamera
from traffic_count import ClassVote, LineCounter


CALIBRATION = {"roi": [[.1, .1], [.9, .1], [.9, .9], [.1, .9]],
               "line": [[.2, .5], [.8, .5]], "direction": [[.5, .2], [.5, .8]],
               "frame_size": [1000, 1000]}
NAMES = {0: "Car", 1: "Truck"}


def result_for(*vehicles):
    # x centre, bottom y, class and confidence, with overlapping boxes across frames.
    boxes = [[x - 60, y - 180, x + 60, y, confidence, label]
             for x, y, label, confidence in vehicles]
    return Results(np.zeros((1000, 1000, 3), dtype=np.uint8), path="fixture", names=NAMES,
                   boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 6))


class ConsensusTests(unittest.TestCase):
    def test_quality_weight_and_time_bins_resist_many_weak_duplicate_frames(self):
        votes = ClassVote()
        for i in range(100):
            votes.add("Truck", .3, .2, i * .001)
        votes.add("Car", .9, 1, .21)
        votes.add("Car", .9, 1, .41)
        self.assertEqual(votes.decision()["label"], "Car")

    def test_post_crossing_class_change_cannot_replace_roi_consensus(self):
        counter = LineCounter(CALIBRATION, min_roi_observations=2)
        counter.observe(1, [.5, .48], 0, "Car", .9)
        counter.observe(1, [.5, .49], .2, "Car", .9)
        counter.observe(1, [.5, .501], .4, "Truck", .99)
        event = counter.observe(1, [.5, .51], .6, "Truck", .99)
        self.assertEqual(event["label"], "Car")

    def test_outside_roi_evidence_and_single_inside_observation_do_not_count(self):
        counter = LineCounter(CALIBRATION, min_roi_observations=2)
        counter.observe(1, [.5, .05], 0, "Truck", .99)
        counter.observe(1, [.5, .49], .2, "Car", .7)
        self.assertIsNone(counter.observe(1, [.5, .51], .4, "Truck", .99))
        counter.observe(2, [.5, .48], 0, "Car", .7)
        counter.observe(2, [.5, .49], .2, "Car", .7)
        self.assertEqual(counter.observe(2, [.5, .51], .4, "Truck", .99)["label"], "Car")

    def test_crossing_line_in_wrong_arrow_direction_is_rejected(self):
        calibration = dict(CALIBRATION, direction=[[.2, .4], [.8, .6]])
        counter = LineCounter(calibration)
        counter.observe(1, [.7, .49], 0, "Car")
        self.assertIsNone(counter.observe(1, [.3, .51], .2, "Car"))

    def test_continuous_counter_expires_old_tracks_and_keeps_no_event_history(self):
        counter = LineCounter(CALIBRATION, keep_history=False)
        counter.observe(1, [.5, .49], 0, "Car")
        counter.observe(1, [.5, .51], .2, "Car")
        counter.expire(31)
        self.assertEqual(counter.total, 1)
        self.assertFalse(counter.events)
        self.assertFalse(counter.tracks)


class TrafficAPITests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "events.sqlite3"
        self.store = EventStore(self.path)
        self.addCleanup(self.store.close)
        self.calibrations = CalibrationStore(Path(self.folder.name) / "regions.json")
        for camera_id in ("a", "b"):
            self.calibrations.save(camera_id, CALIBRATION)
        self.a = TrafficCamera("a", "cam-a", self.calibrations, self.store)
        self.b = TrafficCamera("b", "cam-b", self.calibrations, self.store)

    def test_real_tracker_emits_one_consensus_class_per_camera(self):
        for i, y in enumerate((480, 486, 492, 498, 504, 510, 516, 522)):
            label = 1 if y >= 504 else 0
            self.a.process(result_for((400, y, label, .9)), 100 + i * .1, 0)
            self.b.process(result_for((600, y, 1, .9)), 100 + i * .1, 0)
        self.a.flush(102)
        self.b.flush(102)
        events = self.store.read(0, 100)["events"]
        self.assertEqual([(e["camera_id"], e["sequence"], e["class_name"]) for e in events],
                         [("a", 1, "Car"), ("b", 1, "Truck")])
        self.assertFalse({"confidence", "unknown", "class_confirmed"} & events[0].keys())

    def test_reset_one_camera_does_not_reuse_other_camera_track_ids(self):
        old = self.b.process(result_for((600, 480, 1, .9)), 100, 0)[0, 4]
        self.a.process(result_for((400, 480, 0, .9)), 100, 0)
        self.a.process(result_for((400, 480, 0, .9)), 100.1, 1)
        self.b.process(result_for((600, 480, 1, .9), (300, 480, 1, .9)), 100.1, 0)
        tracks = self.b.process(result_for((600, 482, 1, .9), (300, 482, 1, .9)), 100.2, 0)
        self.assertEqual(len(set(tracks[:, 4])), 2)
        self.assertIn(old, tracks[:, 4])
        self.assertEqual(self.b.resets, 0)

    def test_delayed_deadband_confirmation_preserves_actual_crossing_order(self):
        def tracks(boxes, image):
            return np.asarray([[*row[:4], i + 1, *row[4:], i]
                               for i, row in enumerate(boxes.data)], dtype=np.float32)

        with patch.object(self.a.tracker, "update", side_effect=tracks):
            for i, (first, second) in enumerate(((488, 480), (495, 490), (501, 498), (502, 510))):
                self.a.process(result_for((400, first, 0, .9), (600, second, 1, .9)), 100 + i * .2, 0)
            self.assertFalse(self.store.read(0, 100)["events"])
            self.a.process(result_for((400, 505, 0, .9), (600, 520, 1, .9)), 100.8, 0)
        self.a.flush(102)
        events = self.store.read(0, 100)["events"]
        self.assertEqual([e["class_name"] for e in events], ["Car", "Truck"])
        self.assertLess(events[0]["crossed_at"], events[1]["crossed_at"])

    def test_gap_reconnect_and_calibration_change_do_not_invent_crossings(self):
        for reason in ("gap", "reconnect", "calibration"):
            with self.subTest(reason=reason):
                camera = TrafficCamera("a", "cam-a", self.calibrations, self.store)
                camera.process(result_for((400, 490, 0, .9)), 100, 0)
                camera.process(result_for((400, 496, 0, .9)), 100.1, 0)
                if reason == "calibration":
                    self.calibrations.save("a", dict(CALIBRATION, direction=[[.5, .21], [.5, .8]]))
                camera.process(result_for((400, 505, 0, .9)), 101 if reason == "gap" else 100.2,
                               1 if reason == "reconnect" else 0)
                camera.flush(102)
                self.assertEqual(camera.resets, 1)
                self.assertFalse(self.store.read(0, 100)["events"])

    def test_missing_or_mismatched_calibration_disables_counting(self):
        camera = TrafficCamera("missing", "missing", self.calibrations, self.store)
        camera.process(result_for((400, 490, 0, .9)), 100, 0)
        self.assertEqual(camera.status()["status"], "needs_calibration")
        self.calibrations.save("a", dict(CALIBRATION, frame_size=[1920, 1080]))
        self.a.process(result_for((400, 490, 0, .9)), 100, 0)
        self.assertEqual(self.a.state, "frame_size_mismatch")

    def test_sqlite_restart_pagination_retry_and_failed_transaction(self):
        self.store.append("a", [{"class_name": "Car"}, {"class_name": "Truck"}])
        first = self.store.read(0, 1)
        self.assertTrue(first["has_more"])
        self.assertEqual(first, self.store.read(0, 1))
        with self.assertRaises(ValueError):
            self.store.append("a", [{"class_name": "Bus"}, {"invalid": float("nan")}])
        reopened = EventStore(self.path)
        try:
            reopened.append("a", [{"class_name": "Bus"}])
            rest = reopened.read(first["next_after"], 100)
            self.assertEqual([event["sequence"] for event in rest["events"]], [2, 3])
        finally:
            reopened.close()

    def test_actual_http_feed_and_parameter_validation(self):
        self.store.append("a", [{"class_name": "Car"}])
        with patch.object(MJPEGHandler, "event_store", self.store), \
                patch.object(MJPEGHandler, "traffic_cameras", [self.a, self.b]):
            server = serve("127.0.0.1", 0)
            try:
                for route, status in (("cameras", 200), ("events?after=0&limit=1&camera_id=a", 200),
                                      ("events?after=-1", 400), ("events?limit=1001", 400),
                                      ("events?after=0&after=1", 400), ("events?camera_id=missing", 404),
                                      ("events?after=999999999999999999999", 400), ("missing", 404)):
                    with self.subTest(route=route):
                        connection = http.client.HTTPConnection(*server.server_address, timeout=3)
                        connection.request("GET", "/api/traffic/" + route)
                        response = connection.getresponse()
                        self.assertEqual(response.status, status)
                        self.assertTrue(response.getheader("Content-Type").startswith("application/json"))
                        body = json.loads(response.read())
                        if route.startswith("events?after=0&limit"):
                            self.assertEqual(body["events"][0]["class_name"], "Car")
                        connection.close()
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
