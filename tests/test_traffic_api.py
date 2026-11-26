import http.client
import json
import sqlite3
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from ultralytics.engine.results import Results

from region_calibration import CalibrationStore
from realtime_web import MJPEGHandler, serve
from traffic_api import EventStore, TrafficCamera
from traffic_count import ClassVote, LineCounter


CALIBRATION = {"roi": [[.1, .1], [.9, .1], [.9, .9], [.1, .9]],
               "line": [[.2, .5], [.8, .5]], "direction": [[.5, .2], [.5, .8]],
               "frame_size": [1000, 1000]}
NAMES = {0: "Car", 1: "Truck"}


def stored_event(label=0, name="Car"):
    return {"class_id": label, "class_name": name, "crossed_at": "2026-11-24T08:00:00.000+00:00"}


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
        self.store = EventStore(self.path, NAMES)
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
        counts = self.store.counts([("a", "cam-a"), ("b", "cam-b")])["cameras"]
        self.assertEqual([[v["count"] for v in c["counts"]] for c in counts], [[1, 0], [0, 1]])

    def test_reset_one_camera_does_not_reuse_other_camera_track_ids(self):
        old = self.b.process(result_for((600, 480, 1, .9)), 100, 0)[0, 4]
        self.a.process(result_for((400, 480, 0, .9)), 100, 0)
        self.a.process(result_for((400, 480, 0, .9)), 100.1, 1)
        self.b.process(result_for((600, 480, 1, .9), (300, 480, 1, .9)), 100.1, 0)
        tracks = self.b.process(result_for((600, 482, 1, .9), (300, 482, 1, .9)), 100.2, 0)
        self.assertEqual(len(set(tracks[:, 4])), 2)
        self.assertIn(old, tracks[:, 4])
        self.assertEqual(self.b.resets, 0)

    def test_overlapping_class_duplicates_count_once_without_merging_nearby_vehicle(self):
        for i, y in enumerate((480, 486, 492, 498, 504, 510, 516, 522)):
            result = result_for((400, y, 0, .95), (401, y + 1, 1, .65), (440, y, 1, .9))
            tracks = self.a.process(result, 100 + i * .1, 0)
            self.assertEqual(len(result.boxes), 2)
            self.assertEqual(len(tracks), 2)
            self.assertTrue(all(0 <= int(track[-1]) < len(result.boxes) for track in tracks))
        self.a.flush(102)
        self.assertEqual([e["class_name"] for e in self.store.read(0, 100)["events"]], ["Car", "Truck"])

    def test_large_half_precision_boxes_keep_distinct_overlapping_vehicles(self):
        boxes = torch.tensor([[100, 100, 600, 800, .95, 0], [101, 101, 601, 801, .8, 1],
                              [350, 100, 850, 800, .9, 1]], dtype=torch.float16)
        result = Results(np.zeros((1000, 1000, 3), dtype=np.uint8), path="fixture", names=NAMES, boxes=boxes)
        self.a.process(result, 100, 0)
        self.assertEqual(len(result.boxes), 2)
        self.assertEqual(result.boxes.cls.tolist(), [0, 1])

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

    def test_short_gap_keeps_identity_without_guessing_a_crossing(self):
        tracks = self.a.process(result_for((400, 490, 0, .9)), 100, 0)
        track_id = tracks[0, 4]
        self.a.process(result_for((400, 496, 0, .9)), 100.1, 0)
        tracks = self.a.process(result_for((400, 505, 0, .9)), 101, 0)
        self.a.flush(102)
        self.assertEqual(self.a.resets, 0)
        self.assertEqual(tracks[0, 4], track_id)
        self.assertFalse(self.store.read(0, 100)["events"])
        self.assertEqual(self.a.status()["tracking_gaps"], 1)
        self.assertEqual(self.a.status()["last_tracking_gap_s"], .9)

    def test_counted_vehicle_cannot_be_counted_again_after_short_pause(self):
        for i, y in enumerate((480, 490, 500, 510, 520)):
            tracks = self.a.process(result_for((400, y, 0, .9)), 100 + i * .1, 0)
        track_id = tracks[0, 4]
        self.a.flush(101)
        self.assertEqual(len(self.store.read(0, 100)["events"]), 1)
        for i, y in enumerate((520, 490, 486, 510, 520)):
            tracks = self.a.process(result_for((400, y, 0, .9)), 101.2 + i * .1, 0)
        self.a.flush(103)
        self.assertEqual(tracks[0, 4], track_id)
        self.assertEqual(self.a.resets, 0)
        self.assertEqual(len(self.store.read(0, 100)["events"]), 1)

    def test_gap_reconnect_and_calibration_change_do_not_invent_crossings(self):
        for reason in ("gap", "reconnect", "calibration"):
            with self.subTest(reason=reason):
                camera = TrafficCamera("a", "cam-a", self.calibrations, self.store)
                camera.process(result_for((400, 490, 0, .9)), 100, 0)
                camera.process(result_for((400, 496, 0, .9)), 100.1, 0)
                if reason == "calibration":
                    self.calibrations.save("a", dict(CALIBRATION, direction=[[.5, .21], [.5, .8]]))
                camera.process(result_for((400, 505, 0, .9)), 103 if reason == "gap" else 100.2,
                               1 if reason == "reconnect" else 0)
                camera.flush(104)
                self.assertEqual(camera.resets, 1)
                expected = {"gap": "frame_gap", "reconnect": "stream_reconnected",
                            "calibration": "calibration_changed"}[reason]
                self.assertEqual(camera.status()["tracking_reset_reasons"], {expected: 1})
                self.assertEqual(camera.status()["last_reset_reason"], expected)
                self.assertFalse(self.store.read(0, 100)["events"])

    def test_missing_or_mismatched_calibration_disables_counting(self):
        camera = TrafficCamera("missing", "missing", self.calibrations, self.store)
        camera.process(result_for((400, 490, 0, .9)), 100, 0)
        self.assertEqual(camera.status()["status"], "needs_calibration")
        self.calibrations.save("a", dict(CALIBRATION, frame_size=[1920, 1080]))
        self.a.process(result_for((400, 490, 0, .9)), 100, 0)
        self.assertEqual(self.a.state, "frame_size_mismatch")

    def test_sqlite_restart_pagination_retry_and_failed_transaction(self):
        self.store.append("a", [stored_event(), stored_event(1, "Truck")])
        first = self.store.read(0, 1)
        self.assertTrue(first["has_more"])
        self.assertEqual(first, self.store.read(0, 1))
        with self.assertRaises(ValueError):
            self.store.append("a", [stored_event(2, "Bus"), {"invalid": float("nan")}])
        self.assertEqual(self.store.counts([("a", "cam-a")])["cameras"][0]["total"], 2)
        reopened = EventStore(self.path)
        try:
            reopened.append("a", [stored_event(2, "Bus")])
            rest = reopened.read(first["next_after"], 100)
            self.assertEqual([event["sequence"] for event in rest["events"]], [2, 3])
        finally:
            reopened.close()

    def test_actual_http_feed_and_parameter_validation(self):
        self.store.append("a", [stored_event()])
        with patch.object(MJPEGHandler, "event_store", self.store), \
                patch.object(MJPEGHandler, "traffic_cameras", [self.a, self.b]):
            server = serve("127.0.0.1", 0)
            try:
                for route, status in (("cameras", 200), ("events?after=0&limit=1&camera_id=a", 200),
                                      ("events?after=-1", 400), ("events?limit=1001", 400),
                                      ("events?after=0&after=1", 400), ("events?camera_id=missing", 404),
                                      ("events?after=999999999999999999999", 400), ("missing", 404),
                                      ("counts", 200), ("counts?camera_id=b", 200),
                                      ("counts?camera_id=missing", 404), ("counts?camera_id=", 404),
                                      ("counts?camera_id=a&camera_id=b", 400), ("counts?after=0", 400)):
                    with self.subTest(route=route):
                        connection = http.client.HTTPConnection(*server.server_address, timeout=3)
                        connection.request("GET", "/api/traffic/" + route)
                        response = connection.getresponse()
                        self.assertEqual(response.status, status)
                        self.assertTrue(response.getheader("Content-Type").startswith("application/json"))
                        body = json.loads(response.read())
                        if route.startswith("events?after=0&limit"):
                            self.assertEqual(body["events"][0]["class_name"], "Car")
                        if route == "counts":
                            self.assertEqual(body["scope"], "all_time")
                            self.assertEqual([c["total"] for c in body["cameras"]], [1, 0])
                            self.assertEqual(body["revision"], 1)
                        if route == "counts?camera_id=b":
                            self.assertEqual(len(body["cameras"]), 1)
                            self.assertEqual([v["count"] for v in body["cameras"][0]["counts"]], [0, 0])
                        connection.close()
            finally:
                server.shutdown()
                server.server_close()

    def test_legacy_events_backfill_restart_and_new_events_preserve_totals(self):
        legacy = Path(self.folder.name) / "legacy.sqlite3"
        with sqlite3.connect(legacy) as db:
            db.execute("""CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL, sequence INTEGER NOT NULL, payload TEXT NOT NULL,
                UNIQUE(camera_id, sequence))""")
            for camera_id, sequence, event in [("a", 1, stored_event()), ("a", 2, stored_event(1, "Truck")),
                                                ("b", 1, stored_event(1, "Truck"))]:
                db.execute("INSERT INTO events(camera_id,sequence,payload) VALUES(?,?,?)",
                           (camera_id, sequence, json.dumps(dict(event, camera_id=camera_id, sequence=sequence))))
        db.close()
        for restart in range(2):
            store = EventStore(legacy, NAMES)
            try:
                before = store.read(0, 100)
                counts = store.counts([("a", "cam-a"), ("b", "cam-b"), ("c", "empty")])
                self.assertEqual([c["total"] for c in counts["cameras"]], [2 + restart, 1, 0])
                self.assertEqual(counts["cameras"][2]["counting_since"], None)
                store.append("a", [stored_event()])
                self.assertEqual(store.read(0, 3)["events"], before["events"][:3])
                self.assertEqual(store.counts([("a", "cam-a")])["cameras"][0]["total"], 3 + restart)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
