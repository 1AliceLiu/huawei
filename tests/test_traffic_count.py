import unittest

from traffic_count import LineCounter


CALIBRATION = {
    "roi": [[.1, .1], [.9, .1], [.9, .9], [.1, .9]],
    "line": [[.2, .5], [.8, .5]],
    "direction": [[.5, .2], [.5, .8]],
    "frame_size": [1000, 1000],
}


class LineCounterTests(unittest.TestCase):
    def test_forward_crossing_counts_once_per_track(self):
        counter = LineCounter(CALIBRATION)
        self.assertIsNone(counter.observe(7, [.5, .4], 0, 'Truck'))
        event = counter.observe(7, [.5, .6], .04, 'Truck')
        self.assertEqual(event['track_id'], 7)
        self.assertEqual(event['label'], 'Truck')
        self.assertEqual(counter.total, 1)
        counter.observe(7, [.5, .4], .08, 'Truck')
        counter.observe(7, [.5, .6], .12, 'Truck')
        self.assertEqual(counter.total, 1)

    def test_reverse_traffic_and_line_extensions_do_not_count(self):
        counter = LineCounter(CALIBRATION)
        for track_id, points in ((1, [[.5, .6], [.5, .4]]),
                                 (2, [[.85, .4], [.85, .6]]),
                                 (3, [[.95, .4], [.95, .6]])):
            for i, point in enumerate(points):
                counter.observe(track_id, point, i * .04, 'Car')
        self.assertEqual(counter.total, 0)

    def test_deadband_prevents_line_jitter_from_counting(self):
        counter = LineCounter(CALIBRATION)
        for i, y in enumerate([.49, .501, .499, .502, .498]):
            counter.observe(1, [.5, y], i * .04, 'Car')
        self.assertEqual(counter.total, 0)
        counter.observe(1, [.5, .51], .2, 'Car')
        self.assertEqual(counter.total, 1)

    def test_curve_around_line_endpoint_does_not_invent_a_crossing(self):
        counter = LineCounter(CALIBRATION)
        for i, point in enumerate([[.15, .49], [.15, .502], [.5, .502], [.5, .51]]):
            counter.observe(1, point, i * .04, 'Car')
        self.assertEqual(counter.total, 0)

    def test_crossing_candidate_preserves_actual_intersection_through_deadband(self):
        counter = LineCounter(CALIBRATION)
        for i, point in enumerate([[.5, .49], [.5, .502], [.85, .502], [.85, .51]]):
            counter.observe(1, point, i * .04, 'Car')
        self.assertEqual(counter.total, 1)
        self.assertEqual(counter.events[0]['crossing_x'], .5)

    def test_exact_line_touch_requires_continuing_to_opposite_side(self):
        counter = LineCounter(CALIBRATION)
        for i, y in enumerate([.49, .5, .49]):
            counter.observe(1, [.5, y], i * .04, 'Car')
        self.assertEqual(counter.total, 0)
        counter.observe(1, [.5, .5], .12, 'Car')
        counter.observe(1, [.5, .51], .16, 'Car')
        self.assertEqual(counter.total, 1)

    def test_tracker_reset_does_not_suppress_new_vehicle_reusing_numeric_id(self):
        counter = LineCounter(CALIBRATION)
        counter.observe(1, [.5, .4], 0, 'Car')
        counter.observe(1, [.5, .6], .04, 'Car')
        counter.new_segment()
        counter.observe(1, [.5, .4], 10, 'Car')
        counter.observe(1, [.5, .6], 10.04, 'Car')
        self.assertEqual(counter.total, 2)
        self.assertEqual([event['tracking_epoch'] for event in counter.events], [0, 1])

    def test_long_tracking_gap_does_not_invent_a_crossing(self):
        counter = LineCounter(CALIBRATION)
        counter.observe(1, [.5, .4], 0, 'Car')
        counter.observe(1, [.5, .6], 1, 'Car')
        self.assertEqual(counter.total, 0)

    def test_crossing_is_valid_when_vehicle_exits_roi_just_after_line(self):
        counter = LineCounter(CALIBRATION)
        counter.observe(1, [.5, .4], 0, 'Car')
        counter.observe(1, [.5, .95], .04, 'Car')
        self.assertEqual(counter.total, 1)

    def test_tracking_jump_from_outside_roi_does_not_count(self):
        counter = LineCounter(CALIBRATION)
        counter.observe(1, [.5, .05], 0, 'Car')
        counter.observe(1, [.5, .6], .04, 'Car')
        self.assertEqual(counter.total, 0)

    def test_direction_can_be_reversed_in_calibration(self):
        calibration = dict(CALIBRATION, direction=list(reversed(CALIBRATION['direction'])))
        counter = LineCounter(calibration)
        counter.observe(1, [.5, .6], 0, 'Car')
        counter.observe(1, [.5, .4], .04, 'Car')
        self.assertEqual(counter.total, 1)

    def test_incomplete_calibration_is_rejected(self):
        with self.assertRaises(ValueError):
            LineCounter(dict(CALIBRATION, line=[], direction=[]))


if __name__ == '__main__':
    unittest.main()
