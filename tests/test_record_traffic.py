import unittest
import importlib.util
from pathlib import Path
import tempfile

from record_traffic import ReceiveTimeline, assess_frame_times


class ReceiveTimelineTests(unittest.TestCase):
    def test_receive_clock_ignores_camera_timestamp_jumps(self):
        clock = ReceiveTimeline()
        self.assertEqual(clock.timestamp(10_000_000_000), 0)
        self.assertEqual(clock.timestamp(10_040_000_000), 3600)
        self.assertEqual(clock.timestamp(10_080_000_000), 7200)

    def test_burst_frames_get_distinct_muxable_timestamps(self):
        clock = ReceiveTimeline()
        values = [clock.timestamp(value) for value in [1_000_000_000, 1_000_000_000, 1_000_000_001, 1_040_000_000]]
        self.assertEqual(values, [0, 1, 2, 3600])

    def test_backwards_receive_clock_is_rejected(self):
        clock = ReceiveTimeline()
        clock.timestamp(1_000_000_000)
        with self.assertRaises(ValueError):
            clock.timestamp(999_999_999)


class TimingQualityTests(unittest.TestCase):
    def test_continuous_25fps_clip_passes(self):
        report = assess_frame_times([index / 25 for index in range(101)])
        self.assertTrue(report['usable_for_counting'])
        self.assertEqual(report['effective_fps'], 25)

    def test_long_freeze_is_rejected_even_with_many_surrounding_frames(self):
        times = [index / 25 for index in range(101)] + [6 + index / 25 for index in range(100)]
        report = assess_frame_times(times)
        self.assertFalse(report['usable_for_counting'])
        self.assertEqual(report['gaps_over_0_5s'], 1)

    def test_low_delivery_rate_is_rejected_without_large_gaps(self):
        report = assess_frame_times([index / 8 for index in range(81)])
        self.assertFalse(report['usable_for_counting'])
        self.assertEqual(report['gaps_over_0_5s'], 0)

    def test_missing_or_non_monotonic_timeline_is_rejected(self):
        for values in ([], [0], [0, .04, .03], [0, float('nan')]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                assess_frame_times(values)


@unittest.skipUnless(importlib.util.find_spec('av'), 'Run encoder integration test in the project virtualenv')
class RecorderIntegrationTests(unittest.TestCase):
    def test_burst_delivery_keeps_every_frame_and_writes_monotonic_video(self):
        import av
        import numpy as np
        from record_traffic import record_clip
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source.mp4'
            target = Path(folder) / 'recorded.mp4'
            with av.open(str(source), 'w') as container:
                stream = container.add_stream('libx264', rate=25)
                stream.width, stream.height, stream.pix_fmt = 320, 180, 'yuv420p'
                for index in range(12):
                    frame = av.VideoFrame.from_ndarray(np.full((180, 320, 3), index * 15, dtype=np.uint8), format='bgr24')
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            report = record_clip(str(source), target, 30)
            self.assertEqual(report['received_frames'], 7)
            self.assertEqual(report['frames'], 7)
            self.assertFalse(report['capture_complete'])
            from record_traffic import check_counting_timing
            self.assertFalse(check_counting_timing(target)['usable_for_counting'])
            with av.open(str(target)) as container:
                frames = list(container.decode(video=0))
                times = [float(frame.time) for frame in frames]
            self.assertEqual(len(frames), 7)
            self.assertTrue(all(b > a for a, b in zip(times, times[1:])))

    def test_existing_audit_sidecars_are_preserved_even_without_video(self):
        from record_traffic import record_clip
        for suffix in ('.frames.csv', '.timing.json'):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as folder:
                target = Path(folder) / 'recorded.mp4'
                evidence = target.with_suffix(suffix)
                evidence.write_text('original audit evidence', encoding='utf-8')
                with self.assertRaises(FileExistsError):
                    record_clip(str(Path(folder) / 'missing_source.mp4'), target, 30)
                self.assertEqual(evidence.read_text(encoding='utf-8'), 'original audit evidence')


if __name__ == '__main__':
    unittest.main()
