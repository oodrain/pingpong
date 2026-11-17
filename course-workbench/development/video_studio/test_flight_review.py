import unittest

from flight_review import flight_hints
from review_geometry.table_geometry import build_plane
from table_view import project_events


class FlightReviewTest(unittest.TestCase):
    def hint(self, vertical, *, stride=1, missing=()):
        times = {}
        tracks = []
        for index in range(9):
            frame = index * stride
            t = (index - 4) * .04
            times[frame] = t + 1
            tracks.append(dict(frame_id=frame, timestamp_s=times[frame], x=310 + 100 * t,
                               y=vertical(t), detected=index not in missing, confidence=.95))
        candidate = dict(frame_id=4 * stride, x=310, y=vertical(0))
        return flight_hints([candidate], tracks, times, 640).get(candidate["frame_id"])

    def test_smooth_flight_and_bounce_at_source_frame_rates(self):
        for stride in (1, 2):
            with self.subTest(stride=stride):
                self.assertEqual(self.hint(lambda t: 100 + 400 * t**2, stride=stride), "smooth_apex")
                self.assertEqual(self.hint(lambda t: 100 - 35 * t, stride=stride), "smooth_pass")
                self.assertIsNone(self.hint(lambda t: 100 - 120 * abs(t), stride=stride))
                self.assertIsNone(self.hint(lambda t: 100 + 400 * t**2, stride=stride, missing=(2,)))

    def test_suspected_flight_requires_manual_confirmation_to_project(self):
        plane = build_plane(dict(corners=[(100, 120), (410, 120), (410, 290), (100, 290)],
                                 edge01_axis="length"))
        event = dict(frame_id=10, x=210, y=150, flight_hint="smooth_apex")
        rows = project_events(plane, [dict(event, status=state)
                                      for state in ("pending", "confirmed", "rejected")])
        self.assertEqual([r["inside"] for r in rows], [False, True, False])
        self.assertEqual([r["frame_id"] for r in rows], [10, 10, 10])
        self.assertEqual(rows[0]["map_note"], "疑似空中")


if __name__ == "__main__":
    unittest.main()
