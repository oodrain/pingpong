"""Regression coverage for invisible historic labels and broken trajectory lines."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from development.video_studio.core import VideoEntry, load_reference
from development.video_studio.overlays import (interval_of, load_history,
                                                read_own_session, timed_tail, trajectory_segments)


class OverlayTests(unittest.TestCase):
    def test_timed_trail_stops_at_the_current_time_and_breaks_on_missing_frames(self):
        rows = [{"frame_id": f, "timestamp_s": f / 50, "x": f, "y": 2} for f in range(31)]
        rows[23]["detected"] = False
        del rows[25]
        trail = timed_tail(rows, .56, duration=.13)
        self.assertEqual([(point[1], previous[1] if previous is not None else None) for point, previous in trail],
                         [(22, None), (24, None), (26, None), (27, 26), (28, 27)])

    def test_tail_uses_frame_id_and_never_connects_missing_samples(self):
        rows = [{"frame_id": f, "input_frame_id": f + 1, "x": f, "y": 2, "detected": True} for f in range(31)]
        rows[12].update(x=None, y=None, detected=False)
        segments = trajectory_segments(rows, 28, tail=20)
        self.assertEqual([[p[0] for p in s] for s in segments], [list(range(8, 12)), list(range(13, 29))])
        del rows[17]
        self.assertEqual(len(trajectory_segments(rows, 28, tail=20)), 3)

    def test_project_review_intervals_are_not_converted_to_point_events(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "confirmed.json"
            path.write_text(json.dumps({"videos": {"00_001": [{"interval_frames": [82, 83], "event_type": "table_bounce"}], "00_002": [{"frame_id": 9}]}}), encoding="utf-8")
            rows = load_reference(path, "00_001")
            self.assertEqual(len(rows), 1)
            self.assertEqual(interval_of(rows[0]), (82, 83))
            self.assertNotIn("frame_id", rows[0])
            self.assertNotIn("x", rows[0])

    def test_read_only_history_includes_old_sessions_and_adjudicated_reviews(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "outputs/local_annotation/A/00_005/initial/session.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"video": {"video_id": "00_005"}, "inputs": {"video_sha256": "same"},
                "events": {"initial": [{"frame_id": 7, "x": 11, "y": 13}]},
                "uncertain": {"initial": [{"frame_start": 8, "frame_end": 9, "reason": "遮挡"}]}}), encoding="utf-8")
            review = root / "outputs/team_integration_20261022/stage/labels/local_development_20261022/precise_scopes.json"
            review.parent.mkdir(parents=True)
            review.write_text(json.dumps({"videos": [{"video_id": "00_005", "video_sha256": "same", "events": [{"interval_frames": [10, 11], "event_type": "racket_contact"}]}]}), encoding="utf-8")
            before = path.read_bytes()
            video = SimpleNamespace(sha256="same", n_frames=20)
            entry = VideoEntry("00_005", root / "video.mp4")
            rows, warnings = load_history(root, entry, video)
            self.assertEqual(len(rows), 3)
            self.assertFalse(warnings)
            self.assertEqual(path.read_bytes(), before)
            rows, warnings = load_history(root, entry, SimpleNamespace(sha256="different", n_frames=20))
            self.assertFalse(rows)
            self.assertEqual(len(warnings), 2)

    def test_view_mode_can_read_saved_work_without_acquiring_writer_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = VideoEntry("00_001", root / "video.mp4")
            video = SimpleNamespace(sha256="a" * 64, width=100, height=80, n_frames=20)
            folder = root / "sessions" / ("00_001-" + "a" * 12)
            folder.mkdir(parents=True)
            state = {"video_sha256": video.sha256, "width": 100, "height": 80, "n_frames": 20, "events": [{"frame_id": 3, "x": 4, "y": 5}]}
            (folder / "session.json").write_text(json.dumps(state), encoding="utf-8")
            before = set(folder.iterdir())
            self.assertEqual(read_own_session(root, entry, video)["events"][0]["frame_id"], 3)
            self.assertEqual(set(folder.iterdir()), before)
            self.assertIsNone(read_own_session(root, VideoEntry("00_002", entry.path), video))

    def test_same_filename_different_media_cannot_receive_project_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "data/splits/team_split_v1.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"videos": [{"video_id": "00_001", "video_sha256": "original"}]}), encoding="utf-8")
            rows, warnings = load_history(root, VideoEntry("00_001", root / "other.mp4"), SimpleNamespace(sha256="different", n_frames=100))
            self.assertFalse(rows)
            self.assertTrue(warnings)


if __name__ == "__main__":
    unittest.main()
