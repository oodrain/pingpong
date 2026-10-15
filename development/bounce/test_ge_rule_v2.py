"""Regression checks for the three independently reproduced D defects."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from development.bounce.ge_rule_v2 import RuleConfig, replay
from development.bounce.table_geometry import GeometryError, GeometryTimeline, TablePlane, build_plane
from development.evaluation.evaluate_landings import score_video


def raw_points():
    # Three true local peaks, close in time and separated in table coordinates.
    return [{"input_frame_id": i + 1, "detected": True, "confidence": .9, "x": x, "y": y}
            for i, (x, y) in enumerate([(100, 0), (100, 5), (100, 0),
                                       (1000, 0), (1000, 5), (1000, 0),
                                       (1800, 0), (1800, 5), (1800, 0)])]


def physical_document():
    return {"segments": [{"corners": [[0, 0], [2740, 0], [2740, 1525], [0, 1525]],
                          "edge01_axis": "length"}]}


class GeometryTests(unittest.TestCase):
    def test_all_cyclic_orders_have_same_region_and_distance(self):
        q = np.array([[0., 0.], [100., 0.], [100., 50.], [0., 50.]])
        for winding in (q, q[::-1]):
            for shift in range(4):
                corners = np.roll(winding, shift, axis=0)
                plane = build_plane({"corners": corners.tolist(), "net": [[50, 0], [50.1, 50]]})
                for x, y in q:
                    self.assertAlmostEqual(plane.outside_distance(x, y), 0, places=5)
                self.assertAlmostEqual(plane.outside_distance(130, 25), 822, places=5)

    def test_horizontal_net_identifies_width_edge(self):
        plane = build_plane({"corners": [[0, 0], [100, 0], [100, 50], [0, 50]],
                             "net": [[0, 25], [100, 25]]})
        self.assertAlmostEqual(plane.outside_distance(50, 65), 822, places=5)

    def test_perspective_corner_coordinates_remain_inside(self):
        plane = build_plane({"corners": [[310, 180], [850, 220], [1080, 610], [100, 590]],
                             "edge01_axis": "width"})
        for x, y in ((310, 180), (850, 220), (1080, 610), (100, 590)):
            self.assertLess(plane.outside_distance(x, y), .001)

    def test_axis_unknown_is_fallback_not_screen_shape_guess(self):
        timeline = GeometryTimeline({"segments": [{"corners": [[0,0],[100,0],[100,50],[0,50]]}]})
        self.assertIsNone(timeline.lookup(1))
        self.assertTrue(timeline.issues)

    def test_invalid_geometry_and_axis_conflict(self):
        for segment in ({"corners": [[0,0],[1,1],[0,1],[1,0]], "edge01_axis": "length"},
                        {"corners": [[0,0],[100,0],[100,50],[0,50]], "net": [[50,0],[50,50]], "edge01_axis": "width"},
                        {"corners": [[0,0],[100,0],[100,50],[0,50]], "net": "bad"}):
            with self.assertRaises(GeometryError):
                build_plane(segment)

    def test_projective_infinity_is_unknown(self):
        plane = TablePlane(np.array([[1,0,0],[0,1,0],[1,0,-100.]]), "infinite", "test")
        with self.assertRaises(GeometryError):
            plane.project(100, 5)

    def test_timeline_gaps_ranges_and_overlap(self):
        seg = physical_document()["segments"][0]
        timeline = GeometryTimeline({"segments": [dict(seg, start_frame=0, end_frame=4),
                                                   dict(seg, start_frame=7, end_frame=10)]})
        self.assertEqual(timeline.lookup(4).segment_id, "0")
        self.assertIsNone(timeline.lookup(5))
        self.assertEqual(timeline.lookup(7).segment_id, "1")
        with self.assertRaises(GeometryError):
            GeometryTimeline({"segments": [dict(seg, start_frame=0, end_frame=5),
                                           dict(seg, start_frame=5, end_frame=10)]})


class ReplayTests(unittest.TestCase):
    def test_smart_dedup_emits_once_in_order(self):
        preds, _ = replay(raw_points(), GeometryTimeline(physical_document()), "v1b")
        self.assertEqual([p["frame_id"] for p in preds], [1, 4, 7])

    def test_missing_geometry_preserves_v0_even_for_smart_dedup(self):
        baseline, _ = replay(raw_points(), mode="v0")
        for mode in ("v1a", "v1b"):
            pred, decisions = replay(raw_points(), mode=mode)
            self.assertEqual(pred, baseline)
            self.assertTrue(any(d["reason"] == "accepted_fallback" for d in decisions))

    def test_zero_based_raw_and_one_time_compensation(self):
        raw = [{"input_frame_id": 0, "detected": False}] + raw_points()
        pred, _ = replay(raw)
        self.assertEqual(pred[0]["frame_id"], 1)
        self.assertEqual(pred[0]["x"], 100.)

    def test_far_outside_candidate_is_rejected(self):
        raw = raw_points()
        for r in raw:
            r["x"] = 4000.
        pred, decisions = replay(raw, GeometryTimeline(physical_document()), "v1a", RuleConfig(margin_mm=0))
        self.assertEqual(pred, [])
        self.assertTrue(all(d["reason"] == "outside_table_margin" for d in decisions))

    def test_no_spatial_exception_across_camera_segments(self):
        seg = physical_document()["segments"][0]
        timeline = GeometryTimeline({"segments": [dict(seg, start_frame=0, end_frame=2),
                                                   dict(seg, start_frame=3, end_frame=20)]})
        pred, _ = replay(raw_points(), timeline, "v1b")
        self.assertEqual([p["frame_id"] for p in pred], [1])

    def test_transition_candidate_cannot_use_cross_camera_turn(self):
        seg = physical_document()["segments"][0]
        timeline = GeometryTimeline({"segments": [dict(seg, start_frame=0, end_frame=1),
                                                   dict(seg, start_frame=2, end_frame=20)]})
        _, ds = replay(raw_points(), timeline, "v1a")
        self.assertEqual(ds[0]["reason"], "geometry_segment_transition")

    def test_nonfinite_or_nonchronological_raw_fails(self):
        for raw in (raw_points()[::-1], [dict(raw_points()[0], x=float("nan"))]):
            with self.assertRaises(ValueError):
                replay(raw)


class ScoringTests(unittest.TestCase):
    def test_maximum_matching_and_valid_priority(self):
        pred = [{"frame_id": 10, "x": x, "y": 100} for x in (110,70)]
        gt = [{"frame_id": 10, "x": x, "y": 100} for x in (100,140)]
        self.assertEqual(score_video(pred, gt, [], exempt_invalid=False)["tp"], 2)
        valid = [{"frame_id": 10, "x": 120, "y": 100}]
        invalid = [{"frame_id": 10, "x": 110, "y": 100}]
        for ignore in (False, True):
            scored = score_video(pred[:1], valid, invalid, exempt_invalid=ignore)
            self.assertEqual((scored["tp"], scored["fp"], scored["fn"]), (1,0,0))


class CliTests(unittest.TestCase):
    def test_full_geometry_artifacts_and_strict_missing_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "raw").mkdir()
            (folder / "geometry").mkdir()
            raw = [{"input_frame_id": 0, "detected": False}] + raw_points()
            (folder / "raw/synthetic.jsonl").write_text("\n".join(json.dumps(r) for r in raw), encoding="utf-8")
            manifest = {"videos": [{"video_id": "synthetic", "angle": "0度", "n_frames": len(raw),
                                     "width": 5000, "height": 2000, "n_valid_landings": 3}]}
            (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            gt = [dict(video_id="synthetic", frame_id=f, x=x, y=5, valid=True) for f,x in ((1,100),(4,1000),(7,1800))]
            (folder / "gt.jsonl").write_text("\n".join(json.dumps(r) for r in gt), encoding="utf-8")
            (folder / "geometry/synthetic.json").write_text(json.dumps(physical_document()), encoding="utf-8")
            script = Path(__file__).resolve().parents[1] / "evaluation/run_geometry_d_v2.py"
            cmd = [sys.executable, str(script), "--raw-dir", str(folder / "raw"),
                   "--manifest", str(folder / "manifest.json"), "--gt", str(folder / "gt.jsonl"),
                   "--require-geometry", "--output", str(folder / "out")]
            failed = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((folder / "out").exists())
            result = subprocess.run(cmd + ["--geometry-dir", str(folder / "geometry")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((folder / "out/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "geometry_experiment_complete")
            metrics = json.loads((folder / "out/metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["v1b"]["without_ignore_exemption"]["overall"]["tp"], 3)
            self.assertTrue((folder / "out/changed_matches.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
