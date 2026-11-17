"""Regression coverage for staging portable video results."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from portable_results import SCHEMA, commit, export_result, latest, rows_for_video


class ResultStagingTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.work = self.root / "工作区"
        self.exports = self.root / "导出"
        self.source_hash = "a" * 64
        self.manifest = dict(schema=SCHEMA, status="succeeded", run_id=str(uuid4()),
                             source_sha256=self.source_hash, video_id="样片", width=800, height=600,
                             source_frames=30, event_count=1)
        self.event = dict(frame_id=5, x=320, y=240, video_id="样片")

    @unittest.skipUnless(os.name == "nt", "Windows uses directory modes for ACL selection")
    def test_save_and_export_inherit_destination_permissions(self):
        mkdir = os.mkdir
        modes = []

        def restricted_mkdir(path, mode=0o777, *args, **kwargs):
            candidate = Path(path)
            if candidate.name.startswith(".pending-") and not candidate.exists():
                modes.append(mode)
                if mode == 0o700:
                    self.fail("Windows result staging must inherit the destination ACL")
            return mkdir(path, mode, *args, **kwargs)

        with patch("os.mkdir", side_effect=restricted_mkdir):
            commit(self.work, self.manifest, [], [self.event])
            result = latest(self.work, self.source_hash)
            exported = export_result(result, self.exports, "样片")

        self.assertEqual(modes, [0o777, 0o777])
        self.assertEqual(rows_for_video(result, "样片", 800, 600, 30), [self.event])
        self.assertEqual(json.loads((exported / "predictions.jsonl").read_text(encoding="utf-8")),
                         self.event)

    def test_failed_write_preserves_previous_result_and_cleans_staging(self):
        commit(self.work, self.manifest, [], [self.event])
        previous = latest(self.work, self.source_hash)
        retry = dict(self.manifest, run_id=str(uuid4()))

        with patch("portable_results._jsonl", side_effect=PermissionError("Cannot write predictions")):
            with self.assertRaises(PermissionError):
                commit(self.work, retry, [], [self.event])
            with self.assertRaises(PermissionError):
                export_result(previous, self.exports, "样片")

        self.assertEqual(latest(self.work, self.source_hash), previous)
        self.assertEqual(list(previous[0].parent.iterdir()), [previous[0]])
        self.assertEqual(list(self.exports.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
