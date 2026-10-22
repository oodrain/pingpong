"""Full-clip reinspection must not resurrect corrected window labels."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT if (ROOT/'development/bounce/frame_event_tcn.py').exists() else ROOT/'worktrees/table-geometry-v2'
sys.path.insert(0,str(PROJECT/'experiments/tt3d_net_review_20261013'))
from build_frame_tcn_dataset import effective_reviews

class WholeSequenceDatasetTests(unittest.TestCase):
    def test_complete_reinspection_replaces_old_events_but_draft_does_not(self):
        old={'case_id':'old','video_id':'00_001','state':{'status':'completed','events':[{'interval_frames':[5,5]}]}}
        other={'case_id':'other','video_id':'06_001','state':{'status':'completed','events':[]}}
        full={'case_id':'full','video_id':'00_001','review_mode':'complete_video',
              'state':{'status':'draft','events':[]}}
        before=deepcopy([old,other,full]);used,stale=effective_reviews([old,other,full])
        self.assertEqual(used,[old,other]);self.assertEqual(stale,[])
        full['state']['status']='completed';used,stale=effective_reviews([old,other,full])
        self.assertEqual(used,[other,full]);self.assertEqual(stale,[old])
        self.assertEqual(old,before[0]);self.assertEqual(other,before[1])
    def test_two_complete_adjudications_require_reconciliation(self):
        full={'case_id':'full','video_id':'00_001','review_mode':'complete_video','state':{'status':'completed'}}
        second=deepcopy(full);second['case_id']='second'
        with self.assertRaisesRegex(ValueError,'reconciliation'):effective_reviews([full,second])

if __name__=='__main__':unittest.main()
