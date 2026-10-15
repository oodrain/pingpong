import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np

from development.evaluation.audit_table_reference_bounds import envelope_metrics,table_boxes
from development.evaluation.geometry_reference_review import validate_submission,latest_reviews,evaluate
from development.evaluation.run_auto_table_v2 import sha


class ReferenceReviewTests(unittest.TestCase):
    def setUp(self):
        self.case={'case_id':'train_test_10','group':'train','video_id':'00_001','frame_id':10,
                   'width':100,'height':100,'image_sha256':'image','video_sha256':'video'}
        self.cases={self.case['case_id']:self.case}
        self.reference={'case_id':self.case['case_id'],'reviewer':'test reviewer','status':'corners_visible',
                        'corners':[[10,10],[90,10],[90,90],[10,90]]}

    def test_reference_keeps_exact_frame_and_does_not_claim_double_review(self):
        r=validate_submission(self.reference,self.cases)
        self.assertEqual(r['frame_id'],10)
        self.assertEqual(r['reference_source'],'user_submitted_single_review_not_double_checked')

    def test_crossed_and_outside_corners_are_rejected(self):
        for q in ([[10,10],[90,90],[90,10],[10,90]],[[10,10],[100,10],[90,90],[10,90]]):
            with self.assertRaises(ValueError):validate_submission(dict(self.reference,corners=q),self.cases)

    def test_occluded_status_cannot_save_guessed_corners(self):
        with self.assertRaises(ValueError):validate_submission(dict(self.reference,status='occluded'),self.cases)
        r=validate_submission(dict(self.reference,status='occluded',corners=[]),self.cases)
        self.assertEqual(r['corners'],[])

    def test_reference_requires_reviewer_and_known_case(self):
        for changed in ({'reviewer':''},{'reviewer':7},{'case_id':'foreign'},{'status':'pending'}):
            with self.assertRaises(ValueError):validate_submission(dict(self.reference,**changed),self.cases)

    def test_changed_image_reference_is_not_silently_used(self):
        with tempfile.TemporaryDirectory() as directory:
            pack=Path(directory);r=validate_submission(self.reference,self.cases);r['image_sha256']='changed'
            (pack/'references.jsonl').write_text(json.dumps(r)+'\n',encoding='utf-8')
            with self.assertRaises(ValueError):latest_reviews(pack,{'cases':[self.case]})

    def test_saved_reference_roundtrip_scores_exact_frame_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pack=root/'pack';pack.mkdir();run=root/'run';run.mkdir()
            path=pack/'frame.png';cv2.imwrite(str(path),np.zeros((100,100,3),np.uint8))
            case=dict(self.case,image_sha256=sha(path),image_url='frame.png')
            sample={'group':'train','video_id':'00_001','frame_id':10,'valid_until':20,
                    'usable':True,'epoch':0,'corners':self.reference['corners']}
            (run/'geometry_samples.jsonl').write_text(json.dumps(sample)+'\n',encoding='utf-8')
            manifest={'cases':[case],'geometry_samples_sha256':sha(run/'geometry_samples.jsonl')}
            (pack/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            r=validate_submission(self.reference,{case['case_id']:case})
            (pack/'references.jsonl').write_text(json.dumps(r)+'\n',encoding='utf-8')
            evaluate(pack,run,root/'result')
            summary=json.loads((root/'result/summary.json').read_text(encoding='utf-8'))
            records=json.loads((root/'result/records.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['corner_comparisons'],1)
            self.assertEqual(summary['mean_corner_error_px'],0)
            self.assertFalse(summary['independent_double_review_verified'])
            self.assertEqual([r['frame_id'] for r in records],[10])

    def test_box_agreement_does_not_establish_corner_accuracy(self):
        # A wrong quadrilateral can have a perfect envelope.
        r=envelope_metrics([[0,0],[100,20],[100,100],[20,100]],[0,0,100,100])
        self.assertEqual(r['bbox_iou'],1)
        self.assertEqual(r['max_side_error_normalized'],0)

    def test_missing_and_multiple_table_boxes_are_preserved(self):
        self.assertEqual(table_boxes({'annotations':[{'label':'ball','bbox_xyxy':[[1,2,3,4]]}]}),[])
        bs=table_boxes({'annotations':[{'label':'table','bbox_xyxy':[[0,0,10,20],[30,40,50,60]]}]})
        self.assertEqual(len(bs),2)


if __name__=='__main__':unittest.main()
