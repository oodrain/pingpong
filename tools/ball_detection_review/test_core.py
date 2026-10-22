"""Boundary, missingness and localization tests; no detector or event training."""
import json
from pathlib import Path
import tempfile
import unittest
import core


def row(i,detected=False,x=None,y=None,confidence=None,**extra):
    return dict(frame_id=i,detected=detected,x=x,y=y,confidence=confidence,**extra)


def ann():
    value=dict(frame_width=512,frame_height=288,num_frames=5,
               frames=[dict(frame_id=i,annotations=([dict(label='ball',bbox_xyxy=[[10,10,14,14]])] if i!=3 else [])) for i in range(5)])
    value['_frames']={r['frame_id']:r for r in value['frames']}
    return value


class ContractTests(unittest.TestCase):
    def test_exact_frame_alignment_and_absent_prediction_are_not_silent_misses(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(core.tracks(temp,'x',2),(None,None))
            p=Path(temp)/'x.json';p.write_text(json.dumps([row(1)]))
            with self.assertRaises(ValueError):core.tracks(temp,'x',1)

    def test_heatmap_margin_confidence_and_unknown_labels_are_distinct(self):
        rows=[row(0,True,15,12,.75),row(1,True,12,12,.74),row(2,True,90,90,.99),
              row(3,True,30,30,.99),row(4,missing_reason='tail_not_inferred')]
        result=core.evaluate(ann(),rows)
        self.assertEqual((result['matched'],result['low_confidence'],result['wrong_location'],result['tail_not_inferred']),(1,1,1,1))
        self.assertEqual(result['positive_frames'],4)
        self.assertEqual(result['unknown_frames'],1)
        self.assertEqual(result['false_points_on_reviewed_absent'],0)
        self.assertIsNone(result['complete_detector_f1'])

    def test_visible_annotation_is_excluded_only_by_explicit_review(self):
        rows=[row(i,True,12,12,.9) for i in range(5)]
        result=core.evaluate(ann(),rows,{'0':{'visibility':'occluded'},'1':{'visibility':'uncertain'},'3':{'visibility':'out'}})
        self.assertEqual(result['positive_frames'],2)
        self.assertEqual(result['false_points_on_reviewed_absent'],1)

    def test_detected_nan_rejected_and_low_confidence_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'x.json';p.write_text(json.dumps([row(0,True,float('nan'),2,.9)]))
            with self.assertRaises(ValueError):core.tracks(temp,'x',1)
        self.assertFalse(core.trusted(row(0,True,1,1,.74)))
        self.assertFalse(core.trusted(row(0,True,1,1,.9,in_image=False)))

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):core.data_path(temp,'../unrelated.json')


if __name__=='__main__':unittest.main()
