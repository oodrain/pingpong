import unittest
import numpy as np
from development.evaluation.validate_table_geometry import corner_error
from development.evaluation.compare_window_auto_geometry import filter_candidates
from development.bounce.rule_variants import CONFIGS
from development.detection.auto_table_geometry import AutoGeometryTimeline


class ValidationTests(unittest.TestCase):
    def test_corner_error_handles_only_cycle_and_reflection(self):
        q=np.array([[10,10],[100,20],[130,80],[0,75]],float)
        for winding in (q,q[::-1]):
            for k in range(4):
                self.assertEqual(corner_error(q,np.roll(winding,k,axis=0))['maximum_corner_error_px'],0)
        self.assertGreater(corner_error(q,q[[0,2,1,3]])['mean_corner_error_px'],0)

    def test_window_candidates_fallback_and_geometry_rejection(self):
        cfg=next(c for c in CONFIGS if c.name=='V1_prom3_gapaware2')
        cs=[{'frame_id':f,'x':x,'y':25.,'prev_frame':f-1,'next_frame':f+1} for f,x in ((2,20),(5,90),(14,400))]
        fallback,_=filter_candidates(cs,AutoGeometryTimeline([]),cfg)
        self.assertEqual([p['frame_id'] for p in fallback],[2,14])
        sample={'frame_id':0,'valid_until':30,'usable':True,'epoch':0,'corners':[[0,0],[100,0],[100,50],[0,50]]}
        selected,ds=filter_candidates(cs,AutoGeometryTimeline([sample]),cfg)
        self.assertEqual([p['frame_id'] for p in selected],[2])
        self.assertEqual(ds[-1]['reason'],'outside_table_margin')


if __name__=='__main__':unittest.main()
