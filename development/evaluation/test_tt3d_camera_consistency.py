import unittest
from experiments.tt3d_geometry_revision_20261013.estimate_cameras import summarize_hypothesis


def estimates(focals):
    return [(dict(frame_id=i,epoch=0,corners=[[0,0],[1,0],[1,1],[0,1]]),
             dict(candidates=[dict(edge01_axis='length',usable=True,focal_px=f)])) for i,f in enumerate(focals)]


class ConsistencyTests(unittest.TestCase):
    def test_one_good_fit_is_insufficient(self):
        self.assertFalse(summarize_hypothesis(estimates([1500]),'length')['usable'])

    def test_inconsistent_focals_are_rejected_despite_usable_individual_fits(self):
        result=summarize_hypothesis(estimates([700,1500,5000,1700,9000]),'length')
        self.assertFalse(result['usable'])
        self.assertEqual(result['reason'],'focal_inconsistent_between_frames')

    def test_consistent_focals_retain_a_real_representative_pose(self):
        result=summarize_hypothesis(estimates([1490,1500,1510,1480,1520]),'length')
        self.assertTrue(result['usable'])
        self.assertEqual(result['representative_camera']['focal_px'],1500)
        self.assertEqual(result['representative_frame'],1)


if __name__=='__main__':unittest.main()
