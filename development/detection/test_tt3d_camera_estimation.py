import unittest
import cv2
import numpy as np
from development.detection.tt3d_camera_estimation import estimate_camera,object_corners


class CameraTests(unittest.TestCase):
    def test_known_perspective_camera_is_recovered(self):
        K=np.array([[1400.,0,960],[0,1400,540],[0,0,1]])
        q=cv2.projectPoints(object_corners('length'),np.array([.75,.15,.10]),np.array([.1,.2,5.]),K,np.zeros(5))[0].reshape(4,2)
        result=estimate_camera(q,(1920,1080))
        candidates=[c for c in result['candidates'] if c['usable']]
        self.assertTrue(candidates,result)
        best=min(candidates,key=lambda c:c['reprojection_rmse_px'])
        self.assertAlmostEqual(best['focal_px'],1400,delta=1.)
        self.assertLess(best['reprojection_rmse_px'],.01)
        self.assertGreater(best['camera_centre_m'][2],0)
        self.assertFalse(result['unique_physical_direction_verified'])

    def test_fronto_parallel_plane_cannot_observe_focal(self):
        K=np.array([[1400.,0,960],[0,1400,540],[0,0,1]])
        q=cv2.projectPoints(object_corners('width'),np.zeros(3),np.array([0.,0.,5.]),K,np.zeros(5))[0].reshape(4,2)
        result=estimate_camera(q,(1920,1080))
        self.assertFalse(result['usable'],result)

    def test_degenerate_and_nonfinite_corners_are_rejected(self):
        for q in ([[0,0],[1,0],[2,0],[3,0]],[[0,0],[1,0],[1,1],[0,float('nan')]]):
            with self.assertRaises(ValueError):estimate_camera(q,(1920,1080))


if __name__=='__main__':unittest.main()
