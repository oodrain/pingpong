import unittest
import cv2
import numpy as np
from development.detection.tt3d_net_calibration import estimate_net_camera,estimate_net_sequence,infer_net_axis,net_object_points
from development.detection.auto_table_geometry import order_quad


class NetCalibrationTests(unittest.TestCase):
    def fixture(self,index=0):
        K=np.array([[1400.,0,960],[0,1400,540],[0,0,1]])
        p=cv2.projectPoints(net_object_points('width'),np.array([2.4+.03*index,.12,.08]),np.array([.1+.05*index,.2,5.+.1*index]),K,np.zeros(5))[0].reshape(-1,2)
        q=order_quad(p[:4])
        # Determine correspondence from projected object corners, not side
        # lengths under perspective (a long physical edge can look shorter).
        first=int(np.argmin(np.linalg.norm(p[:4]-q[0],axis=1)))
        second=int(np.argmin(np.linalg.norm(p[:4]-q[1],axis=1)))
        axis='width' if {first,second} in ({0,1},{2,3}) else 'length'
        return q,p[4:],axis

    def test_known_camera_and_net_height_recover_focal(self):
        q,n,axis=self.fixture();result=estimate_net_camera(q,(1920,1080),n,axis)
        self.assertTrue(result['usable'],result)
        self.assertAlmostEqual(result['selected_camera']['focal_px'],1400,delta=1)
        self.assertLess(result['selected_camera']['reprojection_rmse_px'],.01)
        self.assertGreater(result['selected_camera']['camera_centre_m'][2],0)

    def test_endpoint_order_and_corner_winding_do_not_change_estimate(self):
        q,n,axis=self.fixture()
        a=estimate_net_camera(q,(1920,1080),n,axis)
        b=estimate_net_camera(q[::-1],(1920,1080),n[::-1],axis)
        self.assertAlmostEqual(a['selected_camera']['focal_px'],b['selected_camera']['focal_px'],delta=.01)

    def test_unknown_axis_is_not_silently_confirmed(self):
        q,n,axis=self.fixture();result=estimate_net_camera(q,(1920,1080),n)
        self.assertFalse(result['usable']);self.assertIsNone(result['selected_camera'])
        self.assertEqual(infer_net_axis(q,n)['suggested_edge01_axis'],axis)

    def test_wrong_human_axis_is_not_silently_replaced(self):
        q,n,axis=self.fixture();wrong='width' if axis=='length' else 'length'
        result=estimate_net_camera(q,(1920,1080),n,wrong)
        self.assertEqual(result['confirmed_edge01_axis'],wrong)
        self.assertFalse(result['usable'],result)

    def test_nonfinite_or_duplicate_net_points_are_rejected(self):
        q,n,axis=self.fixture()
        for invalid in ([[0,0],[float('nan'),0]],[[10,10],[10,10]]):
            with self.assertRaises(ValueError):estimate_net_camera(q,(1920,1080),invalid,axis)

    def test_three_reviewed_frames_share_the_known_focal(self):
        observations=[]
        for i in range(3):
            q,n,axis=self.fixture(i)
            observations.append(dict(frame_id=i*10,corners=q.tolist(),net_endpoints=n.tolist(),edge01_axis=axis))
        result=estimate_net_sequence(observations,(1920,1080))
        self.assertTrue(result['usable'],result)
        self.assertAlmostEqual(result['shared_focal_px'],1400,delta=1)
        self.assertEqual(len(result['poses']),3)

    def test_bad_frame_is_not_silently_dropped_from_joint_fit(self):
        q,n,axis=self.fixture()
        observations=[dict(frame_id=i*10,corners=q.tolist(),net_endpoints=n.tolist(),edge01_axis=axis) for i in range(3)]
        observations[-1]['net_endpoints']=(n+[0,120]).tolist()
        result=estimate_net_sequence(observations,(1920,1080))
        self.assertFalse(result['usable'],result);self.assertEqual(result['frames'],[0,10,20])

    def test_shared_fit_requires_three_unique_frames_and_confirmed_axes(self):
        q,n,axis=self.fixture();v=dict(frame_id=0,corners=q.tolist(),net_endpoints=n.tolist(),edge01_axis=axis)
        for observations in ([v,v],[v,v,v],[dict(v,frame_id=i,edge01_axis='unknown') for i in range(3)]):
            with self.assertRaises(ValueError):estimate_net_sequence(observations,(1920,1080))


if __name__=='__main__':unittest.main()
