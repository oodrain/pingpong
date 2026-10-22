import unittest
from unittest.mock import patch
import cv2
import numpy as np

from development.bounce.ge_rule_v2 import replay
from development.detection.auto_table_geometry import (
    AmbiguousPlane, AutoGeometryTimeline, CausalGeometryTracker, GeometryConfig, extract_quad,
    validate_quad_observation)


def scene():
    image=np.zeros((360,640,3),np.uint8)
    q=np.array([[190,100],[450,110],[530,290],[100,280]],np.int32)
    mask=np.zeros(image.shape[:2],np.uint8)
    cv2.fillConvexPoly(mask,q,1)
    cv2.fillConvexPoly(image,q,(110,65,15))
    cv2.polylines(image,[q],True,(250,250,250),3)
    return image,mask.astype(float),q


class AutoGeometryTests(unittest.TestCase):
    def test_starting_vertex_flip_does_not_mix_distinct_table_corners(self):
        q=np.array([[200,100],[400,200],[300,400],[100,200]],float)
        image=np.zeros((480,640,3),np.uint8)
        cv2.polylines(image,[q.astype(np.int32)],True,(255,255,255),3)
        tracker=CausalGeometryTracker()
        proposal=dict(usable_proposal=True,reason='test_visible_border',corners=q.tolist())
        for f in (0,2,4):tracker.update(f,image,proposal)
        baseline=np.array(tracker.last_good['corners'])
        moved=q.copy();moved[3,1]+=1
        result=tracker.update(5,image,dict(proposal,corners=moved.tolist()))
        self.assertEqual(result['state'],'stable')
        self.assertLess(np.max(np.linalg.norm(np.array(result['corners'])-baseline,axis=1)),2)

    def test_clothing_diagonal_cannot_certify_a_clipped_corner(self):
        image=np.zeros((360,640,3),np.uint8)
        true=np.array([[250,136],[362,136],[371,206],[239,206]],np.int32)
        cv2.fillConvexPoly(image,true,(100,60,20))
        cv2.polylines(image,[true],True,(255,255,255),2)
        # A player masks part of the left side and adds a ragged clothing edge.
        cv2.fillPoly(image,[np.array([[250,153],[269,153],[292,206],[250,206]])],(20,20,200))
        clipped=[[250,136],[362,136],[371,206],[285,206]]
        seed=dict(usable_proposal=True,reason='segmentation_line_fit',corners=clipped,
                  mask_iou=.87,mask_coverage=.94,edge_support=.96,quality=.9)
        result=validate_quad_observation(image,seed)
        if result['usable_proposal']:
            self.assertLess(np.max(np.linalg.norm(np.array(result['corners'])-true,axis=1)),12,result)
        else:
            self.assertEqual(result['reason'],'unsupported_straight_border')

    def test_dark_straight_border_does_not_require_white_paint(self):
        image,mask,q=scene()
        cv2.line(image,tuple(q[1]),tuple(q[2]),(5,5,5),5)
        seed=dict(usable_proposal=True,reason='segmentation_line_fit',corners=q.tolist())
        result=validate_quad_observation(image,seed)
        self.assertTrue(result['usable_proposal'],result)

    def test_failed_observation_breaks_history_even_with_a_held_plane(self):
        image,mask,_=scene();proposal=extract_quad(image,mask)
        tracker=CausalGeometryTracker()
        for f in (0,2,4):tracker.update(f,image,proposal)
        missing=dict(usable_proposal=False,reason='occluded',corners=None)
        self.assertEqual(tracker.update(5,image,missing)['state'],'held')
        self.assertEqual(tracker.update(6,image,proposal)['state'],'held')
        self.assertEqual(tracker.update(7,image,proposal)['state'],'held')
        self.assertEqual(tracker.update(8,image,proposal)['state'],'stable')

    def test_large_canvas_cannot_hide_table_corner_motion(self):
        image,mask,_=scene();proposal=extract_quad(image,mask)
        canvas=np.zeros((1080,1920,3),np.uint8);canvas[:360,:640]=image
        tracker=CausalGeometryTracker()
        for f in (0,2,4):tracker.update(f,canvas,proposal)
        moved=dict(proposal,corners=(np.array(proposal['corners'])+[25,0]).tolist())
        self.assertFalse(tracker.update(5,canvas,moved)['usable'])

    def test_held_perimeter_requires_current_image_evidence(self):
        image,mask,_=scene();proposal=extract_quad(image,mask)
        tracker=CausalGeometryTracker()
        for f in (0,2,4):tracker.update(f,image,proposal)
        missing=dict(usable_proposal=False,reason='occluded',corners=None)
        result=tracker.update(5,np.zeros_like(image),missing)
        self.assertFalse(result['usable'])
        self.assertEqual(result['reason'],'held_perimeter_unverified')

    def test_contradictory_rgb_border_invalidates_rejected_mask_hold(self):
        image,mask,_=scene();proposal=extract_quad(image,mask)
        tracker=CausalGeometryTracker()
        for f in (0,2,4):tracker.update(f,image,proposal)
        moved=dict(proposal,usable_proposal=False,reason='insufficient_mask_overlap',
                   corners=(np.array(proposal['corners'])+[25,0]).tolist(),
                   image_perimeter=dict(straight_span_by_side=[1,1,1,1]))
        self.assertFalse(tracker.update(5,image,moved)['usable'])

    def test_small_line_offset_at_acute_corner_uses_bounded_hull(self):
        q=np.array([[90,200],[430,100],[580,105],[230,245]],np.int32)
        image=np.zeros((360,640,3),np.uint8)
        mask=np.zeros((360,640),np.uint8)
        cv2.fillConvexPoly(mask,q,1)
        cv2.fillConvexPoly(image,q,(110,65,15))
        cv2.polylines(image,[q],True,(250,250,250),3)
        calls=[0]
        def fitted_line(*args):
            side=calls[0]%4;calls[0]+=1
            a=q[side].astype(float)
            unit=q[(side+1)%4]-a;unit=unit/np.linalg.norm(unit)
            origin=a+np.array([-unit[1],unit[0]])*(7 if side==1 else 0)
            return np.array([*unit,*origin],np.float32)
        with patch('cv2.fitLine',side_effect=fitted_line):
            found=extract_quad(image,mask.astype(float))
        self.assertTrue(found['usable_proposal'],found)
        self.assertEqual(found['fitting_source'],'segmentation_hull_fallback')
        self.assertLess(np.max(np.linalg.norm(np.array(found['corners'])-q,axis=1)),3)

    def test_unrelated_fitted_edges_cannot_rotate_mask_sides(self):
        image,mask,q=scene()
        # Simulate nearby net/player edges dominating the line fit.
        with patch('cv2.fitLine',return_value=np.array([0.,1.,300.,180.],np.float32)):
            found=extract_quad(image,mask)
        self.assertTrue(found['usable_proposal'],found)
        self.assertLess(np.max(np.linalg.norm(np.array(found['corners'])-q,axis=1)),8)
        self.assertGreater(found['rejection_counts']['line_fit_reverted_to_hull'],0)

    def test_hull_fallback_still_requires_visible_image_borders(self):
        image,mask,_=scene()
        with patch('cv2.fitLine',return_value=np.array([0.,1.,300.,180.],np.float32)):
            found=extract_quad(np.zeros_like(image),mask)
        self.assertFalse(found['usable_proposal'])
        self.assertEqual(found['reason'],'insufficient_image_border')

    def test_visible_outer_border_survives_missing_interior_segmentation(self):
        image,mask,q=scene()
        mask[140:,230:400]=0
        found=extract_quad(image,mask)
        self.assertTrue(found['usable_proposal'],found)
        self.assertEqual(found['reason'],'occlusion_supported_border')
        self.assertLess(np.max(np.linalg.norm(np.array(found['corners'])-q,axis=1)),8)

    def test_clipped_mask_cannot_silently_certify_a_remote_corner(self):
        image,mask,q=scene()
        mask[:,440:]=0
        found=extract_quad(image,mask)
        if found['usable_proposal']:
            self.assertLess(np.max(np.linalg.norm(np.array(found['corners'])-q,axis=1)),12,found)
        else:
            self.assertIn(found['reason'],('insufficient_image_border','unsupported_border_continuation',
                                          'ambiguous_outer_border','no_supported_quad'))

    def test_internal_net_line_cannot_replace_missing_image_perimeter(self):
        image,mask,_=scene()
        image[:]=0
        cv2.line(image,(140,180),(480,185),(255,255,255),3)
        found=extract_quad(image,mask)
        self.assertFalse(found['usable_proposal'],found)

    def test_segmentation_guided_corners_on_known_perspective(self):
        image,mask,q=scene()
        found=extract_quad(image,mask)
        self.assertTrue(found['usable_proposal'],found)
        self.assertLess(np.max(np.linalg.norm(np.array(found['corners'])-q,axis=1)),8)

    def test_empty_mask_and_nonfinite_mask(self):
        image,mask,_=scene()
        self.assertFalse(extract_quad(image,np.zeros_like(mask))['usable_proposal'])
        with self.assertRaises(ValueError):
            extract_quad(image,mask*np.nan)

    def test_short_occlusion_hold_then_expiry(self):
        image,mask,_=scene()
        proposal=extract_quad(image,mask)
        tracker=CausalGeometryTracker(GeometryConfig(maximum_hold_frames=20))
        states=[tracker.update(f,image,proposal) for f in (0,10,20)]
        self.assertEqual([s['usable'] for s in states],[False,False,True])
        missing={'usable_proposal':False,'reason':'occluded','corners':None}
        held=tracker.update(30,image,missing)
        self.assertEqual(held['state'],'held')
        self.assertFalse(tracker.update(50,image,missing)['usable'])

    def test_camera_cut_clears_old_plane(self):
        image,mask,_=scene()
        proposal=extract_quad(image,mask)
        tracker=CausalGeometryTracker()
        for f in (0,10,20):tracker.update(f,image,proposal)
        result=tracker.update(30,np.full_like(image,255),{'usable_proposal':False,'reason':'empty','corners':None})
        self.assertFalse(result['usable'])
        self.assertEqual(result['epoch'],1)
        self.assertEqual(result['reason'],'camera_cut')

    def test_missing_observation_breaks_initial_warmup(self):
        image,mask,_=scene();proposal=extract_quad(image,mask)
        tracker=CausalGeometryTracker()
        tracker.update(0,image,proposal)
        tracker.update(10,image,{'usable_proposal':False,'reason':'occluded','corners':None})
        self.assertFalse(tracker.update(20,image,proposal)['usable'])
        self.assertFalse(tracker.update(30,image,proposal)['usable'])
        self.assertTrue(tracker.update(40,image,proposal)['usable'])

    def test_expired_history_requires_new_warmup(self):
        image,mask,_=scene();proposal=extract_quad(image,mask)
        tracker=CausalGeometryTracker()
        for f in (0,10,20):tracker.update(f,image,proposal)
        self.assertFalse(tracker.update(50,image,proposal)['usable'])
        self.assertFalse(tracker.update(60,image,proposal)['usable'])
        self.assertTrue(tracker.update(70,image,proposal)['usable'])

    def test_corner_jitter_is_not_a_camera_cut(self):
        image,mask,_=scene();proposal=extract_quad(image,mask)
        tracker=CausalGeometryTracker()
        states=[tracker.update(f,image,proposal) for f in (0,10,20)]
        moved=dict(proposal,corners=(np.array(proposal['corners'])+[100,0]).tolist())
        states.append(tracker.update(30,image,moved))
        timeline=AutoGeometryTimeline(states)
        self.assertIsNone(timeline.lookup(30))
        self.assertEqual(timeline.interval_id(20),timeline.interval_id(30))
        self.assertNotEqual(states[2]['epoch'],states[3]['epoch'])

    def test_timeline_cannot_use_future_or_expired_samples(self):
        image,mask,_=scene()
        tracker=CausalGeometryTracker()
        proposal=extract_quad(image,mask)
        timeline=AutoGeometryTimeline([tracker.update(f,image,proposal) for f in (0,10,20)])
        self.assertIsNone(timeline.lookup(19))
        self.assertIsNotNone(timeline.lookup(20))
        self.assertIsNone(timeline.lookup(41))

    def test_ambiguous_axes_require_both_assignments_outside(self):
        plane=AmbiguousPlane([[0,0],[100,0],[100,50],[0,50]],'test')
        for x,y in ((0,0),(100,0),(100,50),(0,50)):
            self.assertAlmostEqual(plane.outside_distance(x,y),0,places=4)
        self.assertAlmostEqual(plane.outside_distance(130,25),457.5,places=4)

    def test_axis_ambiguity_cannot_enable_smart_spatial_dedup(self):
        q=[[0,0],[100,0],[100,50],[0,50]]
        sample={'frame_id':0,'valid_until':30,'epoch':0,'usable':True,'corners':q}
        raw=[{'input_frame_id':i+1,'detected':True,'confidence':.9,'x':x,'y':y}
             for i,(x,y) in enumerate([(10,0),(10,5),(10,0),(90,0),(90,5),(90,0)])]
        pred,_=replay(raw,AutoGeometryTimeline([sample]),'v1b')
        self.assertEqual([p['frame_id'] for p in pred],[1])


if __name__=='__main__':unittest.main()
