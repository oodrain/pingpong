"""Dense-event masking, true observations, temporal receptive field and assets."""
import ast,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];PROJECT=ROOT if (ROOT/'development/bounce/frame_event_tcn.py').exists() else ROOT/'worktrees/table-geometry-v2'
sys.path.insert(0,str(PROJECT))
from development.bounce.frame_event_features import automatic_frame_features,interval_targets,decode_frame_events,geometry_supported_touch_scores,FEATURES

Q=[[0,0],[100,0],[100,100],[0,100]]
def tracks(n=12):
    return [{'frame_id':i,'timestamp_s':i*.04,'detected':True,'in_image':True,'confidence':.95,'x':20.+i,'y':80.-2*i} for i in range(n)]


class FrameEvidenceTests(unittest.TestCase):
    def test_current_invalidation_and_future_geometry_never_backfill_features(self):
        rows=tracks();samples=[{'frame_id':0,'usable':True,'valid_until':11,'corners':Q},
                              {'frame_id':4,'usable':False},{'frame_id':8,'usable':True,'valid_until':11,'corners':Q}]
        x=automatic_frame_features(rows,samples,(128,128));index=FEATURES.index('table_available')
        self.assertTrue(np.all(x[:4,index]==1));self.assertTrue(np.all(x[4:8,index]==0));self.assertTrue(np.all(x[8:,index]==1))
        self.assertTrue(np.all(x[4:8,FEATURES.index('rectified_available')]==0))

    def test_missing_ball_is_explicit_and_velocity_does_not_bridge_the_gap(self):
        rows=tracks();rows[4].update(detected=False,confidence=None,x=None,y=None)
        x=automatic_frame_features(rows,[],(128,128))
        self.assertEqual(x[4,FEATURES.index('ball_available')],0)
        self.assertEqual(x[5,FEATURES.index('velocity_available')],0)
        self.assertEqual(x[6,FEATURES.index('acceleration_available')],0)
        self.assertTrue(np.isfinite(x).all())

    def test_actual_pts_change_velocity_and_large_gap_is_marked(self):
        rows=tracks();normal=automatic_frame_features(rows,[],(128,128))
        for row in rows[5:]:row['timestamp_s']+=.16
        changed=automatic_frame_features(rows,[],(128,128))
        self.assertEqual(changed[5,FEATURES.index('time_gap')],1)
        self.assertEqual(changed[5,FEATURES.index('velocity_available')],0)
        self.assertEqual(normal[5,FEATURES.index('velocity_available')],1)

    def test_nonmonotonic_pts_and_noncontiguous_indices_rejected(self):
        rows=tracks();rows[3]['timestamp_s']=rows[2]['timestamp_s']
        with self.assertRaises(ValueError):automatic_frame_features(rows,[],(128,128))
        rows=tracks();rows[3]['frame_id']=99
        with self.assertRaises(ValueError):automatic_frame_features(rows,[],(128,128))

    def test_unknown_frames_and_unvisited_background_are_masked(self):
        review={'state':{'status':'completed','candidate_uncertain':False,'review_scope':[2,9],'visited_frames':[2,3,4,6,7,8,9]}}
        y,mask=interval_targets(12,[review],[])
        self.assertTrue(mask[4].all());self.assertFalse(mask[5].any());self.assertTrue(mask[6:8].all())
        self.assertFalse(mask[:4].any());self.assertFalse(mask[8:].any());self.assertEqual(y.sum(),0)

    def test_bounce_and_hit_intervals_can_overlap_and_boundary_positive_is_retained(self):
        review={'state':{'status':'completed','candidate_uncertain':False,'review_scope':[2,9],'visited_frames':list(range(2,10))}}
        events=[{'event_type':'table_bounce','interval_frames':[2,3]}, {'event_type':'racket_contact','interval_frames':[3,3]}]
        y,mask=interval_targets(12,[review],events)
        self.assertEqual(y[3].tolist(),[1.,1.]);self.assertTrue(mask[3].all());self.assertTrue(mask[2,0])
        self.assertFalse(mask[1].any())

    def test_uncertain_review_is_not_a_negative_window(self):
        review={'state':{'status':'completed','candidate_uncertain':True,'review_scope':[0,11],'visited_frames':list(range(12))}}
        y,mask=interval_targets(12,[review],[]);self.assertFalse(mask.any());self.assertEqual(y.sum(),0)

    def test_occluded_span_and_shoulders_are_masked_but_explicit_nearby_hit_is_preserved(self):
        review={'state':{'status':'completed','review_scope':[0,19],'visited_frames':list(range(20)),
                         'unknown_intervals':[[8,10]]}}
        events=[{'event_type':'racket_contact','interval_frames':[12,12]}]
        y,mask=interval_targets(20,[review],events)
        self.assertFalse(mask[6:12].any());self.assertFalse(mask[12,0]);self.assertTrue(mask[12,1])
        self.assertEqual(y[12,1],1);self.assertTrue(mask[13:18].all())
        review['state']['unknown_intervals']=[[0,19]]
        y,mask=interval_targets(20,[review],[]);self.assertFalse(mask.any())

    def test_decoder_can_emit_a_frame_without_old_candidates_or_usable_table(self):
        rows=tracks();scores=np.zeros(12);scores[4]=.9
        result,decisions=decode_frame_events(scores,rows,.5)
        self.assertEqual(result,[{'frame_id':4,'x':24.,'y':72.}]);self.assertFalse(decisions[0]['observations_interpolated'])

    def test_missing_peak_uses_neighbouring_actual_point_or_rejects(self):
        rows=tracks();rows[4].update(detected=False,confidence=None,x=None,y=None);scores=np.zeros(12);scores[4]=.9;scores[3]=.3
        result,decisions=decode_frame_events(scores,rows,.5)
        self.assertEqual(result[0]['frame_id'],3);self.assertEqual(decisions[0]['peak_frame'],4)
        rows[3]['detected']=False;rows[5]['detected']=False
        result,decisions=decode_frame_events(scores,rows,.5);self.assertEqual(result,[]);self.assertFalse(decisions[0]['included'])

    def test_peak_separation_uses_time_and_nonfinite_scores_are_rejected(self):
        rows=tracks();scores=np.zeros(12);scores[3]=.8;scores[5]=.9
        result,_=decode_frame_events(scores,rows,.5,minimum_separation_s=.1);self.assertEqual(len(result),1)
        result,_=decode_frame_events(scores,rows,.5,minimum_separation_s=.06);self.assertEqual(len(result),2)
        scores[0]=np.nan
        with self.assertRaises(ValueError):decode_frame_events(scores,rows,.5)

    def test_optional_geometry_support_preserves_unknown_and_invalidated_table(self):
        x=automatic_frame_features(tracks(),[{'frame_id':0,'usable':True,'valid_until':11,'corners':Q}],(128,128))
        x[:,FEATURES.index('signed_projection_distance_fraction')]=-.1
        samples=[{'frame_id':0,'state':'stable'},{'frame_id':4,'state':'unknown'}]
        scores=np.ones(12,dtype=np.float32);adjusted=geometry_supported_touch_scores(scores,x,samples,'stable_polygon_soft')
        self.assertTrue(np.all(adjusted[:4]==.25));self.assertTrue(np.all(adjusted[4:]==1));self.assertTrue(np.all(scores==1))
        self.assertTrue(np.array_equal(geometry_supported_touch_scores(scores,x,samples),scores))


@unittest.skipUnless(importlib.util.find_spec('torch'),'Run model checks with .venv-table')
class TemporalModelTests(unittest.TestCase):
    def test_receptive_field_is_local_and_predictions_do_not_depend_on_other_video_length(self):
        import torch
        from development.bounce.frame_event_tcn import FrameEventTCN
        torch.manual_seed(3);model=FrameEventTCN();model.eval();x=torch.randn(1,len(FEATURES),81)
        with torch.no_grad():
            original=model(x)[:,:,40];changed=x.clone();changed[:,:,57:]+=20
            self.assertTrue(torch.allclose(original,model(changed)[:,:,40],atol=1e-6,rtol=0))
            local=model(x[:,:,25:56])[:,:,15]
            self.assertTrue(torch.allclose(original,local,atol=1e-6,rtol=0))
        self.assertEqual(model.receptive_field,31)

    def test_checkpoint_roundtrip_and_schema_mismatch(self):
        import torch
        from development.bounce.frame_event_tcn import FrameEventTCN,load,predict
        from development.bounce.frame_event_features import EVENT_TYPES
        model=FrameEventTCN();stats={'mean':[0.]*len(FEATURES),'scale':[1.]*len(FEATURES)}
        x=automatic_frame_features(tracks(),[],(128,128));expected=predict(model,x,stats)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'model.pt';checkpoint={'features':FEATURES,'event_types':list(EVENT_TYPES),'architecture':model.config,'normalization':stats,'state_dict':model.state_dict()}
            torch.save(checkpoint,path);restored,data=load(path);self.assertTrue(np.array_equal(expected,predict(restored,x,data['normalization'])))
            checkpoint['features']=list(reversed(FEATURES));torch.save(checkpoint,path)
            with self.assertRaises(ValueError):load(path)

    def test_training_normalization_only_uses_supervised_training_rows(self):
        from development.bounce.frame_event_tcn import normalization,normalize
        x=np.ones((10,len(FEATURES)),np.float32);x[5:]=999;mask=np.zeros((10,2),bool);mask[:5]=True
        stats=normalization([{'features':x,'loss_mask':mask}]);self.assertEqual(stats['mean'][1],1.)
        bad=dict(stats,scale=[0.]*len(FEATURES))
        with self.assertRaises(ValueError):normalize(x,bad)

    def test_runtime_modules_have_no_annotation_or_evaluation_imports(self):
        for name in ('frame_event_features.py','frame_event_tcn.py'):
            tree=ast.parse((PROJECT/'development/bounce'/name).read_text(encoding='utf-8'))
            modules=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            self.assertFalse(any(m and any(s in m for s in ('annotation','evaluation','review')) for m in modules))


if __name__=='__main__':unittest.main()
