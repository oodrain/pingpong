"""Multi-contact reviews, revision conflicts and exact sequential frame identity."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import cv2
import numpy as np
from annotation_core import write_json
from auxiliary_event_core import ReviewJournal,blank,sha256
from auxiliary_event_review import AuxiliaryWindow,SequentialFrames,ensure_review_pack

class AuxiliaryReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.video=self.root/'video.mp4';writer=cv2.VideoWriter(str(self.video),cv2.VideoWriter_fourcc(*'mp4v'),25,(128,72))
        self.assertTrue(writer.isOpened())
        for fid in range(48):writer.write(np.full((72,128,3),fid*4,np.uint8))
        writer.release();self.tracks=self.root/'tracks.jsonl'
        self.tracks.write_text(''.join(json.dumps({'frame_id':f,'detected':True,'confidence':.99,'x':64.,'y':36.})+'\n' for f in range(48)),encoding='utf-8')
        cases=[{'case_id':f'v_{f}','video_id':'v','seed_frame':f,'angle':'unknown','video_path':str(self.video),'video_sha256':sha256(self.video),
            'tracks_path':str(self.tracks),'tracks_sha256':sha256(self.tracks),'width':128,'height':72,'n_frames':48,'default_scope':[0,47],'priority':0} for f in (3,5)]
        self.pack=self.root/'review_pack.json';write_json(self.pack,{'schema_version':1,'sources':{},'cases':cases})
        self.store=ReviewJournal(self.pack)
    def window(self):
        w=AuxiliaryWindow(self.pack);self.addCleanup(lambda:w.reader.close());return w
    def complete_video_pack(self):
        pack=json.loads(self.pack.read_text(encoding='utf-8'));pack['collect_other_subtypes']=True
        for case in pack['cases']:
            case.update(review_mode='complete_video',seed_frame=0,
                reference_events=[{'event_id':'prior_human_bounce','event_type':'table_bounce','interval_frames':[3,4],
                                   'reference_source_case_ids':['previous_review']}])
        write_json(self.pack,pack)
    def test_complete_video_requires_all_frames_and_preserves_reference_as_unconfirmed_draft(self):
        self.complete_video_pack();w=self.window()
        self.assertEqual(w.state['status'],'draft');self.assertEqual(w.state['events'][0]['interval_frames'],[3,4])
        self.assertFalse(self.store.journal.exists());self.assertIsNone(w.state['primary_event_id'])
        w.move(47)
        with self.assertRaisesRegex(ValueError,'未浏览'):w.action('complete')
        self.assertFalse(self.store.journal.exists())
        for f in range(48):w.move(f)
        w.action('complete');data=json.loads(w.store.export().read_text(encoding='utf-8'))
        self.assertFalse(data['review_scope_is_candidate_windows_not_full_video_gt']);self.assertEqual(data['complete_video_review_count'],1)
        self.assertTrue(data['records'][0]['state']['all_events_reviewed'])
    def test_complete_video_unknown_span_survives_restart_and_delete_without_labeling_background(self):
        self.complete_video_pack();w=self.window();w.move(10);w.action('start');w.move(13);w.action('end');w.action('uncertain')
        before=w.store.journal.read_bytes();loaded=self.window()
        self.assertEqual(loaded.state['unknown_intervals'],[[10,13]]);self.assertFalse(loaded.state['no_events_in_scope'])
        loaded.action('unknown:10:13');loaded.action('delete')
        self.assertEqual(loaded.state['unknown_intervals'],[]);self.assertTrue(loaded.store.journal.read_bytes().startswith(before))
    def test_complete_video_unknown_cannot_overlap_reference_event_or_mean_no_events(self):
        self.complete_video_pack();w=self.window();w.move(3)
        with self.assertRaisesRegex(ValueError,'重叠'):w.action('uncertain')
        self.assertFalse(w.store.journal.exists());self.assertEqual(w.state['unknown_intervals'],[])
        w.action('event:prior_human_bounce');w.action('delete');w.move(10);w.action('mode:frame');w.action('uncertain')
        for f in range(48):w.move(f)
        with self.assertRaisesRegex(ValueError,'不能声明'):w.action('no_events')
        w.action('complete');self.assertFalse(w.state['no_events_in_scope']);self.assertEqual(w.state['unknown_intervals'],[[10,10]])
    def test_complete_video_scope_is_fixed_and_next_unseen_does_not_mark_skipped_frames(self):
        self.complete_video_pack();w=self.window();w.action('scope');self.assertEqual(w.scope,[0,47])
        w.move(5);w.action('next_unseen');self.assertEqual(w.frame,6);self.assertNotIn(1,w.visited)
        w.scope=[0,20]
        with self.assertRaisesRegex(ValueError,'完整视频'):w.save()
    def test_prior_human_coverage_is_reused_but_missing_frames_still_block_completion(self):
        self.complete_video_pack();pack=json.loads(self.pack.read_text(encoding='utf-8'))
        pack['prior_human_coverage_reused']=True
        for case in pack['cases']:case['reference_visited_frames']=list(range(10,40))
        write_json(self.pack,pack);w=self.window()
        self.assertTrue(set(range(10,40))<=w.visited);self.assertEqual(w.state['events'][0]['event_id'],'prior_human_bounce')
        self.assertFalse(w.store.journal.exists())
        with self.assertRaisesRegex(ValueError,'未浏览'):w.action('complete')
        for f in list(range(10))+list(range(40,48)):w.move(f)
        w.action('complete');self.assertEqual(w.state['status'],'completed')
        loaded=self.window();loaded.select(0);self.assertEqual(len(loaded.visited),48)
    def test_external_coverage_keeps_latest_user_edits_and_never_rewrites_history_on_load(self):
        self.complete_video_pack();reference=json.loads(self.pack.read_text(encoding='utf-8'))
        for case in reference['cases']:case['reference_visited_frames']=list(range(10,40))
        path=self.root/'prior_coverage_pack.json';write_json(path,reference)
        w=self.window();w.action('event:prior_human_bounce');w.action('delete')
        w.action('new');w.move(8);w.action('type:racket_contact');w.action('add')
        before=w.store.journal.read_bytes();loaded=AuxiliaryWindow(self.pack,path);self.addCleanup(loaded.reader.close)
        self.assertEqual([e['event_type'] for e in loaded.state['events']],['racket_contact'])
        self.assertTrue(set(range(10,40))<=loaded.visited);self.assertEqual(loaded.store.journal.read_bytes(),before)
        loaded.action('save');self.assertTrue(loaded.store.journal.read_bytes().startswith(before))
    def test_optional_other_subtype_survives_save_restart_and_keeps_old_reviews_valid(self):
        pack=json.loads(self.pack.read_text(encoding='utf-8'));pack['collect_other_subtypes']=True
        write_json(self.pack,pack);w=self.window();w.action('type:other');w.action('other_kind');w.action('add');w.action('complete')
        w.select(0);self.assertEqual(w.state['events'][0]['other_kind'],'net_collision')
        restored=self.window();restored.select(0);self.assertEqual(restored.other_kind,'net_collision')
        restored.action('event:'+restored.state['events'][0]['event_id']);restored.action('other_kind');restored.action('update')
        self.assertEqual(restored.state['events'][0]['other_kind'],'net_graze');self.assertEqual(restored.state['status'],'draft')
    def test_missing_custom_review_never_creates_a_public_pack(self):
        target=self.root/'missing_train_pack.json'
        with patch('auxiliary_event_review.prepare_pack') as prepare:
            with self.assertRaises(FileNotFoundError):ensure_review_pack(target)
            prepare.assert_not_called();self.assertFalse(target.exists())
    def test_train_review_does_not_open_public_development_analysis(self):
        pack=json.loads(self.pack.read_text(encoding='utf-8'));pack['purpose']='train_only_event_labels'
        write_json(self.pack,pack);w=self.window()
        with patch('tt3d_review_analysis.open_analysis') as analysis:
            w.action('analysis');analysis.assert_not_called()
        self.assertIn('训练集',w.message)
    def test_sequential_decoder_preserves_indices_and_times_after_cache_eviction(self):
        reader=SequentialFrames(self.video,48);self.addCleanup(reader.close)
        for f in (0,40,1,47,2):
            image,t=reader.get(f)
            self.assertLess(abs(float(image.mean())-f*4),4)
            self.assertAlmostEqual(t,f/25,places=3)
        with self.assertRaises(ValueError):reader.get(48)
    def test_two_events_and_frame_interval_survive_completion_and_restart(self):
        w=self.window();w.action('type:table_bounce');w.action('mode:between');w.action('add')
        first=w.state['events'][0]['event_id'];w.move(5);w.action('new');w.action('type:racket_contact');w.action('add');w.action('complete')
        loaded=self.window();self.assertEqual(loaded.index,1)
        # Startup skips the completed case; explicitly return to it to verify recovery.
        loaded.select(0);self.assertEqual(loaded.state['status'],'completed');self.assertEqual(loaded.state['primary_event_id'],first)
        self.assertEqual([(e['event_type'],e['interval_frames']) for e in loaded.state['events']],[('table_bounce',[3,4]),('racket_contact',[5,5])])
        self.assertEqual(json.loads(self.store.export().read_text())['completed_cases'],1)
    def test_edits_and_deletion_append_history_and_invalidate_completion(self):
        w=self.window();w.action('type:table_bounce');w.action('add');w.action('complete')
        original=self.store.journal.read_bytes();w.move(4);w.action('update')
        self.assertTrue(self.store.journal.read_bytes().startswith(original));self.assertEqual(w.state['status'],'draft')
        self.assertEqual(w.state['events'][0]['interval_frames'],[4,4])
        w.action('delete');self.assertEqual(w.state['events'],[])
        before=self.store.journal.read_bytes()
        with self.assertRaises(ValueError):w.action('complete')
        self.assertEqual(self.store.journal.read_bytes(),before)
    def test_uncertain_is_not_exported_as_no_contact_or_false_negative(self):
        w=self.window();w.action('uncertain');w.action('complete')
        export=json.loads(self.store.export().read_text())
        self.assertEqual(export['records'][0]['candidate_label'],'uncertain')
        self.assertEqual(export['records'][1]['status'],'unreviewed')
        self.assertEqual(export['completed_events_exact_deduplicated'],[])
        self.assertTrue(export['review_scope_is_candidate_windows_not_full_video_gt'])
    def test_overlapping_cases_deduplicate_exact_events_and_keep_origins(self):
        for index in range(2):
            w=self.window();w.select(index);w.move(4);w.action('type:racket_contact');w.action('add');w.action('complete')
        export=json.loads(self.store.export().read_text());events=export['completed_events_exact_deduplicated']
        self.assertEqual(len(events),1);self.assertEqual(events[0]['source_case_ids'],['v_3','v_5'])
        self.assertEqual(export['completed_cases'],2)
    def test_duplicate_and_invalid_between_frame_cannot_erase_saved_event(self):
        w=self.window();w.action('type:table_bounce');w.action('add');before=self.store.journal.read_bytes()
        with self.assertRaises(ValueError):w.action('add')
        self.assertEqual(len(w.state['events']),1);self.assertEqual(self.store.journal.read_bytes(),before)
        w.move(47);w.action('mode:between')
        with self.assertRaises(ValueError):w.action('add')
        self.assertEqual(self.store.journal.read_bytes(),before)
    def test_concurrent_revision_and_changed_source_refuse_overwrite(self):
        a=self.window();b=self.window();a.action('type:table_bounce');a.action('add')
        original=self.store.journal.read_bytes();b.action('type:racket_contact')
        with self.assertRaisesRegex(ValueError,'另一窗口'):b.action('add')
        self.assertEqual(b.state['events'],[]);self.assertEqual(self.store.journal.read_bytes(),original)
        self.tracks.write_text('{}\n',encoding='utf-8')
        with self.assertRaises(ValueError):a.save()
        self.assertEqual(self.store.journal.read_bytes(),original)
    def test_render_and_draft_editor_restore_without_automatic_label(self):
        w=self.window();self.assertEqual(w.draw().shape,(900,1440,3));self.assertFalse(self.store.journal.exists())
        w.action('type:racket_contact');w.action('start');w.move(7);w.action('end');w.action('save')
        loaded=self.window();self.assertEqual(loaded.event_type,'racket_contact');self.assertEqual(loaded.interval(),[3,7])
        self.assertEqual(loaded.state['events'],[]);self.assertIsNone(loaded.state['primary_event_id'])
        self.assertEqual(loaded.draw().shape,(900,1440,3))
        self.assertIn('complete',[a for _,a in loaded.buttons])
    def test_return_from_analysis_reloads_reviews_saved_in_nested_viewer(self):
        parent=self.window();parent.action('type:table_bounce');parent.action('add')
        def nested_review(**kwargs):
            child=self.window();child.action('new');child.move(5);child.action('type:racket_contact');child.action('add')
        with patch('tt3d_review_analysis.open_analysis',side_effect=nested_review),patch('auxiliary_event_review.close_window'),patch('auxiliary_event_review.cv2.namedWindow'),patch('auxiliary_event_review.cv2.setMouseCallback'):
            parent.action('analysis')
        self.assertEqual(len(parent.state['events']),2)
        parent.save()  # Uses the nested viewer's latest revision, not a stale revision.
    def test_no_events_needs_full_inspection_and_exports_separately_from_uncertain(self):
        w=self.window()
        with self.assertRaises(ValueError):w.action('no_events')
        self.assertFalse(self.store.journal.exists())
        for f in range(48):w.move(f)
        w.action('no_events');w.action('complete')
        export=json.loads(self.store.export().read_text())
        self.assertEqual(export['records'][0]['candidate_label'],'no_event')
        self.assertEqual(export['records'][0]['state']['review_scope'],[0,47])
        self.assertEqual(export['completed_events_exact_deduplicated'],[])
        restored=self.window();restored.select(0);self.assertTrue(restored.state['no_events_in_scope'])
    def test_no_events_cannot_erase_existing_event_and_is_reset_when_scope_changes(self):
        w=self.window();w.action('type:table_bounce');w.action('add')
        before=self.store.journal.read_bytes()
        with self.assertRaises(ValueError):w.action('no_events')
        self.assertEqual(self.store.journal.read_bytes(),before)
        w.action('delete');w.scope=[0,10]
        for f in range(11):w.move(f)
        w.action('no_events');w.action('complete');w.action('scope')
        self.assertEqual(w.scope,[0,47]);self.assertFalse(w.state['no_events_in_scope']);self.assertEqual(w.state['status'],'draft')
    def test_new_event_or_uncertainty_replaces_negative_judgment_with_history_preserved(self):
        w=self.window()
        for f in range(48):w.move(f)
        w.action('no_events');before=self.store.journal.read_bytes()
        w.action('uncertain');self.assertFalse(w.state['no_events_in_scope']);self.assertTrue(w.state['candidate_uncertain'])
        w.action('no_events');w.action('type:racket_contact');w.action('add')
        self.assertFalse(w.state['no_events_in_scope']);self.assertEqual(len(w.state['events']),1)
        self.assertTrue(self.store.journal.read_bytes().startswith(before))

if __name__=='__main__':unittest.main()
