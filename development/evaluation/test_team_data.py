"""Safety contracts: extra positives, unknown scope and source isolation."""
import unittest
from team_data import matching,preserve,scoped_score,validate_source_use

class Safety(unittest.TestCase):
 def test_outside_scope_positive_cannot_be_deleted(self):
  old={'v':[dict(frame_id=19),dict(frame_id=29)]};pos={'v':[dict(video_id='v',interval_frames=[19,19]),dict(video_id='v',interval_frames=[29,29])]}
  self.assertFalse(preserve(old,{'v':[dict(frame_id=29)]},pos)['passed'])
 def test_one_prediction_does_not_cover_two_events(self):
  self.assertEqual(len(matching([dict(video_id='v',frame_id=20)],[dict(video_id='v',interval_frames=[19,19]),dict(video_id='v',interval_frames=[21,21])])),1)
 def test_unknown_output_is_not_scored_false_positive(self):
  s=[dict(video_id='v',complete_video=False,evaluation_ranges=[[10,12]],full_evaluation_ranges=[],events=[dict(event_type='table_bounce',interval_frames=[11,11])])]
  result=scoped_score({'v':[dict(frame_id=11),dict(frame_id=30)]},s,'trusted')
  self.assertEqual((result['tp'],result['fp'],result['fn']),(1,0,0))
 def test_missing_candidate_video_fails(self):
  with self.assertRaises(ValueError):preserve({'v':[]},{},{'v':[]})
 def test_protected_and_public_sources_not_selectable(self):
  m=dict(videos=[dict(source_group='00',role='development'),dict(source_group='09',role='reserved'),dict(source_group='public',role='public_regression')])
  for bad in ('09','public'):
   with self.assertRaises(ValueError):validate_source_use(m,['00'],[bad],[])
 def test_source_overlap_rejected(self):
  with self.assertRaises(ValueError):validate_source_use(dict(videos=[]),['00'],['00'],[])
 def test_original_preserved(self):
  p={'v':[dict(frame_id=19)]};e={'v':[dict(video_id='v',interval_frames=[19,19])]}
  self.assertTrue(preserve(p,p,e)['passed'])
if __name__=='__main__':unittest.main()
