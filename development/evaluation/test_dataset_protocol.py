import copy
import json
from pathlib import Path
import tempfile
import unittest

from dataset_protocol import check_model_for_formal_validation,load_protocol,select,sha,verify_inputs


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.manifest={'split_id':'v1','videos':[
            {'video_id':'00_001','source_group':'00','split':'train'},
            {'video_id':'00_002','source_group':'00','split':'train'},
            {'video_id':'01_001','source_group':'01','split':'validation_reserved'},
            {'video_id':'v000001','source_group':'public:1','split':'public_regression'}]}

    def test_training_rejects_validation_and_public(self):
        for vid in ('01_001','v000001'):
            with self.assertRaises(ValueError):select(self.manifest,[vid],'training')

    def test_threshold_selection_cannot_use_validation(self):
        with self.assertRaises(ValueError):select(self.manifest,['01_001'],'threshold_selection')

    def test_validation_cannot_score_a_training_video_as_holdout(self):
        with self.assertRaises(ValueError):select(self.manifest,['00_001'],'validation')
        self.assertEqual(select(self.manifest,['01_001'],'validation')[0]['source_group'],'01')

    def test_empty_duplicate_and_unknown_ids_rejected(self):
        for ids in ([],['00_001','00_001'],['missing']):
            with self.assertRaises(ValueError):select(self.manifest,ids,'training')

    def test_public_trained_or_untracked_model_cannot_claim_formal_validation(self):
        for provenance in ({'candidate_count':31},{'split_id':'v1','training_video_ids':['v000001']},
                           {'split_id':'v1','training_video_ids':['01_001']}):
            with self.assertRaises(ValueError):check_model_for_formal_validation({'training_provenance':provenance},self.manifest)
        self.assertTrue(check_model_for_formal_validation({'training_provenance':{'split_id':'v1','training_video_ids':['00_001']}},self.manifest))

    def test_changed_frozen_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'video';p.write_bytes(b'original')
            rows=[{'video_id':'00_001','video_path':str(p),'video_sha256':sha(p)}]
            verify_inputs(rows);p.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_inputs(rows)

    def test_manifest_hash_and_cross_source_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);path=folder/'split_manifest.json'
            def freeze(value):
                path.write_text(json.dumps(value),encoding='utf-8')
                (folder/'freeze.json').write_text(json.dumps({'files':{'split_manifest.json':sha(path)}}),encoding='utf-8')
            freeze(self.manifest);self.assertEqual(load_protocol(folder)['split_id'],'v1')
            path.write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError):load_protocol(folder)
            changed=copy.deepcopy(self.manifest);changed['videos'][1]['split']='validation_reserved'
            freeze(changed)
            with self.assertRaises(ValueError):load_protocol(folder)


if __name__=='__main__':unittest.main()
