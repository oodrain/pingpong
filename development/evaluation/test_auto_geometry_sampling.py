from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
import cv2
from development.detection.test_auto_table_geometry import scene
from development.detection.auto_table_geometry import GeometryConfig,AutoGeometryTimeline
from development.evaluation.run_auto_table_v2 import collect,rows,main


class SamplingTests(unittest.TestCase):
    def test_adaptive_cache_cannot_be_reused_as_fixed_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory)
            (cache/'summary.json').write_text(json.dumps(dict(sample_seconds=.4,warmup_sample_seconds=.12)))
            with patch('sys.argv',['run_auto_table_v2','--data-root',str(cache/'missing_data'),
                                    '--reuse-geometry',str(cache)]), patch(
                    'development.evaluation.run_auto_table_v2.TT3DSegmenter') as model:
                with self.assertRaises(SystemExit) as caught:main()
                self.assertEqual(caught.exception.code,2)
                model.assert_not_called()

    def test_adaptive_warmup_is_causal_and_preserves_fixed_comparison_frames(self):
        image,mask,_=scene()
        class Model:
            def probability(self,image):return mask
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            (out/'diagnostics').mkdir();(out/'fixed_sampling').mkdir()
            path=out/'test.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),20,(640,360))
            self.assertTrue(writer.isOpened())
            for _ in range(30):writer.write(image)
            writer.release()
            meta=dict(video_id='test',n_frames=30,width=640,height=360,fps=20)
            samples,_=collect(Model(),meta,path,'train',out,GeometryConfig(),.4,.1)
            fixed=rows(out/'fixed_sampling/train_test.jsonl')
            self.assertEqual([s['frame_id'] for s in fixed],[0,8,16,24])
            timeline=AutoGeometryTimeline(samples)
            self.assertIsNone(timeline.lookup(3))
            self.assertIsNotNone(timeline.lookup(4))
            self.assertIsNone(AutoGeometryTimeline(fixed).lookup(15))
            self.assertIsNotNone(AutoGeometryTimeline(fixed).lookup(16))


if __name__=='__main__':unittest.main()
