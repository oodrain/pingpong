"""Cache exact review-frame model outputs, without exposing human coordinates."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from development.detection.tt3d_segmenter import TT3DSegmenter
from development.evaluation.run_auto_table_v2 import sha, read, write

def main():
    pack = PROJECT / 'outputs/experiments/geometry_reference_audit_20261013/review_pack'
    out = PROJECT / 'outputs/experiments/tt3d_geometry_revision_20261013/review_cache'
    out.mkdir(parents=True, exist_ok=False)
    manifest = read(pack / 'manifest.json')
    model = TT3DSegmenter(PROJECT / 'outputs/models/table_segmentation.ckpt')
    for case in manifest['cases']:
        path = pack / case['image_url']
        if sha(path) != case['image_sha256']:
            raise ValueError('Image changed')
        image = cv2.imread(str(path))
        prob = model.probability(image)
        np.savez_compressed(out / (case['case_id']+'.npz'), probability=prob)
        scale = 640 / image.shape[1]
        im = cv2.resize(image, (640, round(image.shape[0]*scale)))
        mask = cv2.resize(prob, (640, im.shape[0])) > .8
        tint = im.copy()
        tint[mask] = (tint[mask]*.6 + np.array([0,180,0])*.4).astype(np.uint8)
        cv2.imwrite(str(out / (case['case_id']+'.jpg')), tint)
        print(case['case_id'], flush=True)
    write(out / 'provenance.json', {'model':model.provenance, 'manifest_sha256':sha(pack/'manifest.json'),
          'cache_sha256':{p.name:sha(p) for p in out.glob('*.npz')}})

if __name__ == '__main__': main()
