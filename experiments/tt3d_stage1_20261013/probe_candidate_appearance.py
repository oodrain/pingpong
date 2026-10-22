"""Inspect proposed expansion appearance; references are diagnostic only."""
from pathlib import Path
import cv2
import numpy as np
from development.detection.table_border_lines import enclosing_candidates
from development.evaluation.run_auto_table_v2 import read
from development.evaluation.geometry_reference_review import latest_reviews
from development.evaluation.validate_table_geometry import corner_error

root=Path(__file__).resolve().parents[2]
pack=root/'outputs/experiments/geometry_reference_audit_20261013/review_pack'
cache=root/'outputs/experiments/tt3d_geometry_revision_20261013/review_cache'
refs=latest_reviews(pack,read(pack/'manifest.json'))
for case in read(pack/'manifest.json')['cases']:
    if case['case_id'] not in ('train_17_002_000072','public_v000094_000048','train_03_001_000049'):continue
    image=cv2.imread(str(pack/case['image_url']));scale=640/image.shape[1]
    image=cv2.resize(image,(640,round(image.shape[0]*scale)))
    mask=(cv2.resize(np.load(cache/(case['case_id']+'.npz'))['probability'],(640,image.shape[0]))>.8).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours=sorted(contours,key=cv2.contourArea,reverse=True)
    cleaned=np.zeros_like(mask);cv2.drawContours(cleaned,[contours[0]],-1,255,-1)
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(float)
    colors=lab[cleaned>0];colors=colors[::max(1,len(colors)//256)]
    rs=enclosing_candidates(image,cleaned>0)
    print('\n',case['case_id'],len(rs))
    for r in rs[:20]:
        polygon=np.zeros_like(mask);cv2.fillConvexPoly(polygon,np.rint(r['corners']).astype(np.int32),255)
        added=lab[(polygon>0)&(cleaned==0)];added=added[::max(1,len(added)//512)]
        similarity=np.mean(np.linalg.norm(added[:,None,:]-colors[None,:,:],axis=2).min(axis=1)<25) if len(added) else 1
        print(round(r['quality'],3),round(r['mask_iou'],3),round(similarity,3),corner_error(np.asarray(r['corners'])/scale,refs[case['case_id']]['corners']),np.rint(r['corners']).tolist())
