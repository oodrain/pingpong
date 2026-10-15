"""Inspect image boundary lines against the segmentation, without references."""
import itertools
import sys
from pathlib import Path
import cv2
import numpy as np
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from development.evaluation.run_auto_table_v2 import read,write

def outer_lines(image, mask):
    ys,xs=np.nonzero(mask)
    points=np.column_stack([xs,ys]).astype(float)[::max(1,len(xs)//1000)]
    centre=points.mean(axis=0)
    edges=cv2.Canny(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),60,160)
    roi=np.zeros_like(mask,np.uint8)
    pad=round(max(np.ptp(xs),np.ptp(ys))*.25)
    roi[max(0,ys.min()-pad):ys.max()+pad+1,max(0,xs.min()-pad):xs.max()+pad+1]=255
    segments=cv2.HoughLinesP(cv2.bitwise_and(edges,roi),1,np.pi/720,20,minLineLength=20,maxLineGap=12)
    candidates=[]
    for seg in ([] if segments is None else segments[:,0,:]):
        a,b=seg.reshape(2,2).astype(float)
        u=(b-a)/np.linalg.norm(b-a);n=np.array([-u[1],u[0]])
        if (centre-a)@n<0:n=-n
        c=a@n;dist=points@n-c
        if np.mean(dist>=-3)<.96 or np.quantile(dist,.02)>18:continue
        if any(abs(n@line['normal'])>.999 and abs(c-line['offset'])<4 for line in candidates):continue
        candidates.append({'normal':n,'offset':c,'segment':seg,'length':np.linalg.norm(b-a),
                           'mask_inside':np.mean(dist>=-3),'near':np.quantile(dist,.02)})
    return sorted(candidates,key=lambda l:l['length'],reverse=True)[:32]

def main():
    pack=PROJECT/'outputs/experiments/geometry_reference_audit_20261013/review_pack'
    cache=PROJECT/'outputs/experiments/tt3d_geometry_revision_20261013/review_cache'
    out=cache.parent/'line_probe';out.mkdir(exist_ok=True)
    for case in read(pack/'manifest.json')['cases']:
        im=cv2.imread(str(pack/case['image_url']));im=cv2.resize(im,(640,round(im.shape[0]*640/im.shape[1])))
        prob=np.load(cache/(case['case_id']+'.npz'))['probability']
        mask=cv2.resize(prob,(640,im.shape[0]))>.8
        lines=outer_lines(im,mask)
        for i,line in enumerate(lines):
            a,b=line['segment'].reshape(2,2).astype(int)
            color=(int((i*77)%255),int((i*43+100)%255),255)
            cv2.line(im,tuple(a),tuple(b),color,1)
            cv2.putText(im,str(i),tuple(((a+b)/2).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
        cv2.imwrite(str(out/(case['case_id']+'.jpg')),im)
        write(out/(case['case_id']+'.json'),[{k:np.asarray(v).tolist() for k,v in l.items()} for l in lines])
        print(case['case_id'],len(lines))

if __name__=='__main__':main()
