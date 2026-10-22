"""Measure straight neutral border evidence; annotations are diagnostic only."""
from pathlib import Path
import sys
import cv2
import numpy as np
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from development.evaluation.run_auto_table_v2 import read,write
from development.detection.table_border_lines import enclosing_candidates,image_segments
from development.detection.auto_table_geometry import order_quad


def neutral_support(image,quad):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    neutral=(hsv[:,:,1]<75)&(hsv[:,:,2]>130)
    distance=cv2.distanceTransform((~neutral).astype(np.uint8)*255,cv2.DIST_L2,3)
    segments=image_segments(image)
    scores=[];spans=[];generic=[]
    for a,b in zip(quad,np.roll(quad,-1,axis=0)):
        vector=b-a;length=np.linalg.norm(vector);unit=vector/length
        points=np.rint(np.linspace(a,b,120)).astype(int)
        points[:,0]=np.clip(points[:,0],0,image.shape[1]-1);points[:,1]=np.clip(points[:,1],0,image.shape[0]-1)
        scores.append(float(np.mean(distance[points[:,1],points[:,0]]<=2)))
        offsets=segments-a;across=abs(offsets[:,:,0]*unit[1]-offsets[:,:,1]*unit[0]);along=offsets@unit
        vectors=segments[:,1]-segments[:,0];lens=np.linalg.norm(vectors,axis=1)
        aligned=(abs(vectors@unit)>=lens*np.cos(np.radians(5)))&(across.max(axis=1)<=4)&(lens>=15)
        best=0;all_best=0
        for seg,t in zip(segments[aligned],along[aligned]):
            all_best=max(all_best,max(0,min(t.max(),length)-max(0,t.min()))/length)
            xy=np.rint(np.linspace(*seg,40)).astype(int);xy[:,0]=np.clip(xy[:,0],0,image.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,image.shape[0]-1)
            if np.mean(distance[xy[:,1],xy[:,0]]<=2)<.6:continue
            best=max(best,max(0,min(t.max(),length)-max(0,t.min()))/length)
        spans.append(float(best))
        generic.append(float(all_best))
    return scores,spans,generic


def main():
    root=PROJECT/'outputs/experiments'
    pack=root/'geometry_reference_audit_20261013/review_pack'
    record={r['case_id']:r for r in read(root/'tt3d_geometry_revision_20261013/regression_evaluation_01/records.json')}
    for case in read(pack/'manifest.json')['cases']:
        im=cv2.imread(str(pack/case['image_url']));scale=640/im.shape[1];im=cv2.resize(im,(640,round(im.shape[0]*scale)))
        p=record[case['case_id']]['revised_proposal']
        if not p.get('corners'):continue
        q=order_quad(p['corners'])*scale
        score,span,generic=neutral_support(im,q)
        print(case['case_id'],p['usable_proposal'],np.round(score,2).tolist(),np.round(span,2).tolist(),np.round(generic,2).tolist())
        if case['case_id']=='train_00_005_000018':
            mask=np.zeros(im.shape[:2],np.uint8);cv2.fillConvexPoly(mask,np.rint(q).astype(np.int32),1)
            candidates=enclosing_candidates(im,mask>0)
            for alt in candidates[:12]:
                qq=order_quad(alt['corners']);s,sp,g=neutral_support(im,qq)
                print('ALT',round(alt['quality'],3),np.round(s,2).tolist(),np.round(sp,2).tolist(),np.rint(qq).tolist())


if __name__=='__main__':main()
