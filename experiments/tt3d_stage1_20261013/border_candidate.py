"""Image-supported enclosing quadrilaterals, independent of reference corners."""
import itertools
import cv2
import numpy as np
from experiments.tt3d_stage1_20261013.probe_border_lines import outer_lines

def candidates(image, mask):
    lines=outer_lines(image,mask)
    if len(lines)<4:return []
    lines=sorted(lines,key=lambda l:np.arctan2(l['normal'][1],l['normal'][0]))
    normals=np.array([l['normal'] for l in lines]); offsets=np.array([l['offset'] for l in lines])
    ix=np.array(list(itertools.combinations(range(len(lines)),4)))
    n=normals[ix];c=offsets[ix]; nn=np.roll(n,-1,axis=1);cc=np.roll(c,-1,axis=1)
    angles=np.arctan2(n[:,:,1],n[:,:,0]);gaps=(np.roll(angles,-1,axis=1)-angles)%(2*np.pi)
    det=n[:,:,0]*nn[:,:,1]-n[:,:,1]*nn[:,:,0]
    valid=np.all((gaps<np.pi-.04)&(gaps>.04)&(abs(det)>.04),axis=1)
    n=n[valid];c=c[valid];nn=nn[valid];cc=cc[valid];det=det[valid];ix=ix[valid]
    q=np.stack([(c*nn[:,:,1]-n[:,:,1]*cc)/det,(n[:,:,0]*cc-c*nn[:,:,0])/det],axis=2)
    h,w=mask.shape
    valid=np.all((q[:,:,0]>=0)&(q[:,:,0]<w)&(q[:,:,1]>=0)&(q[:,:,1]<h),axis=1)
    # Each corner must satisfy every enclosing half-plane.
    valid &=np.all(np.einsum('bki,bji->bkj',q,n)>=c[:,None,:]-2,axis=(1,2))
    q=q[valid];ix=ix[valid]
    edges=cv2.Canny(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),60,160)
    distance=cv2.distanceTransform(255-edges,cv2.DIST_L2,3)
    result=[]
    mask_area=np.sum(mask)
    for quad,ids in zip(q,ix):
        area=abs(cv2.contourArea(quad.astype(np.float32)))
        if not max(w*h*.008,mask_area*.95)<=area<=min(w*h*.55,mask_area*1.8):continue
        polygon=np.zeros_like(mask,np.uint8);cv2.fillConvexPoly(polygon,np.rint(quad).astype(np.int32),1)
        overlap=np.sum(polygon.astype(bool)&mask); coverage=overlap/mask_area
        if coverage<.94:continue
        supports=[];spans=[]
        for j,(a,b) in enumerate(zip(quad,np.roll(quad,-1,axis=0))):
            pts=np.rint(np.linspace(a,b,80)).astype(int)
            supports.append(float(np.mean(distance[pts[:,1],pts[:,0]]<=2)))
            # Segment for edge from intersection j to j+1 is line j+1.
            line=lines[ids[(j+1)%4]]; seg=line['segment'].reshape(2,2)
            unit=(b-a)/np.linalg.norm(b-a)
            along=(seg-a)@unit
            spans.append(float(max(0,min(np.max(along),np.linalg.norm(b-a))-max(0,np.min(along)))/np.linalg.norm(b-a)))
        if min(supports)<.35 or np.mean(supports)<.7 or min(spans)<.4:continue
        iou=overlap/(mask_area+np.sum(polygon)-overlap)
        quality=float(np.mean(supports)*.5+np.mean(spans)*.25+coverage*.15+iou*.1)
        result.append({'corners':quad.tolist(),'mask_iou':float(iou),'mask_coverage':float(coverage),
            'edge_support':float(np.mean(supports)),'edge_support_by_side':supports,
            'observed_span_by_side':spans,'quality':quality,'area_fraction':area/(w*h),
            'fitting_source':'enclosing_image_lines','usable_proposal':True,'reason':'enclosing_image_lines'})
    return sorted(result,key=lambda p:p['quality'],reverse=True)

if __name__=='__main__':
    import sys
    from pathlib import Path
    from development.evaluation.run_auto_table_v2 import read,write
    root=Path(__file__).resolve().parents[2]
    pack=root/'outputs/experiments/geometry_reference_audit_20261013/review_pack'
    cache=root/'outputs/experiments/tt3d_geometry_revision_20261013/review_cache'
    out=cache.parent/'border_probe';out.mkdir(exist_ok=True)
    records=[]
    for case in read(pack/'manifest.json')['cases']:
        im=cv2.imread(str(pack/case['image_url']));scale=640/im.shape[1]
        im=cv2.resize(im,(640,round(im.shape[0]*scale)))
        prob=np.load(cache/(case['case_id']+'.npz'))['probability']
        mask=cv2.resize(prob,(640,im.shape[0]))>.8
        result=candidates(im,mask)
        if result:
            cv2.polylines(im,[np.rint(result[0]['corners']).astype(np.int32)],True,(0,220,255),2)
            for r in result:r['corners']=(np.asarray(r['corners'])/scale).tolist()
        cv2.imwrite(str(out/(case['case_id']+'.jpg')),im)
        records.append(dict(case_id=case['case_id'],candidates=result[:5]))
        print(case['case_id'],len(result),flush=True)
    write(out/'records.json',records)
