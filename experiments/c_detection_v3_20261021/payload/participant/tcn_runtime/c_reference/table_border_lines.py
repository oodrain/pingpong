"""Find four enclosing image lines near a segmented table.

These are alternative image hypotheses, not calibrated corner probabilities.
No reference coordinates or event labels are inputs. Long observed segments
and support along every complete side limit extrapolation into player edges.
"""
from functools import lru_cache
from itertools import combinations
import cv2
import numpy as np
from .image_evidence import profiled


@lru_cache(maxsize=32)
def _combinations(count):
    return np.asarray(list(combinations(range(count), 4)), dtype=np.int32)


def image_segments(image, frame_evidence=None):
    if frame_evidence is not None:
        return frame_evidence.segments
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    detected = [cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0],
                cv2.HoughLinesP(cv2.Canny(gray,60,160),1,np.pi/720,20,minLineLength=20,maxLineGap=12)]
    available = [v.reshape(-1,2,2) for v in detected if v is not None]
    return np.concatenate(available) if available else np.empty((0,2,2))


def border_extension(quad, segments):
    """Observable, nearly collinear border length beyond a proposed corner."""
    extensions = []
    for a,b in zip(quad,np.roll(quad,-1,axis=0)):
        vector = b-a;length = np.linalg.norm(vector);unit = vector/length
        offsets = segments-a
        across = abs(offsets[:,:,0]*unit[1]-offsets[:,:,1]*unit[0])
        along = offsets @ unit
        segment_vectors = segments[:,1]-segments[:,0]
        segment_lengths = np.linalg.norm(segment_vectors,axis=1)
        aligned = abs(segment_vectors @ unit)>=segment_lengths*np.cos(np.radians(3))
        aligned &= (across.max(axis=1)<=3) & (segment_lengths>=20)
        # It must overlap this border, not merely lie on its infinite extension.
        aligned &= (along.max(axis=1)>length*.1) & (along.min(axis=1)<length*.9)
        over = np.maximum(-along[aligned].min(axis=1),along[aligned].max(axis=1)-length)
        extensions.append(float(max(0,over.max())) if len(over) else 0.)
    return extensions


@profiled('outer_lines')
def _outer_lines(image, mask, frame_evidence=None):
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return [], None
    points = np.column_stack([xs, ys]).astype(float)
    points = points[::max(1, len(points)//1000)]
    centre = points.mean(axis=0)
    edges = frame_evidence.edges if frame_evidence is not None else cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 60, 160)
    roi = np.zeros_like(mask, np.uint8)
    pad = round(max(np.ptp(xs), np.ptp(ys))*.35)
    roi[max(0,ys.min()-pad):ys.max()+pad+1, max(0,xs.min()-pad):xs.max()+pad+1] = 255
    segments = frame_evidence.roi_hough(roi) if frame_evidence is not None else cv2.HoughLinesP(cv2.bitwise_and(edges,roi),1,np.pi/720,20,
                              minLineLength=20,maxLineGap=12)
    candidates = []
    for segment in ([] if segments is None else segments.reshape(-1,4)):
        a, b = segment.reshape(2,2).astype(float)
        length = np.linalg.norm(b-a)
        midpoint = np.rint((a+b)/2).astype(int)
        if length < 20 or not roi[np.clip(midpoint[1],0,mask.shape[0]-1),
                                  np.clip(midpoint[0],0,mask.shape[1]-1)]:
            continue
        unit = (b-a)/length
        normal = np.array([-unit[1],unit[0]])
        if (centre-a) @ normal < 0:
            normal = -normal
        offset = a @ normal
        distance = points @ normal-offset
        if np.mean(distance >= -3) < .96 or np.quantile(distance,.02) > 18:
            continue
        if any(normal @ line['normal'] > .999 and abs(offset-line['offset']) < 4
               for line in candidates):
            continue
        candidates.append(dict(normal=normal, offset=offset, segment=segment, length=length))
    lines = sorted(candidates, key=lambda line:line['length'], reverse=True)[:32]
    return sorted(lines, key=lambda line:np.arctan2(*line['normal'][::-1])), edges


@profiled('enclosing_candidates')
def enclosing_candidates(image, mask, minimum_area=.008, maximum_area=.55, frame_evidence=None):
    """Return quality-ranked alternatives in the supplied fitting coordinates."""
    lines, edges = _outer_lines(image, mask, frame_evidence=frame_evidence)
    if len(lines) < 4:
        return []
    normals = np.array([line['normal'] for line in lines])
    offsets = np.array([line['offset'] for line in lines])
    indices = _combinations(len(lines))
    n, c = normals[indices], offsets[indices]
    nn, cc = np.roll(n,-1,axis=1), np.roll(c,-1,axis=1)
    angles = np.arctan2(n[:,:,1],n[:,:,0])
    gaps = (np.roll(angles,-1,axis=1)-angles) % (2*np.pi)
    determinant = n[:,:,0]*nn[:,:,1]-n[:,:,1]*nn[:,:,0]
    valid = np.all((gaps < np.pi-.04) & (gaps > .04) & (abs(determinant) > .04),axis=1)
    n,c,nn,cc,determinant,indices = [v[valid] for v in (n,c,nn,cc,determinant,indices)]
    quad = np.stack([(c*nn[:,:,1]-n[:,:,1]*cc)/determinant,
                     (n[:,:,0]*cc-c*nn[:,:,0])/determinant],axis=2)
    h,w = mask.shape
    valid = np.all((quad[:,:,0]>=0) & (quad[:,:,0]<w) & (quad[:,:,1]>=0) & (quad[:,:,1]<h),axis=1)
    valid &= np.all(np.einsum('bki,bji->bkj',quad,n)>=c[:,None,:]-2,axis=(1,2))
    area = abs(np.sum(quad[:,:,0]*np.roll(quad[:,:,1],-1,axis=1)
                      -quad[:,:,1]*np.roll(quad[:,:,0],-1,axis=1),axis=1))*.5
    mask_area = int(np.sum(mask))
    valid &= (area>=max(w*h*minimum_area,mask_area*.95)) & (area<=min(w*h*maximum_area,mask_area*1.8))
    quad, indices, area = quad[valid], indices[valid], area[valid]
    if not len(quad):
        return []
    # Evaluate complete sides and observed segment lengths in batches. Raster
    # overlap is needed only for the strongest survivors, not every combination.
    ends = np.roll(quad,-1,axis=1)
    vectors = ends-quad
    lengths = np.linalg.norm(vectors,axis=2)
    valid = np.all(lengths>=12,axis=1)
    quad,indices,area,vectors,lengths = [v[valid] for v in (quad,indices,area,vectors,lengths)]
    if not len(quad):
        return []
    points = np.rint(quad[:,:,None,:]+vectors[:,:,None,:]*np.linspace(0,1,80)[None,None,:,None]).astype(int)
    points[:,:,:,0] = np.clip(points[:,:,:,0],0,w-1)
    points[:,:,:,1] = np.clip(points[:,:,:,1],0,h-1)
    distance = frame_evidence.edge_distance if frame_evidence is not None else cv2.distanceTransform(255-edges,cv2.DIST_L2,3)
    supports = np.mean(distance[points[:,:,:,1],points[:,:,:,0]]<=2,axis=2)
    segments = np.array([line['segment'].reshape(2,2) for line in lines])[np.roll(indices,-1,axis=1)]
    along = np.sum((segments-quad[:,:,None,:])*(vectors/lengths[:,:,None])[:,:,None,:],axis=3)
    spans = np.maximum(0,np.minimum(along.max(axis=2),lengths)-np.maximum(0,along.min(axis=2)))/lengths
    valid = (supports.min(axis=1)>=.35) & (supports.mean(axis=1)>=.70) & (spans.min(axis=1)>=.40)
    quad,area,supports,spans = [v[valid] for v in (quad,area,supports,spans)]
    approximate_quality = supports.mean(axis=1)*.5+spans.mean(axis=1)*.25+.15+mask_area/area*.1
    result = []
    segments = image_segments(image, frame_evidence=frame_evidence)
    for index in np.argsort(-approximate_quality)[:40]:
        q = quad[index]
        if not cv2.isContourConvex(q.astype(np.float32)):
            continue
        polygon = np.zeros_like(mask,np.uint8)
        cv2.fillConvexPoly(polygon,np.rint(q).astype(np.int32),1)
        overlap = np.sum(polygon.astype(bool)&mask)
        coverage = overlap/mask_area
        if coverage < .94:
            continue
        iou = overlap/(mask_area+np.sum(polygon)-overlap)
        quality = supports[index].mean()*.5+spans[index].mean()*.25+coverage*.15+iou*.1
        result.append(dict(corners=q.tolist(),mask_iou=float(iou),mask_coverage=float(coverage),
            edge_support=float(supports[index].mean()),edge_support_by_side=supports[index].tolist(),
            observed_span_by_side=spans[index].tolist(),quality=float(quality),
            area_fraction=float(area[index]/(w*h)),fitting_source='enclosing_image_lines',
            border_extension_by_side=border_extension(q,segments)))
    return sorted(result,key=lambda proposal:proposal['quality'],reverse=True)
