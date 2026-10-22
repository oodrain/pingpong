"""Segmentation-guided line fitting, causal stability, and conservative axes.

Only image/mask inputs; no landing labels. Confidence is a quality heuristic,
not a calibrated probability of correct corners or physical calibration.
"""
from dataclasses import dataclass
from collections import Counter
import bisect
import math
import cv2
import numpy as np
from .image_evidence import profiled
from .table_geometry import build_plane
from .table_border_lines import enclosing_candidates, border_extension, image_segments


@dataclass(frozen=True)
class GeometryConfig:
    mask_threshold: float = .8
    fitting_width: int = 640
    minimum_area: float = .008
    maximum_area: float = .55
    minimum_mask_iou: float = .70
    minimum_edge_support: float = .40
    minimum_side_support: float = .75
    maximum_border_extension: float = 35.0  # pixels at fitting_width
    minimum_visible_mask_coverage: float = .94
    minimum_occluded_mask_iou: float = .50
    minimum_occluded_side_support: float = .75
    minimum_occluded_edge_support: float = .85
    maximum_reconstruction_growth: float = 1.50
    stability_samples: int = 3
    maximum_corner_jitter: float = .025
    maximum_hold_frames: int = 20
    minimum_straight_span: float = .25


def order_quad(points):
    q = np.asarray(points, np.float32).reshape(4,2)
    centre = q.mean(axis=0)
    q = q[np.argsort(np.arctan2(q[:,1]-centre[1], q[:,0]-centre[0]))]
    return np.roll(q, -np.argmin(q.sum(axis=1)), axis=0)


@profiled('extract_seed')
def _extract_seed(image, probability, config=None, frame_evidence=None):
    cfg = config or GeometryConfig()
    height, width = image.shape[:2]
    scale = min(1.0, cfg.fitting_width / width)
    im = frame_evidence.fitting_image if frame_evidence is not None else cv2.resize(image, (round(width*scale), round(height*scale)))
    h,w = im.shape[:2]
    prob = np.asarray(probability)
    if prob.ndim != 2 or not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1:
        raise ValueError("Malformed segmentation probability")
    mask = (cv2.resize(prob.astype(np.float32), (w,h)) > cfg.mask_threshold).astype(np.uint8)*255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if not contours:
        return {"usable_proposal": False, "reason": "empty_segmentation", "corners": None}
    largest = cv2.contourArea(contours[0])
    centre = contours[0][:,0,:].mean(axis=0)
    selected = [c for c in contours if cv2.contourArea(c) >= max(w*h*.001, largest*.05)
                and np.linalg.norm(c[:,0,:].mean(axis=0)-centre) < .5*w]
    if not selected:
        return {"usable_proposal": False, "reason": "segmentation_too_small", "corners": None}
    contour = np.concatenate(selected)[:,0,:].astype(np.float32)
    hull = cv2.convexHull(contour)
    cleaned = np.zeros_like(mask)
    cv2.drawContours(cleaned, selected, -1, 255, -1)
    edges = frame_evidence.edges if frame_evidence is not None else cv2.Canny(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),60,160)
    if frame_evidence is not None:
        image_edges = frame_evidence.edge_points
    else:
        edge_y, edge_x = np.nonzero(edges)
        image_edges = np.column_stack([edge_x,edge_y]).astype(np.float32)
    edge_distance = frame_evidence.edge_distance if frame_evidence is not None else cv2.distanceTransform(255-edges, cv2.DIST_L2,3)
    proposals = []
    rejected = Counter()
    cleaned_area = cv2.countNonZero(cleaned)
    fit_cache = {}
    def fit_quad(q):
        if not cv2.isContourConvex(q) or min(np.linalg.norm(q-np.roll(q,1,axis=0),axis=1))<12:
            rejected['short_or_nonconvex_side'] += 1
            return None
        area = cv2.contourArea(q)/(w*h)
        if not cfg.minimum_area <= area <= cfg.maximum_area:
            rejected['area_out_of_range'] += 1
            return None
        lines = []
        for a,b in zip(q,np.roll(q,-1,axis=0)):
            vector = b-a
            length = np.linalg.norm(vector)
            unit = vector/length
            def nearby(points, radius):
                offsets = points-a
                along = offsets @ unit
                across = np.abs(offsets[:,0]*unit[1]-offsets[:,1]*unit[0])
                return points[(along>length*.1)&(along<length*.9)&(across<radius)]
            # Use actual image borders where supported; otherwise retain a mask fit.
            pts = nearby(image_edges,5.0)
            if len(pts)<max(8,length*.25):
                pts = nearby(contour,7.0)
            if len(pts)<8:
                break
            vx,vy,x0,y0 = cv2.fitLine(pts,cv2.DIST_HUBER,0,.01,.01).ravel()
            origin,direction = np.array([x0,y0]),np.array([vx,vy])
            # Nearby player/net edges can rotate a pooled fit away from the
            # mask side. Keep the original hull side rather than extrapolating
            # an unrelated image edge into a remote corner. All final mask and
            # image-border quality gates still apply to this fallback.
            endpoint_distance = np.abs((np.array([a,b])-origin)[:,0]*direction[1]
                                       -(np.array([a,b])-origin)[:,1]*direction[0])
            if abs(float(direction @ unit)) < math.cos(math.radians(10)) or endpoint_distance.max()>10:
                rejected['line_fit_reverted_to_hull'] += 1
                origin,direction = a,unit
            lines.append((origin,direction))
        if len(lines)!=4:
            rejected['insufficient_line_points'] += 1
            return None
        refined = []
        for index in range(4):
            p,u = lines[index-1]
            r,v = lines[index]
            matrix = np.column_stack([u,-v])
            if abs(np.linalg.det(matrix))<.05:
                break
            t = np.linalg.solve(matrix,r-p)[0]
            refined.append(p+t*u)
        if len(refined)!=4:
            rejected['nearly_parallel_sides'] += 1
            return None
        refined = np.asarray(refined,np.float32)
        fitting_source = 'image_mask_lines'
        if not cv2.isContourConvex(refined) or np.max(np.linalg.norm(refined-q,axis=1))>15:
            rejected['refinement_deformed_quad'] += 1
            # Acute perspective corners amplify small line-fit errors. The
            # segmentation hull is already a bounded candidate: score it with
            # the same gates instead of discarding a good mask after bad fitting.
            refined = q.copy()
            fitting_source = 'segmentation_hull_fallback'
        if np.any(refined[:,0]<0) or np.any(refined[:,0]>=w) or np.any(refined[:,1]<0) or np.any(refined[:,1]>=h):
            rejected['refinement_out_of_image'] += 1
            return None
        polygon = np.zeros_like(mask)
        cv2.fillConvexPoly(polygon,refined.round().astype(np.int32),255)
        overlap = cv2.countNonZero(cv2.bitwise_and(polygon, cleaned))
        union = cv2.countNonZero(polygon)+cleaned_area-overlap
        iou = float(overlap/union) if union else 0.
        coverage = float(overlap/cleaned_area)
        supports = []
        for a,b in zip(refined,np.roll(refined,-1,axis=0)):
            points = np.linspace(a,b,80).round().astype(int)
            points[:,0] = np.clip(points[:,0],0,w-1)
            points[:,1] = np.clip(points[:,1],0,h-1)
            supports.append(float(np.mean(edge_distance[points[:,1],points[:,0]]<=3)))
        quality = iou*.6+float(np.mean(supports))*.4
        return {"corners":(refined/scale).tolist(),"mask_iou":iou,
                          "mask_coverage":coverage,
                          "edge_support":float(np.mean(supports)),"edge_support_by_side":supports,
                          "area_fraction":area,"quality":quality,"fitting_source":fitting_source}

    for epsilon in (.005,.01,.015,.025,.04,.06,.08):
        approx = cv2.approxPolyDP(hull,epsilon*cv2.arcLength(hull,True),True)
        if len(approx)!=4:
            rejected['not_four_vertices'] += 1
            continue
        q = order_quad(approx[:,0,:])
        key = (q.dtype.str, q.tobytes())
        if key not in fit_cache:
            before = rejected.copy()
            item = fit_quad(q)
            fit_cache[key] = (item, rejected-before)
        else:
            item, counts = fit_cache[key]
            rejected.update(counts)
            if frame_evidence is not None: frame_evidence.profile.add('identical_seed_fit_reused')
        if item is not None: proposals.append(item)
    def accepted(p):
        ordinary = (p['mask_iou']>=cfg.minimum_mask_iou and p['edge_support']>=cfg.minimum_edge_support
                    and min(p['edge_support_by_side'])>=cfg.minimum_side_support)
        occlusion = (p['mask_iou']>=cfg.minimum_occluded_mask_iou
                     and p['mask_coverage']>=cfg.minimum_visible_mask_coverage
                     and min(p['edge_support_by_side'])>=cfg.minimum_occluded_side_support
                     and p['edge_support']>=cfg.minimum_occluded_edge_support)
        return ordinary or occlusion
    supported = [p for p in proposals if accepted(p)]
    best = max(supported or proposals,key=lambda p:p['quality']) if proposals else None
    base_extension = border_extension(order_quad(best['corners'])*scale,image_segments(im,frame_evidence=frame_evidence)) if best else []
    requires_outer = best is None or min(best['edge_support_by_side'])<cfg.minimum_side_support or max(base_extension)>cfg.maximum_border_extension
    if frame_evidence is not None:
        frame_evidence.profile.add('outer_search_required' if requires_outer else 'outer_search_skipped')
    alternatives = enclosing_candidates(im,cleaned>0,cfg.minimum_area,cfg.maximum_area,frame_evidence=frame_evidence) if requires_outer else []
    # An enclosing candidate can expose a clipped hull, but a large expansion
    # into players/aprons is ambiguous. Do not force a four-corner answer.
    strong = [p for p in alternatives if min(p['edge_support_by_side'])>=cfg.minimum_side_support
              and max(p['border_extension_by_side'])<=cfg.maximum_border_extension]
    diagnostic = {'image_line_candidates':len(alternatives) if requires_outer else None, 'strong_image_line_candidates':len(strong) if requires_outer else None, 'outer_search_executed':requires_outer}

    diagnostic['base_border_extension_by_side'] = base_extension
    if best and strong and (min(best['edge_support_by_side'])<cfg.minimum_side_support
                            or max(base_extension)>cfg.maximum_border_extension):
        alternative = strong[0]
        q = order_quad(alternative['corners'])/scale
        base = order_quad(best['corners'])
        displacement = float(np.max(np.linalg.norm(q-base,axis=1)))
        growth = alternative['area_fraction']/best['area_fraction']
        diagnostic.update(enclosing_growth=growth,enclosing_max_corner_shift_px=displacement)
        if growth>=1.08 and displacement>math.hypot(width,height)*.015:
            if growth>cfg.maximum_reconstruction_growth:
                return dict(best,usable_proposal=False,reason='ambiguous_outer_border',
                            border_diagnostic=diagnostic,rejection_counts=dict(rejected))
            if alternative['mask_iou']>=cfg.minimum_mask_iou and best['mask_iou']>=cfg.minimum_occluded_mask_iou:
                alternative=dict(alternative,corners=q.tolist(),usable_proposal=True,
                                 reason='reconstructed_outer_border',border_diagnostic=diagnostic,
                                 rejection_counts=dict(rejected))
                return alternative
    if best is None:
        fallback = [p for p in strong if p['mask_iou']>=cfg.minimum_mask_iou
                    and min(p['observed_span_by_side'])>=.5]
        if fallback:
            p=fallback[0]
            return dict(p,corners=(order_quad(p['corners'])/scale).tolist(),usable_proposal=True,
                        reason='enclosing_image_lines',border_diagnostic=diagnostic,rejection_counts=dict(rejected))
        return dict(usable_proposal=False,reason='no_supported_quad',corners=None,
                    border_diagnostic=diagnostic,rejection_counts=dict(rejected))
    good = accepted(best)
    if max(base_extension)>cfg.maximum_border_extension:
        return dict(best,usable_proposal=False,reason='unsupported_border_continuation',
                    border_diagnostic=diagnostic,rejection_counts=dict(rejected))
    if good:
        reason = 'segmentation_line_fit' if best['mask_iou']>=cfg.minimum_mask_iou else 'occlusion_supported_border'
    elif best['mask_iou']<cfg.minimum_mask_iou:
        reason = 'insufficient_mask_overlap'
    else:
        reason = 'insufficient_image_border'
    return dict(best,usable_proposal=good,reason=reason,border_diagnostic=diagnostic,rejection_counts=dict(rejected))


@profiled('perimeter_evidence')
def perimeter_evidence(image, corners, segments=None, frame_evidence=None):
    """Measured image evidence at fitting resolution, independent of labels."""
    if frame_evidence is not None:
        key = frame_evidence.perimeter_key(corners, segments)
        cached = frame_evidence.perimeter_get(key)
        if cached is not None:
            return cached
        distance = frame_evidence.neutral_distance
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        neutral = (hsv[:,:,1] < 75) & (hsv[:,:,2] > 130)
        distance = cv2.distanceTransform((~neutral).astype(np.uint8)*255, cv2.DIST_L2, 3)
    segments = image_segments(image,frame_evidence=frame_evidence) if segments is None else segments
    support, spans = [], []
    vectors=segments[:,1]-segments[:,0]; lengths=np.linalg.norm(vectors,axis=1)
    for a,b in zip(corners,np.roll(corners,-1,axis=0)):
        vector=b-a; length=float(np.linalg.norm(vector)); unit=vector/max(length,1e-6)
        points=np.rint(np.linspace(a,b,120)).astype(int)
        points[:,0]=np.clip(points[:,0],0,image.shape[1]-1)
        points[:,1]=np.clip(points[:,1],0,image.shape[0]-1)
        support.append(float(np.mean(distance[points[:,1],points[:,0]]<=2)))
        offsets=segments-a
        across=abs(offsets[:,:,0]*unit[1]-offsets[:,:,1]*unit[0])
        along=offsets@unit
        aligned=(abs(vectors@unit)>=lengths*math.cos(math.radians(5))) & (across.max(axis=1)<=4) & (lengths>=15)
        spans.append(max((float(max(0,min(t.max(),length)-max(0,t.min()))/max(length,1e-6))
                          for t in along[aligned]),default=0.))
    result = dict(neutral_border_support_by_side=support,straight_span_by_side=spans)
    if frame_evidence is not None:
        frame_evidence.perimeter_put(key, result)
    return result


@profiled('validate_quad')
def validate_quad_observation(image, seed, config=None, frame_evidence=None):
    """Revalidate a frozen NN proposal using current RGB straight borders.

    A reconstructed polygon is an image/prior fit; its overlap is NOT a fresh
    segmentation score. Keep the original segmentation evidence separately.
    """
    cfg=config or GeometryConfig()
    if not seed.get('corners'):
        return dict(seed)
    scale=min(1.,cfg.fitting_width/image.shape[1])
    im=frame_evidence.fitting_image if frame_evidence is not None else cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
    q=order_quad(seed['corners'])*scale
    segments=image_segments(im,frame_evidence=frame_evidence)
    evidence=perimeter_evidence(im,q,segments,frame_evidence=frame_evidence)
    result=dict(seed,image_perimeter=evidence)
    if not seed['usable_proposal'] or all(s>=cfg.minimum_straight_span or n>=.75 for s,n in zip(
            evidence['straight_span_by_side'],evidence['neutral_border_support_by_side'])):
        return result
    # Three neutral, straight table borders plus one unsupported side expose
    # the common clipped mask/clothing failure. Dark straight borders are valid.
    neutral=np.array(evidence['neutral_border_support_by_side'])
    spans=np.array(evidence['straight_span_by_side'])
    if sum((neutral>=.65)&(spans>=cfg.minimum_straight_span))>=3:
        prior=np.zeros(im.shape[:2],np.uint8)
        cv2.fillConvexPoly(prior,np.rint(q).astype(np.int32),1)
        alternatives=[]
        for candidate in enclosing_candidates(im,prior>0,cfg.minimum_area,cfg.maximum_area,frame_evidence=frame_evidence):
            altq=order_quad(candidate['corners'])
            growth=cv2.contourArea(altq)/max(cv2.contourArea(q),1)
            ev=perimeter_evidence(im,altq,segments,frame_evidence=frame_evidence)
            if (1.08<=growth<=cfg.maximum_reconstruction_growth
                and min(ev['straight_span_by_side'])>=cfg.minimum_straight_span
                and min(ev['neutral_border_support_by_side'])>=.55
                and min(candidate['edge_support_by_side'])>=cfg.minimum_side_support
                and max(candidate['border_extension_by_side'])<=cfg.maximum_border_extension):
                alternatives.append((candidate,altq,ev))
        if alternatives:
            candidate,altq,ev=max(alternatives,key=lambda item:item[0]['quality'])
            return dict(result,corners=(altq/scale).tolist(),reason='straight_outer_border_reconstruction',
                        fitting_source='rgb_lines_with_seed_polygon_prior',image_perimeter=ev,
                        seed_corners=seed['corners'],seed_segmentation_evidence={k:seed.get(k) for k in
                            ('mask_iou','mask_coverage','edge_support','quality')},
                        prior_polygon_iou=candidate['mask_iou'],
                        reconstruction_growth=cv2.contourArea(altq)/cv2.contourArea(q))
    return dict(result,usable_proposal=False,reason='unsupported_straight_border')


@profiled('extract_quad')
def extract_quad(image, probability, config=None, frame_evidence=None):
    return validate_quad_observation(image,_extract_seed(image,probability,config,frame_evidence=frame_evidence),config,frame_evidence=frame_evidence)


def align_quad(corners, reference):
    q=order_quad(corners)
    # Starting vertex may flip at an acute perspective corner. Winding stays fixed.
    return min((np.roll(q,k,axis=0) for k in range(4)),
               key=lambda p:float(np.sum((p-reference)**2)))


def table_diagonal(q):
    return max(float(np.linalg.norm(q[0]-q[2])),float(np.linalg.norm(q[1]-q[3])),1.)


class CausalGeometryTracker:
    def __init__(self, config=None):
        self.config = config or GeometryConfig()
        self.history = []
        self.previous_frame = -1
        self.previous_image = None
        self.last_good = None
        self.epoch = 0
        self.camera_epoch = 0

    def update(self, frame_id, image, proposal, frame_evidence=None):
        if frame_id<=self.previous_frame:
            raise ValueError("Samples must increase")
        self.previous_frame = frame_id
        thumb = frame_evidence.scene_thumb if frame_evidence is not None else cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(64,36)).astype(float)
        cut = self.previous_image is not None and float(np.mean(np.abs(thumb-self.previous_image)>45))>.65
        self.previous_image = thumb
        if cut:
            self.history = []
            self.last_good = None
            self.epoch += 1
            self.camera_epoch += 1
        cfg = self.config
        if self.last_good and frame_id-self.last_good['last_good_frame']>cfg.maximum_hold_frames:
            self.history=[]
            self.last_good=None
        if not proposal['usable_proposal']:
            self.history=[]
            # A rejected mask can still show a contradictory, well observed RGB
            # perimeter. Such evidence must invalidate the held geometry.
            ev=proposal.get('image_perimeter',{})
            if self.last_good and proposal.get('corners') and min(ev.get('straight_span_by_side',[0]))>=cfg.minimum_straight_span:
                old=np.array(self.last_good['corners'])
                observed=align_quad(proposal['corners'],old)
                if np.max(np.linalg.norm(observed-old,axis=1))/table_diagonal(old)>cfg.maximum_corner_jitter:
                    self.last_good=None
                    self.epoch+=1
        if proposal["usable_proposal"]:
            q = order_quad(proposal["corners"])
            reference=self.history[-1] if self.history else np.array(self.last_good['corners']) if self.last_good else None
            if reference is not None:
                q=align_quad(q,reference)
            diag = table_diagonal(q)
            if self.history and np.max(np.linalg.norm(q-self.history[-1],axis=1))/diag>cfg.maximum_corner_jitter:
                self.history = []
                self.last_good = None
                self.epoch += 1
            self.history.append(q)
            self.history = self.history[-cfg.stability_samples:]
            median = np.median(self.history,axis=0)
            ready = len(self.history)==cfg.stability_samples and max(
                np.max(np.linalg.norm(p-median,axis=1))/diag for p in self.history)<=cfg.maximum_corner_jitter
            if ready:
                self.last_good = dict(proposal,corners=median.tolist(),last_good_frame=frame_id)
                return dict(self.last_good,frame_id=frame_id,epoch=self.epoch,usable=True,state="stable",
                            valid_until=frame_id+cfg.maximum_hold_frames,physical_axes="ambiguous",camera_epoch=self.camera_epoch)
        if self.last_good and frame_id-self.last_good["last_good_frame"]<=cfg.maximum_hold_frames:
            scale=min(1.,cfg.fitting_width/image.shape[1])
            im=frame_evidence.fitting_image if frame_evidence is not None else cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
            held_evidence=perimeter_evidence(im,np.array(self.last_good['corners'],np.float32)*scale,frame_evidence=frame_evidence)
            if not all(s>=cfg.minimum_straight_span or n>=.75 for s,n in zip(
                    held_evidence['straight_span_by_side'],held_evidence['neutral_border_support_by_side'])):
                self.history=[]
                self.last_good=None
                return dict(proposal,frame_id=frame_id,epoch=self.epoch,usable=False,state='unknown',
                            reason='held_perimeter_unverified',valid_until=frame_id,
                            physical_axes='unknown',camera_epoch=self.camera_epoch)
            return dict(self.last_good,frame_id=frame_id,epoch=self.epoch,usable=True,state="held",
                        reason="short_observation_gap",valid_until=self.last_good["last_good_frame"]+cfg.maximum_hold_frames,
                        physical_axes="ambiguous",camera_epoch=self.camera_epoch)
        if self.last_good:
            self.history=[]
            self.last_good=None
        return dict(proposal,frame_id=frame_id,epoch=self.epoch,usable=False,state="unknown",
                    reason="camera_cut" if cut else proposal["reason"] if not proposal["usable_proposal"] else "awaiting_stability",
                    valid_until=frame_id,physical_axes="unknown",camera_epoch=self.camera_epoch)


class AmbiguousPlane:
    """Reject only if outside BOTH possible assignments; no unique mm coordinates."""
    def __init__(self, corners, segment_id):
        self.segment_id = segment_id
        self.axis_source = "ambiguous_conservative"
        self.planes = [build_plane({"corners":corners,"edge01_axis":axis},segment_id) for axis in ("length","width")]

    def project(self,x,y):
        # Internal hypothesis used by the common interface, never a measured 3D point.
        return self.planes[0].project(x,y)

    def outside_distance(self,x,y):
        return min(p.outside_distance(x,y) for p in self.planes)


class AutoGeometryTimeline:
    def __init__(self,samples):
        self.samples=samples
        self.frames=[s["frame_id"] for s in samples]
        if self.frames!=sorted(set(self.frames)):
            raise ValueError("Nonchronological geometry samples")
        self.planes={s["frame_id"]:AmbiguousPlane(s["corners"],"auto:"+str(s["epoch"]))
                     for s in samples if s["usable"]}

    def sample(self,frame_id):
        index=bisect.bisect_right(self.frames,frame_id)-1
        return self.samples[index] if index>=0 else None

    def lookup(self,frame_id):
        sample=self.sample(frame_id)
        return self.planes.get(sample["frame_id"]) if sample and frame_id<=sample["valid_until"] else None

    def interval_id(self,frame_id):
        sample=self.sample(frame_id)
        return "camera:"+str(sample.get("camera_epoch",0)) if sample else None
