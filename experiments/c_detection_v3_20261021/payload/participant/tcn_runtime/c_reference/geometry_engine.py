"""A fixed two-mode geometry engine; actual RGB and automatic proposals only."""
from copy import deepcopy
import time
import cv2
import numpy as np
from .auto_table_geometry import GeometryConfig,CausalGeometryTracker,perimeter_evidence,order_quad,align_quad,table_diagonal
from .geometry import SecondsGeometryTracker
from .image_evidence import FrameEvidence,TimingProfile
from .table_memory import TableMemory

POLICY=dict(full_refresh_seconds=2.,light_check_seconds=.2,verified_expiry_seconds=.45,
            tracking_width=320,minimum_flow_points=12,flow_inlier_fraction=.75,
            maximum_forward_backward_error=1.,camera_motion_fraction=.005,
            maximum_full_corner_shift_fraction=.025,minimum_visible_adjacent_edges=2,
            cold_backoff_start_failures=3,cold_maximum_retry_seconds=.8,
            initial_retry_seconds=.12,stable_legacy_seconds=.4)


def unknown(f,reason):return dict(frame_id=f,usable=False,state='unknown',reason=reason,valid_until=f)


class GeometryEngine:
    def __init__(self,timestamps,size,provider,mode='safe'):
        if mode not in ('safe','tracked'):raise ValueError('Unregistered geometry mode')
        self.times=np.asarray(timestamps,float);self.size=tuple(size);self.provider=provider;self.mode=mode
        if not len(self.times) or np.any(np.diff(self.times)<=0):raise ValueError('Need actual increasing timestamps')
        self.cfg=GeometryConfig();self.cadence=CausalGeometryTracker(self.cfg);self.legacy=SecondsGeometryTracker(self.times,self.cfg)
        self.memory=TableMemory(self.times,size);self.samples=[];self.review_samples=[];self.raw=[]
        self.profile=TimingProfile();self.next_full=0;self.next_light=0.;self.anchor=None;self.gray=None
        self.last_full_time=-float('inf');self.failures=0;self.frames=0;self.model_seconds=0.;self.geometry_seconds=0.

    def expires(self,time_s):return int(np.searchsorted(self.times,time_s+1e-9,side='right')-1)

    def fast_check(self,image,q):
        scale=min(1.,POLICY['tracking_width']/image.shape[1])
        im=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        neutral=(hsv[:,:,1]<75)&(hsv[:,:,2]>130)
        white_distance=cv2.distanceTransform((~neutral).astype(np.uint8)*255,cv2.DIST_L2,3)
        edges=cv2.Canny(gray,60,160);edge_distance=cv2.distanceTransform(255-edges,cv2.DIST_L2,3)
        scaled=np.asarray(q,np.float32)*scale;visible=[];supports=[]
        for a,b in zip(scaled,np.roll(scaled,-1,axis=0)):
            points=np.rint(np.linspace(a,b,120)).astype(int);points[:,0]=np.clip(points[:,0],0,im.shape[1]-1);points[:,1]=np.clip(points[:,1],0,im.shape[0]-1)
            white=float(np.mean(white_distance[points[:,1],points[:,0]]<=2*scale/(min(1.,640/image.shape[1]))))
            edge=edge_distance[points[:,1],points[:,0]]<=2
            runs=np.diff(np.r_[0,edge.astype(int),0]);starts=np.where(runs==1)[0];ends=np.where(runs==-1)[0]
            span=float(max(ends-starts,default=0)/120)
            visible.append(bool(white>=.75 or span>=self.cfg.minimum_straight_span));supports.append(dict(white=white,span=span))
        camera_ok=False;motion=None;matches=0
        if self.gray is not None and self.gray.shape==gray.shape:
            mask=np.full(gray.shape,255,np.uint8)
            cv2.fillConvexPoly(mask,np.rint(scaled).astype(np.int32),0)
            pts=cv2.goodFeaturesToTrack(self.gray,80,.01,5,mask=mask)
            if pts is not None and len(pts)>=POLICY['minimum_flow_points']:
                nxt,status,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,pts,None)
                back,back_status,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,nxt,None)
                good=status.ravel().astype(bool)&back_status.ravel().astype(bool)&(np.linalg.norm(back-pts,axis=2).ravel()<=POLICY['maximum_forward_backward_error'])
                if sum(good)>=POLICY['minimum_flow_points']:
                    transform,inliers=cv2.estimateAffinePartial2D(pts[good],nxt[good],method=cv2.RANSAC,ransacReprojThreshold=2.)
                    if transform is not None and inliers is not None:
                        matches=int(inliers.sum());warped=np.c_[scaled,np.ones(4)]@transform.T
                        motion=float(np.linalg.norm(warped-scaled,axis=1).max()/scale)
                        camera_ok=bool(matches>=POLICY['minimum_flow_points'] and matches/sum(good)>=POLICY['flow_inlier_fraction'] and motion<=POLICY['camera_motion_fraction']*table_diagonal(np.asarray(q)))
        self.gray=gray
        adjacent=any(visible[i] and visible[(i+1)%4] for i in range(4))
        return bool(camera_ok and sum(visible)>=POLICY['minimum_visible_adjacent_edges'] and adjacent),dict(visible=visible,support=supports,camera_ok=camera_ok,motion_px=motion,flow_inliers=matches)

    def process(self,frame,rgb):
        if frame!=self.frames or frame>=len(self.times):raise ValueError('Geometry frame sequence mismatch')
        self.frames+=1;now=float(self.times[frame]);start=time.perf_counter()
        image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        cut=self.memory.observe_frame(frame,image)
        if cut:
            self.anchor=None;self.gray=None;self.next_full=frame;self.failures=0
        full=frame>=self.next_full
        if self.mode=='tracked' and self.anchor is not None:
            if now-self.last_full_time>=POLICY['full_refresh_seconds']-1e-9:full=True
            elif now>=self.next_light-1e-9:
                with self.profile.span('light_tracking'):
                    okay,evidence=self.fast_check(image,self.anchor['corners'])
                self.next_light=now+POLICY['light_check_seconds']
                if okay:
                    g=deepcopy(self.anchor);g.update(frame_id=frame,usable=True,state='light_tracked',reason='current_visible_edges_and_static_camera',
                        valid_until=self.expires(min(now+POLICY['verified_expiry_seconds'],self.last_full_time+POLICY['full_refresh_seconds'])),
                        corner_uncertainty_px=.003*table_diagonal(np.asarray(g['corners'])),light_evidence=evidence)
                    self.samples.append(g);self.review_samples.append(deepcopy(g));full=False
                else:full=True
            else:full=False
        if full:
            e=FrameEvidence(image,self.cfg.fitting_width,self.profile)
            before=time.perf_counter();proposal,model_seconds=self.provider(frame,image,e)
            self.model_seconds+=model_seconds
            self.raw.append(dict(frame_id=frame,proposal=deepcopy(proposal)))
            old_anchor=self.anchor
            cadence=self.cadence.update(frame,image,deepcopy(proposal),frame_evidence=e)
            g=self.legacy.update(frame,image,deepcopy(proposal),frame_evidence=e)
            if self.mode=='tracked' and old_anchor is not None and proposal.get('usable_proposal') and proposal.get('corners'):
                q=align_quad(proposal['corners'],np.asarray(old_anchor['corners']))
                compatible=np.linalg.norm(q-np.asarray(old_anchor['corners']),axis=1).max()<=POLICY['maximum_full_corner_shift_fraction']*table_diagonal(q)
                ev=perimeter_evidence(e.fitting_image,np.asarray(old_anchor['corners'],np.float32)*e.scale,frame_evidence=e)
                verified=all(s>=self.cfg.minimum_straight_span or n>=.75 for s,n in zip(ev['straight_span_by_side'],ev['neutral_border_support_by_side']))
                if compatible and verified:
                    g=deepcopy(old_anchor);g.update(frame_id=frame,usable=True,state='stable',reason='fresh_automatic_proposal_reverified_prior_table',
                        confirmed_timestamp_s=now,last_good_frame=frame,valid_until=self.expires(now+.8),current_perimeter=ev)
            self.samples.append(g)
            aux=self.memory.update(frame,g,proposal,lambda q:perimeter_evidence(e.fitting_image,np.asarray(q,np.float32)*e.scale,frame_evidence=e))
            self.review_samples.append(aux)
            if self.mode=='tracked' and g.get('usable'):
                self.anchor=deepcopy(g);self.last_full_time=now;self.next_light=now+POLICY['light_check_seconds'];self.failures=0
                self.fast_check(image,g['corners'])
                self.next_full=self.expires(now+POLICY['full_refresh_seconds'])
            else:
                if self.mode=='tracked':self.anchor=None;self.failures+=1
                interval=POLICY['stable_legacy_seconds'] if cadence['state']=='stable' else POLICY['initial_retry_seconds']
                if self.mode=='tracked' and self.failures>=POLICY['cold_backoff_start_failures']:
                    interval=min(POLICY['cold_maximum_retry_seconds'],interval*2**min(3,self.failures-POLICY['cold_backoff_start_failures']+1))
                fps=1/np.median(np.diff(self.times)) if len(self.times)>1 else 25.
                self.next_full=frame+max(1,round(fps*interval))
        self.geometry_seconds+=time.perf_counter()-start-(model_seconds if full else 0.)

    def result(self):
        if self.frames!=len(self.times):raise ValueError('Incomplete geometry video')
        return dict(samples=self.samples,review_samples=self.review_samples,cuts=self.memory.cuts,
                    raw_proposals=self.raw,profile=self.profile.snapshot(),table_model_sec=self.model_seconds,
                    table_geometry_sec=self.geometry_seconds,full_calls=len(self.raw),frames=self.frames)
