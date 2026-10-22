"""Lazy evidence scoped to one immutable sampled BGR frame and fitting size."""
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import time
import cv2
import numpy as np


class TimingProfile:
    def __init__(self):self.rows={}
    def add(self,name,seconds=0.):
        row=self.rows.setdefault(name,dict(calls=0,seconds=0.))
        row['calls']+=1;row['seconds']+=seconds
    @contextmanager
    def span(self,name):
        start=time.perf_counter()
        try:yield
        finally:self.add(name,time.perf_counter()-start)
    def snapshot(self):
        return dict(inclusive_timings_not_additive=True,operations={k:dict(calls=v['calls'],seconds=round(v['seconds'],9)) for k,v in sorted(self.rows.items())})


def profiled(name):
    def decorator(fn):
        @wraps(fn)
        def call(*args,**kwargs):
            context=kwargs.get('frame_evidence')
            if context is None:return fn(*args,**kwargs)
            with context.profile.span(name):return fn(*args,**kwargs)
        return call
    return decorator


class FrameEvidence:
    def __init__(self,image,fitting_width=640,profile=None):
        if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
            raise ValueError('Expected original sampled BGR uint8 frame')
        self.image=image
        self.scale=min(1.,fitting_width/image.shape[1])
        self.profile=profile if profile is not None else TimingProfile()
        self.cache={};self.rois={};self.perimeters={}
        self.profile.add('sampled_frames')

    def get(self,name,fn):
        if name not in self.cache:
            with self.profile.span(name):self.cache[name]=fn()
        return self.cache[name]

    @property
    def fitting_image(self):return self.get('resize_fitting',lambda:cv2.resize(self.image,(round(self.image.shape[1]*self.scale),round(self.image.shape[0]*self.scale))))
    @property
    def fitting_gray(self):return self.get('gray_fitting',lambda:cv2.cvtColor(self.fitting_image,cv2.COLOR_BGR2GRAY))
    @property
    def edges(self):return self.get('canny',lambda:cv2.Canny(self.fitting_gray,60,160))
    @property
    def edge_points(self):
        def build():
            y,x=np.nonzero(self.edges)
            return np.column_stack([x,y]).astype(np.float32)
        return self.get('edge_points',build)
    @property
    def edge_distance(self):return self.get('edge_distance',lambda:cv2.distanceTransform(255-self.edges,cv2.DIST_L2,3))
    @property
    def hsv(self):return self.get('hsv_fitting',lambda:cv2.cvtColor(self.fitting_image,cv2.COLOR_BGR2HSV))
    @property
    def neutral_distance(self):
        def build():
            hsv=self.hsv;neutral=(hsv[:,:,1]<75)&(hsv[:,:,2]>130)
            return cv2.distanceTransform((~neutral).astype(np.uint8)*255,cv2.DIST_L2,3)
        return self.get('neutral_distance',build)
    @property
    def segments(self):
        def build():
            lsd=self.get('lsd',lambda:cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(self.fitting_gray)[0])
            hough=self.get('hough_full',lambda:cv2.HoughLinesP(self.edges.copy(),1,np.pi/720,20,minLineLength=20,maxLineGap=12))
            available=[x.reshape(-1,2,2) for x in (lsd,hough) if x is not None]
            return np.concatenate(available) if available else np.empty((0,2,2))
        return self.get('segments',build)
    def roi_hough(self,roi):
        # Hough on a different mask is a different operation. Never replace it
        # by globally detected segments merely because Canny is shared.
        key=(roi.shape,roi.dtype.str,roi.tobytes())
        if key not in self.rois:
            with self.profile.span('hough_roi'):
                self.rois[key]=cv2.HoughLinesP(cv2.bitwise_and(self.edges,roi),1,np.pi/720,20,minLineLength=20,maxLineGap=12)
        else:self.profile.add('hough_roi_cache_hits')
        return self.rois[key]
    @property
    def scene_thumb(self):
        def build():
            gray=self.get('gray_full',lambda:cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY))
            return cv2.resize(gray,(64,36)).astype(float)
        return self.get('scene_thumb',build)
    def perimeter_key(self,corners,segments):
        q=np.asarray(corners)
        common=segments is None or segments is self.cache.get('segments')
        return (q.dtype.str,q.shape,q.tobytes(),0 if common else id(segments))
    def perimeter_get(self,key):
        if key not in self.perimeters:return None
        self.profile.add('perimeter_cache_hits')
        return deepcopy(self.perimeters[key])
    def perimeter_put(self,key,value):self.perimeters[key]=deepcopy(value)
