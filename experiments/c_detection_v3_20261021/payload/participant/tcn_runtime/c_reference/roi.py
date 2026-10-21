"""Automatic retry ROI; no event labels or manual box inputs."""
import numpy as np

def automatic_roi(points,width,height,minimum_fraction=.6,margin_fraction=.1):
    """Aspect-preserving ROI from coarse automatic observations, no labels.

    This is an offline two-pass experiment: its region can depend on later
    coarse predictions and has not been accepted for production inference.
    """
    if width<=0 or height<=0 or not 0<minimum_fraction<=1 or margin_fraction<0:
        raise ValueError('Invalid automatic ROI configuration')
    xy=np.asarray(points,float)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<5:return None
    if not np.isfinite(xy).all() or np.any(xy<0) or np.any(xy[:,0]>=width) or np.any(xy[:,1]>=height):
        raise ValueError('Invalid coarse automatic points')
    center=np.median(xy,axis=0);span=np.ptp(xy,axis=0)
    fraction=min(1.,max(minimum_fraction,(span[0]+2*margin_fraction*width)/width,
        (span[1]+2*margin_fraction*height)/height))
    w,h=round(width*fraction),round(height*fraction)
    x=max(0,min(width-w,round(center[0]-w/2)));y=max(0,min(height-h,round(center[1]-h/2)))
    return [x,y,w,h]
