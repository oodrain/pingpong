"""Video-only decoder compatibility for duplicate first PTS in OpenCV 4.10.

Repair is permitted only for a duplicate first PTS followed by consistent initial
positive intervals. Later timing gaps and all frames are preserved unchanged.
"""
import cv2
import numpy as np

NATIVE_CAPTURE=cv2.VideoCapture


def checked_timestamps(times,fps):
    times=np.array(times,dtype=float)
    if not len(times) or not np.isfinite(times).all():raise ValueError('Invalid decoded timestamp sequence')
    bad=np.flatnonzero(np.diff(times)<=0)+1;repair=None
    if len(bad):
        if len(times)<4 or list(bad)!=[1] or times[0]!=times[1] or fps<=0:
            raise ValueError('Nonmonotonic PTS is not a supported startup-only decoder error')
        dt=np.diff(times[1:]);initial=dt[:min(5,len(dt))];cadence=float(np.median(initial))
        # Container average FPS can include edit-list/tail duration. Actual
        # initial decoded PTS cadence is authoritative. Later gaps are retained.
        if cadence<=0 or not np.allclose(initial,cadence,rtol=1e-6,atol=1e-6):
            raise ValueError('Duplicate startup PTS repair requires consistent initial intervals')
        corrected=times[1]-cadence
        if corrected < -1e-6:raise ValueError('Startup PTS repair would yield a negative timestamp')
        repair={'reason':'Duplicate first PTS followed by consistent initial OpenCV intervals',
                'original_first_pts_ms':float(times[0]),'corrected_first_pts_ms':float(max(0,corrected)),
                'actual_cadence_ms':cadence,'container_average_fps':float(fps),
                'other_pts_changed':0,'frames_reordered':False}
        times[0]=max(0,corrected)
    if not np.all(np.diff(times)>0):raise ValueError('Timestamp compatibility check failed')
    return times.tolist(),repair


class StartupPTSCapture:
    repairs=[]
    def __init__(self,path):
        probe=NATIVE_CAPTURE(str(path));times=[]
        if not probe.isOpened():raise ValueError('Cannot open video for timestamp compatibility preflight')
        fps=probe.get(cv2.CAP_PROP_FPS);declared=int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ok,_=probe.read()
            if not ok:break
            times.append(probe.get(cv2.CAP_PROP_POS_MSEC))
        probe.release()
        if len(times)!=declared:raise ValueError('Video frame count mismatch in decoder preflight')
        self.times,repair=checked_timestamps(times,fps)
        if repair:self.repairs.append(repair)
        self.cap=NATIVE_CAPTURE(str(path));self.frame_id=-1
    def isOpened(self):return self.cap.isOpened()
    def read(self):
        ok,image=self.cap.read()
        if ok:self.frame_id+=1
        return ok,image
    def get(self,key):
        if key==cv2.CAP_PROP_POS_MSEC and 0<=self.frame_id<len(self.times):return self.times[self.frame_id]
        return self.cap.get(key)
    def release(self):self.cap.release()
