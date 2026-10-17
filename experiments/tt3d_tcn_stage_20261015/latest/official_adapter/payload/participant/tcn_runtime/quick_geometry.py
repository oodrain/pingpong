"""One fixed conservative current-mask/current-border verification recipe."""
from copy import deepcopy
import cv2
import numpy as np
from .auto_table_geometry import order_quad,table_diagonal,perimeter_evidence
from .table_border_lines import border_extension,image_segments

class QuickGeometry:
 def __init__(self):self.gray=None;self.accepted=0;self.attempted=0
 def camera(self,image,q):
  scale=min(1.,320/image.shape[1]);small=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
  gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY);previous=self.gray;self.gray=gray
  if previous is None or previous.shape!=gray.shape:return False
  mask=np.full(gray.shape,255,np.uint8);cv2.fillConvexPoly(mask,np.rint(q*scale).astype(np.int32),0)
  pts=cv2.goodFeaturesToTrack(previous,80,.01,5,mask=mask)
  if pts is None or len(pts)<12:return False
  nxt,status,_=cv2.calcOpticalFlowPyrLK(previous,gray,pts,None)
  back,bs,_=cv2.calcOpticalFlowPyrLK(gray,previous,nxt,None)
  good=status.ravel().astype(bool)&bs.ravel().astype(bool)&(np.linalg.norm(back-pts,axis=2).ravel()<=1.)
  if sum(good)<12:return False
  matrix,inliers=cv2.estimateAffinePartial2D(pts[good],nxt[good],method=cv2.RANSAC,ransacReprojThreshold=2.)
  if matrix is None or inliers is None or int(inliers.sum())<12 or float(inliers.mean())<.75:return False
  moved=np.c_[q*scale,np.ones(4)]@matrix.T
  return bool(np.linalg.norm(moved-q*scale,axis=1).max()/scale<=.005*table_diagonal(q))
 def propose(self,image,probability,e,prior,now):
  self.attempted+=1;cfg=e.cfg if hasattr(e,'cfg') else None
  from .auto_table_geometry import GeometryConfig
  cfg=cfg or GeometryConfig()
  q=order_quad(prior['corners']) if prior and prior.get('corners') else None
  if q is None:
   # Establish the camera history even while the original C builds its anchor.
   small=cv2.resize(image,(round(image.shape[1]*min(1.,320/image.shape[1])),round(image.shape[0]*min(1.,320/image.shape[1]))))
   self.gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY);return None
  camera_ok=self.camera(image,q)
  if not camera_ok or now-float(prior['confirmed_timestamp_s'])>.8+1e-9:return None
  im=e.fitting_image;h,w=im.shape[:2];scaled=q*e.scale
  mask=(cv2.resize(np.asarray(probability,np.float32),(w,h))>cfg.mask_threshold).astype(np.uint8)*255
  mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
  contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);contours=sorted(contours,key=cv2.contourArea,reverse=True)
  if not contours:return None
  largest=cv2.contourArea(contours[0]);centre=contours[0][:,0,:].mean(axis=0)
  selected=[c for c in contours if cv2.contourArea(c)>=max(w*h*.001,largest*.05) and np.linalg.norm(c[:,0,:].mean(axis=0)-centre)<.5*w]
  if not selected:return None
  cleaned=np.zeros_like(mask);cv2.drawContours(cleaned,selected,-1,255,-1)
  polygon=np.zeros_like(mask);cv2.fillConvexPoly(polygon,np.rint(scaled).astype(np.int32),255)
  overlap=cv2.countNonZero(cv2.bitwise_and(polygon,cleaned));area=cv2.countNonZero(polygon);ma=cv2.countNonZero(cleaned)
  iou=overlap/max(1,area+ma-overlap);coverage=overlap/max(1,ma)
  if iou<cfg.minimum_mask_iou or coverage<cfg.minimum_visible_mask_coverage:return None
  supports=[]
  for a,b in zip(scaled,np.roll(scaled,-1,axis=0)):
   pts=np.rint(np.linspace(a,b,80)).astype(int);pts[:,0]=np.clip(pts[:,0],0,w-1);pts[:,1]=np.clip(pts[:,1],0,h-1)
   supports.append(float(np.mean(e.edge_distance[pts[:,1],pts[:,0]]<=3)))
  if min(supports)<cfg.minimum_side_support:return None
  ev=perimeter_evidence(im,scaled,frame_evidence=e)
  if not all(s>=cfg.minimum_straight_span or n>=.75 for s,n in zip(ev['straight_span_by_side'],ev['neutral_border_support_by_side'])):return None
  extension=border_extension(scaled,image_segments(im,frame_evidence=e))
  if max(extension)>cfg.maximum_border_extension:return None
  af=cv2.contourArea(scaled)/(w*h)
  if not cfg.minimum_area<=af<=cfg.maximum_area:return None
  self.accepted+=1
  return dict(corners=q.tolist(),usable_proposal=True,reason='current_mask_four_borders_and_camera_verified',
   mask_iou=float(iou),mask_coverage=float(coverage),edge_support=float(np.mean(supports)),edge_support_by_side=supports,
   area_fraction=float(af),quality=float(iou*.6+np.mean(supports)*.4),fitting_source='verified_previous_table_current_segmentation',
   image_perimeter=ev,border_diagnostic=dict(base_border_extension_by_side=extension),quick_current_segmentation=True)
