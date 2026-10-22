"""Projected net-zone hypotheses from automatic table geometry only."""
import cv2
import numpy as np
from .automatic_ballistic_contact import camera_hypotheses
from development.detection.tt3d_camera_estimation import object_corners


def net_zone_evidence(tracks,samples,seed,image_size):
    geometry=max((s for s in samples if s['frame_id']<=seed),key=lambda s:s['frame_id'],default=None)
    if geometry is None or not geometry.get('usable') or seed>geometry.get('valid_until',geometry['frame_id']):
        return {'available':False,'reason':'current_geometry_unusable'}
    q=np.asarray(geometry['corners']);scale=max(float(np.linalg.norm(np.ptp(q,axis=0))),1)
    ball=tracks[seed]
    if not ball.get('detected'):return {'available':False,'reason':'ball_missing'}
    try:cameras=camera_hypotheses(tuple(q.ravel()),tuple(image_size))
    except (ValueError,cv2.error,np.linalg.LinAlgError):return {'available':False,'reason':'camera_estimation_failed'}
    zones=[]
    for camera in cameras:
        # Weak focal constraints may support an image-zone hypothesis, not a
        # metric ballistic veto. Never reuse a solver that failed reprojection.
        if camera['reason'] not in ('estimated','weak_focal_constraint'):continue
        if camera['reprojection_rmse_px']>3 or camera['camera_centre_m'][2]<=0:continue
        axis=camera['edge01_axis'];obj=object_corners(axis);x,y=abs(obj[0,0]),abs(obj[0,1])
        top=np.array([[0,-y,.1525],[0,y,.1525]]) if axis=='length' else np.array([[-x,0,.1525],[x,0,.1525]])
        bottom=top.copy();bottom[:,2]=0
        polygon=np.vstack([top,bottom[::-1]])
        uv=cv2.projectPoints(polygon,np.asarray(camera['rvec']),np.asarray(camera['tvec_m']),np.asarray(camera['K']),np.zeros(5))[0].reshape(4,2)
        if not np.isfinite(uv).all():continue
        xy=np.array([ball['x'],ball['y']]);signed=float(cv2.pointPolygonTest(uv.astype(np.float32),tuple(map(float,xy)),True))
        edge=uv[1]-uv[0];length=float(edge@edge)
        if length<=1e-9:continue
        nearest=uv[0]+np.clip(float((xy-uv[0])@edge)/length,0,1)*edge
        zones.append({'edge01_axis':axis,'signed_mesh_distance_fraction':signed/scale,
                      'top_segment_distance_fraction':float(np.linalg.norm(xy-nearest))/scale,
                      'camera_focal_verified':False,'projected_mesh_polygon_px':uv.tolist()})
    if not zones:return {'available':False,'reason':'no_projectable_net_hypothesis'}
    return {'available':True,'hypotheses':zones,
            'maximum_signed_mesh_distance_fraction':max(z['signed_mesh_distance_fraction'] for z in zones),
            'minimum_top_distance_fraction':min(z['top_segment_distance_fraction'] for z in zones),
            'net_contact_confirmed':False,'human_net_points_used':False}
