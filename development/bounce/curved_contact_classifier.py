"""Small table-event classifier from automatic curved-flight evidence.

No IDs, labels, reviewed geometry, pose-cache absence, or manual coordinates
are predictor features. Rectified edge components remain image evidence.
"""
import numpy as np
from .contact_consensus import cosine

FEATURES=['image_direction_cosine','log_speed_ratio','relative_velocity_change',
          'preserved_projected_edge_fraction','curvature_relative_to_speed','fit_rmse_px',
          'signed_projection_distance_fraction','geometry_missing','absolute_subframe_offset','bic_gain_scaled']
EXTENDED_FEATURES=FEATURES+['bic_gain_over_smooth_cubic_scaled']
CONTEXT_FEATURES=EXTENDED_FEATURES+['minimum_rectified_centreline_distance','maximum_rectified_centreline_distance',
                                 'wide_flight_missing','absolute_wide_contact_offset','wide_to_short_rmse_log_ratio']
LEGACY_PHYSICS_FEATURES=CONTEXT_FEATURES+['automatic_ballistic_missing','automatic_ballistic_rmse_log','automatic_ballistic_opposing_velocity_margin']
PHYSICS_FEATURES=LEGACY_PHYSICS_FEATURES+['maximum_ballistic_opposing_velocity_margin']
NET_FEATURES=PHYSICS_FEATURES+['net_projection_hypothesis_missing','net_mesh_signed_distance_fraction','net_top_distance_fraction']


def features(proposal,position,corners,feature_names=FEATURES):
    unit=proposal['nominal_interval_s'];a=np.array(proposal['velocity_before_px_s'])*unit
    b=np.array(proposal['velocity_after_px_s'])*unit;before=float(np.linalg.norm(a));after=float(np.linalg.norm(b))
    edge_a=proposal['table_axis_image_components_before'];edge_b=proposal['table_axis_image_components_after']
    preserved=max((min(abs(x),abs(y))/(before+after+2) if x*y>0 else 0) for x,y in zip(edge_a,edge_b))
    scale=max(float(np.linalg.norm(np.ptp(np.asarray(corners,float),axis=0))),1)
    available=position.get('available',False);d=cosine(a,b)
    result=[0 if d is None else d,np.log((after+2)/(before+2)),float(np.linalg.norm(b-a))/(before+after+2),
        preserved,float(np.linalg.norm(proposal['shared_image_curvature']))/(before+after+2),proposal['fit_rmse_px'],
        np.clip(position.get('signed_distance_px',0)/scale,-1,1) if available else 0,
        not available,abs(proposal['contact_offset_intervals']),np.clip(proposal['bic_gain']/40,-2,4)]
    if feature_names==EXTENDED_FEATURES:result.append(np.clip(proposal['bic_gain_over_smooth_cubic']/40,-4,4))
    elif feature_names!=FEATURES:raise ValueError('Unsupported curved feature schema')
    return np.array(result,float)


def fit(X,y,regularization=1.,feature_names=FEATURES):
    X=np.asarray(X,float);y=np.asarray(y,float)
    if feature_names not in (FEATURES,EXTENDED_FEATURES,CONTEXT_FEATURES,PHYSICS_FEATURES,LEGACY_PHYSICS_FEATURES,NET_FEATURES) or X.ndim!=2 or X.shape!=(len(y),len(feature_names)) or not np.isfinite(X).all() or set(y)!={0.,1.}:
        raise ValueError('Invalid curved event training matrix')
    mean=X.mean(0);scale=X.std(0);scale[scale<1e-8]=1
    A=np.column_stack((np.ones(len(y)),(X-mean)/scale));w=np.zeros(A.shape[1]);penalty=np.diag([0]+[regularization]*len(feature_names))
    weights=np.where(y==1,len(y)/(2*sum(y)),len(y)/(2*sum(1-y)))
    for _ in range(80):
        p=1/(1+np.exp(-np.clip(A@w,-35,35)));gradient=A.T@(weights*(p-y))+penalty@w
        hessian=A.T@((weights*p*(1-p))[:,None]*A)+penalty+np.eye(len(w))*1e-9
        step=np.linalg.solve(hessian,gradient);w-=step
        if np.linalg.norm(step)<1e-8:break
    return {'schema_version':1,'features':feature_names,'mean':mean.tolist(),'scale':scale.tolist(),
            'weights':w.tolist(),'regularization':regularization,'score_is_calibrated_probability':False}


def score_matrix(model,X):
    if model['features'] not in (FEATURES,EXTENDED_FEATURES,CONTEXT_FEATURES,PHYSICS_FEATURES,LEGACY_PHYSICS_FEATURES,NET_FEATURES):raise ValueError('Curved classifier feature order mismatch')
    if model.get('kind')=='shallow_boosted_trees':
        from .curved_tree_classifier import score_matrix as tree_score
        return tree_score(model,X)
    X=np.atleast_2d(np.asarray(X,float));mean=np.asarray(model['mean']);scale=np.asarray(model['scale']);w=np.asarray(model['weights'])
    n=len(model['features'])
    if X.shape[1]!=n or mean.shape!=(n,) or scale.shape!=(n,) or w.shape!=(n+1,) or not np.isfinite(X).all() or not np.isfinite([*mean,*scale,*w]).all() or min(scale)<=0:raise ValueError('Invalid curved classifier inputs')
    return 1/(1+np.exp(-np.clip(np.column_stack((np.ones(len(X)),(X-mean)/scale))@w,-35,35)))
