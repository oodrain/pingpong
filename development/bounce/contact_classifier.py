"""Small supervised contact-candidate model using automatic evidence only.

Input contains no video identity, human axis, spatial annotation or event label.
Model scores are not calibrated contact probabilities.
"""
import json
from pathlib import Path
import numpy as np

FEATURES=['direction_cosine','log_speed_ratio','relative_velocity_change','wrist_visit',
          'near_wrist_ratio','wrist_evidence_missing','person_entry_exit','outside_projection',
          'geometry_missing','context_incomplete']
THRESHOLD=.8

def features(case):
    d=case.get('direction') or {};speeds=d.get('speeds_px_per_nominal_interval',[0,0])
    before,after=speeds;cos=d.get('image_direction_cosine')
    visits=case['wrist_visit']['visits']+case.get('full_frame_wrist_visit',{}).get('visits',[])
    ratios=[v['distance_in_forearms_at_closest'] for v in visits]
    seed=case['seed_frame'];position=next((p for p in case['local_positions'] if p['frame_id']==seed),{})
    # Delta from the fitted image vectors, without selecting a physical table axis.
    cosine=0 if cos is None else max(-1.,min(1.,cos))
    delta=np.sqrt(max(0.,before**2+after**2-2*before*after*cosine))
    return np.array([cosine,np.log((after+2)/(before+2)),delta/(before+after+2),bool(visits),
        min(ratios) if ratios else 2.,not ratios,case['person_visit']['supported'],
        bool(position.get('available') and position.get('outside_projection')),not position.get('available',False),
        bool(case['window_reasons'])],np.float64)

def fit(X,y,regularization=1.):
    X=np.asarray(X,float);y=np.asarray(y,float)
    if X.ndim!=2 or X.shape[1]!=len(FEATURES) or len(X)!=len(y) or not np.isfinite(X).all():raise ValueError('Invalid training matrix')
    if set(y)!={0.,1.}:raise ValueError('Both labelled classes required')
    mean=X.mean(0);scale=X.std(0);scale[scale<1e-8]=1
    A=np.column_stack([np.ones(len(X)),(X-mean)/scale]);w=np.zeros(A.shape[1])
    weights=np.where(y==1,len(y)/(2*sum(y)),len(y)/(2*sum(1-y)))
    penalty=np.diag([0]+[regularization]*X.shape[1])
    for _ in range(80):
        z=np.clip(A@w,-35,35);p=1/(1+np.exp(-z));gradient=A.T@(weights*(p-y))+penalty@w
        hessian=A.T@((weights*p*(1-p))[:,None]*A)+penalty+np.eye(len(w))*1e-9
        step=np.linalg.solve(hessian,gradient);w-=step
        if np.linalg.norm(step)<1e-8:break
    return {'schema_version':1,'features':FEATURES,'mean':mean.tolist(),'scale':scale.tolist(),
            'weights':w.tolist(),'threshold':THRESHOLD,'regularization':regularization,
            'score_is_calibrated_probability':False,'production_exclusion_enabled':False}

def score_matrix(model,X):
    if model['features']!=FEATURES:raise ValueError('Classifier feature order mismatch')
    X=np.atleast_2d(np.asarray(X,float));mean=np.asarray(model['mean']);scale=np.asarray(model['scale']);w=np.asarray(model['weights'])
    if X.shape[1]!=len(FEATURES) or mean.shape!=(len(FEATURES),) or scale.shape!=mean.shape or w.shape!=(len(FEATURES)+1,):raise ValueError('Classifier dimensions mismatch')
    if not np.isfinite(X).all() or not np.isfinite([*mean,*scale,*w]).all() or min(scale)<=0:raise ValueError('Invalid classifier parameters')
    z=np.column_stack([np.ones(len(X)),(X-mean)/scale])@w
    return 1/(1+np.exp(-np.clip(z,-35,35)))

def predict(model,case):return float(score_matrix(model,features(case))[0])

def apply_model(model,cases):
    for case in cases:
        value=predict(model,case)
        valid=bool(case.get('direction') and not case['window_reasons'])
        evidence=bool(case['wrist_visit']['supported'] or case.get('full_frame_wrist_visit',{}).get('supported')
            or case['person_visit']['supported'] or case['racket_visit']['supported'])
        flag=bool(valid and evidence and value>=model['threshold'])
        case['learned_contact']={'score':value,'threshold':model['threshold'],'complete_automatic_context':valid,
            'gesture_evidence_available':evidence,'hit_suggestion':flag,'calibrated_probability':False,
            'training_candidates':model.get('training_provenance',{}).get('candidate_count'),
            'independent_validation_passed':False,'adopt':False}
        case['decision']['learned_hit_supplement']=flag
        if flag:
            case['decision']['review_hit_suggestion']=True;case['event']='racket_contact_candidate'
    return cases

def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
