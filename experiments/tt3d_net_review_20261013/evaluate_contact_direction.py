"""Fixed direction/wrist gate on existing inference; GT opened after decisions."""
import argparse,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from development.bounce.contact_direction import direction_evidence,fuse_direction_wrist
from development.evaluation.evaluate_landings import evaluate
from fit_contact_pilot import sha

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(p,r):p.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def run(root,pilot,out):
    project=Path(__file__).resolve().parents[2]
    wrist_path=pilot/'public_auxiliary_wrists_01/wrist_evidence.json';screen_path=pilot/'auxiliary_screening_01/screening.json'
    inference_path=pilot/'public_regression_01/inference.json'
    geometry_path=project/'outputs/experiments/tt3d_geometry_refinement_20261013/runtime_check_02/geometry_samples.jsonl'
    reference_paths=[project/'outputs/experiments/tt3d_net_review_20261013'/p/'net_references.jsonl' for p in ('review_pack_03','review_pack_04')]
    wrist=read(wrist_path);screen=read(screen_path);inference=read(inference_path);samples=lines(geometry_path)
    if sha(geometry_path)!=inference['sources'][str(geometry_path)]:raise ValueError('Changed geometry')
    references={}
    for path in reference_paths:
        for row in lines(path):references[row['case_id']]=row
    out.mkdir(parents=True,exist_ok=False);cases=[]
    for w in wrist['cases']:
        vid=w['video_id'];seed=w['seed_frame'];video=next(v for v in inference['videos'] if v['video_id']==vid)
        proposal=next(p for p in video['proposal_result']['proposals'] if p['frame_id']==seed)
        geometry=next(s for s in samples if s['group']=='public' and s['video_id']==vid and s['frame_id']==video['reference_frame'] and s.get('usable'))
        # Only transfer reviewed axis if corresponding physical edge directions agree
        # between ALL reviewed polygons and the automatic reference. No screen-size rule.
        import numpy as np
        q=np.asarray(geometry['corners']);base=np.array([q[1]-q[0],q[3]-q[0]]);base/=np.linalg.norm(base,axis=1)[:,None]
        labels=[];ref_ids=[]
        for r in references.values():
            if r.get('video_id')!=vid or not r.get('axis_confirmed') or not r.get('corner_confirmed'):continue
            if not r.get('edge01_axis'):continue
            p=np.asarray(r['corners']);a=np.array([p[1]-p[0],p[3]-p[0]]);a/=np.linalg.norm(a,axis=1)[:,None]
            # Axis is unoriented. Require 0->1 to correspond unambiguously, within 30deg.
            correspondence=np.abs(a@base.T)
            if not (correspondence[0,0]>.866 and correspondence[1,1]>.866 and correspondence[0,0]>correspondence[0,1] and correspondence[1,1]>correspondence[1,0]):continue
            labels.append(r['edge01_axis']);ref_ids.append(r['case_id'])
        axis=labels[0] if labels and len(set(labels))==1 else None
        d=direction_evidence(proposal,geometry['corners'],axis)
        cases.append({'video_id':vid,'seed_frame':seed,'angle':w['angle'],'direction':d,'wrist_status':w['status'],
            'axis_reference_ids':ref_ids,'axis_source':'reviewed_edge_correspondence' if axis else 'unknown',
            'geometry_reference_frame':geometry['frame_id'],'geometry_valid_until':geometry.get('valid_until'),
            'decision':fuse_direction_wrist(d,w)})
    sources=[wrist_path,screen_path,inference_path,geometry_path,*reference_paths]
    rule=project/'development/bounce/contact_direction.py'
    save(out/'direction_evidence.json',{'sources':{str(p):sha(p) for p in sources},'rule_sha256':sha(rule),'runner_sha256':sha(__file__),
        'known_public_regression_not_independent_holdout':True,'labels_used_for_inference':False,
        'thresholds_retuned':False,'production_events_modified':False,'status_counts':dict(Counter(c['decision']['event'] for c in cases)),
        'cases':cases})
    excluded={(c['video_id'],c['seed_frame']) for c in cases if c['decision']['hypothetical_auxiliary_exclusion']}
    selected=screen['selected_review_candidates'];remaining=[p for p in selected if (p['video_id'],p['frame_id']) not in excluded]
    public=root/'participant/participant/pingpang/public_data';gt=lines(public/'gt_public.jsonl')
    manifest=read(root/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1/selected_manifest.json')['videos']
    old=read(pilot/'public_auxiliary_wrists_01/evaluation.json')
    score=evaluate(screen['primary_predictions']+remaining,gt,manifest,exempt_invalid=False)
    save(out/'evaluation.json',{'direction_evidence_sha256':sha(out/'direction_evidence.json'),'gt_sha256':sha(public/'gt_public.jsonl'),
        'scores':{**old['scores'],'hypothetical_direction_and_wrist_exclusion':score},'exclusion_enabled':False,'excluded_auxiliary_count':len(excluded),
        'interpretation':'Candidate-pool diagnostic; no verified racket labels and no final event accuracy claim.'})
    print(json.dumps({'statuses':dict(Counter(c['decision']['event'] for c in cases)),'axis_known':sum(c['axis_source']!='unknown' for c in cases),
        'excluded':sorted(excluded),'score':score['overall']},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','pilot','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.pilot.resolve(),a.output.resolve())
