"""Label-free audit of geometry failures and paired coverage changes.

Operational categories describe extraction/tracker observations, not verified
scene causes or corner accuracy. No event GT is loaded by this command.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
from PIL import Image, ImageDraw


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(baseline, candidate=None):
    quality=rows(baseline/'geometry_quality.jsonl')
    samples=defaultdict(list)
    for s in rows(baseline/'geometry_samples.jsonl'):
        samples[(s['group'],s['video_id'])].append(s)
    failures=[]
    for q in quality:
        if q['group']!='train' or q['any_usable']:
            continue
        ss=samples[(q['group'],q['video_id'])]
        good=sum(s['raw_proposal']['usable_proposal'] for s in ss)
        reasons=Counter(s['raw_proposal']['reason'] for s in ss)
        resets=max(s['epoch'] for s in ss)
        if len(ss)<3:category='too_few_samples'
        elif good==0:category='no_qualified_proposal'
        elif resets>0:category='proposal_with_geometry_resets'
        else:category='intermittent_proposals_without_stability'
        failures.append(dict(q,category=category,qualified_proposals=good,
                             proposal_reason_counts=dict(reasons),geometry_resets=resets,
                             camera_cuts=max(s.get('camera_epoch',0) for s in ss)))
    result={'failed_train_videos':len(failures),
            'categories':dict(Counter(f['category'] for f in failures)),
            'qualified_proposal_videos':sum(f['qualified_proposals']>0 for f in failures),
            'proposal_reasons':dict(sum((Counter(f['proposal_reason_counts']) for f in failures),Counter())),
            'independent_corner_accuracy_verified':False,
            'no_event_labels_used':True,
            'category_status':'operational diagnostics, not verified scene-cause labels'}
    changes=[]
    if candidate:
        old_summary=json.loads((baseline/'summary.json').read_text(encoding='utf-8-sig'))
        new_summary=json.loads((candidate/'summary.json').read_text(encoding='utf-8-sig'))
        if old_summary['model']!=new_summary['model']:
            raise ValueError('Different segmentation model/runtime provenance')
        for path,digest in old_summary['input_sha256'].items():
            if path.lower().endswith('.mp4') and new_summary['input_sha256'].get(path)!=digest:
                raise ValueError('Video inputs changed: '+path)
        cq={(q['group'],q['video_id']):q for q in rows(candidate/'geometry_quality.jsonl')}
        if set(cq)!={(q['group'],q['video_id']) for q in quality}:
            raise ValueError('Different video populations')
        for old in quality:
            new=cq[(old['group'],old['video_id'])]
            changes.append({'group':old['group'],'video_id':old['video_id'],
                            'previous_any_usable':old['any_usable'],'candidate_any_usable':new['any_usable'],
                            'previous_usable_samples':old['usable_samples'],'candidate_usable_samples':new['usable_samples']})
        result['coverage_changes']={g:{
            'newly_usable_videos':sum(c['candidate_any_usable'] and not c['previous_any_usable'] for c in changes if c['group']==g),
            'lost_usable_videos':sum(c['previous_any_usable'] and not c['candidate_any_usable'] for c in changes if c['group']==g),
            'previous_usable_samples':sum(c['previous_usable_samples'] for c in changes if c['group']==g),
            'candidate_usable_samples':sum(c['candidate_usable_samples'] for c in changes if c['group']==g)} for g in ('public','train')}
        cs=rows(candidate/'geometry_samples.jsonl')
        result['candidate_fit_attempt_diagnostics']=dict(sum((Counter(s['raw_proposal'].get('rejection_counts',{})) for s in cs),Counter()))
        result['candidate_samples_by_fit_diagnostic']=dict(Counter(k for s in cs for k in s['raw_proposal'].get('rejection_counts',{})))
        result['candidate_qualified_fitting_sources']=dict(Counter(s['raw_proposal'].get('fitting_source','unknown') for s in cs if s['raw_proposal']['usable_proposal']))
    return failures,changes,result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-run',type=Path,required=True)
    ap.add_argument('--candidate-run',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    failures,changes,result=audit(args.baseline_run,args.candidate_run)
    args.output.mkdir(parents=True,exist_ok=False)
    for name,data in [('failed_train_videos',failures),('coverage_changes',changes)]:
        (args.output/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8')
    for start in range(0,len(failures),12):
        montage=Image.new('RGB',(480*3,298*4),(25,25,25));draw=ImageDraw.Draw(montage)
        for i,f in enumerate(failures[start:start+12]):
            path=args.baseline_run/'diagnostics'/('train_'+f['video_id']+'.png')
            with Image.open(path) as image:
                image=image.convert('RGB');image.thumbnail((480,270));montage.paste(image,((i%3)*480,(i//3)*298))
            draw.text(((i%3)*480+4,(i//3)*298+272),f['video_id']+' '+f['category'],fill='white')
        montage.save(args.output/('failed_%02d.jpg'%start))
    inputs=[args.baseline_run/'geometry_quality.jsonl',args.baseline_run/'geometry_samples.jsonl']
    if args.candidate_run:inputs += [args.candidate_run/'geometry_quality.jsonl',args.candidate_run/'geometry_samples.jsonl']
    result['input_sha256']={str(p.resolve()):sha(p) for p in inputs}
    result['source_sha256']=sha(Path(__file__))
    shutil.copy2(__file__,args.output/'audit_geometry_failures.py')
    (args.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
