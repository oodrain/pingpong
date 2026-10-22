"""Portable offline annotation validation/scoring. Never import in inference."""
import argparse
import json
from pathlib import Path

SNAPSHOT='labels/local_development_20261022'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def matching(points,events,tolerance=1):
 edges=[]
 for e in events:
  lo,hi=e['interval_frames']
  edges.append(sorted([j for j,p in enumerate(points) if p['video_id']==e['video_id'] and lo-tolerance<=p['frame_id']<=hi+tolerance],
   key=lambda j:(max(lo-points[j]['frame_id'],points[j]['frame_id']-hi,0),j)))
 owner={}
 def augment(i,seen):
  for j in edges[i]:
   if j in seen:continue
   seen.add(j)
   if j not in owner or augment(owner[j],seen):owner[j]=i;return True
  return False
 for i in range(len(events)):augment(i,set())
 return {i:j for j,i in owner.items()}
def preserve(baseline,candidate,positives):
 lost=[];total=0;retained=0
 for video,points in baseline.items():
  if video not in candidate:raise ValueError('Candidate video missing: '+video)
  truth=positives[video];tag=lambda pp:[dict(p,video_id=video) for p in pp]
  a=matching(tag(points),truth);b=matching(tag(candidate[video]),truth)
  total+=len(a);retained+=len(set(a)&set(b))
  lost.extend(dict(truth[i],event_index=i) for i in sorted(set(a)-set(b)))
 return dict(passed=not lost,baseline_matched=total,retained=retained,lost=lost,
  positive_only=True,unmatched_predictions_are_not_false_positives=True)
def contains(frame,ranges):return any(lo<=frame<=hi for lo,hi in ranges)
def scoped_score(predictions,precise,scope):
 if scope not in ('trusted','complete'):raise ValueError('Use precise trusted or complete scopes')
 points=[];truth=[];used=[]
 for s in precise:
  video=s['video_id']
  if video not in predictions or (scope=='complete' and not s['complete_video']):continue
  ranges=s['evaluation_ranges' if scope=='trusted' else 'full_evaluation_ranges'];used.append(video)
  points.extend(dict(p,video_id=video) for p in predictions[video] if contains(p['frame_id'],ranges))
  truth.extend(dict(e,video_id=video) for e in s['events'] if e['event_type']=='table_bounce' and
   any(contains(i,ranges) for i in range(e['interval_frames'][0],e['interval_frames'][1]+1)))
 tp=len(matching(points,truth));fp=len(points)-tp;fn=len(truth)-tp
 return dict(scope=scope,videos=len(used),tp=tp,fp=fp,fn=fn,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
  spatial_scoring=False,independent_blind_test=False)
def validate_source_use(manifest,fit,selection,held):
 groups=[set(fit),set(selection),set(held)]
 if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):raise ValueError('Source groups overlap')
 dev={v['source_group'] for v in manifest['videos'] if v['role']=='development'}
 if not set().union(*groups)<=dev:raise ValueError('Only development sources allowed for inner experiments')
 return True
def validate_snapshot(repo):
 repo=Path(repo);manifest=read(repo/'data/splits/team_split_v1.json');base=repo/SNAPSHOT
 precise=read(base/'precise_scopes.json')['videos'];positives=read(base/'confirmed_positive_union.json')['videos']
 by={v['video_id']:v for v in manifest['videos']}
 assert len(by)==169 and len(precise)==42 and len(positives)==123
 assert sum(v['role']=='development' for v in by.values())==123
 assert sum(v['role']=='reserved' for v in by.values())==31
 assert sum(v['role']=='public_regression' for v in by.values())==15
 assert {v['source_group'] for v in by.values() if v['role']=='reserved'}=={'09','12','15','17','24'}
 assert sum(map(len,positives.values()))==625
 table_total=0;complete_total=0
 for s in precise:
  assert by[s['video_id']]['role']=='development' and s['video_sha256']==by[s['video_id']]['video_sha256']
  for k in ('evaluation_ranges','full_evaluation_ranges','training_known_ranges'):
   assert all(0<=lo<=hi<s['n_frames'] for lo,hi in s[k])
  for e in s['events']:
   lo,hi=e['interval_frames'];assert 0<=lo<=hi<s['n_frames']
   if e['event_type']=='table_bounce':
    table_total+=1;complete_total+=int(s['complete_video'])
 for v,ee in positives.items():
  assert by[v]['role']=='development'
  assert all(e['video_id']==v and 0<=e['interval_frames'][0]<=e['interval_frames'][1]<by[v]['n_frames'] for e in ee)
 for name,status in [('supplemental_completed.jsonl','completed'),('supplemental_drafts.jsonl','draft')]:
  records=[json.loads(l) for l in (base/name).read_text(encoding='utf-8').splitlines() if l.strip()]
  assert all(r['state']['status']==status and r['video_id'] in by for r in records)
  assert all(r['video_sha256']==by[r['video_id']]['video_sha256'] for r in records)
 assert table_total==166 and complete_total==112
 return dict(passed=True,development=123,reserved=31,public=15,precise_videos=42,
  complete_videos=sum(s['complete_video'] for s in precise),precise_table_events=table_total,
  complete_table_events=complete_total,confirmed_positive_union=625)
def load_predictions(path):
 p=Path(path)
 if p.suffix=='.jsonl':
  out={}
  for line in p.read_text(encoding='utf-8-sig').splitlines():
   if not line.strip():continue
   row=json.loads(line);out.setdefault(row['video_id'],[]).extend(row['points'])
  return out
 result=read(p)
 if not isinstance(result,dict) or not all(isinstance(v,list) for v in result.values()):raise ValueError('Expected {video_id: [points]}')
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--baseline',type=Path);p.add_argument('--candidate',type=Path);p.add_argument('--output',type=Path)
 a=p.parse_args();result=validate_snapshot(a.repo)
 if bool(a.baseline)!=bool(a.candidate):p.error('Provide both baseline and candidate')
 if a.baseline:
  base=a.repo/SNAPSHOT;old=load_predictions(a.baseline);new=load_predictions(a.candidate)
  precise=read(base/'precise_scopes.json')['videos'];pos=read(base/'confirmed_positive_union.json')['videos']
  result=dict(snapshot=result,preservation=preserve(old,new,pos),
   baseline={s:scoped_score(old,precise,s) for s in ('trusted','complete')},
   candidate={s:scoped_score(new,precise,s) for s in ('trusted','complete')})
 text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text,encoding='utf-8')
 print(text,end='')
 if 'preservation' in result and not result['preservation']['passed']:raise SystemExit(2)
if __name__=='__main__':main()
