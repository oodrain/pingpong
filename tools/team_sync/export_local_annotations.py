"""Export existing local annotations into a portable team snapshot.

This is a data export, never an annotation tool or inference input. Run against
the development archive with --data-root; the published snapshot needs no archive.
"""
import argparse
import hashlib
import json
from pathlib import Path

VERSION='local_development_20261022_v1'
JOURNALS=[
 'outputs/data_split_audit_20261014/train_event_review_01/human_event_reviews.jsonl',
 'outputs/contact_rule_revision_20261014/train_followup_review_02/human_event_reviews.jsonl',
 'outputs/contact_rule_revision_20261014/train_false_positive_review_04/human_event_reviews.jsonl',
 'outputs/frame_tcn_precision_20261015/complete_sequence_review_01/human_event_reviews.jsonl',
 'outputs/frame_tcn_revision_20261015/source_expansion_review_01/human_event_reviews.jsonl',
 'outputs/frame_tcn_source_robust_20261015/new_source_review_01/human_event_reviews.jsonl',
 'outputs/frame_tcn_contact_objects_20261016/adjudication_01/human_event_reviews.jsonl',
 'outputs/frame_tcn_event_ownership_20261016/state_review_01/human_event_reviews.jsonl',
 'outputs/frame_tcn_net_rgb_review_20261016/subtype_review_01/human_net_subtype_reviews.jsonl',
 'outputs/table_contact_scene_20261017/safety_review/human_event_reviews.jsonl',
 'worktrees/table-geometry-v2/outputs/experiments/tt3d_trajectory_pilot_20261013/auxiliary_event_review_01/human_event_reviews.jsonl',
 'worktrees/table-geometry-v2/outputs/experiments/tt3d_trajectory_pilot_20261013/racket_evidence_review_01/human_racket_evidence.jsonl',
]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def jsonlines(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
def ranges(mask):
 result=[];start=None
 for i,value in enumerate([*map(bool,mask),False]):
  if value and start is None:start=i
  if not value and start is not None:result.append([start,i-1]);start=None
 return result
def export(root,dest):
 import numpy as np
 root=Path(root).resolve();dest=Path(dest).resolve();sources={}
 def source(p):
  p=Path(p);rel=p.relative_to(root).as_posix();sources[rel]=sha(p);return rel
 def clean(x):
  if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
  if isinstance(x,list):return [clean(v) for v in x]
  if isinstance(x,str):
   for prefix in (str(root)+'\\',str(root)+'/',root.as_posix()+'/'):
    if x.startswith(prefix):return x[len(prefix):].replace('\\','/')
   if len(x)>2 and x[1:3] in (':\\',':/'):return '[external_path_omitted]'
  return x
 oldpath=root/'outputs/data_split_audit_20261014/split_manifest.json';old=read(oldpath);source(oldpath)
 videos=[]
 for r in old['videos']:
  videos.append({k:r.get(k) for k in ('video_id','source_group','video_sha256','width','height','n_frames','fps','object_annotation_sha256')}|dict(
   role={'train':'development','validation_reserved':'reserved','public_regression':'public_regression'}[r['split']],
   video_relative_path=Path(r['video_path']).relative_to(root).as_posix(),
   object_annotation_relative_path=Path(r['object_annotation_path']).relative_to(root).as_posix() if r.get('object_annotation_path') else None))
 write(dest/'data/splits/team_split_v1.json',dict(schema_version=1,split_id=old['split_id'],version=VERSION,
  reserved_sources=old['validation_source_groups'],source_identity_basis=old['source_confirmation']['statement'],
  policy=dict(development_is_not_fixed_training_set=True,reserved_training=False,reserved_selection=False,
   public_role='engineering_and_development_regression',unknown_frames_are_negative=False,independent_blind_test=False,
   historical_usage='Recorded by experiment; reservation does not erase previous development use'),videos=videos))
 indexpath=root/'outputs/six_task_delivery_20261020/index.json';idx=read(indexpath);source(indexpath)
 precise=[]
 for r in idx:
  if not r['is_primary']:continue
  ep=Path(r['prepared_events']);sp=Path(r['supervision_path']);source(ep);source(sp)
  with np.load(sp) as z:
   evaluation=ranges(z['evaluation_mask']);full=ranges(z['full_evaluation_mask'])
   # Explicit training coverage is separate from evaluation scope.
   known=ranges(z['loss_mask'].all(axis=1))
  precise.append(dict(video_id=r['video_id'],source_group=r['source_group'],video_sha256=r['video_sha256'],
   n_frames=r['n_frames'],complete_video=r['complete_video'],angle=r['angle'],
   evaluation_ranges=evaluation,full_evaluation_ranges=full,training_known_ranges=known,
   events=read(ep),provenance=dict(events_source=ep.relative_to(root).as_posix(),masks_source=sp.relative_to(root).as_posix())))
 target=dest/'labels/local_development_20261022'
 write(target/'precise_scopes.json',dict(version=VERSION,scope='42 development videos; annotation, not a fixed training split',videos=precise))
 pp=root/'outputs/c_quality_round2_20261022/positive_inventory.json';source(pp);positive=read(pp)
 write(target/'confirmed_positive_union.json',dict(version=VERSION,tolerance_frames=1,positive_only=True,
  absence_means_unknown=True,not_a_full_precision_recall_dataset=True,videos=positive))
 # Export all journal revisions for provenance. Only latest completed states
 # are separately exposed as reusable supplemental labels.
 completed=[];drafts=[];journal_summary=[]
 for rel in JOURNALS:
  p=root/rel;source(p);records=[json.loads(l) for l in p.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
  pack=p.with_name('review_pack.json');source(pack);packdata=read(pack);ci={c['case_id']:c for c in packdata.get('cases',[])}
  last={}
  for r in records:
   state=r.get('state',r);key=r.get('case_id',state.get('frame_id'));assert key is not None;last[str(key)]=r
  exportid=rel.split('/')[-2];jsonlines(target/'journals'/f'{exportid}.jsonl',clean(records))
  for key,r in last.items():
   state=clean(r.get('state',r));case=ci.get(key,packdata.get('case',{}));v=case.get('video_id',r.get('video_id'))
   row=dict(case_id=key,video_id=v,video_sha256=r.get('video_sha256',case.get('video_sha256')),
    revision=r.get('revision'),review_pack_sha256=r.get('review_pack_sha256',r.get('pack_sha256')),
    source_journal=rel,annotation_kind=state.get('annotation_kind','spatial_evidence' if 'ball_status' in state else 'event_review'),state=state)
   (completed if state.get('status')=='completed' else drafts).append(row)
  journal_summary.append(dict(source=rel,revisions=len(records),latest_cases=len(last),completed=sum(r.get('state',r).get('status')=='completed' for r in last.values())))
 jsonlines(target/'supplemental_completed.jsonl',completed);jsonlines(target/'supplemental_drafts.jsonl',drafts)
 geometry=[]
 for rel in ('worktrees/table-geometry-v2/outputs/experiments/geometry_reference_audit_20261013/review_pack/references.jsonl',
  'worktrees/table-geometry-v2/outputs/experiments/tt3d_net_review_20261013/review_pack_03/net_references.jsonl',
  'worktrees/table-geometry-v2/outputs/experiments/tt3d_net_review_20261013/review_pack_04/net_references.jsonl'):
  p=root/rel;source(p)
  for l in p.read_text(encoding='utf-8-sig').splitlines():
   if not l.strip():continue
   row=json.loads(l);geometry.append(dict(source=rel,reference={k:v for k,v in row.items() if k not in ('calibration',)}))
 jsonlines(target/'geometry_references.jsonl',clean(geometry))
 write(target/'source_hashes.json',dict(version=VERSION,local_archive_relative_paths=True,files=sources))
 write(target/'summary.json',dict(version=VERSION,development_videos=len(idx),precise_videos=len(precise),
  complete_videos=sum(r['complete_video'] for r in precise),
  precise_table_events=sum(e['event_type']=='table_bounce' for r in precise for e in r['events']),
  complete_table_events=sum(e['event_type']=='table_bounce' for r in precise if r['complete_video'] for e in r['events']),
  confirmed_positive_union=sum(map(len,positive.values())),completed_supplemental_records=len(completed),
  draft_supplemental_records=len(drafts),geometry_reference_records=len(geometry),journals=journal_summary))
 assert len(idx)==123 and len(precise)==42 and sum(map(len,positive.values()))==625
 print('Exported',len(precise),'precise videos;',len(completed),'completed supplements;',len(drafts),'drafts')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
 a=p.parse_args();export(a.data_root,a.output)
