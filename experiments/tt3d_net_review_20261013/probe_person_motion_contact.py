"""Freeze all 31 suggestions before retrospective human-label comparison."""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from development.bounce.person_motion_contact import person_box_visit, local_table_position, review_suggestion

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p): return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, obj): p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def run(root, pilot, out):
    project = Path(__file__).resolve().parents[2]
    roi_path = pilot/'remaining_contact_diagnosis_01/roi_evidence.json'
    inference_path = pilot/'public_regression_01/inference.json'
    geometry_path = project/'outputs/experiments/tt3d_geometry_refinement_20261013/runtime_check_02/geometry_samples.jsonl'
    roi = read(roi_path); inference = read(inference_path); geometry = lines(geometry_path)
    sources = {str(p): sha(p) for p in (roi_path, inference_path, geometry_path)}
    # Pin the untouched append-only journal hashes without reading their labels.
    journals = [pilot/'auxiliary_event_review_01/human_event_reviews.jsonl', pilot/'fit_02/human_contact_reviews.jsonl']
    before = {str(p): sha(p) for p in journals}
    videos = {r['video_id']: r for r in inference['videos']}
    cache = {}; cases = []
    out.mkdir(parents=True, exist_ok=False)
    for c in roi['cases']:
        vid, seed = c['video_id'], c['seed_frame']
        if vid not in cache:
            track = root/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1/trajectories'/f'{vid}.jsonl'
            poses = pilot/'remaining_contact_diagnosis_01'/f'{vid}_pose_probes.json'
            if sha(track) != roi['sources'][str(track)]: raise ValueError('Changed track')
            sources.update({str(p): sha(p) for p in (track, poses)})
            cache[vid] = (lines(track), read(poses)['frames'])
        tracks, poses = cache[vid]
        frames = [{'frame_id': f, 'ball': tracks[f], 'people': poses[str(f)]} for f in range(seed-2, seed+3)]
        body = person_box_visit(frames, seed)
        local = [dict(local_table_position([r for r in geometry if r['video_id'] == vid], tracks[f], f), frame_id=f)
                 for f in range(seed-2, seed+3)]
        proposal = next(r for r in videos[vid]['proposal_result']['proposals'] if r['frame_id'] == seed)
        decision = review_suggestion(c['decision']['hit_flag'], c['direction'], c['roi_visit'], body, local, proposal)
        cases.append({'case_id': f'{vid}_{seed}', 'video_id': vid, 'seed_frame': seed, 'angle': c['angle'],
                      'person_visit': body, 'wrist_visit': c['roi_visit'], 'local_positions': local,
                      'direction': c['direction'], 'proposal': proposal, 'decision': decision})
    rule = project/'development/bounce/person_motion_contact.py'
    evidence = {'cases': cases, 'sources': sources, 'runner_sha256': sha(__file__), 'rule_sha256': sha(rule),
                'human_labels_used_for_inference': False, 'method_designed_after_prior_human_review': True,
                'independent_holdout': False, 'production_events_modified': False,
                'thresholds_retuned_on_results': False, 'sequence_used': False,
                'sequence_limitation': 'No independently confirmed automatic bounce anchors; incomplete review windows are not full-rally labels.'}
    save(out/'evidence.json', evidence)
    (out/'source_probe_person_motion_contact.py').write_bytes(Path(__file__).read_bytes())
    (out/'source_person_motion_contact.py').write_bytes(rule.read_bytes())
    # Predictions are immutable on disk before labels are opened.
    human_path = pilot/'human_event_evaluation_01/human_reviews_snapshot.json'
    human = read(human_path); lookup = {c['case_id']: c for c in cases}; rows = []
    for h in human['records']:
        c = lookup[h['case_id']]
        rows.append({'case_id': h['case_id'], 'human_label': h['candidate_label'],
                     'baseline_flag': c['decision']['baseline_flag'], 'suggestion_flag': c['decision']['review_hit_suggestion']})
    sys.path.insert(0, str(root/'development/annotation'))
    from evaluate_auxiliary_reviews import flag_metrics, table_pool_metrics
    evaluation = {'evidence_sha256': sha(out/'evidence.json'), 'human_snapshot_sha256': sha(human_path),
                  'baseline_flags': flag_metrics(rows, 'baseline_flag'), 'suggestion_flags': flag_metrics(rows, 'suggestion_flag'),
                  'suggestion_table_pool': table_pool_metrics(rows, 'suggestion_flag'), 'cases': rows,
                  'manual_journals_unchanged': all(sha(p) == v for p, v in before.items()), 'journal_hashes': before,
                  'scope': 'Already reviewed development candidates, review suggestions only; no production exclusion.'}
    save(out/'evaluation.json', evaluation)
    print(json.dumps({k: evaluation[k] for k in ('baseline_flags', 'suggestion_flags', 'suggestion_table_pool', 'manual_journals_unchanged')}, ensure_ascii=False))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for n in ('root', 'pilot', 'output'): p.add_argument('--'+n, type=Path, required=True)
    a = p.parse_args(); run(a.root.resolve(), a.pilot.resolve(), a.output.resolve())
