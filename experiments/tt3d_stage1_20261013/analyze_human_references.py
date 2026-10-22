"""Freeze submitted human references and compare existing automatic geometry."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from development.evaluation.geometry_reference_review import latest_reviews
from development.evaluation.validate_table_geometry import corner_error
from development.detection.auto_table_geometry import AutoGeometryTimeline
from experiments.tt3d_stage1_20261013.audit_review_inputs import sha, read


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    base = PROJECT / 'outputs/experiments'
    pack = base / 'geometry_reference_audit_20261013/review_pack'
    root = base / 'tt3d_stage1_20261013'
    formal = root / 'human_reference_evaluation_20261013_01'
    out = root / 'human_reference_analysis_20261013_01'
    manifest = read(pack / 'manifest.json')
    evaluated = read(formal / 'summary.json')
    refs_hash = sha(pack / 'references.jsonl')
    if refs_hash != evaluated['reviews_sha256']:
        raise ValueError('Human references changed since formal evaluation')
    latest = latest_reviews(pack, manifest)
    if set(latest) != {c['case_id'] for c in manifest['cases']}:
        raise ValueError('Human review coverage is incomplete')
    inference_path = root / 'final_diagnostic_20261013_v2/records.json'
    proposals = {r['case_id']: r for r in read(inference_path)}
    run = Path(manifest['geometry_run'])
    if sha(run / 'geometry_samples.jsonl') != evaluated['geometry_sha256']:
        raise ValueError('Frozen geometry changed')
    samples = defaultdict(list)
    for line in (run / 'geometry_samples.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        samples[row['group'], row['video_id']].append(row)
    timelines = {key: AutoGeometryTimeline(values) for key, values in samples.items()}
    out.mkdir(exist_ok=False)
    (out / 'overlays').mkdir()
    (out / 'references_snapshot.jsonl').write_bytes((pack / 'references.jsonl').read_bytes())
    if sha(out / 'references_snapshot.jsonl') != refs_hash:
        raise ValueError('Reference snapshot changed')
    write(out / 'latest_human_references.json', list(latest.values()))
    records = []
    for case in manifest['cases']:
        image_path = pack / case['image_url']
        if sha(image_path) != case['image_sha256']:
            raise ValueError('Original image changed')
        human = latest[case['case_id']]
        inference = proposals[case['case_id']]
        if any(case[k] != inference[k] for k in ('group', 'video_id', 'frame_id')):
            raise ValueError('Single-frame inference misaligned')
        timeline = timelines[case['group'], case['video_id']]
        sample = timeline.sample(case['frame_id'])
        available = timeline.lookup(case['frame_id']) is not None
        proposal = inference['same_frame_proposal']
        record = dict(case_id=case['case_id'], group=case['group'], video_id=case['video_id'],
                      frame_id=case['frame_id'], human_status=human['status'], reviewer=human['reviewer'],
                      causal_geometry_available=available,
                      same_frame_proposal_accepted=proposal['usable_proposal'],
                      same_frame_reason=proposal['reason'], human_reference_is_single_review=True)
        if human['status'] == 'corners_visible':
            if available:
                record['causal_error'] = corner_error(sample['corners'], human['corners'])
            if proposal['usable_proposal']:
                record['accepted_same_frame_error'] = corner_error(proposal['corners'], human['corners'])
            elif proposal.get('corners'):
                record['rejected_same_frame_error'] = corner_error(proposal['corners'], human['corners'])
        draw = cv2.imread(str(image_path))
        for points, color, width in (
            (proposal.get('corners'), (0, 220, 255), 2),
            (sample['corners'] if available else None, (240, 70, 230), 3),
            (human['corners'], (70, 230, 70), 2),
        ):
            if points:
                cv2.polylines(draw, [np.asarray(points).round().astype(np.int32)], True, color, width)
        for i, p in enumerate(human['corners']):
            xy = tuple(np.asarray(p).round().astype(int))
            cv2.circle(draw, xy, 5, (70, 230, 70), -1)
            cv2.putText(draw, f'H{i+1}', (xy[0]+6, xy[1]-7), cv2.FONT_HERSHEY_SIMPLEX, .55, (70, 230, 70), 2)
        cv2.rectangle(draw, (0, 0), (draw.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(draw, f"{case['case_id']} | human={human['status']} | causal={available}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.putText(draw, "green=human reference; yellow=single-frame candidate; purple=causal geometry",
                    (8, 46), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        if not cv2.imwrite(str(out / 'overlays' / (case['case_id']+'.jpg')), draw):
            raise ValueError('Cannot save overlay')
        records.append(record)
    compared = [r for r in records if 'causal_error' in r]
    distances = [r['causal_error']['mean_corner_error_px'] for r in compared]
    mean = float(np.mean(distances))
    if len(compared) != evaluated['corner_comparisons'] or not np.isclose(mean, evaluated['mean_corner_error_px']):
        raise ValueError('Human diagnostic does not agree with formal evaluation')
    priority = sorted(compared, key=lambda r: r['causal_error']['maximum_corner_error_px'], reverse=True)
    summary = dict(
        cases=len(records), unique_submitted_references=len(latest),
        raw_submissions=len((pack / 'references.jsonl').read_text(encoding='utf-8').splitlines()),
        statuses=dict(Counter(r['human_status'] for r in records)),
        causal_geometry_available=sum(r['causal_geometry_available'] for r in records),
        visible_but_no_causal_geometry=sum(r['human_status']=='corners_visible' and
                                         not r['causal_geometry_available'] for r in records),
        causal_comparisons=len(compared), causal_mean_corner_error_px=mean,
        causal_median_case_mean_error_px=float(np.median(distances)),
        causal_maximum_corner_error_px=max(r['causal_error']['maximum_corner_error_px'] for r in compared),
        accepted_same_frame_comparisons=sum('accepted_same_frame_error' in r for r in records),
        independent_double_review_verified=False, diagnostic_selection_not_representative=True,
        model_inference_repeated=False, production_geometry_changed=False,
        references_sha256=refs_hash, frozen_geometry_sha256=evaluated['geometry_sha256'],
        reused_single_frame_inference_sha256=sha(inference_path),
        largest_causal_errors=priority,
    )
    if sha(pack / 'references.jsonl') != refs_hash:
        raise ValueError('Human references changed during analysis')
    write(out / 'records.json', records)
    write(out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
