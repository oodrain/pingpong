"""Mine unreviewed high-score train proposals; unmatched stage GT is not a label."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT / 'development/evaluation'))
from dataset_protocol import DEFAULT, load_protocol, select, sha
from development.bounce.curved_contact_classifier import score_matrix


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def run(args):
    protocol = load_protocol()
    dataset = read(args.dataset / 'dataset.json')
    automatic = read(args.dataset / 'automatic_curved_cases.json')['videos']
    policy = read(args.policy)
    records = {r['video_id']: r for r in select(protocol, list(automatic), 'training')}
    stage_path = DEFAULT / 'baseline_replay_03/label_snapshots/candidate_events.jsonl'
    stage = [json.loads(line) for line in stage_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    selected = []
    config = policy['config']
    for vid, items in automatic.items():
        if not items:
            continue
        scores = score_matrix(policy['table_model'], [item['vector'] for item in items])
        for item, score in zip(items, scores):
            seed = item['seed_frame']
            proposal = item['proposal']
            position = item['position']
            if score < config['table_threshold'] or not position.get('available') or position.get('outside_projection'):
                continue
            if proposal['tracking_spike_nearby'] or proposal['bic_gain_over_smooth_cubic'] < config.get('minimum_cubic_bic_gain', float('-inf')):
                continue
            if any(lo - 2 <= seed <= hi + 2 for lo, hi in dataset['scopes'].get(vid, [])):
                continue
            if any(row['video_id'] == vid and abs(row['frame_id'] - seed) <= 2 for row in stage):
                continue
            selected.append(dict(item, table_score=float(score)))
    selected.sort(key=lambda item: -item['table_score'])
    if not selected:
        raise ValueError('No previously unreviewed high-score train proposals')
    args.output.mkdir(parents=True, exist_ok=False)
    cache = DEFAULT / 'train_event_review_01/automatic_inputs'
    cases = []
    for index, item in enumerate(selected):
        vid = item['video_id']
        seed = item['seed_frame']
        record = records[vid]
        if sha(record['video_path']) != record['video_sha256']:
            raise ValueError('Frozen training video changed')
        track = cache / f'{vid}_tracks.jsonl'
        cases.append({
            'case_id': item['case_id'], 'video_id': vid, 'seed_frame': seed,
            'source_group': record['source_group'], 'split': 'train',
            'angle': '训练补充 / 来源 ' + record['source_group'],
            'video_path': record['video_path'], 'video_sha256': record['video_sha256'],
            'tracks_path': str(track), 'tracks_sha256': sha(track),
            'width': record['width'], 'height': record['height'], 'n_frames': record['n_frames'],
            'default_scope': [max(0, seed - 12), min(record['n_frames'] - 1, seed + 12)],
            'priority': index, 'selection_reason': 'high_table_score_unreviewed_and_stage_reference_unmatched',
        })
    snapshot = args.output / Path(__file__).name
    snapshot.write_bytes(Path(__file__).read_bytes())
    inputs = [args.policy, args.dataset / 'dataset.json', args.dataset / 'automatic_curved_cases.json',
              stage_path, DEFAULT / 'split_manifest.json', snapshot]
    pack = {
        'schema_version': 1, 'purpose': 'train_only_event_labels', 'split_id': protocol['split_id'],
        'sources': {str(path): sha(path) for path in inputs}, 'cases': cases,
        'known_public_development_set': False, 'human_labels_prefilled': False,
        'selection_uses_train_review_scope_metadata': True,
        'selection_uses_train_stage_reference_presence_only': True,
        'selection_uses_public_or_reserved_event_gt': False, 'validation_videos_included': 0,
        'frame_indexing': 'zero_based_decoded_frames',
        'selection_policy': 'Train-only high-score automatic proposals outside completed scopes; lack of stage reference does not imply no event.',
        'review_instructions': 'Record all events in each window, then identify this candidate event. No event and uncertain are valid. Keep nearby hit and table events separate. No ball/racket coordinates requested.',
    }
    save(args.output / 'review_pack.json', pack)
    save(args.output / 'selection.json', {
        'selected': selected, 'pack_sha256': sha(args.output / 'review_pack.json'),
        'unmatched_stage_reference_is_not_ground_truth': True,
        'labels_used_for_runtime_features': False,
    })
    print(json.dumps({'cases': len(cases), 'ids': [c['case_id'] for c in cases]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
