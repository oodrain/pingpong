"""Freeze the current prototype and verify runtime CV gates and review integrity."""
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from development.bounce.contact_classifier import FEATURES, apply_model, features, fit

BASE = PROJECT / 'outputs/experiments/video_only_contact_20261014'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    folder = BASE / 'v000038_02'
    run = read(folder / 'run.json')
    for name in ('human_annotation_inputs', 'manual_axis_inputs', 'prior_pose_or_track_cache_inputs'):
        assert run[name] == [], name
    assert run['production_exclusion_enabled'] is False
    assert sha(Path(run['video_path'])) == run['video_sha256']
    for path, expected in run['source_hashes'].items():
        assert sha(Path(path)) == expected, path
    for key, path in run['model_paths'].items():
        expected = run['contact_classifier']['model_sha256'] if key == 'contact_classifier' else run['models'][key]['checkpoint_sha256']
        assert sha(Path(path)) == expected, key
    cases = read(folder / 'event_candidates.json')['cases']
    target = next(c for c in cases if c['seed_frame'] == 53)
    assert target['event'] == 'racket_contact_candidate'
    assert target['learned_contact']['hit_suggestion']
    assert all(c['confirmed'] is False and c['decision'].get('adopt', False) is False for c in cases)

    # Training feature extraction and fitting must match the frozen training code.
    frozen = ast.parse((BASE / 'classifier_01/source_contact_classifier.py').read_text(encoding='utf-8'))
    current = ast.parse((PROJECT / 'development/bounce/contact_classifier.py').read_text(encoding='utf-8'))
    for name in ('features', 'fit', 'score_matrix', 'predict'):
        old = next(n for n in frozen.body if isinstance(n, ast.FunctionDef) and n.name == name)
        new = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(old) == ast.dump(new), name

    predictions = read(BASE / 'audit_02/automatic_cache_predictions.json')['cases']
    evaluation = read(BASE / 'classifier_01/evaluation.json')
    labels = {c['case_id']: c for c in evaluation['cases']}
    matrix = np.array([features(c) for c in predictions])
    frozen_features = read(BASE / 'classifier_01/automatic_features.json')
    assert frozen_features['features'] == FEATURES
    assert [c['case_id'] for c in predictions] == frozen_features['case_ids']
    np.testing.assert_array_equal(matrix, frozen_features['vectors'])
    y = np.array([labels[c['case_id']]['human_label'] == 'racket_contact' for c in predictions], float)
    groups = np.array([labels[c['case_id']]['source_group_proposal'] for c in predictions])
    checked = []
    for group in sorted(set(groups)):
        test = groups == group
        model = fit(matrix[~test], y[~test])
        result = apply_model(model, copy.deepcopy([c for i, c in enumerate(predictions) if test[i]]))
        for c in result:
            expected = labels[c['case_id']]
            assert abs(c['learned_contact']['score'] - expected['group_cv_score']) < 1e-12
            flag = c['event'] == 'racket_contact_candidate'
            assert flag == expected['group_cv_combined_flag'], c['case_id']
            checked.append({'case_id': c['case_id'], 'runtime_hit_flag': flag,
                            'human_label': expected['human_label'], 'held_group': str(group)})
    assert sum(c['runtime_hit_flag'] and c['human_label'] == 'racket_contact' for c in checked) == 13
    assert sum(c['runtime_hit_flag'] and c['human_label'] != 'racket_contact' for c in checked) == 0

    experiments = PROJECT / 'outputs/experiments'
    pilot = experiments / 'tt3d_trajectory_pilot_20261013'
    spatial = read(BASE / 'audit_02/spatial_diagnostic.json')
    journals = {
        pilot / 'auxiliary_event_review_01/human_event_reviews.jsonl': '98c7af87323cc6f3de608b2d391b4f1e9e9320b39b55487890fdf4fcd3c6cb26',
        pilot / 'fit_02/human_contact_reviews.jsonl': 'b7693f1589c373b034890eaa75334b5b4172be7326dded68e676fb0d64a9f6e8',
        pilot / 'racket_evidence_review_01/human_racket_evidence.jsonl': spatial['spatial_journal_sha256'],
        experiments / 'geometry_reference_audit_20261013/review_pack/references.jsonl': 'b3f72d5b4ebdec5e9ffc441cf3a9c19a96240bd83bd0850da178f047090d2247',
        experiments / 'tt3d_net_review_20261013/review_pack_03/net_references.jsonl': '23e8baca951f2d6432adc58195e12ee918bc78043def52313827edd7dc898924',
        experiments / 'tt3d_net_review_20261013/review_pack_04/net_references.jsonl': '1c492f17ac47fae29e80136ddc626ef9253da3bbb4e4d87f05af2f19edf63ae3',
    }
    for path, expected in journals.items():
        assert sha(path) == expected, str(path)

    snapshot = folder / 'source'
    sources = [Path(p) for p in run['source_hashes']]
    sources += [ROOT / 'development/annotation' / name for name in
                ('video_only_entry.py', 'video_only_results.py', 'local_annotation.py',
                 'test_video_only_contacts.py', 'test_contact_classifier.py')]
    sources += [Path(__file__), PROJECT / 'experiments/tt3d_net_review_20261013/train_contact_classifier.py']
    source_records = {}
    for path in sources:
        relative = path.relative_to(ROOT)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert sha(destination) == sha(path), str(destination)
        else:
            destination.write_bytes(path.read_bytes())
        source_records[str(relative)] = sha(destination)
    save(folder / 'source_manifest.json', source_records)

    validation = {
        'stage': 'automatic_hit_candidate_recovery_and_video_only_prototype',
        'stage_complete': True, 'full_project_acceptance_complete': False,
        'fresh_video_run': {'frames': run['n_frames'], 'pose_racket_frames': run['automatic_pose_frames'],
                            'inference_seconds': run['inference_seconds'],
                            'frames_per_second_cpu': run['n_frames'] / run['inference_seconds'],
                            'seed_53_event': target['event'], 'seed_53_score': target['learned_contact']['score'],
                            'run_sha256': sha(folder / 'run.json'),
                            'event_candidates_sha256': sha(folder / 'event_candidates.json'),
                            'human_annotation_inputs': [], 'previous_inference_cache_inputs': []},
        'runtime_group_cv_gates_checked': checked,
        'development_group_validation': evaluation['metrics']['group_cv_combined_flag'],
        'table_candidate_retention': evaluation['group_cv_table_pool'],
        'independent_holdout': False, 'new_recording_sources_available': False,
        'five_spatial_frames_used_for_training': False, 'event_labels_used_for_training': 31,
        'human_journal_integrity': {str(p): {'sha256': expected, 'unchanged': True} for p, expected in journals.items()},
        'source_and_model_hashes_verified': True,
        'test_results_this_session': {
            'root_annotation_suite': {'tests_run': 253, 'skipped': 4, 'status': 'OK'},
            'worktree_torch_video_only_suite': {'tests_run': 13, 'skipped': 0, 'status': 'OK',
                                               'overlaps_root_tests': True}},
        'native_viewer_smoke_passed': True, 'summary_and_validation_pages_visually_inspected': True,
        'all_twelve_candidate_pages_rendered': True,
        'candidate_pool_shadow_report': str(BASE / 'classifier_01/public_shadow_regression.json'),
        'shadow_f1_is_not_fresh_final_event_f1': True,
        'production_exclusion_enabled': False, 'camera_calibration_verified': False,
        'remaining': ['Full end-to-end regression on existing 15 public videos and 154 development videos',
                      'Validate and integrate final contact output instead of candidate-only analysis',
                      'Speed optimization and target runtime validation'],
    }
    save(folder / 'validation.json', validation)
    previous = BASE / 'readiness.json'
    archived = BASE / 'readiness_before_classifier.json'
    if not archived.exists():
        archived.write_bytes(previous.read_bytes())
    readiness = read(previous)
    readiness.update({
        'available_video_sets_user_confirmed': {'public_examples': 15, 'development_data': 154},
        'required_new_validation_inputs': [], 'new_video_input_required_to_continue': False,
        'awaiting_input': None, 'awaiting_additional_annotations': False,
        'full_video_only_run': 'v000038_02', 'automatic_cache_regression': 'audit_02',
        'current_stage_complete': True, 'acceptance_complete': False,
        'candidate_classifier': {'trained': True, 'training_event_labels': 31, 'group_count': 11,
                                 'model_sha256': run['contact_classifier']['model_sha256'],
                                 'independent_holdout': False},
        'manual_journals_unchanged': {str(p): True for p in journals},
        'remaining_requirements': validation['remaining'],
        'independent_validation_limitation': 'No verified unused recording source; existing data supports development validation only.',
    })
    readiness['current_spatial_labels']['training_started'] = False
    readiness['current_spatial_labels']['use'] = 'Detector diagnosis only; candidate classifier uses earlier event labels.'
    save(previous, readiness)
    print(json.dumps({'runtime_group_cv_cases': len(checked), 'human_journals_unchanged': len(journals),
                      'seed_53_event': target['event'], 'validation': str(folder / 'validation.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
