"""Attribute CPU time without changing inference, labels, or frozen evaluations.

Uses a fresh complete train video. Timers cover disjoint leaf model calls;
parent pose/racket calls are deliberately not added a second time.
This diagnostic is not the official runner's GPU speed benchmark.
"""
import argparse
from collections import defaultdict
from functools import wraps
import json
import os
from pathlib import Path
import platform
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT / 'development/evaluation'))
from dataset_protocol import load_protocol, select, sha


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-id', default='19_004')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = load_protocol()
    record = select(manifest, [args.video_id], 'training')[0]
    # Verify the video only; annotations are unnecessary for timing.
    if sha(record['video_path']) != record['video_sha256']:
        raise ValueError('Frozen source video changed')
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    import development.detection.video_contact_pipeline as pipeline
    torch.set_num_threads(args.threads)
    environment = {'platform': platform.platform(), 'python': platform.python_version(),
                   'processor_identifier': os.environ.get('PROCESSOR_IDENTIFIER'),
                   'logical_cpus': os.cpu_count(), 'torch': torch.__version__,
                   'cuda_available': torch.cuda.is_available(), 'torch_cuda_version': torch.version.cuda,
                   'onnxruntime': ort.__version__, 'ort_available_providers': ort.get_available_providers(),
                   'opencv': cv2.__version__, 'numpy': np.__version__,
                   'workers': 1, 'threads': args.threads, 'model_warmup': False}
    timings = defaultdict(list)
    restores = []
    last_progress = [time.perf_counter()]

    def instrument(owner, attribute, category):
        original = getattr(owner, attribute)
        @wraps(original)
        def timed(*a, **kw):
            name = category(*a, **kw) if callable(category) else category
            start = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                timings[name].append(time.perf_counter() - start)
                now = time.perf_counter()
                if now - last_progress[0] > 10:
                    save(args.output / 'progress.json', {
                        'status': 'running', 'video_id': args.video_id,
                        'timings': {k: {'calls': len(v), 'seconds': sum(v)} for k, v in timings.items()}})
                    last_progress[0] = now
        setattr(owner, attribute, timed)
        restores.append((owner, attribute, original))

    assets = {
        'ball': ROOT / 'participant/participant/pingpang/participant/weights/ball.onnx',
        'table': PROJECT / 'tmp/tt3d-upstream/weights/table_segmentation.ckpt',
        'pose': PROJECT / 'data/pose_weights/keypointrcnn_resnet50_fpn_coco-fc266e95.pth',
        'racket': PROJECT / 'data/racket_weights/rtmdet_m_racket_state.pt'}
    classifier = PROJECT / 'data/contact_classifier/source25_train_20261014_v1/model.json'
    source_paths = [Path(pipeline.__file__), Path(__file__), ROOT / 'participant/participant/pingpang/core/runner.py']
    plan = {'video_id': args.video_id, 'split': record['split'], 'split_id': manifest['split_id'],
            'video_sha256': record['video_sha256'], 'environment': environment,
            'model_sha256': {k: sha(v) for k, v in assets.items()}, 'classifier_sha256': sha(classifier),
            'sources': {str(p): sha(p) for p in source_paths}, 'reads_human_labels': False,
            'purpose': 'CPU time attribution; no rule/threshold/model changes',
            'official_speed_benchmark': False}
    save(args.output / 'plan.json', plan)
    constructors = {'ball': lambda: pipeline.VideoBallDetector(assets['ball'], args.threads),
                    'table': lambda: pipeline.TT3DSegmenter(assets['table'], args.threads),
                    'pose': lambda: pipeline.PoseDetector(assets['pose']),
                    'racket': lambda: pipeline.RacketDetector(assets['racket'], args.threads)}
    models = {}; initialisation = {}
    for name, create in constructors.items():
        start = time.perf_counter(); models[name] = create()
        initialisation[name] = time.perf_counter() - start
    print('Profiling complete train video', args.video_id, flush=True)
    instrument(models['ball'], 'detect', 'ball_detection')
    instrument(models['table'], 'probability', 'table_segmentation')
    instrument(pipeline, 'extract_quad', 'table_corner_extraction')
    instrument(pipeline.CausalGeometryTracker, 'update', 'table_temporal_update')
    shape = (record['height'], record['width'])
    instrument(models['pose'], 'predict', lambda image: 'pose_full_frame' if image.shape[:2] == shape else 'pose_roi')
    instrument(models['racket'], 'detect', lambda image: 'racket_full_frame' if image.shape[:2] == shape else 'racket_roi')
    instrument(pipeline.StartupPTSCapture, '__init__', 'timestamp_preflight_and_open')
    instrument(pipeline.StartupPTSCapture, 'read', 'video_decode_read')
    instrument(pipeline.StartupPTSCapture, 'get', 'video_metadata_get')
    for attribute in ('replay', 'propose_events', 'classify_candidates', 'merge_pose_alternatives',
                      'apply_model', 'experimental_landings'):
        instrument(pipeline, attribute, 'rules_' + attribute)
    instrument(pipeline, 'save', 'diagnostic_json_write')
    instrument(pipeline, 'save_lines', 'diagnostic_jsonl_write')
    started = time.perf_counter()
    try:
        run = pipeline.run(Path(record['video_path']), args.output / 'video_run', models, classifier, args.threads)
    finally:
        for owner, attribute, original in reversed(restores):
            setattr(owner, attribute, original)
    elapsed = time.perf_counter() - started
    summaries = {k: {'calls': len(v), 'seconds': sum(v), 'mean_ms': 1000 * sum(v) / len(v),
                     'p50_ms': float(np.median(v)) * 1000, 'p95_ms': float(np.percentile(v, 95)) * 1000,
                     'percent_run': 100 * sum(v) / elapsed} for k, v in timings.items()}
    report = {**plan, 'n_frames': run['n_frames'], 'automatic_pose_frames': run['automatic_pose_frames'],
              'geometry_samples': summaries.get('table_segmentation', {}).get('calls', 0),
              'model_initialisation_seconds': initialisation, 'run_wall_seconds': elapsed,
              'run_fps_excluding_model_initialisation': run['n_frames'] / elapsed,
              'timings': summaries, 'unattributed_seconds': elapsed - sum(sum(v) for v in timings.values()),
              'status': 'completed', 'limitations': [
                  'One complete 720p train clip; not a representative 1080p GPU benchmark.',
                  'One worker and two threads, with cold first calls; historical batch used two workers.',
                  'Model calls include preprocessing and postprocessing; no GPU speed extrapolation.',
                  'No human labels read and no accuracy evaluation or parameter selection performed.']}
    save(args.output / 'profile.json', report)
    save(args.output / 'progress.json', {'status': 'completed', 'profile_sha256': sha(args.output / 'profile.json')})
    print(json.dumps({'fps': report['run_fps_excluding_model_initialisation'], 'seconds': elapsed,
                      'timings': summaries}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
