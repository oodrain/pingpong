"""Only a video is supplied after fixed local model assets are configured."""
import argparse
from pathlib import Path
import numpy as np
from source.runtime import infer_with_table_range, load_asset, save, save_lines, sha
from source.video_inputs import automatic_inputs

if __name__ == '__main__':
    local = Path(__file__).resolve().parent / 'local_bundle'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=local / 'model.pt')
    parser.add_argument('--ball-model', type=Path, default=local / 'ball.onnx')
    parser.add_argument('--table-model', type=Path, default=local / 'table_segmentation.ckpt')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output must be a new directory')
    for path in (args.video, args.model, args.ball_model, args.table_model):
        if not path.is_file():
            parser.error('Configure the video and all three fixed model files first')
    models = load_asset(args.model)
    features, tracks, geometry, raw = automatic_inputs(args.video, args.ball_model, args.table_model)
    from source.startup_pts_capture import StartupPTSCapture
    capture = StartupPTSCapture(str(args.video))
    try:
        okay, image = capture.read()
        if not okay:
            raise ValueError('Cannot decode the first video frame')
        image_size = image.shape[1], image.shape[0]
    finally:
        capture.release()
    points, hits, scores, decisions = infer_with_table_range(models, features, tracks, raw, image_size)
    args.output.mkdir(parents=True, exist_ok=False)
    save_lines(args.output / 'landings.jsonl', points)
    save_lines(args.output / 'hits.jsonl', hits)
    save_lines(args.output / 'tracks.jsonl', tracks)
    save(args.output / 'geometry.json', geometry)
    save(args.output / 'raw_table_proposals.json', raw)
    np.savez_compressed(args.output / 'runtime_features.npz', control=features)
    np.savez_compressed(args.output / 'scores.npz', **scores)
    save(args.output / 'decisions.json', decisions)
    save(args.output / 'fixed_table.json', decisions['table_range'])
    save(args.output / 'run.json', {'frames': len(tracks), 'video_sha256': sha(args.video),
         'model_sha256': sha(args.model), 'annotation_inputs': [], 'manual_geometry_inputs': [],
         'historical_detection_inputs': [], 'official_speed_benchmark': False,
         'table_range_policy': decisions['table_range']['policy']['schema'],
         'image_size': list(image_size)})
    print('Video analysis complete.')
