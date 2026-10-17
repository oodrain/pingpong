"""Replay the retained model with existing automatic inputs; not an FPS test."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from source.runtime import infer_with_table_range, load_asset, lines, save, save_lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parent / 'local_bundle/model.pt')
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--tracks', type=Path, required=True)
    parser.add_argument('--raw-table-proposals', type=Path, help='Defaults to raw_table_proposals.json beside --features; an empty list means unknown table')
    parser.add_argument('--image-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with np.load(args.features, allow_pickle=False) as data:
        features = data['control']
    proposals = args.raw_table_proposals or args.features.parent / 'raw_table_proposals.json'
    if not proposals.is_file():
        parser.error('Provide automatic raw table proposals, or a JSON file containing [] for unavailable geometry')
    size = args.image_size
    if size is None:
        for name, key in [('freeze.json', 'size'), ('run.json', 'image_size')]:
            metadata = args.features.parent / name
            if metadata.exists():
                size = json.loads(metadata.read_text(encoding='utf-8-sig')).get(key)
                if size is not None:
                    break
    if size is None:
        parser.error('Provide --image-size WIDTH HEIGHT or matching freeze.json/run.json')
    observations = json.loads(proposals.read_text(encoding='utf-8-sig'))
    points, hits, scores, decisions = infer_with_table_range(load_asset(args.model), features, lines(args.tracks), observations, size)
    args.output.mkdir(parents=True, exist_ok=False)
    save_lines(args.output / 'landings.jsonl', points)
    save_lines(args.output / 'hits.jsonl', hits)
    np.savez_compressed(args.output / 'scores.npz', **scores)
    save(args.output / 'decisions.json', decisions)
    save(args.output / 'fixed_table.json', decisions['table_range'])
    print('Cached replay complete; this does not measure video processing speed.')
