"""Replay the retained model with existing automatic inputs; not an FPS test."""
import argparse
from pathlib import Path
import numpy as np
import torch
from source.runtime import infer, load_asset, lines, save, save_lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parent / 'local_bundle/model.pt')
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--tracks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with np.load(args.features, allow_pickle=False) as data:
        features = data['control']
    points, hits, scores, decisions = infer(load_asset(args.model), features, lines(args.tracks))
    args.output.mkdir(parents=True, exist_ok=False)
    save_lines(args.output / 'landings.jsonl', points)
    save_lines(args.output / 'hits.jsonl', hits)
    np.savez_compressed(args.output / 'scores.npz', **scores)
    save(args.output / 'decisions.json', decisions)
    print('Cached replay complete; this does not measure video processing speed.')
