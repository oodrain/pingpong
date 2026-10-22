"""Run the official lifecycle with the frozen v3 model-input color contract."""
import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys

from verify_bundle import verify, verify_installation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--debug-tracks', action='store_true')
    args = parser.parse_args()
    task = args.task_dir.resolve()
    manifest = verify()
    verify_installation(task, manifest)
    subprocess.run([sys.executable, str(task / 'integrity.py'), '--verify'],
                   cwd=task, check=True)
    os.environ['TCN_DEVICE'] = args.device
    os.environ['TCN_SAVE_DEBUG'] = '1' if args.debug_tracks else '0'
    # Load Torch CUDA libraries before ORT (same ordering as the GPU acceptance).
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    sys.path.insert(0, str(task))
    import core.decoder as decoder
    from participant.tcn_runtime.detection_v3 import model_input_decoder
    original_decode = decoder.decode_video

    def decode(meta, folder, mode):
        if mode == 'cpu':
            yield from model_input_decoder.cpu_frames(meta, folder, mode)
        elif mode == 'gpu':
            yield from original_decode(meta, folder, mode)
        else:
            raise ValueError('Unknown decoder mode: ' + mode)

    model_input_decoder.install(decoder)
    decoder.decode_video = decode
    import core.runner as runner
    runner.decode_video = decode
    # Only the process-local decoder binding changes; framework bytes are intact.
    sys.argv = [str(task / 'run.py'), '--input', str(args.input.resolve()),
                '--output', str(args.output.resolve()),
                '--solution', 'participant.solution:Solution',
                '--decoder', 'gpu' if args.device == 'cuda' else 'cpu']
    runpy.run_path(str(task / 'run.py'), run_name='__main__')


if __name__ == '__main__':
    main()
