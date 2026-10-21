"""Check model loading, warmup and actual selected providers/devices without videos."""
import argparse
import importlib
from pathlib import Path
import sys
import os
import tempfile
import json
from verify_bundle import verify, verify_installation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--allow-version-difference', action='store_true', help='Local developer checks only')
    args = parser.parse_args()
    manifest = verify()
    verify_installation(args.task_dir, manifest)
    import torch
    import numpy
    import onnxruntime as ort
    import segmentation_models_pytorch as smp
    import timm
    if not args.allow_version_difference:
        assert torch.__version__ == '2.1.2+cu118', 'Use the official torch environment'
        assert numpy.__version__ == '1.26.4', 'Use official numpy 1.26.4'
        assert ort.__version__ == '1.18.1', 'Use official onnxruntime-gpu 1.18.1'
        assert smp.__version__ == '0.3.3' and timm.__version__ == '0.9.2', 'Install the pinned extra dependencies'
    os.environ['TCN_DEVICE'] = args.device
    task = args.task_dir.resolve()
    sys.path.insert(0, str(task))
    from core.types import RunContext
    cls = importlib.import_module('participant.solution').Solution
    with tempfile.TemporaryDirectory(prefix='tcn_adapter_check_') as temp:
        if args.device == 'cuda':
            os.environ['TCN_ORT_PROFILE_PREFIX'] = str(Path(temp) / 'ball_cuda_profile')
        context = RunContext(input_dir=str(task / 'public_data'), output_dir=temp, decoder='cpu')
        solution = cls()
        solution.prepare(context)
        os.environ.pop('TCN_ORT_PROFILE_PREFIX', None)
        for states in solution.models.values():
            for model, _ in states:
                assert next(model.parameters()).device.type == args.device
        assert next(solution.table.model.parameters()).device.type == args.device
        assert solution.table.mean.device.type == args.device and solution.table.std.device.type == args.device
        if args.device == 'cuda':
            assert solution.ball.session.get_providers()[0] == 'CUDAExecutionProvider'
            profile = Path(solution.ball.session.end_profiling())
            events = json.loads(profile.read_text(encoding='utf-8'))
            cuda_nodes = [e for e in events if e.get('args', {}).get('provider') == 'CUDAExecutionProvider']
            assert cuda_nodes, 'Ball warmup did not execute any CUDA provider nodes; GPU check rejected'
            print('Ball CUDA provider node events: ' + str(len(cuda_nodes)))
            assert solution.ball.middle.session.get_providers()[0] == 'CUDAExecutionProvider'
            assert context.info['middle_graph_check']['device_IO']
            middle_events = json.loads(Path(solution.ball.middle.session.end_profiling()).read_text(encoding='utf-8'))
            middle_cuda = [e for e in middle_events if e.get('args', {}).get('provider') == 'CUDAExecutionProvider']
            assert middle_cuda, 'Derived middle graph did not execute CUDA nodes'
            print('C2 middle heatmap uses CUDA I/O Binding; actual-video parity is required separately.')
        print('ADAPTER CHECK PASSED')
        print('Device: ' + args.device)
        print('Ball providers: ' + ', '.join(solution.ball.session.get_providers()))
        print('Frozen model and configuration checksums passed; all models warmed up.')
        if args.device == 'cpu':
            print('CPU check only; this does not measure GPU FPS.')


if __name__ == '__main__':
    main()
