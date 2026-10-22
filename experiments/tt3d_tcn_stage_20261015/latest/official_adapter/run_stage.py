"""Run the unmodified official self-check or benchmark with strict GPU selection."""
import argparse
import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import uuid
from verify_bundle import verify, verify_installation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir', required=True, type=Path)
    parser.add_argument('--mode', choices=('cpu-smoke', 'gpu-smoke', 'gpu-bench'), required=True)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--input', type=Path, help='Defaults to task-dir/public_data')
    parser.add_argument('--reference-results', type=Path, help='Optional original A benchmark directory or predictions.jsonl')
    parser.add_argument('--variant', choices=('C','C2'), default='C2')
    args = parser.parse_args()
    spec = 'participant.solution:Solution' if args.variant == 'C2' else 'participant.tcn_runtime.c_reference.solution:Solution'
    if not 1 <= args.runs <= 5:
        parser.error('--runs must be between 1 and 5')
    manifest = verify()
    task = args.task_dir.resolve()
    verify_installation(task, manifest)
    data = (args.input or task / 'public_data').resolve()
    import hashlib
    deployed = task / 'participant/solution.py'
    supplied = Path(__file__).resolve().parent / 'payload/participant/solution.py'
    if not deployed.is_file() or hashlib.sha256(deployed.read_bytes()).digest() != hashlib.sha256(supplied.read_bytes()).digest():
        parser.error('This adapter is not deployed. Run deploy.py first.')
    videos = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))['videos']
    missing = [e['path'] for e in videos if not (data / e['path']).is_file()]
    if not videos or missing:
        parser.error('Missing video files: ' + repr(missing))
    env = os.environ.copy()
    env['TCN_DEVICE'] = 'cpu' if args.mode == 'cpu-smoke' else 'cuda'
    env.pop('TCN_SAVE_DEBUG', None)
    env.pop('TCN_ORT_PROFILE_PREFIX', None)
    torch_lib = Path(sys.executable).resolve().parents[1] / 'lib/python3.10/site-packages/torch/lib'
    if torch_lib.is_dir():
        env['LD_LIBRARY_PATH'] = str(torch_lib) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    def run(argv):
        subprocess.run([sys.executable, *map(str, argv)], cwd=task, env=env, check=True)
    run([task / 'integrity.py', '--verify'])
    if args.mode != 'cpu-smoke':
        run([task / 'scripts/check_env.py'])
    run([Path(__file__).resolve().parent / 'check_adapter.py', '--task-dir', task,
         '--device', env['TCN_DEVICE']])
    if args.mode in ('cpu-smoke', 'gpu-smoke'):
        run([task / 'scripts/check_solution.py', '--input', data, '--solution',
             spec, '--decoder', env['TCN_DEVICE'].replace('cuda', 'gpu')])
        return
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    output = task.parent / 'outputs' / ('tcn_' + args.variant + '_gpu_' + stamp)
    records = []
    for index in range(1, args.runs + 1):
        out = output / ('run_' + str(index))
        run([task / 'run.py', '--input', data, '--output', out,
             '--solution', spec, '--decoder', 'gpu'])
        status = json.loads((out / 'run_status.json').read_text(encoding='utf-8'))
        totals = status['totals']
        info = status['solution_info']
        if (status['status'] != 'ok' or totals['failed'] or totals['skipped'] or totals.get('cpu_fallback', 0)
                or totals['ok'] != len(videos) or totals['frames'] != sum(v['n_frames'] for v in videos)):
            raise RuntimeError('Incomplete run; not accepted as an FPS result')
        if info['tcn_device'] != 'cuda' or not info['table_device'].startswith('cuda') or info['ball_providers'][0] != 'CUDAExecutionProvider':
            raise RuntimeError('Actual GPU execution was not enabled')
        prediction_sha = hashlib.sha256((out / 'predictions.jsonl').read_bytes()).hexdigest()
        records.append({'run': index, 'performance': status['performance'], 'timing': status['timing'],
                        'predictions_sha256': prediction_sha, 'predictions': totals['predictions']})
    summary = {'engineering_variant': info['engineering_variant'], 'schema': 'tcn_gpu_benchmark_v1', 'videos': len(videos), 'runs': records,
               'median_infer_fps': statistics.median(r['performance']['infer_fps'] for r in records),
               'all_run_predictions_identical': len({r['predictions_sha256'] for r in records}) == 1,
               'official_f1_calculated': False, 'independent_blind_test': False}
    from compare_benchmark import compare
    comparisons = [compare(output / 'run_1', output / ('run_' + str(i))) for i in range(1,args.runs+1)]
    summary['all_run_predictions_equivalent'] = all(c['passed'] for c in comparisons)
    if args.reference_results:
        reference_comparisons = [compare(args.reference_results, output / ('run_' + str(i))) for i in range(1,args.runs+1)]
        summary['reference_output_equivalence_passed'] = all(c['passed'] for c in reference_comparisons)
        (output / 'output_comparison.json').write_text(json.dumps(reference_comparisons, indent=2)+'\n',encoding='utf-8')
    breakdown = []
    for i in range(1,args.runs+1):
        status=json.loads((output / ('run_' + str(i)) / 'run_status.json').read_text(encoding='utf-8'))
        videos=status['solution_info']['tcn_video_timings']
        broad={k:sum(v[k] for v in videos) for k in ('timestamp_probe_sec','ball_sec','table_model_sec','table_geometry_sec','roi_repair_sec','features_sec','tcn_sec')}
        broad['official_decode_sec']=sum(v.get('decode_sec',0) for v in status['videos'])
        operations={};methods={}
        for v in status['solution_info'].get('tcn_video_profiles',[]):
            method=v['timestamps']['method'];methods[method]=methods.get(method,0)+1
            for group in ('geometry','ball'):
                for name,row in v[group]['operations'].items():
                    d=operations.setdefault(group+'.'+name,dict(calls=0,seconds=0.))
                    d['calls']+=row['calls'];d['seconds']+=row['seconds']
        breakdown.append(dict(run=i,worker_and_main_stage_seconds=broad,operations=operations,timestamp_methods=methods,
            c_pipeline_timings=status['solution_info'].get('c_pipeline_timings',[]),
            worker_and_main_durations_overlap_and_must_not_be_added=True,
            nested_geometry_timings_are_inclusive_not_additive=True,official_performance=status['performance']))
    (output / 'timing_breakdown.json').write_text(json.dumps(breakdown,indent=2)+'\n',encoding='utf-8')
    (output / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print('BENCHMARK COMPLETE: ' + str(output))
    print('Median official Runner infer_fps: ' + str(summary['median_infer_fps']))
    print('FPS includes actual video processing and finish_video; F1 must be evaluated separately.')
    if summary.get('reference_output_equivalence_passed') is False:
        raise RuntimeError('Reference event output parity failed. Retain all results; do not accept speed at the cost of changed events.')
    if not summary['all_run_predictions_equivalent']:
        raise RuntimeError('Repeated C runs disagree beyond the fixed tolerance; keep results for diagnosis.')


if __name__ == '__main__':
    main()
