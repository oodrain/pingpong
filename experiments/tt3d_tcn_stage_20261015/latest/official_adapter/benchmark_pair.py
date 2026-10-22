"""Run C and C2 on identical official input; compare outputs without event GT."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--task-dir',type=Path,required=True);p.add_argument('--input',type=Path);p.add_argument('--runs',type=int,default=1);a=p.parse_args()
 if not 1<=a.runs<=5:p.error('--runs must be 1..5')
 package=Path(__file__).resolve().parent;outputs=a.task_dir.resolve().parent/'outputs'
 def run(variant,reference=None):
  previous=set(outputs.glob('tcn_'+variant+'_gpu_*'))
  cmd=[sys.executable,str(package/'run_stage.py'),'--task-dir',str(a.task_dir),'--mode','gpu-bench','--variant',variant,'--runs',str(a.runs)]
  if a.input:cmd+=['--input',str(a.input)]
  if reference:cmd+=['--reference-results',str(reference)]
  subprocess.run(cmd,check=True)
  created=set(outputs.glob('tcn_'+variant+'_gpu_*'))-previous
  if len(created)!=1:raise RuntimeError('Could not uniquely locate completed benchmark output')
  return created.pop()
 reference=run('C');candidate=run('C2',reference)
 aa=json.loads((reference/'benchmark_summary.json').read_text(encoding='utf-8'));bb=json.loads((candidate/'benchmark_summary.json').read_text(encoding='utf-8'))
 result=dict(C_result=reference.name,C2_result=candidate.name,C_FPS=aa['median_infer_fps'],C2_FPS=bb['median_infer_fps'],speedup=bb['median_infer_fps']/aa['median_infer_fps'],output_equivalence_passed=bb['reference_output_equivalence_passed'],event_GT_read=False)
 (candidate/'paired_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print('PAIRED BENCHMARK COMPLETE: '+str(candidate));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
