"""Verify published snapshot and frozen C/C2 source without external assets."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def sha(path):
 raw=Path(path).read_bytes()
 if path.suffix.lower() in ('.json','.jsonl','.py','.md','.txt','.csv','.ps1','.cmd','.html','.yml','.yaml') or path.name=='.gitattributes':raw=raw.replace(b'\r\n',b'\n')
 return hashlib.sha256(raw).hexdigest()
def main():
 release=ROOT/'experiments/team_integration_20261022/results'
 manifest=json.loads((release/'publication_manifest.json').read_text(encoding='utf-8'))
 for rel,expected in manifest['files'].items():
  p=(ROOT/rel).resolve();assert p.is_relative_to(ROOT) and p.is_file(),rel
  assert sha(p)==expected,rel
 protected=json.loads((release/'protected_source_manifest.json').read_text(encoding='utf-8'))
 for rel,expected in protected['files'].items():assert sha(ROOT/rel)==expected,rel
 spec=importlib.util.spec_from_file_location('team_data',ROOT/'development/evaluation/team_data.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 checks=module.validate_snapshot(ROOT)
 # Use C's own immutable package manifest, not a copied acceptance claim.
 spec=importlib.util.spec_from_file_location('c_package_verify',ROOT/'experiments/c_detection_v3_20261021/verify_bundle.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);c.verify(require_assets=False)
 print(json.dumps(dict(passed=True,published_files=len(manifest['files']),protected_baseline_files=len(protected['files']),snapshot=checks,
  default_model_replaced=False,training_performed=False,independent_accuracy_claim=False),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
