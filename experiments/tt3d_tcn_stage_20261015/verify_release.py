"""Check the complete stage capsule with the standard library only."""
import hashlib,json,os
from pathlib import Path

if __name__=='__main__':
    root=Path(os.environ.get('TT3D_TCN_BUNDLE',str(Path(__file__).resolve().parent/'local_bundle'))).resolve()
    hashes=json.loads((root/'SHA256SUMS.json').read_text(encoding='utf-8'))
    for name,digest in hashes.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('Invalid capsule file: '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Capsule hash mismatch: '+name)
    print(json.dumps({'capsule_files_verified':len(hashes),'all_hashes_match':True}))
