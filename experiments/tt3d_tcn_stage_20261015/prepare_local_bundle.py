"""Copy an existing, checksum-verified local capsule into ignored local_bundle/."""
import argparse,hashlib,json,shutil
from pathlib import Path

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve();target=Path(__file__).resolve().parent/'local_bundle'
    hashes=json.loads((source/'SHA256SUMS.json').read_text(encoding='utf-8'))
    if target.exists():raise FileExistsError('Local bundle already exists; use TT3D_TCN_BUNDLE for a different source.')
    paths=[]
    for name,digest in hashes.items():
        path=(source/name).resolve()
        if not path.is_relative_to(source) or not path.is_file():raise ValueError('Invalid source capsule path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Source capsule checksum mismatch: '+name)
        paths.append((path,Path(name)))
    for path,relative in paths+[(source/'SHA256SUMS.json',Path('SHA256SUMS.json'))]:
        destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
    print(json.dumps({'local_bundle':str(target),'verified_source_files':len(paths),'git_publication':False}))
