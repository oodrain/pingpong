"""Download the pinned upstream checkpoint and verify its digest before use."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

REVISION='a2ef524ea0400262d6808db6cacf4a0b90bd0ad7'
SHA256='0ed344f09c5fe9079a22a51fa10f0c529d836e318895a1d4321ca75db0a93605'
URL=f'https://raw.githubusercontent.com/cogsys-tuebingen/tt3d/{REVISION}/weights/table_segmentation.ckpt'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path('outputs/models/table_segmentation.ckpt'))
    args=ap.parse_args()
    if args.output.exists():
        if hashlib.sha256(args.output.read_bytes()).hexdigest()!=SHA256:
            raise ValueError('Existing checkpoint digest mismatch')
        print('Verified existing checkpoint:',args.output)
        return
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.download')
    with urllib.request.urlopen(URL,timeout=60) as response,temporary.open('xb') as target:
        while block:=response.read(1024*1024):target.write(block)
    if hashlib.sha256(temporary.read_bytes()).hexdigest()!=SHA256:
        raise ValueError('Downloaded checkpoint digest mismatch; partial file retained for diagnosis')
    temporary.replace(args.output)
    print('Verified checkpoint:',args.output)


if __name__=='__main__':main()
