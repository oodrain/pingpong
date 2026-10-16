"""Prepare an ignored local asset from the existing selected checkpoint."""
import argparse
from pathlib import Path
from source.runtime import export_control

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'local_bundle/model.pt')
    args = parser.parse_args()
    export_control(args.source, args.output)
    print('Local retained-model asset prepared; no training performed.')
