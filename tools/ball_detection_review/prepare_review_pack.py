"""Export only six prediction tracks for authorized sharing; never copy videos, labels or weights."""
import argparse
from pathlib import Path
import core


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True,help='Directory containing video_id.json or video_id/control.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('请选择尚不存在的输出目录，避免覆盖已有结果')
    pending=[]
    for vid,meta in core.videos().items():
        rows,digest=core.tracks(args.predictions,vid,meta['num_frames'])
        if rows is None:parser.error('缺少预测：'+vid)
        # Explicit allowlist strips any unrelated metadata or local paths.
        allowed=('frame_id','timestamp_s','detected','x','y','confidence','in_image','missing_reason','coordinate_source')
        clean=[{k:r[k] for k in allowed if k in r} for r in rows]
        pending.append((vid,clean,digest))
    args.output.mkdir(parents=True)
    for vid,rows,digest in pending:core.save(args.output/(vid+'.json'),rows)
    core.save(args.output/'PACK.json',dict(schema=1,contents='six automatic prediction tracks only',
        videos=[v for v,_,_ in pending],training_or_annotation_data_included=False,
        source_prediction_sha256={v:h for v,_,h in pending},historical_development_exposure=True))
    print('REVIEW PREDICTIONS READY: 6 videos; no videos, labels, models or local paths copied.')


if __name__=='__main__':main()
