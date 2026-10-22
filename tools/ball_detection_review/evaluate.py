"""Evaluate ball localization independently of TCN, with unknown frames explicitly excluded."""
import argparse
from pathlib import Path
import core


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--reviews',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reviews=core.read(args.reviews)['reviews'] if args.reviews else {}
    results=[]
    for vid,meta in core.videos().items():
        ann=core.annotation(args.dataset_root,meta)
        rows,digest=core.tracks(args.predictions,vid,ann['num_frames'])
        if rows is None:
            parser.error('缺少整段逐帧预测：'+vid)
        local={str(r['frame_id']):r for r in reviews.values() if r['video_id']==vid}
        results.append(dict(video_id=vid,angle=meta['angle'],prediction_sha256=digest,
                            **core.evaluate(ann,rows,local)))
    keys=('positive_frames','matched','missing','low_confidence','wrong_location','tail_not_inferred',
          'unknown_frames','reviewed_absent_frames','false_points_on_reviewed_absent')
    def total(items):
        value={key:sum(r[key] for r in items) for key in keys}
        value['positive_localization_rate']=value['matched']/value['positive_frames'] if value['positive_frames'] else None
        return value
    result=dict(schema=1,threshold=.75,matching='same frame, existing box expanded by one 512x288 heatmap pixel',
        evaluated_scope='six development videos; not independent validation',
        annotation_caveat='box presence is a positive reference, not proof of visibility; manual visibility reviews can exclude it',
        unknown_frames_are_negatives=False,complete_detector_f1=None,
        overall=total(results),by_angle={str(a):total([r for r in results if r['angle']==a]) for a in (0,45,90)},
        videos=results)
    core.save(args.output,result)
    print(result['overall'])
    print('仅球检测定位评测；没有计算触台F1或整套检测器F1。')


if __name__=='__main__':main()
