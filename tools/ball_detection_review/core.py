"""Portable video/annotation/point contracts and positive-box evaluation. Standard library only."""
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
VISIBLE = {'visible', 'blurred', 'partial'}
VISIBILITY = {'': '未核查', 'visible': '清晰可见', 'blurred': '模糊但可定位',
              'partial': '部分遮挡但可定位', 'occluded': '完全不可见',
              'out': '确认比赛球不在画面内', 'uncertain': '无法判断'}
JUDGMENTS = {'': '未判断', 'correct': '定位正确', 'missing': '没有输出球点',
             'weak': '置信度不足', 'wrong': '位置错误', 'background': '背景假球',
             'tail': '尾帧未推理', 'unknown': '暂不判断'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def cases():
    with (HERE / 'cases.csv').open(encoding='utf-8-sig', newline='') as stream:
        return [dict(row, frame_id=int(row['frame_id'])) for row in csv.DictReader(stream)]


def videos():
    return {row['video_id']: row for row in read(HERE / 'videos.json')['videos']}


def data_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('数据路径必须位于所选数据根目录内')
    return path


def annotation(root, video):
    path = data_path(root, video['annotation_path'])
    value = read(path)
    if value.get('video_id') != video['video_id']:
        raise ValueError('对象标注的视频编号不一致')
    value['_frames'] = {r['frame_id']: r for r in value['frames']}
    return value


def boxes(annotation_value, frame_id, label='ball'):
    frame = annotation_value['_frames'].get(frame_id, {})
    return [b for obj in frame.get('annotations', []) if obj.get('label') == label
            for b in obj.get('bbox_xyxy', []) if len(b) == 4 and all(math.isfinite(x) for x in b)]


def prediction_file(directory, video_id):
    if not directory:
        return None
    directory = Path(directory)
    choices = [directory / (video_id + ext) for ext in ('.json', '.jsonl')]
    choices += [directory / video_id / name for name in ('control.json', 'tracks.json', 'tracks.jsonl')]
    return next((p for p in choices if p.is_file()), None)


def tracks(directory, video_id, frame_count):
    path = prediction_file(directory, video_id)
    if path is None:
        return None, None
    if path.suffix == '.jsonl':
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    else:
        rows = read(path)
    if not isinstance(rows, list) or len(rows) != frame_count:
        raise ValueError(video_id + ' 预测必须为覆盖整段视频的逐帧数组；不能用触台事件列表代替')
    for index, row in enumerate(rows):
        if row.get('frame_id') != index or type(row.get('detected')) is not bool:
            raise ValueError('预测需要连续的零起始 frame_id 和布尔 detected')
        if row['detected'] and not all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in ('x', 'y')):
            raise ValueError('检测点坐标必须为有限数值')
        confidence = row.get('confidence')
        if confidence is not None and (not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError('confidence 必须为空或 0 到 1 的有限数值')
        if row.get('timestamp_s') is not None and not math.isfinite(row['timestamp_s']):
            raise ValueError('timestamp_s 必须为真实的有限时间戳')
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def trusted(row, threshold=.75):
    return bool(row and row['detected'] and row.get('in_image', True)
                and row.get('confidence') is not None and row['confidence'] >= threshold)


def status(row):
    if row is None:
        return '未载入该视频预测（不等于漏检）'
    reason = row.get('missing_reason')
    if not row['detected']:
        return '未输出球点；原因：' + str(reason or '未记录')
    confidence = row.get('confidence')
    conf = '未知' if confidence is None else f'{confidence:.4f}'
    return f"({row['x']:.1f}, {row['y']:.1f})  置信度 {conf}  " + ('可信' if trusted(row) else '低置信/无效')


def evaluate(ann, rows, review=None):
    """Visible positive-box support. Unknown/unannotated frames never become negatives."""
    review = review or {}
    counts = dict(positive_frames=0, matched=0, missing=0, low_confidence=0,
                  wrong_location=0, tail_not_inferred=0, unknown_frames=0,
                  reviewed_absent_frames=0, false_points_on_reviewed_absent=0)
    errors = []
    mx, my = ann['frame_width'] / 512, ann['frame_height'] / 288
    for i, row in enumerate(rows):
        visibility = review.get(str(i), {}).get('visibility', '')
        bb = boxes(ann, i)
        if visibility == 'out':
            counts['reviewed_absent_frames'] += 1
            counts['false_points_on_reviewed_absent'] += int(trusted(row))
            continue
        if visibility in ('occluded', 'uncertain') or not bb:
            counts['unknown_frames'] += 1
            continue
        counts['positive_frames'] += 1
        if row.get('missing_reason') == 'tail_not_inferred':
            counts['tail_not_inferred'] += 1
        elif not row['detected']:
            counts['missing'] += 1
        elif not trusted(row):
            counts['low_confidence'] += 1
        elif any(b[0]-mx <= row['x'] <= b[2]+mx and b[1]-my <= row['y'] <= b[3]+my for b in bb):
            counts['matched'] += 1
            errors.append(min(math.hypot(row['x']-(b[0]+b[2])/2, row['y']-(b[1]+b[3])/2) for b in bb))
        else:
            counts['wrong_location'] += 1
    counts['positive_localization_rate'] = counts['matched'] / counts['positive_frames'] if counts['positive_frames'] else None
    counts['matched_center_error_mean_px'] = sum(errors)/len(errors) if errors else None
    counts['annotation_scope'] = 'existing positive boxes, modified only by explicit visibility reviews'
    counts['complete_detector_f1'] = None
    return counts
