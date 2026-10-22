"""Render frozen suggestions against their source video, without human labels."""
import argparse, json, hashlib
from pathlib import Path
import cv2
import numpy as np
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(root, pilot, out):
    e = read(out/'evidence.json'); roi = read(pilot/'remaining_contact_diagnosis_01/roi_evidence.json')
    manifest = root/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1/selected_manifest.json'
    if sha(manifest) != roi['sources'][str(manifest)]: raise ValueError('Manifest changed')
    meta = {v['video_id']: v for v in read(manifest)['videos']}
    geometry_path = next(Path(p) for p in e['sources'] if p.endswith('geometry_samples.jsonl'))
    if sha(geometry_path) != e['sources'][str(geometry_path)]: raise ValueError('Geometry changed')
    geometry = [json.loads(s) for s in geometry_path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    outputs = {}; inputs = {str(manifest): sha(manifest), str(out/'evidence.json'): sha(out/'evidence.json')}
    for vid in sorted({c['video_id'] for c in e['cases']}):
        m = meta[vid]; video = root/'participant/participant/pingpang/public_data'/m['path']
        if sha(video) != roi['sources'][str(video)]: raise ValueError('Video changed')
        inputs[str(video)] = sha(video)
        pose = pilot/'remaining_contact_diagnosis_01'/f'{vid}_pose_probes.json'
        track = root/'outputs/trajectory_delivery_audit_20261010/C_public15_20261009_v1/trajectories'/f'{vid}.jsonl'
        for p in (pose, track):
            if sha(p) != e['sources'][str(p)]: raise ValueError('Evidence input changed')
            inputs[str(p)] = sha(p)
        poses = read(pose)['frames']; tracks = [json.loads(s) for s in track.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        cases = [c for c in e['cases'] if c['video_id'] == vid]
        targets = {f for c in cases for f in range(c['seed_frame']-2, c['seed_frame']+3)}
        cap = cv2.VideoCapture(str(video)); frames = {}; fid = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if fid in targets: frames[fid] = frame
            fid += 1
        cap.release()
        if fid != len(tracks): raise ValueError('Video frame count changed')
        for c in cases:
            sheet = np.zeros((430, 1280, 3), np.uint8)
            for i, pos in enumerate(c['local_positions']):
                f = pos['frame_id']; image = frames[f].copy(); ball = tracks[f]
                if pos['available']:
                    g = next(r for r in geometry if r['video_id'] == vid and r['frame_id'] == pos['geometry_frame'])
                    cv2.polylines(image, [np.rint(g['corners']).astype(np.int32)], True, (0, 220, 255), 3)
                for p in poses[str(f)]:
                    if p['score'] >= .9:
                        b = np.rint(p['box']).astype(int)
                        cv2.rectangle(image, tuple(b[:2]), tuple(b[2:]), (255, 220, 0), 2)
                if ball.get('detected'): cv2.circle(image, (round(ball['x']), round(ball['y'])), 8, (0, 255, 0), 2)
                sheet[:144, i*256:(i+1)*256] = cv2.resize(image, (256, 144))
                cv2.putText(sheet, f'frame {f}', (i*256+6, 173), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
                x = round(ball['x']) if ball.get('detected') else m['width']//2
                y = round(ball['y']) if ball.get('detected') else m['height']//2
                left = max(0, min(m['width']-240, x-120)); top = max(0, min(m['height']-180, y-90))
                sheet[200:380, i*256+8:i*256+248] = image[top:top+180, left:left+240]
            cv2.putText(sheet, 'cyan: person box | yellow: current table projection | green: tracked ball', (15,413), cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            path = out/(c['case_id']+'_spatial.jpg'); cv2.imencode('.jpg',sheet)[1].tofile(str(path)); outputs[str(path)] = sha(path)
    (out/'source_render_person_motion_evidence.py').write_bytes(Path(__file__).read_bytes())
    (out/'visual_evidence.json').write_text(json.dumps({'source_hashes': inputs, 'output_hashes': outputs,
        'renderer_sha256': sha(__file__), 'human_labels_read': False}, indent=2), encoding='utf-8')
    print('Rendered', len(outputs), 'five-frame spatial sheets')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for n in ('root','pilot','output'): p.add_argument('--'+n,type=Path,required=True)
    a = p.parse_args(); run(a.root.resolve(),a.pilot.resolve(),a.output.resolve())
