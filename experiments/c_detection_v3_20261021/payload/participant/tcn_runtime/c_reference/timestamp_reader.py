"""Read actual MP4 access-unit PTS without a second complete pixel decode.

The original OpenCV scan is retained for unsupported or ambiguous inputs.
This module never uses historical timestamps, event labels, or frame/fps time.
"""
from pathlib import Path
import hashlib
import cv2
import numpy as np
from .startup_pts_capture import checked_timestamps


def packet_timestamps(path,expected_frames):
    import av
    with av.open(str(path)) as container:
        if 'mp4' not in container.format.name.split(',') or not container.streams.video:
            raise ValueError('Packet fast path requires MP4 video')
        stream=container.streams.video[0]
        if stream.codec_context.name not in ('h264','hevc'):
            raise ValueError('Packet fast path requires AVC/HEVC access-unit samples')
        if stream.codec_context.codec_tag not in ('avc1','avc3','hvc1','hev1'):
            raise ValueError('Unsupported MP4 sample entry')
        if stream.start_time is None or stream.time_base is None or stream.frames!=expected_frames:
            raise ValueError('Missing or inconsistent stream frame/time metadata')
        pts=[]
        for packet in container.demux(stream):
            if not packet.size:continue
            if packet.pts is None or packet.dts is None or packet.is_discard:
                raise ValueError('Missing or discarded access-unit timestamp')
            if packet.time_base!=stream.time_base:
                raise ValueError('Inconsistent packet time base')
            pts.append(int(packet.pts))
        if len(pts)!=expected_frames or len(set(pts))!=len(pts):
            raise ValueError('Packet/decoded-frame correspondence is ambiguous')
        coded_pts=list(pts)
        pts.sort()  # MP4 coded packets may arrive in B-frame decoding order.
        ratio=float(stream.time_base.numerator)/float(stream.time_base.denominator)
        # Match OpenCV FFmpeg dts_to_sec then CAP_PROP_POS_MSEC operation order.
        times=[float(p-int(stream.start_time))*ratio*1000 for p in pts]
        if not times or times[0]<0 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
            raise ValueError('Invalid presentation timestamp sequence')
        return times,dict(method='mp4_packet_pts',decoded_pixels_for_timestamps=False,
                          frames=len(times),time_base=[stream.time_base.numerator,stream.time_base.denominator],
                          reordered_for_presentation=pts!=coded_pts)


def legacy_timestamps(path,expected_frames):
    cap=cv2.VideoCapture(str(path));times=[]
    if not cap.isOpened():raise ValueError('Cannot open video for legacy timestamp scan')
    try:
        fps=cap.get(cv2.CAP_PROP_FPS);declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            okay,_=cap.read()
            if not okay:break
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    finally:cap.release()
    if len(times)!=declared or len(times)!=expected_frames:
        raise ValueError('Video frame count mismatch in decoder preflight')
    times,repair=checked_timestamps(times,fps)
    return times,dict(method='legacy_opencv_scan',decoded_pixels_for_timestamps=True,
                      frames=len(times),startup_repair=repair)


def read_timestamps(path,expected_frames):
    if type(expected_frames) is not int or expected_frames<1:
        raise ValueError('Need expected decoded frame count')
    try:times,info=packet_timestamps(path,expected_frames)
    except Exception as error:
        times,info=legacy_timestamps(path,expected_frames)
        info['fallback_reason']=type(error).__name__+': '+str(error)
    # Input filenames or machine details are not included in profiling output.
    info['timestamp_ms_sha256']=hashlib.sha256(np.asarray(times,dtype='<f8').tobytes()).hexdigest()
    return times,info
