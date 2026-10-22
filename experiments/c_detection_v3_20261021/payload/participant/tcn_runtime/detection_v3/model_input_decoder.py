"""Reproduce the frozen OpenCV model-input color contract explicitly.

This is model compatibility, not an assertion that a tagged BT.709 source was
recorded in BT.601. Both routes keep native frames and timestamps. NVDEC is
required for the GPU route; there is no CPU decoding fallback.
"""
import subprocess, tempfile
import numpy as np

def cpu_frames(meta, folder, mode):
    import av
    from pathlib import Path
    from av.video.reformatter import VideoReformatter, Colorspace
    converter=VideoReformatter();count=0;last_pts=None
    with av.open(str(Path(folder)/meta.path)) as video:
        for frame in video.decode(video=0):
            if frame.pts is None:raise ValueError('Missing source PTS')
            pts=frame.pts*frame.time_base
            if last_pts is not None and pts<=last_pts:raise ValueError('Non-increasing source PTS')
            last_pts=pts
            rgb=converter.reformat(frame,format='rgb24',src_colorspace=Colorspace.ITU601).to_ndarray()
            if rgb.shape!=(meta.height,meta.width,3):raise ValueError('Changed image size')
            yield count,rgb
            count+=1
    if count!=meta.n_frames:raise ValueError('Native decoded frame count mismatch')

def gpu_frames(path,width,height):
    command=['ffmpeg','-hide_banner','-loglevel','error','-hwaccel','cuda','-i',str(path),
             '-vf','scale=in_color_matrix=bt601','-vsync','0','-f','rawvideo','-pix_fmt','rgb24','-']
    size=width*height*3;error_stream=tempfile.TemporaryFile()
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=error_stream)
    try:
        while True:
            data=process.stdout.read(size)
            if not data:break
            if len(data)!=size:raise ValueError('Truncated NVDEC RGB frame')
            yield np.frombuffer(data,np.uint8).reshape(height,width,3).copy()
        code=process.wait(timeout=20)
        error_stream.seek(0);message=error_stream.read().decode('utf-8','replace')
        if code:raise RuntimeError(f'NVDEC failed ({code}): {message[-2000:]}')
    finally:
        if process.poll() is None:process.kill()
        process.wait();process.stdout.close();error_stream.close()

def install(framework_decoder):
    framework_decoder.iter_frames_gpu=gpu_frames
