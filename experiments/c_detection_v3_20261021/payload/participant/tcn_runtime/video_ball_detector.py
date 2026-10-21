"""Frozen official detector preprocessing, decoding and middle-frame delay."""
import hashlib
from pathlib import Path
import cv2
import numpy as np
import time
from .image_evidence import TimingProfile

EXPECTED_SHA256 = 'a5a467f134ae7e2fc5679e395dca9af12b6dfcc65cee51e633c465beb0c61543'


class VideoBallDetector:
    def __init__(self, checkpoint, threads=4, device='cuda', profile_prefix=None, expected_sha256=EXPECTED_SHA256):
        import onnxruntime as ort
        if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != expected_sha256:
            raise ValueError('Unexpected official ball model')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        if profile_prefix:
            options.enable_profiling = True
            options.profile_file_prefix = str(profile_prefix)
        if device == 'cuda':
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('ONNX Runtime CUDA provider unavailable; CPU fallback is not a GPU benchmark')
            providers = [('CUDAExecutionProvider', {'device_id': 0, 'use_tf32': '0'}), 'CPUExecutionProvider']
        elif device == 'cpu':
            providers = ['CPUExecutionProvider']
        else:
            raise ValueError('Device must be cpu or cuda')
        self.session = ort.InferenceSession(str(checkpoint), sess_options=options, providers=providers)
        if device == 'cuda' and self.session.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('Ball detector failed to activate CUDA; refusing silent CPU fallback')
        if self.session.get_inputs()[0].type != 'tensor(float)':
            raise ValueError('Unexpected ball model dtype')
        self.profile = TimingProfile()
        self.name = self.session.get_inputs()[0].name
        self.reset()

    @staticmethod
    def normalize(bgr):
        x = cv2.cvtColor(cv2.resize(bgr, (512, 288)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        x = (x - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        return x.transpose(2, 0, 1).copy()

    def reset(self):
        blank = self.normalize(np.zeros((288, 512, 3), np.uint8))
        self.buffer = [blank, blank]

    def detect(self, image):
        start = time.perf_counter()
        self.buffer = (self.buffer + [self.normalize(image)])[-3:]
        tensor = np.concatenate(self.buffer, axis=0)[None]
        self.profile.add('normalization_and_window', time.perf_counter()-start)
        start = time.perf_counter()
        heat = self.session.run(None, {self.name: tensor})[0][0, 1]
        self.profile.add('inference_and_transfers', time.perf_counter()-start)
        start = time.perf_counter()
        mask = heat > .7
        if not mask.any():
            self.profile.add('heatmap_decode', time.perf_counter()-start)
            return None
        yy, xx = np.nonzero(mask)
        weights = heat[mask].astype(np.float64)
        total = weights.sum()
        height, width = image.shape[:2]
        result = {'x': float(xx @ weights / total * width / 512),
                'y': float(yy @ weights / total * height / 288), 'confidence': float(heat[mask].max())}
        self.profile.add('heatmap_decode', time.perf_counter()-start)
        return result
