"""TT3D table inference with frozen preprocessing and explicit device placement.

Model source: https://github.com/cogsys-tuebingen/tt3d
"""
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from .contract import CONTRACT

EXPECTED_SHA256 = CONTRACT['asset_sha256']['table_segmentation.ckpt']


class TT3DSegmenter:
    def __init__(self, checkpoint, threads=4, device='cuda'):
        import segmentation_models_pytorch as smp
        torch.set_num_threads(threads)
        if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != EXPECTED_SHA256:
            raise ValueError('Unexpected TT3D checkpoint hash')
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
        params = loaded['hyper_parameters']
        if (params['arch'], params['encoder_name'], params['in_channels'], params['out_classes']) != (
                'unetplusplus', 'timm-regnety_002', 3, 1):
            raise ValueError('Unexpected table segmentation architecture')
        self.model = smp.create_model(params['arch'], encoder_name=params['encoder_name'],
                                      in_channels=3, classes=1, encoder_weights=None)
        state = loaded['state_dict']
        self.model.load_state_dict({k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}, strict=True)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.mean = state['mean'].to(self.device)
        self.std = state['std'].to(self.device)

    def probability(self, image):
        rgb = cv2.cvtColor(cv2.resize(image, (640, 320), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            return self.model((tensor - self.mean) / self.std).sigmoid()[0, 0].cpu().numpy()
