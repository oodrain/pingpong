"""Inference-only adapter for the public TT3D table checkpoint.

Upstream: https://github.com/cogsys-tuebingen/tt3d
Architecture, input range and normalization follow its calibration inference.
No Lightning, training services or upstream Python execution are required.
"""
from pathlib import Path
import hashlib
import cv2
import numpy as np

EXPECTED_SHA256 = "0ed344f09c5fe9079a22a51fa10f0c529d836e318895a1d4321ca75db0a93605"


class TT3DSegmenter:
    def __init__(self, checkpoint, threads=4):
        import torch
        import segmentation_models_pytorch as smp
        self.torch = torch
        torch.set_num_threads(threads)
        path = Path(checkpoint)
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.sha256 != EXPECTED_SHA256:
            raise ValueError("Unexpected TT3D checkpoint hash")
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        params = loaded["hyper_parameters"]
        if (params["arch"], params["encoder_name"], params["in_channels"], params["out_classes"]) != (
                "unetplusplus", "timm-regnety_002", 3, 1):
            raise ValueError("Unexpected segmentation architecture")
        self.model = smp.create_model(params["arch"], encoder_name=params["encoder_name"],
                                      in_channels=3, classes=1, encoder_weights=None)
        state = loaded["state_dict"]
        self.model.load_state_dict({k[len("model."):]: v for k,v in state.items() if k.startswith("model.")}, strict=True)
        self.model.eval()
        self.mean, self.std = state["mean"], state["std"]
        self.provenance = {"checkpoint_sha256": self.sha256, "torch": torch.__version__,
                           "segmentation_models_pytorch": smp.__version__, "device": "cpu",
                           "input_size": [640, 320], "input_range": "RGB float32 0..255 (upstream)",
                           "upstream_commit": "a2ef524ea0400262d6808db6cacf4a0b90bd0ad7"}

    def probability(self, image):
        rgb = cv2.cvtColor(cv2.resize(image, (640, 320), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float().unsqueeze(0)
        with self.torch.inference_mode():
            probability = self.model((tensor - self.mean) / self.std).sigmoid()[0,0].numpy()
        return probability
