"""Selected C detection v3: 144 weights, supported-gap repair, unchanged TCN."""
from .tcn_runtime.c2_solution import Solution as C2Solution
from .tcn_runtime.base_solution import sha256
from .tcn_runtime.detection_v3.local_repair import repair_video


class Solution(C2Solution):
    ball_model_name = 'ball_v3.onnx'
    ball_model_sha256 = '3ee4d3166f2ef9cedb98ca5efd23a544da6213d7b332140566eec98232be41c1'
    middle_model_name = 'ball_v3_middle.onnx'
    middle_model_sha256 = '37e7ca4d3f2002a0bdc486e784b93e01a391e6c4c9dbdbc270b965e63078ec53'

    def prepare(self, context):
        from pathlib import Path
        root = Path(__file__).resolve().parent
        if sha256(root / 'weights' / self.middle_model_name) != self.middle_model_sha256:
            raise ValueError('v3 middle model checksum mismatch')
        super().prepare(context)
        context.info.update(detection_version='C_v3_20261021',
                            ball_sha256=self.ball_model_sha256,
                            ball_middle_sha256=self.middle_model_sha256,
                            required_input_color='explicit_legacy601_RGB',
                            events_use_enriched_v3_tracks=True)

    def repair_tracks(self, video, tracks, size, detector):
        return repair_video(video, tracks, size, detector)
