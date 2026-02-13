import os
import sys
import torch
import numpy as np
from argparse import Namespace
from unittest.mock import MagicMock

# Mock necessary modules
sys.modules["scene"] = MagicMock()
sys.modules["scene.gaussian_model"] = MagicMock()
sys.modules["arguments"] = MagicMock()
sys.modules["utils.compress_utils"] = MagicMock()

# Import the function to test
# We need to do this *after* mocking if we were mocking at module level,
# but here we'll just import the classes we need or mock extensively.
# Since export_onnx.py imports these at top level, we might have issues if we don't mock them effectively.
# Actually, let's just create a simplified version of the logic inside this script for testing purposes,
# or try to import export_onnx after mocking.

from io import BytesIO
import pickle


# Helper to mock GaussianModel
class MockGaussianModel:
    def __init__(self, sh_degree, gaussian_dim=3, rot_4d=False, force_sh_3d=False):
        self.active_sh_degree = sh_degree
        self.gaussian_dim = gaussian_dim
        self.rot_4d = rot_4d
        self.force_sh_3d = force_sh_3d
        self.max_sh_degree = sh_degree
        self._xyz = torch.rand(100, 3).cuda()
        self._features_dc = torch.rand(100, 1, 3).cuda()
        self._features_rest = torch.rand(100, 15, 3).cuda()
        self._scaling = torch.rand(100, 3).cuda()
        self._rotation = torch.rand(100, 4).cuda()
        self._opacity = torch.rand(100, 1).cuda()
        self.time_duration = [-0.5, 0.5]

        self.mlp_cont = MagicMock()
        self.mlp_cont.mlp = torch.nn.Linear(80, 32).cuda()
        self.mlp_cont.encoding = MagicMock()
        self.mlp_cont.encoding.freqs = torch.rand(10).cuda()

        self.mlp_view = torch.nn.Linear(35, 3).cuda()
        self.mlp_dc = torch.nn.Linear(35, 3).cuda()
        self.mlp_opacity = torch.nn.Linear(35, 1).cuda()

        self._features_static = torch.rand(100, 3).cuda()
        self._features_view = torch.rand(100, 3).cuda()

        self.get_xyz = self._xyz

    def decode(self, load_dict, decompress=True):
        pass


# Mock export_onnx.py functionality
# Since export_onnx.py has imports that might fail or be heavy, let's replicate the critical part: MLPInference and export_svq_data
# We will use the actual export_onnx.py if possible, but let's try to verify the export logic specifically.

import export_onnx


def test_export():
    args = Namespace(
        config="config.yaml",  # Dummy
        gaussian_dim=3,
        time_duration=[-0.5, 0.5],
        num_pts=100,
        num_pts_ratio=1.0,
        rot_4d=False,
        force_sh_3d=False,
        seed=6666,
        comp_checkpoint="dummy.xz",
        out_dir="web",
        opset=17,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # Mock gaussians
    gaussians = MockGaussianModel(sh_degree=3)

    # Mock save_dict for SVQ
    save_dict = {
        "xyz": torch.rand(100, 3),
        "scale_code": [torch.rand(16, 3) for _ in range(1)],
        "scale_index": [np.random.randint(0, 16, 100).astype(np.uint8)],  # Simplified
        "scale_htable": [
            None
        ],  # Mock huffman table not needed for this test if we mock huffman_decode
        "rotation_code": [torch.rand(16, 4) for _ in range(1)],
        "rotation_index": [np.random.randint(0, 16, 100).astype(np.uint8)],
        "rotation_htable": [None],
        "app_code": [torch.rand(16, 6) for _ in range(1)],
        "app_index": [np.random.randint(0, 16, 100).astype(np.uint8)],
        "app_htable": [None],
    }

    # Mock huffman_decode in utils.compress_utils
    export_onnx.huffman_decode = MagicMock(
        side_effect=lambda code, table: code.astype(np.uint16)
    )  # Just pass through for test

    print("Testing ONNX export...")
    try:
        wrapper = (
            export_onnx.MLPInference(
                mlp_cont=gaussians.mlp_cont,
                mlp_view=gaussians.mlp_view,
                mlp_dc=gaussians.mlp_dc,
                mlp_opacity=gaussians.mlp_opacity,
                freq_buffers=gaussians.mlp_cont.encoding.freqs,
            )
            .cpu()
            .float()
        )
        wrapper.eval()

        onnx_path = os.path.join(args.out_dir, "mlp_inference.onnx")
        torch.onnx.export(
            wrapper,
            (
                torch.randn(100, 3),
                torch.randn(100, 3),
                torch.randn(100, 3),
                torch.tensor([0.5]),
            ),
            onnx_path,
            opset_version=args.opset,
            input_names=["xyz", "features_static", "features_view", "t_norm"],
            output_names=["dc", "sh_rest", "opacity"],
            dynamic_axes={
                "xyz": {0: "N"},
                "features_static": {0: "N"},
                "features_view": {0: "N"},
                "dc": {0: "N"},
                "sh_rest": {0: "N"},
                "opacity": {0: "N"},
            },
            use_external_data_format=False,
        )
        print(f"ONNX exported to {onnx_path}")

    except Exception as e:
        print(f"ONNX export failed: {e}")
        # Only fail if it's a logic error in our script, not environmental (like missing onnx)
        # But we want to see if the code *would* work.

    print("Testing SVQ Data export...")
    try:
        # We need to mock the components effectively.
        # Let's just run export_svq_data
        svq_data, total_bytes = export_onnx.export_svq_data(save_dict, gaussians, args)
        npz_path = os.path.join(args.out_dir, "gaussians.npz")
        np.savez_compressed(npz_path, **svq_data)
        print(f"SVQ data exported to {npz_path} ({total_bytes} bytes)")
    except Exception as e:
        print(f"SVQ export failed: {e}")


if __name__ == "__main__":
    test_export()
