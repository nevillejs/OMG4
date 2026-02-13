"""
Export OMG4 MLP inference pipeline to ONNX + SVQ-compressed Gaussian data.

Produces:
  - <out_dir>/mlp_inference.onnx   : MLP pipeline (~85 KB)
  - <out_dir>/gaussians.npz        : SVQ-compressed Gaussian data (~3 MB for 100K pts)

The gaussians.npz keeps the original SVQ compression: codebooks + uint16 indices.
Client-side reconstruction is a single gather per attribute:
    scales[:, sv*d:(sv+1)*d] = scale_codebook_sv[scale_index_sv]

ONNX model signature:
  Inputs:
    xyz              [N, 3]   float32 — Gaussian positions
    features_static  [N, 3]   float32 — Static appearance features (from app SVQ [:, 0:3])
    features_view    [N, 3]   float32 — View-dependent features (from app SVQ [:, 3:6])
    t_norm           [1]      float32 — Normalized timestamp in [0, 1]
  Outputs:
    dc               [N, 3]   float32 — SH DC component
    sh_rest          [N, 141] float32 — SH higher-order (reshape to [N,47,3])
    opacity          [N, 1]   float32 — Sigmoid-activated opacity

Usage:
    python export_onnx.py \
        --config configs/dynerf/cook_spinach.yaml \
        --comp_checkpoint path/to/comp.xz \
        --out_dir export/
"""

import os, sys, math, lzma, pickle, random
import numpy as np
import torch
from torch import nn
from argparse import ArgumentParser
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.compress_utils import huffman_decode


# ---------- Exportable wrapper ------------------------------------------------

class MLPInference(nn.Module):
    """Wraps the 4 OMG4 MLPs + preprocessing into a single ONNX-exportable module."""

    def __init__(self, mlp_cont, mlp_view, mlp_dc, mlp_opacity, freq_buffers):
        super().__init__()
        self.mlp_cont_mlp = mlp_cont.mlp
        self.mlp_view = mlp_view
        self.mlp_dc = mlp_dc
        self.mlp_opacity = mlp_opacity
        self.register_buffer('freqs', freq_buffers)

    def frequency_encode(self, x):
        x_expanded = x.unsqueeze(-1) * self.freqs
        encoded = torch.cat([torch.sin(x_expanded),
                             torch.cos(x_expanded)], dim=-1)
        return encoded.reshape(x.shape[0], -1)

    def contract_to_unisphere(self, x):
        # aabb = [-1, -1, -1, 1, 1, 1]
        x = (x + 1.0) / 2.0
        x = x * 2.0 - 1.0
        mag = torch.norm(x, dim=-1, keepdim=True)
        mask = (mag > 1.0).squeeze(-1)
        contracted = (2.0 - 1.0 / mag) * (x / mag)
        x = torch.where(mask.unsqueeze(-1), contracted, x)
        x = x / 4.0 + 0.5
        return x

    def forward(self, xyz, features_static, features_view, t_norm):
        N = xyz.shape[0]
        xyz_c = self.contract_to_unisphere(xyz)
        t_expanded = t_norm.expand(N, 1)
        xyzt = torch.cat([xyz_c, t_expanded], dim=1)

        encoded = self.frequency_encode(xyzt)
        cont = self.mlp_cont_mlp(encoded)

        space_feat = torch.cat([cont, features_static], dim=1)
        view_feat = torch.cat([cont, features_view], dim=1)

        dc = self.mlp_dc(space_feat)
        sh_rest = self.mlp_view(view_feat)
        opacity = torch.sigmoid(self.mlp_opacity(space_feat))

        return dc, sh_rest, opacity


# ---------- Load model --------------------------------------------------------

def load_model(dataset, opt, pipe, args):
    time_duration = list(args.time_duration)
    if dataset.frame_ratio > 1:
        time_duration = [t / dataset.frame_ratio for t in time_duration]

    sh_degree_t = 2 if pipe.eval_shfs_4d else 0
    gaussians = GaussianModel(
        dataset.sh_degree, gaussian_dim=args.gaussian_dim,
        time_duration=time_duration, rot_4d=args.rot_4d,
        force_sh_3d=args.force_sh_3d, sh_degree_t=sh_degree_t,
    )
    scene = Scene(dataset, gaussians, num_pts=args.num_pts,
                  num_pts_ratio=args.num_pts_ratio, time_duration=time_duration)
    gaussians.training_setup(opt)

    with lzma.open(args.comp_checkpoint, "rb") as f:
        load_dict = pickle.load(f)
    gaussians.decode(load_dict, decompress=True)
    gaussians.active_sh_degree = 3
    gaussians.active_sh_degree_t = 2
    if gaussians.env_map.device != torch.device("cuda"):
        gaussians.env_map = gaussians.env_map.to("cuda")

    return gaussians, scene, load_dict


# ---------- Export SVQ data ---------------------------------------------------

def export_svq_data(save_dict, gaussians, args):
    """Export Gaussian data preserving SVQ codebook+index format."""
    data = {}

    # --- xyz and t as float16 ---
    data["xyz"] = np.array(save_dict["xyz"].cpu().numpy(), dtype=np.float16)
    data["time_duration"] = np.array(gaussians.time_duration, dtype=np.float32)
    data["gaussian_dim"] = np.array([args.gaussian_dim], dtype=np.int32)
    data["rot_4d"] = np.array([int(args.rot_4d)], dtype=np.int32)

    total_bytes = data["xyz"].nbytes

    # --- SVQ attributes: codebook (fp16) + indices (uint16) ---
    svq_attrs = [
        ("scale",     "scale_code",      "scale_index",      "scale_htable"),
        ("rotation",  "rotation_code",   "rotation_index",   "rotation_htable"),
        ("app",       "app_code",        "app_index",        "app_htable"),
    ]
    if args.gaussian_dim == 4:
        data["t"] = np.array(save_dict["t"], dtype=np.float16)
        total_bytes += data["t"].nbytes
        svq_attrs += [
            ("scaling_t",  "scaling_t_code",  "scaling_t_index",  "scaling_t_htable"),
            ("rotation_r", "rotation_r_code", "rotation_r_index", "rotation_r_htable"),
        ]

    for attr_name, code_key, index_key, htable_key in svq_attrs:
        n_subvec = len(save_dict[code_key])
        data[f"{attr_name}_n_subvec"] = np.array([n_subvec], dtype=np.int32)
        for i in range(n_subvec):
            # Codebook: [K, sub_dim] float16
            codebook = np.array(save_dict[code_key][i], dtype=np.float16)
            data[f"{attr_name}_codebook_{i}"] = codebook
            total_bytes += codebook.nbytes

            # Indices: Huffman decode → [N] uint16
            indices = huffman_decode(save_dict[index_key][i], save_dict[htable_key][i])
            data[f"{attr_name}_index_{i}"] = indices.astype(np.uint16)
            total_bytes += indices.nbytes

    return data, total_bytes


# ---------- Main --------------------------------------------------------------

def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--comp_checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="export")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (17 works with onnxruntime-web)")

    args = parser.parse_args(sys.argv[1:])

    cfg = OmegaConf.load(args.config)
    def _merge(key, host):
        if isinstance(host[key], DictConfig):
            for k in host[key]:
                _merge(k, host[key])
        elif hasattr(args, key):
            setattr(args, key, host[key])
    for k in cfg:
        _merge(k, cfg)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    pipe.env_map_res = 0

    print(f"Loading checkpoint: {args.comp_checkpoint}")
    gaussians, scene, save_dict = load_model(dataset, opt, pipe, args)

    N = gaussians.get_xyz.shape[0]
    print(f"Loaded {N:,} Gaussians, gaussian_dim={args.gaussian_dim}, rot_4d={args.rot_4d}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Build and export ONNX ----
    wrapper = MLPInference(
        mlp_cont=gaussians.mlp_cont,
        mlp_view=gaussians.mlp_view,
        mlp_dc=gaussians.mlp_dc,
        mlp_opacity=gaussians.mlp_opacity,
        freq_buffers=gaussians.mlp_cont.encoding.freqs,
    ).cpu().float()
    wrapper.eval()

    with torch.no_grad():
        xyz_test = gaussians.get_xyz.cpu().float()
        fs_test = gaussians._features_static.data.cpu().float()
        fv_test = gaussians._features_view.data.cpu().float()
        t_test = torch.tensor([0.5])
        dc_test, sh_test, op_test = wrapper(xyz_test, fs_test, fv_test, t_test)
    print(f"  Verification — dc: {dc_test.shape}, sh: {sh_test.shape}, opacity: {op_test.shape}")

    onnx_path = os.path.join(args.out_dir, "mlp_inference.onnx")
    print(f"Exporting ONNX → {onnx_path}")

    torch.onnx.export(
        wrapper,
        (torch.randn(N, 3), torch.randn(N, 3), torch.randn(N, 3), torch.tensor([0.5])),
        onnx_path,
        opset_version=args.opset,
        input_names=["xyz", "features_static", "features_view", "t_norm"],
        output_names=["dc", "sh_rest", "opacity"],
        dynamic_axes={
            "xyz": {0: "N"}, "features_static": {0: "N"},
            "features_view": {0: "N"}, "dc": {0: "N"},
            "sh_rest": {0: "N"}, "opacity": {0: "N"},
        },
    )

    # ---- Export SVQ-compressed Gaussian data ----
    npz_path = os.path.join(args.out_dir, "gaussians.npz")
    print(f"Exporting SVQ data → {npz_path}")

    svq_data, raw_bytes = export_svq_data(save_dict, gaussians, args)
    np.savez_compressed(npz_path, **svq_data)

    # ---- Summary ----
    onnx_size = os.path.getsize(onnx_path)
    npz_size = os.path.getsize(npz_path)
    print(f"\n{'='*60}")
    print(f"  mlp_inference.onnx   {onnx_size / 1024:>8.1f} KB   (MLP weights)")
    print(f"  gaussians.npz        {npz_size / 1024:>8.1f} KB   (SVQ codebooks + indices)")
    print(f"  raw data (pre-zip)   {raw_bytes / 1024:>8.1f} KB")
    print(f"{'='*60}")
    print(f"  Gaussians: {N:,}")
    print(f"  MLP params: {sum(p.numel() for p in wrapper.parameters()):,}")
    print()
    print("Client-side load:")
    print("  1. npz = load('gaussians.npz')")
    print("  2. Reconstruct per-attribute via codebook gather:")
    print("       scale[:, sv*d:(sv+1)*d] = scale_codebook_sv[scale_index_sv]")
    print("  3. Appearance sub-vectors reconstruct to [N,6]:")
    print("       features_static = app[:, 0:3]")
    print("       features_view   = app[:, 3:6]")
    print("  4. Per-frame: session.run({xyz, features_static, features_view, t_norm})")
    print("  5. If 4D: opacity *= exp(-0.5*(t - ts)^2 / sigma_t)")


if __name__ == "__main__":
    main()
