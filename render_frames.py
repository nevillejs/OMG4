"""
Render compressed OMG4 Gaussian Splats at arbitrary timesteps.

Loads a .xz checkpoint, decodes Gaussians + MLPs, and for each
requested timestep runs MLP inference to produce per-Gaussian
attributes (xyz, opacity, SH, scales, rotations). Saves RGB frames
via the built-in rasterizer and optionally dumps raw .pt attributes
for use with your own renderer.

Usage:
    python render_frames.py \
        --config configs/your_config.yaml \
        --comp_checkpoint path/to/compressed.xz \
        --timesteps 0.0 0.25 0.5 0.75 1.0 \
        --out_dir output_frames \
        --save_attrs
"""

import os, sys, math, random, time, lzma, pickle
import numpy as np
import torch
from torch import nn
from argparse import ArgumentParser
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from PIL import Image

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render


# ---------- helpers ----------------------------------------------------------

def save_image(tensor, path):
    img = (tensor.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


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

    return gaussians, scene


# ---------- MLP inference for your own renderer -----------------------------

@torch.no_grad()
def infer_gaussians_at_t(pc, timestamp):
    """
    Run MLP inference at *timestamp* and return every per-Gaussian
    attribute needed to rasterize:

        xyz        [N,3]    positions (with 4D mean-offset when rot_4d)
        opacity    [N,1]    after sigmoid + marginal-t modulation
        shs        [N,48,3] DC (1) + rest (47) SH bands
        scales     [N,3]    exp-activated spatial scales
        rotations  [N,4]    normalised quaternions
        scales_t   [N,1]    temporal scale   (4D, else None)
        rotations_r[N,4]    temporal rotation(4D+rot_4d, else None)
        ts         [N,1]    per-Gaussian centre time (4D, else None)
        cov3D      [N,6]    precomputed 3D covariance (rot_4d, else None)
    """
    t_min, t_max = pc.time_duration
    means3D = pc.get_xyz

    # -- MLP path (mirrors gaussian_renderer/__init__.py:132-151) --
    t_norm = (timestamp - t_min) / (t_max - t_min)
    xyz_c = pc.contract_to_unisphere(
        means3D.clone().detach(),
        torch.tensor([-1., -1., -1., 1., 1., 1.], device="cuda"),
    )
    xyzt = torch.cat([xyz_c, xyz_c.new_full((xyz_c.shape[0], 1), t_norm)], 1)

    cont = pc.mlp_cont(xyzt)                                        # [N,13]
    space_feat = torch.cat([cont, pc._features_static], -1)         # [N,16]
    view_feat  = torch.cat([cont, pc._features_view],   -1)         # [N,16]

    dc       = pc.mlp_dc(space_feat).reshape(-1, 1, 3).float()      # [N,1,3]
    shs_rest = pc.mlp_view(view_feat).reshape(-1, 47, 3).float()    # [N,47,3]
    opacity  = pc.opacity_activation(pc.mlp_opacity(space_feat).float())  # [N,1]
    shs = torch.cat([dc, shs_rest], 1)                               # [N,48,3]

    # -- geometry --
    scales    = pc.get_scaling      # [N,3]
    rotations = pc.get_rotation     # [N,4]
    scales_t = rotations_r = ts = cov3D = delta_mean = None

    if pc.gaussian_dim == 4:
        scales_t = pc.get_scaling_t
        ts       = pc.get_t
        marginal_t = pc.get_marginal_t(timestamp)
        opacity = opacity * marginal_t

        if pc.rot_4d:
            rotations_r = pc.get_rotation_r
            cov3D, delta_mean = pc.get_current_covariance_and_mean_offset(1.0, timestamp)
            means3D = means3D + delta_mean

    return dict(
        xyz=means3D, opacity=opacity, shs=shs,
        scales=scales, rotations=rotations,
        scales_t=scales_t, rotations_r=rotations_r,
        ts=ts, cov3D=cov3D,
    )


# ---------- main ------------------------------------------------------------

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
    parser.add_argument("--timesteps", nargs="+", type=float, default=None,
                        help="Normalised [0,1] timesteps to render")
    parser.add_argument("--num_timesteps", type=int, default=None,
                        help="Evenly space N timesteps in [0,1]")
    parser.add_argument("--camera_idx", type=int, default=0,
                        help="Test-set camera to use as viewpoint")
    parser.add_argument("--out_dir", type=str, default="rendered_frames")
    parser.add_argument("--save_attrs", action="store_true",
                        help="Dump per-Gaussian .pt for your own renderer")

    args = parser.parse_args(sys.argv[1:])

    # merge yaml config
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
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    pipe.env_map_res = 0

    print(f"Loading checkpoint: {args.comp_checkpoint}")
    gaussians, scene = load_model(dataset, opt, pipe, args)

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    os.makedirs(args.out_dir, exist_ok=True)
    test_cams = scene.getTestCameras()

    t_min, t_max = gaussians.time_duration

    # resolve timesteps
    if args.timesteps is not None:
        t_norms = args.timesteps
    elif args.num_timesteps is not None:
        t_norms = np.linspace(0.0, 1.0, args.num_timesteps).tolist()
    else:
        t_norms = np.linspace(0.0, 1.0, 10).tolist()

    t_actual = [t_min + t * (t_max - t_min) for t in t_norms]

    _, ref_cam = test_cams[args.camera_idx]
    print(f"Camera {args.camera_idx} | {len(t_actual)} timesteps "
          f"[{t_actual[0]:.4f} .. {t_actual[-1]:.4f}]")

    total = 0.0
    for i, (tn, ta) in enumerate(zip(t_norms, t_actual)):
        ref_cam.timestamp = ta
        cam = ref_cam.cuda()

        torch.cuda.synchronize()
        st = time.time()

        # ---- built-in rasteriser (produces RGB) ----
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe=pipe, bg_color=bg)
        image = pkg["render"].clamp(0, 1)

        torch.cuda.synchronize()
        dt = time.time() - st
        total += dt

        out = os.path.join(args.out_dir, f"frame_{i:04d}_t{tn:.4f}.png")
        save_image(image, out)

        # ---- raw per-Gaussian attrs for your renderer ----
        if args.save_attrs:
            attrs = infer_gaussians_at_t(gaussians, ta)
            pt_out = os.path.join(args.out_dir, f"frame_{i:04d}_t{tn:.4f}.pt")
            torch.save({k: v.cpu() if v is not None else None
                        for k, v in attrs.items()}, pt_out)

        print(f"  [{i:3d}] t={tn:.4f}  {dt*1000:.1f} ms  -> {out}")

    print(f"\n{len(t_actual)} frames, {total:.2f}s total, "
          f"{len(t_actual)/total:.1f} FPS")


if __name__ == "__main__":
    main()
