"""Blocks as boxes in the cameras: the rendered extents, corner keypoints, analytic depth / instance rendering by ray
casting, and corner visibility against a rendered depth image. Torch, Isaac-free.

The station block (Foundation01, cloned for every block by the generator and by us) renders as its 10-vertex hull: a
150 x 49.76 x 28.3 mm box plus two keel vertices within 1 mm of two corners, not the nominal 150 x 50 x 30 mm. Task 5's
32 mm side blocks are the same mesh scaled by their dims.
"""

from __future__ import annotations

import numpy as np
import torch

from tower_rl import geometry as geo
from tower_rl.perception.cameras import Camera

NOMINAL_DIMS = (0.15, 0.05, 0.03)
VISUAL_HALF = (0.075, 0.02488, 0.01415)
# corner k = (sx, sy, sz) with s = -1 / +1, x slowest (as tower_rl.geometry.box_corners)
CORNER_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)


def visual_half_extents(dims) -> np.ndarray:
    """Rendered half extents ``[B, 3]`` of blocks with layout dims ``[B, 3]`` (m)."""
    return np.asarray(VISUAL_HALF) * np.asarray(dims, dtype=np.float64).reshape(-1, 3) / np.asarray(NOMINAL_DIMS)


def corners_world(pos, quat_wxyz, half) -> np.ndarray:
    """The 8 corners ``[B, 8, 3]`` of each block in the station frame."""
    r = geo.quat_to_mat(torch.as_tensor(np.asarray(quat_wxyz), dtype=torch.float64)).numpy()  # [B, 3, 3]
    local = CORNER_SIGNS[None] * np.asarray(half)[:, None]  # [B, 8, 3]
    return np.einsum("bij,bkj->bki", r, local) + np.asarray(pos)[:, None]


@torch.no_grad()
def raycast(cam: Camera, pos, quat_wxyz, half, device: str | torch.device = "cpu",
            stride: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic render of boxes: depth along the optical axis ``[H, W]`` (inf where no block) and the index of the
    nearest block ``[H, W]`` (-1 where none), at pixel centres (every ``stride``-th pixel)."""
    dt = torch.float32
    k = torch.as_tensor(cam.K, dtype=dt, device=device)
    r_wc = torch.as_tensor(cam.R_wc, dtype=dt, device=device)
    t_wc = torch.as_tensor(cam.t_wc, dtype=dt, device=device)
    vs = torch.arange(0, cam.height, stride, dtype=dt, device=device)
    us = torch.arange(0, cam.width, stride, dtype=dt, device=device)
    v, u = torch.meshgrid(vs, us, indexing="ij")
    d_cam = torch.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], torch.ones_like(u)], -1)  # z = 1: t is depth
    d_w = d_cam @ r_wc.T
    pos = torch.as_tensor(np.asarray(pos), dtype=dt, device=device).reshape(-1, 3)
    rb = geo.quat_to_mat(torch.as_tensor(np.asarray(quat_wxyz), dtype=dt, device=device).reshape(-1, 4))
    half = torch.as_tensor(np.asarray(half), dtype=dt, device=device).reshape(-1, 3)
    depth = torch.full(u.shape, float("inf"), dtype=dt, device=device)
    index = torch.full(u.shape, -1, dtype=torch.long, device=device)
    for b in range(pos.shape[0]):
        o = rb[b].T @ (t_wc - pos[b])               # camera centre in the block frame
        d = d_w @ rb[b]                             # ray directions in the block frame
        d = torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)
        t1 = (-half[b] - o) / d
        t2 = (half[b] - o) / d
        tmin = torch.minimum(t1, t2).amax(-1)
        tmax = torch.maximum(t1, t2).amin(-1)
        hit = (tmax >= tmin) & (tmin > 0)
        t = torch.where(hit, tmin, torch.full_like(tmin, float("inf")))
        closer = t < depth
        depth = torch.where(closer, t, depth)
        index = torch.where(closer, torch.full_like(index, b), index)
    return depth, index


def corner_visibility(cam: Camera, corners: np.ndarray, depth_img: np.ndarray, tol_m: float = 0.004):
    """Projected corners ``[B, 8, 2]``, their depths and visibility ``[B, 8]`` against a rendered depth image of the
    delivered view (nothing rendered in front of the corner by more than ``tol_m``)."""
    uv, z = cam.project(corners)
    h, w = depth_img.shape
    ui, vi = np.rint(uv[..., 0]).astype(int), np.rint(uv[..., 1]).astype(int)
    inside = (z > 0.01) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    d = np.where(inside, depth_img[np.clip(vi, 0, h - 1), np.clip(ui, 0, w - 1)], 0.0)
    return uv, z, inside & (d >= z - tol_m)
