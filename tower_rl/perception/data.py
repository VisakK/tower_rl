"""Training / validation samples for the keypoint network: one delivered camera image with its block-corner and
block-edge keypoints (sub-pixel, visible ones only) and its visible-block mask.

Shards come from ``scripts/render_perception_data.py`` (renders: ids maps, corner visibility from Isaac depth) or
``scripts/fetch_dataset_frames.py`` (organizer JPEGs: no ids maps, so visibility from the boxes alone; validation).
Edge points are sampled every ``EDGE_STEP_PX`` along each projected box edge and kept when the ray through the point
meets no block before it (exact, per point) and no arm or finger covers it (ids value 254 in a 3 x 3 window).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from tower_rl import geometry as geo
from tower_rl.perception.boxes import corners_world
from tower_rl.perception.calib import EDGES
from tower_rl.perception.cameras import CAMERA_IDS, Camera

OCCLUDED, NO_BLOCK = 254, 255
EDGE_STEP_PX = 0.7
MAX_CORNERS, MAX_EDGE_PTS = 512, 24000


def load_shards(roots, variants=None) -> list[tuple[Path, dict]]:
    """``(shard dir, record)`` for every frame under the given roots (``<root>/<variant>/<shard>/frames.jsonl``)."""
    out = []
    for root in roots:
        for fj in sorted(Path(root).glob("*/*/frames.jsonl")):
            if variants and fj.parent.parent.name not in variants:
                continue
            for line in open(fj):
                out.append((fj.parent, json.loads(line)))
    return out


def first_hit_depth(cam: Camera, uv: np.ndarray, pos, quat, half) -> np.ndarray:
    """Depth (along the optical axis) of the first block surface on the ray through each pixel point ``uv [P, 2]``."""
    k = cam.K
    d_cam = np.stack([(uv[:, 0] - k[0, 2]) / k[0, 0], (uv[:, 1] - k[1, 2]) / k[1, 1], np.ones(len(uv))], -1)
    d_w = d_cam @ cam.R_wc.T
    r = geo.quat_to_mat(torch.as_tensor(np.asarray(quat), dtype=torch.float64)).numpy()  # [B, 3, 3]
    best = np.full(len(uv), np.inf)
    for b in range(len(pos)):
        o = r[b].T @ (cam.t_wc - np.asarray(pos[b]))
        d = d_w @ r[b]
        d = np.where(np.abs(d) < 1e-12, 1e-12, d)
        t1 = (-half[b] - o) / d
        t2 = (half[b] - o) / d
        tmin = np.minimum(t1, t2).max(-1)
        tmax = np.maximum(t1, t2).min(-1)
        hit = (tmax >= tmin) & (tmin > 0)
        best = np.where(hit & (tmin < best), tmin, best)
    return best


def keypoints(rec: dict, cam_name: str, ids: np.ndarray | None):
    """Visible corners ``[Nc, 2]`` and visible edge points ``[Ne, 2]`` (delivered-image pixels) of one camera."""
    cam = Camera.from_dict(rec["cams"][cam_name])
    pos, quat, half = np.asarray(rec["pos"]), np.asarray(rec["quat"]), np.asarray(rec["half"])
    cw = corners_world(pos, quat, half)  # [B, 8, 3]
    uv, z = cam.project(cw)
    w, h = cam.width, cam.height
    inside = (z > 0.01) & (uv[..., 0] >= 0) & (uv[..., 0] <= w - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= h - 1)
    if "vis" in rec:
        bits = np.asarray(rec["vis"][cam_name], dtype=np.int64)
        vis = ((bits[:, None] >> np.arange(8)[None]) & 1).astype(bool) & inside
    else:
        dep = first_hit_depth(cam, uv.reshape(-1, 2), pos, quat, half).reshape(uv.shape[:2])
        vis = inside & (dep >= z - 1e-3)
    corners = uv[vis]
    pts = []
    for b in range(len(pos)):
        if not (z[b] > 0.01).all():
            continue
        for i, j in EDGES:
            n = int(math.ceil(np.linalg.norm(uv[b, i] - uv[b, j]) / EDGE_STEP_PX)) + 1
            if n < 2 or n > 4000:
                continue
            s = np.linspace(0.0, 1.0, n)[:, None]
            p3 = cw[b, i] * (1 - s) + cw[b, j] * s
            p2, pz = cam.project(p3)
            ok = (pz > 0.01) & (p2[:, 0] >= 0) & (p2[:, 0] <= w - 1) & (p2[:, 1] >= 0) & (p2[:, 1] <= h - 1)
            if ok.any():
                pts.append((p2[ok], pz[ok]))
    if pts:
        e2 = np.concatenate([p for p, _ in pts])
        ez = np.concatenate([q for _, q in pts])
        dep = first_hit_depth(cam, e2, pos, quat, half)
        keep = dep >= ez - 1e-3
        if ids is not None:
            occ = ids == OCCLUDED
            if occ.any():
                occ = cv2.dilate(occ.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                keep &= ~occ[np.rint(e2[:, 1]).astype(int), np.rint(e2[:, 0]).astype(int)]
        edges = e2[keep]
    else:
        edges = np.zeros((0, 2))
    return corners, edges


def _affine(rng: np.random.Generator, w: int, h: int, scale=(0.9, 1.1), rot_deg=4.0, shift_px=24.0) -> np.ndarray:
    s = rng.uniform(*scale)
    a = math.radians(rng.uniform(-rot_deg, rot_deg))
    c = np.array([w / 2, h / 2])
    m = np.array([[s * math.cos(a), -s * math.sin(a)], [s * math.sin(a), s * math.cos(a)]])
    t = c - m @ c + rng.uniform(-shift_px, shift_px, 2)
    return np.concatenate([m, t[:, None]], 1)  # [2, 3]: x' = M x + t


def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Lighting-like changes (exposure, gamma, white balance, contrast, saturation) plus sensor-like noise / blur."""
    x = img.astype(np.float32) / 255.0
    x = np.clip(x, 0, 1) ** rng.uniform(0.75, 1.35)
    x = x * rng.uniform(0.55, 1.45) * rng.uniform(0.85, 1.15, 3)
    m = x.mean()
    x = (x - m) * rng.uniform(0.75, 1.25) + m
    g = x.mean(-1, keepdims=True)
    x = g + (x - g) * rng.uniform(0.6, 1.4)
    if rng.random() < 0.3:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 0.9))
    x = x + rng.normal(0, rng.uniform(0, 0.012), x.shape).astype(np.float32)
    x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    if rng.random() < 0.3:
        ok, enc = cv2.imencode(".jpg", x[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 96))])
        x = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
    return x


class KeypointDataset(Dataset):
    """``(image [3, H, W] float, mask [H, W] float (0 no block, 1 visible block, 2 block hidden by an arm or finger,
    -1 unknown), corners [MAX_CORNERS, 2], n_corners, edge points [MAX_EDGE_PTS, 2], n_edge)``; ``augment`` adds
    photometric changes and a mild similarity warp (labels warped alike)."""

    def __init__(self, frames: list[tuple[Path, dict]], cams=CAMERA_IDS, augment: bool = True, seed: int = 0,
                 crop: tuple[int, int] | None = None):
        self.items = [(i, c) for i in range(len(frames)) for c in cams]
        self.frames = frames
        self.augment = augment
        self.seed = seed
        self.crop = crop

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        fi, cam = self.items[k]
        shard, rec = self.frames[fi]
        # fresh entropy per draw: workers keep their own copy of the dataset (an epoch counter would not reach them)
        rng = np.random.default_rng((self.seed, k, int.from_bytes(os.urandom(4), "little")) if self.augment else (self.seed, k))
        img = cv2.imread(str(shard / "img" / f"{rec['idx']:06d}_{cam}.jpg"), cv2.IMREAD_COLOR)[..., ::-1]
        ids_path = shard / "ids" / f"{rec['idx']:06d}_{cam}.png"
        ids = cv2.imread(str(ids_path), cv2.IMREAD_UNCHANGED) if ids_path.exists() else None
        corners, edges = keypoints(rec, cam, ids)
        h, w = img.shape[:2]
        if ids is not None:
            mask = (ids < OCCLUDED).astype(np.float32)
            mask[ids == OCCLUDED] = 2.0  # a block surface hidden by an arm / finger
        else:
            mask = np.full((h, w), -1.0, np.float32)
        if self.augment:
            a = _affine(rng, w, h)
            img = cv2.warpAffine(np.ascontiguousarray(img), a, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, a, (w, h), flags=cv2.INTER_NEAREST, borderValue=-1.0)
            corners = corners @ a[:, :2].T + a[:, 2]
            edges = edges @ a[:, :2].T + a[:, 2]
            img = photometric(img, rng)
        if self.crop is not None:
            cw_, ch_ = self.crop
            x0 = int(rng.integers(0, w - cw_ + 1))
            y0 = int(rng.integers(0, h - ch_ + 1))
            img, mask = img[y0:y0 + ch_, x0:x0 + cw_], mask[y0:y0 + ch_, x0:x0 + cw_]
            corners, edges = corners - [x0, y0], edges - [x0, y0]
            h, w = ch_, cw_

        def clip_pts(p, n_max):
            p = p[(p[:, 0] >= 0) & (p[:, 0] <= w - 1) & (p[:, 1] >= 0) & (p[:, 1] <= h - 1)]
            if len(p) > n_max:
                p = p[rng.choice(len(p), n_max, replace=False)]
            out = np.zeros((n_max, 2), np.float32)
            out[: len(p)] = p
            return out, len(p)

        c, nc = clip_pts(corners, MAX_CORNERS)
        e, ne = clip_pts(edges, MAX_EDGE_PTS)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
        return x, torch.from_numpy(np.ascontiguousarray(mask)), torch.from_numpy(c), nc, torch.from_numpy(e), ne


def splat(points: torch.Tensor, counts: torch.Tensor, h: int, w: int, sigma: float, radius: int) -> torch.Tensor:
    """Gaussian heatmaps ``[N, H, W]`` (max over points) from padded sub-pixel points ``[N, P, 2]`` (x, y)."""
    n, p, _ = points.shape
    dev = points.device
    valid = torch.arange(p, device=dev)[None] < counts[:, None]
    o = torch.arange(-radius, radius + 1, device=dev, dtype=points.dtype)
    oy, ox = torch.meshgrid(o, o, indexing="ij")
    base = points.round()
    px = base[..., 0:1] + ox.reshape(1, 1, -1)  # [N, P, K]
    py = base[..., 1:2] + oy.reshape(1, 1, -1)
    val = torch.exp(-((px - points[..., 0:1]) ** 2 + (py - points[..., 1:2]) ** 2) / (2 * sigma * sigma))
    ok = valid[..., None] & (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)
    idx = (torch.arange(n, device=dev).view(n, 1, 1) * h * w + py.clamp(0, h - 1).long() * w + px.clamp(0, w - 1).long())
    out = torch.zeros(n * h * w, device=dev, dtype=points.dtype)
    out.scatter_reduce_(0, idx[ok], val[ok], reduce="amax")
    return out.view(n, h, w)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
