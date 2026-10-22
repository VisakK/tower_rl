"""Image-model checks: how well projected block edges sit on image edges (a camera model is right when the best
sub-pixel shift of the projection is ~(0, 0))."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from tower_rl.perception.boxes import raycast
from tower_rl.perception.cameras import Camera

EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]  # the 12 box edges


def visible_edge_points(cam: Camera, corners: np.ndarray, pos, quat, half, n: int = 40) -> np.ndarray:
    """Pixels ``[P, 2]`` along the blocks' edges that no block hides (blocks only: arms are not modelled)."""
    depth = raycast(cam, pos, quat, half)[0].numpy()
    pts = []
    s = np.linspace(0.05, 0.95, n)[:, None]
    for b in range(len(corners)):
        for i, j in EDGES:
            uv, z = cam.project(corners[b, i] * (1 - s) + corners[b, j] * s)
            ui, vi = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
            ok = (z > 0) & (ui >= 1) & (ui < cam.width - 1) & (vi >= 1) & (vi < cam.height - 1)
            ok &= depth[np.clip(vi, 0, cam.height - 1), np.clip(ui, 0, cam.width - 1)] >= z - 0.003
            pts.append(uv[ok])
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def edge_shift(img_rgb: np.ndarray, pts: np.ndarray, r: float = 3.0, step: float = 0.25) -> dict:
    """Mean image gradient along ``pts`` at the model and the shift (px) that maximises it."""
    g = ndimage.gaussian_filter(np.asarray(img_rgb, dtype=np.float64).mean(-1), 0.7)
    grad = np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))

    def score(dx, dy):
        return float(ndimage.map_coordinates(grad, [pts[:, 1] + dy, pts[:, 0] + dx], order=1).mean())

    best = max((score(dx, dy), dx, dy) for dy in np.arange(-r, r + 1e-9, step) for dx in np.arange(-r, r + 1e-9, step))
    return {"points": int(len(pts)), "grad_at_model": round(score(0.0, 0.0), 1), "best_shift_px": [float(best[1]), float(best[2])],
            "grad_at_best": round(best[0], 1)}
