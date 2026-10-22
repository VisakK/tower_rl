"""Per-view labels from a rendered depth image: per-corner visibility and the per-pixel block-id map (block index;
254 = a block surface hidden by something else, an arm or a finger; 255 = no block).

The depth comes from an Isaac render (``cam_ren``: principal point at the image centre). The labelled image may be
another camera model with the same pose (``cam_lab``: the organizers' principal point, for their JPEGs): block ids are
then ray cast exactly in ``cam_lab`` and only the hidden-by-an-arm mask is carried over from the render (nearest pixel,
<= 0.5 px off), so no depth is resampled across silhouettes.
"""

from __future__ import annotations

import cv2
import numpy as np

from tower_rl.perception.boxes import corner_visibility, corners_world, raycast
from tower_rl.perception.cameras import Camera

OCCLUDED, NO_BLOCK = 254, 255


def _to_lab(mask: np.ndarray, cam_ren: Camera, cam_lab: Camera) -> np.ndarray:
    """A mask of ``cam_ren``'s image in ``cam_lab``'s pixel coordinates (same pose, different intrinsics)."""
    kr, kl = cam_ren.K, cam_lab.K
    sx, sy = kr[0, 0] / kl[0, 0], kr[1, 1] / kl[1, 1]
    m = np.array([[sx, 0.0, kr[0, 2] - sx * kl[0, 2]], [0.0, sy, kr[1, 2] - sy * kl[1, 2]]])  # lab pixel -> render pixel
    return cv2.warpAffine(mask.astype(np.uint8), m, (cam_lab.width, cam_lab.height),
                          flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0).astype(bool)


def view_labels(cam_ren: Camera, depth_ren: np.ndarray, pos, quat, half, cam_lab: Camera | None = None,
                device: str = "cpu", tol_m: float = 0.003):
    """``(ids [H, W] uint8, corner visibility bits [B] (8 per block), visible pixels per block [B])`` of the view."""
    da, ia = (x.cpu().numpy() for x in raycast(cam_ren, pos, quat, half, device=device))
    with np.errstate(invalid="ignore"):
        occ = (ia >= 0) & (depth_ren < da - tol_m)
        agree = (ia >= 0) & (np.abs(depth_ren - da) < tol_m)
    _, _, v = corner_visibility(cam_ren, corners_world(pos, quat, half), depth_ren)
    if cam_lab is None:
        ids = np.where(agree, ia, NO_BLOCK).astype(np.uint8)
        ids[occ] = OCCLUDED
    else:
        _, il = (x.cpu().numpy() for x in raycast(cam_lab, pos, quat, half, device=device))
        ids = np.where(il >= 0, il, NO_BLOCK).astype(np.uint8)
        ids[(il >= 0) & _to_lab(occ, cam_ren, cam_lab)] = OCCLUDED
    bits = [int(sum(int(x) << k for k, x in enumerate(row))) for row in v]
    npix = np.bincount(ids[ids < OCCLUDED].ravel(), minlength=len(pos)).tolist()
    return ids, bits, npix
