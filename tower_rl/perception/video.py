"""Videos of the pose estimator in the loop (``rehearse.py --perception <model.pt> --video <dir>``): for every estimator
call, the four delivered camera images with every block's estimated pose drawn as its visible box edges, coloured by
its state, the simulator's true pose (white) where an estimate is more than 2 mm off, and a status bar: episode, time,
the executor node with its target block's error, confidence counts. One mp4 per episode at 10 fps, one frame per call:
learned skills (called every 0.1 s row) play in real time, planned transits (one call per chunk of rows) faster.
Isaac-free (numpy, OpenCV, imageio).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from tower_rl.perception.boxes import corners_world, raycast
from tower_rl.perception.calib import EDGES
from tower_rl.perception.cameras import CAMERA_IDS
from tower_rl.perception.model import peaks

COLORS = {"good": (40, 220, 40), "foundation": (70, 150, 255), "held": (0, 225, 225), "target": (255, 60, 255),
          "weak": (255, 215, 0), "unseen": (255, 130, 0), "lost": (255, 40, 40), "truth": (255, 255, 255)}  # RGB
LEGEND = (("good", "tracked"), ("foundation", "foundation"), ("held", "held"), ("target", "node target"),
          ("weak", "weak 0.6"), ("unseen", "not seen 0.3"), ("lost", "lost 0"), ("truth", "truth (> 2 mm off)"))
PANELS = ("right_front_camera", "left_back_camera", "left_wrist_camera", "back_wrist_camera")  # 2 x 2 (trace_episode)
BAR_H = 80
TRUTH_MM = 2.0
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_EI = np.array([i for i, _ in EDGES])
_EJ = np.array([j for _, j in EDGES])


def _text(img: np.ndarray, s: str, org, color=(255, 255, 255), scale: float = 0.5):
    cv2.putText(img, s, (int(org[0]), int(org[1])), _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, (int(org[0]), int(org[1])), _FONT, scale, color, 1, cv2.LINE_AA)


def draw_boxes(img: np.ndarray, cam, pos, quat, half, colors: list, widths: list, device="cpu", n: int = 12,
               tol_m: float = 0.004):
    """Draw the visible edges (not behind any of the boxes; arms are not modelled) of boxes ``pos [B, 3]``, ``quat
    [B, 4]`` (wxyz), ``half [B, 3]`` on ``img`` in place; ``colors[b]`` None skips block b."""
    depth = raycast(cam, pos, quat, half, device=device, stride=2)[0].cpu().numpy()
    cw = corners_world(pos, quat, half)  # [B, 8, 3]
    s = np.linspace(0.0, 1.0, n)[:, None]
    uv, z = cam.project(cw[:, _EI, None] * (1 - s) + cw[:, _EJ, None] * s)  # [B, 12, n, 2], [B, 12, n]
    h, w = depth.shape
    ui = np.clip(np.rint(uv[..., 0] / 2).astype(int), 0, w - 1)
    vi = np.clip(np.rint(uv[..., 1] / 2).astype(int), 0, h - 1)
    inside = (z > 0.02) & (uv[..., 0] >= -50) & (uv[..., 0] <= cam.width + 50) & (uv[..., 1] >= -50) & (uv[..., 1] <= cam.height + 50)
    vis = inside & (depth[vi, ui] >= z - tol_m)
    fix = np.rint(np.nan_to_num(uv) * 16).astype(np.int64)  # sub-pixel lines (shift 4)
    for b in range(len(cw)):
        if colors[b] is None:
            continue
        for k in range(len(EDGES)):
            v = vis[b, k]
            if v.all():
                cv2.line(img, tuple(map(int, fix[b, k, 0])), tuple(map(int, fix[b, k, -1])), colors[b], widths[b], cv2.LINE_AA, 4)
            elif v.any():
                for m in np.nonzero(v[:-1] & v[1:])[0]:
                    cv2.line(img, tuple(map(int, fix[b, k, m])), tuple(map(int, fix[b, k, m + 1])), colors[b], widths[b],
                             cv2.LINE_AA, 4)


def frame(images: dict, cams: dict, half: np.ndarray, pos, quat, conf, fixed, held: list, block_ids: list,
          truth=None, focus: int | None = None, lines=(), maps=None, device="cpu") -> np.ndarray:
    """One video frame ``[960 + BAR_H, 1280, 3]`` uint8 RGB: the four ``images`` (delivered, by camera id) with the
    estimated boxes; ``truth`` (pos, quat) adds the true boxes of blocks estimated > 2 mm off; ``lines``: status bar
    text; ``maps`` (corner, edge probability maps ``[4, H, W]`` in CAMERA_IDS order, numpy) blends the network's edge
    map (blue) and marks its corner peaks (yellow)."""
    pos, quat, conf = np.asarray(pos), np.asarray(quat), np.asarray(conf)
    B = len(pos)
    err = np.linalg.norm(pos - truth[0], axis=-1) * 1000 if truth is not None else np.zeros(B)
    state = []
    for b in range(B):
        if conf[b] <= 0:
            state.append("lost")
        elif conf[b] < 0.5:
            state.append("unseen")
        elif conf[b] < 1:
            state.append("weak")
        else:
            state.append("held" if b in held else "foundation" if fixed[b] else "good")
    colors = [COLORS["target" if b == focus and state[b] in ("good", "held", "foundation") else state[b]] for b in range(B)]
    widths = [2 if b == focus else 1 for b in range(B)]
    off = [b for b in range(B) if truth is not None and err[b] > TRUTH_MM]
    out = np.zeros((960 + BAR_H, 1280, 3), np.uint8)
    for p, name in enumerate(PANELS):
        img = np.ascontiguousarray(images[name]).copy()
        cam = cams[name]
        if maps is not None:
            ci = CAMERA_IDS.index(name)
            img[..., 2] = np.maximum(img[..., 2], (255 * maps[1][ci]).clip(0, 255).astype(np.uint8))
        draw_boxes(img, cam, pos, quat, half, colors, widths, device=device)
        if off:
            draw_boxes(img, cam, truth[0], truth[1], half, [COLORS["truth"] if b in off else None for b in range(B)],
                       [1] * B, device=device)
        if maps is not None:
            for x, y, _ in peaks(torch.as_tensor(maps[0][CAMERA_IDS.index(name)])).numpy():
                cv2.circle(img, (int(round(x * 16)), int(round(y * 16))), 32, (255, 255, 0), 1, cv2.LINE_AA, 4)
        uv, z = cam.project(pos)
        for b in ([focus] if focus is not None else []) + off:  # labels at the block centres
            if z[b] > 0.02 and 0 <= uv[b, 0] < 640 and 0 <= uv[b, 1] < 480:
                lab = block_ids[b] + (f" {err[b]:.0f} mm" if b in off else "")
                _text(img, lab, (uv[b, 0] + 8, uv[b, 1] - 8), colors[b] if b == focus else COLORS["truth"], 0.45)
        _text(img, name.replace("_camera", ""), (8, 472), (230, 230, 230), 0.45)
        out[BAR_H + (p // 2) * 480: BAR_H + (p // 2 + 1) * 480, (p % 2) * 640: (p % 2 + 1) * 640] = img
    out[:BAR_H] = 28
    for i, s in enumerate(lines[:3]):
        _text(out, s, (8, 19 + 19 * i), (235, 235, 235))
    x = 8
    for key, label in LEGEND:
        cv2.line(out, (x, BAR_H - 12), (x + 18, BAR_H - 12), COLORS[key], 2, cv2.LINE_AA)
        _text(out, label, (x + 22, BAR_H - 7), COLORS[key], 0.45)
        x += 22 + cv2.getTextSize(label, _FONT, 0.45, 1)[0][0] + 16
    return out


class EpisodeVideos:
    """One mp4 per env and episode: ``add(e, name, frame)`` per call; ``close(outcomes)`` after the batch appends the
    episode's result to the file names."""

    def __init__(self, out_dir, maps: bool = False, fps: int = 10):
        self.dir = Path(out_dir)
        self.maps = maps
        self.fps = fps
        self.tier = ""
        self.writers = {}  # env -> (writer, path)

    def add(self, e: int, name: str, img: np.ndarray):
        if e not in self.writers:
            import imageio

            self.dir.mkdir(parents=True, exist_ok=True)
            path = self.dir / f"{name}.mp4"
            self.writers[e] = (imageio.get_writer(path, fps=self.fps, macro_block_size=8), path)
        self.writers[e][0].append_data(img)

    def close(self, outcomes=None):
        for e, (w, path) in sorted(self.writers.items()):
            w.close()
            if outcomes is not None and e < len(outcomes):
                new = path.with_name(f"{path.stem}_{'success' if outcomes[e].task_success else 'fail'}.mp4")
                path = path.rename(new)
            print(f"[video] wrote {path}", flush=True)
        self.writers = {}
