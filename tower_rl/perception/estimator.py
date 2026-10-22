"""Pose estimator v1: every block's pose from the four delivered camera images, the joint angles and the gripper
widths (the challenge's observation), as the executor's ``State`` fields.

Per call the keypoint network (tower_rl.perception.model) gives corner and edge heatmaps of the four images. Each block
is then fitted by robust Gauss-Newton (6-DoF, all blocks batched) to (a) the corner peaks nearest to its projected
visible corners and (b) the edge maxima found along the normals of its projected visible edges, in all cameras at
once, starting from a prior: the previous estimate, or for a block held in a gripper the gripper's FK pose times the
block's measured offset. "Visible" is predicted by casting rays against all blocks at their priors (arms are not
modelled: points they hide find no match). The first call starts from the variant's nominal layout and a coarse planar
search (x, y, yaw) per block; a block whose fit fails is searched again around its prior.

Camera models: tower_rl.perception.cameras (source "evaluator" for the challenge, "render" for our Isaac renders).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from tower_rl import geometry as geo
from tower_rl.data.layouts import Layout
from tower_rl.kinematics import ArmBase, ArmKinematics
from tower_rl.perception.boxes import CORNER_SIGNS, visual_half_extents
from tower_rl.perception.calib import EDGES
from tower_rl.perception.cameras import CAMERA_IDS, CameraRig
from tower_rl.perception.model import KeypointNet, peaks

ARMS = ("left", "back")


@dataclass
class EstCfg:
    corner_thr: float = 0.3
    corner_gate_px: float = 6.0
    edge_thr: float = 0.25
    edge_search_px: int = 6
    iters: int = 6
    huber_px: float = 1.5
    tukey_px: float = 2.5
    w_corner: float = 1.0
    w_edge: float = 0.3
    prior_pos_m: float = 0.01
    prior_rot_rad: float = 0.05
    hold_width_m: float = 0.055      # measured opening below this: fingers on a block (block 0.0497 m)
    release_width_m: float = 0.065
    empty_width_m: float = 0.046     # fingers closed past the block (49.7 mm): nothing held (dropped)
    hold_dist_m: float = 0.03        # block centre within this of the pinch point when the fingers close
    offset_gain: float = 0.5         # held-block offset update per good fit: a change (slip) beyond 1 mm / 0.5 deg,
    offset_gain_small: float = 0.15  # else (measurement noise)
    settle_calls: int = 3            # after a grasp is detected the fingers are still closing (the block shifts ~8 mm in
                                     # the pinch frame within a call: 4-3 trace): the first fits are taken whole
    static_gain: float = 0.3         # a resting block's estimate moves this share toward a fit within static_gate
    static_gate_m: float = 0.0015    # (noise); beyond the gate the fit is taken whole (the block moved)
    static_gate_deg: float = 0.75
    good_inliers: int = 24           # weighted inlier count for confidence 1
    min_inliers: int = 8             # below: search again, then keep the prior (confidence 0.3)
    min_ratio: float = 0.5           # inliers / predicted visible points: below, the fit is rejected (oracle maps: all
    good_ratio: float = 0.6          # fits below 0.5 were wrong, half of 0.5-0.6); below good_ratio: search again
    corner_ratio: float = 0.75       # with >= 3 corners predicted visible, fewer matched than this: search again (a
                                     # block slid along its length keeps its long edges but loses its corners)
    flat_weight: float = 1e4         # foundations lie flat on the table: prior weight multiplier on z, roll, pitch
    rest_calls: int = 10             # a free block unmoved (< 0.5 mm, 0.3 deg per call) this many calls is anchored:
    rest_prior_m: float = 0.0005     # its fit is held toward the pose it came to rest at (this sigma; rotation alike),
                                     # so a neighbour's edges cannot drag it call by call (Task 5: a side block crept
                                     # 13 mm while the insert slid in next to it); a real move fails the fit and is
                                     # found by the re-search (weak prior), which ends the anchor
    freeze_after: int = 10           # foundations never move: after this many ok fits, the mean (x, y, yaw) is frozen
                                     # and no longer fitted (blocks stacked on a pad hide it and pull its fit; 0: off)
    foundation_prior_m: float = 0.005  # first call: foundations held toward the nominal layout (dataset spread ~2 mm)
    search_xy_m: float = 0.024
    search_step_m: float = 0.003
    search_yaw_deg: float = 15.0
    search_yaw_step_deg: float = 1.5
    search_axis_m: float = 0.10      # ... and along the block's own long axis this far (a Task 4 push moved a block
                                     # 31 mm in one 0.1 s call, then 125 mm in 0.4 s)
    max_step_m: float = 0.04         # a block that moved last call is predicted to move as much again (at most this)
    coast_calls: int = 5             # a moving block not found keeps that motion (decaying x0.8 per call) this many
                                     # calls: a frozen pose made the Task 4 push actor push harder (block off the table)
    table_search_m: float = 0.16     # a lost movable block is also looked for lying on the table this far around
    table_step_m: float = 0.008      # (on heatmaps max-pooled over 5 x 5 px, so a coarse hypothesis still scores)
    table_yaw_step_deg: float = 6.0
    table_max_blocks: int = 2        # per call
    lean_deg: tuple = (25.0, 40.0, 55.0, 70.0, 82.0)  # ... or leaning (low end on the table, tilted about its short
                                     # axis: a block tipped off a tower edge rests against it, Task 4 trace)
    lean_yaw_step_deg: float = 12.0
    table_backoff_max: int = 64      # calls between searches for a block not found on the table (doubling up to this)
    ghost_after: int = 3             # failed searches after which a lost block stops occluding and blocking other poses
    touch_m: float = 0.06            # a still block can only have moved if a gripper came this close to its centre
    touch_block_m: float = 0.10      # ... or a moving block's centre this close; such a block counts as touched for
    touch_calls: int = 20            # this many calls. Only moving or touched blocks are searched wide when lost; an
                                     # untouched one is occluded (pose kept): still blocks searched wide were re-found in
                                     # a pushed-off neighbour's place, identities swapped (Task 4 traces)


def block_order(layout: Layout) -> list[str]:
    """The station scene's block order: foundations (source then destination), movables by (layer, id)."""
    f = list(layout.pads["source"]["foundation_blocks"]) + list(layout.pads["destination"]["foundation_blocks"])
    return f + sorted(layout.movable_ids, key=lambda b: (layout.layer[b], b))


def _edge_model() -> tuple[np.ndarray, np.ndarray]:
    """Sample points on the 12 edges in sign space ``[P, 3]`` (corners excluded) and their edge axes ``[P, 3]``."""
    pts, dirs = [], []
    for i, j in EDGES:
        axis = int(np.argmax(np.abs(CORNER_SIGNS[i] - CORNER_SIGNS[j])))
        n = {0: 24, 1: 8, 2: 6}[axis]
        for s in np.linspace(0.06, 0.94, n):
            pts.append(CORNER_SIGNS[i] * (1 - s) + CORNER_SIGNS[j] * s)
            d = np.zeros(3)
            d[axis] = 1.0
            dirs.append(d)
    return np.asarray(pts), np.asarray(dirs)


EDGE_SIGNS, EDGE_DIRS = _edge_model()
N_CORNER, N_EDGE = 8, len(EDGE_SIGNS)
# faces adjacent to each model point (corners: 3, edge samples: 2), as unit normals in the block frame [Q, 3, 3]
_ALL_SIGNS = np.concatenate([CORNER_SIGNS, EDGE_SIGNS])
FACE_NORMALS = np.stack([np.diag(np.where(np.abs(_ALL_SIGNS[q]) > 0.999, np.sign(_ALL_SIGNS[q]), 0.0)) for q in range(len(_ALL_SIGNS))])


def _rodrigues(rv: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``[..., 3, 3]`` of rotation vectors ``[..., 3]``."""
    return geo.quat_to_mat(geo.quat_from_rotvec(rv))


def load_net(checkpoint: str, device) -> KeypointNet:
    net = KeypointNet(pretrained=False).to(device).eval()
    net.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    return net


class PoseEstimator:
    def __init__(self, checkpoint: str | None, device: str = "cuda", source: str = "evaluator", cfg: EstCfg | None = None,
                 net: KeypointNet | None = None):
        """``checkpoint`` None and no ``net``: no network (heatmaps are passed to each call, e.g. oracle maps for tests);
        ``net``: a loaded network shared between estimators (one per env)."""
        self.dev = torch.device(device)
        self.cfg = cfg or EstCfg()
        self.net = net
        if net is None and checkpoint is not None:
            self.net = load_net(checkpoint, self.dev)
        self.rig = CameraRig(source)
        self.kin = ArmKinematics(dtype=torch.float64)
        self.bases = {a: ArmBase(a, dtype=torch.float64) for a in ARMS}
        self.edge_signs = torch.as_tensor(EDGE_SIGNS, dtype=torch.float32, device=self.dev)
        self.edge_dirs = torch.as_tensor(EDGE_DIRS, dtype=torch.float32, device=self.dev)
        self.corner_signs = torch.as_tensor(CORNER_SIGNS, dtype=torch.float32, device=self.dev)
        self.face_normals = torch.as_tensor(FACE_NORMALS, dtype=torch.float32, device=self.dev)
        # the box's proper symmetries (half turns about its axes): the same box, another quaternion
        self.sym = torch.stack([torch.diag(torch.tensor(d, dtype=torch.float32)) for d in
                                ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]).to(self.dev)
        self.variant = None
        self._hidden = None  # max-pooled arm-hidden-block map of the current call (per camera), or None

    # ------------------------------------------------------------------ episode state
    def reset(self, variant: str):
        lay = Layout.load(variant)
        self.variant = variant
        self.block_ids = block_order(lay)
        dims = np.array([lay.dims.get(b, (0.15, 0.05, 0.03)) for b in self.block_ids])
        self.half = torch.as_tensor(visual_half_extents(dims), dtype=torch.float32, device=self.dev)
        p7 = torch.as_tensor(np.array([lay.blocks[b] for b in self.block_ids]), dtype=torch.float32, device=self.dev)
        self.pos = p7[:, :3].clone()
        self.R = geo.quat_to_mat(geo.normalize(p7[:, 3:7]))
        self.fixed = torch.tensor([bool(lay.fixed[b]) for b in self.block_ids], device=self.dev)
        self.z0 = self.pos[:, 2].clone()  # nominal heights (foundations: resting on the table)
        self.axis_w = torch.ones(len(self.block_ids), 6, device=self.dev)  # prior weight per (x, y, z, roll, pitch, yaw)
        self.axis_w[self.fixed, 2:5] = self.cfg.flat_weight
        self.conf = torch.full((len(self.block_ids),), 0.3, device=self.dev)
        self.held = {a: None for a in ARMS}
        self.hold_age = {a: 0 for a in ARMS}  # calls since the grasp was detected
        self.vel = torch.zeros(len(self.block_ids), 3, device=self.dev)  # last call's displacement of moving blocks
        self.coast = torch.zeros(len(self.block_ids), device=self.dev)  # calls a lost moving block has coasted
        self.rest_n = torch.zeros(len(self.block_ids), device=self.dev)  # consecutive calls unmoved
        self.fail_n = torch.zeros(len(self.block_ids), device=self.dev)  # consecutive calls without an ok fit
        self.touch_n = torch.zeros(len(self.block_ids), device=self.dev)  # calls left in which a block counts as touched
        self.anchored = torch.zeros(len(self.block_ids), dtype=torch.bool, device=self.dev)
        self.anchor_p = self.pos.clone()
        self.anchor_R = self.R.clone()
        self.offset = {}  # block index -> (p [3], R [3, 3]) in the pinch frame
        self.calls = 0
        self.last = {}
        self.table_fail = {}  # block index -> consecutive failed table searches (the next waits 2^n calls)
        self.ghost = torch.zeros(len(self.block_ids), dtype=torch.bool, device=self.dev)  # long lost: pose unknown
        self.frozen = torch.zeros(len(self.block_ids), dtype=torch.bool, device=self.dev)  # foundations, see freeze_after
        self.f_n = torch.zeros(len(self.block_ids), device=self.dev)  # ok fits of each foundation so far
        self.f_sum = torch.zeros(len(self.block_ids), 4, device=self.dev)  # their sums of x, y, cos 2yaw, sin 2yaw
        self.table_next = {}  # block index -> call index of its next table search

    def _pinch(self, q: np.ndarray, arm: str) -> tuple[torch.Tensor, torch.Tensor]:
        i = ARMS.index(arm)
        p, qq = self.kin.fk(torch.as_tensor(np.asarray(q[i], dtype=np.float64)[None]), "pinch")
        pw, qw = self.bases[arm].to_world(p, qq)
        return pw[0].float().to(self.dev), geo.quat_to_mat(qw[0].float()).to(self.dev)

    # ------------------------------------------------------------------ geometry helpers
    def _points(self, pos, R, which: str = "all") -> torch.Tensor:
        """World points ``[..., Q, 3]`` (corners then edge samples) of blocks with poses ``pos [..., 3]``, ``R [..., 3, 3]``
        (the block index is the dimension before the last of ``pos``)."""
        signs = torch.cat([self.corner_signs, self.edge_signs]) if which == "all" else self.corner_signs
        local = signs[None] * self.half[:, None]  # [B, Q, 3]
        if pos.dim() == 3:  # [B, K, 3]: K poses per block
            local = local[:, None].expand(-1, pos.shape[1], -1, -1)
        return torch.einsum("...ij,...qj->...qi", R, local) + pos[..., None, :]

    @staticmethod
    def _project(xw: torch.Tensor, cam: dict) -> tuple[torch.Tensor, torch.Tensor]:
        xc = (xw - cam["t"]) @ cam["R"]
        z = xc[..., 2]
        uv = xc[..., :2] / z.clamp_min(1e-4)[..., None] * cam["f"] + cam["c"]
        return uv, z

    def _first_hit(self, uv: torch.Tensor, cam: dict, pos: torch.Tensor, R: torch.Tensor, occ: torch.Tensor | None = None) -> torch.Tensor:
        """Depth of the first block surface on the rays through pixel points ``uv [N, 2]`` (blocks at ``pos``, ``R``;
        only the occluders ``occ`` [B] if given)."""
        half = self.half
        if occ is not None:
            pos, R, half = pos[occ], R[occ], half[occ]
        d_cam = torch.cat([(uv - cam["c"]) / cam["f"], torch.ones_like(uv[:, :1])], -1)
        d_w = d_cam @ cam["R"].T
        o = torch.einsum("bji,bj->bi", R, cam["t"][None] - pos)  # camera centre in each block frame [B, 3]
        d = torch.einsum("nj,bji->bni", d_w, R)
        d = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
        h = half[:, None]
        t1 = (-h - o[:, None]) / d
        t2 = (h - o[:, None]) / d
        tmin = torch.minimum(t1, t2).amax(-1)
        tmax = torch.maximum(t1, t2).amin(-1)
        hit = (tmax >= tmin) & (tmin > 0)
        return torch.where(hit, tmin, torch.full_like(tmin, float("inf"))).amin(0)

    def _cams(self, q: np.ndarray) -> list[dict]:
        out = []
        for name, c in self.rig.cameras(q).items():
            out.append({"name": name, "R": torch.as_tensor(c.R_wc, dtype=torch.float32, device=self.dev),
                        "t": torch.as_tensor(c.t_wc, dtype=torch.float32, device=self.dev),
                        "f": torch.as_tensor(np.diag(c.K)[:2].copy(), dtype=torch.float32, device=self.dev),
                        "c": torch.as_tensor(c.K[:2, 2].copy(), dtype=torch.float32, device=self.dev),
                        "w": c.width, "h": c.height})
        return out

    def _visible(self, cams, pos, R, active: torch.Tensor) -> list[torch.Tensor]:
        """Predicted visibility ``[B, Q]`` per camera of every block's corners and edge samples: in the image, not
        behind a block (ray cast at the given poses) and not where this call's "hidden" map (a block surface covered
        by an arm or finger, learned) fires: arms are not modelled geometrically. (The visible-block mask must not be
        used for this: it also says "no block" on background, which excuses points of a misplaced pose.)"""
        xw = self._points(pos, R)  # [B, Q, 3]
        out = []
        for ci, cam in enumerate(cams):
            uv, z = self._project(xw, cam)
            inside = (z > 0.02) & (uv[..., 0] >= 2) & (uv[..., 0] <= cam["w"] - 3) & (uv[..., 1] >= 2) & (uv[..., 1] <= cam["h"] - 3)
            # ghosts (long lost, pose unknown) hide nothing, unless being fitted (a block hides its own back faces)
            dep = self._first_hit(uv.reshape(-1, 2), cam, pos, R, occ=~self.ghost | active).reshape(z.shape)
            v = inside & (dep >= z - 0.0015) & active[:, None]
            if self._hidden is not None:
                grid = torch.stack([(uv[..., 0] + 0.5) / cam["w"] * 2 - 1, (uv[..., 1] + 0.5) / cam["h"] * 2 - 1], -1)
                hid = F.grid_sample(self._hidden[ci][None, None], grid[None], align_corners=False)[0, 0]
                v = v & (hid < 0.5)
            out.append(v)
        return out

    # ------------------------------------------------------------------ measurements
    @torch.no_grad()
    def heatmaps(self, images: dict) -> tuple[torch.Tensor, ...]:
        """Corner, edge, visible-block and arm-hidden-block probability maps ``[4, H, W]`` of the four delivered images."""
        return self.heatmaps_batch(torch.stack([torch.as_tensor(np.ascontiguousarray(images[n])) for n in CAMERA_IDS]))

    @torch.no_grad()
    def heatmaps_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Maps ``[N, H, W]`` of uint8 images ``[N, H, W, 3]``."""
        x = x.to(self.dev).permute(0, 3, 1, 2).float().div_(255.0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            lg = self.net(x)
        p = lg.float().sigmoid()
        return p[:, 0], p[:, 1], p[:, 2], p[:, 3]

    def _edge_targets(self, emap: torch.Tensor, uv: torch.Tensor, nrm: torch.Tensor, r: int):
        """Nearest edge maximum within ``r`` px along each normal: offsets ``[N]`` (px, NaN = none), peak values ``[N]``."""
        s = torch.arange(-r, r + 1, device=self.dev, dtype=torch.float32)
        pts = uv[:, None] + s[None, :, None] * nrm[:, None]  # [N, S, 2]
        h, w = emap.shape
        grid = torch.stack([(pts[..., 0] + 0.5) / w * 2 - 1, (pts[..., 1] + 0.5) / h * 2 - 1], -1)
        v = F.grid_sample(emap[None, None], grid[None], mode="bilinear", align_corners=False)[0, 0]  # [N, S]
        mid = v[:, 1:-1]
        is_max = (mid >= v[:, :-2]) & (mid >= v[:, 2:]) & (mid > self.cfg.edge_thr)
        dist = (s[1:-1].abs()[None]).expand_as(mid).masked_fill(~is_max, float("inf"))
        k = dist.argmin(1)
        found = torch.isfinite(dist.gather(1, k[:, None])[:, 0])
        kk = k + 1
        lv = v.clamp_min(1e-4).log()
        a, b, c = lv.gather(1, (kk - 1)[:, None])[:, 0], lv.gather(1, kk[:, None])[:, 0], lv.gather(1, (kk + 1)[:, None])[:, 0]
        den = a - 2 * b + c
        sub = torch.where(den < -1e-6, 0.5 * (a - c) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
        off = s[kk] + sub
        return torch.where(found, off, torch.full_like(off, float("nan"))), v.gather(1, kk[:, None])[:, 0] * found

    def _associate(self, cams, cmaps, emaps, peaks_c, pos, R, vis, gate_px: float, search_px: int):
        """Targets for every visible point at the current poses: corner targets ``[C, B, 8, 2]`` and weights, edge
        normals ``[C, B, P, 2]``, targets ``[C, B, P, 2]`` and weights."""
        xw = self._points(pos, R)
        e_dir_w = torch.einsum("bij,pj->bpi", R, self.edge_dirs)  # [B, P, 3]
        ct, cw, en, et, ew = [], [], [], [], []
        for ci, cam in enumerate(cams):
            uv, _ = self._project(xw, cam)
            cuv = uv[:, :N_CORNER]
            tgt = torch.zeros_like(cuv)
            wgt = torch.zeros(cuv.shape[:2], device=self.dev)
            pk = peaks_c[ci]
            if len(pk):
                d = torch.cdist(cuv.reshape(-1, 2), pk[:, :2])
                dmin, j = d.min(1)
                ok = (dmin < gate_px).reshape(cuv.shape[:2]) & vis[ci][:, :N_CORNER]
                tgt = pk[j, :2].reshape(cuv.shape)
                wgt = torch.where(ok, pk[j, 2].reshape(cuv.shape[:2]), wgt)
            ct.append(tgt)
            cw.append(wgt)
            euv = uv[:, N_CORNER:]
            uv2, _ = self._project(xw[:, N_CORNER:] + 0.01 * e_dir_w, cam)
            dimg = uv2 - euv
            dimg = dimg / dimg.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            nrm = torch.stack([-dimg[..., 1], dimg[..., 0]], -1)
            m = vis[ci][:, N_CORNER:]
            off = torch.full(m.shape, float("nan"), device=self.dev)
            val = torch.zeros(m.shape, device=self.dev)
            if m.any():
                o_, v_ = self._edge_targets(emaps[ci], euv[m], nrm[m], search_px)
                off[m], val[m] = o_, v_
            en.append(nrm)
            et.append(euv + torch.nan_to_num(off)[..., None] * nrm)
            ew.append(torch.where(torch.isnan(off), torch.zeros_like(val), val))
        return torch.stack(ct), torch.stack(cw), torch.stack(en), torch.stack(et), torch.stack(ew)

    # ------------------------------------------------------------------ fitting
    def _fit(self, cams, cmaps, emaps, peaks_c, pos, R, active: torch.Tensor, prior_scale: torch.Tensor, ref=None):
        """Robust Gauss-Newton from ``pos``, ``R`` [B] with a prior toward ``ref`` (default: the start poses).
        Coarse to fine: wide association windows and Huber weights for two iterations, then narrow windows and Tukey
        weights (wrong matches drop out); visibility re-predicted every two iterations. Returns poses, weighted inlier
        counts and inlier ratios (inliers / predicted visible points) [B], from a final association pass."""
        cfg = self.cfg
        B = pos.shape[0]
        p0, R0 = ref if ref is not None else (pos.clone(), R.clone())
        p0, R0 = self._flat(p0, R0)
        pos, R = self._flat(pos, R)
        eps = torch.tensor([1e-4] * 3 + [1e-3] * 3, device=self.dev)
        lam_prior = torch.cat([torch.full((3,), 1 / cfg.prior_pos_m ** 2), torch.full((3,), 1 / cfg.prior_rot_rad ** 2)]).to(self.dev)
        for it in range(cfg.iters):
            coarse = it < 2
            if it % 2 == 0:
                vis = self._visible(cams, pos, R, active)
            ct, cw, en, et, ew = self._associate(cams, cmaps, emaps, peaks_c, pos, R, vis, cfg.corner_gate_px if coarse else 3.0,
                                                 cfg.edge_search_px if coarse else 3)
            # perturbed poses: 0 = current, 1..12 = +-eps per parameter
            dl = torch.zeros(13, 6, device=self.dev)
            for k in range(6):
                dl[1 + 2 * k, k] = eps[k]
                dl[2 + 2 * k, k] = -eps[k]
            pp = pos[:, None] + dl[None, :, :3]
            RR = _rodrigues(dl[:, 3:])[None] @ R[:, None]  # [B, 13, 3, 3]
            xw = self._points(pp, RR)  # [B, 13, Q, 3]
            H = torch.zeros(B, 6, 6, device=self.dev)
            g = torch.zeros(B, 6, device=self.dev)
            for ci, cam in enumerate(cams):
                uv, _ = self._project(xw, cam)  # [B, 13, Q, 2]
                rc = uv[:, :, :N_CORNER] - ct[ci][:, None]  # [B, 13, 8, 2]
                re = ((uv[:, :, N_CORNER:] - et[ci][:, None]) * en[ci][:, None]).sum(-1)  # [B, 13, P]
                r = torch.cat([rc.flatten(2), re], -1)  # [B, 13, 16 + P]
                J = torch.stack([(r[:, 1 + 2 * k] - r[:, 2 + 2 * k]) / (2 * eps[k]) for k in range(6)], -1)  # [B, N, 6]
                r0 = r[:, 0]
                wc = cw[ci].repeat_interleave(2, dim=1) * cfg.w_corner
                w = torch.cat([wc, ew[ci] * cfg.w_edge], -1)
                cn = rc[:, 0].norm(dim=-1).repeat_interleave(2, dim=1)
                mag = torch.cat([cn, re[:, 0].abs()], -1)
                if coarse:  # Huber
                    w = w * torch.clamp(cfg.huber_px / mag.clamp_min(1e-6), max=1.0)
                else:  # Tukey biweight
                    w = w * (1 - (mag / cfg.tukey_px).clamp(max=1.0) ** 2) ** 2
                H = H + torch.einsum("bni,bn,bnj->bij", J, w, J)
                g = g + torch.einsum("bni,bn,bn->bi", J, w, r0)
            # prior toward the start pose (weak), damping
            dp = pos - p0
            dr = geo.rotvec_from_quat(geo.mat_to_quat(R @ R0.transpose(-1, -2)))
            lam = lam_prior[None] * prior_scale[:, None] * self.axis_w
            H = H + torch.diag_embed(lam) + 1e-6 * torch.eye(6, device=self.dev)
            g = g + lam * torch.cat([dp, dr], -1)
            step = -torch.linalg.solve(H, g[..., None])[..., 0]
            step = torch.where(active[:, None], step, torch.zeros_like(step))
            step[:, :3] = step[:, :3].clamp(-0.01, 0.01)
            step[:, 3:] = step[:, 3:].clamp(-0.1, 0.1)
            pos = pos + step[:, :3]
            R = _rodrigues(step[:, 3:]) @ R
        return (pos, R) + self._score(cams, cmaps, emaps, peaks_c, pos, R, active)

    def _score(self, cams, cmaps, emaps, peaks_c, pos, R, active):
        """Weighted inliers (corner within 2 px: 2, edge sample within 1.5 px: 0.25) and their ratio to the predicted
        visible points, with narrow windows at the given poses. (A minimum over point groups, so that long edges alone
        cannot vouch for a block slid along its length, scored worse on oracle maps: points hidden by an arm, which the
        visibility prediction does not model, drag single groups down.) Also leaves ``self.corner_stats``: the share of
        predicted visible corners matched and their number [B], for the slide test in ``__call__``."""
        vis = self._visible(cams, pos, R, active)
        ct, cw, en, et, ew = self._associate(cams, cmaps, emaps, peaks_c, pos, R, vis, 3.0, 3)
        xw = self._points(pos, R)
        inl = torch.zeros(pos.shape[0], device=self.dev)
        exp = torch.zeros_like(inl)
        c_in, c_vis = torch.zeros_like(inl), torch.zeros_like(inl)
        for ci, cam in enumerate(cams):
            uv, _ = self._project(xw, cam)
            gc = (cw[ci] > 0) & ((uv[:, :N_CORNER] - ct[ci]).norm(dim=-1) < 2.0)
            ge = (ew[ci] > 0) & (((uv[:, N_CORNER:] - et[ci]) * en[ci]).sum(-1).abs() < 1.5)
            inl = inl + 2 * gc.sum(-1) + 0.25 * ge.sum(-1)
            exp = exp + 2 * vis[ci][:, :N_CORNER].sum(-1) + 0.25 * vis[ci][:, N_CORNER:].sum(-1)
            c_in, c_vis = c_in + gc.sum(-1), c_vis + vis[ci][:, :N_CORNER].sum(-1)
        self.corner_stats = (c_in / c_vis.clamp_min(1.0), c_vis)  # matched / predicted visible corners, the latter
        return inl, inl / exp.clamp_min(1.0)

    def _flat(self, pos: torch.Tensor, R: torch.Tensor):
        """Foundation poses put on the table: nominal height, rotation reduced to its yaw (copies)."""
        if not bool(self.fixed.any()):
            return pos, R
        pos, R = pos.clone(), R.clone()
        f = self.fixed
        pos[f, 2] = self.z0[f]
        R[f] = geo.quat_to_mat(geo.quat_from_yaw(torch.atan2(R[f][:, 1, 0], R[f][:, 0, 0])))
        return pos, R

    def _search(self, cams, cmaps, emaps, b: int, pos: torch.Tensor, R: torch.Tensor):
        """Coarse planar search (x, y, yaw about the block centre) for block ``b`` scoring heatmap values at its
        predicted visible corners and edge samples."""
        cfg = self.cfg
        xs = torch.arange(-cfg.search_xy_m, cfg.search_xy_m + 1e-9, cfg.search_step_m, device=self.dev)
        ys = torch.deg2rad(torch.arange(-cfg.search_yaw_deg, cfg.search_yaw_deg + 1e-9, cfg.search_yaw_step_deg, device=self.dev))
        gx, gy, gyaw = torch.meshgrid(xs, xs, ys, indexing="ij")
        n = gx.numel()
        dp = torch.stack([gx.flatten(), gy.flatten(), torch.zeros(n, device=self.dev)], -1)
        Rz = geo.quat_to_mat(geo.quat_from_yaw(gyaw.flatten()))
        ph = pos[b][None] + dp
        Rh = Rz @ R[b][None]
        if cfg.search_axis_m > cfg.search_xy_m:  # and further along the block's own long axis (pushed, slid)
            s = torch.arange(-cfg.search_axis_m, cfg.search_axis_m + 1e-9, cfg.search_step_m, device=self.dev)
            s = s[s.abs() > cfg.search_xy_m]
            ph = torch.cat([ph, pos[b][None] + s[:, None] * R[b][:, 0][None]])
            Rh = torch.cat([Rh, R[b][None].expand(len(s), 3, 3)])
        free = ~self._overlap(b, ph, Rh, pos, R)
        if bool(free.any()):
            ph, Rh = ph[free], Rh[free]
        n = len(ph)
        active = torch.zeros(len(pos), dtype=torch.bool, device=self.dev)
        active[b] = True
        vis = self._visible(cams, pos, R, active)
        signs = torch.cat([self.corner_signs, self.edge_signs])
        local = signs * self.half[b]
        xw = torch.einsum("hij,qj->hqi", Rh, local) + ph[:, None]  # [n, Q, 3]
        score = torch.zeros(n, device=self.dev)
        for ci, cam in enumerate(cams):
            m = vis[ci][b]
            if not m.any():
                continue
            uv, _ = self._project(xw[:, m], cam)
            h, w = cam["h"], cam["w"]
            grid = torch.stack([(uv[..., 0] + 0.5) / w * 2 - 1, (uv[..., 1] + 0.5) / h * 2 - 1], -1)
            mc = int(m[:N_CORNER].sum())
            vals_c = F.grid_sample(cmaps[ci][None, None], grid[None, :, :mc], align_corners=False)[0, 0]
            vals_e = F.grid_sample(emaps[ci][None, None], grid[None, :, mc:], align_corners=False)[0, 0]
            score = score + 2.0 * vals_c.sum(-1) + 0.25 * vals_e.sum(-1)
        k = int(score.argmax())
        return ph[k], Rh[k], float(score[k])

    def _overlap(self, b: int, ph: torch.Tensor, Rh: torch.Tensor, pos: torch.Tensor, R: torch.Tensor,
                 pairwise: bool = False) -> torch.Tensor:
        """Whether poses ``ph [N, 3]``, ``Rh [N, 3, 3]`` of block ``b`` interpenetrate another block (at ``pos``, ``R``)
        by more than ~6 mm: a point of b's centre line or one of its corners, pulled in by 3 mm, inside another block
        shrunk by 3 mm, or another block's centre inside b. Touching blocks pass; a pose on another block's own image
        features (a duplicate, e.g. a lost block's refit sliding onto a foundation) does not. ``[N]``, or with
        ``pairwise`` ``[N, B]`` (which blocks; b's own column False)."""
        others = ~self.ghost.clone()  # (a ghost's stale pose blocks nothing)
        others[b] = False
        out = torch.zeros((len(ph), len(pos)) if pairwise else len(ph), dtype=torch.bool, device=self.dev)
        if not others.any():
            return out
        h = (self.half[b] - 0.003).clamp_min(0.001)
        line = torch.zeros(7, 3, device=self.dev)
        line[:, 0] = torch.linspace(-1.0, 1.0, 7, device=self.dev)
        local = torch.cat([line, self.corner_signs]) * h  # [15, 3]
        po, Ro, ho = pos[others], R[others], (self.half[others] - 0.003).clamp_min(0.001)
        for c0 in range(0, len(ph), 4096):
            pc, Rc = ph[c0:c0 + 4096], Rh[c0:c0 + 4096]
            pts = torch.einsum("nij,qj->nqi", Rc, local) + pc[:, None]  # [n, 15, 3]
            loc = torch.einsum("oji,noqj->noqi", Ro, pts[:, None] - po[None, :, None])  # in each other block's frame
            inside = (loc.abs() < ho[None, :, None]).all(-1).any(-1)  # [n, O]
            loc2 = torch.einsum("nji,noj->noi", Rc, po[None] - pc[:, None])  # the other centres in b's frame
            hit = inside | (loc2.abs() < h).all(-1)
            if pairwise:
                out[c0:c0 + 4096, others] = hit
            else:
                out[c0:c0 + 4096] = hit.any(-1)
        return out

    def _search_table(self, cams, cmaps, emaps, b: int, pos: torch.Tensor, R: torch.Tensor, top: int = 3):
        """Search for a lost block lying on the table (flat or on its side) or flat on top of other blocks near its
        last position (knocked onto a layer), any yaw, within ``table_search_m``, not inside another block; points
        count when one of their faces faces the camera (nothing else is modelled). Returns the ``top`` hypotheses
        ``(pos [k, 3], R [k, 3, 3])``."""
        cfg = self.cfg
        h = self.half[b]
        xs = torch.arange(-cfg.table_search_m, cfg.table_search_m + 1e-9, cfg.table_step_m, device=self.dev)
        yaws = torch.deg2rad(torch.arange(0.0, 180.0, cfg.table_yaw_step_deg, device=self.dev))
        gx, gy, gyaw = torch.meshgrid(xs, xs, yaws, indexing="ij")
        Rz = geo.quat_to_mat(geo.quat_from_yaw(gyaw.flatten()))
        Rx = torch.tensor([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]], device=self.dev)  # on its side: rolled 90 deg about x
        others = ~self.ghost.clone()
        others[b] = False
        near = others & ((pos[:, :2] - pos[b, :2]).norm(dim=-1) < cfg.table_search_m + 0.08)
        levels = []  # flat on the top face of a nearby block (merged within 3 mm)
        if near.any():
            for t in sorted(self._points(pos, R, "corners")[near][..., 2].amax(-1).tolist()):
                if t > 0.01 and (not levels or t - levels[-1] > 0.003):
                    levels.append(t)
        zs = [float(h[2]), float(h[1])] + [t + float(h[2]) for t in levels]
        Rh = torch.cat([Rz, Rz @ Rx] + [Rz] * len(levels))
        z = torch.cat([torch.full((len(Rz),), zz, device=self.dev) for zz in zs])
        ph = torch.stack([pos[b, 0] + gx.flatten(), pos[b, 1] + gy.flatten()], -1).repeat(len(zs), 1)
        ph = torch.cat([ph, z[:, None]], -1)
        if cfg.lean_deg:  # leaning: yaw over the full turn (the lean has a direction), tilt about the short axis
            lyaw = torch.deg2rad(torch.arange(0.0, 360.0, cfg.lean_yaw_step_deg, device=self.dev))
            lx, ly, lyw = torch.meshgrid(xs, xs, lyaw, indexing="ij")
            Rl = geo.quat_to_mat(geo.quat_from_yaw(lyw.flatten()))
            for th in cfg.lean_deg:
                t = math.radians(th)
                Ry = torch.tensor([[math.cos(t), 0.0, math.sin(t)], [0.0, 1.0, 0.0], [-math.sin(t), 0.0, math.cos(t)]], device=self.dev)
                zl = float(h[0]) * math.sin(t) + float(h[2]) * math.cos(t)
                pl = torch.stack([pos[b, 0] + lx.flatten(), pos[b, 1] + ly.flatten(), torch.full_like(lx.flatten(), zl)], -1)
                Rh = torch.cat([Rh, Rl @ Ry])
                ph = torch.cat([ph, pl])
        free = ~self._overlap(b, ph, Rh, pos, R)
        ph, Rh = ph[free], Rh[free]
        if len(ph) == 0:
            return ph, Rh
        signs = torch.cat([self.corner_signs, self.edge_signs])
        local = signs * h
        cmaps, emaps = F.max_pool2d(cmaps[:, None], 5, 1, 2)[:, 0], F.max_pool2d(emaps[:, None], 5, 1, 2)[:, 0]
        score = torch.zeros(len(ph), device=self.dev)
        for c0 in range(0, len(ph), 8192):
            Rc, pc = Rh[c0:c0 + 8192], ph[c0:c0 + 8192]
            xw = torch.einsum("hij,qj->hqi", Rc, local) + pc[:, None]  # [n, Q, 3]
            nw = torch.einsum("hij,qfj->hqfi", Rc, self.face_normals)  # [n, Q, 3 faces, 3]
            for ci, cam in enumerate(cams):
                facing = ((nw * (cam["t"] - xw)[:, :, None]).sum(-1) > 0).any(-1)  # [n, Q]
                uv, zc = self._project(xw, cam)
                ok = facing & (zc > 0.02) & (uv[..., 0] >= 0) & (uv[..., 0] <= cam["w"] - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= cam["h"] - 1)
                grid = torch.stack([(uv[..., 0] + 0.5) / cam["w"] * 2 - 1, (uv[..., 1] + 0.5) / cam["h"] * 2 - 1], -1)
                vc = F.grid_sample(cmaps[ci][None, None], grid[None, :, :N_CORNER], align_corners=False)[0, 0]
                ve = F.grid_sample(emaps[ci][None, None], grid[None, :, N_CORNER:], align_corners=False)[0, 0]
                score[c0:c0 + 8192] += (2.0 * vc * ok[:, :N_CORNER]).sum(-1) + (0.25 * ve * ok[:, N_CORNER:]).sum(-1)
        k = score.topk(min(top, len(score))).indices
        return ph[k], Rh[k]

    # ------------------------------------------------------------------ one call
    @torch.no_grad()
    def __call__(self, images: dict | None, q: np.ndarray, widths: np.ndarray, maps=None, focus=None):
        """Block poses ``pos [B, 3]``, ``quat [B, 4]`` (wxyz) and confidences ``[B]`` (numpy, ``self.block_ids`` order).
        ``maps``: optional (corner, edge[, mask, hidden]) probability maps ``[4, H, W]`` used instead of the network.
        ``focus``: indices of the blocks the policy is acting on (its node's target): never anchored (they may start
        to move at any call: an anchored pushed block lagged the push's first millimetres) and searched wide when
        lost."""
        cfg = self.cfg
        q = np.asarray(q, dtype=np.float64).reshape(2, 7)
        widths = np.asarray(widths, dtype=np.float64).reshape(2)
        maps = tuple(maps) if maps is not None else self.heatmaps(images)
        cmaps, emaps = maps[0].to(self.dev), maps[1].to(self.dev)
        hmaps = maps[3].to(self.dev) if len(maps) > 3 and maps[3] is not None else None
        self._hidden = F.max_pool2d(hmaps[:, None], 3, 1, 1)[:, 0] if hmaps is not None else None
        peaks_c = [peaks(cmaps[c], cfg.corner_thr) for c in range(len(CAMERA_IDS))]
        cams = self._cams(q)
        B = len(self.block_ids)
        pinch = {a: self._pinch(q, a) for a in ARMS}
        prev_pos = self.pos.clone()
        self.R_prev = self.R.clone()
        pos, R = self.pos.clone(), self.R.clone()
        moving = self.vel.norm(dim=-1) > 0.003  # moved > 3 mm last call (pushed, sliding, falling): carry it on
        pos[moving] = pos[moving] + self.vel[moving]
        for a in ARMS:  # held blocks follow their gripper
            b = self.held[a]
            w = widths[ARMS.index(a)]
            if b is not None and (w > cfg.release_width_m or w < cfg.empty_width_m):
                self.held[a] = None
            elif b is not None:
                op, oR = self.offset[b]
                pos[b] = pinch[a][0] + pinch[a][1] @ op
                R[b] = pinch[a][1] @ oR
        active = ~self.frozen  # frozen foundations keep their pose (they still hide what lies behind them)
        prior_scale = torch.ones(B, device=self.dev)
        ref = (pos.clone(), R.clone())
        focus_m = torch.zeros(B, dtype=torch.bool, device=self.dev)
        if focus is not None:
            focus_m[[int(f) for f in focus]] = True
        self.anchored &= ~focus_m
        anch = self.anchored & ~moving
        for b in self.held.values():
            if b is not None:
                anch[b] = False
        if bool(anch.any()):  # blocks at rest: held toward where they came to rest
            ref[0][anch], ref[1][anch] = self.anchor_p[anch], self.anchor_R[anch]
            pos[anch], R[anch] = self.anchor_p[anch], self.anchor_R[anch]
            prior_scale[anch] = (cfg.prior_pos_m / cfg.rest_prior_m) ** 2
        if self.calls == 0:  # nominal layout: search every block; foundations held toward nominal (they move ~2 mm)
            prior_scale[self.fixed] = (cfg.prior_pos_m / cfg.foundation_prior_m) ** 2
            for b in range(B):
                pos[b], R[b], _ = self._search(cams, cmaps, emaps, b, pos, R)
        pos_f, R_f, inl, ratio = self._fit(cams, cmaps, emaps, peaks_c, pos, R, active, prior_scale, ref)
        c_ratio, c_vis = self.corner_stats
        slid = (c_vis >= 3) & (c_ratio < cfg.corner_ratio)
        bad = ((inl < cfg.min_inliers) | (ratio < cfg.good_ratio) | slid) & self._expected_visible(cams, pos, R) & ~self.frozen
        if self.calls > 0 and bool(bad.any()):  # lost blocks: search around the prior, refit
            for b in torch.nonzero(bad)[:, 0].tolist():
                pos[b], R[b], _ = self._search(cams, cmaps, emaps, b, pos, R)
            pos2, R2, inl2, ratio2 = self._fit(cams, cmaps, emaps, peaks_c, pos, R, bad, prior_scale * 0.01, ref)
            c_ratio2 = self.corner_stats[0]
            # (a hidden block: weak evidence everywhere, keep the prior)
            better = bad & (ratio2 + 0.5 * c_ratio2 > ratio + 0.5 * c_ratio) & (ratio2 >= cfg.good_ratio)
            for b in torch.nonzero(better)[:, 0].tolist():
                if bool(self._overlap(b, pos2[b][None], R2[b][None], pos2, R2)[0]):
                    better[b] = False
            pos_f = torch.where(better[:, None], pos2, pos_f)
            R_f = torch.where(better[:, None, None], R2, R_f)
            inl, ratio = torch.where(better, inl2, inl), torch.where(better, ratio2, ratio)
        ok = ((inl >= cfg.min_inliers) & (ratio >= cfg.min_ratio)) | self.frozen
        if self.calls > 0:  # a fit that moved into another block took that block's image features: rejected, unless
            # its evidence is clearly better than that block's (then the other estimate is the wrong one)
            moved = ((pos_f - ref[0]).norm(dim=-1) > 0.005) & ok
            for b in torch.nonzero(moved)[:, 0].tolist():
                hit = self._overlap(b, pos_f[b][None], R_f[b][None], pos_f, R_f, pairwise=True)[0]
                if bool((hit & (ratio > ratio[b] - 0.15)).any()):
                    ok[b], bad[b] = False, True
        held_ids = [b for b in self.held.values() if b is not None]
        lost = bad & ~ok & ~self.fixed
        lost[held_ids] = False
        self.fail_n = torch.where(ok, torch.zeros_like(self.fail_n), self.fail_n + 1)
        # touched: a gripper near its last pose, or a block that moved last call near it
        near = torch.zeros(B, dtype=torch.bool, device=self.dev)
        for a in ARMS:
            near |= (prev_pos - pinch[a][0]).norm(dim=-1) < cfg.touch_m
        if bool(moving.any()):
            near |= (torch.cdist(prev_pos, prev_pos[moving]) < cfg.touch_block_m).any(-1)
        self.touch_n = torch.where(near, torch.full_like(self.touch_n, cfg.touch_calls), (self.touch_n - 1).clamp_min(0))
        wide = lost & (moving | (self.touch_n > 0) | self.ghost | focus_m)
        if self.calls > 0 and bool(wide.any()):  # fell or was knocked off: look for it on the table, a layer, leaning
            due = [b for b in torch.nonzero(wide)[:, 0].tolist() if self.table_next.get(b, 0) <= self.calls]
            for b in due[: cfg.table_max_blocks]:
                for cp, cR in zip(*self._search_table(cams, cmaps, emaps, b, pos_f, R_f)):
                    p_try, R_try = pos_f.clone(), R_f.clone()
                    p_try[b], R_try[b] = cp, cR
                    only = torch.zeros(B, dtype=torch.bool, device=self.dev)
                    only[b] = True
                    p3, R3, inl3, ratio3 = self._fit(cams, cmaps, emaps, peaks_c, p_try, R_try, only, prior_scale * 0.01)
                    if (float(ratio3[b]) >= cfg.good_ratio and float(inl3[b]) >= cfg.good_inliers and float(ratio3[b]) > float(ratio[b])
                            and not bool(self._overlap(b, p3[b][None], R3[b][None], p3, R3)[0])):
                        pos_f[b], R_f[b], inl[b], ratio[b] = p3[b], R3[b], inl3[b], ratio3[b]
                        ok[b] = True
                if not bool(ok[b]):  # not found (still moving, off the table, on debris): back off
                    n = self.table_fail.get(b, 0) + 1
                    self.table_fail[b] = n
                    self.table_next[b] = self.calls + min(2 ** n, cfg.table_backoff_max)
                    if n >= cfg.ghost_after:
                        self.ghost[b] = True
        for b in torch.nonzero(ok)[:, 0].tolist():
            self.table_fail.pop(b, None)
            self.table_next.pop(b, None)
            self.ghost[b] = False
        if self.calls > 0:  # resting blocks: low-pass small changes (noise), take large ones (moved)
            static = ok.clone() & ~moving
            static[held_ids] = False
            dp = (pos_f - ref[0]).norm(dim=-1)
            rv = geo.rotvec_from_quat(geo.mat_to_quat(R_f @ ref[1].transpose(-1, -2)))
            near = static & (dp < cfg.static_gate_m) & (rv.norm(dim=-1) < math.radians(cfg.static_gate_deg))
            g = cfg.static_gain
            pos_f = torch.where(near[:, None], ref[0] + g * (pos_f - ref[0]), pos_f)
            R_f = torch.where(near[:, None, None], _rodrigues(g * rv) @ ref[1], R_f)
        self.pos = torch.where(ok[:, None], pos_f, ref[0])
        self.R = torch.where(ok[:, None, None], R_f, ref[1])
        good = ((inl >= cfg.good_inliers) & (ratio >= cfg.good_ratio)) | self.frozen
        self.conf = torch.where(good, torch.ones_like(inl), torch.where(ok, torch.full_like(inl, 0.6), torch.full_like(inl, 0.3)))
        self.conf[self.ghost] = 0.0
        self.stats = {"inliers": inl, "ratio": ratio}
        for a in ARMS:  # grasp: closed fingers around a block centre near the pinch; refresh held offsets
            i = ARMS.index(a)
            pp, pR = pinch[a]
            b = self.held[a]
            if b is None and cfg.empty_width_m <= widths[i] < cfg.hold_width_m:
                d = (self.pos - pp).norm(dim=-1)
                d[self.fixed] = float("inf")
                j = int(d.argmin())
                if float(d[j]) < cfg.hold_dist_m:
                    self.held[a] = j
                    self.hold_age[a] = -1  # (counted up below: the next settle_calls fits are taken whole)
                    self.offset[j] = (pR.T @ (self.pos[j] - pp), pR.T @ self.R[j])
            elif b is not None and bool(ok[b]):
                op, oR = self.offset[b]
                op_new = pR.T @ (self.pos[b] - pp)
                rv = geo.rotvec_from_quat(geo.mat_to_quat(pR.T @ self.R[b] @ oR.T))
                slip = float((op_new - op).norm()) > 0.001 or float(rv.norm()) > math.radians(0.5)
                g = 1.0 if self.hold_age[a] < cfg.settle_calls else (cfg.offset_gain if slip else cfg.offset_gain_small)
                self.offset[b] = (op + g * (op_new - op), _rodrigues(g * rv) @ oR)
            if self.held[a] is not None:
                self.hold_age[a] += 1
            b = self.held[a]
            if b is not None:  # report the held block as the gripper's pose times the filtered offset
                op, oR = self.offset[b]
                self.pos[b] = pp + pR @ op
                self.R[b] = pR @ oR
        if cfg.freeze_after > 0:  # foundations: average the first ok fits, then freeze
            learn = self.fixed & ~self.frozen & ok & (ratio >= cfg.good_ratio)
            if bool(learn.any()):
                yaw2 = 2 * torch.atan2(self.R[:, 1, 0], self.R[:, 0, 0])  # (a half turn maps the box onto itself)
                self.f_sum[learn] += torch.stack([self.pos[:, 0], self.pos[:, 1], torch.cos(yaw2), torch.sin(yaw2)], -1)[learn]
                self.f_n[learn] += 1
                done = learn & (self.f_n >= cfg.freeze_after)
                if bool(done.any()):
                    m = self.f_sum[done] / self.f_n[done][:, None]
                    self.pos[done] = torch.stack([m[:, 0], m[:, 1], self.z0[done]], -1)
                    self.R[done] = geo.quat_to_mat(geo.quat_from_yaw(0.5 * torch.atan2(m[:, 3], m[:, 2])))
                    self.frozen |= done
        # report the representative nearest the previous output (skills observe raw quaternions and were trained on the
        # simulator's continuous ones; searches and the foundation freeze can return a half-turned twin)
        Rc = self.R[:, None] @ self.sym[None]  # [B, 4, 3, 3]
        k = torch.einsum("bkij,bij->bk", Rc, self.R_prev).argmax(1)
        self.R = Rc[torch.arange(B, device=self.dev), k]
        # blocks at rest: count unmoved calls, anchor after rest_calls; any move, hold or loss ends it
        d_p = (self.pos - prev_pos).norm(dim=-1)
        d_r = geo.rotvec_from_quat(geo.mat_to_quat(self.R @ self.R_prev.transpose(-1, -2))).norm(dim=-1)
        still = ok & ~self.fixed & (d_p < 0.0005) & (d_r < math.radians(0.3))
        for b in self.held.values():
            if b is not None:
                still[b] = False
        self.rest_n = torch.where(still, self.rest_n + 1, torch.zeros_like(self.rest_n))
        new = still & ~self.anchored & ~focus_m & (self.rest_n >= cfg.rest_calls)
        self.anchor_p[new], self.anchor_R[new] = self.pos[new], self.R[new]
        self.anchored = (self.anchored & still) | new
        # motion of free movable blocks this call (the next call's prediction); held blocks follow their gripper
        step = torch.where((ok & ~self.fixed)[:, None], self.pos - prev_pos, torch.zeros_like(prev_pos))
        coasting = ~ok & moving & (self.coast < cfg.coast_calls)  # lost while moving: keep going (it was predicted)
        step = torch.where(coasting[:, None], 0.8 * self.vel, step)
        self.coast = torch.where(coasting, self.coast + 1, torch.zeros_like(self.coast))
        step[[b for b in self.held.values() if b is not None]] = 0.0
        n = step.norm(dim=-1, keepdim=True)
        self.vel = torch.where(n > cfg.max_step_m, step * cfg.max_step_m / n.clamp_min(1e-9), step)
        self.calls += 1
        quat = geo.mat_to_quat(self.R)
        return self.pos.cpu().numpy().astype(np.float64), quat.cpu().numpy().astype(np.float64), self.conf.cpu().numpy().astype(np.float64)

    def _expected_visible(self, cams, pos, R) -> torch.Tensor:
        vis = self._visible(cams, pos, R, torch.ones(len(pos), dtype=torch.bool, device=self.dev))
        return torch.stack([v[:, :N_CORNER].sum(-1) + 0.25 * v[:, N_CORNER:].sum(-1) for v in vis]).sum(0) >= 8
