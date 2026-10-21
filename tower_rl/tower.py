"""Tower geometry: layout configs -> sampled block poses for a task stage and a domain sample (pure torch).

Every movable source block has a nominal position in its pad frame (pad centre + foundation yaw) and a nominal world
yaw. A sample composes, in the source pad frame: per-block jitter, the per-layer irregularity (shift + yaw about the
pad centre) and the assembly shift (shift + yaw about the pad centre); heights are the measured resting heights of each
layer. The destination side places already transferred blocks at their goals plus a placement error.
Poses are ``[x, y, z, qw, qx, qy, qz]`` in the station frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from tower_rl import geometry as geo
from tower_rl.config import load_yaml
from tower_rl.data.layouts import Layout, yaw_of_wxyz
from tower_rl.domain import DomainSample

BLOCK_HALF = (0.075, 0.025, 0.015)


def _rot2(yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


@dataclass
class TowerSpec:
    variant: str
    src_center: torch.Tensor   # [2]
    src_yaw: float
    dst_center: torch.Tensor   # [2]
    dst_yaw: float
    foundation_ids: list[str]  # 3 source then 3 destination
    foundation: torch.Tensor   # [6, 7] nominal world poses
    movable_ids: list[str]     # source movable blocks (layout order: layer 1 first)
    layer: torch.Tensor        # [M] long, 1-based
    rel_xy: torch.Tensor       # [M, 2] nominal position in the source pad frame
    yaw: torch.Tensor          # [M] nominal world yaw
    transfer_block: torch.Tensor  # [K] index into movable_ids, in plan order
    transfer_arm: list[str]
    transfer_dest_layer: torch.Tensor  # [K]
    goal: torch.Tensor         # [K, 7] world goal poses
    goal_rel_xy: torch.Tensor  # [K, 2] goal position in the destination pad frame
    # Task 4/5 (dataset v1.1.0). Tasks 1-3 keep the defaults and their exact previous behaviour.
    family: str = "transfer"                   # transfer | pushdown | insertion
    nominal: torch.Tensor | None = None        # [M, 7] layout poses of the movable blocks (heights for Task 4/5)
    perturb: torch.Tensor | None = None        # [M] bool: the robustness cases move this block (source side only)
    push_block: torch.Tensor | None = None     # [P] index into movable_ids, push order (Task 4)
    push_arm: list[str] | None = None
    push_dir: torch.Tensor | None = None       # [P, 3] world push direction (unit, horizontal)

    @property
    def num_movable(self) -> int:
        return len(self.movable_ids)

    @property
    def num_layers(self) -> int:
        return int(self.layer.max())

    def to(self, device) -> TowerSpec:
        kw = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in self.__dict__.items()}
        return TowerSpec(**kw)


def load_spec(variant: str, device: str = "cpu") -> TowerSpec:
    lay = Layout.load(variant)
    sp, dp = lay.pads["source"], lay.pads["destination"]
    src_c = torch.tensor(sp["center"][:2], dtype=torch.float32)
    dst_c = torch.tensor(dp["center"][:2], dtype=torch.float32)
    src_yaw, dst_yaw = math.radians(sp["foundation_yaw_deg"]), math.radians(dp["foundation_yaw_deg"])
    f_ids = list(sp["foundation_blocks"]) + list(dp["foundation_blocks"])
    foundation = torch.tensor(np.stack([lay.blocks[b] for b in f_ids]), dtype=torch.float32)
    mov = sorted(lay.movable_ids, key=lambda b: (lay.layer[b], b))
    pos = torch.tensor(np.stack([lay.blocks[b][:2] for b in mov]), dtype=torch.float32)
    rel = ((pos - src_c) @ _rot2(torch.tensor(src_yaw))).float()  # world -> pad frame: R^T (p - c)
    yaw = torch.tensor([yaw_of_wxyz(lay.blocks[b][3:7]) for b in mov], dtype=torch.float32)
    goal = torch.tensor(np.stack([t.goal for t in lay.transfers]), dtype=torch.float32) if lay.transfers else torch.zeros(0, 7)
    goal_rel = (goal[:, :2] - dst_c) @ _rot2(torch.tensor(dst_yaw))
    extra = {}
    if lay.family != "transfer":
        extra = dict(
            family=lay.family,
            nominal=torch.tensor(np.stack([lay.blocks[b] for b in mov]), dtype=torch.float32),
            # docs "Task prompts": "Task5 pose cases vary only the source blocks" (the inserts); Task 4's whole tower
            perturb=torch.tensor([lay.side[b] == "source" for b in mov]),
            push_block=torch.tensor([mov.index(p.block) for p in lay.pushes], dtype=torch.long),
            push_arm=[p.arm for p in lay.pushes],
            push_dir=torch.tensor(np.stack([p.direction for p in lay.pushes]), dtype=torch.float32) if lay.pushes else torch.zeros(0, 3),
        )
    spec = TowerSpec(
        variant=variant, src_center=src_c, src_yaw=src_yaw, dst_center=dst_c, dst_yaw=dst_yaw,
        foundation_ids=f_ids, foundation=foundation, movable_ids=mov,
        layer=torch.tensor([lay.layer[b] for b in mov]), rel_xy=rel, yaw=yaw,
        transfer_block=torch.tensor([mov.index(t.block) for t in lay.transfers], dtype=torch.long),
        transfer_arm=[t.arm for t in lay.transfers],
        transfer_dest_layer=torch.tensor([t.dest_layer for t in lay.transfers], dtype=torch.long),
        goal=goal, goal_rel_xy=goal_rel.float(), **extra,
    )
    return spec.to(device)


def rest_heights(device="cpu") -> torch.Tensor:
    """Measured resting centre height per layer index (0 = foundation)."""
    return torch.tensor(load_yaml("physics")["block"]["rest_z"], dtype=torch.float32, device=device)


def pose_from_xy_yaw(xy: torch.Tensor, z: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    return torch.cat([xy, z[..., None], geo.quat_from_yaw(yaw)], dim=-1)


def source_poses(spec: TowerSpec, dom: DomainSample, rest_z: torch.Tensor) -> torch.Tensor:
    """World poses ``[N, M, 7]`` of all source movable blocks under the domain sample (present or not)."""
    n = dom.assembly_xy.shape[0]
    m = spec.num_movable
    li = (spec.layer - 1).clamp(max=dom.layer_xy.shape[1] - 1)  # [M]
    # per-block jitter (the Tasks 1-3 generator's centre-block jitter) would make Task 4's packed full layers (three
    # blocks side by side, 0.25 mm apart) interpenetrate; its generator jitters nothing per block, layers move rigidly
    bxy = torch.zeros_like(dom.block_xy[:, :m]) if spec.family == "pushdown" else dom.block_xy[:, :m]
    p = spec.rel_xy[None].expand(n, m, 2) + bxy
    ly = dom.layer_yaw[:, li]            # [N, M]
    p = (_rot2(ly) @ p[..., None]).squeeze(-1) + dom.layer_xy[:, li]
    ay = dom.assembly_yaw[:, None].expand(n, m)
    p = (_rot2(ay) @ p[..., None]).squeeze(-1) + dom.assembly_xy[:, None]
    p = spec.src_center + (_rot2(torch.tensor(spec.src_yaw, device=p.device)) @ p[..., None]).squeeze(-1)
    yaw = spec.yaw[None] + ly + ay
    if spec.family == "transfer":
        z = rest_z[spec.layer][None].expand(n, m)
        return pose_from_xy_yaw(p, z, yaw)
    # Task 4/5: heights from the layout (Task 5's 32 mm side blocks and its roof sit off the standard layer pitch);
    # blocks the robustness cases do not move (Task 5's destination tower) stay at their layout poses
    z = spec.nominal[:, 2][None].expand(n, m)
    out = pose_from_xy_yaw(p, z, yaw)
    keep = ~spec.perturb
    out[:, keep] = spec.nominal[keep][None].expand(n, int(keep.sum()), 7)
    return out


def goal_poses_with_error(spec: TowerSpec, rest_z: torch.Tensor, n: int, xy_err_m: float, yaw_err_deg: float,
                          gen: torch.Generator | None = None) -> torch.Tensor:
    """Poses ``[N, K, 7]`` of blocks already placed at their goals with a uniform placement error."""
    k = spec.goal.shape[0]
    dev = spec.goal.device
    xy = spec.goal[None, :, :2] + (torch.rand(n, k, 2, generator=gen, device=dev) * 2 - 1) * xy_err_m
    yaw = geo.yaw_from_quat(spec.goal[:, 3:7])[None] + (torch.rand(n, k, generator=gen, device=dev) * 2 - 1) * math.radians(yaw_err_deg)
    z = rest_z[spec.transfer_dest_layer][None].expand(n, k)
    return pose_from_xy_yaw(xy, z, yaw)


def foundation_poses(spec: TowerSpec, n: int, jitter_m: float = 0.0, gen: torch.Generator | None = None) -> torch.Tensor:
    """``[N, 6, 7]`` fixed foundation blocks; each pad's three blocks move together by up to ``jitter_m``."""
    f = spec.foundation[None].repeat(n, 1, 1)
    if jitter_m > 0:
        d = (torch.rand(n, 2, 2, generator=gen, device=f.device) * 2 - 1) * jitter_m
        f[:, :3, :2] += d[:, 0, None]
        f[:, 3:, :2] += d[:, 1, None]
    return f


def present_after(spec: TowerSpec, stage: torch.Tensor) -> torch.Tensor:
    """``[N, M]`` mask of source blocks still on the source tower before transfer ``stage`` (0-based)."""
    k = spec.transfer_block.shape[0]
    done = torch.arange(k, device=stage.device)[None] < stage[:, None]  # [N, K] transfers already executed
    removed = torch.zeros(stage.shape[0], spec.num_movable, dtype=torch.bool, device=stage.device)
    removed.scatter_(1, spec.transfer_block[None].expand(stage.shape[0], k), done)
    return ~removed
