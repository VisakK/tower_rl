"""Grasp and release geometry of the demonstrations: where the pinch point sits relative to the block (Step 1/2.3).

For every operation, at the frame before the block starts to rise (gripper closed, block still supported) the measured
pinch pose (torch FK of ``observations/qpos``) is expressed in the block frame; at the release the block pose is
compared with its goal. These numbers define the skills' grasp target and the transit's pre-grasp pose.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from tower_rl import geometry as geo
from tower_rl.data.episode import ARM_INDEX, Episode
from tower_rl.data.events import first_lift_frame, operations
from tower_rl.kinematics import ArmBase, ArmKinematics

ARM_Q = {"left": slice(0, 7), "back": slice(8, 15)}


def _pinch_world(kin: ArmKinematics, base: ArmBase, q7: np.ndarray):
    p, q = kin.fk(torch.tensor(q7[None], dtype=torch.float64), "pinch")
    return base.to_world(p, q)


def operation_geometry(ep: Episode, kin: ArmKinematics | None = None) -> list[dict]:
    kin = kin or ArmKinematics(dtype=torch.float64)
    bases = {a: ArmBase(a, dtype=torch.float64) for a in ("left", "back")}
    goals = {m["block_id"]: m for m in ep.target_mapping}
    out = []
    for op in operations(ep):
        f = max(op.close_frame, first_lift_frame(ep, op) - 1)
        bp = torch.tensor(ep.block_poses[f, op.block_index], dtype=torch.float64)
        b_pos, b_quat = geo.pose7_to_pq(bp)
        rec = {"episode": ep.name, "arm": op.arm, "block": op.block_id, "close_frame": op.close_frame, "grasp_frame": f,
               "open_frame": op.open_frame, "lift_m": op.lift_m, "held_width_m": op.held_width_m}
        for which, q in (("meas", ep.qpos_obs), ("cmd", ep.qpos_cmd)):
            pw, qw = _pinch_world(kin, bases[op.arm], q[f, ARM_Q[op.arm]])
            p_rel, q_rel = geo.frame_relative(b_pos[None], b_quat[None], pw, qw)
            r = geo.quat_to_mat(q_rel)[0]
            rec[f"pinch_in_block_{which}"] = p_rel[0].numpy().round(5).tolist()
            # approach (pinch z) and jaw (pinch y) axes in the block frame
            rec[f"approach_in_block_{which}"] = r[:, 2].numpy().round(4).tolist()
            rec[f"jaw_in_block_{which}"] = r[:, 1].numpy().round(4).tolist()
            rec[f"approach_tilt_deg_{which}"] = round(math.degrees(math.acos(max(-1.0, min(1.0, -float(qw_z(qw)))))), 3)
        # command vs measured pinch (gravity sag + tracking) at the grasp frame
        pm, _ = _pinch_world(kin, bases[op.arm], ep.qpos_obs[f, ARM_Q[op.arm]])
        pc, _ = _pinch_world(kin, bases[op.arm], ep.qpos_cmd[f, ARM_Q[op.arm]])
        rec["cmd_minus_meas_pinch_mm"] = ((pc - pm)[0].numpy() * 1000).round(2).tolist()
        g = goals.get(op.block_id)
        if g is not None and op.open_frame >= 0:
            fe = min(op.open_frame + 60, ep.num_frames - 1)  # 2 s after the gripper opened
            bpe = ep.block_poses[fe, op.block_index]
            gp, gq = geo.pose7_to_pq(torch.tensor(g["goal"], dtype=torch.float64))
            ep_, eq = geo.pose7_to_pq(torch.tensor(bpe, dtype=torch.float64))
            rec["place_err_mm"] = ((ep_ - gp).numpy() * 1000).round(2).tolist()
            rec["place_yaw_err_deg"] = round(math.degrees(float(geo.yaw_diff_sym(geo.yaw_from_quat(eq), geo.yaw_from_quat(gq)))), 3)
            rec["planned_arm_matches"] = ARM_INDEX[op.arm] == g["arm"]
        out.append(rec)
    return out


def qw_z(qw: torch.Tensor) -> float:
    """World z component of the pinch approach axis (-1 = pointing straight down)."""
    return float(geo.quat_to_mat(qw)[0, 2, 2])
