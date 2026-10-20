"""Every nominal via-point of every planned transfer is reachable with the canonical gripper orientation.

Seeds come from the demo posture priors; this is the invariant the transit controller and the skill reset
distributions rely on.
"""

import math

import pytest
import torch

from tower_rl import geometry as geo
from tower_rl.data.layouts import VARIANTS, Layout
from tower_rl.grasp import above, cfg, grasp_pose, place_pose
from tower_rl.kinematics import ArmBase, ArmKinematics
from tower_rl.postures import prior_for_pose
from tower_rl.tower import rest_heights


def _solve(kin, base, arm, phase, pos, quat, block_quat):
    seed = prior_for_pose(arm, phase, pos, block_quat).double()
    pb, qb = base.to_base(pos.double(), quat.double())
    q, pe, re = kin.ik(seed, pb, qb, iters=200, q_ref=seed)
    return q, float(pe.max()), math.degrees(float(re.max()))


@pytest.mark.parametrize("variant", VARIANTS)
def test_transfer_viapoints_reachable(variant):
    lay = Layout.load(variant)
    kin = ArmKinematics(dtype=torch.float64)
    rz = rest_heights()
    g = cfg()
    for t in lay.transfers:
        base = ArmBase(t.arm, dtype=torch.float64)
        b = torch.tensor(lay.blocks[t.block], dtype=torch.float32)[None]
        bpos, bq = b[:, :3].clone(), b[:, 3:7]
        bpos[:, 2] = rz[lay.layer[t.block]]
        gp, gq = grasp_pose(bpos, bq, t.arm)
        goal = torch.tensor(t.goal, dtype=torch.float32)[None]
        ppos, pq = place_pose(goal[:, :3], goal[:, 3:7], t.arm)
        ppos = ppos + torch.tensor([0.0, 0.0, g["release_clearance_m"]])
        points = {
            "grasp": ("grasp", gp, gq, bq),
            "pregrasp": ("grasp", *above(gp, gq, g["pregrasp_height_m"]), bq),
            "postlift": ("grasp", *above(gp, gq, g["post_lift_height_m"]), bq),
            "release": ("release", ppos, pq, goal[:, 3:7]),
            "preplace": ("release", *above(ppos, pq, g["preplace_height_m"]), goal[:, 3:7]),
        }
        for name, (phase, p, q, bqq) in points.items():
            _, pe, re = _solve(kin, base, t.arm, phase, p, q, bqq)
            assert pe < 1e-3 and re < 0.5, (variant, t.block, t.arm, name, pe, re)
