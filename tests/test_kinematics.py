"""Torch Nero kinematics against the recorded dataset, finite differences and the PhysX statics probe."""

import json
import math

import numpy as np
import pytest
import torch

from tower_rl import geometry as geo
from tower_rl.data.episode import Episode
from tower_rl.kinematics import ArmBase, ArmKinematics
from tower_rl.paths import EPISODE_DIR, RESULTS_DIR

EPISODES = sorted(EPISODE_DIR.glob("*.npz"))
ARM_SLICES = {"left": (slice(0, 7), slice(0, 7)), "back": (slice(8, 15), slice(7, 14))}  # qpos, eef columns


@pytest.fixture(scope="module")
def kin():
    return ArmKinematics(dtype=torch.float64)


@pytest.mark.skipif(not EPISODES, reason="no episodes in data/episodes")
@pytest.mark.parametrize("npz", EPISODES, ids=lambda p: p.stem)
def test_fk_matches_recorded_eef(kin, npz):
    ep = Episode.load(npz)
    q = torch.tensor(ep.qpos_obs, dtype=torch.float64)
    eef = torch.tensor(ep.eef_obs, dtype=torch.float64)
    for arm, (sq, se) in ARM_SLICES.items():
        p, quat = kin.fk(q[:, sq], "link7")
        pe, qe = geo.pose7_to_pq(eef[:, se])
        assert float((p - pe).norm(dim=-1).max()) < 5e-5, arm  # 0.05 mm (float32 storage)
        assert float(geo.quat_angle(quat, geo.normalize(qe)).max()) < math.radians(0.01), arm


@pytest.mark.skipif(not EPISODES, reason="no episodes in data/episodes")
def test_pinch_matches_generator_measurement(kin):
    n = 0
    for npz in EPISODES:
        ep = Episode.load(npz)
        arm_of = {m["block_id"]: ("left", "back")[m["arm"]] for m in ep.target_mapping}
        for e in ep.events:
            if e["event"] != "precontact_pinch_measurement":
                continue
            f = int(np.argmin(np.abs(ep.time - e["time"])))
            arm = arm_of[e["block_id"]]
            p, quat = kin.fk(torch.tensor(ep.qpos_obs[f:f + 1, ARM_SLICES[arm][0]], dtype=torch.float64), "pinch")
            pw, _ = ArmBase(arm, dtype=torch.float64).to_world(p, quat)
            assert np.linalg.norm(pw[0].numpy() - np.array(e["actual_pinch_center_world_m"])) < 5e-5
            n += 1
    if n == 0:
        pytest.skip("no precontact_pinch_measurement events")


def test_jacobian_matches_finite_differences(kin):
    torch.manual_seed(0)
    q = (torch.rand(8, 7, dtype=torch.float64) - 0.5) * 2.0
    jac = kin.jacobian(q, "pinch")
    eps = 1e-6
    p0, q0 = kin.fk(q, "pinch")
    for i in range(7):
        dq = torch.zeros_like(q)
        dq[:, i] = eps
        p1, q1 = kin.fk(q + dq, "pinch")
        dp = (p1 - p0) / eps
        dw = geo.quat_error_rotvec(q0, q1) / eps
        assert torch.allclose(jac[:, :3, i], dp, atol=1e-5)
        assert torch.allclose(jac[:, 3:, i], dw, atol=1e-5)


def test_ik_roundtrip(kin):
    torch.manual_seed(1)
    lim = kin.limits
    q_true = lim[:, 0] * 0.6 + (lim[:, 1] - lim[:, 0]) * 0.6 * torch.rand(64, 7, dtype=torch.float64)
    p, quat = kin.fk(q_true, "pinch")
    q_seed = q_true + 0.15 * (torch.rand_like(q_true) - 0.5)
    q_sol, pe, re = kin.ik(q_seed, p, quat, iters=100, tol_pos=1e-6, tol_rot=1e-5)
    assert float(pe.max()) < 5e-4 and float(re.max()) < math.radians(0.5)
    assert bool(((q_sol >= lim[:, 0] - 1e-9) & (q_sol <= lim[:, 1] + 1e-9)).all())


def test_gravity_model_matches_physx_probe():
    probes = sorted((RESULTS_DIR / "statics").glob("probe*.json"))
    if not probes:
        pytest.skip("run scripts/probe_arm_statics.py first")
    rep = json.loads(probes[-1].read_text())
    for arm in ("left", "back"):
        scale = rep[arm]["physx_over_model_scale_per_joint"]  # PhysX reports compensation forces (= -tau_g)
        for j in range(1, 7):
            assert abs(scale[j] + 1.0) < 0.05, (arm, j, scale)


def test_grasp_relation_error_invariances():
    """Zero for the canonical grasp, unchanged when the held block (with the gripper) tilts in the world or the fingers
    swap, and it reports an offset along the block in the block frame."""
    import math

    from tower_rl import geometry as geo
    from tower_rl.grasp import grasp_pose, grasp_relation_error

    bp = torch.tensor([[0.17, -0.52, 0.0441]])
    bq = geo.quat_from_yaw(torch.tensor([0.4]))
    gp, gq = grasp_pose(bp, bq, "left")
    d, rot = grasp_relation_error(bp, bq, gp, gq, "left")
    assert float(d.norm()) < 1e-6 and float(rot) < 1e-3
    # rigid tilt of block + gripper about a horizontal axis through the block (a held block hanging tilted)
    t = geo.quat_from_axis_angle(torch.tensor([[0.3, 1.0, 0.0]]), torch.tensor([math.radians(20.0)]))
    bq2, gq2 = geo.quat_mul(t, bq), geo.quat_mul(t, gq)
    gp2 = bp + geo.quat_apply(t, gp - bp)
    d, rot = grasp_relation_error(bp, bq2, gp2, gq2, "left")
    assert float(d.norm()) < 1e-5 and float(rot) < 2e-3
    # fingers swapped (pi about the approach axis): same grasp
    flip = geo.quat_from_axis_angle(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([math.pi]))
    _, rot = grasp_relation_error(bp, bq, gp, geo.quat_mul(gq, flip), "left")
    assert float(rot) < 1e-3
    # 15 mm along the block
    d, _ = grasp_relation_error(bp, bq, gp + geo.quat_apply(bq, torch.tensor([[0.015, 0.0, 0.0]])), gq, "left")
    assert abs(float(d[0, 0]) - 0.015) < 1e-6 and abs(float(d[0, 2])) < 1e-6
