"""Step 2 smoke tests: open the light skill scene or the station scene headless, step, and check the basics.

skill   : one arm + a source tower of ``--variant`` at ``--stage`` + both foundations; the arm holds the pre-grasp pose of
          the stage's target block through the shared 10 Hz controller (IK + sag compensation). Reports NaNs, the arm
          base pose vs the station, PhysX vs torch FK, the pinch hold error and tower drift, and physics throughput.
station : the full station with both arms and an episode's blocks at frame 0, holding the recorded pose.

Usage:
  python scripts/smoke_scenes.py --headless --scene skill --arm left --num_envs 64 --out results/scenes/smoke_skill_left.json
  python scripts/smoke_scenes.py --headless --scene station --num_envs 2 --out results/scenes/smoke_station.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--scene", choices=["skill", "station"], default="skill")
parser.add_argument("--arm", choices=["left", "back"], default="left")
parser.add_argument("--variant", default=None, help="default: task1-3 for left, task2-3 for back")
parser.add_argument("--stage", type=int, default=0, help="transfer index whose block is the target")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--ticks", type=int, default=300, help="30 Hz control ticks to simulate")
parser.add_argument("--episode", default="t23_ep1", help="station scene: episode providing blocks and arm pose")
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from tower_rl import geometry as geo  # noqa: E402
from tower_rl.config import load_yaml  # noqa: E402
from tower_rl.kinematics import ArmBase, ArmKinematics  # noqa: E402
from tower_rl.sim.assets import q_full, sim_cfg  # noqa: E402

phys = load_yaml("physics")
SUB = phys["control_substeps"]
report = {"scene": args.scene, "num_envs": args.num_envs, "ticks": args.ticks}


def finite(t):
    return bool(torch.isfinite(t).all())


if args.scene == "skill":
    from tower_rl.control import ArmController, GripperController
    from tower_rl.domain import DomainSampler
    from tower_rl.grasp import above, cfg as gcfg, grasp_pose
    from tower_rl.postures import prior_for_pose
    from tower_rl.sim.skill_scene import FOUNDATION_SLOTS, MOVABLE_SLOTS, make_skill_scene_cfg, park_pose
    from tower_rl.tower import foundation_poses, load_spec, present_after, rest_heights, source_poses

    variant = args.variant or ("task1-3_tower_transfer_3" if args.arm == "left" else "task2-3_cross_tower_transfer_3")
    sim = sim_utils.SimulationContext(sim_cfg(phys, device=args.device))
    t0 = time.time()
    scene = InteractiveScene(make_skill_scene_cfg(args.arm, args.num_envs))
    sim.reset()
    report["build_s"] = round(time.time() - t0, 1)
    dev = sim.device
    N = args.num_envs
    robot, blocks = scene["robot"], scene["blocks"]
    report["robot_joint_names"] = robot.joint_names
    report["block_slots"] = blocks.object_names
    # --- tower
    spec = load_spec(variant, device=dev)
    rz = rest_heights(dev)
    dom = DomainSampler("D0", device=dev).sample(N, torch.Generator(device=dev).manual_seed(0))
    stage = torch.full((N,), args.stage, device=dev)
    present = present_after(spec, stage)
    src = source_poses(spec, dom, rz)
    found = foundation_poses(spec, N)
    poses = torch.zeros(N, len(FOUNDATION_SLOTS) + len(MOVABLE_SLOTS), 7, device=dev)
    poses[:, :6] = found
    park = torch.tensor([park_pose(i) for i in range(len(MOVABLE_SLOTS))], device=dev)
    poses[:, 6:] = park
    m = spec.num_movable
    poses[:, 6:6 + m] = torch.where(present[..., None], src, park[:m])
    names = blocks.object_names
    order = [names.index(s) for s in FOUNDATION_SLOTS + MOVABLE_SLOTS]
    world = poses.clone()
    world[..., :3] += scene.env_origins[:, None]
    state = torch.zeros(N, len(names), 13, device=dev)
    state[:, order, :7] = world
    blocks.write_object_state_to_sim(state)
    # --- arm at the pre-grasp pose of the stage's target
    k = int(spec.transfer_block[args.stage])
    assert spec.transfer_arm[args.stage] == args.arm, f"transfer {args.stage} of {variant} uses the {spec.transfer_arm[args.stage]} arm"
    tb = src[:, k]
    g = gcfg()
    gp, gq = grasp_pose(tb[:, :3], tb[:, 3:7], args.arm)
    pp, pq = above(gp, gq, g["pregrasp_height_m"])
    kin = ArmKinematics(device=dev)
    kin.set_arm(args.arm)
    base = ArmBase(args.arm, device=dev)
    seed = prior_for_pose(args.arm, "grasp", tb[:, :3], tb[:, 3:7])
    pb, qb = base.to_base(pp, pq)
    q_des, pe, re = kin.ik(seed, pb, qb, iters=100, q_ref=seed)
    report["ik_pregrasp_err_mm_max"] = round(float(pe.max()) * 1000, 3)
    report["ik_pregrasp_err_deg_max"] = round(math.degrees(float(re.max())), 3)
    width = torch.full((N, 1), 0.1, device=dev)
    robot.write_joint_state_to_sim(q_full(q_des, width), torch.zeros(N, 10, device=dev))
    ctrl = ArmController(args.arm, N, device=dev)
    ctrl.reset(q_des)
    ctrl.set_target(pp, pq)
    grip = GripperController(N, device=dev)
    q_cmd = ctrl.command()
    robot.set_joint_position_target(q_full(q_cmd, grip.step(torch.zeros(N, dtype=torch.bool, device=dev))[:, None]))
    scene.write_data_to_sim()
    sim.forward()
    scene.update(phys["dt"])
    block_p0 = blocks.data.object_link_pos_w.clone()
    report["robot_root_pos_env0_local"] = (robot.data.root_link_pos_w[0] - scene.env_origins[0]).tolist()
    report["robot_root_quat_env0"] = robot.data.root_link_quat_w[0].tolist()
    nan_free = True
    t1 = time.time()
    for tick in range(args.ticks):
        if tick % phys["policy_ticks"] == 0:
            q_cmd = ctrl.command(robot.data.joint_pos[:, :7])
            w = grip.step(torch.zeros(N, dtype=torch.bool, device=dev))
            robot.set_joint_position_target(q_full(q_cmd, w[:, None]))
        for _ in range(SUB):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(phys["dt"])
        nan_free &= finite(robot.data.joint_pos) and finite(blocks.data.object_link_pos_w)
    torch.cuda.synchronize()
    wall = time.time() - t1
    q_meas = robot.data.joint_pos[:, :7]
    # PhysX vs torch FK (pinch/link7 in world)
    il = robot.body_names.index("link7")
    p_px = robot.data.body_link_pos_w[:, il] - scene.env_origins
    q_px = robot.data.body_link_quat_w[:, il]
    p_fk, q_fk = base.to_world(*kin.fk(q_meas, "link7"))
    pos_err, rot_err = ctrl.pinch_error(q_meas)
    disp = (blocks.data.object_link_pos_w - block_p0).norm(dim=-1)
    mov_idx = [names.index(s) for s in MOVABLE_SLOTS]
    report.update({
        "variant": variant, "stage": args.stage, "target_block": spec.movable_ids[k],
        "nan_free": nan_free,
        "physics_steps_per_s": round(args.ticks * SUB / wall, 1),
        "env_physics_steps_per_s": round(args.ticks * SUB * N / wall, 1),
        "fk_link7_err_mm_max": round(float((p_px - p_fk).norm(dim=-1).max()) * 1000, 4),
        "fk_link7_err_deg_max": round(math.degrees(float(geo.quat_angle(q_px, q_fk).max())), 4),
        "hold_pinch_err_mm": [round(float(pos_err.mean()) * 1000, 2), round(float(pos_err.max()) * 1000, 2)],
        "hold_pinch_err_deg": [round(math.degrees(float(rot_err.mean())), 3), round(math.degrees(float(rot_err.max())), 3)],
        "sag_cmd_minus_des_rad_max": round(float((q_cmd - ctrl.q_des).abs().max()), 4),
        "tower_drift_mm_max": round(float(disp[:, mov_idx].max()) * 1000, 3),
        "tower_drift_mm_mean": round(float(disp[:, mov_idx].mean()) * 1000, 3),
        "foundation_drift_mm_max": round(float(disp[:, [names.index(s) for s in FOUNDATION_SLOTS]].max()) * 1000, 4),
        "gpu_mem_alloc_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    })
else:
    from tower_rl.paths import STATION_JSON, resolve_episode
    from tower_rl.scene import CONTROL_SUBSTEPS, PHYSICS_DT, Episode, apply_friction, load_station_meta, make_scene_cfg, set_arm_targets, write_recorded_state

    meta = load_station_meta(STATION_JSON)
    episode = Episode.load(*resolve_episode(args.episode))
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, render_interval=CONTROL_SUBSTEPS, device=args.device,
                                                              physx=sim_utils.PhysxCfg()))
    t0 = time.time()
    scene = InteractiveScene(make_scene_cfg(meta, episode, args.num_envs))
    sim.reset()
    report["build_s"] = round(time.time() - t0, 1)
    apply_friction(scene, finger=(1.2, 1.0))
    write_recorded_state(scene, episode, 0)
    q0 = torch.tensor(episode.qpos_cmd[0], dtype=torch.float32, device=sim.device)
    p0 = scene["blocks"].data.object_link_pos_w.clone()
    nan_free = True
    t1 = time.time()
    for tick in range(args.ticks):
        set_arm_targets(scene, q0)
        for _ in range(CONTROL_SUBSTEPS):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(PHYSICS_DT)
        nan_free &= finite(scene["left"].data.joint_pos) and finite(scene["back"].data.joint_pos) and finite(scene["blocks"].data.object_link_pos_w)
    wall = time.time() - t1
    disp = (scene["blocks"].data.object_link_pos_w - p0).norm(dim=-1)
    report.update({"episode": args.episode, "nan_free": nan_free, "physics_steps_per_s": round(args.ticks * CONTROL_SUBSTEPS / wall, 1),
                   "block_drift_mm_max": round(float(disp.max()) * 1000, 3)})

report["ok"] = bool(report["nan_free"])
print("REPORT_JSON " + json.dumps(report), flush=True)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(report, indent=1))
os._exit(0)
