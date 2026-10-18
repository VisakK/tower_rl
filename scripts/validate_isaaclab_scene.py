"""Validate the TOWER station + blocks as an Isaac Lab InteractiveScene (meters), with evaluator drive parameters.

Modes:
  inspect      : print parsed articulation/rigid-object properties and hold the initial pose for 5 s
  fk           : validate kinematics against recorded observations/eef (TCP in arm-base frame)
  replay       : open-loop replay of a recorded episode's leader/qpos (30 Hz ZOH, 4 x 120 Hz substeps)
  bench        : physics throughput for --num_envs parallel envs
  render       : render the 4 evaluator cameras at --render_frames (needs --enable_cameras)
  render_bench : 4-camera rendering throughput at --cam_scale resolution (needs --enable_cameras)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

from tower_rl.paths import STATION_JSON, resolve_episode  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--mode", choices=["inspect", "fk", "replay", "bench", "render", "render_bench"], default="inspect")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--station_json", default=str(STATION_JSON), help="output json of make_meter_assets.py")
parser.add_argument("--episode", required=True, help="episode name in data/episodes (e.g. t23_ep1) or npz path")
parser.add_argument("--attrs", default=None, help="attrs json of the episode (default: <npz stem>_attrs.json)")
parser.add_argument("--out", required=True)
parser.add_argument("--block_friction", nargs=2, type=float, default=None, help="override movable+fixed block friction")
parser.add_argument("--finger_friction", nargs=2, type=float, default=None, help="override finger shape friction")
parser.add_argument("--delay_substeps", nargs=2, type=int, default=[0, 0], help="command delay (left, back) in 120 Hz substeps")
parser.add_argument("--bench_steps", type=int, default=600)
parser.add_argument("--render_frames", nargs="+", type=int, default=[0, 1500])
parser.add_argument("--dome_light", type=float, default=0.0, help="add a dome light of this intensity")
parser.add_argument("--render_out", default=None, help="directory for rendered images")
parser.add_argument("--cam_scale", type=float, default=1.0, help="resolution scale of the 640x480 evaluator cameras")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from tower_rl.scene import (  # noqa: E402
    CONTROL_SUBSTEPS,
    EVAL_RENDER_CFG,
    PHYSICS_DT,
    DelayedCommands,
    Episode,
    apply_friction,
    evaluator_camera_cfgs,
    load_station_meta,
    make_scene_cfg,
    measured_q16,
    set_arm_targets,
    write_recorded_state,
)

meta = load_station_meta(args.station_json)
episode = Episode.load(*resolve_episode(args.episode, args.attrs))
ep = episode.arrays
block_ids = episode.block_ids
rendering = args.mode in ("render", "render_bench")

extras = {}
if rendering:
    extras.update(evaluator_camera_cfgs(meta, episode, scale=args.cam_scale))
    if args.dome_light > 0:
        extras["dome"] = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=args.dome_light))

render_cfg = sim_utils.RenderCfg(**EVAL_RENDER_CFG) if rendering else sim_utils.RenderCfg()
sim = sim_utils.SimulationContext(
    sim_utils.SimulationCfg(dt=PHYSICS_DT, render_interval=CONTROL_SUBSTEPS, device=args.device, physx=sim_utils.PhysxCfg(), render=render_cfg)
)
t0 = time.time()
scene = InteractiveScene(make_scene_cfg(meta, episode, args.num_envs, extras=extras))
t_scene = time.time() - t0
sim.reset()
t_reset = time.time() - t0
left, back, blk = scene["left"], scene["back"], scene["blocks"]
dev = sim.device
report = {"mode": args.mode, "num_envs": args.num_envs, "scene_build_s": round(t_scene, 2), "scene_build_plus_reset_s": round(t_reset, 2)}
apply_friction(scene, finger=args.finger_friction, block=args.block_friction)


def step_physics():
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(sim.get_physics_dt())


def tcp_like(r, link):
    """pose of `link` in the robot base_link frame (pos, quat wxyz)"""
    ib = r.body_names.index("base_link")
    il = r.body_names.index(link)
    pb, qb = r.data.body_link_pos_w[:, ib], r.data.body_link_quat_w[:, ib]
    pl, ql = r.data.body_link_pos_w[:, il], r.data.body_link_quat_w[:, il]
    return math_utils.subtract_frame_transforms(pb, qb, pl, ql)


if args.mode == "inspect":
    for name, r in (("left", left), ("back", back)):
        v = r.root_physx_view
        report[name] = {
            "joint_names": r.joint_names, "body_names": r.body_names, "num_bodies": r.num_bodies,
            "fixed_base": r.is_fixed_base,
            "joint_limits": r.data.joint_pos_limits[0].cpu().numpy().round(5).tolist(),
            "stiffness": v.get_dof_stiffnesses()[0].numpy().round(4).tolist(),
            "damping": v.get_dof_dampings()[0].numpy().round(5).tolist(),
            "armature": v.get_dof_armatures()[0].numpy().round(5).tolist(),
            "friction_props": v.get_dof_friction_properties()[0].numpy().round(5).tolist(),
            "max_force": v.get_dof_max_forces()[0].numpy().round(4).tolist(),
            "max_vel": v.get_dof_max_velocities()[0].numpy().round(4).tolist(),
            "masses": dict(zip(r.body_names, v.get_masses()[0].numpy().round(5).tolist())),
            "inertia_diag": dict(zip(r.body_names, v.get_inertias()[0].numpy()[:, [0, 4, 8]].round(7).tolist())),
            "root_pos_env0": (r.data.root_link_pos_w[0] - scene.env_origins[0]).cpu().numpy().round(5).tolist(),
        }
    report["blocks"] = {"names": blk.object_names,
                        "unique_masses": torch.unique(blk.root_physx_view.get_masses()).numpy().round(5).tolist(),
                        "unique_inertia_diag": torch.unique(blk.root_physx_view.get_inertias()[:, [0, 4, 8]], dim=0).numpy().round(8).tolist(),
                        "material(static,dyn,rest)": blk.root_physx_view.get_material_properties()[0].numpy().round(4).tolist()}
    write_recorded_state(scene, episode, 0)
    p0 = (blk.data.object_link_pos_w - scene.env_origins[:, None, :]).clone()
    q_hold = {n: r.data.joint_pos.clone() for n, r in (("left", left), ("back", back))}
    speeds = []
    for i in range(600):
        step_physics()
        if i % 60 == 59:
            speeds.append(float(blk.data.object_link_lin_vel_w[0].norm(dim=-1).max()))
    p1 = blk.data.object_link_pos_w - scene.env_origins[:, None, :]
    report["block_initial_pos_env0_m"] = p0[0].cpu().numpy().round(5).tolist()
    report["block_disp_mm_after_5s_env0"] = dict(zip(blk.object_names, ((p1 - p0)[0].norm(dim=-1) * 1000).cpu().numpy().round(3).tolist()))
    report["block_disp_mm_max_over_envs"] = float(((p1 - p0).norm(dim=-1) * 1000).max())
    report["block_max_speed_m_s_per_0.5s"] = [round(s, 5) for s in speeds]
    for n, r in (("left", left), ("back", back)):
        report[f"{n}_joint_hold_abs_err"] = (r.data.joint_pos[0] - q_hold[n][0]).abs().cpu().numpy().round(5).tolist()

elif args.mode == "fk":
    T = episode.num_frames
    frames = list(range(0, T, max(1, T // 60)))
    cands = ["link7", "gripper_flange", "gripper_base"]
    rec = {c: {"left": [], "back": []} for c in cands}
    for t in frames:
        write_recorded_state(scene, episode, t)
        sim.forward()
        scene.update(sim.get_physics_dt())
        e = torch.tensor(ep["observations__eef"][t], dtype=torch.float32, device=dev)
        for arm, r, sl in (("left", left, slice(0, 7)), ("back", back, slice(7, 14))):
            pe, qe = e[sl][:3], e[sl][[6, 3, 4, 5]]
            for c in cands:
                pl, ql = tcp_like(r, c)
                # offset of recorded TCP in candidate link frame: T_link^-1 * T_eef
                po, qo = math_utils.subtract_frame_transforms(pl[0:1], ql[0:1], pe[None], qe[None])
                rec[c][arm].append(torch.cat([po[0], qo[0]]).cpu().numpy())
    res = {}
    for c in cands:
        res[c] = {}
        for arm in ("left", "back"):
            a = np.stack(rec[c][arm])
            qs = a[:, 3:] * np.sign(a[:, 3:4])
            res[c][arm] = {"offset_mean_pos_m": a[:, :3].mean(0).round(6).tolist(), "offset_pos_std_mm": (a[:, :3].std(0) * 1000).round(4).tolist(),
                           "offset_mean_quat_wxyz": qs.mean(0).round(5).tolist(), "offset_quat_std": qs.std(0).round(6).tolist()}
    report["tcp_offset_candidates"] = res
    report["tcp_calibration_from_metadata"] = episode.provenance.get("tcp_calibration")
    report["frames_checked"] = len(frames)

elif args.mode == "replay":
    write_recorded_state(scene, episode, 0)
    lq = torch.tensor(ep["leader__qpos"], dtype=torch.float32, device=dev)
    oq = torch.tensor(ep["observations__qpos"], dtype=torch.float32, device=dev)
    bp = torch.tensor(ep["sim__block_poses"], dtype=torch.float32, device=dev)
    T = lq.shape[0]
    commands = DelayedCommands(lq[0], tuple(args.delay_substeps))
    qerr, berr, bpos_sim = [], [], []
    t1 = time.time()
    for t in range(T - 1):
        for _ in range(CONTROL_SUBSTEPS):
            set_arm_targets(scene, commands.push(lq[t]))
            step_physics()
        qerr.append((measured_q16(scene) - oq[t + 1]).abs().cpu().numpy())
        pos = blk.data.object_link_pos_w[0] - scene.env_origins[0]
        bpos_sim.append(pos.cpu().numpy())
        berr.append((pos - bp[t + 1, :, :3]).norm(dim=-1).cpu().numpy())
    qerr, berr, bpos_sim = np.stack(qerr), np.stack(berr), np.stack(bpos_sim)
    report["replay_wall_s"] = round(time.time() - t1, 1)
    report["frames"] = int(T)
    report["arm_joint_abs_err_rad_mean"] = qerr[:, [*range(7), *range(8, 15)]].mean(0).round(4).tolist()
    report["arm_joint_abs_err_rad_p95"] = np.percentile(qerr[:, [*range(7), *range(8, 15)]], 95, axis=0).round(4).tolist()
    report["gripper_width_abs_err_mm_mean"] = (qerr[:, [7, 15]].mean(0) * 1000).round(3).tolist()
    report["block_pos_err_mm_final"] = dict(zip(block_ids, (berr[-1] * 1000).round(2).tolist()))
    report["block_pos_err_mm_max_over_time"] = dict(zip(block_ids, (berr.max(0) * 1000).round(2).tolist()))
    report["block_final_pos_sim_m"] = dict(zip(block_ids, bpos_sim[-1].round(4).tolist()))
    report["block_final_pos_recorded_m"] = dict(zip(block_ids, ep["sim__block_poses"][-1][:, :3].round(4).tolist()))
    report["goal_distances_sim"] = [
        {"block": g["block_id"], "dist_to_goal_mm": round(float(np.linalg.norm(bpos_sim[-1][g["block_index"]] - np.array(g["goal"][:3]))) * 1000, 2)}
        for g in episode.target_mapping
    ]
    np.savez_compressed(args.out.replace(".json", "_series.npz"), qerr=qerr, berr=berr, bpos_sim=bpos_sim)

elif args.mode == "bench":
    write_recorded_state(scene, episode, 0)
    for _ in range(10):
        sim.step(render=False)
    torch.cuda.synchronize()
    t1 = time.time()
    n = args.bench_steps
    lq = torch.tensor(ep["leader__qpos"], dtype=torch.float32, device=dev)
    for i in range(n):
        if i % CONTROL_SUBSTEPS == 0:
            set_arm_targets(scene, lq[min(i // CONTROL_SUBSTEPS, lq.shape[0] - 1)])
        step_physics()
    torch.cuda.synchronize()
    dt = time.time() - t1
    report["physics_steps"] = n
    report["wall_s"] = round(dt, 2)
    report["env_physics_steps_per_s"] = round(n * scene.num_envs / dt, 1)
    report["env_control_steps_per_s_at_30Hz"] = round(n * scene.num_envs / dt / CONTROL_SUBSTEPS, 1)
    report["realtime_factor_total"] = round(n * scene.num_envs / 120.0 / dt, 2)
    report["gpu_mem_alloc_GB_torch"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

elif args.mode == "render":
    from PIL import Image

    os.makedirs(args.render_out, exist_ok=True)
    names = {"cam_high": "high", "cam_low": "low", "cam_left_wrist": "left_wrist", "cam_back_wrist": "back_wrist"}
    for t in args.render_frames:
        write_recorded_state(scene, episode, t)
        sim.forward()
        for _ in range(12):  # let RTX accumulate / DLSS settle without advancing physics
            sim.render()
        scene.update(sim.get_physics_dt())
        for key, nm in names.items():
            rgb = scene[key].data.output["rgb"][0].cpu().numpy()
            Image.fromarray(rgb[..., :3]).save(os.path.join(args.render_out, f"{nm}_{t:05d}.png"))
        report[f"frame_{t}_mean_rgb"] = {nm: scene[k].data.output["rgb"][0].float().mean(dim=(0, 1)).cpu().numpy().round(1).tolist() for k, nm in names.items()}

elif args.mode == "render_bench":
    import subprocess

    write_recorded_state(scene, episode, 0)
    lq = torch.tensor(ep["leader__qpos"], dtype=torch.float32, device=dev)
    cams = ["cam_high", "cam_low", "cam_left_wrist", "cam_back_wrist"]

    def control_step(i):
        set_arm_targets(scene, lq[min(i, lq.shape[0] - 1)])
        for _ in range(CONTROL_SUBSTEPS):
            scene.write_data_to_sim()
            sim.step(render=False)
        sim.render()
        scene.update(CONTROL_SUBSTEPS * sim.get_physics_dt())
        return sum(float(scene[k].data.output["rgb"][:, 0, 0, 0].float().sum()) for k in cams)  # forces readback

    for i in range(10):
        control_step(i)
    torch.cuda.synchronize()
    t1 = time.time()
    n = 60
    for i in range(n):
        control_step(10 + i)
    torch.cuda.synchronize()
    dt = time.time() - t1
    report["resolution"] = [scene["cam_high"].cfg.width, scene["cam_high"].cfg.height]
    report["control_steps"] = n
    report["wall_s"] = round(dt, 2)
    report["env_control_steps_per_s_with_4_cams"] = round(n * scene.num_envs / dt, 1)
    report["images_per_s"] = round(4 * n * scene.num_envs / dt, 1)
    report["gpu_mem_used_MiB_nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                                           capture_output=True, text=True).stdout.strip()

print("REPORT_JSON " + json.dumps(report), flush=True)
with open(args.out, "w") as f:
    json.dump(report, f, indent=1)
# Kit can hang in SimulationApp.close(); results are already on disk
os._exit(0)
