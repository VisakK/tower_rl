"""Measure the arms' static behaviour in the station scene and compare with ``tower_rl.kinematics``.

For ``--num_envs`` joint commands (demo poses plus random perturbations), hold the command for ``--hold_s`` seconds and
record the steady-state joint angles. Reports: PhysX vs USD link masses, PhysX generalized gravity forces vs the torch
model, measured sag (q - q_cmd) vs the model's ``tau_g / kp`` prediction, and FK of PhysX link7 vs torch FK.

Usage: python scripts/probe_arm_statics.py --headless --num_envs 32 --out results/statics/probe.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

from tower_rl.paths import STATION_JSON, resolve_episode  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episode", default="t23_ep1")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--hold_s", type=float, default=4.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="results/statics/probe.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from tower_rl.kinematics import ArmKinematics, load_model  # noqa: E402
from tower_rl.scene import ARMS, CONTROL_SUBSTEPS, PHYSICS_DT, Episode, apply_friction, arm_q_full, load_station_meta, make_scene_cfg  # noqa: E402

meta = load_station_meta(STATION_JSON)
episode = Episode.load(*resolve_episode(args.episode))
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, render_interval=CONTROL_SUBSTEPS, device=args.device,
                                                          physx=sim_utils.PhysxCfg()))
scene = InteractiveScene(make_scene_cfg(meta, episode, args.num_envs))
sim.reset()
apply_friction(scene, finger=(1.2, 1.0))
dev = sim.device
model = load_model()
kin = ArmKinematics(model, device=dev, dtype=torch.float32)
gen = torch.Generator(device="cpu").manual_seed(args.seed)

# commands: demo frames (spread over the episode) + uniform perturbations within +-0.4 rad, clipped to limits
T = episode.num_frames
frames = np.linspace(0, T - 1, args.num_envs).astype(int)
q_cmd16 = torch.tensor(episode.qpos_cmd[frames], dtype=torch.float32)
pert = (torch.rand(args.num_envs, 16, generator=gen) - 0.5) * 0.8
pert[:, [7, 15]] = 0.0
pert[: args.num_envs // 4] = 0.0  # keep a quarter of the demo poses unperturbed
q_cmd16 = q_cmd16 + pert
lim = kin.limits.cpu()
for s in (slice(0, 7), slice(8, 15)):
    q_cmd16[:, s] = torch.max(torch.min(q_cmd16[:, s], lim[:, 1] - 0.05), lim[:, 0] + 0.05)
q_cmd16[:, [7, 15]] = 0.1
q_cmd16 = q_cmd16.to(dev)

report = {"num_envs": args.num_envs, "hold_s": args.hold_s, "episode": args.episode}
env_ids = torch.arange(args.num_envs, device=dev)
for i, arm in enumerate(ARMS):
    r = scene[arm]
    qf = arm_q_full(q_cmd16[:, 8 * i: 8 * i + 8])
    r.write_joint_state_to_sim(qf, torch.zeros_like(qf))
    r.set_joint_position_target(qf)
scene.write_data_to_sim()
n_steps = int(args.hold_s / PHYSICS_DT)
for k in range(n_steps):
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(PHYSICS_DT)

for i, arm in enumerate(ARMS):
    r = scene[arm]
    view = r.root_physx_view
    kin.set_arm(arm)
    q_meas = r.data.joint_pos[:, :7]
    qd = r.data.joint_vel[:, :7]
    qc = q_cmd16[:, 8 * i: 8 * i + 7]
    tau_model = kin.gravity_torque(q_meas)
    sag_model = tau_model / kin.kp
    sag_meas = q_meas - qc
    # PhysX generalized gravity forces: [N, dofs] for the current state (sign: as reported by PhysX)
    try:
        g_physx = view.get_generalized_gravity_forces().to(dev)[:, :7]
    except Exception as e:  # noqa: BLE001
        g_physx = None
        report[f"{arm}_gravity_error"] = repr(e)
    masses = view.get_masses()[0].cpu().numpy()
    coms = view.get_coms()[0].cpu().numpy()
    usd_masses = {b: model["links"].get(b, {}).get("mass") for b in r.body_names}
    ib = r.body_names.index("base_link")
    il = r.body_names.index("link7")
    from tower_rl import geometry as geo
    pb, qb = r.data.body_link_pos_w[:, ib], r.data.body_link_quat_w[:, ib]
    pl, ql = r.data.body_link_pos_w[:, il], r.data.body_link_quat_w[:, il]
    p_rel, q_rel = geo.frame_relative(pb, qb, pl, ql)
    p_fk, q_fk = kin.fk(q_meas, "link7")
    rec = {
        "body_names": r.body_names,
        "physx_masses": dict(zip(r.body_names, masses.round(5).tolist())),
        "usd_masses": usd_masses,
        "physx_coms_env0": dict(zip(r.body_names, coms[:, :3].round(5).tolist())),
        "max_joint_speed_at_end": float(qd.abs().max()),
        "sag_meas_mean_abs": sag_meas.abs().mean(0).cpu().numpy().round(5).tolist(),
        "sag_model_mean_abs": sag_model.abs().mean(0).cpu().numpy().round(5).tolist(),
        "sag_residual_mean_abs(meas-model)": (sag_meas - sag_model).abs().mean(0).cpu().numpy().round(5).tolist(),
        "sag_residual_max_abs(meas-model)": (sag_meas - sag_model).abs().amax(0).cpu().numpy().round(5).tolist(),
        "fk_link7_err_mm_max": float((p_rel - p_fk).norm(dim=-1).max() * 1000),
        "fk_link7_err_deg_max": float(geo.quat_angle(q_rel, q_fk).max() * 180 / np.pi),
        "tau_model_env0": tau_model[0].cpu().numpy().round(4).tolist(),
        "passive_gripper_joint_pos": r.data.joint_pos[:, r.joint_names.index("gripper")].cpu().numpy().round(4).tolist(),
        "per_env_residual": (sag_meas - sag_model).cpu().numpy().round(4).tolist(),
        "per_env_speed": qd.abs().amax(-1).cpu().numpy().round(4).tolist(),
        "per_env_unperturbed": [bool(x) for x in (pert[:, 8 * i: 8 * i + 7].abs().sum(-1) == 0).tolist()],
        "sag_meas_env0": sag_meas[0].cpu().numpy().round(5).tolist(),
        "sag_model_env0": sag_model[0].cpu().numpy().round(5).tolist(),
    }
    if g_physx is not None:
        rec["gravity_physx_env0"] = g_physx[0].cpu().numpy().round(4).tolist()
        # least-squares scale between PhysX gravity forces and the model per joint
        num = (g_physx * tau_model).sum(0)
        den = (tau_model * tau_model).sum(0).clamp_min(1e-9)
        rec["physx_over_model_scale_per_joint"] = (num / den).cpu().numpy().round(4).tolist()
        rec["sag_from_physx_gravity_residual_mean_abs"] = (sag_meas - g_physx / kin.kp).abs().mean(0).cpu().numpy().round(5).tolist()
        rec["sag_from_neg_physx_gravity_residual_mean_abs"] = (sag_meas + g_physx / kin.kp).abs().mean(0).cpu().numpy().round(5).tolist()
    report[arm] = rec
    print(f"[probe] {arm}: " + json.dumps({k: v for k, v in rec.items() if 'sag' in k or 'scale' in k or 'fk' in k or 'gravity' in k}), flush=True)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(report, indent=1))
print(f"[probe] wrote {args.out}", flush=True)
os._exit(0)
