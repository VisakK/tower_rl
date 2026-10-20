"""Step 2.3 (b): replay recorded episodes open-loop in the station scene and score every operation.

Each episode of ``--variant`` in data/subset is replayed in ``--num_replays`` parallel envs (PhysX is not
reset-deterministic; the build file judges >= 8 of 10). Commands are the recorded ``leader/qpos`` at 30 Hz,
zero-order hold over 4 x 120 Hz substeps with the evaluator's ~1-substep delay. Per operation (from the recorded
gripper events): G = the block rose >= 2 cm while the gripper was closed; P = 1 s after the release the block is
within ``--tol_mm`` / ``--tol_deg`` of its goal (xy and yaw; z within 10 mm). Writes per-episode results to ``--out``.

Usage: python scripts/replay_demo.py --headless --variant task2-2_cross_tower_transfer_2 --num_replays 10 \
           --out results/calibration/replay_task2-2.json
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--variant", default="task2-2_cross_tower_transfer_2")
parser.add_argument("--episodes", nargs="*", default=None, help="npz paths (default: all subset episodes of the variant)")
parser.add_argument("--num_replays", type=int, default=10)
parser.add_argument("--finger_friction", nargs=2, type=float, default=[1.2, 1.0])
parser.add_argument("--delay_substeps", nargs=2, type=int, default=[1, 1])
parser.add_argument("--settle_s", type=float, default=1.0, help="settle blocks (raised 0.5 mm) before replaying; 0 = teleport only")
parser.add_argument("--physx", default="{}", help="JSON dict of PhysxCfg overrides, e.g. '{\"solver_type\": 0}'")
parser.add_argument("--tol_mm", type=float, default=5.0, help="primary placement tolerance (generator: 5 mm)")
parser.add_argument("--extra_tol_mm", type=float, default=10.0, help="also scored (build file: 10 mm)")
parser.add_argument("--tol_deg", type=float, default=5.0)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from tower_rl import geometry as geo  # noqa: E402
from tower_rl.data.events import operations  # noqa: E402
from tower_rl.paths import DATA_DIR, STATION_JSON  # noqa: E402
from tower_rl.scene import (CONTROL_SUBSTEPS, PHYSICS_DT, DelayedCommands, Episode, apply_friction,  # noqa: E402
                            load_station_meta, make_scene_cfg, set_arm_targets, settle_recorded_state, write_recorded_state)

files = [Path(p) for p in args.episodes] if args.episodes else sorted((DATA_DIR / "subset" / args.variant).glob("*.npz"))
episodes = [Episode.load(f) for f in files]
meta = load_station_meta(STATION_JSON)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, render_interval=CONTROL_SUBSTEPS, device=args.device,
                                                          physx=sim_utils.PhysxCfg(**json.loads(args.physx))))
scene = InteractiveScene(make_scene_cfg(meta, episodes[0], args.num_replays))
sim.reset()
apply_friction(scene, finger=tuple(args.finger_friction))
dev = sim.device
blk = scene["blocks"]
results = []
t_all = time.time()
for ep in episodes:
    assert ep.block_ids == episodes[0].block_ids, "all episodes of a variant share the block set"
    t0 = time.time()
    if args.settle_s > 0:
        settle_recorded_state(scene, sim, ep, 0, seconds=args.settle_s)
    else:
        write_recorded_state(scene, ep, 0)
    lq = torch.tensor(ep.qpos_cmd, dtype=torch.float32, device=dev)
    T = lq.shape[0]
    ops = operations(ep)
    goals = {m["block_id"]: m["goal"] for m in ep.target_mapping}
    # frames at which block poses are needed: every frame for lift detection is cheap enough on the GPU
    z_hist = torch.zeros(T, args.num_replays, len(ep.block_ids), device=dev)
    rec_q = torch.tensor(ep.qpos_obs, dtype=torch.float32, device=dev)
    q_sq_err = torch.zeros(args.num_replays, 14, device=dev)
    rec_bp = torch.tensor(ep.block_poses[:, :, :3], dtype=torch.float32, device=dev)
    traj_err_max = torch.zeros(args.num_replays, len(ep.block_ids), device=dev)
    snap = {}
    want = {min(o.open_frame + 30, T - 1) for o in ops} | {T - 1}
    cmd = DelayedCommands(lq[0], tuple(args.delay_substeps))
    for t in range(T - 1):
        for _ in range(CONTROL_SUBSTEPS):
            set_arm_targets(scene, cmd.push(lq[t]))
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(PHYSICS_DT)
        pos = blk.data.object_link_pos_w - scene.env_origins[:, None]
        z_hist[t + 1] = pos[..., 2]
        traj_err_max = torch.maximum(traj_err_max, (pos - rec_bp[t + 1]).norm(dim=-1))
        qm = torch.cat([scene["left"].data.joint_pos[:, :7], scene["back"].data.joint_pos[:, :7]], -1)
        q_sq_err += (qm - rec_q[t + 1][[*range(7), *range(8, 15)]]) ** 2
        if t + 1 in want:
            snap[t + 1] = (pos.clone(), blk.data.object_link_quat_w.clone())
    rec = {"episode": ep.name, "frames": T, "wall_s": round(time.time() - t0, 1), "ops": []}
    all_ok = torch.ones(args.num_replays, dtype=torch.bool, device=dev)
    all_ok_extra = torch.ones(args.num_replays, dtype=torch.bool, device=dev)
    for o in ops:
        bi = o.block_index
        z0 = z_hist[max(o.close_frame, 1), :, bi]
        rise = z_hist[o.close_frame:o.open_frame, :, bi].amax(0) - z0
        g_ok = rise >= 0.02
        fs = min(o.open_frame + 30, T - 1)
        p, q = snap[fs]
        goal = torch.tensor(goals[o.block_id], dtype=torch.float32, device=dev)
        gp, gq = geo.pose7_to_pq(goal)
        dxy = (p[:, bi, :2] - gp[:2]).norm(dim=-1)
        dz = (p[:, bi, 2] - gp[2]).abs()
        dyaw = geo.yaw_diff_sym(geo.yaw_from_quat(q[:, bi]), geo.yaw_from_quat(gq))
        p_ok = (dxy <= args.tol_mm / 1000) & (dz <= 0.010) & (dyaw <= math.radians(args.tol_deg))
        all_ok &= g_ok & p_ok
        all_ok_extra &= g_ok & (dxy <= args.extra_tol_mm / 1000) & (dz <= 0.010) & (dyaw <= math.radians(args.tol_deg))
        rec["ops"].append({"block": o.block_id, "arm": o.arm, "grasp_ok": int(g_ok.sum()), "place_ok": int(p_ok.sum()),
                           "lift_mm_mean": round(float(rise.mean()) * 1000, 1),
                           "place_xy_mm": [round(float(dxy.mean()) * 1000, 2), round(float(dxy.max()) * 1000, 2)],
                           "place_yaw_deg_max": round(math.degrees(float(dyaw.max())), 2)})
    pT, qT = snap[T - 1]
    mov = [i for i, f in enumerate(ep.block_fixed) if not f]
    fin_rec = (pT[:, mov] - rec_bp[T - 1, mov]).norm(dim=-1)  # [replays, movable]
    rec["final_vs_recorded_mm"] = [round(float(fin_rec.mean()) * 1000, 2), round(float(fin_rec.max()) * 1000, 2)]
    rec["traj_vs_recorded_max_mm"] = round(float(traj_err_max[:, mov].median()) * 1000, 2)
    rec["arm_joint_rms_rad"] = round(float((q_sq_err / (T - 1)).sqrt().mean()), 5)
    final = []
    for bid, g in goals.items():
        bi = ep.block_ids.index(bid)
        gp, _ = geo.pose7_to_pq(torch.tensor(g, dtype=torch.float32, device=dev))
        final.append((pT[:, bi, :2] - gp[:2]).norm(dim=-1))
    final = torch.stack(final, -1)  # [replays, blocks]
    rec["replays_all_ops_ok"] = int(all_ok.sum())
    rec["replays_all_ops_ok_extra_tol"] = int(all_ok_extra.sum())
    rec["final_xy_err_mm_max_per_replay"] = (final.amax(-1) * 1000).round(decimals=2).tolist()
    rec["final_layout_ok"] = int((final.amax(-1) <= args.tol_mm / 1000).sum())
    results.append(rec)
    print(f"[replay] {ep.name}: all ops ok in {rec['replays_all_ops_ok']}/{args.num_replays} "
          f"({rec['replays_all_ops_ok_extra_tol']} at {args.extra_tol_mm:g} mm), final layout ok "
          f"{rec['final_layout_ok']}/{args.num_replays}, {rec['wall_s']} s", flush=True)

summary = {"variant": args.variant, "num_replays": args.num_replays, "finger_friction": args.finger_friction,
           "settle_s": args.settle_s, "physx": json.loads(args.physx),
           "tol_mm": args.tol_mm, "tol_deg": args.tol_deg, "episodes": len(results),
           "episodes_passing_8_of_10": sum(r["replays_all_ops_ok"] >= math.ceil(0.8 * args.num_replays) for r in results),
           "extra_tol_mm": args.extra_tol_mm,
           "episodes_passing_8_of_10_extra_tol": sum(r["replays_all_ops_ok_extra_tol"] >= math.ceil(0.8 * args.num_replays) for r in results),
           "final_vs_recorded_mm_mean": round(sum(r["final_vs_recorded_mm"][0] for r in results) / len(results), 2),
           "arm_joint_rms_rad_mean": round(sum(r["arm_joint_rms_rad"] for r in results) / len(results), 5),
           "op_success_rate": round(sum(o["grasp_ok"] + o["place_ok"] for r in results for o in r["ops"])
                                    / (2 * args.num_replays * sum(len(r["ops"]) for r in results)), 4),
           "wall_s": round(time.time() - t_all, 1), "results": results}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(summary, indent=1))
print("REPORT_JSON " + json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
os._exit(0)
