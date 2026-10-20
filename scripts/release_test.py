"""Release test (CPU station scene): the executor's planned place, many times, to measure the rare "pop" on release.

Each env: the destination stack at the stage of ``--transfer`` (earlier transfers placed); the transfer's block in the
arm's closed gripper (canonical held offset + random slip) at the pre-place pose ``--descend_m`` above the executor's
release pose (``release_pose``, clearance per configuration), held there at rest; then the executor's
``PlannedPlaceNode`` runs as in a rehearsal (descent, ``pre_open_hold_s`` pause, open, wait, retreat, settle check).
Reports the final xy / yaw error against the goal and the block's peak upward speed (a pop throws it upward).
Rehearsals showed ~2-4 % of planned places ending 6-25 mm off after a bounce; a free drop (scripts/drop_test.py) and
a release from rest in the hand never do.

Usage: python scripts/release_test.py --headless --device cpu --variant task2-2_cross_tower_transfer_2 --transfer 0 \
           --num_envs 16 --trials 4 --clearances 0.002 0.005 --hold_s 0.0 0.3
"""
import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--variant", default="task2-2_cross_tower_transfer_2")
parser.add_argument("--transfer", type=int, default=0)
parser.add_argument("--tier", default="D0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--trials", type=int, default=4, help="resets per configuration")
parser.add_argument("--clearances", nargs="+", type=float, default=[0.002, 0.005])
parser.add_argument("--hold_s", nargs="+", type=float, default=[0.0, 0.3])
parser.add_argument("--slip_m", type=float, default=0.003, help="held offset sigma along the block / vertical (x0.5)")
parser.add_argument("--slip_deg", type=float, default=2.0, help="held rotation sigma about the jaw axis")
parser.add_argument("--offset_x_m", type=float, default=0.0, help="grasp this far off-centre along the block (random end)")
parser.add_argument("--descend_m", type=float, default=0.05, help="start this far above the release pose (pre-place)")
parser.add_argument("--max_s", type=float, default=12.0, help="row budget per trial")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--finger_friction", nargs=2, type=float, default=[2.4, 2.0])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)  # tiny tensors (see rehearse.py)
from isaaclab.scene import InteractiveScene  # noqa: E402

from tower_rl import geometry as geo  # noqa: E402
from tower_rl.config import load_yaml  # noqa: E402
from tower_rl.domain import DomainSampler  # noqa: E402
from tower_rl.graph.executor import Executor, release_pose  # noqa: E402
from tower_rl.graph.interface import ARMS  # noqa: E402
from tower_rl.grasp import above  # noqa: E402
from tower_rl.kinematics import ArmKinematics  # noqa: E402
from tower_rl.paths import RESULTS_DIR  # noqa: E402
from tower_rl.postures import prior_for_pose  # noqa: E402
from tower_rl.sim.assets import q_full, sim_cfg  # noqa: E402
from tower_rl.sim.station_scene import StationWorld, make_station_scene_cfg  # noqa: E402
from tower_rl.skills.obs import grasp_target_world  # noqa: E402

phys = load_yaml("physics")
sim = sim_utils.SimulationContext(sim_cfg(phys, device=args.device))
scene = InteractiveScene(make_station_scene_cfg(args.variant, args.num_envs))
sim.reset()
world = StationWorld(scene, sim, args.variant, finger_friction=tuple(args.finger_friction))
n, dev = world.n, world.dev
graph = Executor(args.variant).graph
place = next(nd for nd in graph["nodes"] if nd["skill"] == "place_release" and nd["transfer"] == args.transfer)
arm, bid = place["arm"], place["block"]
ai = ARMS.index(arm)
bi = world.block_ids.index(bid)
kin = ArmKinematics(device=dev, arm=arm)  # this arm's drives (sag compensation)
gen = torch.Generator(device=dev).manual_seed(args.seed)
sampler = DomainSampler(args.tier, device=dev)
payload = 0.1056
results = {}
for clearance in args.clearances:
    for hold_s in args.hold_s:
        # a one-node graph: this transfer's place, run by the executor's PlannedPlaceNode exactly as in a rehearsal
        g1 = copy.deepcopy(graph)
        g1["nodes"] = [dict(place, pre_open_hold_s=hold_s, release_clearance_m=clearance, first_of_transfer=True)]
        recs = []
        for trial in range(args.trials):
            dom = sampler.sample(n, gen)
            world.reset(dom, stage=args.transfer, gen=gen)
            ex = Executor(args.variant, graph=g1)
            ex.reset(n, args.variant, list(range(n)))
            pos, quat = world.block_pose()
            q_start = torch.zeros(n, 7, device=dev)
            offsets = []
            for e in range(n):
                run = ex.runs[e]
                s = world.state(e)
                gp, gq = run.dest_goal(s, place)
                bp, bq = pos[e:e + 1, bi], quat[e:e + 1, bi]
                cp, cq = grasp_target_world(bp, bq, arm)
                op, oq = geo.frame_relative(cp, cq, bp, bq)  # canonical held relation (block in the pinch frame)
                g = torch.randn(3, generator=gen, device=dev)
                sx = args.offset_x_m * (1 if torch.rand(1, generator=gen, device=dev) < 0.5 else -1)  # either end
                op = op + geo.quat_apply(oq, torch.tensor([[-sx, 0.0, 0.0]], device=dev))  # pinch sx along the block
                op = op + torch.tensor([[g[0] * args.slip_m, 0.0, g[1] * args.slip_m * 0.5]], device=dev)
                oq = geo.quat_mul(geo.quat_from_axis_angle(torch.tensor([[0.0, 1.0, 0.0]], device=dev),
                                                           g[2:3] * math.radians(args.slip_deg)), oq)
                run.held_offset[arm] = (op, oq)
                pp, pq = release_pose(run, arm, gp, gq, clearance)
                run.release_target[arm] = (pp, pq)
                sp, sq = above(pp, pq, args.descend_m)  # where the pre-place transit leaves the pinch
                seed = prior_for_pose(arm, "release", gp, gq)
                ctrl = run.ctrl[arm]
                pb, qb = ctrl.base.to_base(sp, sq)
                q_start[e] = ctrl.kin.ik(seed, pb, qb, iters=200, q_ref=seed, limit_margin=0.03)[0][0]
                offsets.append((op, oq))
            # teleport: arm at the pre-place pose holding the block (fingers on its faces), held at rest for 1 s
            w_block = torch.full((n, 1), 0.0497, device=dev)
            world.scene[arm].write_joint_state_to_sim(q_full(q_start, w_block), torch.zeros(n, 10, device=dev))
            pinch_b, pinch_qb = world.kin.fk(q_start, "pinch")
            pw, pq_w = world.bases[arm].to_world(pinch_b, pinch_qb)
            poses = torch.cat([pos, quat], -1).clone()
            for e in range(n):
                bpos, bquat = geo.frame_compose(pw[e:e + 1], pq_w[e:e + 1], *offsets[e])
                poses[e, bi, :3], poses[e, bi, 3:7] = bpos[0], bquat[0]
            world._write_blocks(poses)
            cmd = world._cmd.clone()
            cmd[:, 8 * ai: 8 * ai + 7] = kin.sag_compensated_command(q_start, torch.full((n,), payload, device=dev))
            cmd[:, 8 * ai + 7] = 0.040
            world._cmd = cmd
            world._hist = [cmd.clone() for _ in range(len(world._hist))]
            for _ in range(10):
                world.apply_row(cmd)
            for e in range(n):  # executor state as after a held pre-place transit
                run = ex.runs[e]
                s = world.state(e)
                run._cmd = {}
                for a in ARMS:
                    run.ctrl[a].reset(run.q_meas(s, a))
                    run.command(a, payload=payload if a == arm else 0.0)
                run.grip[arm].reset(width=0.040, closed=True)
                run.grip[ARMS[1 - ai]].reset(width=0.1, closed=False)
                run.ready = True
            vz_max = torch.zeros(n, device=dev)
            opened = torch.zeros(n, dtype=torch.bool, device=dev)  # peak upward speed counts from the opening on
            opened_at = [None] * n
            for row in range(int(args.max_s * 10)):
                states = [world.state(e) for e in range(n)]
                rows = np.stack([ex.runs[e].act(states[e])[0] for e in range(n)])
                for e in range(n):
                    if opened_at[e] is None and float(rows[e, 8 * ai + 7]) > 0.045:
                        opened_at[e] = row
                        opened[e] = True
                world.apply_row(torch.as_tensor(rows, device=dev), on_tick=lambda t: vz_max.copy_(torch.where(
                    opened, torch.maximum(vz_max, world.blocks.data.object_link_lin_vel_w[:, bi, 2]), vz_max)))
                if args.debug and trial == 0:
                    nd = ex.runs[0].node
                    print(f"[dbg] row {row} phase {getattr(nd, 'phase', None)} k {getattr(nd, 'k', None)} w_cmd {rows[0, 8 * ai + 7]:.3f} "
                          f"block z {float(world.block_pose()[0][0, bi, 2]) * 1000:.1f} mm vz {float(world.blocks.data.object_link_lin_vel_w[0, bi, 2]):.3f} "
                          f"vz_max {float(vz_max[0]):.3f}", flush=True)
                if all(r.i >= 1 for r in ex.runs):
                    break
            fp, fq = world.block_pose()
            for e in range(n):
                run = ex.runs[e]
                gp, gq = run.dest_goal(world.state(e), place)
                dxy = float((fp[e, bi, :2] - gp[0, :2]).norm())
                dyaw = math.degrees(float(geo.yaw_diff_sym(geo.yaw_from_quat(fq[e:e + 1, bi]), geo.yaw_from_quat(gq))[0]))
                ev = run.events[-1] if run.events else None
                recs.append({"dxy_mm": round(dxy * 1000, 2), "dyaw_deg": round(dyaw, 2), "vz_max": round(float(vz_max[e]), 3),
                             "status": ev[2] if ev else "unfinished", "open_row": opened_at[e], "case": dom.case[e]})
        dx = torch.tensor([r["dxy_mm"] for r in recs])
        vz = torch.tensor([r["vz_max"] for r in recs])
        key = f"off{args.offset_x_m * 1000:.0f}mm_clear{clearance * 1000:.1f}mm_hold{hold_s:.1f}s"
        results[key] = {"n": len(recs), "in_tol": round(float(((dx <= 5.0) & (torch.tensor([r["dyaw_deg"] for r in recs]) <= 5.0)).float().mean()), 4),
                        "pops_vz_gt_0.1": int((vz > 0.1).sum()), "dxy_mm_p50_p90_max": [round(float(torch.quantile(dx, q)), 2) for q in (0.5, 0.9, 1.0)],
                        "vz_max_p50_p90_max": [round(float(torch.quantile(vz, q)), 3) for q in (0.5, 0.9, 1.0)],
                        "worst": sorted(recs, key=lambda r: -r["dxy_mm"])[:3]}
        print(f"[release] {key}: " + json.dumps({k: v for k, v in results[key].items() if k != "worst"}), flush=True)
out = Path(args.out or RESULTS_DIR / "release_test" / f"{args.variant}_t{args.transfer}_{args.tier}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"config": vars(args), "results": results}, indent=1, default=str))
print("REPORT_JSON " + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "worst"} for k, v in results.items()}), flush=True)
os._exit(0)
