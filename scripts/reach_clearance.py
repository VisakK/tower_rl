"""P3 baseline (Reach-PPO-plan.MD): how close the gripper, and a held block, come to the other blocks on the reacher's
direct paths and on the planned transit's (rise, travel, descend) for the same recorded task goals. Pure torch.

Gripper: points on its axis from gripper_base to the finger tips (9 mm past the pinch), radius ``--grip_r``. Held
block: a ``BLOCK_SIZE`` box carried at its recorded offset from the pinch (at the transit's start). Obstacles: every
block and foundation at its recorded pose (``BLOCK_SIZE`` boxes) except the held one; the transit's own block (to grasp,
push or push in) is reported apart, since the goal is next to it by design. Clearance = signed distance - radius.
Both paths are kinematic (the reacher's measured joints; the planned pinch poses), the arm links are not checked.

  python scripts/reach_clearance.py results/reach/tasks/check_left_<tag>.json --goals 'results/reach/tasks/heldout_*.jsonl'
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl import geometry as geo  # noqa: E402
from tower_rl.data.layouts import BLOCK_SIZE  # noqa: E402
from tower_rl.kinematics import ArmBase, ArmKinematics, load_model  # noqa: E402
from tower_rl.skills.reach_tasks import load_goals  # noqa: E402
from tower_rl.skills.transit import TransitCfg, plan_waypoints  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dump", help="reach_task_check.py --dump ... --traj output")
ap.add_argument("--goals", nargs="+", required=True)
ap.add_argument("--grip_r", type=float, default=0.025, help="gripper body radius about its axis (m)")
ap.add_argument("--margin", type=float, default=0.0, help="report paths closer than this (m)")
ap.add_argument("--away", type=float, default=0.03, help="mid-path: rows with the pinch this far from both the start and "
                                                         "the goal (m); near them the gripper is next to blocks by design")
args = ap.parse_args()

HALF = torch.tensor(BLOCK_SIZE) / 2
GB = float(load_model()["tcp"]["pinch_in_gripper_base"][2])  # gripper_base -> pinch along the gripper axis
AX = torch.linspace(-GB, 0.009, 20)                            # gripper axis points, pinch frame z


def box_sd(p, c, q):
    """Signed distance of points ``p [..., 3]`` to boxes (centres ``c [B, 3]``, quats ``q [B, 4]``) -> ``[..., B]``."""
    rel = p[..., None, :] - c
    loc = geo.quat_apply(geo.quat_conj(q).expand(*rel.shape[:-1], 4), rel)
    d = loc.abs() - HALF
    return d.clamp(min=0).norm(dim=-1) + d.max(-1).values.clamp(max=0)


def block_corners(pos, quat):
    s = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=torch.float32) * HALF
    return pos[:, None] + geo.quat_apply(quat[:, None].expand(-1, 8, 4), s[None].expand(len(pos), 8, 3))


def clearance(pp, pq, g, held_off):
    """Min mid-path clearance (m) of the gripper (+ held block) along pinch poses ``pp [T, 3]``, ``pq [T, 4]`` (station
    frame) to the other blocks, and to the transit's own block."""
    mid = ((pp - pp[:1]).norm(dim=-1) > args.away) & ((pp - pp[-1:]).norm(dim=-1) > args.away)
    if not bool(mid.any()):
        return 9.0, None
    pp, pq = pp[mid], pq[mid]
    T = len(pp)
    z = geo.quat_apply(pq, torch.tensor([0.0, 0.0, 1.0]).expand(T, 3))
    pts = pp[:, None] + AX[None, :, None] * z[:, None]                  # [T, 20, 3]
    rad = torch.full(pts.shape[:2], args.grip_r)
    if held_off is not None:  # the held block's corners and centre, carried with the pinch
        hp, hq = geo.frame_compose(pp, pq, held_off[0].expand(T, 3), held_off[1].expand(T, 4))
        hpts = torch.cat([block_corners(hp, hq), hp[:, None]], 1)
        pts = torch.cat([pts, hpts], 1)
        rad = torch.cat([rad, torch.zeros(T, 9)], 1)
    ids = [b for b in g["blocks"] if b != (g["block"] if g["held"] else None)]
    own = g["block"] if not g["held"] else None
    bp = torch.tensor([g["blocks"][b][:3] for b in ids])
    bq = torch.tensor([g["blocks"][b][3:] for b in ids])
    sd = box_sd(pts, bp, bq) - rad[..., None]                            # [T, P, B]
    per_block = sd.amin((0, 1))
    other = [i for i, b in enumerate(ids) if b != own]
    c_other = float(per_block[other].min()) if other else 9.0
    c_own = float(per_block[ids.index(own)]) if own in ids else None
    return c_other, c_own


goals = {g["id"]: g for g in load_goals(sorted(f for p in args.goals for f in glob.glob(p))) if "q" in g}
recs = json.loads(Path(args.dump).read_text())
assert recs and "q_path" in recs[0], "needs reach_task_check.py --traj"
arm = None
out = []
for r in recs:
    g = goals[r["id"]]
    if arm != g["arm"]:
        arm = g["arm"]
        kin, base = ArmKinematics(device="cpu").set_arm(arm), ArmBase(arm, device="cpu")
        ai = ("left", "back").index(arm)
    qp = torch.tensor(r["q_path"])
    pp, pq = base.to_world(*kin.fk(qp, "pinch"))
    p0, q0 = pp[:1], pq[:1]
    held_off = None
    if g["held"] and g["block"] in g["blocks"]:
        b = torch.tensor(g["blocks"][g["block"]])
        held_off = geo.frame_relative(p0, q0, b[None, :3], b[None, 3:])
    c_re, own_re = clearance(pp, pq, g, held_off)
    # the planned transit from the same start (pinch FK of the recorded start joints) to the same goal
    gp, gq = torch.tensor(g["goal_pos"]), torch.tensor(g["goal_quat"])
    s0p, s0q = base.to_world(*kin.fk(torch.tensor([g["q"][ai]]), "pinch"))
    ps, qs = plan_waypoints(s0p[0], s0q[0], gp, gq, g["held"], TransitCfg(dt=0.02))
    c_pl, own_pl = clearance(torch.cat([s0p, ps]), torch.cat([s0q, qs]), g, held_off)
    out.append({"id": r["id"], "target": g["target"], "held": g["held"], "reach": c_re, "planned": c_pl,
                "reach_own": own_re, "planned_own": own_pl, "in_band": r["in_band"]})


def frac(v, x):
    return sum(c < x for c in v) / max(len(v), 1)


print(f"{'target':16s} {'n':>4s}  contact (clearance < {args.margin * 1000:.0f} mm), other blocks: reacher / planned; "
      f"p5 clearance mm")
for tg in sorted({o["target"] + ("_held" if o["held"] else "") for o in out}):
    rs = [o for o in out if o["target"] + ("_held" if o["held"] else "") == tg]
    re_, pl = [o["reach"] for o in rs], [o["planned"] for o in rs]
    q5 = lambda v: sorted(v)[int(0.05 * (len(v) - 1))] * 1000  # noqa: E731
    only = sum(o["reach"] < args.margin <= o["planned"] for o in rs) / len(rs)
    print(f"{tg:16s} {len(rs):4d}  {frac(re_, args.margin):.3f} / {frac(pl, args.margin):.3f}   "
          f"p5 {q5(re_):6.1f} / {q5(pl):6.1f}   reacher only {only:.3f}")
allr, allp = [o["reach"] for o in out], [o["planned"] for o in out]
print(f"{'all':16s} {len(out):4d}  {frac(allr, args.margin):.3f} / {frac(allp, args.margin):.3f}")
Path(args.dump).with_suffix(".clearance.json").write_text(json.dumps(out))
