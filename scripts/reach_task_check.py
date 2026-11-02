"""P2 task-goal check (Reach-PPO-plan.MD): the teacher's recorded transits (``rehearse.py --transit_log``) replayed on
the reacher, each from its measured start joints with its held block, one goal per env for ``--rows`` rows.

Per goal: settled (the env's settle test) and when, the end error, whether the end pose is inside the next skill's
tight band (``reach_tasks.BANDS``) and the graph's arrival tolerance, the lowest pinch, and the planned transit's time
on the same goal. Envs whose episode aborts (error past ``goal.abort_m``) count as failures.

  python scripts/reach_task_check.py --headless --arm left --policy <ckpt> --goals results/reach/tasks/*.jsonl \
      [--device cpu] [--dump out.json]

``--block CASE`` (P2d, Reacher-V1.MD §11.1): held transits carry a real block (the env's ``held_block``: grasp offset
like the grasp skill's, fingers in the variant's regime, a D1 case's mass / friction: ``BLOCK_CASES``, or ``d1`` for a
random case per transit), settled for ``held_block.settle_rows`` before the move. Per held transit: dropped (width
< 44 mm or the block 25 mm off its settled offset in the pinch frame) and when, drift, worst slip per row.
``--driver planned``: the planned transit (``transit.plan_waypoints`` at the graph node's speeds + the IK controller, as
``executor.TransitNode``) instead of the reacher: the reference on the same goals and blocks.
"""
import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="left")
ap.add_argument("--policy", default=None)
ap.add_argument("--goals", nargs="+", required=True)
ap.add_argument("--num_envs", type=int, default=256)
ap.add_argument("--rows", type=int, default=80, help="rows per goal (10 Hz)")
ap.add_argument("--override", default="{}", help="JSON merged into configs/skills/reach.yaml (@file: from a file)")
ap.add_argument("--v_held", type=float, default=None, help="pinch speed limit (m/s) for held transits (speed.obs models)")
ap.add_argument("--v_empty", type=float, default=None, help="... and for the others; default: none (speed.free_m_s)")
ap.add_argument("--dump", default=None)
ap.add_argument("--via", type=float, default=None, help="via-points: rise to a travel height, travel, descend. The height is "
                                                         "this much (m) above the tallest block top (+ 3 cm holding a "
                                                         "block), at least 2 cm above the goal, at most --via_cap; the "
                                                         "reacher gets the next one within --via_tol of the current one")
ap.add_argument("--via_cap", type=float, default=0.33, help="travel height cap (m; the training goals reach 0.35)")
ap.add_argument("--via_rise", type=float, default=0.05, help="a start below the travel height + this rises at least this "
                                                             "much first (m), straight up; higher starts go straight to "
                                                             "the travel height above the goal")
ap.add_argument("--via_tol", nargs=2, type=float, default=[0.01, 20.0], help="via-point reached: m, deg (10 deg: two Task 5 "
                                                                            "held pre-inserts stalled, the wrist at its limit)")
ap.add_argument("--via_stall", type=float, default=0.0, help="also the next via-point once the moving goal has sat on the "
                                                               "current one this long (s) and the pinch is within --via_tol's "
                                                               "position part (executor reach.via_stall_s; 0: off)")
ap.add_argument("--carrot", nargs=2, type=float, default=None, metavar=("V_HELD", "V_EMPTY"),
                help="a moving goal (reach_tasks.Carrot) toward each via-point / the goal at these pinch speeds (m/s)")
ap.add_argument("--carrot_a", type=float, default=0.6, help="the carrot's acceleration (m/s^2)")
ap.add_argument("--carrot_w", type=float, default=1.0, help="the carrot's turn rate (rad/s)")
ap.add_argument("--parity", action="store_true", help="check reach_policy.Reacher (the deployable observation and "
                                                     "command) against the env's, row by row")
ap.add_argument("--replay", default=None, help="a --traj dump: replay its recorded actions open-loop (no policy) on the "
                                                "same goals, e.g. GPU actions on CPU PhysX")
ap.add_argument("--traj", action="store_true", help="with --dump: each goal's joint path (one row per 10 Hz row; "
                                                    "scripts/reach_clearance.py)")
ap.add_argument("--block", default=None, help="held transits carry a real block of this D1 case (BLOCK_CASES; d1: a random "
                                              "case per transit)")
ap.add_argument("--held_only", action="store_true", help="only the held transits")
ap.add_argument("--driver", choices=["policy", "planned"], default="policy")
ap.add_argument("--graphs", default="results/bundles/student_v1_3/graphs", help="--driver planned: per-node transit speeds")
ap.add_argument("--finger_friction", nargs=2, type=float, default=[2.4, 2.0], help="--block: static, dynamic (the station's)")
ap.add_argument("--seed", type=int, default=0, help="--block: grasp offsets and case draws")
ap.add_argument("--chunk", type=int, default=1, help="rows per policy call (Reach-PPO-plan.MD §2.6): the deployable reacher acts "
                                                     "on predicted joints (its q_des) inside a chunk and on measured ones "
                                                     "at its start; closed loop while the moving goal sits on a via-point / "
                                                     "the goal (as executor.LearnedTransitNode)")
ap.add_argument("--chunk_pred", choices=["qdes", "offset"], default="offset", help="--chunk: the predicted joints inside a chunk: "
                                                                               "q_des less the offset q_des - q measured at the "
                                                                               "chunk's start (sag, lag; reach_policy.Reacher and the "
                                                                               "env's chunk), or q_des (before 10-31 15:00)")
ap.add_argument("--chunk_final", choices=["closed", "chunk"], default="closed",
                help="--chunk: closed loop while the moving goal sits on a via-point / the goal (the executor's rule for "
                     "models trained closed loop), or chunks to the end (chunk-trained models). With --parity: the env's "
                     "own chunks (fixed length --chunk) are compared with the deployable's")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
if args.driver == "policy" and args.policy is None:
    ap.error("--policy is required unless --driver planned")
if args.parity and args.chunk > 1 and args.chunk_final != "chunk":
    ap.error("--parity with --chunk compares against the env's fixed-length chunks: needs --chunk_final chunk")
app = AppLauncher(args).app

import torch  # noqa: E402

from tower_rl import geometry as geo  # noqa: E402
from tower_rl.graph.interface import ARMS  # noqa: E402
from tower_rl.sim.assets import q_full  # noqa: E402
from tower_rl.skills.reach_env import ReachEnv, ReachEnvCfg  # noqa: E402
from tower_rl.skills.reach_tasks import BANDS, Carrot, load_goals, structure_top, via_points  # noqa: E402

goals = [g for g in load_goals(sorted(f for p in args.goals for f in glob.glob(p)), arm=args.arm) if "q" in g]
if args.held_only:
    goals = [g for g in goals if g["held"]]
assert goals, "no recorded transits for this arm"
ov = json.loads(Path(args.override[1:]).read_text() if args.override.startswith("@") else args.override)
ov["arm"] = args.arm
ov.setdefault("reset", {})["uniform_frac"] = 0.0
ov["payload"] = dict(ov.get("payload", {}), frac=0.0)  # set per goal below
blk = args.block or ("nominal" if float(ov.get("held_block", {}).get("frac", 0.0)) > 0 else None)
if blk:  # a block in the scene; this script records drops itself (the env would reset the env)
    ov["held_block"] = dict(ov.get("held_block", {}), frac=1.0, drop_m=10.0, drop_width_m=-1.0)
n = min(args.num_envs, len(goals))
cfg = ReachEnvCfg()
cfg.seed = args.seed
cfg.setup(num_envs=n, device=args.device, overrides=ov)
env = ReachEnv(cfg)
env.reset()
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from tower_rl.skills.ppo_cfg import runner_cfg  # noqa: E402

wrapped = RslRlVecEnvWrapper(env)
if args.driver == "policy":
    runner = OnPolicyRunner(wrapped, runner_cfg(cfg.skill, "reach_task_check").to_dict(), log_dir=None, device=args.device)
    runner.load(args.policy, load_optimizer=False)
    policy = runner.get_inference_policy(device=args.device)
dev, ai = env.device, ARMS.index(args.arm)
gen = torch.Generator(device=dev).manual_seed(args.seed + 4242)

# D1 cases (configs/tiers.yaml) as block dynamics: mass scale, block friction; d1 draws a case per transit with D1's
# weights (shifted_assembly, irregular_layers and lighting have the nominal block)
BLOCK_CASES = {"nominal": {}, "lightweight": {"mass": (0.4, 0.7)}, "heavy": {"mass": (1.4, 2.0)},
               "uneven_weights": {"mass": (0.4, 2.0)}, "slippery": {"fs": (0.15, 0.30), "fd": (0.07, 0.14)},
               "high_friction": {"fs": (0.9, 1.3), "fd": (0.5, 0.8)}, "combined": {"mass": (0.5, 1.8), "mixed": True}}
D1_W = {"nominal": 4, "lightweight": 1, "heavy": 1, "uneven_weights": 1, "slippery": 1, "high_friction": 1, "combined": 1}


def block_case(name: str, k: int):
    """Per transit: case names, mass scale ``[k]``, block friction ``[k, 2]``."""
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(k, generator=gen, device=dev)  # noqa: E731
    if name == "d1":
        w = torch.tensor([float(D1_W[c]) for c in BLOCK_CASES], device=dev)
        names = [list(BLOCK_CASES)[i] for i in torch.multinomial(w, k, replacement=True, generator=gen).tolist()]
    else:
        names = [name] * k
    mass, fs, fd = torch.ones(k, device=dev), torch.full((k,), 0.40, device=dev), torch.full((k,), 0.23, device=dev)
    for c, spec in BLOCK_CASES.items():
        sel = torch.tensor([x == c for x in names], device=dev)
        if not bool(sel.any()):
            continue
        if "mass" in spec:
            mass = torch.where(sel, u(*spec["mass"]), mass)
        if "fs" in spec:
            fs, fd = torch.where(sel, u(*spec["fs"]), fs), torch.where(sel, u(*spec["fd"]), fd)
        if spec.get("mixed"):  # per block slippery or high friction
            slip = torch.rand(k, generator=gen, device=dev) < 0.5
            fs = torch.where(sel, torch.where(slip, u(0.15, 0.30), u(0.9, 1.3)), fs)
            fd = torch.where(sel, torch.where(slip, u(0.07, 0.14), u(0.5, 0.8)), fd)
    return names, mass, torch.stack([fs, torch.minimum(fd, fs)], -1)


if args.driver == "planned":
    from tower_rl.control import ArmController  # noqa: E402
    from tower_rl.skills.transit import TransitCfg, plan_waypoints  # noqa: E402

    ctrl = ArmController(args.arm, n, device=dev, kin=env.kin)
    graph_nodes = {}

    def node_spec(g: dict) -> dict:
        if g["variant"] not in graph_nodes:
            p = Path(args.graphs) / f"{g['variant']}.json"
            graph_nodes[g["variant"]] = {x["id"]: x for x in json.loads(p.read_text())["nodes"]} if p.exists() else {}
        return graph_nodes[g["variant"]].get(g["node"], {})

    def planned_pre(actions):
        """The planned transit's row: IK on the plan's next target (the start until the block settled)."""
        k = plan_state["k"] - plan_state["start"]
        j = torch.clamp(torch.full((n,), k, device=dev), min=0)
        j = torch.minimum(j, plan_state["len"] - 1)
        ar = torch.arange(n, device=dev)
        moving = k >= 0
        tp = plan_state["P"][ar, j] if moving else plan_state["p0"]
        tq = plan_state["Q"][ar, j] if moving else plan_state["q0"]
        ctrl.set_target(tp, tq)
        env.q_cmd_prev[:] = env.q_cmd
        env.q_cmd[:] = ctrl.command(env.robot.data.joint_pos[:, :7], env.payload)
        env.q_des[:] = ctrl.q_des
        env.substep = 0
        plan_state["k"] += 1

    plan_state = {}
    env._pre_physics_step = planned_pre
lim = env.kin.limits
recs = []
par = {"obs": 0.0, "cmd": 0.0, "q_des": 0.0}
rep_a = {r["id"]: r["a_path"] for r in json.loads(Path(args.replay).read_text())} if args.replay else {}
if args.replay:
    goals = [g for g in goals if g["id"] in rep_a]
for b0 in range(0, len(goals), n):
    batch = goals[b0:b0 + n]
    m = len(batch)
    idx = torch.arange(n, device=dev) % m  # pad the last batch by repeating its goals
    t = lambda key: torch.tensor([batch[i][key] for i in idx.tolist()], dtype=torch.float32, device=dev)  # noqa: E731
    q0 = env._clamp_q(torch.tensor([batch[i]["q"][ai] for i in idx.tolist()], dtype=torch.float32, device=dev))
    env.payload[:] = t("payload_kg")
    gp, gq = env.base.to_base(t("goal_pos"), t("goal_quat"))
    ids = torch.arange(n, device=dev)
    held_t = t("payload_kg") > 0
    w_init = env.w_open
    if blk:  # held: the block in the fingers, the variant's finger regime, the case's dynamics; empty: the node's fingers
        env.held[:] = held_t
        w_init = env._spawn_held(ids, q0)
        t45 = torch.tensor([batch[i]["variant"].startswith(("task4", "task5")) for i in idx.tolist()], device=dev)
        env.regime[:] = t45.float()
        rg = env.hb["regimes"]
        tool = torch.tensor([{"prepush": 0.0, "prepushin": 0.02}.get(batch[i]["target"], 0.1) for i in idx.tolist()],
                            device=dev)
        env.w_cmd[:, 0] = torch.where(held_t, torch.where(t45, rg["t45"]["closed_m"], rg["t13"]["closed_m"]), tool)
        w_init = torch.where(held_t[:, None], w_init, env.w_cmd)
        case_names, ms, bf = block_case(blk, n)
        ff = torch.tensor([args.finger_friction], device=dev).expand(n, 2)
        env.set_held_dynamics(ids, ms, bf, ff)
    env.robot.write_joint_state_to_sim(q_full(q0, w_init), torch.zeros(n, 10, device=dev), env_ids=ids)
    q_cmd = torch.max(torch.min(env.kin.sag_compensated_command(q0, env.payload), lim[:, 1]), lim[:, 0])
    env.robot.set_joint_position_target(q_full(q_cmd, env.w_cmd))
    settle = int(env.hb["settle_rows"]) if blk else 0
    if args.driver == "planned":  # plans from the start pinch pose, padded with their last row
        ctrl.reset(q0)
        p0w, q0w = ctrl.pinch_world(q0)
        gw_p, gw_q = t("goal_pos"), t("goal_quat")
        plans = []
        for i in range(n):
            spec = node_spec(batch[int(idx[i])])
            tc = TransitCfg(**{k: spec[k] for k in ("v_max", "v_max_held", "a_max") if k in spec})
            plans.append(plan_waypoints(p0w[i], q0w[i], gw_p[i], gw_q[i], bool(held_t[i]), tc))
        L = max(len(p) for p, _ in plans)
        P = torch.stack([torch.cat([p, p[-1:].expand(L - len(p), 3)]) for p, _ in plans])
        Q = torch.stack([torch.cat([q, q[-1:].expand(L - len(q), 4)]) for _, q in plans])
        plan_state.update(k=0, start=settle, P=P, Q=Q, p0=p0w, q0=q0w,
                          len=torch.tensor([len(p) for p, _ in plans], device=dev))
    for buf, v in ((env.q_des, q0), (env.q_cmd, q_cmd), (env.q_cmd_prev, q_cmd), (env.prev_q, q0)):
        buf[:] = v
    for buf in (env.prev_qd, env.actions, env.prev_actions, env.raw, env.prev_raw):
        buf.zero_()
    env.goal_p[:], env.goal_q[:] = gp, gq
    env.trk[:] = False  # the goals here are this script's (its own moving goal: --carrot)
    env.goal_v.zero_()
    cur = torch.full((n,), 2, dtype=torch.long, device=dev)
    if args.via is not None:  # slots: above the start (start orientation), above the goal (goal orientation), the goal
        s0w, s0q = env.base.to_world(*env.kin.fk(q0, "pinch"))
        tops = torch.tensor([structure_top(batch[i]["blocks"], batch[i]["block"] if batch[i]["held"] else None)
                             for i in idx.tolist()], device=dev)
        vw, valid = via_points(s0w, s0q, t("goal_pos"), t("goal_quat"), tops, t("payload_kg") > 0, margin=args.via,
                               rise=args.via_rise, cap=args.via_cap)
        vias = torch.cat(env.base.to_base(vw[..., :3].reshape(-1, 3), vw[..., 3:].reshape(-1, 4)), -1).reshape(n, 3, 7)
        cur = valid.float().argmax(1)
        ar = torch.arange(n, device=dev)
        env.goal_p[:], env.goal_q[:] = vias[ar, cur, :3], vias[ar, cur, 3:]
    slot_p, slot_q = env.goal_p.clone(), env.goal_q.clone()  # the current via-point / goal
    car = None
    if args.carrot:  # the env's goal is the carrot, the via switch test is on the slot
        vc = torch.full((n,), args.carrot[1], device=dev)
        vc[t("payload_kg") > 0] = args.carrot[0]
        car = Carrot(*env.kin.fk(q0, "pinch"), vc, args.carrot_a, args.carrot_w)
        hold = (env.settle_left > 0)[:, None] if blk else torch.zeros(n, 1, dtype=torch.bool, device=dev)
        cp, cq = car.p.clone(), car.q.clone()
        car.step(torch.where(hold, car.p, slot_p), torch.where(hold, car.q, slot_q), env.row_dt)
        env.goal_p[:], env.goal_q[:] = car.p, car.q
        env.goal_v[:] = torch.cat([(car.p - cp) / env.row_dt, geo.quat_error_rotvec(cq, car.q) / env.row_dt], -1)
    env.prev_pb[:] = env.kin.fk(q0, "pinch")[0]
    vh, ve = (env.v_free if v is None else v for v in (args.v_held, args.v_empty))
    env.v_lim[:] = torch.where(t("payload_kg") > 0, vh, ve)
    env.goal_t.zero_()
    env.settle_n.zero_()
    env.first_settle.fill_(-1)
    env.goal_dur.fill_(10 ** 6)  # no goal switch inside the window
    env.episode_length_buf.zero_()
    if env.ck_on:  # the env's chunks (chunk-trained models): from a measured row here, --chunk rows long (1: closed loop)
        env.ck_lengths, env.ck_weights = torch.tensor([args.chunk], device=dev), torch.ones(1, device=dev)
        env.ck_k.zero_()
        env.ck_len[:] = env._ck_draw(n)
        env.ck_off.zero_()
        env.q_obs[:], env.q_obs_prev[:] = q0, q0
    obs = wrapped.get_observations()
    if args.parity or args.chunk > 1:
        from tensordict import TensorDict

        from tower_rl.skills.reach_policy import Reacher

        rch = Reacher(None, args.arm, cfg.skill, n, args.device)
        rch.reset(q0)
    last_obs = torch.full((n,), -10 ** 6, dtype=torch.long, device=dev)  # --chunk: row of the last measured observation
    calls = torch.zeros(n, dtype=torch.long, device=dev)  # ... and the policy calls until the first settle
    car_on = torch.zeros(n, dtype=torch.bool, device=dev)
    slot_rows = torch.zeros(n, dtype=torch.long, device=dev)  # rows the carrot has sat on the current via-point
    w_obs = env.width().clone()
    q_off = torch.zeros(n, 7, device=dev)  # --chunk_pred offset: q_des - q at the chunk's start
    dead = torch.zeros(n, dtype=torch.bool, device=dev)
    z_min = torch.full((n,), 9.0, device=dev)
    # pinch speed / acceleration from row to row (10 Hz), vs the planned transit's limits (0.25 m/s empty, 0.12 held,
    # 0.6 m/s^2): a held block may slip at the reacher's speed
    p_prev = env.base.to_world(*env.kin.fk(q0, "pinch"))[0]
    v_prev = torch.zeros(n, 3, device=dev)
    v_max, a_max = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
    path = [q0.clone()]
    acts = []
    if args.replay:
        A = torch.tensor([rep_a[batch[i]["id"]] for i in idx.tolist()], dtype=torch.float32, device=dev)  # [n, rows, 7]
    dropped = torch.zeros(n, dtype=torch.bool, device=dev)  # --block: held transits only
    drop_row = torch.full((n,), -1, dtype=torch.long, device=dev)
    drift_max, slip_max = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
    for k in range(args.rows):
        with torch.inference_mode():
            if args.driver == "planned":
                a = torch.zeros(n, 7, device=dev)
            elif args.chunk > 1:  # a new call: measured joints; inside a chunk: the reacher's own prediction
                qm = env.robot.data.joint_pos[:, :7].clone()
                final = car_on if car is not None and args.chunk_final == "closed" else torch.zeros_like(car_on)
                fresh = (k - last_obs >= args.chunk) | final  # final: the carrot on its via-point / goal
                last_obs = torch.where(fresh, torch.full_like(last_obs, k), last_obs)
                calls += (fresh & (env.first_settle < 0)).long()
                w_obs = torch.where(fresh, env.width(), w_obs)
                if args.chunk_pred == "offset":
                    q_off = torch.where(fresh[:, None], rch.q_des - qm, q_off)
                q_obs = torch.where(fresh[:, None], qm, rch.q_des - q_off)
                ob = rch.obs(q_obs, env.goal_p, env.goal_q, env.v_lim, w_obs, env.regime, env.goal_v, (k - last_obs).float())
                if args.parity:  # the env's own chunk view on the same rows (chunk-trained models, --chunk_final chunk)
                    par["obs"] = max(par["obs"], float((ob - obs["policy"]).abs().max()))
                a = policy(TensorDict({"policy": ob, "critic": obs["critic"]}, batch_size=[n]))
                a_eff = torch.where((env.settle_left > 0)[:, None], torch.zeros_like(a), a) if blk else a
                rch.apply(q_obs, a_eff.clamp(-1.0, 1.0), env.payload)
            else:
                a = policy(obs) if not args.replay else A[:, k]
        if args.parity and args.chunk > 1:
            cmd = rch.kin.sag_compensated_command(rch.q_des, env.payload)
            cmd = torch.max(torch.min(cmd, lim[:, 1]), lim[:, 0])
        elif args.parity:
            qm = env.robot.data.joint_pos[:, :7].clone()
            ob = rch.obs(qm, env.goal_p, env.goal_q, env.v_lim, env.width(), env.regime, env.goal_v)
            par["obs"] = max(par["obs"], float((ob - obs["policy"]).abs().max()))
            a_eff = torch.where((env.settle_left > 0)[:, None], torch.zeros_like(a), a) if blk else a  # the env's settle
            cmd = rch.apply(qm, a_eff.clamp(-1.0, 1.0), env.payload)
        if args.traj:
            acts.append(a.clone())
        saved = (env.prev_q.clone(), env.prev_qd.clone(), env.prev_pb.clone())
        obs, _, dones, _ = wrapped.step(a)
        dead |= dones.bool()
        if blk:  # the env's drift (to the settled offset) and slip, measured on held transits after the settle
            live = held_t & (env.settle_left == 0)
            lost = live & ~dropped & ((env.drift > 0.025) | (env.width() < 0.044))
            drop_row[lost] = k
            dropped |= lost
            drift_max = torch.maximum(drift_max, torch.where(live & ~dropped, env.drift, torch.zeros_like(env.drift)))
            slip_max = torch.maximum(slip_max, torch.where(live & ~dropped, env.slip_vec.norm(dim=-1), torch.zeros_like(env.drift)))
        if args.parity:
            par["cmd"] = max(par["cmd"], float((cmd - env.q_cmd).abs().max()))
            par["q_des"] = max(par["q_des"], float((rch.q_des - env.q_des).abs().max()))
        st = env._st
        changed = False
        if args.via is not None:
            pe_s, re_s = (st["pb"] - slot_p).norm(dim=-1), geo.quat_angle(st["qb"], slot_q)
            stalled = (args.via_stall > 0) & (slot_rows * env.row_dt >= args.via_stall)
            hit = (cur < 2) & (pe_s <= args.via_tol[0]) & ((re_s <= math.radians(args.via_tol[1])) | stalled)
            slot_rows[hit] = 0
            if bool(hit.any()):
                nxt = torch.where(valid[ar, (cur + 1).clamp(max=2)], cur + 1, (cur + 2).clamp(max=2))
                cur = torch.where(hit, nxt, cur)
                slot_p[hit], slot_q[hit] = vias[ar, cur, :3][hit], vias[ar, cur, 3:][hit]
                if car is None:
                    env.goal_p[hit], env.goal_q[hit] = slot_p[hit], slot_q[hit]
                env.settle_n[hit] = 0
                env.first_settle[hit] = -1
                changed = True
        if car is not None:  # held: the carrot waits while the block settles in the fingers
            hold = (env.settle_left > 0)[:, None] if blk else torch.zeros(n, 1, dtype=torch.bool, device=dev)
            cp, cq = car.p.clone(), car.q.clone()
            car_on = car.step(torch.where(hold, car.p, slot_p), torch.where(hold, car.q, slot_q), env.row_dt)
            slot_rows = torch.where(car_on, slot_rows + 1, torch.zeros_like(slot_rows))
            moving = ~car_on
            env.goal_p[:], env.goal_q[:] = car.p, car.q
            env.goal_v[:] = torch.cat([(car.p - cp) / env.row_dt, geo.quat_error_rotvec(cq, car.q) / env.row_dt], -1)
            env.settle_n[moving] = 0  # settled counts once the carrot is on the goal
            env.first_settle[moving] = -1
            changed = True
        if changed:
            env.prev_q[:], env.prev_qd[:], env.prev_pb[:] = saved  # the same row's observation, the new goals
            obs = wrapped.get_observations()
        pw = env.base.to_world(st["pb"], st["qb"])[0]
        z_min = torch.minimum(z_min, pw[:, 2])
        v = (pw - p_prev) / env.row_dt
        v_max = torch.maximum(v_max, v.norm(dim=-1))
        if k:
            a_max = torch.maximum(a_max, ((v - v_prev) / env.row_dt).norm(dim=-1))
        p_prev, v_prev = pw, v
        if args.traj:
            path.append(st["q"].clone())
    pe, re = (st["pb"] - gp).norm(dim=-1), geo.quat_angle(st["qb"], gq)  # to the final goal (via: maybe not there yet)
    for i in range(m):
        g = batch[i]
        _, bp, br = BANDS[g["target"]]
        fs = int(env.first_settle[i])
        end = g.get("end") or {}
        fs = fs if int(cur[i]) == 2 else -1
        recs.append({"id": g["id"], "target": g["target"], "held": g["held"], "variant": g["variant"],
                     "aborted": bool(dead[i]), "settled": fs >= 0 and not bool(dead[i]),
                     "settle_s": round(fs * env.row_dt, 2) if fs >= 0 else None,
                     "pos_mm": round(float(pe[i]) * 1000, 2), "rot_deg": round(math.degrees(float(re[i])), 3),
                     "in_band": not bool(dead[i]) and float(pe[i]) <= bp and math.degrees(float(re[i])) <= br,
                     "in_tol": not bool(dead[i]) and float(pe[i]) <= g["tol_pos_m"] and math.degrees(float(re[i])) <= g["tol_rot_deg"],
                     "z_min_m": round(float(z_min[i]), 4), "v_max_m_s": round(float(v_max[i]), 3),
                     "a_max_m_s2": round(float(a_max[i]), 2), "start_pos_mm": g.get("start_pos_mm"),
                     "start_rot_deg": g.get("start_rot_deg"), "planned_s": round(g["planned_rows"] * env.row_dt, 2),
                     "planned_status": end.get("status"), "planned_t_s": end.get("t_s"),
                     **({"calls": int(calls[i])} if args.chunk > 1 else {})})
        if blk and g["held"]:  # a dropped block is no arrival
            dr = bool(dropped[i])
            recs[-1].update(case=case_names[i], mass=round(float(ms[i]), 3), fric=[round(float(x), 3) for x in bf[i].tolist()],
                            dropped=dr, drop_s=round(float(drop_row[i]) * env.row_dt, 1) if dr else None,
                            drift_max_mm=round(float(drift_max[i]) * 1000, 2), slip_max_mm=round(float(slip_max[i]) * 1000, 2),
                            drift_end_mm=None if dr else round(float(env.drift[i]) * 1000, 2))
            recs[-1]["in_band"] &= not dr
            recs[-1]["in_tol"] &= not dr
            recs[-1]["settled"] &= not dr
        if args.via is not None:
            recs[-1]["via_cur"] = int(cur[i])
            recs[-1]["via_slots"] = [[round(v, 4) for v in env.base.to_world(vias[i, j, :3][None], vias[i, j, 3:][None])[0][0].tolist()]
                                     for j in range(3)]
            recs[-1]["via_valid"] = valid[i].tolist()
        if args.traj:
            recs[-1]["q_path"] = [[round(v, 4) for v in row[i].tolist()] for row in path]
            recs[-1]["a_path"] = [[round(v, 5) for v in row[i].tolist()] for row in acts]
    print(f"[task_check] {min(b0 + n, len(goals))} / {len(goals)}", flush=True)


def summary(rs):
    k = len(rs)
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else float("nan")  # noqa: E731
    st = [r["settle_s"] for r in rs if r["settled"]]
    return {"n": k, "in_band": round(sum(r["in_band"] for r in rs) / k, 4), "in_tol": round(sum(r["in_tol"] for r in rs) / k, 4),
            "settled": round(sum(r["settled"] for r in rs) / k, 4), "aborted": sum(r["aborted"] for r in rs),
            "pos_mm_p50": q([r["pos_mm"] for r in rs], 0.5), "pos_mm_p95": q([r["pos_mm"] for r in rs], 0.95),
            "rot_deg_p50": q([r["rot_deg"] for r in rs], 0.5), "rot_deg_p95": q([r["rot_deg"] for r in rs], 0.95),
            "settle_s_p50": q(st, 0.5), "settle_s_p90": q(st, 0.9),
            "planned_s_p50": q([r["planned_s"] for r in rs], 0.5),
            "faster_frac": round(sum(r["settled"] and r["settle_s"] <= r["planned_s"] for r in rs) / k, 4),
            "z_min_m": min(r["z_min_m"] for r in rs),
            "v_max_p50": q([r["v_max_m_s"] for r in rs], 0.5), "v_max_p95": q([r["v_max_m_s"] for r in rs], 0.95),
            "a_max_p50": q([r["a_max_m_s2"] for r in rs], 0.5), "a_max_p95": q([r["a_max_m_s2"] for r in rs], 0.95),
            "turn_gt90": sum((r["start_rot_deg"] or 0) > 90 for r in rs),
            **({"calls_p50": q([r["calls"] for r in rs if r["settled"]], 0.5)} if rs and "calls" in rs[0] else {}),
            **held_summary([r for r in rs if "dropped" in r], q)}


def held_summary(hs, q):
    if not hs:
        return {}
    ok = [r for r in hs if not r["dropped"]]
    return {"held_n": len(hs), "drops": sum(r["dropped"] for r in hs),
            "drift_max_p50": q([r["drift_max_mm"] for r in hs], 0.5), "drift_max_p95": q([r["drift_max_mm"] for r in hs], 0.95),
            "drift_end_p50": q([r["drift_end_mm"] for r in ok], 0.5), "drift_end_p95": q([r["drift_end_mm"] for r in ok], 0.95),
            "slip_max_p50": q([r["slip_max_mm"] for r in hs], 0.5), "slip_max_p95": q([r["slip_max_mm"] for r in hs], 0.95)}


rep = {"policy": args.policy, "arm": args.arm, "device": args.device, "v_held": args.v_held, "v_empty": args.v_empty,
       "via": args.via, "carrot": args.carrot, "block": blk, "driver": args.driver, "chunk": args.chunk,
       "all": summary(recs)}
for c in BLOCK_CASES if blk == "d1" else []:
    rs = [r for r in recs if r.get("case") == c]
    if rs:
        rep["case_" + c] = summary(rs)
for tg in BANDS:
    for held in (False, True):
        rs = [r for r in recs if r["target"] == tg and r["held"] == held]
        if rs:
            rep[tg + ("_held" if held else "")] = summary(rs)
if args.parity:
    rep["parity_max_abs"] = par
print("TASK_JSON " + json.dumps(rep), flush=True)
if args.dump:
    Path(args.dump).write_text(json.dumps(recs))
os._exit(0)  # app.close() hangs in Isaac Sim 5.1
