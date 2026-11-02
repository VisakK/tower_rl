"""Goal-conditioned reaching for one Nero arm (corl_challenge/Reach-PPO-plan.MD; JAiLeR, arXiv 2011.06332). Import
after ``AppLauncher``.

Arm-only scene: the arm at its station base pose and the tabletop collider. Each 10 Hz row the policy outputs desired
joint velocities; they are integrated into the desired joint position ``q_des`` (kept ``limit_margin`` inside the
limits) and the row sent is its gravity-sag-compensated command, held for 12 substeps after the 1-substep command
delay, as the evaluator executes a row. The pinch pose error is JAiLeR's three-point error: points ``kp_len_m`` along
the goal frame's axes against the same points on the pinch frame.

Goals follow each other without resets: with ``goal.switch: settle`` a goal ends when the arm has settled on it
(``goal.settle``) or after ``goal.limit_s``; with ``timer`` it ends after a random ``min_frac``..1 of ``limit_s``
whatever the arm does (ending a goal by settling gave up the near-goal reward, and PPO learned to jitter on the goal
instead, 10-30 run 4), and the ``hold`` reward pays for being still on it. The next goal is drawn around where the arm
is, from the current curriculum region (nested regions of goal displacement; the next opens when the mean end-of-goal
error over ``curriculum.window`` goals, less the worst ``trim`` of them, is below ``curriculum.advance``). Candidates are FK of perturbed joint configurations (and of the demo posture priors), kept if
the pinch is over the table, at a working height, with the approach axis within ``max_tilt_deg`` of straight down.

``held_block`` (P2d, corl_challenge/Reacher-V1.MD §11.1): one block per env. A held episode starts with the block between
the fingers at a grasp offset like the grasp skill's, the fingers squeezing it in the Tasks 1-3 or the Task 4/5 regime,
and the arm still for ``settle_rows``; its block mass and block / finger friction ramp from the station's nominal values
to the D1 ranges. The block's motion in the pinch frame is penalised per row (slip) and against where it settled
(drift); a block ``drop_m`` off that offset, or fingers closed past it, ends the episode with ``reward.drop``. Empty
episodes park the block on a shelf, fingers open or closed (the Task 4 / 5 pushers). The command compensates the
nominal block mass, as the executor does.

``chunk`` (corl_challenge/Reacher-V2.MD §7.1): rows grouped into chunks as the evaluator executes a call's rows. A
chunk's first row sees the measured joints; the rows after it see the reacher's own prediction (q_des less the offset
q_des - q measured on the first row, the opening measured there), as ``reach_policy.Reacher`` computes it without a
measurement. Chunk lengths are drawn per chunk from ``chunk.lengths`` / ``weights``; a new goal starts a new chunk.
``chunk.obs``: + rows since the measured row (actor), the true joints and pose error (critic). Rewards use the true state.
"""

from __future__ import annotations

import math

import torch

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from tower_rl import geometry as geo
from tower_rl.config import deep_update, load_yaml
from tower_rl.grasp import cfg as grasp_cfg
from tower_rl.grasp import gripper_quat
from tower_rl.kinematics import GRAVITY, ArmBase, ArmKinematics, load_model
from tower_rl.paths import STATION_JSON
from tower_rl.postures import priors
from tower_rl.scene import load_station_meta
from tower_rl.sim.assets import arm_cfg, block_cfg, q_full, sim_cfg
from tower_rl.sim.skill_scene import PARK_X, TABLE_SIZE, _static_box, park_pose

ACTOR_DIM = 9 + 7 * 4 + 9 + 9   # pose error, (q, qd, q_des - q, previous action), pinch pose, goal pose
CRITIC_DIM = ACTOR_DIM + 7 + 3  # + true joint velocities, goal time, curriculum level, payload
FINE_DIM = 9                    # obs.fine_m > 0: + tanh(pose error / fine_m), appended to the actor observation
SPEED_DIM = 1                   # speed.obs: + the goal's pinch speed limit / speed.free_m_s, appended after those
HELD_DIM = 2                    # held_block.obs: + finger opening ((w - 5 cm) / 5 cm) and regime (1: Task 4/5), after those
HELD_CRITIC_DIM = 20            # ... and in the critic after the actor's inputs: block in the pinch frame (3 + 6), its
                                # motion there this row (3), drift (1), mass scale, block / finger friction (2 + 2), held,
                                # settling
TRACK_DIM = 6                   # track.obs: + the goal's linear (m/s) and angular (rad/s) velocity, base frame, after those
CHUNK_DIM = 1                   # chunk.obs: + rows since the last measured row / chunk.k_norm, last
CHUNK_CRITIC_DIM = 16           # ... and in the critic right after the actor's inputs: the true joints and pose error
BLOCK_HALF = (0.075, 0.02485, 0.01415)   # the block's collision hull (assets.py), half extents
PAD_HALF = (0.0084, 0.009)               # finger pads: half width (along the block), half height (along the approach)


def rot6d_mat(m: torch.Tensor) -> torch.Tensor:
    return torch.cat([m[..., :, 0], m[..., :, 1]], dim=-1)


def make_reach_scene_cfg(arm: str, num_envs: int, env_spacing: float = 2.5, held: bool = False) -> InteractiveSceneCfg:
    meta = load_station_meta(STATION_JSON)
    attrs = {
        "table": _static_box("{ENV_REGEX_NS}/Table", TABLE_SIZE, (0.48, -0.33, -0.0125), (0.55, 0.55, 0.6)),
        "robot": arm_cfg(arm, meta, contact_sensors=False),
    }
    if held:  # held_block: the station block asset; a shelf out of reach (as in the skill scene) parks it
        attrs["block"] = block_cfg(meta, "{ENV_REGEX_NS}/Block", park_pose(0), False, load_yaml("physics"))
        attrs["shelf"] = _static_box("{ENV_REGEX_NS}/Shelf", (0.2, 0.2, 0.025), (PARK_X, -0.12, -0.0125), (0.4, 0.4, 0.45))
    cls = configclass(type("ReachSceneCfg", (InteractiveSceneCfg,), attrs))
    return cls(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True)


@configclass
class ReachEnvCfg(DirectRLEnvCfg):
    skill_yaml: str = "skills/reach"
    arm: str = "left"
    decimation: int = 12
    episode_length_s: float = 30.0
    action_space: int = 7
    observation_space: int = ACTOR_DIM
    state_space: int = CRITIC_DIM
    is_finite_horizon: bool = False

    def setup(self, num_envs: int | None = None, device: str = "cuda:0", overrides: dict | None = None):
        """Fill sim/scene from the YAML config (call before creating the env)."""
        self.skill = deep_update(load_yaml(self.skill_yaml), overrides or {})
        self.arm = self.skill["arm"]
        phys = load_yaml("physics")
        self.decimation = phys["control_substeps"] * phys["policy_ticks"]
        self.episode_length_s = self.skill["episode_steps"] * self.decimation * phys["dt"]
        self.sim = sim_cfg(phys, device=device, render_interval=self.decimation)
        hb = self.skill.get("held_block", {})
        held = float(hb.get("frac", 0.0)) > 0
        self.scene = make_reach_scene_cfg(self.arm, num_envs or self.skill["num_envs"], held=held)
        fine = FINE_DIM if self.skill["obs"].get("fine_m", 0.0) > 0 else 0
        fine += SPEED_DIM if self.skill.get("speed", {}).get("obs", False) else 0
        hobs = held and bool(hb.get("obs", False))
        tobs = TRACK_DIM if self.skill.get("track", {}).get("obs", False) else 0
        cobs = CHUNK_DIM if self.skill.get("chunk", {}).get("obs", False) else 0
        self.observation_space = ACTOR_DIM + fine + (HELD_DIM if hobs else 0) + tobs + cobs
        self.state_space = (CRITIC_DIM + fine + (HELD_DIM + HELD_CRITIC_DIM if hobs else 0) + tobs
                            + (cobs + CHUNK_CRITIC_DIM if cobs else 0))
        return self


class ReachEnv(DirectRLEnv):
    cfg: ReachEnvCfg

    def __init__(self, cfg: ReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.sk = sk = cfg.skill
        phys = load_yaml("physics")
        n, dev = self.num_envs, self.device
        self.arm = cfg.arm
        self.robot = self.scene["robot"]
        self.kin = ArmKinematics(device=dev).set_arm(self.arm)
        self.base = ArmBase(self.arm, device=dev)
        # payload.force: the held block's weight at the pinch (gripper_base frame); without it only the command
        # compensates a payload, and the arm overshoots upward. held_block: the real block instead
        hb = self.hb = sk.get("held_block", {})
        self.held_on = float(hb.get("frac", 0.0)) > 0
        self.held_obs = self.held_on and bool(hb.get("obs", False))
        self.payload_force = bool(sk.get("payload", {}).get("force", False)) and not self.held_on
        if self.payload_force:
            self.gb_id = self.robot.find_bodies("gripper_base")[0][0]
            self.pinch_off = torch.tensor(load_model()["tcp"]["pinch_in_gripper_base"], device=dev).expand(n, 1, 3)
        lim = self.kin.limits
        self.lo = lim[:, 0] + sk["action"]["limit_margin"]
        self.hi = lim[:, 1] - sk["action"]["limit_margin"]
        self.q_mid, self.q_half = (lim[:, 0] + lim[:, 1]) / 2, (lim[:, 1] - lim[:, 0]) / 2
        self.row_dt = cfg.decimation * phys["dt"]
        self.delay = phys["command_delay_substeps"][self.arm]
        self.v_max = float(sk["action"]["v_max"])
        self.amap = sk["action"].get("map", "linear")
        self.lin_frac = float(sk["action"].get("lin_frac", 0.0))  # map blend: lin_frac * a + (1 - lin_frac) * a|a|
        self.kp_len = float(sk["obs"]["kp_len_m"])
        self.fine_m = float(sk["obs"].get("fine_m", 0.0))
        self.q_noise = float(sk.get("rand", {}).get("q_noise_rad", 0.0))
        sp = sk.get("speed", {})
        self.speed_obs = bool(sp.get("obs", False))
        self.v_free = float(sp.get("free_m_s", 2.0))
        self.v_lim = torch.full((n,), self.v_free, device=dev)  # the current goal's pinch speed limit (m/s)
        self.prev_pb = torch.zeros(n, 3, device=dev)
        g = sk["goal"]
        self.goal_rows = int(round(g["limit_s"] / self.row_dt))
        self.switch = g.get("switch", "settle")
        ws = g["workspace"]
        self.ws_lo = torch.tensor([ws["x"][0], ws["y"][0], ws["z"][0]], device=dev)
        self.ws_hi = torch.tensor([ws["x"][1], ws["y"][1], ws["z"][1]], device=dev)
        self.cos_tilt = math.cos(math.radians(ws["max_tilt_deg"]))
        home = load_yaml("scene")["home_qpos"][self.arm][:7]
        post = [home] + [v["q"][:7] for ph in priors()[self.arm].values() for pad in ph.values() for v in pad.values()]
        self.postures = torch.tensor(post, dtype=torch.float32, device=dev)
        tk = g.get("tasks", {})  # Phase 2: the teacher's own transits (rehearse.py --transit_log) as goals / starts
        self.task_frac, self.task_start_frac = float(tk.get("frac", 0.0)), float(sk["reset"].get("task_frac", 0.0))
        if self.task_frac > 0 or self.task_start_frac > 0 or self.held_on:
            from tower_rl.skills.reach_tasks import load_task_pool

            self.task_q, self.task_gp, self.task_gq, th = load_task_pool(tk["files"], self.arm, dev, with_held=True)
            # held_block: held episodes draw recorded held transits (pre-place / pre-insert), empty ones the others
            self.task_ids = {True: th.nonzero(as_tuple=True)[0], False: (~th).nonzero(as_tuple=True)[0]}
            if not self.held_on:
                self.task_ids = {True: torch.arange(len(th), device=dev), False: torch.arange(len(th), device=dev)}
            self.task_pos_m, self.task_yaw = float(tk.get("pos_m", 0.02)), math.radians(float(tk.get("yaw_deg", 10.0)))
        self.levels = sk["curriculum"]["levels"]
        self.level = int(sk["curriculum"].get("start_level", 0))
        w = int(sk["curriculum"]["window"])
        self._ring = torch.full((w, 2), float("nan"), device=dev)  # end-of-goal (pos m, rot rad) at the current level
        self._ring_i = 0
        self.gen = torch.Generator(device=dev).manual_seed(int(cfg.seed or 0) + 777)
        z7 = lambda: torch.zeros(n, 7, device=dev)  # noqa: E731
        self.q_des, self.q_cmd, self.q_cmd_prev = z7(), z7(), z7()
        self.prev_q, self.prev_qd = z7(), z7()
        self.actions, self.prev_actions = z7(), z7()
        self.raw_excess, self.raw, self.prev_raw = z7(), z7(), z7()
        self.goal_p = torch.zeros(n, 3, device=dev)  # base frame
        self.goal_q = torch.zeros(n, 4, device=dev)
        self.goal_q[:, 0] = 1
        self.goal_t = torch.zeros(n, dtype=torch.long, device=dev)
        self.settle_n = torch.zeros(n, dtype=torch.long, device=dev)
        self.goal_dur = torch.full((n,), self.goal_rows, dtype=torch.long, device=dev)
        self.first_settle = torch.full((n,), -1, dtype=torch.long, device=dev)  # goal row the arm first settled at
        self.payload = torch.zeros(n, device=dev)
        self.w_open = torch.full((n, 1), 0.1, device=dev)
        self.w_cmd = self.w_open.clone()  # finger opening command per env (held_block: the grip, open, or a closed pusher)
        self.held = torch.zeros(n, dtype=torch.bool, device=dev)
        self.regime = torch.zeros(n, device=dev)  # held_block: 1 = Task 4/5 fingers (3 N limit, 42.5 mm command)
        if self.held_on:
            self.block = self.scene["block"]
            self.finger_jids = self.robot.find_joints(["gripper_joint1", "gripper_joint2"], preserve_order=True)[0]
            self.settle_left = torch.zeros(n, dtype=torch.long, device=dev)
            z3 = lambda: torch.zeros(n, 3, device=dev)  # noqa: E731
            self.off0, self.prev_off, self.slip_vec = z3(), z3(), z3()  # block in the pinch frame: settled, last row
            self.rel0 = torch.zeros(n, 4, device=dev)
            self.rel0[:, 0] = 1
            self.prev_rel = self.rel0.clone()
            self.drift = torch.zeros(n, device=dev)
            self.mass_scale = torch.ones(n, device=dev)
            self.bfric = torch.tensor([[0.40, 0.23]], device=dev).repeat(n, 1)
            self.ffric = torch.tensor([[2.4, 2.0]], device=dev).repeat(n, 1)
            self.cos_tilt_held = math.cos(math.radians(hb["max_tilt_deg"]))
            # the pinch orientation in the block frame of a centred grasp (grasp_relation_error's canonical relation; a
            # flat block at yaw 0 has the identity orientation, so it is the grasp orientation itself)
            ref = torch.tensor([[0.48, -0.33, 0.1]], device=dev)
            self.can = gripper_quat(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev), ref, self.arm)[0]
            self.grasp_off = torch.tensor(grasp_cfg()["pinch_offset_in_block_m"], device=dev)
            self.park = torch.tensor(park_pose(0), device=dev)
            self._dyn = None  # PhysX property tensors (CPU), read once
        tr = self.tr = sk.get("track", {})  # P2e: moving goals (a carrot from the pinch to the sampled goal)
        self.track_on = float(tr.get("frac", 0.0)) > 0 or float(tr.get("held_frac", 0.0)) > 0
        self.track_obs = bool(tr.get("obs", False))
        self.trk = torch.zeros(n, dtype=torch.bool, device=dev)  # the current goal moves
        self.tgt_p, self.tgt_q = self.goal_p.clone(), self.goal_q.clone()  # where it is going (base frame)
        self.goal_v = torch.zeros(n, 6, device=dev)  # its velocity this row, base frame (linear, angular)
        if self.track_on:
            from tower_rl.skills.reach_tasks import Carrot

            self.car = Carrot(self.goal_p, self.goal_q, torch.ones(n, device=dev), torch.ones(n, device=dev),
                              float(tr.get("w_rad_s", 1.0)))
        # chunk (Reacher-V2.MD §7.1): rows inside a chunk act on the reacher's own joint prediction, as deployed
        ck = sk.get("chunk", {})
        self.ck_lengths = torch.tensor(ck.get("lengths", [1]), dtype=torch.long, device=dev)
        self.ck_weights = torch.tensor(ck.get("weights", [1.0] * len(self.ck_lengths)), dtype=torch.float32, device=dev)
        self.ck_obs = bool(ck.get("obs", False))
        self.ck_on = self.ck_obs or bool((self.ck_lengths > 1).any())
        self.ck_norm = float(ck.get("k_norm", 16))
        self.ck_k = torch.zeros(n, dtype=torch.long, device=dev)  # rows since the last measured row (0: this one)
        self.ck_len = torch.ones(n, dtype=torch.long, device=dev)  # the current chunk's length (rows)
        self.ck_off = z7()  # q_des - q at the chunk's measured row
        self.ck_w = torch.full((n,), 0.1, device=dev)  # the finger opening measured there
        self.q_obs, self.q_obs_prev = z7(), z7()  # the joints the actor sees this row / saw last row
        self.substep = 0
        self._st = None
        self._log = {"settled": 0, "timeout": 0, "pos_sum": 0.0, "rot_sum": 0.0, "time_sum": 0.0, "ever": 0,
                     "settle_sum": 0.0, "v_rows": 0, "v_over": 0, "h_rows": 0, "h_slip": 0.0, "h_slip1": 0, "h_drops": 0,
                     "h_lost": 0, "h_eps": 0, "hg": 0, "hg_settled": 0, "hg_pos": 0.0, "eg_settled": 0}

    # ------------------------------------------------------------------ helpers
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.gen, device=self.device)

    def _clamp_q(self, q):
        return torch.max(torch.min(q, self.hi), self.lo)

    def keypoints(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """``[..., 3, 3]``: the frame origin moved ``kp_len`` along each of its axes."""
        return p[..., None, :] + self.kp_len * geo.quat_to_mat(q).transpose(-1, -2)

    def _ck_draw(self, m: int) -> torch.Tensor:
        """``m`` chunk lengths from ``chunk.lengths`` / ``chunk.weights``."""
        return self.ck_lengths[torch.multinomial(self.ck_weights, m, replacement=True, generator=self.gen)]

    def _state(self, q: torch.Tensor | None = None, prev: torch.Tensor | None = None) -> dict:
        """Measured arm state after a row and its errors to the goal (base frame); ``q``: joint angles to use instead
        (observation noise, the chunk's prediction), ``prev``: the previous row's (default: the measured ones)."""
        q = self.robot.data.joint_pos[:, :7] if q is None else q
        pb, qb = self.kin.fk(q, "pinch")
        dk = self.keypoints(self.goal_p, self.goal_q) - self.keypoints(pb, qb)
        qd = (q - (self.prev_q if prev is None else prev)) / self.row_dt
        return {"q": q, "pb": pb, "qb": qb, "dk": dk, "d": dk.norm(dim=-1).sum(-1),
                "pos_err": (self.goal_p - pb).norm(dim=-1), "rot_err": geo.quat_angle(qb, self.goal_q),
                "qd": qd, "qdd": (qd - self.prev_qd) / self.row_dt, "v": (pb - self.prev_pb).norm(dim=-1) / self.row_dt}

    # ------------------------------------------------------------------ goals
    def _sample_goals(self, env_ids: torch.Tensor, q0: torch.Tensor):
        """New goals for ``env_ids`` around the joint configuration ``q0``, from the current curriculum region."""
        g, lv = self.sk["goal"], self.levels[self.level]
        r, k = len(env_ids), int(g["candidates"])
        if lv["q_sigma"] > 0:
            qc = q0[:, None] + self._randn(r, k, 7) * lv["q_sigma"]
        else:
            qc = self.lo + self._rand(r, k, 7) * (self.hi - self.lo)
        npri = int(round(k * g["prior_frac"]))
        if npri:
            idx = torch.randint(len(self.postures), (r, npri), generator=self.gen, device=self.device)
            qc[:, :npri] = self.postures[idx] + self._randn(r, npri, 7) * g["prior_sigma"]
        qc = self._clamp_q(qc)
        pb, qb = self.kin.fk(qc.reshape(-1, 7), "pinch")
        pw, qw = self.base.to_world(pb, qb)
        pb, qb, pw, qw = pb.reshape(r, k, 3), qb.reshape(r, k, 4), pw.reshape(r, k, 3), qw.reshape(r, k, 4)
        approach_down = -geo.quat_to_mat(qw)[..., 2, 2] >= self.cos_tilt
        in_ws = ((pw >= self.ws_lo) & (pw <= self.ws_hi)).all(-1)
        p0, r0 = self.kin.fk(q0, "pinch")
        dp = (pb - p0[:, None]).norm(dim=-1)
        dr = geo.quat_angle(qb, r0[:, None].expand(-1, k, -1))
        in_region = (dp <= lv["pos_m"]) & (dr <= math.radians(lv["rot_deg"]))
        moves = (dp >= g["min_move_m"]) | (dr >= math.radians(g["min_move_deg"]))
        valid = approach_down & in_ws & in_region & moves
        held = self.held[env_ids]
        if self.held_on:  # holding a block: high enough for the block to clear the table, pointing nearly down
            ok_h = (pw[..., 2] >= self.hb["min_z_m"]) & (-geo.quat_to_mat(qw)[..., 2, 2] >= self.cos_tilt_held)
            valid &= ~held[:, None] | ok_h
        j = (self._rand(r, k) + 2.0 * valid.float()).argmax(-1)
        ar = torch.arange(r, device=self.device)
        ok = valid[ar, j]
        # no valid candidate: a small move from where the arm is
        qf = self._clamp_q(q0 + self._randn(r, 7) * 0.05)
        pf, rf = self.kin.fk(qf, "pinch")
        self.goal_p[env_ids] = torch.where(ok[:, None], pb[ar, j], pf)
        self.goal_q[env_ids] = torch.where(ok[:, None], qb[ar, j], rf)
        if self.task_frac > 0 and self.level == len(self.levels) - 1:
            # a recorded teacher transit goal instead (top level only: they lie anywhere), jittered by +-pos_m and a yaw
            # about the station z of +-yaw_deg (left pre-push goals point 45 deg off vertical, where random goals are rare);
            # held_block: a held transit's goal while holding, an empty one's otherwise
            use = self._rand(r) < self.task_frac
            for h in (False, True):
                sel = use & (held == h)
                m, pool = int(sel.sum()), self.task_ids[h]
                if m == 0 or len(pool) == 0:
                    continue
                k = pool[torch.randint(len(pool), (m,), generator=self.gen, device=self.device)]
                pw_t = self.task_gp[k] + (2 * self._rand(m, 3) - 1) * self.task_pos_m
                qw_t = geo.quat_mul(geo.quat_from_yaw((2 * self._rand(m) - 1) * self.task_yaw), self.task_gq[k])
                ids = env_ids[sel]
                self.goal_p[ids], self.goal_q[ids] = self.base.to_base(pw_t, qw_t)
        if self.held_on:  # held: a straight rise or descent from here (a carry's via-points), the orientation kept
            vert = held & (self._rand(r) < self.hb["vertical_frac"])
            m = int(vert.sum())
            if m:
                vlo, vhi = self.hb["vertical_m"]
                p0w, q0w = self.base.to_world(p0[vert], r0[vert])
                dz = vlo + (vhi - vlo) * self._rand(m)
                up = self._rand(m) < 0.5
                cap, zmin = float(self.ws_hi[2]) - 0.02, self.hb["min_z_m"]
                up = torch.where(p0w[:, 2] + dz > cap, torch.zeros_like(up),
                                 torch.where(p0w[:, 2] - dz < zmin, torch.ones_like(up), up))
                p1 = p0w.clone()
                p1[:, 2] = torch.where(up, p0w[:, 2] + dz, p0w[:, 2] - dz).clamp(zmin, cap)
                ids = env_ids[vert]
                self.goal_p[ids], self.goal_q[ids] = self.base.to_base(p1, q0w)
        self.goal_t[env_ids] = 0
        self.settle_n[env_ids] = 0
        self.first_settle[env_ids] = -1
        if self.switch == "timer":
            lo = float(g.get("min_frac", 0.5))
            self.goal_dur[env_ids] = (self.goal_rows * (lo + (1.0 - lo) * self._rand(r))).round().long()
            if self.held_on:  # a gentle carry takes longer (the planned transit: 10-12 s to a pre-place pose)
                scale = torch.where(held, float(self.hb.get("goal_time_scale", 1.0)), 1.0)
                self.goal_dur[env_ids] = (self.goal_dur[env_ids].float() * scale).round().long()
        if self.track_on:  # a moving goal: it starts at the pinch and slides to the sampled goal (stepped in _get_rewards)
            tr = self.tr
            mv = self._rand(r) < torch.where(held, float(tr.get("held_frac", 0.0)), float(tr.get("frac", 0.0)))
            self.trk[env_ids] = mv
            self.goal_v[env_ids] = 0
            m = int(mv.sum())
            if m:
                ids = env_ids[mv]
                self.tgt_p[ids], self.tgt_q[ids] = self.goal_p[ids], self.goal_q[ids]
                hm = held[mv]
                vlo = torch.where(hm, float(tr["v_held_m_s"][0]), float(tr["v_m_s"][0]))
                vhi = torch.where(hm, float(tr["v_held_m_s"][1]), float(tr["v_m_s"][1]))
                v = vlo * (vhi / vlo) ** self._rand(m)
                alo, ahi = tr["a_m_s2"]
                self.car.p[ids], self.car.q[ids] = p0[mv], r0[mv]
                self.car.f[ids], self.car.v[ids], self.car.a[ids] = 0.0, v, alo + (ahi - alo) * self._rand(m)
                self.goal_p[ids], self.goal_q[ids] = p0[mv], r0[mv]
                if self.switch == "timer":  # long enough to get there at that speed
                    dist = (self.tgt_p[ids] - p0[mv]).norm(dim=-1)
                    turn = geo.quat_angle(r0[mv], self.tgt_q[ids])
                    need = (torch.maximum(dist / v, turn / self.car.w) + float(tr.get("extra_s", 2.0))) / self.row_dt
                    self.goal_dur[ids] = torch.maximum(self.goal_dur[ids], need.ceil().long())
        sp = self.sk.get("speed", {})
        if sp.get("frac", 0.0) > 0:  # a pinch speed limit for this goal (log-uniform), or none (free_m_s)
            vlo, vhi = sp["range_m_s"]
            lim = vlo * (vhi / vlo) ** self._rand(r)
            self.v_lim[env_ids] = torch.where(self._rand(r) < sp["frac"], lim, self.v_free)
            if self.switch == "timer":  # time to get there at the limit
                need = ((self.goal_p[env_ids] - p0).norm(dim=-1) / self.v_lim[env_ids] + sp.get("extra_s", 2.0)) / self.row_dt
                self.goal_dur[env_ids] = torch.maximum(self.goal_dur[env_ids], need.ceil().long())

    def _record(self, pos_err: torch.Tensor, rot_err: torch.Tensor):
        """End-of-goal errors into the curriculum window; opens the next level when the mean is good enough."""
        cur = self.sk["curriculum"]
        w = self._ring.shape[0]
        m = min(len(pos_err), w)
        if m == 0:
            return
        idx = (self._ring_i + torch.arange(m, device=self.device)) % w
        self._ring[idx] = torch.stack([pos_err[:m], rot_err[:m]], -1)
        self._ring_i = (self._ring_i + m) % w
        if self.level < len(self.levels) - 1 and not bool(torch.isnan(self._ring).any()):
            ring, trim = self._ring, float(cur.get("trim", 0.0))
            if trim > 0:  # the worst goals by error relative to the bar (runaways into a joint limit, 10-30 run 4)
                score = ring[:, 0] / cur["advance"]["pos_m"] + ring[:, 1] / math.radians(cur["advance"]["rot_deg"])
                ring = ring[score.argsort()[: int(round(len(score) * (1.0 - trim)))]]
            mp, mr = ring.mean(0).tolist()
            if mp < cur["advance"]["pos_m"] and mr < math.radians(cur["advance"]["rot_deg"]):
                self.level += 1
                self._ring[:] = float("nan")
                print(f"[reach] curriculum level {self.level} (window mean {mp * 1000:.1f} mm / "
                      f"{math.degrees(mr):.2f} deg)", flush=True)

    # ------------------------------------------------------------------ actions
    def _pre_physics_step(self, actions: torch.Tensor):
        # penalties on the raw action: beyond the clamp every sample executes the same, so a cost on the clamped action
        # gives PPO no gradient (10-30 runs 6-7: joint 1 kept a saturated -1 / +1 dither at action_rate 0.5)
        self.raw_excess = (actions.abs() - 1.0).clamp(min=0)
        self.prev_raw[:] = self.raw
        self.raw[:] = actions
        if self.ck_on:  # this row is spent: the next one is the chunk's next, or a new chunk's measured row
            self.q_obs_prev[:] = self.q_obs
            k = self.ck_k + 1
            new = k >= self.ck_len
            self.ck_k = torch.where(new, torch.zeros_like(k), k)
            if bool(new.any()):
                self.ck_len[new] = self._ck_draw(int(new.sum()))
        a = actions.clamp(-1.0, 1.0)
        if self.held_on:  # held_block: the arm stays still while the fingers settle on the block
            a = torch.where((self.settle_left > 0)[:, None], torch.zeros_like(a), a)
        self.prev_actions[:] = self.actions
        self.actions[:] = a
        v = a * a.abs() if self.amap == "square" else a
        if self.amap == "blend":
            v = self.lin_frac * a + (1.0 - self.lin_frac) * a * a.abs()
        self.q_des = self._clamp_q(self.q_des + v * self.v_max * self.row_dt)
        self.q_cmd_prev[:] = self.q_cmd
        lim = self.kin.limits
        self.q_cmd[:] = torch.max(torch.min(self.kin.sag_compensated_command(self.q_des, self.payload), lim[:, 1]),
                                  lim[:, 0])
        self.substep = 0

    def _apply_action(self):
        q = self.q_cmd_prev if self.substep < self.delay else self.q_cmd
        self.robot.set_joint_position_target(q_full(q, self.w_cmd))
        if self.payload_force:  # world-down weight, rotated into the gripper_base frame every substep
            f = torch.zeros(self.num_envs, 3, device=self.device)
            f[:, 2] = -self.payload * GRAVITY
            fb = geo.quat_apply(geo.quat_conj(self.robot.data.body_quat_w[:, self.gb_id]), f)[:, None]
            self.robot.set_external_force_and_torque(fb, torch.zeros_like(fb), positions=self.pinch_off,
                                                     body_ids=[self.gb_id])
        self.substep += 1

    # ------------------------------------------------------------------ MDP
    def _get_dones(self):
        st = self._st = self._state()
        s = self.sk["goal"]["settle"]
        ok = (st["pos_err"] <= s["pos_m"]) & (st["rot_err"] <= math.radians(s["rot_deg"])) & \
             (st["qd"].abs().amax(-1) <= s["qd"])
        self.settle_n = torch.where(ok, self.settle_n + 1, torch.zeros_like(self.settle_n))
        self.goal_t += 1
        holding = self.settle_n >= s["rows"]
        self.first_settle = torch.where(holding & (self.first_settle < 0), self.goal_t, self.first_settle)
        end = (self.goal_t >= self.goal_dur) if self.switch == "timer" else holding | (self.goal_t >= self.goal_rows)
        st["settled"] = end & holding  # the goal ends now, with the arm settled on it ...
        st["goal_timeout"] = end & ~holding  # ... or not
        st["first_settle"] = self.first_settle.clone()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = st["d"] > self.sk["goal"]["abort_m"]
        if self.held_on:
            hb = self.hb
            pw, qw = self.base.to_world(st["pb"], st["qb"])
            off, rel = self.block_in_pinch(pw, qw)
            w = self.width()
            settling = self.settle_left > 0
            live = self.held & ~settling
            self.slip_vec = torch.where(live[:, None], off - self.prev_off, torch.zeros_like(off))
            self.drift = torch.where(self.held, (off - self.off0).norm(dim=-1), torch.zeros_like(self.drift))
            lost = self.held & ((self.drift > hb["drop_m"]) | (w < hb["drop_width_m"]))
            st.update(pw=pw, off=off, rel=rel, width=w, live=live, slip=self.slip_vec.norm(dim=-1),
                      slip_rot=torch.where(live, geo.quat_angle(self.prev_rel, rel), torch.zeros_like(w)),
                      dropped=lost & ~settling, lost_settling=lost & settling)
            done_settling = self.held & (self.settle_left == 1)  # the reference offset: where the block settled
            self.off0[done_settling], self.rel0[done_settling] = off[done_settling], rel[done_settling]
            self.settle_left = (self.settle_left - 1).clamp(min=0)
            terminated = terminated | st["dropped"]
            # a block the fingers could not hold at rest is not the policy's doing: ends as a time-out (bootstrapped)
            time_out = time_out | st["lost_settling"]
        return terminated, time_out

    # ------------------------------------------------------------------ held block
    def width(self) -> torch.Tensor:
        """Measured finger opening (m), as the evaluator reports it."""
        return self.robot.data.joint_pos[:, 8] - self.robot.data.joint_pos[:, 9]

    def block_in_pinch(self, pw: torch.Tensor, qw: torch.Tensor):
        """The block's pose in the pinch frame (pinch pose ``pw``, ``qw`` env-local): offset ``[n, 3]``, rotation ``[n, 4]``."""
        qc = geo.quat_conj(qw)
        bp = self.block.data.root_link_pos_w - self.scene.env_origins
        return geo.quat_apply(qc, bp - pw), geo.quat_mul(qc, self.block.data.root_link_quat_w)

    def _apply_held_dynamics(self, env_ids: torch.Tensor):
        """Block mass / friction, finger friction and finger effort limit per env (PhysX property tensors read once)."""
        hb, r, dev = self.hb, len(env_ids), self.device
        held = self.held[env_ids]
        ramp = min(1.0, float(self.common_step_counter) / max(1.0, float(hb["ramp_steps"])))
        hard = held & (self._rand(r) < (1.0 - hb["nominal_frac"]) * ramp)
        u = lambda lohi: lohi[0] + (lohi[1] - lohi[0]) * self._rand(r)  # noqa: E731
        mass = torch.where(hard, u(hb["mass_scale"]), torch.ones(r, device=dev))
        fs = torch.where(hard, u(hb["friction_static"]), torch.full((r,), 0.40, device=dev))
        fd = torch.where(hard, fs * u(hb["friction_dynamic_ratio"]), torch.full((r,), 0.23, device=dev))
        ffs = torch.where(hard, u(hb["finger_friction_static"]), torch.full((r,), 2.4, device=dev))
        ffd = torch.where(hard, ffs * u(hb["finger_friction_dynamic_ratio"]), torch.full((r,), 2.0, device=dev))
        self.set_held_dynamics(env_ids, mass, torch.stack([fs, fd], -1), torch.stack([ffs, ffd], -1))

    def set_held_dynamics(self, env_ids: torch.Tensor, mass_scale: torch.Tensor, block_fric: torch.Tensor,
                          finger_fric: torch.Tensor):
        """Write block mass scale ``[k]``, block friction ``[k, 2]`` (static, dynamic) and finger friction ``[k, 2]`` (every
        robot shape, as the station scene does), and the finger effort limit of each env's regime."""
        view, rv = self.block.root_physx_view, self.robot.root_physx_view
        if self._dyn is None:
            m0 = view.get_masses().clone()
            self._dyn = {"m0": m0, "m": m0.clone(), "i0": view.get_inertias().clone(), "bmat": view.get_material_properties().clone(),
                         "rmat": rv.get_material_properties().clone()}
            self._dyn["i"] = self._dyn["i0"].clone()
        d, eid = self._dyn, env_ids.cpu()
        s = mass_scale.cpu()
        d["m"][eid] = d["m0"][eid] * s.reshape(-1, *([1] * (d["m"].dim() - 1)))
        d["i"][eid] = d["i0"][eid] * s.reshape(-1, *([1] * (d["i"].dim() - 1)))
        view.set_masses(d["m"], eid)
        view.set_inertias(d["i"], eid)
        bf, ff = block_fric.cpu(), finger_fric.cpu()
        d["bmat"][eid, :, 0], d["bmat"][eid, :, 1] = bf[:, 0:1], bf[:, 1:2]
        view.set_material_properties(d["bmat"], eid)
        d["rmat"][eid, :, 0], d["rmat"][eid, :, 1] = ff[:, 0:1], ff[:, 1:2]
        rv.set_material_properties(d["rmat"], eid)
        self.mass_scale[env_ids], self.bfric[env_ids], self.ffric[env_ids] = mass_scale, block_fric, finger_fric
        rg = self.hb["regimes"]
        e = torch.where(self.regime[env_ids] > 0.5, float(rg["t45"]["effort_n"]), float(rg["t13"]["effort_n"]))
        self.robot.write_joint_effort_limit_to_sim(e[:, None].expand(-1, len(self.finger_jids)).contiguous(),
                                                   joint_ids=self.finger_jids, env_ids=env_ids)

    def _spawn_held(self, env_ids: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Held envs (``self.held``): the block between the fingers at a sampled grasp offset (the grasp skill's), finger
        command closed; the others: the block on the shelf, fingers open or a closed pusher. Returns the initial opening
        ``[k, 1]`` (held: the block's extent along the jaw axis between the pads + 0.1 mm a side, no initial overlap)."""
        hb, r, dev = self.hb, len(env_ids), self.device
        held = self.held[env_ids]
        pw, qw = self.base.to_world(*self.kin.fk(q, "pinch"))
        o = hb["offset"]
        u = lambda lohi: lohi[0] + (lohi[1] - lohi[0]) * self._rand(r)  # noqa: E731
        dx = torch.where(self._rand(r) < o["wide_frac"], (2 * self._rand(r) - 1) * o["wide_x_mm"], u(o["x_mm"])) / 1000
        dz = u(o["z_mm"]) / 1000
        dq = geo.quat_from_axis_angle(self._randn(r, 3), self._randn(r).abs() * math.radians(o["rot_deg"]))
        bq = geo.normalize(geo.quat_mul(geo.quat_mul(qw, geo.quat_conj(self.can).expand(r, 4)), dq))
        bp = pw - geo.quat_apply(bq, torch.stack([dx, torch.zeros_like(dx), dz], -1) + self.grasp_off)
        rel = geo.quat_to_mat(geo.quat_mul(geo.quat_conj(qw), bq))  # block axes (columns) in the pinch frame
        c = geo.quat_apply(geo.quat_conj(qw), bp - pw)
        half = (BLOCK_HALF[1] * rel[:, 1, 1].abs() + PAD_HALF[0] * rel[:, 1, 0].abs() + PAD_HALF[1] * rel[:, 1, 2].abs()
                + c[:, 1].abs() + 1e-4)
        park = self.park.expand(r, 7)
        pose = torch.where(held[:, None], torch.cat([bp, bq], -1), park).clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        self.block.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        self.block.write_root_com_velocity_to_sim(torch.zeros(r, 6, device=dev), env_ids=env_ids)
        rg = hb["regimes"]
        self.regime[env_ids] = (self._rand(r) < hb["regime_frac"]).float()
        closed = torch.where(self.regime[env_ids] > 0.5, float(rg["t45"]["closed_m"]), float(rg["t13"]["closed_m"]))
        lo, hi = hb["empty_closed_m"]
        pusher = ~held & (self._rand(r) < hb["empty_closed_frac"])
        tool = torch.where(self._rand(r) < 0.5, float(lo), float(hi))
        self.w_cmd[env_ids, 0] = torch.where(held, closed, torch.where(pusher, tool, torch.full_like(closed, 0.1)))
        off, rel_q = self.block_in_pinch_from(pw, qw, bp, bq)
        self.off0[env_ids], self.prev_off[env_ids], self.rel0[env_ids], self.prev_rel[env_ids] = off, off, rel_q, rel_q
        self.slip_vec[env_ids] = 0
        self.drift[env_ids] = 0
        self.settle_left[env_ids] = torch.where(held, int(hb["settle_rows"]), 0)
        return torch.where(held, 2 * half, self.w_cmd[env_ids, 0])[:, None]

    @staticmethod
    def block_in_pinch_from(pw, qw, bp, bq):
        qc = geo.quat_conj(qw)
        return geo.quat_apply(qc, bp - pw), geo.quat_mul(qc, bq)

    def _get_rewards(self):
        st, c = self._st, self.sk["reward"]
        d = st["d"]
        r = c["coarse"] * torch.exp(-c["err_lambda"] * d * d) + c["fine"] * torch.exp(-(d / c["fine_sigma_m"]) ** 2)
        r = r + c.get("mid", 0.0) * torch.exp(-(d / c.get("mid_sigma_m", 0.06)) ** 2) - c.get("linear", 0.0) * d
        r = r - c["effort"] * st["qdd"].norm(dim=-1)
        r = r - c["action_rate"] * (self.raw - self.prev_raw).pow(2).sum(-1)
        r = r - c.get("action_bound", 0.0) * self.raw_excess.sum(-1)
        near = torch.minimum(self.q_des - self.lo, self.hi - self.q_des)
        r = r - c["joint_limit"] * (1.0 - near / 0.1).clamp(min=0).sum(-1)
        pw, _ = self.base.to_world(st["pb"], st["qb"])
        r = r - c["table"] * ((0.015 - pw[:, 2]) / 0.015).clamp(min=0)
        r = r + c["settle_bonus"] * st["settled"].float()
        over = ((st["v"] - self.v_lim) / self.v_lim).clamp(min=0)  # speed.*: relative excess over the goal's limit
        r = r - c.get("speed", 0.0) * over
        # still on the goal: a smooth version of the settle test. hold_wide: the same with wider sigmas, a gradient where
        # the narrow product is ~0 (10-30 run 11 parked at 3 mm / 3 deg, wrist outputs ~0, joints 1 / 3 turning against
        # each other at ~0.05 rad/s: the narrow term paid 0.02 per row there whatever it did)
        for key in ("hold", "hold_wide"):
            if c.get(key, 0.0):
                hs = c[key + "_sigma"]
                r = r + c[key] * torch.exp(-(st["pos_err"] / hs["pos_m"]) ** 2
                                           - (st["rot_err"] / math.radians(hs["rot_deg"])) ** 2
                                           - (st["qd"].abs().amax(-1) / hs["qd"]) ** 2)
        if self.held_on:  # the block's motion in the pinch frame (slip, per row) and its offset from where it settled
            live = st["live"]
            slip_mm = st["slip"] * 1000.0
            r = r - c["slip"] * slip_mm - c["slip_rot"] * torch.rad2deg(st["slip_rot"])
            r = r - c["drift"] * (self.drift * 1000.0 - c["drift_free_mm"]).clamp(min=0) * live.float()
            r = r - c["drop"] * st["dropped"].float()
            r = r - c["table_held"] * (live & (st["pw"][:, 2] < self.hb["min_z_m"] - 0.02)).float()
            lg = self._log
            lg["h_rows"] += int(live.sum())
            lg["h_slip"] += float(slip_mm[live].sum())
            lg["h_slip1"] += int((live & (slip_mm > 1.0)).sum())
            lg["h_drops"] += int(st["dropped"].sum())
            lg["h_lost"] += int(st["lost_settling"].sum())
            self.prev_off[:], self.prev_rel[:] = st["off"], st["rel"]
        # goal transitions (after the reward of the row that finished the goal)
        done = st["settled"] | st["goal_timeout"]
        ids = done.nonzero(as_tuple=True)[0]
        if len(ids):
            self._record(st["pos_err"][ids], st["rot_err"][ids])
            lg = self._log
            lg["settled"] += int(st["settled"].sum())
            lg["timeout"] += int(st["goal_timeout"].sum())
            lg["pos_sum"] += float(st["pos_err"][ids].sum())
            lg["rot_sum"] += float(st["rot_err"][ids].sum())
            lg["time_sum"] += float(self.goal_t[ids].sum()) * self.row_dt
            fs = st["first_settle"][ids]
            lg["ever"] += int((fs >= 0).sum())
            lg["settle_sum"] += float(fs.clamp(min=0).sum()) * self.row_dt
            if self.held_on:  # the same split by held / empty goals
                hd = self.held[ids]
                lg["hg"] += int(hd.sum())
                lg["hg_settled"] += int((st["settled"][ids] & hd).sum())
                lg["hg_pos"] += float(st["pos_err"][ids][hd].sum())
                lg["eg_settled"] += int((st["settled"][ids] & ~hd).sum())
            self._sample_goals(ids, st["q"][ids])
            if self.ck_on:  # a new goal starts with a measured row (the executor: a new node's first call)
                self.ck_k[ids] = 0
                self.ck_len[ids] = self._ck_draw(len(ids))
        if self.track_on:  # moving goals advance one row (new ones included), before the next observation
            pp, pq = self.goal_p.clone(), self.goal_q.clone()
            on = self.car.step(self.tgt_p, self.tgt_q, self.row_dt)
            t = self.trk[:, None]
            self.goal_p[:] = torch.where(t, self.car.p, self.goal_p)
            self.goal_q[:] = torch.where(t, self.car.q, self.goal_q)
            self.goal_v[:] = torch.cat([(self.goal_p - pp) / self.row_dt,
                                        geo.quat_error_rotvec(pq, self.goal_q) / self.row_dt], -1) * t
            moving = self.trk & ~on  # settled counts once the goal stopped
            self.settle_n[moving] = 0
            self.first_settle[moving] = -1
        lg = self._log
        lim = self.v_lim < self.v_free
        lg["v_rows"] += int(lim.sum())
        lg["v_over"] += int((lim & (over > 0.1)).sum())
        nd = lg["settled"] + lg["timeout"]
        if self.common_step_counter % 24 == 0 and nd > 0:
            self.extras["log"] = {"reach/level": float(self.level), "reach/settled_frac": lg["settled"] / nd,
                                  "reach/end_pos_mm": lg["pos_sum"] / nd * 1000.0,
                                  "reach/end_rot_deg": math.degrees(lg["rot_sum"] / nd),
                                  "reach/goal_time_s": lg["time_sum"] / nd, "reach/goals": float(nd),
                                  "reach/ever_settled_frac": lg["ever"] / nd,
                                  "reach/settle_time_s": lg["settle_sum"] / max(lg["ever"], 1),
                                  "reach/speed_over_frac": lg["v_over"] / max(lg["v_rows"], 1)}  # limited rows > 1.1x
            if self.held_on:  # per held episode started in the window; slip per held row
                self.extras["log"] |= {"held/drops_per_ep": lg["h_drops"] / max(lg["h_eps"], 1),
                                       "held/lost_settling_per_ep": lg["h_lost"] / max(lg["h_eps"], 1),
                                       "held/slip_mm_per_row": lg["h_slip"] / max(lg["h_rows"], 1),
                                       "held/slip_gt1mm_frac": lg["h_slip1"] / max(lg["h_rows"], 1),
                                       "held/goal_settled_frac": lg["hg_settled"] / max(lg["hg"], 1),
                                       "held/goal_end_pos_mm": lg["hg_pos"] / max(lg["hg"], 1) * 1000.0,
                                       "held/empty_goal_settled_frac": lg["eg_settled"] / max(nd - lg["hg"], 1),
                                       "held/hard_frac": min(1.0, float(self.common_step_counter) / max(1.0, float(self.hb["ramp_steps"])))
                                       * (1.0 - self.hb["nominal_frac"])}
            self._log = {"settled": 0, "timeout": 0, "pos_sum": 0.0, "rot_sum": 0.0, "time_sum": 0.0, "ever": 0,
                         "settle_sum": 0.0, "v_rows": 0, "v_over": 0, "h_rows": 0, "h_slip": 0.0, "h_slip1": 0, "h_drops": 0,
                         "h_lost": 0, "h_eps": 0, "hg": 0, "hg_settled": 0, "hg_pos": 0.0, "eg_settled": 0}
        return r

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        r = len(env_ids)
        # start at a demo posture or home (+ noise), or anywhere within the limits with the pinch above the table
        pi = torch.randint(len(self.postures), (r,), generator=self.gen, device=self.device)
        q = self.postures[pi] + self._randn(r, 7) * self.sk["reset"]["posture_sigma"]
        rs = self.sk["reset"]
        anywhere = self._rand(r) < rs["uniform_frac"]
        if rs.get("uniform", "joint") == "workspace":
            # anywhere the goals can be (pinch in the workspace, pointing down): from joint-uniform starts 27 / 28 first
            # goals ran away, mostly sideways / upward pinches outside the goal distribution (10-30 run 11, level 2)
            k = 2 * int(self.sk["goal"]["candidates"])
            qc = self.lo + self._rand(r, k, 7) * (self.hi - self.lo)
            pb, qb = self.kin.fk(qc.reshape(-1, 7), "pinch")
            pw, qw = self.base.to_world(pb, qb)
            pw, qw = pw.reshape(r, k, 3), qw.reshape(r, k, 4)
            valid = (-geo.quat_to_mat(qw)[..., 2, 2] >= self.cos_tilt) & ((pw >= self.ws_lo) & (pw <= self.ws_hi)).all(-1)
            j = (self._rand(r, k) + 2.0 * valid.float()).argmax(-1)
            ar = torch.arange(r, device=self.device)
            anywhere &= valid[ar, j]  # no valid candidate: the posture start
            qu = qc[ar, j]
        else:
            qu = self.lo + self._rand(r, 7) * (self.hi - self.lo)
        q = torch.where(anywhere[:, None], qu, q)
        if self.task_start_frac > 0:  # where a recorded teacher transit started (+ 0.02 rad noise)
            ts = self._rand(r) < self.task_start_frac
            k = torch.randint(len(self.task_q), (r,), generator=self.gen, device=self.device)
            q = torch.where(ts[:, None], self.task_q[k] + self._randn(r, 7) * 0.02, q)
        q = self._clamp_q(q)
        pb, qb = self.kin.fk(q, "pinch")
        pw, _ = self.base.to_world(pb, qb)
        low = pw[:, 2] < 0.05
        q = torch.where(low[:, None], self.postures[0].expand(r, -1), q)
        w_init = self.w_open[env_ids]
        p = self.sk.get("payload", {})
        self.payload[env_ids] = torch.where(self._rand(r) < p.get("frac", 0.0), float(p.get("kg", 0.1056)), 0.0)
        if self.held_on:
            hb = self.hb
            held = self._rand(r) < hb["frac"]
            self.held[env_ids] = held
            # held starts: where a recorded held transit started (+ 0.02 rad), or the start above if the block hangs clear
            # of the table there and the approach points nearly down; else a recorded held start
            pool = self.task_ids[True]
            if len(pool):
                k = pool[torch.randint(len(pool), (r,), generator=self.gen, device=self.device)]
                qt = self._clamp_q(self.task_q[k] + self._randn(r, 7) * 0.02)
                q = torch.where((held & (self._rand(r) < hb["start_tasks"]))[:, None], qt, q)
                pw, qw = self.base.to_world(*self.kin.fk(q, "pinch"))
                bad = held & ((pw[:, 2] < hb["min_z_m"]) | (-geo.quat_to_mat(qw)[:, 2, 2] < self.cos_tilt_held))
                q = torch.where(bad[:, None], qt, q)
            w_init = self._spawn_held(env_ids, q)
            self._apply_held_dynamics(env_ids)
            self.payload[env_ids] = torch.where(held, float(p.get("kg", 0.1056)), 0.0)  # nominal mass, as the executor
            self._log["h_eps"] += int(held.sum())
        self.robot.write_joint_state_to_sim(q_full(q, w_init), torch.zeros(r, 10, device=self.device), env_ids=env_ids)
        self.prev_pb[env_ids] = self.kin.fk(q, "pinch")[0]
        lim = self.kin.limits
        q_cmd = torch.max(torch.min(self.kin.sag_compensated_command(q, self.payload[env_ids]), lim[:, 1]), lim[:, 0])
        self.robot.set_joint_position_target(q_full(q_cmd, self.w_cmd[env_ids]), env_ids=env_ids)
        self.q_des[env_ids] = q
        self.q_cmd[env_ids] = q_cmd
        self.q_cmd_prev[env_ids] = q_cmd
        self.prev_q[env_ids] = q
        if self.ck_on:
            self.ck_k[env_ids] = 0
            self.ck_len[env_ids] = self._ck_draw(r)
            self.ck_off[env_ids] = 0
            self.q_obs[env_ids], self.q_obs_prev[env_ids] = q, q
        self.prev_qd[env_ids] = 0
        self.actions[env_ids] = 0
        self.prev_actions[env_ids] = 0
        self.raw[env_ids] = 0
        self.prev_raw[env_ids] = 0
        self._sample_goals(env_ids, q)

    def _get_observations(self):
        st = st_true = self._state()
        q_true, qd_true, pb_true, qb_true = st["q"], st["qd"], st["pb"], st["qb"]
        if self.q_noise > 0:  # rand.q_noise_rad: the policy sees noisy joint angles (rewards use the true ones)
            st = self._state(q_true + self._randn(self.num_envs, 7) * self.q_noise)
        width = self.width()
        if self.ck_on:  # chunk: measured joints on a chunk's first row; inside it q_des less the offset q_des - q measured
            # there (sag, lag), the opening measured there, and qd from the previous row's view (reach_policy.Reacher)
            meas = self.ck_k == 0
            self.ck_off = torch.where(meas[:, None], self.q_des - st["q"], self.ck_off)
            self.ck_w = torch.where(meas, width, self.ck_w)
            self.q_obs = torch.where(meas[:, None], st["q"], self.q_des - self.ck_off)
            st, width = self._state(self.q_obs, prev=self.q_obs_prev), self.ck_w
        q, qd = st["q"], st["qd"]
        actor = torch.cat([
            st["dk"].reshape(-1, 9),
            (q - self.q_mid) / self.q_half, qd, self.q_des - q, self.actions,
            st["pb"], rot6d_mat(geo.quat_to_mat(st["qb"])),
            self.goal_p, rot6d_mat(geo.quat_to_mat(self.goal_q)),
        ] + ([torch.tanh(st["dk"].reshape(-1, 9) / self.fine_m)] if self.fine_m > 0 else [])
          + ([self.v_lim[:, None] / self.v_free] if self.speed_obs else [])
          + ([((width - 0.05) / 0.05)[:, None], self.regime[:, None]] if self.held_obs else [])
          + ([self.goal_v] if self.track_obs else [])
          + ([(self.ck_k.float() / self.ck_norm)[:, None]] if self.ck_obs else []), dim=-1)
        chunk_critic = [(q_true - self.q_mid) / self.q_half, st_true["dk"].reshape(-1, 9)] if self.ck_obs else []
        held_critic = []
        if self.held_obs:  # privileged: the block in the pinch frame, its motion this row, drift, dynamics
            off, rel = self.block_in_pinch(*self.base.to_world(pb_true, qb_true))
            held_critic = [off, rot6d_mat(geo.quat_to_mat(rel)), self.slip_vec, self.drift[:, None], self.mass_scale[:, None],
                           self.bfric, self.ffric, self.held.float()[:, None], (self.settle_left > 0).float()[:, None]]
        critic = torch.cat([
            actor, *chunk_critic, *held_critic, self.robot.data.joint_vel[:, :7],
            (self.goal_t.float() / self.goal_rows)[:, None],
            torch.full((self.num_envs, 1), float(self.level), device=self.device),
            self.payload[:, None],
        ], dim=-1)
        self.prev_q[:] = q_true
        self.prev_qd[:] = qd_true
        self.prev_pb[:] = pb_true
        return {"policy": torch.nan_to_num(actor), "critic": torch.nan_to_num(critic)}
