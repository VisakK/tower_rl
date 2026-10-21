"""place_release skill env (Step 6): from above the goal with the block already grasped, set it down, release, retreat.

Reset: the stage's earlier transfers already sit at their goals (placement error +-3 mm / 2 deg); the held block is
in the closed gripper with a slip offset; the pinch starts 1-6 cm above the release pose (reverse curriculum) with
tight / wide perturbations. Success = the settle check (build file), with the evaluator's P semantics
(``tower_rl.eval.operations``): after the release the block has stayed within 5 mm / 5 deg of its goal (z within
10 mm of the layer's rest height), released (gripper open) and static (< 1 cm/s) for 1.0 s *continuously*, every placed
block moved < 10 mm, and the pinch has risen >= 3 cm. Failure (penalty, episode ends): the block comes to rest out of
tolerance >= 1 s after the release, no success within 3 s of the release, a placed block disturbed, or the block
dropped. (An earlier version checked "every block static" at exactly 1.0 s after the release; GPU contact jitter of
other blocks and the still-retreating fingers failed ~16 % of a scripted oracle's accurate placements.)

Actor observation (73, base frame; ``tower_rl.skills.obs.place_release_actor_obs``): pinch pose / velocity / target
lead (21), width, gripper request, width command (3), goal in the pinch frame (9), held block in the pinch frame (9),
block-to-goal error (5), two nearest placed blocks relative to the goal (12), goal position (3), previous action (7),
phase one-hot (holding / released / retreated) (3), progress (1).
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from tower_rl import geometry as geo
from tower_rl.grasp import cfg as grasp_cfg
from tower_rl.postures import PADS, priors
from tower_rl.sim.assets import q_full
from tower_rl.sim.skill_scene import MAX_MOVABLE
from tower_rl.skills.base_env import SkillEnvBase, SkillEnvCfg, squash
from tower_rl.skills.obs import grasp_target_world, place_release_actor_obs, rot6d
from tower_rl.tower import foundation_poses, goal_poses_with_error

ACTOR_DIM = 73
CRITIC_DIM = ACTOR_DIM + 9 + MAX_MOVABLE * 5 + MAX_MOVABLE + 2 + 1 + 5 + 7 + 7 + 7 + 1 + 1 + 1 + 1
BLOCK_W = 0.0497


@configclass
class PlaceReleaseEnvCfg(SkillEnvCfg):
    skill_yaml: str = "skills/place_release"
    action_space: int = 7
    observation_space: int = ACTOR_DIM
    state_space: int = CRITIC_DIM


class PlaceReleaseEnv(SkillEnvBase):
    cfg: PlaceReleaseEnvCfg

    def __init__(self, cfg: PlaceReleaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self.tilt_deg = grasp_cfg()["approach_tilt_deg"][self.arm]
        self.release_clear = grasp_cfg()["release_clearance_m"]
        pr = priors()[self.arm]["release"]
        self.seed_table = {(pad, int(yc)): torch.tensor(v["q"], device=dev) for pad, d in pr.items() for yc, v in d.items()}
        self.pad_xy = torch.tensor([PADS["A"], PADS["B"]], device=dev)
        self.goal_pos = torch.zeros(n, 3, device=dev)
        self.goal_quat = torch.zeros(n, 4, device=dev)
        self.placed = torch.zeros(n, MAX_MOVABLE, dtype=torch.bool, device=dev)
        self.released = torch.zeros(n, dtype=torch.bool, device=dev)
        self.since_release = torch.zeros(n, dtype=torch.long, device=dev)
        self.settled = torch.zeros(n, dtype=torch.long, device=dev)  # consecutive steps released + in tolerance + static
        self.prev_tpos = torch.zeros(n, 3, device=dev)  # target block position one step ago (static = displacement / step)
        self.z_release = torch.zeros(n, device=dev)
        self.noise_goal = torch.zeros(n, 3, device=dev)
        self.noise_goal_yaw = torch.zeros(n, device=dev)
        kp = [[0, 0, 0], [0.06, 0, 0], [-0.06, 0, 0], [0, 0.02, 0]]
        self.kp = torch.tensor(kp, dtype=torch.float32, device=dev)
        self.flip = geo.quat_from_yaw(torch.tensor([math.pi], device=dev))

    # ------------------------------------------------------------------ reset
    def _seed(self, pos, quat):
        pad = torch.cdist(pos[:, :2], self.pad_xy).argmin(-1)
        y = torch.remainder(torch.rad2deg(geo.yaw_from_quat(quat)) + 45, 180) - 45
        ycls = torch.where(y.abs() < 45, 0, 90)
        out = torch.zeros(len(pos), 7, device=self.device)
        for (p, yc), q in self.seed_table.items():
            out[(pad == (0 if p == "A" else 1)) & (ycls == yc)] = q
        miss = out.abs().sum(-1) == 0
        out[miss] = next(iter(self.seed_table.values()))
        return out

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        r, dev = len(env_ids), self.device
        rs = self.sk["reset"]
        pi = torch.randint(len(self.pairs), (r,), generator=self.gen, device=dev)
        self.env_variant[env_ids] = self.pair_v[pi]
        self.env_stage[env_ids] = self.pair_k[pi]
        tier = torch.multinomial(self.tier_p, r, replacement=True, generator=self.gen)
        doms = {t: self.samplers[t].sample(r, self.gen) for t in self.tier_names}
        mov = self.park[None].repeat(r, 1, 1)
        placed = torch.zeros(r, MAX_MOVABLE, dtype=torch.bool, device=dev)
        held = torch.zeros(r, dtype=torch.long, device=dev)
        goal = torch.zeros(r, 7, device=dev)
        found = torch.zeros(r, 6, 7, device=dev)
        mass = torch.ones(r, MAX_MOVABLE, device=dev)
        bfric = torch.zeros(r, 2, device=dev)
        ffric = torch.zeros(r, 2, device=dev)
        for vi, spec in enumerate(self.specs):
            sel = (self.env_variant[env_ids] == vi).nonzero(as_tuple=True)[0]
            if len(sel) == 0:
                continue
            stage = self.env_stage[env_ids][sel]
            gpe = goal_poses_with_error(spec, self.rz, len(sel), rs["placed_xy_err_m"], rs["placed_yaw_err_deg"], self.gen)
            for k in range(len(spec.transfer_arm)):
                b = int(spec.transfer_block[k])
                done = stage > k
                mov[sel[done], b] = gpe[done, k]
                placed[sel[done], b] = True
                cur = stage == k
                held[sel[cur]] = b
                g = torch.cat([spec.goal[k, :2], self.rz[spec.transfer_dest_layer[k]][None], spec.goal[k, 3:7]])
                goal[sel[cur]] = g
            found[sel] = foundation_poses(spec, len(sel), jitter_m=0.003, gen=self.gen)
            for ti, t in enumerate(self.tier_names):
                s2 = sel[tier[sel] == ti]
                if len(s2):
                    d = doms[t].subset(s2)
                    for j, e in enumerate(env_ids[s2].tolist()):
                        self.env_case[e] = f"{t}/{d.case[j]}"
                    m = spec.num_movable
                    mass[s2, :m] = d.mass_scale[:, :m]
                    bfric[s2] = d.friction[:, 0]
                    ffric[s2] = d.finger_friction
        # pinch start: canonical grasp of a virtual block at the goal, raised and perturbed
        gpos, gq = goal[:, :3], goal[:, 3:7]
        pp, pq = grasp_target_world(gpos, gq, self.arm, self.tilt_deg)
        t_pb = geo.frame_relative(pp, pq, gpos, gq)  # block in the pinch frame (centred grasp)
        slip = torch.zeros(r, 3, device=dev)
        slip[:, 0] = self._randn(r) * rs["grasp_slip_m"]
        slip[:, 2] = self._randn(r) * rs["grasp_slip_m"] * 0.5
        srot = geo.quat_from_axis_angle(torch.tensor([0.0, 1.0, 0.0], device=dev).expand(r, 3),
                                        self._randn(r) * math.radians(rs["grasp_slip_deg"]))
        wide_cfg = rs.get("grasp_offset_wide")
        if wide_cfg:  # held offsets as wide as the grasp skill's quality gate admits (uniform)
            wide_g = self._rand(r) < wide_cfg["frac"]
            u = lambda a: (self._rand(r) * 2 - 1) * a  # noqa: E731
            wslip = torch.stack([u(wide_cfg["x_m"]), torch.zeros(r, device=dev), u(wide_cfg["z_m"])], -1)
            # the pads keep the block's side faces parallel: it pivots about the jaw axis (pinch y), rolls little
            ex, ey = (torch.tensor([v], device=dev).expand(r, 3) for v in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
            rmax = math.radians(wide_cfg["rot_deg"])
            wrot = geo.quat_mul(geo.quat_from_axis_angle(ey, u(rmax)), geo.quat_from_axis_angle(ex, u(rmax / 3)))
            slip = torch.where(wide_g[:, None], wslip, slip)
            srot = torch.where(wide_g[:, None], wrot, srot)
        sp = t_pb[0] + slip
        sq = geo.quat_mul(srot, t_pb[1])
        h = rs["start_height_m"][0] + (rs["start_height_m"][1] - rs["start_height_m"][0]) * self._rand(r)
        wide = self._rand(r) > rs["tight"]["frac"]
        rad = torch.where(wide, rs["wide"]["pos_m"], rs["tight"]["pos_m"])
        rot = torch.deg2rad(torch.where(wide, torch.tensor(rs["wide"]["rot_deg"], device=dev), torch.tensor(rs["tight"]["rot_deg"], device=dev)))
        u = self._randn(r, 3)
        u[:, 2] = u[:, 2].abs() * 0.3
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9) * (self._rand(r, 1) ** (1 / 3)) * rad[:, None]
        p0 = pp.clone()
        p0[:, 2] += self.release_clear + h
        p0 = p0 + u
        q0 = geo.normalize(geo.quat_mul(geo.quat_from_axis_angle(self._randn(r, 3), (self._rand(r) * 2 - 1) * rot), pq))
        seed = self._seed(gpos, gq)
        pb, qb = self.base.to_base(p0, q0)
        q_des, pe, re = self.kin.ik(seed, pb, qb, iters=30, q_ref=seed)
        bad = (pe > 0.005) | (re > math.radians(3))
        if bad.any():
            p1 = pp.clone()
            p1[:, 2] += self.release_clear + 0.03
            pb1, qb1 = self.base.to_base(p1[bad], pq[bad])
            q_des[bad] = self.kin.ik(seed[bad], pb1, qb1, iters=30, q_ref=seed[bad])[0]
        # the block where the (IK-achieved) pinch holds it
        pa_b, qa_b = self.kin.fk(q_des, "pinch")
        pa, qa = self.base.to_world(pa_b, qa_b)
        bp, bq = geo.frame_compose(pa, qa, sp, sq)
        ar = torch.arange(r, device=dev)
        mov[ar, held, :3] = bp
        mov[ar, held, 3:7] = bq
        self.prev_tpos[env_ids] = bp
        poses = torch.zeros(r, len(self.blocks.object_names), 7, device=dev)
        poses[:, self.found_idx] = found
        poses[:, self.slot_idx] = mov
        world = poses.clone()
        world[..., :3] += self.scene.env_origins[env_ids][:, None]
        self.blocks.write_object_link_pose_to_sim(world, env_ids=env_ids)
        self.blocks.write_object_com_velocity_to_sim(torch.zeros(r, MAX_MOVABLE, 6, device=dev), env_ids=env_ids, object_ids=self.slot_idx)
        self.present[env_ids] = placed.clone()
        self.present[env_ids, held] = True
        self.placed[env_ids] = placed
        self.target_slot[env_ids] = held
        self.block_p0[env_ids] = mov[..., :3]
        self.goal_pos[env_ids], self.goal_quat[env_ids] = gpos, gq
        self.mass_scale[env_ids], self.block_fric[env_ids], self.finger_fric[env_ids] = mass, bfric, ffric
        on = self.sk["obs_noise"]
        self.noise_pos[env_ids] = self._randn(r, MAX_MOVABLE, 3) * on["episode_pos_m"]
        self.noise_pos[env_ids, :, 2] *= 0.5
        self.noise_yaw[env_ids] = self._randn(r, MAX_MOVABLE) * math.radians(on["episode_yaw_deg"])
        self.noise_goal[env_ids] = self._randn(r, 3) * on["episode_pos_m"] * torch.tensor([1.0, 1.0, 0.5], device=dev)
        self.noise_goal_yaw[env_ids] = self._randn(r) * math.radians(on["episode_yaw_deg"])
        # optional error shared by every perceived pose of the episode, the goal included (an estimator- or
        # calibration-level bias): absolute positions are then off while block-to-goal relations are not
        sp, sy = on.get("shared_pos_m", 0.0), on.get("shared_yaw_deg", 0.0)
        if sp > 0 or sy > 0:
            sh = self._randn(r, 3) * sp * torch.tensor([1.0, 1.0, 0.5], device=dev)
            self.noise_pos[env_ids] += sh[:, None]
            self.noise_goal[env_ids] += sh
            shy = self._randn(r) * math.radians(sy)
            self.noise_yaw[env_ids] += shy[:, None]
            self.noise_goal_yaw[env_ids] += shy
        self._apply_dynamics(env_ids)
        # arm: holding (fingers on the block faces, closed command, payload compensated)
        self._set_arm(env_ids, q_des)
        wq = torch.full((r, 1), BLOCK_W, device=dev)
        self.robot.write_joint_state_to_sim(q_full(q_des, wq), torch.zeros(r, 10, device=dev), env_ids=env_ids)
        cw = self.grip.cfg.closed_width
        q_cmd = self.kin.sag_compensated_command(q_des, torch.full((r,), self.ctrl.cfg.payload_mass, device=dev))
        lim = self.kin.limits
        q_cmd = torch.max(torch.min(q_cmd, lim[:, 1]), lim[:, 0])
        self.q_cmd[env_ids] = q_cmd
        self.q_cmd_prev[env_ids] = q_cmd
        self.w_cmd[env_ids] = cw
        self.w_cmd_prev[env_ids] = cw
        self.grip.reset(env_ids, width=cw, closed=True)
        self.robot.set_joint_position_target(q_full(q_cmd, torch.full((r, 1), cw, device=dev)), env_ids=env_ids)
        self.actions[env_ids, 6] = 1.0  # "holding" is the previous request
        self.released[env_ids] = False
        self.since_release[env_ids] = 0
        self.settled[env_ids] = 0
        self.z_release[env_ids] = 0

    # ------------------------------------------------------------------ actions
    def _action_scale(self):
        ac = self.sk["action"]
        npos, _ = self.noisy_block_poses()
        ar = torch.arange(self.num_envs, device=self.device)
        d = (npos[ar, self.target_slot] - (self.goal_pos + self.noise_goal)).norm(dim=-1)
        a = ((d - ac["near_radius_m"]) / (ac["far_radius_m"] - ac["near_radius_m"])).clamp(0, 1)
        return ac["pos_near_m"] + (ac["pos_far_m"] - ac["pos_near_m"]) * a

    def _payload_kg(self):
        return self.grip.closed.float() * self.ctrl.cfg.payload_mass

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        q, _ = self.measured()
        pb, _ = self.kin.fk(q, "pinch")
        pw = self.base.to_world(pb, self.prev_pinch_qb)[0]
        newly = (~self.released) & (~self.grip.closed)
        self.z_release = torch.where(newly, pw[:, 2], self.z_release)
        self.released |= newly

    # ------------------------------------------------------------------ state, dones, rewards
    def _compute_state(self):
        n, dev = self.num_envs, self.device
        ar = torch.arange(n, device=dev)
        q, w = self.measured()
        pb, qb, pw, qw = self.pinch(q)
        bpos, bquat = self.block_poses_local()
        tpos, tquat = bpos[ar, self.target_slot], bquat[ar, self.target_slot]
        # block keypoints vs goal keypoints (goal or its pi-flip, whichever is closer)
        kpb = tpos[:, None] + geo.quat_apply(tquat[:, None].expand(n, 4, 4), self.kp[None].expand(n, 4, 3))
        g1 = self.goal_quat
        g2 = geo.quat_mul(self.goal_quat, self.flip.expand(n, 4))
        d = []
        for g in (g1, g2):
            kpg = self.goal_pos[:, None] + geo.quat_apply(g[:, None].expand(n, 4, 4), self.kp[None].expand(n, 4, 3))
            d.append((kpb - kpg).norm(dim=-1).mean(-1))
        kp_d = torch.minimum(d[0], d[1])
        dxy = (tpos[:, :2] - self.goal_pos[:, :2]).norm(dim=-1)
        dz = tpos[:, 2] - self.goal_pos[:, 2]
        dyaw = geo.yaw_diff_sym(geo.yaw_from_quat(tquat), geo.yaw_from_quat(self.goal_quat))
        speed = self.blocks.data.object_link_lin_vel_w[:, self.slot_idx].norm(dim=-1)
        disp = (bpos - self.block_p0).norm(dim=-1)
        disp_pl = torch.where(self.placed, disp, torch.zeros_like(disp))
        tspeed = speed[ar, self.target_slot]
        f = self.finger_forces(self.target_slot)
        self.st = dict(q=q, w=w, pb=pb, qb=qb, pw=pw, qw=qw, bpos=bpos, bquat=bquat, tpos=tpos, tquat=tquat, kp_d=kp_d,
                       dxy=dxy, dz=dz, dyaw=dyaw, disp_pl=disp_pl, tspeed=tspeed, f=f, tilt=geo.tilt_angle(tquat))
        return self.st

    def _get_dones(self):
        s = self._compute_state()
        pl, tm = self.sk["place"], self.sk["termination"]
        self.since_release = torch.where(self.released, self.since_release + 1, self.since_release)
        in_tol = (s["dxy"] <= pl["tol_pos_m"]) & (s["dz"].abs() <= pl["tol_z_m"]) & (s["dyaw"] <= math.radians(pl["tol_yaw_deg"]))
        undisturbed = (s["disp_pl"] < tm["disturb_mm"] / 1000).all(-1)
        # static from the displacement over the last step (0.1 s): GPU TGS reports ~1-2 cm/s velocity jitter for blocks
        # resting on dynamic blocks that do not move (the CPU evaluator does not); the mean speed is what matters
        s["tvel"] = (s["tpos"] - self.prev_tpos).norm(dim=-1) / self.step_dt
        self.prev_tpos = s["tpos"].clone()
        static = s["tvel"] < pl["static_speed"]
        # the evaluator's P: released (no closed gripper on it), in tolerance and static, continuously for settle_s
        self.settled = torch.where(self.released & ~self.grip.closed & in_tol & static, self.settled + 1, torch.zeros_like(self.settled))
        retreated = s["pw"][:, 2] >= self.z_release + pl["retreat_m"]
        success = (self.settled >= pl["settle_steps"]) & retreated & undisturbed
        failed = self.released & (self.since_release >= pl["settle_steps"]) & static & ~in_tol  # came to rest off its goal
        failed |= self.released & (self.since_release >= 3 * pl["settle_steps"]) & ~success   # never settled / retreated
        dropped = (s["dz"] < -tm["drop_below_goal_m"]) | (s["tilt"] > math.radians(tm["tilt_deg"]))
        disturbed = ~undisturbed
        self.term = dict(success=success, failed=(failed | dropped) & ~success, disturbed=disturbed & ~success)
        terminated = success | failed | dropped | disturbed
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        done = terminated | time_out
        self.extras.pop("log", None)
        if done.any():
            dn = done.float().sum()
            self.extras["log"] = {
                "success_rate": float((success & done).float().sum() / dn),
                "failed_rate": float((self.term["failed"] & done).float().sum() / dn),
                "disturbed_rate": float((self.term["disturbed"] & done).float().sum() / dn),
                "timeout_rate": float((time_out & ~terminated & done).float().sum() / dn),
                "place_xy_mm": float(s["dxy"][done].mean() * 1000),
            }
        return terminated, time_out

    def _get_rewards(self):
        s = self.st
        rw = self.sk["reward"]
        r_track = squash(s["kp_d"], 50.0, 2.0) + squash(s["kp_d"], 100.0, 0.0)
        aligned = (s["dxy"] < 0.005) & (s["dyaw"] < math.radians(5))
        r_desc = aligned.float() * (1 - (s["dz"] / 0.05)).clamp(0, 1)
        r_retreat = self.released.float() * ((s["pw"][:, 2] - self.z_release) / self.sk["place"]["retreat_m"]).clamp(0, 1)
        pen_d = ((s["disp_pl"] - 0.003).clamp(min=0) / 0.01).sum(-1)
        lim = self.kin.limits
        margin = torch.minimum(s["q"] - lim[:, 0], lim[:, 1] - s["q"])
        pen_jl = ((0.1 - margin) / 0.1).clamp(0, 1).sum(-1)
        rate = (self.actions - self.prev_actions).pow(2).sum(-1)
        remaining = (self.max_episode_length - self.episode_length_buf).float()
        r = (rw["track"] * r_track + rw["descend"] * r_desc + r_retreat - rw["disturb_per_cm"] * pen_d
             - rw["action_rate"] * rate - rw["joint_limit"] * pen_jl)
        # precision bonus at success: a placement that just passes moves out of tolerance when the next layer lands on it
        prec = (1 - s["dxy"] / self.sk["place"]["tol_pos_m"]).clamp(0, 1)
        r = r + self.term["success"].float() * (rw["success"] + rw["success_per_remaining_step"] * remaining
                                                + rw.get("success_precision", 0.0) * prec)
        r = r - self.term["failed"].float() * rw["fail"] - self.term["disturbed"].float() * rw["disturb_terminal"]
        return r

    # ------------------------------------------------------------------ observations
    def _get_observations(self):
        s = self._compute_state()
        n, dev = self.num_envs, self.device
        ar = torch.arange(n, device=dev)
        pb, qb = s["pb"], s["qb"]
        tgt_pb, tgt_qb = self.base.to_base(self.ctrl.pos, self.ctrl.quat)
        npos, nquat = self.noisy_block_poses()
        npb, nqb = self.base.to_base(npos, nquat)
        gq_noisy = geo.quat_mul(geo.quat_from_yaw(self.noise_goal_yaw), self.goal_quat)
        gpb, gqb = self.base.to_base(self.goal_pos + self.noise_goal, gq_noisy)
        valid = self.placed.clone()
        retreated = s["pw"][:, 2] >= self.z_release + self.sk["place"]["retreat_m"]
        phase = torch.stack([~self.released, self.released & ~retreated, self.released & retreated], -1).float()
        actor = place_release_actor_obs(
            pb=pb, qb=qb, prev_pb=self.prev_pinch_b, prev_qb=self.prev_pinch_qb, dt=self.step_dt, tgt_pb=tgt_pb, tgt_qb=tgt_qb,
            width=s["w"], closed=self.grip.closed, width_cmd=self.grip.width, goal_pb=gpb, goal_qb=gqb,
            block_pb=npb[ar, self.target_slot], block_qb=nqb[ar, self.target_slot], other_pb=npb, other_qb=nqb,
            other_valid=valid, phase=phase, prev_actions=self.actions,
            progress=self.episode_length_buf.float() / self.max_episode_length)
        self.prev_pinch_b, self.prev_pinch_qb = pb.clone(), qb.clone()
        tp_b, tq_b = self.base.to_base(s["tpos"], s["tquat"])
        g_b, gq_b = self.base.to_base(self.goal_pos, self.goal_quat)
        trel_p, trel_q = geo.frame_relative(g_b, gq_b, tp_b, tq_b)
        allp, allq = self.base.to_base(s["bpos"], s["bquat"])
        arel_p, arel_q = geo.frame_relative(g_b[:, None].expand_as(allp), gq_b[:, None].expand_as(allq), allp, allq)
        ayaw = geo.yaw_from_quat(arel_q)
        slots = torch.cat([arel_p, torch.cos(2 * ayaw)[..., None], torch.sin(2 * ayaw)[..., None]], -1).reshape(n, -1)
        other = self.finger_any_other(self.target_slot)
        dyn = torch.cat([self.mass_scale[ar, self.target_slot][:, None], self.block_fric, self.finger_fric], -1)
        jv = self.robot.data.joint_vel[:, :7]
        critic = torch.cat([
            actor, trel_p, rot6d(trel_q), slots, self.placed.float(), s["f"], other[:, None], dyn, s["q"], jv,
            self.q_cmd - s["q"], s["dz"][:, None], s["disp_pl"].max(-1).values[:, None],
            (self.since_release.float() / self.sk["place"]["settle_steps"])[:, None],
            (self.settled.float() / self.sk["place"]["settle_steps"])[:, None],
        ], -1)
        assert actor.shape[1] == ACTOR_DIM and critic.shape[1] == CRITIC_DIM, (actor.shape, critic.shape)
        return {"policy": torch.nan_to_num(actor), "critic": torch.nan_to_num(critic)}
