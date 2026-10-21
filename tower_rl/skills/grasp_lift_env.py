"""grasp_lift skill env (Step 5): from near the pre-grasp pose, grasp the stage's target block and lift it 8 cm.

Actor observations (camera-estimable only; base frame of the arm):
  pinch pose (3 + 6), pinch velocity from finite differences (6), target lead (3 + 3), measured width, last gripper
  request, width command (3), noisy target block relative to the pinch (pos 3, block x/z axes in the pinch frame 6),
  noisy target in the base frame (3), canonical grasp pose error from the noisy block (3 + 3), two nearest other
  blocks relative to the target (2 x (3 + 2 + 1)), previous action (7), phase one-hot (3), episode progress (1).
Critic: actor observations + true target relative pose, true poses of all slots relative to the target, presence,
  finger contact forces, any contact with other blocks, mass / friction scalars, joint angles / velocities, command
  lead in joint space, lift height, worst disturbance.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from tower_rl import geometry as geo
from tower_rl.grasp import cfg as grasp_cfg
from tower_rl.grasp import grasp_relation_error
from tower_rl.postures import PADS, priors
from tower_rl.sim.skill_scene import MAX_MOVABLE
from tower_rl.skills.base_env import SkillEnvBase, SkillEnvCfg, squash
from tower_rl.skills.obs import grasp_lift_actor_obs, grasp_target_world, rot6d

ACTOR_DIM = 3 + 6 + 6 + 3 + 3 + 3 + 3 + 6 + 3 + 6 + 12 + 7 + 3 + 1
CRITIC_DIM = ACTOR_DIM + 9 + MAX_MOVABLE * 5 + MAX_MOVABLE + 2 + 1 + 5 + 7 + 7 + 7 + 1 + 1


@configclass
class GraspLiftEnvCfg(SkillEnvCfg):
    skill_yaml: str = "skills/grasp_lift"
    action_space: int = 7
    observation_space: int = ACTOR_DIM
    state_space: int = CRITIC_DIM


class GraspLiftEnv(SkillEnvBase):
    cfg: GraspLiftEnvCfg

    def __init__(self, cfg: GraspLiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self.hold_count = torch.zeros(n, dtype=torch.long, device=dev)
        self.ever_lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self.z0 = torch.zeros(n, device=dev)
        self.noisy_z0 = torch.zeros(n, device=dev)
        self.tilt_deg = grasp_cfg()["approach_tilt_deg"][self.arm]
        self.pregrasp_h = grasp_cfg()["pregrasp_height_m"]
        g = self.sk["grasp"]
        self.lift_success = g["lift_success_m"]
        # prior seeds per pad / yaw class for this arm (grasp phase)
        pr = priors()[self.arm]["grasp"]
        self.seed_table = {(pad, int(yc)): torch.tensor(v["q"], device=dev) for pad, d in pr.items() for yc, v in d.items()}
        self.pad_xy = torch.tensor([PADS["A"], PADS["B"]], device=dev)
        self.stats = {}
        # keypoints (pinch frame): centre, both jaw sides, along the approach
        self.kp = torch.tensor([[0, 0, 0], [0, 0.04, 0], [0, -0.04, 0], [0, 0, 0.04]], dtype=torch.float32, device=dev)

    # ------------------------------------------------------------------ geometry
    def _seed(self, block_pos, block_quat):
        """Posture prior per env (pad + yaw class of the target block)."""
        d = torch.cdist(block_pos[:, :2], self.pad_xy)  # [N, 2]
        pad = d.argmin(-1)
        yaw = geo.yaw_from_quat(block_quat)
        y = torch.remainder(torch.rad2deg(yaw) + 45, 180) - 45
        ycls = torch.where(y.abs() < 45, 0, 90)
        out = torch.zeros(len(block_pos), 7, device=self.device)
        for (p, yc), q in self.seed_table.items():
            m = (pad == (0 if p == "A" else 1)) & (ycls == yc)
            out[m] = q
        missing = (out.abs().sum(-1) == 0)
        if missing.any():  # fall back to any prior of this arm
            out[missing] = next(iter(self.seed_table.values()))
        return out

    def grasp_target(self, block_pos, block_quat):
        """Canonical pinch pose that grasps the block centred (station frame)."""
        return grasp_target_world(block_pos, block_quat, self.arm, self.tilt_deg)

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        r = len(env_ids)
        tpos, tquat = self._sample_towers(env_ids)
        gp, gq = self.grasp_target(tpos, tquat)
        # initial pinch pose: around the pre-grasp (tight / wide), mid-height above the block, or at the grasp pose
        rs = self.sk["reset"]
        mix = torch.tensor([rs["mix"]["pregrasp"], rs["mix"]["mid"], rs["mix"]["at_grasp"]], device=self.device)
        mode = torch.multinomial(mix, r, replacement=True, generator=self.gen)
        wide = self._rand(r) > rs["tight"]["frac"]
        rad = torch.where(wide, rs["wide"]["pos_m"], rs["tight"]["pos_m"])
        rot = torch.deg2rad(torch.where(wide, torch.tensor(rs["wide"]["rot_deg"], device=self.device),
                                        torch.tensor(rs["tight"]["rot_deg"], device=self.device)))
        u = self._randn(r, 3)
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9) * (self._rand(r, 1) ** (1 / 3)) * rad[:, None]
        p0 = gp.clone()
        p0[:, 2] += self.pregrasp_h
        p0 = p0 + u
        p0[:, 2] = torch.maximum(p0[:, 2], gp[:, 2] + rs["min_height_above_grasp_m"])
        # mid / at-grasp modes: offsets along the gripper's x (block length) and y (jaw) axes
        rg = geo.quat_to_mat(gq)
        ex, ey = rg[..., :, 0], rg[..., :, 1]
        ag = rs["at_grasp"]
        sym = lambda a: (self._rand(r) * 2 - 1) * a  # noqa: E731
        lo, hi = rs["mid_height_m"]
        p_mid = gp + ex * sym(rs["mid_lateral_m"])[:, None] + ey * sym(rs["mid_lateral_m"])[:, None]
        p_mid[:, 2] += lo + (hi - lo) * self._rand(r)
        p_at = gp + ex * sym(ag["length_m"])[:, None] + ey * sym(ag["jaw_m"])[:, None]
        p_at[:, 2] += sym(ag["z_m"])
        p0 = torch.where((mode == 1)[:, None], p_mid, p0)
        p0 = torch.where((mode == 2)[:, None], p_at, p0)
        rot = torch.where(mode == 0, rot, torch.full_like(rot, math.radians(ag["rot_deg"])))
        ax = self._randn(r, 3)
        q0 = geo.normalize(geo.quat_mul(geo.quat_from_axis_angle(ax, (self._rand(r) * 2 - 1) * rot), gq))
        seed = self._seed(tpos, tquat)
        pb, qb = self.base.to_base(p0, q0)
        q_des, pe, re = self.kin.ik(seed, pb, qb, iters=30, q_ref=seed)
        bad = (pe > 0.005) | (re > math.radians(3))
        if bad.any():  # fall back to the nominal pre-grasp
            p1 = gp.clone()
            p1[:, 2] += self.pregrasp_h
            pb1, qb1 = self.base.to_base(p1[bad], gq[bad])
            q1, _, _ = self.kin.ik(seed[bad], pb1, qb1, iters=30, q_ref=seed[bad])
            q_des[bad] = q1
        self._set_arm(env_ids, q_des)
        self.hold_count[env_ids] = 0
        self.ever_lifted[env_ids] = False
        self.z0[env_ids] = tpos[:, 2] - 0.0005
        nz = self.block_p0[env_ids, :, 2] + self.noise_pos[env_ids, :, 2]
        self.noisy_z0[env_ids] = nz[torch.arange(r, device=self.device), self.target_slot[env_ids]] - 0.0005

    # ------------------------------------------------------------------ actions
    def _action_scale(self):
        ac = self.sk["action"]
        npos, nq = self.noisy_block_poses()
        ar = torch.arange(self.num_envs, device=self.device)
        gp, _ = self.grasp_target(npos[ar, self.target_slot], nq[ar, self.target_slot])
        d = (self.ctrl.pos - gp).norm(dim=-1)
        a = ((d - ac["near_radius_m"]) / (ac["far_radius_m"] - ac["near_radius_m"])).clamp(0, 1)
        return ac["pos_near_m"] + (ac["pos_far_m"] - ac["pos_near_m"]) * a

    def _payload_kg(self):
        return (self.grip.closed & self.ever_lifted).float() * self.ctrl.cfg.payload_mass

    # ------------------------------------------------------------------ state shared by dones / rewards / obs
    def _compute_state(self):
        n, dev = self.num_envs, self.device
        ar = torch.arange(n, device=dev)
        q, w = self.measured()
        pb, qb, pw, qw = self.pinch(q)
        bpos, bquat = self.block_poses_local()
        tpos, tquat = bpos[ar, self.target_slot], bquat[ar, self.target_slot]
        gp, gq = self.grasp_target(tpos, tquat)
        f = self.finger_forces(self.target_slot)
        thr = self.sk["grasp"]["contact_force_n"]
        both = (f > thr).all(-1)
        closed = self.grip.closed
        grasped = both & closed & (w < 0.056)
        lift = (tpos[:, 2] - self.z0).clamp(min=0)
        # keypoint distance pinch vs canonical grasp pose
        kp_p = pw[:, None] + geo.quat_apply(qw[:, None].expand(n, 4, 4), self.kp[None].expand(n, 4, 3))
        kp_g = gp[:, None] + geo.quat_apply(gq[:, None].expand(n, 4, 4), self.kp[None].expand(n, 4, 3))
        kp_d = (kp_p - kp_g).norm(dim=-1).mean(-1)
        disp = (bpos - self.block_p0).norm(dim=-1)
        nontarget = self.present.clone()
        nontarget[ar, self.target_slot] = False
        disp_nt = torch.where(nontarget, disp, torch.zeros_like(disp))
        tilt = geo.tilt_angle(tquat)
        # grasp quality: the pinch pose in the block frame against the canonical grasp (offset x along the block, z up
        # the block; rotation with interchangeable fingers), invariant to the held block's tilt in the world
        d_b, rot = grasp_relation_error(tpos, tquat, pw, qw, self.arm, self.tilt_deg)
        qc = self.sk.get("quality")
        good = torch.ones(n, dtype=torch.bool, device=dev) if qc is None else \
            (d_b[:, 0].abs() <= qc["x_m"]) & (d_b[:, 2].abs() <= qc["z_m"]) & (rot <= math.radians(qc["rot_deg"]))
        self.st = dict(q=q, w=w, pb=pb, qb=qb, pw=pw, qw=qw, bpos=bpos, bquat=bquat, tpos=tpos, tquat=tquat, gp=gp, gq=gq,
                       f=f, both=both, grasped=grasped, lift=lift, kp_d=kp_d, disp_nt=disp_nt, tilt=tilt,
                       q_dx=d_b[:, 0], q_dz=d_b[:, 2], q_rot=rot, good=good)
        return self.st

    def _get_dones(self):
        s = self._compute_state()
        g = self.sk["grasp"]
        tm = self.sk["termination"]
        lifted = s["grasped"] & (s["lift"] >= g["lift_success_m"])
        # held at the lift height for hold_steps; a grasp far from the canonical one (quality gate, the place needs a
        # near-canonical held offset) ends the episode as a bad grasp instead of a success
        self.hold_count = torch.where(lifted, self.hold_count + 1, torch.zeros_like(self.hold_count))
        self.ever_lifted |= s["grasped"] & (s["lift"] >= tm["drop_after_lift_m"])
        held = self.hold_count >= g["hold_steps"]
        success = held & s["good"]
        bad_grasp = held & ~s["good"]
        disturbed = (s["disp_nt"] > tm["disturb_mm"] / 1000).any(-1)
        dropped = self.ever_lifted & ~s["grasped"] & (s["lift"] < 0.01)
        fell = (s["tilt"] > math.radians(tm["tilt_deg"])) | (s["tpos"][:, 2] < self.z0 - 0.02)
        self.term = dict(success=success, disturbed=disturbed & ~success, dropped=(dropped | fell) & ~success,
                         bad_grasp=bad_grasp & ~disturbed)
        terminated = success | bad_grasp | disturbed | dropped | fell
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # episode statistics for logging
        done = terminated | time_out
        self.extras["log"] = {"step_grasped_frac": float(s["grasped"].float().mean()),
                              "step_both_contact_frac": float(s["both"].float().mean()),
                              "step_close_request_frac": float(self.grip.closed.float().mean())}
        if done.any():
            d = done.float().sum()
            self.extras["log"] |= {
                "success_rate": float((success & done).float().sum() / d),
                "disturbed_rate": float((self.term["disturbed"] & done).float().sum() / d),
                "dropped_rate": float((self.term["dropped"] & done).float().sum() / d),
                "timeout_rate": float((time_out & ~terminated & done).float().sum() / d),
                "bad_grasp_rate": float((self.term["bad_grasp"] & done).float().sum() / d),
                "max_lift_mm": float(s["lift"][done].mean() * 1000),
            }
            if held.any():  # grasp quality of the grasps that reached the hold (success or bad)
                self.extras["log"] |= {"held_dx_mm": float(s["q_dx"][held].abs().mean() * 1000),
                                       "held_dz_mm": float(s["q_dz"][held].abs().mean() * 1000),
                                       "held_rot_deg": float(torch.rad2deg(s["q_rot"][held]).mean())}
        return terminated, time_out

    def _get_rewards(self):
        s = self.st
        rw = self.sk["reward"]
        r_align = squash(s["kp_d"], 50.0, 2.0) + squash(s["kp_d"], 100.0, 0.0)
        r_lift = (s["lift"] / self.lift_success).clamp(0, 1) * s["grasped"].float()
        pen_d = ((s["disp_nt"] - 0.003).clamp(min=0) / 0.01).sum(-1)
        lim = self.kin.limits
        margin = torch.minimum(s["q"] - lim[:, 0], lim[:, 1] - s["q"])
        pen_jl = ((0.1 - margin) / 0.1).clamp(0, 1).sum(-1)
        rate = (self.actions - self.prev_actions).pow(2).sum(-1)
        remaining = (self.max_episode_length - self.episode_length_buf).float()
        close_aligned = (self.grip.closed & (s["kp_d"] < 0.01)).float()
        r = (rw["align"] * r_align + rw["contact"] * s["both"].float() + rw["grasp"] * s["grasped"].float()
             + rw["close_aligned"] * close_aligned
             + rw["lift"] * r_lift - rw["disturb_per_cm"] * pen_d - rw["action_rate"] * rate - rw["joint_limit"] * pen_jl)
        r = r + self.term["success"].float() * (rw["success"] + rw["success_per_remaining_step"] * remaining)
        r = r - self.term["disturbed"].float() * rw["disturb_terminal"] - self.term["dropped"].float() * rw["drop"]
        r = r - self.term["bad_grasp"].float() * rw.get("bad_grasp", 0.0)
        return r

    # ------------------------------------------------------------------ observations
    def _get_observations(self):
        s = self._compute_state()  # recomputed: resets happened after the dones/rewards pass
        n, dev = self.num_envs, self.device
        ar = torch.arange(n, device=dev)
        pb, qb = s["pb"], s["qb"]
        tgt_pb, tgt_qb = self.base.to_base(self.ctrl.pos, self.ctrl.quat)
        # noisy blocks (actor)
        npos, nquat = self.noisy_block_poses()
        npb, nqb = self.base.to_base(npos, nquat)
        tpb, tqb = npb[ar, self.target_slot], nqb[ar, self.target_slot]
        gpw, gqw = grasp_target_world(npos[ar, self.target_slot], nquat[ar, self.target_slot], self.arm, self.tilt_deg)
        gpb, gqb = self.base.to_base(gpw, gqw)
        valid = self.present.clone()
        valid[ar, self.target_slot] = False
        actor = grasp_lift_actor_obs(
            pb=pb, qb=qb, prev_pb=self.prev_pinch_b, prev_qb=self.prev_pinch_qb, dt=self.step_dt, tgt_pb=tgt_pb, tgt_qb=tgt_qb,
            width=s["w"], closed=self.grip.closed, width_cmd=self.grip.width, target_pb=tpb, target_qb=tqb,
            grasp_pb=gpb, grasp_qb=gqb, other_pb=npb, other_qb=nqb, other_valid=valid,
            target_z_world=npos[ar, self.target_slot, 2], target_z0_world=self.noisy_z0, prev_actions=self.actions,
            progress=self.episode_length_buf.float() / self.max_episode_length)
        self.prev_pinch_b, self.prev_pinch_qb = pb.clone(), qb.clone()
        # privileged
        tp_true, tq_true = self.base.to_base(s["tpos"], s["tquat"])
        trel_p, trel_q = geo.frame_relative(pb, qb, tp_true, tq_true)
        allp, allq = self.base.to_base(s["bpos"], s["bquat"])
        arel_p, arel_q = geo.frame_relative(tp_true[:, None].expand_as(allp), tq_true[:, None].expand_as(allq), allp, allq)
        ayaw = geo.yaw_from_quat(arel_q)
        slots = torch.cat([arel_p, torch.cos(2 * ayaw)[..., None], torch.sin(2 * ayaw)[..., None]], -1).reshape(n, -1)
        other = self.finger_any_other(self.target_slot)
        dyn = torch.cat([self.mass_scale[ar, self.target_slot][:, None], self.block_fric, self.finger_fric], -1)
        jv = self.robot.data.joint_vel[:, :7]
        critic = torch.cat([
            actor, trel_p, rot6d(trel_q), slots, self.present.float(), s["f"], other[:, None], dyn, s["q"], jv,
            self.q_cmd - s["q"], s["lift"][:, None], s["disp_nt"].max(-1).values[:, None],
        ], -1)
        assert actor.shape[1] == ACTOR_DIM and critic.shape[1] == CRITIC_DIM, (actor.shape, critic.shape)
        return {"policy": torch.nan_to_num(actor), "critic": torch.nan_to_num(critic)}
