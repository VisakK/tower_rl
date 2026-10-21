"""push skill env (Task 4): from near the pre-push pose behind a block's end face, push the block off the full tower
along its long axis, leaving the rest of the tower in place.

Reset: a Task 4 tower (2 or 3 full layers of three) at a stage of the prescribed push order (blocks pushed before
it are gone, or lie on the table where they landed: ``reset.debris``), the domain tiers as for the other skills; the arm holds the closed fingers (width 0, the demos' tool)
around the pre-push pose (``tower_rl.push``) or close to the end face. Action: 6-D delta pose of the pinch (no
gripper channel: the fingers stay closed). Success: the target left the tower (centre >= 0.15 m from the tower axis,
or fallen >= 2 cm beyond 0.10 m) while every other block stayed in place (moved <= 10 mm, not tilted > 30 deg or
dropped > 15 mm). Failure: the remaining tower moved (structure broken) or the target was shoved sideways >= 4 cm.

Actor observation (65, base frame; ``tower_rl.skills.obs.push_actor_obs``). Critic: actor + true target pose in the
pinch frame, all slots relative to the target, presence, finger contact forces, dynamics scalars, joint state, true
travel, worst disturbance.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from tower_rl import geometry as geo
from tower_rl import push as P
from tower_rl.sim.assets import q_full
from tower_rl.skills.base_env import SkillEnvBase, SkillEnvCfg, squash
from tower_rl.skills.obs import push_actor_obs, rot6d
from tower_rl.tower import foundation_poses, source_poses

S_MAX = 9
ACTOR_DIM = 65
CRITIC_DIM = ACTOR_DIM + 9 + S_MAX * 5 + S_MAX + 2 + 1 + 5 + 7 + 7 + 7 + 1 + 1


@configclass
class PushEnvCfg(SkillEnvCfg):
    skill_yaml: str = "skills/push"
    action_space: int = 6
    observation_space: int = ACTOR_DIM
    state_space: int = CRITIC_DIM


class PushEnv(SkillEnvBase):
    cfg: PushEnvCfg

    def _pairs(self):
        return [(vi, k) for vi, s in enumerate(self.specs) for k, a in enumerate(s.push_arm) if a == self.arm]

    def __init__(self, cfg: PushEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        assert self.n_slots == S_MAX, self.n_slots
        n, dev = self.num_envs, self.device
        self.dir = torch.zeros(n, 3, device=dev)
        self.tower_xy = torch.zeros(n, 2, device=dev)
        self.z0 = torch.zeros(n, device=dev)
        self.yaw0 = torch.zeros(n, device=dev)
        self.layer = torch.ones(n, dtype=torch.long, device=dev)
        self.prev_along = torch.zeros(n, device=dev)
        self.noisy_p0 = torch.zeros(n, 3, device=dev)
        # keypoints in the pinch frame: centre, both jaw sides, along the approach
        self.kp = torch.tensor([[0, 0, 0], [0, 0.03, 0], [0, -0.03, 0], [0, 0, 0.03]], dtype=torch.float32, device=dev)
        self.seed_by_layer = {int(k): torch.tensor(v["q"], device=dev) for k, v in P.priors()[self.arm].items()}

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        r, dev = len(env_ids), self.device
        pi = torch.randint(len(self.pairs), (r,), generator=self.gen, device=dev)
        self.env_variant[env_ids] = self.pair_v[pi]
        self.env_stage[env_ids] = self.pair_k[pi]
        tier = torch.multinomial(self.tier_p, r, replacement=True, generator=self.gen)
        doms = {t: self.samplers[t].sample(r, self.gen) for t in self.tier_names}
        S = self.n_slots
        mov = torch.zeros(r, S, 7, device=dev)
        present = torch.zeros(r, S, dtype=torch.bool, device=dev)
        target = torch.zeros(r, dtype=torch.long, device=dev)
        direction = torch.zeros(r, 3, device=dev)
        tower = torch.zeros(r, 2, device=dev)
        layer = torch.ones(r, dtype=torch.long, device=dev)
        mass = torch.ones(r, S, device=dev)
        bfric = torch.zeros(r, 2, device=dev)
        ffric = torch.zeros(r, 2, device=dev)
        found = torch.zeros(r, 6, 7, device=dev)
        db = self.sk["reset"].get("debris") or {}
        deb = torch.zeros(r, S, 7, device=dev)                     # debris poses of earlier pushed blocks (if on)
        is_deb = torch.zeros(r, S, dtype=torch.bool, device=dev)
        for vi, spec in enumerate(self.specs):
            sel = (self.env_variant[env_ids] == vi).nonzero(as_tuple=True)[0]
            if len(sel) == 0:
                continue
            m = spec.num_movable
            stage = self.env_stage[env_ids][sel]
            for ti, t in enumerate(self.tier_names):
                s2 = sel[tier[sel] == ti]
                if len(s2) == 0:
                    continue
                d = doms[t].subset(s2)
                for j, e in enumerate(env_ids[s2].tolist()):
                    self.env_case[e] = f"{t}/{d.case[j]}"
                mov[s2, :m] = source_poses(spec, d, self.rz)
                mass[s2, :m] = d.mass_scale[:, :m]
                bfric[s2] = d.friction[:, 0]
                ffric[s2] = d.finger_friction
            done = torch.arange(len(spec.push_arm), device=dev)[None] < stage[:, None]   # pushes already executed
            pres = torch.zeros(len(sel), S, dtype=torch.bool, device=dev)
            pres[:, :m] = True
            pres[:, :m].scatter_(1, spec.push_block[None].expand(len(sel), -1), ~done)
            present[sel] = pres
            if db.get("frac", 0.0) > 0:  # earlier pushed blocks on the table where they land, never two in one spot
                placed = []
                for k in range(len(spec.push_arm)):
                    use = done[:, k] & (self._rand(len(sel)) < db["frac"])
                    if not bool(use.any()):
                        continue
                    b = int(spec.push_block[k])
                    pk, qk = P.debris_poses(mov[sel, b, :3], mov[sel, b, 3:7], spec.push_dir[k][None].expand(len(sel), 3),
                                            self._rand(len(sel), 4), db)
                    for m_prev, xy_prev in placed:
                        use = use & ~(m_prev & ((pk[:, :2] - xy_prev).norm(dim=-1) < db["min_gap_m"]))
                    deb[sel[use], b] = torch.cat([pk, qk], -1)[use]
                    is_deb[sel[use], b] = True
                    placed.append((use, pk[:, :2]))
            target[sel] = spec.push_block[stage]
            direction[sel] = spec.push_dir[stage]
            tower[sel] = spec.src_center
            layer[sel] = spec.layer[spec.push_block[stage]]
            found[sel] = foundation_poses(spec, len(sel), jitter_m=0.003, gen=self.gen)
        mov[..., 2] += 0.0005
        mov = torch.where(present[..., None], mov, self.park[None].expand(r, -1, -1))
        mov = torch.where(is_deb[..., None], deb, mov)
        # start mode: around the pre-push pose / close to the end face / "mid": the target already slid out along the
        # push (practice for the end of the push, where the block must clear the tower)
        rs = self.sk["reset"]
        mix = torch.tensor([rs["mix"]["prepush"], rs["mix"]["near"], rs["mix"].get("mid", 0.0)], device=dev)
        mode = torch.multinomial(mix, r, replacement=True, generator=self.gen)
        ar = torch.arange(r, device=dev)
        lo, hi = rs.get("mid_shift_m", [0.0, 0.0])
        shift = torch.where(mode == 2, lo + (hi - lo) * self._rand(r), torch.zeros(r, device=dev))
        mov[ar, target, :2] += direction[:, :2] * shift[:, None]
        poses = torch.zeros(r, len(self.blocks.object_names), 7, device=dev)
        poses[:, self.found_idx] = found
        poses[:, self.slot_idx] = mov
        world = poses.clone()
        world[..., :3] += self.scene.env_origins[env_ids][:, None]
        self.blocks.write_object_link_pose_to_sim(world, env_ids=env_ids)
        self.blocks.write_object_com_velocity_to_sim(torch.zeros(r, S, 6, device=dev), env_ids=env_ids, object_ids=self.slot_idx)
        self.present[env_ids] = present
        self.target_slot[env_ids] = target
        self.block_p0[env_ids] = mov[..., :3]
        self.mass_scale[env_ids] = mass
        self.block_fric[env_ids] = bfric
        self.finger_fric[env_ids] = ffric
        on = self.sk["obs_noise"]
        self.noise_pos[env_ids] = self._randn(r, S, 3) * on["episode_pos_m"]
        self.noise_pos[env_ids, :, 2] *= 0.5
        self.noise_yaw[env_ids] = self._randn(r, S) * math.radians(on["episode_yaw_deg"])
        self._apply_dynamics(env_ids)
        tpos, tquat = mov[ar, target, :3], mov[ar, target, 3:7]
        self.dir[env_ids], self.tower_xy[env_ids], self.layer[env_ids] = direction, tower, layer
        self.z0[env_ids] = tpos[:, 2] - 0.0005
        self.yaw0[env_ids] = geo.yaw_from_quat(tquat)
        self.prev_along[env_ids] = 0.0
        self.noisy_p0[env_ids] = tpos + self.noise_pos[env_ids][ar, target]
        # arm: closed fingers around the pre-push pose (tight / wide) or close to the end face ("near" and "mid")
        g = self.sk["geometry"]
        zoff = torch.tensor([P.z_offset(self.arm, int(k)) for k in layer.tolist()], device=dev)
        p_pre, q_pre = P.push_pose(tpos, direction, self.arm, g["prepush_along_m"], zoff)
        wide = self._rand(r) > rs["tight"]["frac"]
        rad = torch.where(wide, rs["wide"]["pos_m"], rs["tight"]["pos_m"])
        rot = torch.deg2rad(torch.where(wide, torch.tensor(rs["wide"]["rot_deg"], device=dev), torch.tensor(rs["tight"]["rot_deg"], device=dev)))
        u = self._randn(r, 3)
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9) * (self._rand(r, 1) ** (1 / 3)) * rad[:, None]
        p0 = p_pre + u
        near = mode >= 1
        nr = rs["near"]
        along = nr["along_m"][0] + (nr["along_m"][1] - nr["along_m"][0]) * self._rand(r)
        lat_dir = torch.stack([-direction[:, 1], direction[:, 0], torch.zeros(r, device=dev)], -1)
        p_near, _ = P.push_pose(tpos, direction, self.arm, 0.0, zoff)
        p_near = p_near + direction * along[:, None] + lat_dir * ((self._rand(r) * 2 - 1) * nr["lat_m"])[:, None]
        p_near[:, 2] += (self._rand(r) * 2 - 1) * nr["z_m"]
        p0 = torch.where(near[:, None], p_near, p0)
        rot = torch.where(near, torch.full_like(rot, math.radians(nr["rot_deg"])), rot)
        # never start with the fingers inside the target: at most max_along_m from its centre along the push
        a0 = ((p0 - tpos) * direction).sum(-1)
        p0 = p0 + direction * (rs["max_along_m"] - a0).clamp(max=0)[:, None]
        q0 = geo.normalize(geo.quat_mul(geo.quat_from_axis_angle(self._randn(r, 3), (self._rand(r) * 2 - 1) * rot), q_pre))
        seed = torch.stack([self.seed_by_layer.get(int(k), next(iter(self.seed_by_layer.values()))) for k in layer.tolist()])
        pb, qb = self.base.to_base(p0, q0)
        q_des, pe, re = self.kin.ik(seed, pb, qb, iters=40, q_ref=seed)
        bad = (pe > 0.005) | (re > math.radians(3))
        if bad.any():  # fall back to the nominal pre-push pose
            pb1, qb1 = self.base.to_base(p_pre[bad], q_pre[bad])
            q_des[bad] = self.kin.ik(seed[bad], pb1, qb1, iters=60, q_ref=seed[bad])[0]
        self._set_arm(env_ids, q_des)
        tw = g["tool_width_m"]
        self.robot.write_joint_state_to_sim(q_full(q_des, torch.full((r, 1), tw, device=dev)), torch.zeros(r, 10, device=dev),
                                            env_ids=env_ids)
        self.w_cmd[env_ids] = tw
        self.w_cmd_prev[env_ids] = tw
        self.robot.set_joint_position_target(q_full(self.q_cmd[env_ids], torch.full((r, 1), tw, device=dev)), env_ids=env_ids)

    # ------------------------------------------------------------------ actions
    def _action_scale(self):
        ac = self.sk["action"]
        npos, _ = self.noisy_block_poses()
        ar = torch.arange(self.num_envs, device=self.device)
        zoff = self._zoff()
        cp, _ = P.push_pose(npos[ar, self.target_slot], self.dir, self.arm, self.sk["geometry"]["contact_along_m"], zoff)
        d = (self.ctrl.pos - cp).norm(dim=-1)
        a = ((d - ac["near_radius_m"]) / (ac["far_radius_m"] - ac["near_radius_m"])).clamp(0, 1)
        return ac["pos_near_m"] + (ac["pos_far_m"] - ac["pos_near_m"]) * a

    def _zoff(self) -> torch.Tensor:
        if not hasattr(self, "_zoff_table"):
            self._zoff_table = torch.tensor([P.z_offset(self.arm, k) for k in range(4)], device=self.device)
        return self._zoff_table[self.layer.clamp(0, 3)]

    def _pre_physics_step(self, actions: torch.Tensor):
        a = actions.clamp(-1.0, 1.0)
        self.prev_actions[:] = self.actions
        self.actions[:] = a
        ac = self.sk["action"]
        self.ema_actions = ac["ema"] * a[:, :6] + (1 - ac["ema"]) * self.ema_actions
        dpos_b = self.ema_actions[:, :3] * self._action_scale()[:, None]
        drot_b = self.ema_actions[:, 3:6] * ac["rot_rad"]
        q_meas, _ = self.measured()
        self.ctrl.apply_delta(self.base.vec_to_world(dpos_b), self.base.vec_to_world(drot_b), q_meas=q_meas, max_lead=ac["max_lead_m"])
        self.q_cmd_prev[:] = self.q_cmd
        self.w_cmd_prev[:] = self.w_cmd
        self.q_cmd[:] = self.ctrl.command(q_meas, payload_kg=0.0)
        self.substep = 0

    # ------------------------------------------------------------------ state, dones, rewards
    def _compute_state(self):
        n, dev = self.num_envs, self.device
        ar = torch.arange(n, device=dev)
        q, w = self.measured()
        pb, qb, pw, qw = self.pinch(q)
        bpos, bquat = self.block_poses_local()
        tpos, tquat = bpos[ar, self.target_slot], bquat[ar, self.target_slot]
        along, lat = P.progress(tpos, self.block_p0[ar, self.target_slot], self.dir)
        cp, cq = P.push_pose(tpos, self.dir, self.arm, self.sk["geometry"]["contact_along_m"], self._zoff())
        kp_p = pw[:, None] + geo.quat_apply(qw[:, None].expand(n, 4, 4), self.kp[None].expand(n, 4, 3))
        kp_c = cp[:, None] + geo.quat_apply(cq[:, None].expand(n, 4, 4), self.kp[None].expand(n, 4, 3))
        kp_d = (kp_p - kp_c).norm(dim=-1).mean(-1)
        disp = (bpos - self.block_p0).norm(dim=-1)
        other = self.present.clone()
        other[ar, self.target_slot] = False
        pu = self.sk["push"]
        disp_nt = torch.where(other, disp, torch.zeros_like(disp))
        tilt = geo.tilt_angle(bquat)
        drop = self.block_p0[..., 2] - bpos[..., 2]
        broken = other & ((disp > pu["structure_mm"] / 1000) | (tilt > math.radians(pu["collapse_deg"])) | (drop > pu["collapse_drop_m"]))
        dyaw = geo.yaw_diff_sym(geo.yaw_from_quat(tquat), self.yaw0)
        departed = P.departed(tpos, self.z0, self.tower_xy)
        self.st = dict(q=q, w=w, pb=pb, qb=qb, pw=pw, qw=qw, bpos=bpos, bquat=bquat, tpos=tpos, tquat=tquat, along=along, lat=lat,
                       kp_d=kp_d, disp_nt=disp_nt, broken=broken.any(-1), dyaw=dyaw, departed=departed, cp=cp, cq=cq)
        return self.st

    def _get_dones(self):
        s = self._compute_state()
        pu = self.sk["push"]
        success = s["departed"] & ~s["broken"]
        shoved = (s["lat"].abs() > pu["lateral_fail_m"]) & ~s["departed"]
        failed = (s["broken"] | shoved) & ~success
        self.term = dict(success=success, failed=failed)
        terminated = success | failed
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        done = terminated | time_out
        self.extras["log"] = {"step_along_mm": float(s["along"].mean() * 1000)}
        if done.any():
            dn = done.float().sum()
            self.extras["log"] |= {
                "success_rate": float((success & done).float().sum() / dn),
                "failed_rate": float((failed & done).float().sum() / dn),
                "broken_rate": float((s["broken"] & done).float().sum() / dn),
                "timeout_rate": float((time_out & ~terminated & done).float().sum() / dn),
                "final_along_mm": float(s["along"][done].mean() * 1000),
                "max_disturb_mm": float(s["disp_nt"].max(-1).values[done].mean() * 1000),
            }
        return terminated, time_out

    def _get_rewards(self):
        s = self.st
        rw = self.sk["reward"]
        cap = self.sk["geometry"]["depart_m"]
        along_c = s["along"].clamp(-0.05, cap)
        r_prog = rw["progress"] * (along_c - self.prev_along)
        self.prev_along = along_c.detach().clone()
        engaged = s["along"] > 0.005
        r_align = (squash(s["kp_d"], 50.0, 2.0) + squash(s["kp_d"], 100.0, 0.0)) * (~engaged).float() \
            + 0.5 * (squash(s["kp_d"], 50.0, 2.0)) * engaged.float()
        pen_lat = ((s["lat"].abs() - 0.005).clamp(min=0) / 0.01)
        pen_yaw = (torch.rad2deg(s["dyaw"]) - 2.0).clamp(min=0)
        pen_d = ((s["disp_nt"] - 0.002).clamp(min=0) / 0.01).sum(-1)
        lim = self.kin.limits
        margin = torch.minimum(s["q"] - lim[:, 0], lim[:, 1] - s["q"])
        pen_jl = ((0.1 - margin) / 0.1).clamp(0, 1).sum(-1)
        rate = (self.actions - self.prev_actions).pow(2).sum(-1)
        remaining = (self.max_episode_length - self.episode_length_buf).float()
        r = (rw["align"] * r_align + r_prog - rw["lateral"] * pen_lat - rw["yaw"] * pen_yaw - rw["disturb_per_cm"] * pen_d
             - rw["action_rate"] * rate - rw["joint_limit"] * pen_jl)
        r = r + self.term["success"].float() * (rw["success"] + rw["success_per_remaining_step"] * remaining)
        r = r - self.term["failed"].float() * rw["fail"]
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
        tnp, tnq = npos[ar, self.target_slot], nquat[ar, self.target_slot]
        along, lat = P.progress(tnp, self.noisy_p0, self.dir)
        dyaw = geo.wrap_angle(geo.yaw_from_quat(tnq) - self.yaw0)
        cpw, cqw = P.push_pose(tnp, self.dir, self.arm, self.sk["geometry"]["contact_along_m"], self._zoff())
        cpb, cqb = self.base.to_base(cpw, cqw)
        valid = self.present.clone()
        valid[ar, self.target_slot] = False
        dir_b = self.base.vec_to_base(self.dir)
        actor = push_actor_obs(
            pb=pb, qb=qb, prev_pb=self.prev_pinch_b, prev_qb=self.prev_pinch_qb, dt=self.step_dt, tgt_pb=tgt_pb, tgt_qb=tgt_qb,
            target_pb=npb[ar, self.target_slot], target_qb=nqb[ar, self.target_slot], dir_b=dir_b, along=along, lateral=lat,
            dyaw=dyaw, contact_pb=cpb, contact_qb=cqb, other_pb=npb, other_qb=nqb, other_valid=valid, z0_world=self.z0,
            prev_actions=self.actions, progress=self.episode_length_buf.float() / self.max_episode_length)
        self.prev_pinch_b, self.prev_pinch_qb = pb.clone(), qb.clone()
        tp_true, tq_true = self.base.to_base(s["tpos"], s["tquat"])
        trel_p, trel_q = geo.frame_relative(pb, qb, tp_true, tq_true)
        allp, allq = self.base.to_base(s["bpos"], s["bquat"])
        arel_p, arel_q = geo.frame_relative(tp_true[:, None].expand_as(allp), tq_true[:, None].expand_as(allq), allp, allq)
        ayaw = geo.yaw_from_quat(arel_q)
        slots = torch.cat([arel_p, torch.cos(2 * ayaw)[..., None], torch.sin(2 * ayaw)[..., None]], -1).reshape(n, -1)
        f = self.finger_forces(self.target_slot)
        other = self.finger_any_other(self.target_slot)
        dyn = torch.cat([self.mass_scale[ar, self.target_slot][:, None], self.block_fric, self.finger_fric], -1)
        jv = self.robot.data.joint_vel[:, :7]
        critic = torch.cat([
            actor, trel_p, rot6d(trel_q), slots, self.present.float(), f, other[:, None], dyn, s["q"], jv,
            self.q_cmd - s["q"], s["along"][:, None], s["disp_nt"].max(-1).values[:, None],
        ], -1)
        assert actor.shape[1] == ACTOR_DIM and critic.shape[1] == CRITIC_DIM, (actor.shape, critic.shape)
        return {"policy": torch.nan_to_num(actor), "critic": torch.nan_to_num(critic)}
