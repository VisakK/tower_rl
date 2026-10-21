"""Task 5 skill envs (Tasks 5-2 / 5-3): ``insert`` (slide the held block into the slot and release it) and ``push_in``
(push the released block home with the nearly closed fingers). Import after ``AppLauncher``.

Scene (``configs/skills/insert.yaml: scene``): one arm, both pads' foundations, nine movable slots: m0-m3 the 32 mm side
blocks, m4-m6 the roof, m7 the insert of the current stage, m8 an insert already in its slot (Task 5-3 stage 1); cuboid
colliders (the Task 5-3 generator's, the tighter fit: 2 mm lateral / 2 mm vertical clearance) and 3 N fingers. The
destination tower stays at its layout pose (docs: Task 5 pose cases vary only the source blocks) up to the generator's
3 mm tower jitter; mass / friction cases as for every skill.

Shared actor observation (``insert_actor_obs``, base frame): pinch pose / velocity / lead (21), width, gripper request,
width command (3), the insert block in the pinch frame (9), slot centre and axis (6), the block relative to the slot
(along / across / vertical, cos / sin 2 dyaw) (5), phase one-hot (3), previous action (7), progress (1) = 55.
"""

from __future__ import annotations

import math

import torch

from tower_rl import geometry as geo
from tower_rl.config import load_yaml
from tower_rl.graph.graph import insertion_direction
from tower_rl.data.layouts import Layout, yaw_of_wxyz
from tower_rl.skills.base_env import SkillEnvBase, rot6d
from tower_rl.skills.task5_obs import ACTOR_DIM, insert_actor_obs, pad_actions, slot_relative  # noqa: F401

SIDE_SLOTS, ROOF_SLOTS, INSERT_SLOT, DONE_SLOT = (0, 1, 2, 3), (4, 5, 6), 7, 8


class Task5Spec:
    """Per (variant, stage): the tower's blocks in their slots, the insert's goal (slot) and direction."""

    def __init__(self, variant: str, stage: int, device):
        lay = Layout.load(variant)
        t = lay.transfers[stage]
        self.arm = t.arm
        sides = sorted(b for b in lay.movable_ids if lay.role(b) == "side")
        roof = sorted(b for b in lay.movable_ids if lay.role(b) == "roof")
        P = lambda b: torch.tensor(lay.blocks[b], dtype=torch.float32, device=device)  # noqa: E731
        self.side = torch.stack([P(b) for b in sides] + [torch.zeros(7, device=device)] * (4 - len(sides)))
        self.n_side = len(sides)
        self.roof = torch.stack([P(b) for b in roof])
        self.goal = torch.tensor(t.goal, dtype=torch.float32, device=device)
        self.goal[2] = lay.raw["transfers"][stage]["rest_z_final"]
        self.dir = torch.tensor(insertion_direction(t.goal[:3], yaw_of_wxyz(t.goal[3:7]), t.arm), dtype=torch.float32, device=device)
        self.slot_sides = [sides.index(b) for b in lay.raw["transfers"][stage]["side_blocks"]]
        # an insert done earlier (Task 5-3 stage 1: the layer-2 block sits in its slot)
        self.done = None
        if stage > 0:
            g0 = torch.tensor(lay.transfers[0].goal, dtype=torch.float32, device=device)
            g0[2] = lay.raw["transfers"][0]["rest_z_final"]
            self.done = g0
        f_ids = list(lay.pads["source"]["foundation_blocks"]) + list(lay.pads["destination"]["foundation_blocks"])
        self.foundation = torch.stack([P(b) for b in f_ids])


class Task5EnvBase(SkillEnvBase):
    """Tower reset shared by the insert and push_in envs (subclasses place the insert block and the arm)."""

    def _pairs(self):
        out = []
        for vi, s in enumerate(self.specs):
            for k, a in enumerate(s.transfer_arm):
                if a == self.arm:
                    out.append((vi, k))
        return out

    def _t5_init(self):
        dev = self.device
        self.t5 = {(vi, k): Task5Spec(self.specs[vi].variant, k, dev) for vi, k in self.pairs}
        self.ig = load_yaml("skills/insert")["geometry"]
        n = self.num_envs
        self.slot_pos = torch.zeros(n, 3, device=dev)
        self.slot_quat = torch.zeros(n, 4, device=dev)
        self.slot_quat[:, 0] = 1
        self.axis = torch.zeros(n, 3, device=dev)
        self.structure0 = torch.zeros(n, 7, 3, device=dev)  # the tower's side + roof blocks at reset (slots m0-m6)
        self.n_side = torch.zeros(n, dtype=torch.long, device=dev)

    def _tower_reset(self, env_ids: torch.Tensor):
        """Write the tower (sides, roof, earlier insert) and return per-env (goal [r, 7], dir [r, 3])."""
        r, dev = len(env_ids), self.device
        pi = torch.randint(len(self.pairs), (r,), generator=self.gen, device=dev)
        self.env_variant[env_ids] = self.pair_v[pi]
        self.env_stage[env_ids] = self.pair_k[pi]
        tier = torch.multinomial(self.tier_p, r, replacement=True, generator=self.gen)
        doms = {t: self.samplers[t].sample(r, self.gen) for t in self.tier_names}
        S = self.n_slots
        mov = self.park[None].repeat(r, 1, 1)
        present = torch.zeros(r, S, dtype=torch.bool, device=dev)
        found = torch.zeros(r, 6, 7, device=dev)
        goal = torch.zeros(r, 7, device=dev)
        direction = torch.zeros(r, 3, device=dev)
        mass = torch.ones(r, S, device=dev)
        bfric = torch.zeros(r, 2, device=dev)
        ffric = torch.zeros(r, 2, device=dev)
        jit = (self._rand(r, 2) * 2 - 1) * self.sk["reset"]["tower_jitter_m"]  # the whole destination tower
        for j, (vi, k) in enumerate(self.pairs):
            sel = (pi == j).nonzero(as_tuple=True)[0]
            if len(sel) == 0:
                continue
            sp = self.t5[(vi, k)]
            ns = sp.n_side
            mov[sel, :ns] = sp.side[:ns]
            present[sel, :ns] = True
            mov[sel, 4:7] = sp.roof
            present[sel, 4:7] = True
            if sp.done is not None:
                mov[sel, DONE_SLOT] = sp.done
                present[sel, DONE_SLOT] = True
            found[sel] = sp.foundation
            goal[sel] = sp.goal
            direction[sel] = sp.dir
            self.n_side[env_ids[sel]] = ns
        # tower jitter: foundation of pad A (slots 3-5), the tower blocks and the goal move together
        for i in range(3, 6):
            found[:, i, :2] += jit
        tower = present.clone()
        tower[:, INSERT_SLOT] = False
        mov[..., :2] = torch.where(tower[..., None], mov[..., :2] + jit[:, None], mov[..., :2])
        goal[:, :2] += jit
        for ti, t in enumerate(self.tier_names):
            s2 = (tier == ti).nonzero(as_tuple=True)[0]
            if len(s2) == 0:
                continue
            d = doms[t].subset(s2)
            for j, e in enumerate(env_ids[s2].tolist()):
                self.env_case[e] = f"{t}/{d.case[j]}"
            mass[s2] = d.mass_scale[:, :S]
            bfric[s2] = d.friction[:, 0]
            ffric[s2] = d.finger_friction
        self._t5_mov, self._t5_present, self._t5_found = mov, present, found
        self.mass_scale[env_ids], self.block_fric[env_ids], self.finger_fric[env_ids] = mass, bfric, ffric
        self.slot_pos[env_ids] = goal[:, :3]
        self.slot_quat[env_ids] = goal[:, 3:7]
        self.axis[env_ids] = direction
        return goal, direction

    def _t5_write(self, env_ids: torch.Tensor):
        r, dev = len(env_ids), self.device
        mov = self._t5_mov
        mov = mov.clone()
        mov[..., 2] += torch.where(self._t5_present, 0.0003, 0.0)  # a hair above contact, then the physics settles
        poses = torch.zeros(r, len(self.blocks.object_names), 7, device=dev)
        poses[:, self.found_idx] = self._t5_found
        poses[:, self.slot_idx] = mov
        world = poses.clone()
        world[..., :3] += self.scene.env_origins[env_ids][:, None]
        self.blocks.write_object_link_pose_to_sim(world, env_ids=env_ids)
        self.blocks.write_object_com_velocity_to_sim(torch.zeros(r, self.n_slots, 6, device=dev), env_ids=env_ids,
                                                     object_ids=self.slot_idx)
        self.present[env_ids] = self._t5_present
        self.target_slot[env_ids] = INSERT_SLOT
        self.block_p0[env_ids] = mov[..., :3]
        self.structure0[env_ids] = mov[:, :7, :3]
        on = self.sk["obs_noise"]
        self.noise_pos[env_ids] = self._randn(r, self.n_slots, 3) * on["episode_pos_m"]
        self.noise_pos[env_ids, :, 2] *= 0.5
        self.noise_yaw[env_ids] = self._randn(r, self.n_slots) * math.radians(on["episode_yaw_deg"])
        self._apply_dynamics(env_ids)

    def structure_moved(self, bpos: torch.Tensor) -> torch.Tensor:
        """Largest displacement ``[N]`` of the tower's present side / roof blocks since the reset."""
        d = (bpos[:, :7] - self.structure0).norm(dim=-1)
        return torch.where(self.present[:, :7], d, torch.zeros_like(d)).max(-1).values


# ---------------------------------------------------------------------------------------------------------- insert
from isaaclab.utils import configclass  # noqa: E402

from tower_rl.grasp import cfg as grasp_cfg  # noqa: E402
from tower_rl.postures import prior_for_pose  # noqa: E402
from tower_rl.sim.assets import q_full  # noqa: E402
from tower_rl.skills.base_env import SkillEnvCfg, squash  # noqa: E402
from tower_rl.skills.obs import grasp_target_world  # noqa: E402

CRITIC_DIM = ACTOR_DIM + 9 + 5 + 9 * 5 + 9 + 2 + 1 + 5 + 7 + 7 + 7 + 1 + 1 + 1


class _Task5Common(Task5EnvBase):
    """Observation / critic assembly and the shared state of the insert and push_in envs."""

    def _phase(self) -> torch.Tensor:
        raise NotImplementedError

    def _common_state(self):
        n, dev = self.num_envs, self.device
        q, w = self.measured()
        pb, qb, pw, qw = self.pinch(q)
        bpos, bquat = self.block_poses_local()
        tpos, tquat = bpos[:, INSERT_SLOT], bquat[:, INSERT_SLOT]
        rel = slot_relative(tpos, tquat, self.slot_pos, self.slot_quat, self.axis)
        tilt = geo.tilt_angle(tquat)
        moved = self.structure_moved(bpos)
        return dict(q=q, w=w, pb=pb, qb=qb, pw=pw, qw=qw, bpos=bpos, bquat=bquat, tpos=tpos, tquat=tquat, rel=rel, tilt=tilt,
                    moved=moved, along=rel[:, 0], lat=rel[:, 1], vert=rel[:, 2], dyaw=torch.atan2(rel[:, 4], rel[:, 3]).abs() / 2)

    def _obs(self, s):
        n, dev = self.num_envs, self.device
        pb, qb = s["pb"], s["qb"]
        tgt_pb, tgt_qb = self.base.to_base(self.ctrl.pos, self.ctrl.quat)
        npos, nquat = self.noisy_block_poses()
        npb, nqb = self.base.to_base(npos, nquat)
        # the slot as perceived: the midpoint of the (noisy) side blocks is what the executor uses; here the true slot
        # plus the side blocks' mean noise (their shared perception error moves the perceived slot)
        sides = torch.tensor(SIDE_SLOTS, device=dev)
        sn = self.noise_pos[:, sides].mean(1)
        sp_b, sq_b = self.base.to_base(self.slot_pos + sn, self.slot_quat)
        axis_b = self.base.vec_to_base(self.axis)
        actor = insert_actor_obs(
            pb=pb, qb=qb, prev_pb=self.prev_pinch_b, prev_qb=self.prev_pinch_qb, dt=self.step_dt, tgt_pb=tgt_pb, tgt_qb=tgt_qb,
            width=s["w"], closed=self.grip.closed, width_cmd=self.w_cmd, block_pb=npb[:, INSERT_SLOT], block_qb=nqb[:, INSERT_SLOT],
            slot_pb=sp_b, slot_qb=sq_b, axis_b=axis_b, phase=self._phase(), prev_actions=pad_actions(self.actions),
            progress=self.episode_length_buf.float() / self.max_episode_length)
        self.prev_pinch_b, self.prev_pinch_qb = pb.clone(), qb.clone()
        tp_b, tq_b = self.base.to_base(s["tpos"], s["tquat"])
        trel_p, trel_q = geo.frame_relative(pb, qb, tp_b, tq_b)
        allp, allq = self.base.to_base(s["bpos"], s["bquat"])
        arel_p, arel_q = geo.frame_relative(tp_b[:, None].expand_as(allp), tq_b[:, None].expand_as(allq), allp, allq)
        ayaw = geo.yaw_from_quat(arel_q)
        slots = torch.cat([arel_p, torch.cos(2 * ayaw)[..., None], torch.sin(2 * ayaw)[..., None]], -1).reshape(n, -1)
        f = self.finger_forces(torch.full((n,), INSERT_SLOT, device=dev, dtype=torch.long))
        other = self.finger_any_other(torch.full((n,), INSERT_SLOT, device=dev, dtype=torch.long))
        dyn = torch.cat([self.mass_scale[:, INSERT_SLOT:INSERT_SLOT + 1], self.block_fric, self.finger_fric], -1)
        jv = self.robot.data.joint_vel[:, :7]
        critic = torch.cat([actor, trel_p, rot6d(trel_q), s["rel"], slots, self.present.float(), f, other[:, None], dyn, s["q"], jv,
                            self.q_cmd - s["q"], s["moved"][:, None], s["tilt"][:, None], s["along"][:, None]], -1)
        assert actor.shape[1] == ACTOR_DIM and critic.shape[1] == CRITIC_DIM, (actor.shape, critic.shape)
        return {"policy": torch.nan_to_num(actor), "critic": torch.nan_to_num(critic)}


@configclass
class InsertEnvCfg(SkillEnvCfg):
    skill_yaml: str = "skills/insert"
    action_space: int = 7
    observation_space: int = ACTOR_DIM
    state_space: int = CRITIC_DIM


class InsertEnv(_Task5Common):
    """Held entry: from the pre-insert pose (block on the slot axis, ``outside_m`` out) or part-way in, slide the block
    into the slot, release it at ``entry_m`` (+-) and retreat. Success: released, the block's leading end >= 45 mm inside
    (centre along >= -0.105 m) and not past -0.06 m, across <= 2.5 mm, not tilted > 8 deg, the tower moved <= 3 mm, the
    pinch 3 cm up, static. Failure: the tower moved > 10 mm, the block dropped / tipped (> 20 deg) or released short of
    the slot, timeout."""

    cfg: InsertEnvCfg

    def __init__(self, cfg: InsertEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self._t5_init()
        self.released = torch.zeros(n, dtype=torch.bool, device=dev)
        self.since_release = torch.zeros(n, dtype=torch.long, device=dev)
        self.settled = torch.zeros(n, dtype=torch.long, device=dev)
        self.z_release = torch.zeros(n, device=dev)
        self.prev_along = torch.zeros(n, device=dev)
        self.prev_tpos = torch.zeros(n, 3, device=dev)
        self.tilt_deg = {a: grasp_cfg()["approach_tilt_deg"][a] for a in ("left", "back")}

    def _phase(self):
        q, _ = self.measured()
        pw = self.pinch(q)[2]
        retreated = self.released & (pw[:, 2] >= self.z_release + self.sk["insert"]["retreat_m"])
        return torch.stack([~self.released, self.released & ~retreated, retreated], -1).float()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        r, dev = len(env_ids), self.device
        goal, d = self._tower_reset(env_ids)
        rs = self.sk["reset"]
        lat = torch.stack([-d[:, 1], d[:, 0], torch.zeros(r, device=dev)], -1)
        u = lambda a: (self._rand(r) * 2 - 1) * a  # noqa: E731
        mid = self._rand(r) < rs["mid_frac"]
        a_out = -self.ig["outside_m"] + u(rs["outside"]["along_m"])
        a_mid = rs["mid"]["along_m"][0] + (rs["mid"]["along_m"][1] - rs["mid"]["along_m"][0]) * self._rand(r)
        along = torch.where(mid, a_mid, a_out)
        lat_off = torch.where(mid, u(rs["mid"]["lat_m"]), u(rs["outside"]["lat_m"]))
        z_off = torch.where(mid, rs["mid"]["z_m"][0] + (rs["mid"]["z_m"][1] - rs["mid"]["z_m"][0]) * self._rand(r), u(rs["outside"]["z_m"]))
        dyaw = torch.where(mid, u(math.radians(rs["mid"]["yaw_deg"])), u(math.radians(rs["outside"]["yaw_deg"])))
        bpos = goal[:, :3] + d * along[:, None] + lat * lat_off[:, None]
        bpos[:, 2] += z_off
        bquat = geo.quat_mul(geo.quat_from_yaw(dyaw), goal[:, 3:7])
        # the canonical centred grasp of a virtual block there, with the arm's tilt; the block then sits in the
        # achieved pinch at the canonical offset plus a grasp offset within the grasp skill's quality gate
        tilt = self.tilt_deg[self.arm]
        pp, pq = grasp_target_world(bpos, bquat, self.arm, tilt)
        t_pb = geo.frame_relative(pp, pq, bpos, bquat)
        gw = rs["grasp_offset"]
        slip = torch.stack([u(gw["x_m"]), torch.zeros(r, device=dev), u(gw["z_m"])], -1)
        ey = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(r, 3)
        srot = geo.quat_from_axis_angle(ey, u(math.radians(gw["rot_deg"])))
        seed = prior_for_pose(self.arm, "release", bpos, bquat)
        pb_, qb_ = self.base.to_base(pp - geo.quat_apply(pq, slip), pq)
        q_des, pe, re = self.kin.ik(seed, pb_, qb_, iters=60, q_ref=seed)
        pa_b, qa_b = self.kin.fk(q_des, "pinch")
        pa, qa = self.base.to_world(pa_b, qa_b)
        bp, bq = geo.frame_compose(pa, qa, t_pb[0] + slip, geo.quat_mul(srot, t_pb[1]))
        self._t5_mov[torch.arange(r, device=dev), INSERT_SLOT] = torch.cat([bp, bq], -1)
        self._t5_present[:, INSERT_SLOT] = True
        self._t5_write(env_ids)
        # arm holding the block (fingers on its faces, the 3 N-safe closed command, payload compensated)
        self._set_arm(env_ids, q_des)
        cw = self.ig["held_close_width_m"]
        self.grip.cfg.closed_width = cw
        self.robot.write_joint_state_to_sim(q_full(q_des, torch.full((r, 1), 0.0497, device=dev)), torch.zeros(r, 10, device=dev), env_ids=env_ids)
        q_cmd = self.kin.sag_compensated_command(q_des, torch.full((r,), self.ctrl.cfg.payload_mass, device=dev))
        lim = self.kin.limits
        q_cmd = torch.max(torch.min(q_cmd, lim[:, 1]), lim[:, 0])
        self.q_cmd[env_ids] = q_cmd
        self.q_cmd_prev[env_ids] = q_cmd
        self.w_cmd[env_ids] = cw
        self.w_cmd_prev[env_ids] = cw
        self.grip.reset(env_ids, width=cw, closed=True)
        self.robot.set_joint_position_target(q_full(q_cmd, torch.full((r, 1), cw, device=dev)), env_ids=env_ids)
        self.actions[env_ids, 6] = 1.0
        self.released[env_ids] = False
        self.since_release[env_ids] = 0
        self.settled[env_ids] = 0
        self.z_release[env_ids] = 0
        self.prev_along[env_ids] = along
        self.prev_tpos[env_ids] = bp

    def _action_scale(self):
        ac = self.sk["action"]
        npos, _ = self.noisy_block_poses()
        along = ((npos[:, INSERT_SLOT] - self.slot_pos) * self.axis).sum(-1)
        # small steps once the block is near / in the slot entrance (its leading end within 3 cm of the tower face)
        near = (along > -0.18).float()
        return ac["pos_far_m"] + (ac["pos_near_m"] - ac["pos_far_m"]) * near

    def _payload_kg(self):
        return self.grip.closed.float() * self.ctrl.cfg.payload_mass

    def _grip_mask(self, close):
        # no release before the (perceived) block is in the slot: from a random start the actor opened at once, the
        # block fell outside and PPO settled on failing fast (v1: 0 % at it. 79, episodes 4 steps long)
        rmin = self.sk["insert"].get("release_min_along_m")
        if rmin is None:
            return close
        npos, _ = self.noisy_block_poses()
        along = ((npos[:, INSERT_SLOT] - self.slot_pos) * self.axis).sum(-1)
        return close | ((along < rmin) & ~self.released)

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        q, _ = self.measured()
        pw = self.pinch(q)[2]
        newly = (~self.released) & (~self.grip.closed)
        self.z_release = torch.where(newly, pw[:, 2], self.z_release)
        self.released |= newly

    def _get_dones(self):
        s = self.st = self._common_state()
        ins, tm = self.sk["insert"], self.sk["termination"]
        self.since_release = torch.where(self.released, self.since_release + 1, self.since_release)
        retreated = self.released & (s["pw"][:, 2] >= self.z_release + ins["retreat_m"])
        static = (s["tpos"] - self.prev_tpos).norm(dim=-1) / self.step_dt < 0.01
        self.prev_tpos = s["tpos"].clone()
        in_slot = (s["along"] >= ins["along_min_m"]) & (s["along"] <= ins["along_max_m"]) & (s["lat"].abs() <= ins["lat_m"]) & \
            (s["tilt"] <= math.radians(ins["tilt_deg"]))
        self.settled = torch.where(self.released & retreated & in_slot & static, self.settled + 1, torch.zeros_like(self.settled))
        success = (self.settled >= ins["settle_steps"]) & (s["moved"] <= ins["structure_ok_m"])
        broke = s["moved"] > tm["structure_m"]
        dropped = (s["vert"] < -tm["drop_m"]) | (s["tilt"] > math.radians(tm["tilt_deg"]))
        short = self.released & (self.since_release >= ins["settle_steps"]) & (s["along"] < ins["along_min_m"] - 0.02)
        failed = (broke | dropped | short) & ~success
        self.term = dict(success=success, failed=failed)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        done = success | failed | time_out
        self.extras["log"] = {"step_along_mm": float(s["along"].mean() * 1000)}
        if done.any():
            dn = done.float().sum()
            self.extras["log"] |= {"success_rate": float((success & done).float().sum() / dn),
                                   "failed_rate": float((failed & done).float().sum() / dn),
                                   "broke_rate": float((broke & done).float().sum() / dn),
                                   "timeout_rate": float((time_out & ~(success | failed) & done).float().sum() / dn),
                                   "final_along_mm": float(s["along"][done].mean() * 1000),
                                   "final_lat_mm": float(s["lat"][done].abs().mean() * 1000)}
        return success | failed, time_out

    def _get_rewards(self):
        s = self.st
        rw, ins = self.sk["reward"], self.sk["insert"]
        # aim ``entry_lift_m`` above the goal height while entering (demo +1 mm: at the goal height the block's bottom
        # face is flush with the floor's top edge and catches it)
        lift = self.ig["entry_lift_m"][self.sk["scene"].get("collider", "hull")] * (~self.released).float()
        err = torch.stack([s["lat"], s["vert"] - lift, 0.075 * torch.sin(s["dyaw"])], -1).norm(dim=-1)  # yaw as tip offset
        r_align = squash(err, 300.0, 2.0) + squash(err, 1000.0, 0.0)
        aligned = (err < ins["align_gate_m"]).float()
        target = -self.ig["entry_m"]
        a_c = s["along"].clamp(max=target)
        r_prog = rw["progress"] * (a_c - self.prev_along.clamp(max=target)) * torch.where(a_c > self.prev_along, aligned, torch.ones_like(aligned))
        self.prev_along = s["along"].detach().clone()
        released_ok = self.released & (s["along"] >= ins["along_min_m"])
        pen_struct = ((s["moved"] - 0.001).clamp(min=0) / 0.01)
        lim = self.kin.limits
        margin = torch.minimum(s["q"] - lim[:, 0], lim[:, 1] - s["q"])
        pen_jl = ((0.1 - margin) / 0.1).clamp(0, 1).sum(-1)
        rate = (self.actions - self.prev_actions).pow(2).sum(-1)
        remaining = (self.max_episode_length - self.episode_length_buf).float()
        r = (rw["align"] * r_align * (~self.released).float() + r_prog + rw["release_ok"] * released_ok.float()
             - rw["structure_per_cm"] * pen_struct - rw["action_rate"] * rate - rw["joint_limit"] * pen_jl)
        r = r + self.term["success"].float() * (rw["success"] + rw["success_per_remaining_step"] * remaining)
        r = r - self.term["failed"].float() * rw["fail"]
        return r

    def _get_observations(self):
        s = self.st = self._common_state()
        return self._obs(s)


# ---------------------------------------------------------------------------------------------------------- push_in
@configclass
class PushInEnvCfg(SkillEnvCfg):
    skill_yaml: str = "skills/push_in"
    action_space: int = 6
    observation_space: int = ACTOR_DIM
    state_space: int = CRITIC_DIM


class PushInEnv(_Task5Common):
    """Push home: the released block rests part-way in the slot (centre ~``entry_m`` out, wedged), the fingers at the
    pusher width behind its outer end face (or above it, at the pre-push-in pose); push it along the slot until its
    centre is at the slot centre. Success: |along| <= 2 mm, across <= 2.5 mm, tilt <= 5 deg, the tower moved <= 3 mm,
    static 0.3 s. Failure: overshoot (> 8 mm past the centre), the tower moved > 10 mm, the block dropped / tipped,
    timeout. Action: 6-D delta pose (the fingers stay at the pusher width)."""

    cfg: PushInEnvCfg

    def __init__(self, cfg: PushInEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self._t5_init()
        self.prev_along = torch.zeros(n, device=dev)
        self.prev_tpos = torch.zeros(n, 3, device=dev)
        self.settled = torch.zeros(n, dtype=torch.long, device=dev)
        self.tilt_deg = {a: grasp_cfg()["approach_tilt_deg"][a] for a in ("left", "back")}

    def _phase(self):
        n = self.num_envs
        return torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(n, 3)  # "released" (the pusher phase)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        r, dev = len(env_ids), self.device
        goal, d = self._tower_reset(env_ids)
        rs = self.sk["reset"]
        u = lambda a: (self._rand(r) * 2 - 1) * a  # noqa: E731
        lat = torch.stack([-d[:, 1], d[:, 0], torch.zeros(r, device=dev)], -1)
        lo, hi = rs["block_along_m"]
        along = lo + (hi - lo) * self._rand(r)
        bpos = goal[:, :3] + d * along[:, None] + lat * u(rs["block_lat_m"])[:, None]
        bquat = geo.quat_mul(geo.quat_from_yaw(u(math.radians(rs["block_yaw_deg"]))), goal[:, 3:7])
        ar = torch.arange(r, device=dev)
        self._t5_mov[ar, INSERT_SLOT] = torch.cat([bpos, bquat], -1)
        self._t5_present[:, INSERT_SLOT] = True
        self._t5_write(env_ids)
        # pusher: at the pre-push-in pose (behind and above the outer end face) or near the end face, at centre height
        _, gq = grasp_target_world(bpos, goal[:, 3:7], self.arm, self.tilt_deg[self.arm])
        near = self._rand(r) < rs["near_frac"]
        pa = bpos + d * self.ig["prepushin_along_m"]
        pa[:, 2] = bpos[:, 2] + self.ig["prepushin_up_m"]
        pn = bpos + d * (rs["near_along_m"][0] + (rs["near_along_m"][1] - rs["near_along_m"][0]) * self._rand(r))[:, None]
        pn[:, 2] = bpos[:, 2] + u(rs["near_z_m"])
        p0 = torch.where(near[:, None], pn, pa) + lat * u(rs["lat_m"])[:, None]
        p0[:, 2] += torch.where(near, torch.zeros(r, device=dev), u(rs["z_m"]))
        q0 = geo.normalize(geo.quat_mul(geo.quat_from_axis_angle(self._randn(r, 3), u(math.radians(rs["rot_deg"]))), gq))
        seed = prior_for_pose(self.arm, "release", bpos, goal[:, 3:7])
        pb_, qb_ = self.base.to_base(p0, q0)
        q_des, _, _ = self.kin.ik(seed, pb_, qb_, iters=60, q_ref=seed)
        self._set_arm(env_ids, q_des)
        pw_ = self.ig["pusher_width_m"]
        self.robot.write_joint_state_to_sim(q_full(q_des, torch.full((r, 1), pw_, device=dev)), torch.zeros(r, 10, device=dev), env_ids=env_ids)
        self.w_cmd[env_ids] = pw_
        self.w_cmd_prev[env_ids] = pw_
        self.robot.set_joint_position_target(q_full(self.q_cmd[env_ids], torch.full((r, 1), pw_, device=dev)), env_ids=env_ids)
        self.prev_along[env_ids] = along
        self.prev_tpos[env_ids] = bpos
        self.settled[env_ids] = 0

    def _action_scale(self):
        ac = self.sk["action"]
        npos, _ = self.noisy_block_poses()
        tp = npos[:, INSERT_SLOT]
        cp = tp + self.axis * self.ig["pushin_contact_along_m"]
        dd = (self.ctrl.pos - cp).norm(dim=-1)
        a = ((dd - ac["near_radius_m"]) / (ac["far_radius_m"] - ac["near_radius_m"])).clamp(0, 1)
        return ac["pos_near_m"] + (ac["pos_far_m"] - ac["pos_near_m"]) * a

    def _pre_physics_step(self, actions: torch.Tensor):
        a = actions.clamp(-1.0, 1.0)
        self.prev_actions[:] = self.actions
        self.actions[:] = a
        ac = self.sk["action"]
        self.ema_actions = ac["ema"] * a[:, :6] + (1 - ac["ema"]) * self.ema_actions
        q_meas, _ = self.measured()
        self.ctrl.apply_delta(self.base.vec_to_world(self.ema_actions[:, :3] * self._action_scale()[:, None]),
                              self.base.vec_to_world(self.ema_actions[:, 3:6] * ac["rot_rad"]), q_meas=q_meas, max_lead=ac["max_lead_m"])
        self.q_cmd_prev[:] = self.q_cmd
        self.w_cmd_prev[:] = self.w_cmd
        self.q_cmd[:] = self.ctrl.command(q_meas, payload_kg=0.0)
        self.substep = 0

    def _get_dones(self):
        s = self.st = self._common_state()
        pi, tm = self.sk["push_in"], self.sk["termination"]
        static = (s["tpos"] - self.prev_tpos).norm(dim=-1) / self.step_dt < 0.01
        self.prev_tpos = s["tpos"].clone()
        home = (s["along"].abs() <= pi["along_tol_m"]) & (s["lat"].abs() <= pi["lat_m"]) & (s["tilt"] <= math.radians(pi["tilt_deg"]))
        self.settled = torch.where(home & static, self.settled + 1, torch.zeros_like(self.settled))
        success = (self.settled >= pi["settle_steps"]) & (s["moved"] <= pi["structure_ok_m"])
        broke = s["moved"] > tm["structure_m"]
        over = s["along"] > pi["overshoot_m"]
        dropped = (s["vert"] < -tm["drop_m"]) | (s["tilt"] > math.radians(tm["tilt_deg"]))
        failed = (broke | over | dropped) & ~success
        self.term = dict(success=success, failed=failed)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        done = success | failed | time_out
        self.extras["log"] = {"step_along_mm": float(s["along"].mean() * 1000)}
        if done.any():
            dn = done.float().sum()
            self.extras["log"] |= {"success_rate": float((success & done).float().sum() / dn),
                                   "failed_rate": float((failed & done).float().sum() / dn),
                                   "broke_rate": float((broke & done).float().sum() / dn),
                                   "overshoot_rate": float((over & done).float().sum() / dn),
                                   "timeout_rate": float((time_out & ~(success | failed) & done).float().sum() / dn),
                                   "final_along_mm": float(s["along"][done].mean() * 1000)}
        return success | failed, time_out

    def _get_rewards(self):
        s = self.st
        rw = self.sk["reward"]
        a_c = s["along"].clamp(max=0.0)
        r_prog = rw["progress"] * (a_c - self.prev_along.clamp(max=0.0))
        self.prev_along = s["along"].detach().clone()
        cp = s["tpos"] + self.axis * self.ig["pushin_contact_along_m"]
        r_align = squash((s["pw"] - cp).norm(dim=-1), 50.0, 2.0)
        pen_over = (s["along"] - 0.002).clamp(min=0) / 0.01
        pen_struct = ((s["moved"] - 0.001).clamp(min=0) / 0.01)
        lim = self.kin.limits
        margin = torch.minimum(s["q"] - lim[:, 0], lim[:, 1] - s["q"])
        pen_jl = ((0.1 - margin) / 0.1).clamp(0, 1).sum(-1)
        rate = (self.actions - self.prev_actions).pow(2).sum(-1)
        remaining = (self.max_episode_length - self.episode_length_buf).float()
        r = (rw["align"] * r_align + r_prog - rw["overshoot_per_cm"] * pen_over - rw["structure_per_cm"] * pen_struct
             - rw["action_rate"] * rate - rw["joint_limit"] * pen_jl)
        r = r + self.term["success"].float() * (rw["success"] + rw["success_per_remaining_step"] * remaining)
        r = r - self.term["failed"].float() * rw["fail"]
        return r

    def _get_observations(self):
        s = self.st = self._common_state()
        return self._obs(s)
