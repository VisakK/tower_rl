"""Shared machinery of the contact-skill envs (light skill scene, one arm, 10 Hz policy). Import after ``AppLauncher``.

* Control: actions move the shared :class:`tower_rl.control.ArmController` target (IK + gravity-sag compensation);
  one joint row per 10 Hz step is held for 12 physics substeps (3 x 30 Hz ticks), applied after the evaluator's
  1-substep command delay, exactly as the evaluator executes a row. The gripper is binary with rate-limited widths.
* Towers: each env samples a (variant, stage) pair handled by its arm, a domain tier and a source-tower sample
  (``tower_rl.tower``); movable blocks occupy slots m0..m5, unused slots are parked.
* Perception stand-in: actor block poses carry a per-episode offset plus per-step jitter.
"""

from __future__ import annotations

import math

import torch

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils import configclass

from tower_rl import geometry as geo
from tower_rl.config import load_yaml
from tower_rl.control import ArmController, ArmControllerCfg, GripperController
from tower_rl.grasp import gripper_cfg
from tower_rl.data.layouts import VARIANTS
from tower_rl.domain import DomainSampler
from tower_rl.kinematics import ArmBase, ArmKinematics
from tower_rl.sim.assets import q_full, sim_cfg
from tower_rl.sim.skill_scene import FOUNDATION_SLOTS, MAX_MOVABLE, make_skill_scene_cfg, park_pose
from tower_rl.tower import foundation_poses, load_spec, present_after, rest_heights, source_poses


def rot6d(q: torch.Tensor) -> torch.Tensor:
    """First two columns of the rotation matrix (continuous rotation encoding)."""
    m = geo.quat_to_mat(q)
    return torch.cat([m[..., :, 0], m[..., :, 1]], dim=-1)


def squash(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Factory keypoint squashing ``1 / (exp(a x) + b + exp(-a x))``."""
    return 1.0 / (torch.exp(a * x) + b + torch.exp(-a * x))


@configclass
class SkillEnvCfg(DirectRLEnvCfg):
    skill_yaml: str = "skills/grasp_lift"
    arm: str = "left"
    decimation: int = 12
    episode_length_s: float = 15.0
    action_space: int = 7
    observation_space: int = 1
    state_space: int = 1
    is_finite_horizon: bool = False

    def setup(self, num_envs: int | None = None, device: str = "cuda:0", overrides: dict | None = None):
        """Fill sim/scene from the YAML config (call before creating the env)."""
        from tower_rl.config import deep_update
        self.skill = deep_update(load_yaml(self.skill_yaml), overrides or {})
        self.arm = self.skill["arm"]
        phys = load_yaml("physics")
        self.decimation = phys["control_substeps"] * phys["policy_ticks"]
        self.episode_length_s = self.skill["episode_steps"] * self.decimation * phys["dt"]
        self.sim = sim_cfg(phys, device=device, render_interval=self.decimation)
        n = num_envs or self.skill["num_envs"]
        sc = self.skill.get("scene", {})  # Task 4/5 skills: more slots, 3 N fingers, 32 mm side blocks, cuboid colliders
        self.scene = make_skill_scene_cfg(self.arm, n, max_movable=sc.get("max_movable", MAX_MOVABLE),
                                          finger_effort=sc.get("finger_effort_n"),
                                          slot_dims={int(k): v for k, v in (sc.get("slot_dims") or {}).items()},
                                          slot_mass={int(k): v for k, v in (sc.get("slot_mass") or {}).items()},
                                          collider=sc.get("collider", "hull"))
        return self


class SkillEnvBase(DirectRLEnv):
    cfg: SkillEnvCfg

    def __init__(self, cfg: SkillEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        sk = self.cfg.skill
        self.sk = sk
        self.phys = load_yaml("physics")
        n, dev = self.num_envs, self.device
        self.arm = cfg.arm
        self.robot = self.scene["robot"]
        self.blocks = self.scene["blocks"]
        self.fingers = [self.scene["finger1_contact"], self.scene["finger2_contact"]]
        names = list(self.blocks.object_names)
        self.slot_names = [s for s in names if s not in FOUNDATION_SLOTS]  # m0, m1, ... (6 for Tasks 1-3 skills)
        self.n_slots = S = len(self.slot_names)
        self.found_idx = torch.tensor([names.index(s) for s in FOUNDATION_SLOTS], device=dev)
        self.slot_idx = torch.tensor([names.index(s) for s in self.slot_names], device=dev)
        self.park = torch.tensor([park_pose(i) for i in range(S)], device=dev)
        self.kin = ArmKinematics(device=dev)
        self.kin.set_arm(self.arm)
        self.base = ArmBase(self.arm, device=dev)
        self.ctrl = ArmController(self.arm, n, device=dev, cfg=ArmControllerCfg(), kin=self.kin)
        self.grip = GripperController(n, device=dev, cfg=gripper_cfg())
        self.rz = rest_heights(dev)
        # (variant, stage) pairs this arm executes (skill YAML ``variants``: default the Tasks 1-3 transfer variants)
        self.specs = [load_spec(v, device=dev) for v in sk.get("variants", VARIANTS)]
        self.pairs = self._pairs()
        self.pair_v = torch.tensor([p[0] for p in self.pairs], device=dev)
        self.pair_k = torch.tensor([p[1] for p in self.pairs], device=dev)
        self.samplers = {t: DomainSampler(t, device=dev, max_blocks=max(6, S)) for t in sk["tiers"]}
        self.tier_names = list(sk["tiers"])
        self.tier_p = torch.tensor([sk["tiers"][t] for t in self.tier_names], device=dev)
        # per-env episode state
        self.env_case = [""] * n  # robustness case of each env's current episode (for evaluation breakdowns)
        self.env_variant = torch.zeros(n, dtype=torch.long, device=dev)
        self.env_stage = torch.zeros(n, dtype=torch.long, device=dev)
        self.target_slot = torch.zeros(n, dtype=torch.long, device=dev)
        self.present = torch.zeros(n, S, dtype=torch.bool, device=dev)
        self.block_p0 = torch.zeros(n, S, 3, device=dev)
        self.noise_pos = torch.zeros(n, S, 3, device=dev)
        self.noise_yaw = torch.zeros(n, S, device=dev)
        self._noisy_cache = None  # (env step, pos, quat): one noise draw per env step
        self._frozen = torch.zeros(n, S, dtype=torch.long, device=dev)  # dropout steps left per block
        self.mass_scale = torch.ones(n, S, device=dev)
        self.block_fric = torch.zeros(n, 2, device=dev)
        self.finger_fric = torch.zeros(n, 2, device=dev)
        self.actions = torch.zeros(n, cfg.action_space, device=dev)
        self.prev_actions = torch.zeros_like(self.actions)
        self.ema_actions = torch.zeros(n, 6, device=dev)
        self.q_cmd = torch.zeros(n, 7, device=dev)
        self.q_cmd_prev = torch.zeros(n, 7, device=dev)
        self.w_cmd = torch.full((n,), 0.1, device=dev)
        self.w_cmd_prev = torch.full((n,), 0.1, device=dev)
        self.substep = 0
        self.delay = self.phys["command_delay_substeps"][self.arm]
        self.prev_pinch_b = torch.zeros(n, 3, device=dev)
        self.prev_pinch_qb = torch.zeros(n, 4, device=dev)
        self.prev_pinch_qb[:, 0] = 1
        self.gen = torch.Generator(device=dev).manual_seed(int(cfg.seed or 0) + 12345)
        self._default_mass = None

    def _pairs(self) -> list[tuple[int, int]]:
        """(variant index, stage) pairs this arm executes: the transfers of ``self.specs`` assigned to the arm."""
        return [(vi, k) for vi, s in enumerate(self.specs) for k, a in enumerate(s.transfer_arm) if a == self.arm]

    # ------------------------------------------------------------------ helpers
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.gen, device=self.device)

    def block_poses_local(self):
        """True movable-slot poses (env-local station frame): ``pos [N,S,3]``, ``quat [N,S,4]``."""
        pos = self.blocks.data.object_link_pos_w[:, self.slot_idx] - self.scene.env_origins[:, None]
        return pos, self.blocks.data.object_link_quat_w[:, self.slot_idx]

    def noisy_block_poses(self):
        """Actor's view of the movable blocks: true pose + per-episode offset + per-step jitter (yaw only); with
        ``obs_noise.outlier_p`` one-step jumps of ``outlier_m`` and with ``obs_noise.dropout_p`` dropouts that freeze a
        block's pose for ``dropout_steps`` (the Step 8.1 perception model; both off by default). One draw per env step:
        the observation and the next action's step scaling see the same estimate, as the executor does."""
        step = int(self.common_step_counter)
        if self._noisy_cache is not None and self._noisy_cache[0] == step:
            return self._noisy_cache[1], self._noisy_cache[2]
        pos, quat = self.block_poses_local()
        on = self.sk["obs_noise"]
        p = pos + self.noise_pos + self._randn(*pos.shape) * on["step_pos_m"]
        dyaw = self.noise_yaw + self._randn(*self.noise_yaw.shape) * math.radians(on["step_yaw_deg"])
        q = geo.quat_mul(geo.quat_from_yaw(dyaw), quat)
        if on.get("outlier_p", 0.0) > 0:
            out = self._rand(*p.shape[:2]) < on["outlier_p"]
            d = self._randn(*p.shape)
            p = p + out[..., None] * d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9) * on["outlier_m"]
        if on.get("dropout_p", 0.0) > 0:
            # a dropout can only start with a pose of the current episode to freeze
            fresh = (self.episode_length_buf > 0)[:, None]
            start = (self._rand(*p.shape[:2]) < on["dropout_p"]) & (self._frozen == 0) & fresh
            self._frozen = torch.where(start, on["dropout_steps"], self._frozen)
            self._frozen = torch.where(fresh, self._frozen, torch.zeros_like(self._frozen))
            fr = self._frozen > 0
            if self._noisy_cache is not None:
                p = torch.where(fr[..., None], self._noisy_cache[1], p)
                q = torch.where(fr[..., None], self._noisy_cache[2], q)
            self._frozen = (self._frozen - 1).clamp(min=0)
        self._noisy_cache = (step, p, q)
        return p, q

    def measured(self):
        q = self.robot.data.joint_pos[:, :7]
        w = self.robot.data.joint_pos[:, 8] - self.robot.data.joint_pos[:, 9]
        return q, w

    def pinch(self, q7):
        """Measured pinch pose in the base frame and in the env-local station frame."""
        pb, qb = self.kin.fk(q7, "pinch")
        pw, qw = self.base.to_world(pb, qb)
        return pb, qb, pw, qw

    def finger_forces(self, slot: torch.Tensor) -> torch.Tensor:
        """Contact force magnitudes ``[N, 2]`` of each finger with the block in ``slot``."""
        ar = torch.arange(self.num_envs, device=self.device)
        return torch.stack([f.data.force_matrix_w[ar, 0, slot].norm(dim=-1) for f in self.fingers], -1)

    def finger_any_other(self, slot: torch.Tensor) -> torch.Tensor:
        """Total finger contact force with movable blocks other than ``slot``."""
        tot = torch.zeros(self.num_envs, device=self.device)
        mask = torch.ones(self.num_envs, self.n_slots, device=self.device)
        mask[torch.arange(self.num_envs, device=self.device), slot] = 0
        for f in self.fingers:
            tot += (f.data.force_matrix_w[:, 0].norm(dim=-1) * mask).sum(-1)
        return tot

    # ------------------------------------------------------------------ physics properties
    def _apply_dynamics(self, env_ids: torch.Tensor):
        """Per-env block mass/inertia/friction and finger friction. The CPU-side property tensors are read from PhysX
        once and cached (reading them back every reset dominated the step time)."""
        view = self.blocks.root_physx_view
        rv = self.robot.root_physx_view
        n = self.num_envs
        if self._default_mass is None:
            self._default_mass = view.get_masses().clone()
            self._default_inertia = view.get_inertias().clone()
            self._masses = self._default_mass.clone()
            self._inertias = self._default_inertia.clone()
            self._block_mat = view.get_material_properties().clone()
            self._robot_mat = rv.get_material_properties().clone()
        eid = env_ids.cpu()
        slots = self.slot_idx.cpu()
        ids = (slots[:, None] * n + eid[None]).flatten()  # object-major view ids
        scale = self.mass_scale[env_ids].T.reshape(-1).cpu()
        self._masses[ids] = self._default_mass[ids] * scale.reshape(self._masses[ids].shape)
        self._inertias[ids] = self._default_inertia[ids] * scale[:, None]
        view.set_masses(self._masses, ids)
        view.set_inertias(self._inertias, ids)
        fr = self.block_fric[env_ids].cpu().repeat(len(slots), 1)  # same friction for all slots of an env
        self._block_mat[ids, :, 0] = fr[:, 0:1]
        self._block_mat[ids, :, 1] = fr[:, 1:2]
        view.set_material_properties(self._block_mat, ids)
        ff = self.finger_fric[env_ids].cpu()
        self._robot_mat[eid, :, 0] = ff[:, 0:1]
        self._robot_mat[eid, :, 1] = ff[:, 1:2]
        rv.set_material_properties(self._robot_mat, eid)

    # ------------------------------------------------------------------ tower reset
    def _sample_towers(self, env_ids: torch.Tensor):
        """Variant/stage/domain per env, block poses written; returns the (true) target block pose."""
        r = len(env_ids)
        dev = self.device
        pi = torch.randint(len(self.pairs), (r,), generator=self.gen, device=dev)
        self.env_variant[env_ids] = self.pair_v[pi]
        self.env_stage[env_ids] = self.pair_k[pi]
        # domain: per env tier
        tier = torch.multinomial(self.tier_p, r, replacement=True, generator=self.gen)
        doms = {t: self.samplers[t].sample(r, self.gen) for t in self.tier_names}
        poses = torch.zeros(r, len(self.blocks.object_names), 7, device=dev)
        mov = torch.zeros(r, self.n_slots, 7, device=dev)
        present = torch.zeros(r, self.n_slots, dtype=torch.bool, device=dev)
        target = torch.zeros(r, dtype=torch.long, device=dev)
        mass = torch.ones(r, self.n_slots, device=dev)
        bfric = torch.zeros(r, 2, device=dev)
        ffric = torch.zeros(r, 2, device=dev)
        found = torch.zeros(r, 6, 7, device=dev)
        for vi, spec in enumerate(self.specs):
            sel = (self.env_variant[env_ids] == vi).nonzero(as_tuple=True)[0]
            if len(sel) == 0:
                continue
            m = spec.num_movable
            stage = self.env_stage[env_ids][sel]
            for ti, t in enumerate(self.tier_names):
                s2 = sel[(tier[sel] == ti)]
                if len(s2) == 0:
                    continue
                d = doms[t].subset(s2)
                for j, e in enumerate(env_ids[s2].tolist()):
                    self.env_case[e] = f"{t}/{d.case[j]}"
                mov[s2, :m] = source_poses(spec, d, self.rz)
                mass[s2, :m] = d.mass_scale[:, :m]
                bfric[s2] = d.friction[:, 0]
                ffric[s2] = d.finger_friction
            present[sel, :m] = present_after(spec, stage)
            target[sel] = spec.transfer_block[stage]
            found[sel] = foundation_poses(spec, len(sel), jitter_m=0.003, gen=self.gen)
        mov[..., 2] += 0.0005
        mov = torch.where(present[..., None], mov, self.park[None].expand(r, -1, -1))
        poses[:, self.found_idx] = found
        poses[:, self.slot_idx] = mov
        world = poses.clone()
        world[..., :3] += self.scene.env_origins[env_ids][:, None]
        self.blocks.write_object_link_pose_to_sim(world, env_ids=env_ids)
        self.blocks.write_object_com_velocity_to_sim(torch.zeros(r, self.n_slots, 6, device=dev), env_ids=env_ids,
                                                     object_ids=self.slot_idx)
        self.present[env_ids] = present
        self.target_slot[env_ids] = target
        self.block_p0[env_ids] = mov[..., :3]
        self.mass_scale[env_ids] = mass
        self.block_fric[env_ids] = bfric
        self.finger_fric[env_ids] = ffric
        on = self.sk["obs_noise"]
        self.noise_pos[env_ids] = self._randn(r, self.n_slots, 3) * on["episode_pos_m"]
        self.noise_pos[env_ids, :, 2] *= 0.5
        self.noise_yaw[env_ids] = self._randn(r, self.n_slots) * math.radians(on["episode_yaw_deg"])
        self._apply_dynamics(env_ids)
        ar = torch.arange(r, device=dev)
        return mov[ar, target, :3], mov[ar, target, 3:7]

    def _set_arm(self, env_ids: torch.Tensor, q_des: torch.Tensor):
        """Arm state at ``q_des`` with the sag-compensated command (no transient), gripper open."""
        r = len(env_ids)
        w = torch.full((r, 1), 0.1, device=self.device)
        self.robot.write_joint_state_to_sim(q_full(q_des, w), torch.zeros(r, 10, device=self.device), env_ids=env_ids)
        q_cmd = self.kin.sag_compensated_command(q_des)
        lim = self.kin.limits
        q_cmd = torch.max(torch.min(q_cmd, lim[:, 1]), lim[:, 0])
        self.q_cmd[env_ids] = q_cmd
        self.q_cmd_prev[env_ids] = q_cmd
        self.w_cmd[env_ids] = 0.1
        self.w_cmd_prev[env_ids] = 0.1
        self.robot.set_joint_position_target(q_full(q_cmd, w), env_ids=env_ids)
        # controller state for these envs
        self.ctrl.q_des[env_ids] = q_des
        self.ctrl.q_ref[env_ids] = q_des
        pw, qw = self.ctrl.pinch_world(q_des)
        self.ctrl.pos[env_ids], self.ctrl.quat[env_ids] = pw, qw
        self.ctrl.i_pos[env_ids] = 0
        self.ctrl.i_rot[env_ids] = 0
        self.grip.reset(env_ids)
        pb, qb = self.kin.fk(q_des, "pinch")
        self.prev_pinch_b[env_ids], self.prev_pinch_qb[env_ids] = pb, qb
        self.actions[env_ids] = 0
        self.prev_actions[env_ids] = 0
        self.ema_actions[env_ids] = 0

    # ------------------------------------------------------------------ actions
    def _action_scale(self) -> torch.Tensor:
        """Per-env max translation per step (subclasses: smaller near the contact region)."""
        return torch.full((self.num_envs,), self.sk["action"]["pos_far_m"], device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor):
        a = actions.clamp(-1.0, 1.0)
        self.prev_actions[:] = self.actions
        self.actions[:] = a
        ac = self.sk["action"]
        self.ema_actions = ac["ema"] * a[:, :6] + (1 - ac["ema"]) * self.ema_actions
        dpos_b = self.ema_actions[:, :3] * self._action_scale()[:, None]
        drot_b = self.ema_actions[:, 3:6] * ac["rot_rad"]
        q_meas, _ = self.measured()
        self.ctrl.apply_delta(self.base.vec_to_world(dpos_b), self.base.vec_to_world(drot_b), q_meas=q_meas,
                              max_lead=ac["max_lead_m"])
        close = self._grip_mask(a[:, 6] > ac.get("grip_threshold", 0.0))
        self.q_cmd_prev[:] = self.q_cmd
        self.w_cmd_prev[:] = self.w_cmd
        self.q_cmd[:] = self.ctrl.command(q_meas, payload_kg=self._payload_kg())
        self.w_cmd[:] = self.grip.step(close)
        self.substep = 0

    def _payload_kg(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _grip_mask(self, close: torch.Tensor) -> torch.Tensor:
        """Skills may veto the gripper action (the insert keeps the block until it is in the slot)."""
        return close

    def _apply_action(self):
        if self.substep < self.delay:
            q, w = self.q_cmd_prev, self.w_cmd_prev
        else:
            q, w = self.q_cmd, self.w_cmd
        self.robot.set_joint_position_target(q_full(q, w[:, None]))
        self.substep += 1
