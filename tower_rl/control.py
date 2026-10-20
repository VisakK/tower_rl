"""10 Hz arm and gripper controllers shared by the skill envs, the transit controller and the executor (pure torch).

One call produces one action row (absolute joint targets + gripper width) that the evaluator holds for 0.1 s, so the
same code runs in training, in rehearsal and behind the policy server:

* :class:`ArmController` keeps a Cartesian target for the pinch point (station frame). Each row it solves damped
  least-squares IK seeded with the previous solution (so the posture family is kept) and returns the
  gravity-sag-compensated command ``q_des - tau_g(q_des) / kp``; the measured arm then settles on ``q_des``.
  Optional integral correction removes the residual (joint friction) using the measured joint angles.
* :class:`GripperController` turns a binary open/close request into a rate-limited width command (demos: close
  0.09 m/s, 0.03 m/s in the contact zone; open 0.16 m/s).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tower_rl import geometry as geo
from tower_rl.kinematics import ArmBase, ArmKinematics


@dataclass
class ArmControllerCfg:
    ik_iters: int = 4
    ik_damping: float = 0.02
    ik_rot_weight: float = 0.5
    null_gain: float = 0.05
    limit_margin: float = 0.03          # rad kept from every joint limit
    max_joint_step: float = 0.15        # rad per 10 Hz row (1.5 rad/s)
    integral_gain: float = 0.0          # 0 disables; fraction of the measured pinch error added to the IK target per row
    integral_limit: float = 0.01        # m (and rad) bound of the integral correction
    sag_compensation: bool = True
    payload_mass: float = 0.1056        # nominal block mass compensated while holding


class ArmController:
    def __init__(self, arm: str, num_envs: int, device: str = "cpu", cfg: ArmControllerCfg | None = None,
                 kin: ArmKinematics | None = None):
        self.arm = arm
        self.cfg = cfg or ArmControllerCfg()
        self.n = num_envs
        self.device = torch.device(device)
        self.kin = kin or ArmKinematics(device=device)
        self.kin.set_arm(arm)
        self.base = ArmBase(arm, device=device)
        self.q_des = torch.zeros(num_envs, 7, device=self.device)
        self.q_ref = torch.zeros(num_envs, 7, device=self.device)
        self.pos = torch.zeros(num_envs, 3, device=self.device)   # target pinch position, station frame
        self.quat = torch.zeros(num_envs, 4, device=self.device)  # target pinch orientation, station frame
        self.quat[:, 0] = 1
        self.i_pos = torch.zeros(num_envs, 3, device=self.device)  # integral corrections (station frame)
        self.i_rot = torch.zeros(num_envs, 3, device=self.device)

    # ------------------------------------------------------------------ state
    def reset(self, q_meas: torch.Tensor, env_ids: torch.Tensor | None = None, q_ref: torch.Tensor | None = None):
        """Hold the measured configuration: target = FK(q_meas), IK seed = q_meas."""
        ids = slice(None) if env_ids is None else env_ids
        self.q_des[ids] = q_meas
        self.q_ref[ids] = q_meas if q_ref is None else q_ref
        p, q = self.pinch_world(q_meas)
        self.pos[ids], self.quat[ids] = p, q
        self.i_pos[ids] = 0
        self.i_rot[ids] = 0

    def pinch_world(self, q7: torch.Tensor):
        p, q = self.kin.fk(q7, "pinch")
        return self.base.to_world(p, q)

    def set_target(self, pos: torch.Tensor, quat: torch.Tensor, env_ids: torch.Tensor | None = None):
        ids = slice(None) if env_ids is None else env_ids
        self.pos[ids] = pos
        self.quat[ids] = geo.normalize(quat)

    def apply_delta(self, dpos: torch.Tensor, drot: torch.Tensor, q_meas: torch.Tensor | None = None,
                    max_lead: float | None = None):
        """Move the target by ``dpos`` (m) and the world rotation vector ``drot`` (rad).

        ``max_lead`` bounds the distance between target and measured pinch (anti-windup in contact).
        """
        self.pos = self.pos + dpos
        self.quat = geo.normalize(geo.quat_mul(geo.quat_from_rotvec(drot), self.quat))
        if max_lead is not None and q_meas is not None:
            pm, _ = self.pinch_world(q_meas)
            d = self.pos - pm
            dn = d.norm(dim=-1, keepdim=True)
            self.pos = torch.where(dn > max_lead, pm + d * (max_lead / dn.clamp_min(1e-9)), self.pos)

    # ------------------------------------------------------------------ command
    def command(self, q_meas: torch.Tensor | None = None, payload_kg: torch.Tensor | float = 0.0) -> torch.Tensor:
        """Joint command row ``[N, 7]`` for the current target; ``payload_kg`` is compensated at the pinch point
        (callers pass ``holding * cfg.payload_mass``: the real block mass is unknown at test time)."""
        c = self.cfg
        tgt_p, tgt_q = self.pos, self.quat
        if c.integral_gain > 0 and q_meas is not None:
            pm, qm = self.pinch_world(q_meas)
            self.i_pos = (self.i_pos + c.integral_gain * (self.pos - pm)).clamp(-c.integral_limit, c.integral_limit)
            self.i_rot = (self.i_rot + c.integral_gain * geo.quat_error_rotvec(qm, self.quat)).clamp(-c.integral_limit, c.integral_limit)
            tgt_p = self.pos + self.i_pos
            tgt_q = geo.normalize(geo.quat_mul(geo.quat_from_rotvec(self.i_rot), self.quat))
        pb, qb = self.base.to_base(tgt_p, tgt_q)
        q_prev = self.q_des
        q_new, _, _ = self.kin.ik(q_prev, pb, qb, iters=c.ik_iters, damping=c.ik_damping, rot_weight=c.ik_rot_weight,
                                  q_ref=self.q_ref, null_gain=c.null_gain, limit_margin=c.limit_margin, tol_pos=0.0, tol_rot=0.0)
        dq = (q_new - q_prev).clamp(-c.max_joint_step, c.max_joint_step)
        self.q_des = q_prev + dq
        if not c.sag_compensation:
            return self.q_des.clone()
        mass = torch.as_tensor(payload_kg, dtype=self.q_des.dtype, device=self.device)
        q_cmd = self.kin.sag_compensated_command(self.q_des, mass.expand(self.n) if mass.ndim == 0 else mass)
        lim = self.kin.limits
        return torch.max(torch.min(q_cmd, lim[:, 1]), lim[:, 0])

    def pinch_error(self, q_meas: torch.Tensor):
        """(position error m, rotation error rad) between the target and the measured pinch pose."""
        pm, qm = self.pinch_world(q_meas)
        return (self.pos - pm).norm(dim=-1), geo.quat_angle(qm, self.quat)


@dataclass
class GripperCfg:
    open_width: float = 0.1
    closed_width: float = 0.040       # ~4 N per finger on the 0.0497 m block with the evaluator's kp = 800 N/m
    contact_zone_width: float = 0.06  # below this the closing speed drops
    close_speed: float = 0.09         # m/s
    contact_close_speed: float = 0.03
    open_speed: float = 0.16
    dt: float = 0.1                   # one 10 Hz row


class GripperController:
    def __init__(self, num_envs: int, device="cpu", cfg: GripperCfg | None = None):
        self.cfg = cfg or GripperCfg()
        self.width = torch.full((num_envs,), self.cfg.open_width, device=device)
        self.closed = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids=None, width: float | torch.Tensor | None = None, closed: bool = False):
        ids = slice(None) if env_ids is None else env_ids
        self.width[ids] = self.cfg.open_width if width is None else width
        self.closed[ids] = closed

    def step(self, close: torch.Tensor) -> torch.Tensor:
        """Advance one row toward open/closed; returns the width command ``[N]``."""
        c = self.cfg
        self.closed = close.bool()
        fast = self.width > c.contact_zone_width
        down = torch.where(fast, torch.full_like(self.width, c.close_speed * c.dt), torch.full_like(self.width, c.contact_close_speed * c.dt))
        closing = (self.width - down).clamp(min=c.closed_width)
        # a fast step may not cross the contact-zone boundary by more than the slow step would
        closing = torch.where(fast, torch.maximum(closing, torch.full_like(closing, c.contact_zone_width - c.contact_close_speed * c.dt)), closing)
        opening = (self.width + c.open_speed * c.dt).clamp(max=c.open_width)
        self.width = torch.where(self.closed, closing, opening)
        return self.width.clone()
