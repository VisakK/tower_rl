"""Nero arm kinematics and static gravity model in torch (no Isaac imports).

Loaded from ``configs/robot/nero.json`` (``scripts/extract_robot_model.py``): the USD joint chain, link inertials and
the evaluator's drive gains. Everything is batched over a leading dimension and expressed in the arm's ``base_link``
frame unless a function says otherwise; :class:`ArmBase` maps between base and station (world) frames.

Frames:
  * ``link7``  : the recorded ``observations/eef`` frame (validated against the dataset to < 0.01 mm).
  * ``flange`` : ``gripper_flange``.
  * ``pinch``  : the point between the finger pads, ``gripper_base * (0, 0, 0.178)`` (the data generator's
                 ``pinch_center_in_flange_m``; its "flange" is gripper_base). Same rotation as gripper_base: z is the
                 approach axis (wrist -> finger tips), y the jaw closing axis.

Joint motion follows UsdPhysics: ``child = parent * T(pos0, rot0) * Rz(q) * T(pos1, rot1)^-1``.
Static gravity sag of the PD-driven joints: at equilibrium ``kp (q_cmd - q) + tau_g(q) = 0``, so a desired pose
``q_des`` needs ``q_cmd = q_des - tau_g(q_des) / kp`` (:meth:`ArmKinematics.sag_compensated_command`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from tower_rl import geometry as geo
from tower_rl.paths import REPO_ROOT

MODEL_JSON = Path(os.environ.get("TOWER_CONFIG_DIR", REPO_ROOT / "configs")) / "robot" / "nero.json"
GRAVITY = 9.81
_CHAIN_LINKS = ["link1", "link2", "link3", "link4", "link5", "link6", "link7"]
# links that move with each joint i (everything distal); fingers are lumped at their mid-open pose
_DISTAL_LINKS = _CHAIN_LINKS + ["gripper_flange", "gripper_base", "gripper_link1", "gripper_link2"]


def load_model(path: str | Path = MODEL_JSON) -> dict:
    return json.loads(Path(path).read_text())


class ArmKinematics:
    """Batched FK, Jacobians, DLS IK and gravity torques for one Nero arm (both arms share the model)."""

    def __init__(self, model: dict | None = None, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32,
                 arm: str | None = None):
        model = model or load_model()
        self.model = model
        self.device = torch.device(device)
        self.dtype = dtype
        joints = {j["name"]: j for j in model["joints"]}

        def tf(j, which):
            p = torch.tensor(j[f"pos{which}"], dtype=dtype, device=self.device)
            r = geo.quat_to_mat(torch.tensor(j[f"rot{which}_wxyz"], dtype=dtype, device=self.device))
            return p, r

        self.joint_names = model["arm_joints"]
        self._j0 = [tf(joints[n], 0) for n in self.joint_names]
        self._j1inv = []
        for n in self.joint_names:
            p1, r1 = tf(joints[n], 1)
            self._j1inv.append((-(r1.T @ p1), r1.T))
        self.limits = torch.tensor([joints[n]["limits"] for n in self.joint_names], dtype=dtype, device=self.device)
        fl = joints["gripper_flange_joint"]
        self._flange = tf(fl, 0)
        gb = joints["gripper_base_joint"]
        self._gbase = tf(gb, 0)
        self.pinch_offset = torch.tensor(model["tcp"]["pinch_in_gripper_base"], dtype=dtype, device=self.device)
        # inertials (COM in the link frame)
        self.masses = {k: v["mass"] for k, v in model["links"].items()}
        self.coms = {k: torch.tensor(v["com"], dtype=dtype, device=self.device) for k, v in model["links"].items()}
        # finger links sit at +-0.138 along gripper_base z with the fingers half open; their COM is lumped there
        self._finger_mount = torch.tensor([0.0, 0.0, 0.138], dtype=dtype, device=self.device)
        # PhysX's default 1 kg on the massless passive gripper_link, at its slider position along gripper_base z
        vg = model.get("virtual_gripper_link", {"mass": 0.0, "slider_pos": 0.0})
        self.virtual_mass = float(vg["mass"])
        self.virtual_slider = torch.tensor([0.0, 0.0, float(vg["slider_pos"])], dtype=dtype, device=self.device)
        self.kp = None
        if arm is not None:
            self.set_arm(arm)

    def set_arm(self, arm: str):
        d = self.model["drives"][arm]
        self.kp = torch.tensor(d["kp"], dtype=self.dtype, device=self.device)
        self.arm = arm
        return self

    # ------------------------------------------------------------------ forward kinematics
    def chain(self, q: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Link frames ``name -> (pos [N,3], rot [N,3,3])`` in base_link for joint angles ``q [N,7]``.

        Also returns the joint frames ``jointK`` (origin and rotation whose z axis is the joint axis).
        """
        q = q.to(self.device, self.dtype)
        n = q.shape[0]
        p = torch.zeros(n, 3, dtype=self.dtype, device=self.device)
        r = torch.eye(3, dtype=self.dtype, device=self.device).expand(n, 3, 3)
        out = {"base_link": (p, r)}
        for i, name in enumerate(_CHAIN_LINKS):
            p0, r0 = self._j0[i]
            pj = p + (r @ p0)
            rj = r @ r0
            out[f"joint{i + 1}"] = (pj, rj)
            c, s = torch.cos(q[:, i]), torch.sin(q[:, i])
            z0, o1 = torch.zeros_like(c), torch.ones_like(c)
            rz = torch.stack([torch.stack([c, -s, z0], -1), torch.stack([s, c, z0], -1), torch.stack([z0, z0, o1], -1)], -2)
            rm = rj @ rz
            p1i, r1i = self._j1inv[i]
            p = pj + rm @ p1i
            r = rm @ r1i
            out[name] = (p, r)
        pf, rf = self._flange
        p_fl = p + r @ pf
        r_fl = r @ rf
        out["gripper_flange"] = (p_fl, r_fl)
        pg, rg = self._gbase
        p_gb, r_gb = p_fl + r_fl @ pg, r_fl @ rg
        out["gripper_base"] = (p_gb, r_gb)
        out["pinch"] = (p_gb + r_gb @ self.pinch_offset, r_gb)
        return out

    def fk(self, q: torch.Tensor, frame: str = "pinch") -> tuple[torch.Tensor, torch.Tensor]:
        """Pose ``(pos [N,3], quat_wxyz [N,4])`` of ``frame`` in base_link."""
        frame = {"flange": "gripper_flange"}.get(frame, frame)
        p, r = self.chain(q)[frame]
        return p, geo.mat_to_quat(r)

    def jacobian(self, q: torch.Tensor, frame: str = "pinch") -> torch.Tensor:
        """Geometric Jacobian ``[N, 6, 7]`` (linear rows first) of ``frame``'s origin, in base_link."""
        frame = {"flange": "gripper_flange"}.get(frame, frame)
        ch = self.chain(q)
        pt = ch[frame][0]
        cols = []
        for i in range(7):
            pj, rj = ch[f"joint{i + 1}"]
            z = rj[:, :, 2]
            cols.append(torch.cat([torch.cross(z, pt - pj, dim=-1), z], dim=-1))
        return torch.stack(cols, dim=-1)

    # ------------------------------------------------------------------ inverse kinematics
    def ik(self, q_seed: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor, frame: str = "pinch", iters: int = 30,
           damping: float = 0.02, rot_weight: float = 0.5, max_step: float = 0.25, q_ref: torch.Tensor | None = None,
           null_gain: float = 0.05, limit_margin: float = 0.02, tol_pos: float = 1e-4, tol_rot: float = 1e-3,
           free_axis: torch.Tensor | None = None, free_weight: float = 0.05):
        """Damped least squares IK toward a base_link pose; returns ``(q [N,7], pos_err [N], rot_err [N])``.

        ``q_ref`` (default: the seed) biases the redundant DOF through the Jacobian null space so solutions stay in
        the posture family of the seed. Joint limits are enforced with ``limit_margin`` (rad).
        ``free_axis`` ([N,3] or [3], base frame): rotation about this axis is only weakly constrained (weight
        ``free_weight``), e.g. the tilt about the jaw axis, which the arm needs near the edge of its workspace.
        ``rot_err`` then excludes the free component.
        """
        q = q_seed.to(self.device, self.dtype).clone()
        q_ref = q.clone() if q_ref is None else q_ref.to(self.device, self.dtype)
        lo, hi = self.limits[:, 0] + limit_margin, self.limits[:, 1] - limit_margin
        eye3 = torch.eye(3, dtype=self.dtype, device=self.device)
        eye6 = torch.eye(6, dtype=self.dtype, device=self.device)
        eye7 = torch.eye(7, dtype=self.dtype, device=self.device)
        pos = pos.to(self.device, self.dtype)
        quat = quat.to(self.device, self.dtype)
        # rotation-row weighting matrix [N, 3, 3]
        wr = rot_weight * eye3.expand(q.shape[0], 3, 3)
        if free_axis is not None:
            a = free_axis.to(self.device, self.dtype).expand(q.shape[0], 3)
            a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            wr = rot_weight * (eye3 - (1.0 - free_weight) * a[:, :, None] * a[:, None, :])
        frame = {"flange": "gripper_flange"}.get(frame, frame)
        r_t = geo.quat_to_mat(quat)
        for it in range(iters):
            ch = self.chain(q)  # one pass gives the pose and the Jacobian
            p, r_c = ch[frame]
            # orientation error 0.5 * sum_i c_i x t_i (= sin(angle) * axis; a descent direction up to 180 deg)
            er = (wr @ (0.5 * torch.cross(r_c, r_t, dim=1).sum(-1))[..., None]).squeeze(-1)
            e = torch.cat([pos - p, er], dim=-1)
            if it % 10 == 9:  # convergence check (a GPU sync) only now and then
                pe, re = e[:, :3].norm(dim=-1), e[:, 3:].norm(dim=-1) / rot_weight
                if bool(((pe < tol_pos) & (re < tol_rot)).all()):
                    break
            cols = []
            for j in range(7):
                pj, rj = ch[f"joint{j + 1}"]
                z = rj[:, :, 2]
                cols.append(torch.cat([torch.cross(z, p - pj, dim=-1), z], dim=-1))
            jac = torch.stack(cols, dim=-1)
            jac = torch.cat([jac[:, :3], wr @ jac[:, 3:]], dim=1)
            jt = jac.transpose(1, 2)
            jjt_inv = torch.linalg.inv(jac @ jt + (damping**2) * eye6)
            j_pinv = jt @ jjt_inv
            dq = (j_pinv @ e[..., None]).squeeze(-1)
            if null_gain > 0:
                # posture bias: fades out over the iterations (it selects the family, it must not bias the solution)
                # and skips joints near a limit, where clamping breaks the null-space projection
                free = ((q - lo) > 0.05) & ((hi - q) > 0.05)
                g = null_gain * (1.0 - it / max(1, iters - 1))
                dq = dq + ((eye7 - j_pinv @ jac) @ (g * (q_ref - q) * free)[..., None]).squeeze(-1)
            scale = (max_step / dq.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
            q = torch.max(torch.min(q + dq * scale, hi), lo)
        p, qc = self.fk(q, frame)
        rot_err = geo.quat_angle(qc, quat) if free_axis is None else \
            (wr @ geo.quat_error_rotvec(qc, quat)[..., None]).squeeze(-1).norm(dim=-1) / rot_weight
        return q, (pos - p).norm(dim=-1), rot_err

    # ------------------------------------------------------------------ statics
    def gravity_torque(self, q: torch.Tensor, payload_mass: torch.Tensor | float = 0.0,
                       payload_offset: torch.Tensor | None = None) -> torch.Tensor:
        """Generalized gravity torque ``[N, 7]`` (the torque gravity exerts on each joint) for an upright base.

        ``payload_mass`` (kg) is attached at the pinch point (+ ``payload_offset`` in the pinch frame).
        """
        ch = self.chain(q)
        n = q.shape[0]
        g = torch.tensor([0.0, 0.0, -GRAVITY], dtype=self.dtype, device=self.device)
        # world (base) COM positions and masses of every distal body
        bodies = []
        for name in _DISTAL_LINKS:
            m = self.masses[name]
            if m <= 1e-5:
                continue
            if name in ("gripper_link1", "gripper_link2"):
                pg, rg = ch["gripper_base"]
                c = pg + rg @ self._finger_mount
            else:
                pl, rl = ch[name]
                c = pl + rl @ self.coms[name]
            bodies.append((name, c, torch.full((n,), m, dtype=self.dtype, device=self.device)))
        if self.virtual_mass > 0:
            pg, rg = ch["gripper_base"]
            bodies.append(("gripper_link", pg + rg @ self.virtual_slider, torch.full((n,), self.virtual_mass, dtype=self.dtype, device=self.device)))
        pm = torch.as_tensor(payload_mass, dtype=self.dtype, device=self.device)
        if bool((pm > 0).any()):
            pf, rf = ch["pinch"]
            c = pf if payload_offset is None else pf + rf @ payload_offset.to(self.device, self.dtype)
            bodies.append(("payload", c, pm.expand(n)))
        tau = []
        for i in range(7):
            pj, rj = ch[f"joint{i + 1}"]
            z = rj[:, :, 2]
            t = torch.zeros(n, dtype=self.dtype, device=self.device)
            for name, c, m in bodies:
                if name in _CHAIN_LINKS and _CHAIN_LINKS.index(name) < i:
                    continue
                f = m[:, None] * g
                t = t + (torch.cross(c - pj, f, dim=-1) * z).sum(-1)
            tau.append(t)
        return torch.stack(tau, dim=-1)

    def sag_compensated_command(self, q_des: torch.Tensor, payload_mass: torch.Tensor | float = 0.0) -> torch.Tensor:
        """Joint command whose PD equilibrium under gravity is ``q_des`` (requires :meth:`set_arm`)."""
        assert self.kp is not None, "call set_arm() first"
        return q_des - self.gravity_torque(q_des, payload_mass) / self.kp

    def sag_equilibrium(self, q_cmd: torch.Tensor, payload_mass: torch.Tensor | float = 0.0, iters: int = 4) -> torch.Tensor:
        """Predicted steady-state joint angles for a command (fixed-point of ``q = q_cmd + tau_g(q)/kp``)."""
        q = q_cmd.clone()
        for _ in range(iters):
            q = q_cmd + self.gravity_torque(q, payload_mass) / self.kp
        return q


class ArmBase:
    """Fixed transform between an arm's base_link and the station frame."""

    def __init__(self, arm: str, model: dict | None = None, device="cpu", dtype=torch.float32):
        model = model or load_model()
        b = model["base_world"][arm]
        self.arm = arm
        self.pos = torch.tensor(b["pos"], dtype=dtype, device=device)
        self.quat = torch.tensor(b["quat_wxyz"], dtype=dtype, device=device)

    def to_world(self, p: torch.Tensor, q: torch.Tensor):
        return geo.frame_compose(self.pos.expand_as(p), self.quat.expand(*q.shape[:-1], 4), p, q)

    def to_base(self, p: torch.Tensor, q: torch.Tensor):
        return geo.frame_relative(self.pos.expand_as(p), self.quat.expand(*q.shape[:-1], 4), p, q)

    def vec_to_world(self, v: torch.Tensor) -> torch.Tensor:
        return geo.quat_apply(self.quat.expand(*v.shape[:-1], 4), v)

    def vec_to_base(self, v: torch.Tensor) -> torch.Tensor:
        return geo.quat_apply(geo.quat_conj(self.quat).expand(*v.shape[:-1], 4), v)
