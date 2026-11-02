"""The learned reacher outside Isaac (Reach-PPO-plan.MD P4): executor nodes and the policy server.

``Reacher`` rebuilds ``reach_env.ReachEnv``'s actor observation from measured joint angles and its own command history,
runs the frozen actor (``tower_rl.skills.export``: TorchScript with the normaliser), integrates the joint-speed action
into ``q_des`` (kept ``action.limit_margin`` inside the limits) and returns the gravity-sag-compensated command, as the
env does each row. Goals are pinch poses in the station frame; ``scripts/reach_task_check.py --parity`` checks the
observation against the env's row by row. No Isaac imports.

Chunks (the env's ``chunk``, Reacher-V2.MD §7.1): ``step(None, ...)`` is a row without a measurement, on the predicted
joints q_des - (q_des - q at the last measured row); chunk-trained models also see the rows since that measurement.
"""
from __future__ import annotations

import torch

from tower_rl import geometry as geo
from tower_rl.kinematics import ArmBase, ArmKinematics

ROW_DT = 0.1


def rot6d_mat(m: torch.Tensor) -> torch.Tensor:
    return torch.cat([m[..., :, 0], m[..., :, 1]], dim=-1)


class Reacher:
    """One arm, ``n`` envs. ``skill``: the run's skill config (``config.json`` ``skill``, also in the export's JSON)."""

    def __init__(self, policy, arm: str, skill: dict, n: int = 1, device: str = "cpu"):
        self.policy, self.arm, self.n, self.dev = policy, arm, n, torch.device(device)
        self.kin = ArmKinematics(device=device).set_arm(arm)
        self.base = ArmBase(arm, device=device)
        act, obs, sp = skill["action"], skill["obs"], skill.get("speed", {})
        lim = self.kin.limits
        self.lo, self.hi = lim[:, 0] + act["limit_margin"], lim[:, 1] - act["limit_margin"]
        self.q_mid, self.q_half = (lim[:, 0] + lim[:, 1]) / 2, (lim[:, 1] - lim[:, 0]) / 2
        self.v_max, self.amap, self.lin_frac = float(act["v_max"]), act.get("map", "linear"), float(act.get("lin_frac", 0.0))
        self.kp_len, self.fine_m = float(obs["kp_len_m"]), float(obs.get("fine_m", 0.0))
        self.speed_obs, self.v_free = bool(sp.get("obs", False)), float(sp.get("free_m_s", 2.0))
        hb = skill.get("held_block", {})  # P2d models: + the measured finger opening and the finger regime
        self.held_obs = float(hb.get("frac", 0.0)) > 0 and bool(hb.get("obs", False))
        self.track_obs = bool(skill.get("track", {}).get("obs", False))  # P2e models: + the goal's velocity
        ck = skill.get("chunk", {})  # chunk-trained models: + rows since the last measured row / k_norm
        self.ck_obs, self.ck_norm = bool(ck.get("obs", False)), float(ck.get("k_norm", 16))
        z = lambda: torch.zeros(n, 7, device=self.dev)  # noqa: E731
        self.q_des, self.prev_q, self.actions = z(), z(), z()
        self.off, self.k = z(), 0  # q_des - q at the last measured row; rows since it

    def reset(self, q_meas: torch.Tensor):
        """Start from rest at the measured joints (the env's reset: q_des = q, no previous motion or action)."""
        self.q_des = self.clamp(q_meas.clone())
        self.prev_q = q_meas.clone()
        self.actions = torch.zeros_like(self.actions)
        self.off, self.k = torch.zeros_like(self.off), 0

    def clamp(self, q):
        return torch.max(torch.min(q, self.hi), self.lo)

    def keypoints(self, p, q):
        return p[..., None, :] + self.kp_len * geo.quat_to_mat(q).transpose(-1, -2)

    def predicted(self) -> torch.Tensor:
        """The joints a row without a measurement acts on: q_des less the offset measured at the last measured row."""
        return self.q_des - self.off

    def obs(self, q: torch.Tensor, goal_p: torch.Tensor, goal_q: torch.Tensor, v_lim=None, width=None,
            regime=0.0, goal_v=None, k: int = 0) -> torch.Tensor:
        """Actor observation; ``goal_p`` / ``goal_q`` in the arm's base frame; ``width``: the measured finger opening (m,
        default open), ``regime``: 1 with the Task 4/5 fingers (3 N limit); ``goal_v [n, 6]``: the goal's linear and
        angular velocity in the base frame (default still); ``k``: rows since the last measured row (chunk-trained
        models). Does not advance the history."""
        pb, qb = self.kin.fk(q, "pinch")
        dk = (self.keypoints(goal_p, goal_q) - self.keypoints(pb, qb)).reshape(-1, 9)
        qd = (q - self.prev_q) / ROW_DT
        parts = [dk, (q - self.q_mid) / self.q_half, qd, self.q_des - q, self.actions,
                 pb, rot6d_mat(geo.quat_to_mat(qb)), goal_p, rot6d_mat(geo.quat_to_mat(goal_q))]
        if self.fine_m > 0:
            parts.append(torch.tanh(dk / self.fine_m))
        if self.speed_obs:
            v = torch.full((self.n,), self.v_free, device=self.dev) if v_lim is None else torch.as_tensor(v_lim, device=self.dev)
            parts.append((v.expand(self.n) / self.v_free)[:, None])
        if self.held_obs:
            w = torch.as_tensor(0.1 if width is None else width, dtype=torch.float32, device=self.dev).expand(self.n)
            parts += [((w - 0.05) / 0.05)[:, None],
                      torch.as_tensor(regime, dtype=torch.float32, device=self.dev).expand(self.n)[:, None]]
        if self.track_obs:
            parts.append(torch.zeros(self.n, 6, device=self.dev) if goal_v is None else goal_v.to(self.dev).expand(self.n, 6))
        if self.ck_obs:
            parts.append(torch.as_tensor(k, dtype=torch.float32, device=self.dev).expand(self.n)[:, None] / self.ck_norm)
        return torch.nan_to_num(torch.cat(parts, dim=-1))

    def measure(self, q_meas: torch.Tensor | None) -> torch.Tensor:
        """The joints this row acts on: ``q_meas`` (a measured row: the offset is re-measured), or None inside a chunk
        (the prediction); advances the rows-since-measurement count."""
        if q_meas is None:
            self.k += 1
            return self.predicted()
        self.off, self.k = self.q_des - q_meas, 0
        return q_meas

    @torch.no_grad()
    def step(self, q_meas: torch.Tensor | None, goal_p_w: torch.Tensor, goal_q_w: torch.Tensor, payload_kg=0.0, v_lim=None,
             width=None, regime=0.0, goal_v_w=None):
        """One row: the command ``[n, 7]`` for measured joints ``q_meas`` (None: a row inside a chunk, on the prediction;
        ``width`` then the last measured one) and a station-frame goal pinch pose (moving at ``goal_v_w [n, 6]``, station
        frame, for P2e models)."""
        q = self.measure(q_meas)
        gp, gq = self.base.to_base(goal_p_w, goal_q_w)
        gv = None if goal_v_w is None else torch.cat([self.base.vec_to_base(goal_v_w[..., :3]),
                                                       self.base.vec_to_base(goal_v_w[..., 3:])], -1)
        a = self.policy(self.obs(q, gp, gq, v_lim, width, regime, gv, self.k)).clamp(-1.0, 1.0)
        return self.apply(q, a, payload_kg)

    def apply(self, q_meas: torch.Tensor, a: torch.Tensor, payload_kg=0.0) -> torch.Tensor:
        """Advance the history with action ``a`` (clamped) and return the sag-compensated command."""
        v = a * a.abs() if self.amap == "square" else a
        if self.amap == "blend":
            v = self.lin_frac * a + (1.0 - self.lin_frac) * a * a.abs()
        self.q_des = self.clamp(self.q_des + v * self.v_max * ROW_DT)
        self.prev_q = q_meas.clone()
        self.actions = a.clone()
        m = torch.as_tensor(payload_kg, dtype=self.q_des.dtype, device=self.dev).expand(self.n)
        lim = self.kin.limits
        return torch.max(torch.min(self.kin.sag_compensated_command(self.q_des, m), lim[:, 1]), lim[:, 0])
