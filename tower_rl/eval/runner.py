"""Local evaluator loop (Step 3.4): in-process policies with the challenge's chunk semantics.

``policy.act(states, env_ids)`` returns one chunk ``[H, 16]`` (1 <= H <= 16, absolute joints + widths in m) per
requesting env; every row is held for 0.1 s (10 Hz = 3 x 30 Hz ticks). An env stops on task success, at the time
budget (210 s) or on a policy error (exception, wrong shape, non-finite values), which counts as a failed episode.
Operations are scored by :class:`tower_rl.eval.operations.OperationTracker` every 30 Hz tick.
Import after ``AppLauncher`` (uses the station scene).
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections import deque
from typing import Protocol

import numpy as np
import torch

from tower_rl import geometry as geo
from tower_rl.data.episode import Episode
from tower_rl.domain import DomainSampler
from tower_rl.eval.metrics import EpisodeOutcome
from tower_rl.eval.operations import OpCfg, OperationTracker, Push, PushTracker, Transfer
from tower_rl.eval.perception_noise import NoiseCfg, PerceptionNoise
from tower_rl.graph.interface import State
from tower_rl.sim.station_scene import StationWorld

HOLD_WIDTH = 0.055  # measured opening below this = fingers on a block (block 0.0497 m)


class Policy(Protocol):
    def reset(self, n_envs: int, variant: str, seeds: list[int]) -> None: ...
    def act(self, states: list[State], env_ids: list[int]) -> list[np.ndarray]: ...


def tracker_transfers(world: StationWorld, goals: dict[str, list] | None = None) -> list[Transfer]:
    """Planned transfers with goal z at the measured rest height of the goal layer (Task 5: the slot's goal height,
    its 32 mm side blocks put layer 2 off the standard pitch)."""
    out = []
    for k, bi in enumerate(world.spec.transfer_block.tolist()):
        bid = world.spec.movable_ids[bi]
        g = torch.as_tensor(goals[bid], dtype=torch.float32) if goals else world.spec.goal[k].cpu()
        if goals:  # dataset format x y z qx qy qz qw
            gp, gq = geo.pose7_to_pq(g)
        else:
            gp, gq = g[:3], g[3:7]
        layer = int(world.spec.transfer_dest_layer[k])
        z = float(world.rz[layer]) if world.spec.family == "transfer" else float(gp[2])
        pos = np.array([float(gp[0]), float(gp[1]), z])
        out.append(Transfer(world.block_ids.index(bid), world.spec.transfer_arm[k], pos, float(geo.yaw_from_quat(gq)), layer))
    return out


def make_trackers(world: StationWorld, n: int, goals: dict | None = None, op_cfg: OpCfg | None = None) -> list:
    """One operation tracker per env for the variant's family (transfer / insertion: G + P per transfer, Task 5 with
    its tower as structure; pushdown: G + P per push)."""
    lay = world.layout
    op_cfg = op_cfg or OpCfg()
    if lay.family != "transfer" and op_cfg.final_settle_s is None:
        op_cfg = dataclasses.replace(op_cfg, final_settle_s=lay.generator.get("final_settle_seconds") or 2.0)
    mov = list(world.mov_idx)
    if lay.family == "pushdown":
        pushes = [Push(world.block_ids.index(p.block), p.arm, np.asarray(p.direction, dtype=float)) for p in lay.pushes]
        center = np.asarray(lay.pads["source"]["center"][:2], dtype=float)
        return [PushTracker(world.variant, pushes, mov, center, op_cfg) for _ in range(n)]
    transfers = tracker_transfers(world, goals)
    structure = [world.block_ids.index(b) for b in lay.movable_ids if lay.role(b) in ("side", "roof")]
    return [OperationTracker(world.variant, transfers, mov, op_cfg, structure=structure) for _ in range(n)]


ARM_POINTS = ["joint3", "joint4", "joint5", "joint6", "joint7", "gripper_flange", "gripper_base", "pinch"]


def arm_distance(world: StationWorld, qj: torch.Tensor) -> torch.Tensor:
    """Smallest distance ``[N]`` between the two arms, from the elbow to the pinch (joint origins and gripper frames;
    a proxy for arm-arm contact, which the station scene does not report)."""
    pts = []
    for i, arm in enumerate(("left", "back")):
        ch = world.kin.chain(qj[:, i])
        pb = torch.stack([ch[k][0] for k in ARM_POINTS], 1)  # [N, P, 3] base frame
        pts.append(world.bases[arm].vec_to_world(pb) + world.bases[arm].pos)
    return torch.cdist(pts[0], pts[1]).flatten(1).min(-1).values


def _validate(chunk) -> np.ndarray:
    a = np.asarray(chunk, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 16 or not 1 <= a.shape[0] <= 16:
        raise ValueError(f"chunk shape {a.shape}, expected [H, 16] with 1 <= H <= 16")
    if not np.isfinite(a).all():
        raise ValueError("non-finite action")
    return a


def run_batch(world: StationWorld, policy: Policy, sampler: DomainSampler | None, seed: int, max_time_s: float = 210.0,
              op_cfg: OpCfg | None = None, episode: Episode | None = None, stage: int = 0, rate_hz: int = 10,
              noise: NoiseCfg | None = None, log=print, observe=None) -> list[EpisodeOutcome]:
    """One episode per env. ``episode`` replaces the domain sample by that recording's initial state and goals.
    ``rate_hz`` is the track: 10 (rows held 0.1 s) or 30 (rows held 1/30 s). ``observe(world, envs, states)``
    replaces the requesting envs' states (e.g. block poses estimated from the rendered cameras, tower_rl.perception);
    its ``reset(variant, seeds)`` runs with the policy's."""
    assert rate_hz in (10, 30)
    ticks_per_row = 30 // rate_hz
    n = world.n
    gen = torch.Generator(device=world.dev).manual_seed(seed)
    goals = None
    if episode is not None:
        world.reset_from_episode(episode)
        goals = {m["block_id"]: m["goal"] for m in episode.target_mapping if "goal" in m} or None  # Task 4: push targets
        cases = ["episode"] * n
    else:
        dom = sampler.sample(n, gen)
        world.reset(dom, stage=stage, gen=gen)
        cases = dom.case
    pos, quat = world.block_pose()
    trackers = make_trackers(world, n, goals, op_cfg)
    for e in range(n):
        trackers[e].reset(pos[e].cpu().numpy())
    # Task 4 / 5: a shadow tracker without our 10 mm "remaining tower in place" rule (the generator's records only show
    # its collapse rule: 30 deg tilt / 15 mm drop), reported as extra["task_success_collapse_only"]
    shadows = None
    if world.layout.family != "transfer":
        shadows = make_trackers(world, n, goals, dataclasses.replace(op_cfg or OpCfg(), structure_mm=math.inf))
        for e in range(n):
            shadows[e].reset(pos[e].cpu().numpy())
    shadow_done = [False] * n
    seeds = [seed * 1000 + e for e in range(n)]
    policy.reset(n, world.variant, seeds)
    if observe is not None:
        observe.reset(world.variant, seeds)
        if episode is None and hasattr(observe, "set_domain"):  # e.g. the case "lighting" on the rendered cameras
            observe.set_domain(world, dom)
    queues = [deque() for _ in range(n)]
    noisers = [PerceptionNoise(noise, len(world.block_ids), seeds[e] + 7) for e in range(n)] if noise else None
    act_wall = []
    done = [False] * n
    errors: list[str | None] = [None] * n
    t_end = [max_time_s] * n
    rows = world._cmd.clone()
    wall0 = time.time()
    arm_gap = torch.full((n,), float("inf"), device=world.dev)  # closest approach of the two arms (Step 8.2)

    def on_tick(t):
        p, q4 = world.block_pose()
        sp = world.block_speed()
        qj, w = world.joints()
        pinch = world.pinch_world(qj)
        arm_gap.copy_(torch.minimum(arm_gap, arm_distance(world, qj)))
        p, q4, sp, w, pinch = (x.cpu().numpy() for x in (p, q4, sp, w, pinch))
        for e in range(n):
            if shadows is not None and not done[e] and not shadow_done[e]:
                shadow_done[e] = shadows[e].update(t, p[e], q4[e], sp[e], pinch[e], w[e] < HOLD_WIDTH)
            if not done[e] and trackers[e].update(t, p[e], q4[e], sp[e], pinch[e], w[e] < HOLD_WIDTH):
                done[e] = True
                t_end[e] = t

    while world.time < max_time_s - 1e-6 and not all(done):
        need = [e for e in range(n) if not done[e] and not queues[e]]
        if need:
            states = [world.state(e) for e in need]
            if observe is not None:
                states = observe(world, need, states)
            if noisers:
                states = [noisers[e](st) for e, st in zip(need, states)]
            try:
                t_act = time.time()
                chunks = policy.act(states, need)
                act_wall.append((time.time() - t_act) / len(need))
                for e, c in zip(need, chunks):
                    queues[e].extend(_validate(c))
            except Exception as ex:  # noqa: BLE001  (a policy error ends those episodes)
                for e in need:
                    if not queues[e]:
                        errors[e] = repr(ex)
                        done[e] = True
                        t_end[e] = world.time
                log(f"[runner] policy error at t={world.time:.1f}s: {ex!r}")
        for e in range(n):
            if not done[e] and queues[e]:
                rows[e] = torch.as_tensor(queues[e].popleft(), device=world.dev)
        world.apply_row(rows, on_tick=on_tick, ticks=ticks_per_row)
    outs = []
    for e in range(n):
        o = trackers[e].outcome(t_end[e], errors[e])
        o.extra.update({"case": cases[e], "seed": seeds[e], "wall_s": round(time.time() - wall0, 1),
                        "min_arm_gap_m": round(float(arm_gap[e]), 3),
                        "act_ms_mean": round(1000 * float(np.mean(act_wall)), 2) if act_wall else None,
                        "act_ms_max": round(1000 * float(np.max(act_wall)), 2) if act_wall else None,
                        "act_calls": len(act_wall)})
        if hasattr(policy, "episode_log"):
            o.extra["policy_log"] = policy.episode_log(e)
        if shadows is not None:
            o.extra["task_success_collapse_only"] = bool(o.task_success or shadow_done[e])
        outs.append(o)
    return outs


# ---------------------------------------------------------------------- reference policies
class HoldStill:
    """Keeps the home command (scores 0 everywhere)."""

    def reset(self, n_envs, variant, seeds):
        pass

    def act(self, states, env_ids):
        return [np.concatenate([np.r_[s.q[0], 0.1], np.r_[s.q[1], 0.1]])[None].repeat(16, 0) for s in states]


class DemoReplay:
    """Replays a recording's 30 Hz ``leader/qpos``: every command on the 30 Hz track, every third on the 10 Hz track;
    16 rows per chunk."""

    def __init__(self, episode: Episode, rate_hz: int = 10):
        self.cmd = episode.qpos_applied[:: 30 // rate_hz].astype(np.float32)
        self.i = None

    def reset(self, n_envs, variant, seeds):
        self.i = [0] * n_envs

    def act(self, states, env_ids):
        out = []
        for e in env_ids:
            i = self.i[e]
            c = self.cmd[min(i, len(self.cmd) - 1): min(i + 16, len(self.cmd))]
            if len(c) == 0:
                c = self.cmd[-1:]
            self.i[e] = i + len(c)
            out.append(c)
        return out
