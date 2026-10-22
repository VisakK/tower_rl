"""Perception in the loop: the executor sees block poses estimated from the four evaluator cameras rendered in the
station scene instead of the simulator's (``tower_rl.eval.runner.run_batch(observe=CameraPerception(...))``).

The truth stays in ``State.privileged`` (true_block_pos / true_block_quat) for the evaluator, and every call's
estimation errors are kept for the report: per block, position (mm) and rotation (deg, over the box's symmetries)
errors, confidence, and whether the block is held (fingers closed within 3 cm of its centre). Isaac is reached only
through the ``world`` passed to each call.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from tower_rl import geometry as geo
from tower_rl.perception.cameras import CAMERA_IDS, deliver
from tower_rl.perception.data import keypoints, splat
from tower_rl.perception.estimator import ARMS, EstCfg, PoseEstimator, load_net
from tower_rl.perception.model import SIGMA

CAM_KEY = {"left_back_camera": "cam_low", "right_front_camera": "cam_high", "left_wrist_camera": "cam_left_wrist",
           "back_wrist_camera": "cam_back_wrist"}
SYM = [np.diag(d) for d in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]  # the box's proper symmetries


def rot_err_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Rotation errors ``[B]`` (deg) between ``Ra``, ``Rb [B, 3, 3]``, the smallest over the box's symmetries."""
    c = np.stack([(np.einsum("bij,bik,kj->b", Ra, Rb, S) - 1) / 2 for S in SYM])
    return np.degrees(np.arccos(np.clip(c.max(0), -1, 1)))


class CameraPerception:
    """``checkpoint`` "oracle": no rendering and no network; heatmaps splatted from the true visible corners and edges
    (boxes only: arms hide nothing) test the estimator and its integration with the executor on CPU."""

    def __init__(self, checkpoint: str, n_envs: int, device: str = "cuda", subframes: int = 4, cfg: EstCfg | None = None):
        self.truth = checkpoint == "truth"  # pass-through (states unchanged, logged as the others: a reference run)
        self.oracle = checkpoint == "oracle" or self.truth
        net = None if self.oracle else load_net(checkpoint, torch.device(device))
        self.est = [PoseEstimator(None, device, source="render", cfg=cfg, net=net) for _ in range(n_envs)]
        self.subframes = subframes
        self.log: list[dict] = []
        self.policy = None  # optional: the executor, to log the node (skill, target block) each call belongs to
        self.lights = None
        self.light_log = []
        self.video = None  # optional tower_rl.perception.video.EpisodeVideos: a frame per call (network mode)
        self.cases = []

    def reset(self, variant: str, seeds: list[int]):
        for est in self.est:
            est.reset(variant)
        self.seeds = seeds
        if self.video is not None:
            self.video.close()

    def set_domain(self, world, dom):
        """Apply the episodes' lighting (robustness case "lighting") to the rendered station lights."""
        self.cases = list(dom.case)
        if self.oracle:
            return
        if self.lights is None:
            import omni.usd

            from tower_rl.sim.lights import StationLights

            self.lights = StationLights(omni.usd.get_context().get_stage(), world.scene.env_origins.cpu().numpy())
        self.light_log = self.lights.apply(dom)

    def _node(self, e: int, est: PoseEstimator) -> dict:
        if self.policy is None or e >= len(self.policy.runs):
            return {}
        run = self.policy.runs[e]
        if run.i >= len(run.g["nodes"]):
            return {}
        nd = run.g["nodes"][run.i]
        return {"skill": nd.get("skill"), "node": nd.get("id"), "block": est.block_ids.index(nd["block"]) if nd.get("block") in est.block_ids else None}

    def _oracle_maps(self, est: PoseEstimator, s):
        rec = {"pos": np.asarray(s.block_pos).tolist(), "quat": np.asarray(s.block_quat).tolist(), "half": est.half.cpu().numpy().tolist(),
               "cams": {k: c.to_dict() for k, c in est.rig.cameras(s.q).items()}}
        cm, em = [], []
        for name in CAMERA_IDS:
            c, e = keypoints(rec, name, None)
            c, e = torch.as_tensor(c, dtype=torch.float32)[None], torch.as_tensor(e, dtype=torch.float32)[None]
            cm.append(splat(c, torch.tensor([c.shape[1]]), 480, 640, SIGMA["corner"], 4)[0])
            em.append(splat(e, torch.tensor([e.shape[1]]), 480, 640, SIGMA["edge"], 3)[0])
        return torch.stack(cm), torch.stack(em)

    def __call__(self, world, envs: list[int], states: list):
        if not self.oracle:
            for _ in range(self.subframes):
                world.sim.render()
            rgb = {}
            for name, key in CAM_KEY.items():
                cam = world.scene[key]
                cam.update(0.0, force_recompute=True)
                rgb[name] = cam.data.output["rgb"][..., :3].cpu().numpy()
            x = torch.stack([torch.as_tensor(np.ascontiguousarray(deliver(n, rgb[n][e]))) for e in envs for n in CAMERA_IDS])
            cm, em, mm, hm = self.est[0].heatmaps_batch(x)  # one forward pass for every requesting env's four images
        out = []
        for k, (e, s) in enumerate(zip(envs, states)):
            est = self.est[e]
            assert est.block_ids == list(s.block_ids), "estimator and scene block orders differ"
            sl = slice(4 * k, 4 * k + 4)
            if self.truth:
                pos, quat, conf = np.asarray(s.block_pos), np.asarray(s.block_quat), np.ones(len(s.block_ids))
                self.log.append({"seed": self.seeds[e], "t": round(float(s.t), 2), "err_mm": np.zeros(len(pos)), "err_deg": np.zeros(len(pos)),
                                 "conf": conf, "held": np.zeros(len(pos), dtype=bool), "fixed": est.fixed.cpu().numpy(), "pos": pos, "quat": quat,
                                 "true_pos": pos, "true_quat": quat, "width": np.asarray(s.width), "est_held": [None, None],
                                 "ratio": np.ones(len(pos)), "inliers": np.ones(len(pos)), **self._node(e, est)})
                out.append(s)
                continue
            maps = self._oracle_maps(est, s) if self.oracle else (cm[sl], em[sl], mm[sl], hm[sl])
            focus = None  # the block the executor's current node acts on
            if self.policy is not None and e < len(self.policy.runs):
                run = self.policy.runs[e]
                if run.i < len(run.g["nodes"]) and run.g["nodes"][run.i].get("block") in est.block_ids:
                    focus = [est.block_ids.index(run.g["nodes"][run.i]["block"])]
            pos, quat, conf = est(None, s.q, s.width, maps=maps, focus=focus)
            R = geo.quat_to_mat(torch.as_tensor(quat)).numpy()
            Rt = geo.quat_to_mat(torch.as_tensor(np.asarray(s.block_quat, dtype=np.float64))).numpy()
            pinch = [est._pinch(s.q, a)[0].cpu().numpy() for a in ARMS]
            held = np.zeros(len(pos), dtype=bool)
            for i in range(2):
                if s.width[i] < 0.055:
                    held |= np.linalg.norm(s.block_pos - pinch[i], axis=-1) < 0.03
            node = {}
            if self.policy is not None and e < len(self.policy.runs):
                run = self.policy.runs[e]
                if run.i < len(run.g["nodes"]):
                    nd = run.g["nodes"][run.i]
                    node = {"skill": nd.get("skill"), "block": est.block_ids.index(nd["block"]) if nd.get("block") in est.block_ids else None}
            self.log.append({"seed": self.seeds[e], "t": round(float(s.t), 2), "err_mm": 1000 * np.linalg.norm(pos - s.block_pos, axis=-1),
                             "err_deg": rot_err_deg(R, Rt), "conf": conf, "held": held, "fixed": est.fixed.cpu().numpy(), **node,
                             "pos": pos, "quat": quat, "true_pos": np.asarray(s.block_pos), "true_quat": np.asarray(s.block_quat),
                             "width": np.asarray(s.width), "est_held": [est.held[a] for a in ARMS],
                             "ratio": est.stats["ratio"].cpu().numpy(), "inliers": est.stats["inliers"].cpu().numpy()})
            if self.video is not None and not self.oracle:
                self._video_frame(e, est, s, pos, quat, conf, focus, x[sl].numpy(), (cm[sl], em[sl]))
            priv = dict(s.privileged)
            priv["true_block_pos"], priv["true_block_quat"] = s.block_pos, s.block_quat
            out.append(replace(s, block_pos=pos, block_quat=quat, block_conf=conf, privileged=priv))
        return out

    def _video_frame(self, e: int, est: PoseEstimator, s, pos, quat, conf, focus, images, maps):
        from tower_rl.perception.video import frame

        r = self.log[-1]
        b = focus[0] if focus else None
        node = self._node(e, est)
        run = self.policy.runs[e] if self.policy is not None and e < len(self.policy.runs) else None
        lines = [f"{est.variant}   {self.video.tier}   seed {self.seeds[e]}   case {self.cases[e] if e < len(self.cases) else '-'}"
                 f"      t {s.t:6.1f} s      estimator call {est.calls}"]
        line = f"node {node.get('node', 'done')} ({node.get('skill', '-')})"
        if b is not None:
            line += f"    target {est.block_ids[b]}: error {r['err_mm'][b]:.1f} mm / {r['err_deg'][b]:.1f} deg, confidence {conf[b]:.1f}"
        if run is not None and run.events:
            line += f"    last: {run.events[-1][1]} {run.events[-1][2]}"
        lines.append(line)
        sure = conf >= 0.5
        worst = int(np.argmax(np.where(sure, r["err_mm"], -1.0)))
        held = [est.held[a] for a in ARMS]
        lines.append(f"blocks: {int((conf >= 1).sum())} good, {int(((conf >= 0.5) & (conf < 1)).sum())} weak, "
                     f"{int(((conf > 0) & (conf < 0.5)).sum())} not seen, {int((conf <= 0).sum())} lost    worst confident "
                     f"error {r['err_mm'][worst]:.1f} mm ({est.block_ids[worst]})    held: "
                     + ", ".join(f"{a} {est.block_ids[h] if h is not None else '-'}" for a, h in zip(ARMS, held)))
        img = frame({n: images[i] for i, n in enumerate(CAMERA_IDS)}, est.rig.cameras(s.q), est.half.cpu().numpy(), pos, quat, conf,
                    est.fixed.cpu().numpy(), [h for h in held if h is not None], est.block_ids,
                    truth=(np.asarray(s.block_pos), np.asarray(s.block_quat)), focus=b, lines=lines,
                    maps=tuple(m.float().cpu().numpy() for m in maps) if self.video.maps else None, device=est.dev)
        self.video.add(e, f"{est.variant}_{self.video.tier}_seed{self.seeds[e]}", img)

    def save(self, path):
        """The per-call log (numpy arrays per call) as a pickle, for diagnosing failed episodes."""
        import pickle

        with open(path, "wb") as f:
            pickle.dump({"block_ids": self.est[0].block_ids, "log": self.log}, f)

    def summary(self) -> dict:
        if not self.log:
            return {}
        err = np.concatenate([r["err_mm"] for r in self.log])
        deg = np.concatenate([r["err_deg"] for r in self.log])
        conf = np.concatenate([r["conf"] for r in self.log])
        held = np.concatenate([r["held"] for r in self.log])
        fixed = np.concatenate([r["fixed"] for r in self.log])

        def st(m):
            if not m.any():
                return None
            e, d = err[m], deg[m]
            return {"n": int(m.sum()), "pos_mm_median": round(float(np.median(e)), 3), "pos_mm_p90": round(float(np.percentile(e, 90)), 3),
                    "pos_mm_p99": round(float(np.percentile(e, 99)), 3), "rot_deg_median": round(float(np.median(d)), 3),
                    "rot_deg_p90": round(float(np.percentile(d, 90)), 3), "frac_gt_5mm": round(float((e > 5).mean()), 4),
                    "frac_conf_lt_0.5": round(float((conf[m] < 0.5).mean()), 4)}

        ok = conf >= 0.5
        out = {"calls": len(self.log), "all": st(np.ones_like(ok)), "confident": st(ok), "foundations": st(ok & fixed),
               "movable_static": st(ok & ~fixed & ~held), "held": st(ok & held)}
        # the target block of each call's node, per skill (what the skill acts on)
        tgt = [(r.get("skill"), r["err_mm"][r["block"]], r["err_deg"][r["block"]]) for r in self.log if r.get("block") is not None]
        for sk in sorted({t[0] for t in tgt}):
            e = np.array([t[1] for t in tgt if t[0] == sk])
            d = np.array([t[2] for t in tgt if t[0] == sk])
            out[f"target@{sk}"] = {"n": int(len(e)), "pos_mm_median": round(float(np.median(e)), 3),
                                   "pos_mm_p90": round(float(np.percentile(e, 90)), 3), "pos_mm_max": round(float(e.max()), 2),
                                   "rot_deg_p90": round(float(np.percentile(d, 90)), 3)}
        return out
