"""Render pose-estimator training data: the teacher runs episodes in the station scene (CPU physics, as in rehearsal)
while the four evaluator cameras (RGB + depth, the generator's RTX settings) are captured at random rows. Labels: the
exact block poses, the camera models (tower_rl.perception.cameras, source "render": Isaac centres the principal
point), per-corner visibility and per-pixel block ids (analytic boxes where Isaac's depth agrees, i.e. not behind an
arm or a finger). Images are stored as delivered by the evaluator (static cameras turned by 180 deg).

The station light (one RectLight) is randomized per env and batch over the robustness case "lighting" (intensity
x0.25-2.0, 3000-8500 K, azimuth <= 60 deg) and beyond (tilt), with nominal light in ~40 % of the episodes.
One variant per process. ``--check`` writes geometry and lighting checks for the first capture and stops.

Output: <out>/<variant>/<tier>_s<seed>/frames.jsonl, img/<idx>_<camera>.jpg (q95), ids/<idx>_<camera>.png
(uint8 block index; 254 = a block surface hidden by an arm or finger; 255 = no block).
Example:
  python scripts/render_perception_data.py --headless --device cpu --variant task2-3_cross_tower_transfer_3 \
      --tiers D0 D1 --episodes 8 --num_envs 4 --bundle results/bundles/teacher_t45_final --out data/perception/r1
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--variant", required=True)
parser.add_argument("--tiers", nargs="+", default=["D0", "D1"])
parser.add_argument("--episodes", type=int, default=8, help="episodes per tier")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=20000)
parser.add_argument("--bundle", default="results/bundles/teacher_t45_final")
parser.add_argument("--capture_p", type=float, default=0.15, help="probability of a capture after each 0.1 s row")
parser.add_argument("--subframes", type=int, default=4, help="RTX renders per capture (the generator's render_subframes)")
parser.add_argument("--nominal_light_p", type=float, default=0.4)
parser.add_argument("--max_time_s", type=float, default=210.0)
parser.add_argument("--out", default="data/perception/r1")
parser.add_argument("--check", action="store_true", help="geometry / lighting checks on the first capture, then exit")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

torch.set_num_threads(1)  # the executor's tiny tensors (see rehearse.py)

from isaaclab.scene import InteractiveScene  # noqa: E402

from tower_rl.config import load_yaml  # noqa: E402
from tower_rl.domain import DomainSampler  # noqa: E402
from tower_rl.eval.runner import run_batch  # noqa: E402
from tower_rl.graph.executor import Executor  # noqa: E402
from tower_rl.perception.boxes import corner_visibility, corners_world, raycast, visual_half_extents  # noqa: E402
from tower_rl.perception.calib import edge_shift, visible_edge_points  # noqa: E402
from tower_rl.perception.cameras import CAMERA_IDS, CameraRig, camera_from_isaac, deliver  # noqa: E402
from tower_rl.scene import EVAL_RENDER_CFG  # noqa: E402
from tower_rl.sim.assets import sim_cfg  # noqa: E402
from tower_rl.sim.station_scene import StationWorld, make_station_scene_cfg  # noqa: E402

CAM_KEY = {"left_back_camera": "cam_low", "right_front_camera": "cam_high", "left_wrist_camera": "cam_left_wrist",
           "back_wrist_camera": "cam_back_wrist"}
LIGHT = "/World/envs/env_{}/Station/Model/StationModel/StationLighting/TopWhitePlaneLight"
OCCLUDED = 254  # ids value: a block surface hidden by something else (arm, finger, gripper)
TABLE_CENTRE = (0.30, -0.40, 0.0)  # between the two pads (station frame, m)
RAY_DEV = "cuda" if torch.cuda.is_available() else "cpu"

phys = load_yaml("physics")
cfg = sim_cfg(phys, device=args.device)
cfg.render = sim_utils.RenderCfg(**EVAL_RENDER_CFG)
sim = sim_utils.SimulationContext(cfg)
scene = InteractiveScene(make_station_scene_cfg(args.variant, args.num_envs, cameras=True,
                                                cam_data_types=("rgb", "distance_to_image_plane"), cam_update_pose=args.check))
sim.reset()
world = StationWorld(scene, sim, args.variant)
policy = Executor.from_bundle(args.bundle, args.variant, device=str(world.dev), rate_hz=10)
rig = CameraRig("render")
n = world.n
half = visual_half_extents(np.array([world.layout.dims.get(b, (0.15, 0.05, 0.03)) for b in world.block_ids]))
stage = omni.usd.get_context().get_stage()
origins = world.scene.env_origins.cpu().numpy()


# ---------------------------------------------------------------------- lighting
class Lights:
    """Per-env RectLight randomization (intensity, colour temperature, rotation about the vertical through the table
    centre = azimuth, tilt about the light's own centre)."""

    def __init__(self):
        self.prims = [stage.GetPrimAtPath(LIGHT.format(e)) for e in range(n)]
        assert all(p.IsValid() for p in self.prims), "station light prim not found"
        self.base = [(UsdGeom.Xformable(p).GetLocalTransformation(), p.GetAttribute("inputs:intensity").Get(),
                      p.GetAttribute("inputs:colorTemperature").Get()) for p in self.prims]

    def set(self, e: int, scale=1.0, kelvin=None, azimuth_deg=0.0, tilt_deg=0.0, tilt_dir_deg=0.0):
        p = self.prims[e]
        local0, inten0, kelvin0 = self.base[e]
        xf = UsdGeom.Xformable(p)
        parent = xf.ComputeParentToWorldTransform(0)
        world0 = local0 * parent
        c_light = world0.ExtractTranslation()
        c_table = Gf.Vec3d(*(np.asarray(TABLE_CENTRE) + origins[e]).tolist())
        tdir = math.radians(tilt_dir_deg)
        tilt = (Gf.Matrix4d().SetTranslate(-c_light) * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(math.cos(tdir), math.sin(tdir), 0), tilt_deg))
                * Gf.Matrix4d().SetTranslate(c_light))
        az = (Gf.Matrix4d().SetTranslate(-c_table) * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), azimuth_deg))
              * Gf.Matrix4d().SetTranslate(c_table))
        xf.MakeMatrixXform().Set(world0 * tilt * az * parent.GetInverse())
        p.GetAttribute("inputs:intensity").Set(float(inten0 * scale))
        p.GetAttribute("inputs:colorTemperature").Set(float(kelvin if kelvin is not None else kelvin0))
        return {"scale": round(scale, 3), "kelvin": round(float(kelvin if kelvin is not None else kelvin0)), "azimuth_deg": round(azimuth_deg, 1),
                "tilt_deg": round(tilt_deg, 1), "tilt_dir_deg": round(tilt_dir_deg, 1)}

    def randomize(self, rng: np.random.Generator) -> list[dict]:
        out = []
        for e in range(n):
            if rng.random() < args.nominal_light_p:
                out.append(self.set(e))
                continue
            out.append(self.set(e, scale=float(np.exp(rng.uniform(math.log(0.25), math.log(2.0)))),
                                kelvin=float(rng.uniform(3000, 8500)), azimuth_deg=float(rng.uniform(-60, 60)),
                                tilt_deg=float(rng.uniform(0, 30)) if rng.random() < 0.5 else 0.0,
                                tilt_dir_deg=float(rng.uniform(0, 360))))
        return out


lights = Lights()


# ---------------------------------------------------------------------- capture
def render_all():
    for _ in range(args.subframes):
        sim.render()
    out = {}
    for name, key in CAM_KEY.items():
        cam = scene[key]
        cam.update(0.0, force_recompute=True)
        # copies: on the CPU device these arrays share memory with Isaac's buffers, which the next render overwrites
        out[name] = (cam.data.output["rgb"][..., :3].cpu().numpy().copy(),
                     cam.data.output["distance_to_image_plane"][..., 0].cpu().numpy().copy())
    return out


def label_env(e: int, frames: dict, pos: np.ndarray, quat: np.ndarray, q: np.ndarray):
    """Delivered images, per-pixel block ids and per-corner visibility of env ``e``."""
    cams = rig.cameras(q)
    cw = corners_world(pos, quat, half)
    imgs, ids, vis, npix = {}, {}, {}, {}
    for name in CAMERA_IDS:
        rgb, dep = frames[name]
        img, d = deliver(name, rgb[e]), deliver(name, dep[e])
        da, ia = raycast(cams[name], pos, quat, half, device=RAY_DEV)
        da, ia = da.cpu().numpy(), ia.cpu().numpy()
        idmap = np.where((ia >= 0) & (np.abs(d - da) < 0.003), ia, 255).astype(np.uint8)
        idmap[(ia >= 0) & (d < da - 0.003)] = OCCLUDED  # an arm / finger in front of a block surface
        _, _, v = corner_visibility(cams[name], cw, d)
        imgs[name], ids[name] = img, idmap
        vis[name] = [int(sum(int(x) << k for k, x in enumerate(row))) for row in v]  # 8 bits per block
        npix[name] = np.bincount(idmap[idmap < OCCLUDED].ravel(), minlength=len(pos)).tolist()
    return cams, imgs, ids, vis, npix


def check(frames, pos, quat, q):
    """Rendered vs modelled camera poses, edge alignment of the render model, light response."""
    rep = {}
    cams = rig.cameras(q[0])
    cw = corners_world(pos[0], quat[0], half)
    for name in CAMERA_IDS:
        c_isaac = camera_from_isaac(name, scene[CAM_KEY[name]].data.pos_w[0].cpu().numpy(),
                                    scene[CAM_KEY[name]].data.quat_w_ros[0].cpu().numpy(), origins[0])
        c = cams[name]
        dr = np.degrees(np.arccos(np.clip((np.trace(c.R_wc.T @ c_isaac.R_wc) - 1) / 2, -1, 1)))
        img = deliver(name, frames[name][0][0])
        rep[name] = {"pose_diff_mm": round(1000 * float(np.linalg.norm(c.t_wc - c_isaac.t_wc)), 3), "rot_diff_deg": round(float(dr), 4),
                     "edges": edge_shift(img, visible_edge_points(c, cw, pos[0], quat[0], half))}
        Image.fromarray(img).save(out_root / f"check_{name}.jpg", quality=92)
    import carb

    st = carb.settings.get_settings()
    rep["rtx_settings"] = {k: st.get(k) for k in ("/rtx/post/histogram/enabled", "/rtx/post/tonemap/op", "/rtx/post/tonemap/filmIso",
                                                  "/rtx/post/tonemap/cameraShutter", "/rtx/post/tonemap/fNumber",
                                                  "/rtx/sceneDb/ambientLightIntensity", "/rtx/rendermode", "/rtx/post/aa/op")}
    means = {}
    for scale in (0.0, 0.25, 1.0, 2.0, 8.0, 1.0):
        lights.set(0, scale=scale)
        for rep_i in range(2):  # a second capture shows whether one capture (4 subframes) has converged
            fr = render_all()
            means[f"x{scale}_{rep_i}"] = [round(float(x), 1) for x in fr["right_front_camera"][0][0].reshape(-1, 3).mean(0)]
    for kelvin in (3000, 8500):
        lights.set(0, kelvin=kelvin)
        fr = render_all()
        means[f"{kelvin}K"] = [round(float(x), 1) for x in fr["right_front_camera"][0][0].reshape(-1, 3).mean(0)]
    lights.set(0, azimuth_deg=60, tilt_deg=30, tilt_dir_deg=0)
    fr = render_all()
    means["az60_tilt30"] = [round(float(x), 1) for x in fr["right_front_camera"][0][0].reshape(-1, 3).mean(0)]
    Image.fromarray(deliver("right_front_camera", fr["right_front_camera"][0][0])).save(out_root / "check_light_az60_tilt30.jpg", quality=92)
    rep["light_mean_rgb_right_front"] = means
    return rep


out_root = Path(args.out) / args.variant
out_root.mkdir(parents=True, exist_ok=True)
pool = ThreadPoolExecutor(max_workers=6)


def save_jpg(img, path):
    Image.fromarray(img).save(path, quality=95)


def save_png(a, path):
    Image.fromarray(a).save(path, optimize=False, compress_level=3)


summary = {}
for ti, tier in enumerate(args.tiers):
    sampler = DomainSampler(tier, device=world.dev, max_blocks=max(6, world.spec.num_movable))
    shard = out_root / f"{tier}_s{args.seed}"
    (shard / "img").mkdir(parents=True, exist_ok=True)
    (shard / "ids").mkdir(parents=True, exist_ok=True)
    fout = open(shard / "frames.jsonl", "a")
    idx = sum(1 for _ in open(shard / "frames.jsonl"))
    nb = (args.episodes + n - 1) // n
    t0 = time.time()
    for b in range(nb):
        bseed = args.seed + 1000 * ti + 100 * b
        rng = np.random.default_rng(bseed)
        light = lights.randomize(rng)
        recs, last = [], [None] * n
        apply_row0 = world.apply_row

        def apply_row_captured(q16, on_tick=None, ticks=None):
            global idx
            apply_row0(q16, on_tick=on_tick, ticks=ticks)
            if rng.random() >= args.capture_p and not args.check:
                return
            frames = render_all()
            pos, quat = (x.cpu().numpy().astype(np.float64) for x in world.block_pose())
            qj, w = (x.cpu().numpy().astype(np.float64) for x in world.joints())
            if args.check:
                rep = check(frames, pos, quat, qj)
                (out_root / "check.json").write_text(json.dumps(rep, indent=1))
                print("CHECK_JSON " + json.dumps(rep), flush=True)
                os._exit(0)
            for e in range(n):
                key = np.concatenate([qj[e].ravel(), pos[e].ravel()])
                if last[e] is not None and np.abs(key - last[e]).max() < 1e-5:
                    continue  # finished episode (env idle): no new information
                last[e] = key
                cams, imgs, ids, vis, npix = label_env(e, frames, pos[e], quat[e], qj[e])
                for name in CAMERA_IDS:
                    pool.submit(save_jpg, imgs[name], shard / "img" / f"{idx:06d}_{name}.jpg")
                    pool.submit(save_png, ids[name], shard / "ids" / f"{idx:06d}_{name}.png")
                recs.append({"idx": idx, "env": e, "batch_seed": bseed, "t": round(world.time, 2), "step": world.step_count,
                             "light": light[e], "q": qj[e].round(6).tolist(), "width": w[e].round(6).tolist(),
                             "pos": pos[e].round(6).tolist(), "quat": quat[e].round(6).tolist(),
                             "cams": {k: c.to_dict() for k, c in cams.items()}, "vis": vis, "npix": npix})
                idx += 1

        world.apply_row = apply_row_captured
        outs = run_batch(world, policy, sampler, seed=bseed, max_time_s=args.max_time_s, rate_hz=10)
        world.apply_row = apply_row0
        for r in recs:
            o = outs[r["env"]]
            r.update({"variant": args.variant, "tier": tier, "case": o.extra["case"], "episode_success": bool(o.task_success),
                      "block_ids": world.block_ids, "half": half.round(6).tolist()})
            fout.write(json.dumps(r) + "\n")
        fout.flush()
        print(f"[render] {args.variant} {tier} batch {b + 1}/{nb}: {len(recs)} frames, TSR {np.mean([o.task_success for o in outs]):.2f}, "
              f"{time.time() - t0:.0f} s", flush=True)
    fout.close()
    summary[tier] = idx
pool.shutdown(wait=True)
print("REPORT_JSON " + json.dumps({"variant": args.variant, "frames": summary}), flush=True)
os._exit(0)
