"""Occlusion-aware labels for organizer frames (shards of scripts/fetch_dataset_frames.py), so their JPEGs (the
evaluator's renderer) can train the keypoint network: every recorded state (joints, gripper widths, block poses) is
written into the station scene and its depth rendered (no physics stepping); tower_rl.perception.labels turns the depth
into per-corner visibility and the block-id map in the organizers' image coordinates. Adds ``vis`` / ``npix`` to each
frames.jsonl record and writes ids/<idx>_<camera>.png. GPU device (kinematic pose writes do not stick on CPU).

Usage: python scripts/label_dataset_frames.py --headless --variant task2-3_cross_tower_transfer_3 --root data/perception/dataset_train
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--variant", required=True)
parser.add_argument("--root", default="data/perception/dataset_train")
parser.add_argument("--subframes", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from isaaclab.scene import InteractiveScene  # noqa: E402

from tower_rl.config import load_yaml  # noqa: E402
from tower_rl.perception.boxes import raycast  # noqa: E402
from tower_rl.perception.cameras import CAMERA_IDS, Camera, CameraRig, deliver  # noqa: E402
from tower_rl.perception.labels import view_labels  # noqa: E402
from tower_rl.sim.assets import q_full, sim_cfg  # noqa: E402
from tower_rl.sim.station_scene import make_station_scene_cfg  # noqa: E402

CAM_KEY = {"left_back_camera": "cam_low", "right_front_camera": "cam_high", "left_wrist_camera": "cam_left_wrist",
           "back_wrist_camera": "cam_back_wrist"}

phys = load_yaml("physics")
sim = sim_utils.SimulationContext(sim_cfg(phys, device=args.device))
scene = InteractiveScene(make_station_scene_cfg(args.variant, 1, cameras=True, cam_data_types=("distance_to_image_plane",)))
sim.reset()
rig = CameraRig("render")
blocks = scene["blocks"]
scene_ids = list(blocks.object_names)
origin = scene.env_origins[0]
dev = scene.device


def label_frame(shard: Path, rec: dict):
    """Write the recorded state, render the depth and add the frame's labels (``rec`` updated in place)."""
    pos, quat, half = np.asarray(rec["pos"]), np.asarray(rec["quat"]), np.asarray(rec["half"])
    q = np.asarray(rec["q"])
    for i, arm in enumerate(("left", "back")):
        qa = torch.tensor(np.r_[q[i], rec["width"][i]], dtype=torch.float32, device=dev)
        qf = q_full(qa[:7], qa[7:8])[None]
        scene[arm].write_joint_state_to_sim(qf, torch.zeros_like(qf))
        scene[arm].set_joint_position_target(qf)
    order = [rec["block_ids"].index(b) for b in scene_ids]
    p7 = torch.tensor(np.c_[pos[order], quat[order]], dtype=torch.float32, device=dev)[None]
    p7[..., :3] += origin
    blocks.write_object_link_pose_to_sim(p7)
    scene.write_data_to_sim()
    sim.forward()
    for _ in range(args.subframes):
        sim.render()
    cams_r = rig.cameras(q)
    vis, npix = {}, {}
    n_box = n_behind = 0
    for name in CAMERA_IDS:
        cam = scene[CAM_KEY[name]]
        cam.update(0.0, force_recompute=True)
        depth = deliver(name, cam.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy().copy())
        ids, vis[name], npix[name] = view_labels(cams_r[name], depth, pos, quat, half, cam_lab=Camera.from_dict(rec["cams"][name]),
                                                 device=DEV)
        # the render must show the blocks where the recorded poses put them: depth beyond a box surface means the
        # written state did not reach the renderer (bevelled block edges alone give a few percent)
        da, ia = (x.cpu().numpy() for x in raycast(cams_r[name], pos, quat, half, device=DEV))
        with np.errstate(invalid="ignore"):
            n_box += int((ia >= 0).sum())
            n_behind += int(((ia >= 0) & (depth > da + 0.003)).sum())
        Image.fromarray(ids).save(shard / "ids" / f"{rec['idx']:06d}_{name}.png", compress_level=3)
    rec["vis"], rec["npix"] = vis, npix
    return n_behind / max(1, n_box)


DEV = "cuda" if torch.cuda.is_available() else "cpu"
n_done, behind = 0, []
for fj in sorted((Path(args.root) / args.variant).glob("*/frames.jsonl")):
    shard = fj.parent
    (shard / "ids").mkdir(exist_ok=True)
    recs = [json.loads(line) for line in open(fj)]
    for rec in recs:
        try:
            behind.append(label_frame(shard, rec))
            n_done += 1
        except Exception as ex:  # noqa: BLE001  (an unlabelled frame trains with box-only visibility)
            print(f"[label] {shard.name} frame {rec['idx']}: {ex!r}", flush=True)
            continue
        if len(behind) == 3 and min(behind) > 0.2:
            print(f"[label] ABORT: {100 * min(behind):.0f}% of box pixels render behind the box: the written state does not "
                  "reach the renderer; no labels written", flush=True)
            os._exit(2)
    with open(fj, "w") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")
    print(f"[label] {args.variant} {shard.name}: {len(recs)} frames", flush=True)
print("REPORT_JSON " + json.dumps({"variant": args.variant, "frames": n_done, "behind_frac_mean": round(float(np.mean(behind)), 4) if behind else None,
                                   "behind_frac_max": round(float(np.max(behind)), 4) if behind else None}), flush=True)
os._exit(0)
