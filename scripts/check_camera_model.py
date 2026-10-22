"""Check the evaluator camera model (tower_rl.perception.cameras, source="evaluator") on dataset JPEGs: project the
recorded block poses (and FK of the recorded joints for the wrist cameras) and measure how well the projected visible
box edges sit on image edges. The best sub-pixel shift of the projection should be ~(0, 0) on every camera.

Usage: python scripts/check_camera_model.py --episode data/subset/<variant>/<stem>.npz --frames_dir data/frames_x \
           --frames 0 900 --out results/perception/calib/model_check_x
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.data.episode import Episode  # noqa: E402
from tower_rl.data.layouts import Layout  # noqa: E402
from tower_rl.perception.boxes import corners_world, visual_half_extents  # noqa: E402
from tower_rl.perception.calib import EDGES, edge_shift, visible_edge_points  # noqa: E402
from tower_rl.perception.cameras import CAMERA_IDS, CameraRig  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--frames", nargs="+", type=int, default=[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ep = Episode.load(args.episode, Path(args.episode).with_name(Path(args.episode).stem + "_attrs.json"))
    lay = Layout.load(ep.variant)
    half = visual_half_extents(np.array([lay.dims.get(b, (0.15, 0.05, 0.03)) for b in ep.block_ids]))
    rig = CameraRig("evaluator")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for f in args.frames:
        bp = ep.arrays["sim__block_poses"][f]  # [B, 7] x y z qx qy qz qw
        pos, quat = bp[:, :3], np.concatenate([bp[:, 6:7], bp[:, 3:6]], 1)
        qpos = ep.arrays["observations__qpos"][f]
        cams = rig.cameras(np.stack([qpos[0:7], qpos[8:15]]))
        cw = corners_world(pos, quat, half)
        tiles = []
        for name in CAMERA_IDS:
            cam = cams[name]
            img = Image.open(Path(args.frames_dir) / f"{name}_{f:05d}.jpg").convert("RGB")
            rep = edge_shift(np.asarray(img), visible_edge_points(cam, cw, pos, quat, half))
            report[f"{name}@{f}"] = rep
            d = ImageDraw.Draw(img)
            for b in range(len(pos)):
                uv, z = cam.project(cw[b])
                if (z > 0).all():
                    for i, j in EDGES:
                        d.line([tuple(uv[i]), tuple(uv[j])], fill=(0, 255, 255), width=1)
            d.text((6, 6), f"{name} f{f} shift {rep['best_shift_px'][0]:+.2f},{rep['best_shift_px'][1]:+.2f}", fill=(255, 255, 0))
            tiles.append(img)
        grid = Image.new("RGB", (1280, 960))
        for i, t in enumerate(tiles):
            grid.paste(t, ((i % 2) * 640, (i // 2) * 480))
        grid.save(out / f"overlay_f{f:05d}.jpg", quality=92)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main())
