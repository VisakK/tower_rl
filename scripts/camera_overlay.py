"""Step 2.3 (c): overlay rendered evaluator cameras on the demo JPEGs of the same frame and measure the misalignment.

Renders come from ``scripts/validate_isaaclab_scene.py --mode render`` (station scene, recorded state of the frame);
the static cameras' renders are rotated 180 deg to the dataset's stored orientation (left_back_camera = station
low_camera, right_front_camera = high_camera). The pixel offset is the shift maximizing the correlation of the two
edge maps within +-25 px.

Usage: python -I scripts/camera_overlay.py --renders results/eval_2026-10-17/render_nodome --frames data/frames_t23_ep1 \
           --frame 0 --out results/calibration/camera_overlay_t23_f0.jpg
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CAMS = [("low", "left_back_camera", True), ("high", "right_front_camera", True),
        ("left_wrist", "left_wrist_camera", False), ("back_wrist", "back_wrist_camera", False)]


def edges(img: Image.Image) -> np.ndarray:
    g = np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    g -= g.mean()
    return g / (g.std() + 1e-6)


def best_shift(a: np.ndarray, b: np.ndarray, r: int = 25):
    """Shift (dx, dy) of ``b`` that maximizes its correlation with ``a``."""
    best, arg = -1e9, (0, 0)
    h, w = a.shape
    ca = a[r:h - r, r:w - r]
    for dy in range(-r, r + 1, 1):
        for dx in range(-r, r + 1, 1):
            cb = b[r + dy:h - r + dy, r + dx:w - r + dx]
            s = float((ca * cb).mean())
            if s > best:
                best, arg = s, (dx, dy)
    return arg, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tiles, report = [], {}
    for render_name, data_name, rotate in CAMS:
        r = Image.open(Path(args.renders) / f"{render_name}_{args.frame:05d}.png").convert("RGB")
        if rotate:
            r = r.rotate(180)
        d = Image.open(Path(args.frames) / f"{data_name}_{args.frame:05d}.jpg").convert("RGB").resize(r.size)
        (dx, dy), score = best_shift(edges(d), edges(r))
        report[data_name] = {"pixel_offset_dxdy": [dx, dy], "edge_corr": round(score, 3), "size": list(r.size)}
        blend = Image.blend(d, r, 0.5)
        draw = ImageDraw.Draw(blend)
        draw.text((8, 8), f"{data_name}: render/demo 50/50, best edge shift ({dx},{dy}) px", fill=(255, 255, 0))
        tiles.append((d, r, blend))
    w, h = tiles[0][0].size
    grid = Image.new("RGB", (3 * w, len(tiles) * h))
    for i, row in enumerate(tiles):
        for j, im in enumerate(row):
            grid.paste(im.resize((w, h)), (j * w, i * h))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out, quality=90)
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main())
