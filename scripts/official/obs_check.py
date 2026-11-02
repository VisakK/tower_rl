"""Per variant: the first saved official frame of each camera vs the public dataset's (local_probe_1): which of
same / rot180 / flip_ud / flip_lr correlates best, and the share of wood-coloured pixels (blocks) in each. Then, per
episode, whether its first frame shows blocks: edge density at the table centre of the back wrist camera, which
holds under the tinted lighting cases (wood share fails there: sepia light makes a blank table "wood", cool light
makes blocks pale). Official frames 2026-10-26: blank <= 0.0032, blocks >= 0.0105 (Tasks 1-2 / 1-3 / 2-x), but Task
3-2's flat block layer only 0.0072: check new layouts (Tasks 4/5) by eye.

  ~/Documents/moves/env_isaaclab/bin/python scripts/official/obs_check.py [since_unix_time]   # default: frames after the 2026-10-24 qualification check
"""
import glob, os, sys
import numpy as np
from PIL import Image
S = "/home/visakii/Documents/tower_rl/results/student/server/"
CAMS = ["left_back_camera", "right_front_camera", "back_wrist_camera", "left_wrist_camera"]

def gray(p):
    a = np.asarray(Image.open(p).convert("L").resize((160, 120)), dtype=np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)

def wood(p):
    h, s, v = [np.asarray(c, dtype=np.float32) for c in Image.open(p).convert("HSV").split()]
    # PIL hue 0-255: orange-yellow wood ~ 15-35 (21-50 deg); saturated and not dark
    return float(((h >= 12) & (h <= 38) & (s >= 70) & (v >= 90)).mean())

def structure(p):
    g = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
    h, w = g.shape
    c = g[int(0.05 * h):int(0.70 * h), int(0.25 * w):int(0.75 * w)]  # table centre, above the fingers
    gx, gy = np.abs(np.diff(c, axis=1)), np.abs(np.diff(c, axis=0))
    return float((gx[:-1] + gy[:, :-1]).mean() / (c.mean() + 1e-6))

BLOCKS_MIN = 0.006  # between blank (<= 0.0032) and blocks (>= 0.0105)

def first_frames(root, variant, since=None):
    dirs = [d for d in glob.glob(f"{S}{root}/episodes/{variant}/*_images") if os.path.exists(d + "/00001_left_back_camera.jpg")]
    if since:
        dirs = [d for d in dirs if os.path.getmtime(d + "/00001_left_back_camera.jpg") >= since]
    return sorted(dirs, key=lambda d: os.path.getmtime(d + "/00001_left_back_camera.jpg"))

since = float(sys.argv[1]) if len(sys.argv) > 1 else 1792900000  # 2026-10-26 (skips the probe / check frames)
for v in sorted(os.listdir(S + "official_1/episodes")):
    off = first_frames("official_1", v, since)
    ref = sorted(first_frames("local_probe_1", v), key=lambda d: len(os.listdir(d)))[-1:]  # the replayed episode (not the zero-image check)
    if not off:
        continue
    print(f"{v}: {len(off)} official episode(s) with frames")
    for c in CAMS:
        o = off[-1] + "/00001_" + c + ".jpg"
        line = f"  {c:20s} official wood {wood(o):.4f}"
        if ref:
            r = ref[0] + "/00001_" + c + ".jpg"
            A, B = gray(o), gray(r)
            sc = {k: float((A * x).mean()) for k, x in {"same": B, "rot180": B[::-1, ::-1], "flip_ud": B[::-1], "flip_lr": B[:, ::-1]}.items()}
            best = max(sc, key=sc.get)
            line += f" | dataset wood {wood(r):.4f} | best {best} {sc[best]:+.2f} (" + " ".join(f"{k} {x:+.2f}" for k, x in sc.items()) + ")"
        print(line)

print("\nper episode (first frame), oldest first: back-wrist edge density -> blocks?")
tally = {}
for v in sorted(os.listdir(S + "official_1/episodes")):
    for d in first_frames("official_1", v, since):
        x = structure(d + "/00001_back_wrist_camera.jpg")
        ok = x >= BLOCKS_MIN
        tally.setdefault(v, [0, 0])[0 if ok else 1] += 1
        print(f"  {v[:7]} {os.path.basename(d)[:8]} {x:.4f} {'blocks' if ok else 'NO BLOCKS'}")
for v, (a, b) in tally.items():
    print(f"{v}: {a} with blocks, {b} without")
