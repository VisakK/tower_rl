"""Evaluator-domain validation frames for the pose estimator: JPEGs of TOWER-SimData episodes (the organizers'
renderer) fetched by HTTP range requests, labelled with the recorded block poses and joints (cameras: the evaluator
model of tower_rl.perception.cameras, checked on these images to ~0.25 px). The frames are evenly spaced over each
episode. Writes the render shards' layout without ids maps: <out>/<variant>/<stem>/frames.jsonl + img/<idx>_<camera>.jpg.

Usage: python scripts/fetch_dataset_frames.py --variant task2-3_cross_tower_transfer_3 --episodes 3 --frames 24 \
           --out data/perception/dataset_val
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_rl.data.episode import Episode  # noqa: E402
from tower_rl.data.layouts import Layout  # noqa: E402
from tower_rl.data.remote import REPO  # noqa: E402
from tower_rl.paths import DATA_DIR  # noqa: E402
from tower_rl.perception.boxes import visual_half_extents  # noqa: E402
from tower_rl.perception.cameras import CAMERA_IDS, CameraRig  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--skip", type=int, default=0, help="skip the first episodes (sorted names): 0-2 are validation")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--out", default="data/perception/dataset_val")
    args = ap.parse_args()
    import h5py
    from huggingface_hub import HfFileSystem

    rig = CameraRig("evaluator")
    lay = Layout.load(args.variant)
    npzs = sorted((DATA_DIR / "subset" / args.variant).glob("episode_*.npz"))[args.skip: args.skip + args.episodes]
    fs = HfFileSystem()
    for npz in npzs:
        ep = Episode.load(npz, npz.with_name(npz.stem + "_attrs.json"))
        shard = Path(args.out) / args.variant / npz.stem
        (shard / "img").mkdir(parents=True, exist_ok=True)
        half = visual_half_extents(np.array([lay.dims.get(b, (0.15, 0.05, 0.03)) for b in ep.block_ids]))
        T = ep.num_frames
        frames = np.linspace(30, T - 31, args.frames).round().astype(int)
        t0 = time.time()
        recs = []
        with fs.open(f"{REPO}/data/{args.variant}/{npz.stem}.hdf5", "rb", block_size=1024 * 1024, cache_type="readahead") as fo, \
                h5py.File(fo, "r") as f:
            for i, t in enumerate(frames):
                for name in CAMERA_IDS:
                    (shard / "img" / f"{i:06d}_{name}.jpg").write_bytes(bytes(f[f"observations/images/{name}"][t]))
                bp = ep.arrays["sim__block_poses"][t]
                qpos = ep.arrays["observations__qpos"][t]
                q = np.stack([qpos[0:7], qpos[8:15]])
                cams = rig.cameras(q)
                recs.append({"idx": i, "frame": int(t), "t": round(float(ep.arrays["sim__time"][t]), 3), "q": q.round(6).tolist(),
                             "width": [float(qpos[7]), float(qpos[15])], "pos": bp[:, :3].round(6).tolist(),
                             "quat": np.concatenate([bp[:, 6:7], bp[:, 3:6]], 1).round(6).tolist(),
                             "cams": {k: c.to_dict() for k, c in cams.items()}, "variant": args.variant, "tier": "dataset",
                             "case": "dataset", "block_ids": ep.block_ids, "half": half.round(6).tolist(), "source": "dataset"})
        with open(shard / "frames.jsonl", "w") as fout:
            for r in recs:
                fout.write(json.dumps(r) + "\n")
        print(f"[fetch] {args.variant} {npz.stem}: {len(recs)} frames in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
