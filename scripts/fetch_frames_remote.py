"""Fetch individual JPEG frames of one TOWER-SimData episode via HTTP range requests.

Usage: python fetch_frames_remote.py <repo-relative hdf5 path> <out dir> <frame> [<frame> ...]
"""
import os
import sys

import h5py
from huggingface_hub import HfFileSystem

REPO = "datasets/tower-benchmark/TOWER-SimData@1a59248e44866b1593e4ac8b851aa519c013220f"
rel, out_dir, frames = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]]
os.makedirs(out_dir, exist_ok=True)
fs = HfFileSystem()
with fs.open(f"{REPO}/{rel}", "rb", block_size=1024 * 1024, cache_type="readahead") as fo, h5py.File(fo, "r") as f:
    for cam in f["observations/images"]:
        ds = f[f"observations/images/{cam}"]
        for t in frames:
            data = bytes(ds[t])
            with open(os.path.join(out_dir, f"{cam}_{t:05d}.jpg"), "wb") as fp:
                fp.write(data)
            print(cam, t, len(data), "bytes")
