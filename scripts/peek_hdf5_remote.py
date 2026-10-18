"""Read attributes + small (non-image) datasets of one TOWER-SimData episode via HTTP range requests.

Usage: python -I peek_hdf5_remote.py <path in the dataset, e.g. data/task2-3_.../episode_....hdf5> <out.npz>
Writes <out.npz> (arrays, "/" replaced by "__") and <out>_attrs.json (root and group attributes).
"""
import json
import os
import sys
import time

import h5py
import numpy as np
from huggingface_hub import HfFileSystem

REPO = "datasets/tower-benchmark/TOWER-SimData@1a59248e44866b1593e4ac8b851aa519c013220f"
rel = sys.argv[1]
os.makedirs(os.path.dirname(os.path.abspath(sys.argv[2])), exist_ok=True)
fs = HfFileSystem()
t0 = time.time()
f_obj = fs.open(f"{REPO}/{rel}", "rb", block_size=2 * 1024 * 1024, cache_type="readahead")
with h5py.File(f_obj, "r") as f:
    print("== root attrs")
    for k, v in f.attrs.items():
        s = v.decode() if isinstance(v, bytes) else str(v)
        if len(s) > 400:
            print(f"  {k}: <{len(s)} chars> {s[:400]}...")
        else:
            print(f"  {k}: {s}")
    print("\n== tree")

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}: shape={obj.shape} dtype={obj.dtype} chunks={obj.chunks} compression={obj.compression}" + (f" attrs={dict(obj.attrs)}" if len(obj.attrs) else ""))
        else:
            if len(obj.attrs):
                print(f"  [{name}] attrs: " + ", ".join(f"{k}=" + (str(v)[:200]) for k, v in obj.attrs.items()))

    f.visititems(visit)
    print("t(meta)=%.1fs" % (time.time() - t0))
    out = {}
    for name in ["observations/qpos", "leader/qpos", "observations/eef", "leader/eef", "observations/gripper_status",
                 "sim/block_poses", "sim/active_block", "sim/task_phase", "sim/task_progress", "sim/success", "sim/terminal",
                 "sim/collapse", "sim/time", "sim/action_valid", "sim/block_released", "sim/joint_efforts", "sim/joint_velocities",
                 "sim/finger_pad_centers_m", "sim/controller_qpos", "sim/gripper_joint_positions_m", "sim/gripper_requested_width_m",
                 "lifter/action_mm", "observations/lifter/position_mm", "keyframes", "sim/event_count", "sim/recovery_attempt"]:
        if name in f:
            out[name] = f[name][()]
    print("t(data)=%.1fs" % (time.time() - t0))
    np.savez_compressed(sys.argv[2], **{k.replace("/", "__"): v for k, v in out.items()})
    # dump all root attrs fully as json for later inspection
    attrs = {k: (v.decode() if isinstance(v, bytes) else (v.tolist() if hasattr(v, "tolist") else v)) for k, v in f.attrs.items()}
    for grp in ["sim", "observations", "observations/images", "leader", "lifter"]:
        if grp in f:
            attrs[f"__group__{grp}"] = {k: (v.decode() if isinstance(v, bytes) else (v.tolist() if hasattr(v, "tolist") else v)) for k, v in f[grp].attrs.items()}
    with open(sys.argv[2].replace(".npz", "_attrs.json"), "w") as fp:
        json.dump(attrs, fp, indent=1, default=str)
print("saved")
