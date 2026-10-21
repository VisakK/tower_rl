"""Read TOWER-SimData episodes over HTTP range requests: attributes and non-image datasets only.

HDF5 datasets are chunked, so ``h5py`` on an ``HfFileSystem`` file object fetches only the byte ranges it touches and
the JPEG streams (0.4-1.2 GB per episode) stay on the server; one episode's arrays take ~5 s. Use processes, not
threads, to parallelize (h5py serializes on a global lock).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# pinned dataset revision (same files as the docs-pinned 5ad6adc9)
REVISION = "1a59248e44866b1593e4ac8b851aa519c013220f"
REPO = f"datasets/tower-benchmark/TOWER-SimData@{REVISION}"
ARRAY_KEYS = [
    "observations/qpos", "leader/qpos", "observations/eef", "leader/eef", "observations/gripper_status",
    "sim/block_poses", "sim/block_velocities", "sim/block_released", "sim/active_block", "sim/task_phase",
    "sim/task_progress", "sim/success", "sim/terminal", "sim/collapse", "sim/time", "sim/action_valid",
    "sim/joint_efforts", "sim/joint_velocities", "sim/finger_pad_centers_m", "sim/event_count", "sim/recovery_attempt",
    "keyframes",
    # Task 4/5 (v1.1.0) simulator diagnostics: the gripper widths the generator's controller actually applied (while
    # gripping it holds ~43.8 mm, while leader/qpos says 0), and the two finger joint positions
    "sim/controller_qpos", "sim/gripper_joint_positions_m", "sim/gripper_requested_width_m",
]
ATTR_GROUPS = ["sim", "observations", "observations/images", "leader", "lifter"]


def _decode(v):
    if isinstance(v, bytes):
        return v.decode()
    return v.tolist() if hasattr(v, "tolist") else v


def fetch_arrays(rel_path: str, out_npz: str | Path, retries: int = 4) -> dict:
    """Fetch one episode's non-image arrays to ``out_npz`` (+ ``<stem>_attrs.json``); returns a small summary."""
    import h5py
    from huggingface_hub import HfFileSystem

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    err = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            fs = HfFileSystem()
            with fs.open(f"{REPO}/{rel_path}", "rb", block_size=2 * 1024 * 1024, cache_type="readahead") as fo, \
                    h5py.File(fo, "r") as f:
                arrays = {k.replace("/", "__"): f[k][()] for k in ARRAY_KEYS if k in f}
                image_keys = sorted(f["observations/images"].keys()) if "observations/images" in f else []
                attrs = {k: _decode(v) for k, v in f.attrs.items()}
                for grp in ATTR_GROUPS:
                    if grp in f:
                        attrs[f"__group__{grp}"] = {k: _decode(v) for k, v in f[grp].attrs.items()}
            np.savez_compressed(out_npz, **arrays)
            out_npz.with_name(f"{out_npz.stem}_attrs.json").write_text(json.dumps(attrs, indent=1, default=str))
            return {"path": rel_path, "out": str(out_npz), "frames": int(arrays["observations__qpos"].shape[0]),
                    "arrays": sorted(arrays), "images": image_keys, "seconds": round(time.time() - t0, 1)}
        except Exception as e:  # noqa: BLE001  (transient HTTP errors)
            err = e
            time.sleep(2 + 3 * attempt)
    return {"path": rel_path, "out": str(out_npz), "error": repr(err)}
