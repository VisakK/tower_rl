"""Extract per-episode metadata + initial state of TOWER-SimData episodes via HTTP range requests (no images).

Usage: python extract_meta_remote.py <MANIFEST.jsonl> <out.jsonl> <task-prefix> [<task-prefix> ...]
"""
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
from huggingface_hub import HfFileSystem

REPO = "datasets/tower-benchmark/TOWER-SimData@1a59248e44866b1593e4ac8b851aa519c013220f"
manifest, out_path, prefixes = sys.argv[1], sys.argv[2], tuple(sys.argv[3:])
rows = [json.loads(line) for line in open(manifest)]
rows = [r for r in rows if r["task"].startswith(prefixes)]
try:  # resume: skip episodes already extracted successfully
    done = {json.loads(line)["path"] for line in open(out_path) if "error" not in json.loads(line)}
except FileNotFoundError:
    done = set()
rows = [r for r in rows if r["path"] not in done]


def attr(g, k):
    v = g.attrs.get(k)
    if v is None:
        return None
    v = v.decode() if isinstance(v, bytes) else v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v.tolist() if hasattr(v, "tolist") else v


def one(row):
    fs = HfFileSystem()
    for attempt in range(4):
        try:
            with fs.open(f"{REPO}/{row['path']}", "rb", block_size=2 * 1024 * 1024, cache_type="readahead") as fo, h5py.File(fo, "r") as f:
                sim = f["sim"]
                cfg = attr(sim, "config_json")
                prov = attr(sim, "provenance_json")
                rb = prov["calibration"]["runtime_readback"]
                gains = {a: {k: rb[a]["readback"][k][0] for k in ("kp", "kd", "armatures", "friction", "max_efforts", "max_velocities")} for a in rb}
                rec = {
                    "task": row["task"], "path": row["path"], "frames": row["frames"],
                    "instruction": attr(f, "instruction"), "task_name": attr(f, "task_name"),
                    "seed": attr(sim, "seed"), "block_ids": attr(sim, "block_ids"),
                    "block_fixed_on_table": attr(sim, "block_fixed_on_table"),
                    "initial_block_poses_attr": attr(sim, "initial_block_poses"),
                    "block_poses_frame0": f["sim/block_poses"][0].tolist(),
                    "block_poses_final": f["sim/block_poses"][-1].tolist(),
                    "qpos_frame0": f["observations/qpos"][0].tolist(),
                    "target_mapping": attr(sim, "target_mapping"),
                    "layout_metadata": attr(sim, "layout_metadata_json"),
                    "outcome": attr(sim, "outcome_json"),
                    "success_final": bool(f["sim/success"][-1]),
                    "stage": cfg.get("stage"), "physics_hz": prov.get("physics_hz"),
                    "cfg": {k: cfg.get(k) for k in ("task", "layers_above_base", "tower_jitter_m", "center_block_jitter_m", "motion_profile",
                                                     "gripper_finger_effort_limit_n", "placement_m", "placement_degrees", "max_seconds",
                                                     "collapse_degrees", "collapse_drop_m", "collapse_seconds", "reset_settle_seconds",
                                                     "settle_seconds", "final_settle_seconds")},
                    "gains_sha": hashlib.sha256(json.dumps(gains, sort_keys=True).encode()).hexdigest()[:12],
                    "gains": gains,
                    "command_delays": prov["calibration"].get("command_delays_seconds"),
                    "joint_gain_mode": prov["calibration"].get("joint_gain_mode"),
                    # Task 4/5 additions (v1.1.0): block sizes / masses (taller Task 5 side blocks), the finger effort the
                    # generator applied, and the task events without the per-row phase / open-loop progress records
                    "block_dimensions_m": attr(sim, "block_dimensions_m"),
                    "block_mass_kg": attr(sim, "block_mass_kg"),
                    "gripper_effort": prov["calibration"].get("generator_gripper_effort_application"),
                    "events": [e for e in (attr(sim, "events_json") or [])
                               if e.get("event") not in ("phase", "placement_ready_open_loop")],
                }
                return rec
        except Exception as e:  # transient HTTP errors
            err = e
            time.sleep(2 + 3 * attempt)
    return {"task": row["task"], "path": row["path"], "error": repr(err)}


t0 = time.time()
n_ok = 0
with open(out_path, "a") as out, ProcessPoolExecutor(max_workers=8) as ex:
    futs = [ex.submit(one, r) for r in rows]
    for i, fu in enumerate(as_completed(futs)):
        rec = fu.result()
        n_ok += "error" not in rec
        out.write(json.dumps(rec) + "\n")
        if i % 25 == 0:
            print(f"{i + 1}/{len(rows)} done, {time.time() - t0:.0f}s", flush=True)
print(f"finished {len(rows)} episodes ({n_ok} ok) in {time.time() - t0:.0f}s")
