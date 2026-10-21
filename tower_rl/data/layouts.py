"""Per-variant block layouts reconstructed from the dataset metadata (Step 1).

The SimData files record every block pose (``sim/block_poses``) and the transfer plan (``target_mapping``), so the
layout is read directly instead of being inferred from gripper events: ``data/meta_tasks123.jsonl``
(``scripts/extract_meta_remote.py``) holds frame-0 poses, goals and arms for all 331 Task1-3 episodes. Per variant the
nominal pose of each block is the mean over episodes; the spread is the generator's 3 mm tower jitter.

YAML layout (``configs/layouts/<variant>.yaml``), all poses ``[x, y, z, qw, qx, qy, qz]`` in the station frame (m):
  pads.{source,destination}: centre (mean of the three foundation blocks) and foundation yaw
  blocks: id -> {pose, fixed, side, layer, std_mm}
  transfers: ordered operations {block, arm, source_layer, dest_layer, goal, goal_in_pad}
  generator: the data generator's tolerances (placement 5 mm / 5 deg, collapse 30 deg / 15 mm)
Execution code must take goals from these files (and later from estimated poses), never from organizer descriptors.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from tower_rl.paths import DATA_DIR, REPO_ROOT

LAYOUT_DIR = Path(os.environ.get("TOWER_CONFIG_DIR", REPO_ROOT / "configs")) / "layouts"
METADATA_JSONL = DATA_DIR / "meta_tasks123.jsonl"
METADATA_T45_JSONL = DATA_DIR / "meta_tasks45.jsonl"
VARIANTS = ["task1-2_tower_transfer_2", "task1-3_tower_transfer_3", "task2-2_cross_tower_transfer_2",
            "task2-3_cross_tower_transfer_3", "task3-2_vertical_stacking_2", "task3-3_vertical_stacking_3"]
# dataset v1.1.0 (2026-10-13): Task 4 pushes every movable block off a full tower, Task 5 inserts blocks into the
# empty middle slots of a tower with taller side blocks and a roof (Task 5-3 on cuboid colliders)
VARIANTS_T4 = ["task4-2_full_tower_pushdown_2", "task4-3_full_tower_pushdown_3"]
VARIANTS_T5 = ["task5-2_tower_insertion_2", "task5-3_tower_insertion_3"]
ALL_VARIANTS = VARIANTS + VARIANTS_T4 + VARIANTS_T5
BLOCK_SIZE = (0.15, 0.05, 0.03)  # nominal Jenga block (2x real size); the collision hull is 0.15 x 0.0497 x 0.0283
FOUNDATION_CENTER_Z = 0.015
LAYER_PITCH = 0.03
ARM_NAMES = ("left", "back")  # target_mapping arm index


def layer_of(block_id: str) -> int:
    """``source_l2_a`` -> 2, ``source_l3_center`` -> 3, foundation blocks -> 0."""
    m = re.search(r"_l(\d+)_", block_id + "_")
    return int(m.group(1)) if m else 0


def side_of(block_id: str) -> str:
    return "destination" if block_id.startswith("destination") else "source"


def family_of(variant: str) -> str:
    """``transfer`` (Tasks 1-3), ``pushdown`` (Task 4) or ``insertion`` (Task 5)."""
    return "pushdown" if "pushdown" in variant else "insertion" if "insertion" in variant else "transfer"


def xyzw_to_wxyz(q) -> list[float]:
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def mean_quat_wxyz(qs: np.ndarray) -> np.ndarray:
    """Sign-aligned normalized mean of quaternions ``[N, 4]`` (wxyz); fine for small spreads."""
    ref = qs[0]
    aligned = np.where((qs @ ref)[:, None] < 0, -qs, qs)
    m = aligned.mean(0)
    m = m / np.linalg.norm(m)
    return m if m[0] >= 0 else -m


def yaw_of_wxyz(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load_metadata(path: str | Path = METADATA_JSONL) -> dict[str, list[dict]]:
    by = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if "error" not in r:
            by[r["task"]].append(r)
    return dict(by)


def _r(x, n=6):
    return [round(float(v), n) for v in x]


def build_layout(variant: str, rows: list[dict]) -> dict:
    ids = rows[0]["block_ids"]
    assert all(r["block_ids"] == ids for r in rows), f"{variant}: block ids differ between episodes"
    fixed = rows[0]["block_fixed_on_table"]
    p0 = np.array([r["block_poses_frame0"] for r in rows])  # [E, B, 7] xyzw
    pos = p0[..., :3]
    quat = p0[..., [6, 3, 4, 5]]
    blocks = {}
    for i, bid in enumerate(ids):
        mq = mean_quat_wxyz(quat[:, i])
        yaws = np.array([yaw_of_wxyz(q) for q in quat[:, i]])
        dy = (yaws - yaw_of_wxyz(mq) + math.pi / 2) % math.pi - math.pi / 2  # block yaw is pi-symmetric
        blocks[bid] = {
            "pose": _r(np.concatenate([pos[:, i].mean(0), mq])),
            "fixed": bool(fixed[i]),
            "side": side_of(bid),
            "layer": layer_of(bid),
            "std_mm": _r(pos[:, i].std(0) * 1000, 2),
            "yaw_std_deg": round(float(np.degrees(dy.std())), 3),
        }
    lm = rows[0]["layout_metadata"]
    fy = lm.get("foundation_yaw_degrees", {})
    pads = {}
    for side in ("source", "destination"):
        fid = [b for b in ids if b.startswith(f"{side}_base_")]
        c = np.mean([blocks[b]["pose"][:2] for b in fid], axis=0)
        pads[side] = {"center": _r([c[0], c[1], 0.0]), "foundation_yaw_deg": float(fy.get(side, 0.0)), "foundation_blocks": fid}
    # transfers: same plan in every episode
    plans = [[(m["block_id"], m["arm"], m["source_layer"], m["destination_layer"]) for m in r["target_mapping"]] for r in rows]
    assert all(p == plans[0] for p in plans), f"{variant}: transfer plans differ between episodes"
    goals = np.array([[m["goal"] for m in r["target_mapping"]] for r in rows])  # [E, K, 7] xyzw
    dest = pads["destination"]
    cy, sy = math.cos(math.radians(dest["foundation_yaw_deg"])), math.sin(math.radians(dest["foundation_yaw_deg"]))
    transfers = []
    for k, (bid, arm, sl, dl) in enumerate(plans[0]):
        gq = mean_quat_wxyz(goals[:, k][:, [6, 3, 4, 5]])
        gp = goals[:, k, :3].mean(0)
        rel = gp - np.array(dest["center"])
        rel_pad = [cy * rel[0] + sy * rel[1], -sy * rel[0] + cy * rel[1], rel[2]]
        yaw_rel = math.degrees((yaw_of_wxyz(gq) - math.radians(dest["foundation_yaw_deg"]) + math.pi / 2) % math.pi - math.pi / 2)
        transfers.append({
            "block": bid, "arm": ARM_NAMES[arm], "source_layer": int(sl), "dest_layer": int(dl),
            "goal": _r(np.concatenate([gp, gq])),
            "goal_in_pad": {"pos": _r(rel_pad), "yaw_deg": round(yaw_rel, 3)},
            "goal_std_mm": _r(goals[:, k, :3].std(0) * 1000, 2),
        })
    cfg = rows[0]["cfg"]
    frames = [r["frames"] for r in rows]
    return {
        "variant": variant,
        "task_name": rows[0]["task_name"],
        "instruction_dataset": rows[0]["instruction"],
        "num_episodes": len(rows),
        "pose_format": "[x, y, z, qw, qx, qy, qz] in the station frame (m); tabletop z = 0",
        "pads": pads,
        "blocks": blocks,
        "transfers": transfers,
        "jitter": {"tower_jitter_m": cfg.get("tower_jitter_m"), "center_block_jitter_m": cfg.get("center_block_jitter_m"),
                   "max_block_std_mm": round(max(max(b["std_mm"][:2]) for b in blocks.values()), 2)},
        "generator": {k: cfg.get(k) for k in ("placement_m", "placement_degrees", "collapse_degrees", "collapse_drop_m",
                                              "collapse_seconds", "gripper_finger_effort_limit_n", "max_seconds")},
        "demo_duration_s": {"min": round(min(frames) / 30, 1), "mean": round(float(np.mean(frames)) / 30, 1), "max": round(max(frames) / 30, 1)},
        "all_demos_successful": all(r["success_final"] for r in rows),
    }


def _pad_yaw_deg(blocks: dict, fids: list[str]) -> float:
    """Foundation yaw of a pad (half-turn symmetric, in [-90, 90)) from its blocks' mean orientation."""
    y = round(float(np.mean([math.degrees(yaw_of_wxyz(blocks[b]["pose"][3:7])) % 180.0 for b in fids])), 3)
    return round((y + 90.0) % 180.0 - 90.0, 3)


def build_layout_t45(variant: str, rows: list[dict]) -> dict:
    """Layout of a Task 4 (push-down) or Task 5 (insertion) variant from ``data/meta_tasks45.jsonl``.

    Task 4: ``pushes`` = the generator's ordered push targets (top layer first, row slot 2 -> 0) with the arm and the
    world push direction measured in the demos (along the block's long axis, away from the arm). Task 5: ``transfers``
    = the inserts (highest gap first) with the slot centre as goal, plus the insertion geometry measured in the demos
    (outside alignment and release depth along the slot axis). Blocks carry their size / mass when they differ from
    the standard block (Task 5's 32 mm side blocks); ``physics`` records the collider type of the generator's stage
    (Task 5-3: cuboid) and the finger effort it applied (3 N for every Task 4/5 demo)."""
    fam = family_of(variant)
    ids = rows[0]["block_ids"]
    assert all(r["block_ids"] == ids for r in rows), f"{variant}: block ids differ between episodes"
    fixed = rows[0]["block_fixed_on_table"]
    dims = rows[0].get("block_dimensions_m") or [None] * len(ids)
    mass = rows[0].get("block_mass_kg") or [None] * len(ids)
    p0 = np.array([r["block_poses_frame0"] for r in rows])  # [E, B, 7] xyzw
    p1 = np.array([r["block_poses_final"] for r in rows])
    blocks = {}
    for i, bid in enumerate(ids):
        quat = p0[:, i][:, [6, 3, 4, 5]]
        mq = mean_quat_wxyz(quat)
        yaws = np.array([yaw_of_wxyz(q) for q in quat])
        dy = (yaws - yaw_of_wxyz(mq) + math.pi / 2) % math.pi - math.pi / 2
        b = {"pose": _r(np.concatenate([p0[:, i, :3].mean(0), mq])), "fixed": bool(fixed[i]), "side": side_of(bid),
             "layer": layer_of(bid), "std_mm": _r(p0[:, i, :3].std(0) * 1000, 2), "yaw_std_deg": round(float(np.degrees(dy.std())), 3)}
        if dims[i] is not None and [round(x, 4) for x in dims[i]] != [0.15, 0.05, 0.03]:
            b["dims_m"] = _r(dims[i], 4)
        if mass[i] is not None:
            b["mass_kg"] = round(float(mass[i]), 5)
        if fam == "insertion":
            b["role"] = "foundation" if fixed[i] else "insert" if "insert" in bid else "side" if "side" in bid else "roof"
        blocks[bid] = b
    pads = {}
    for side in ("source", "destination"):
        fid = [b for b in ids if b.startswith(f"{side}_base_")]
        c = np.mean([blocks[b]["pose"][:2] for b in fid], axis=0)
        pads[side] = {"center": _r([c[0], c[1], 0.0]), "foundation_yaw_deg": _pad_yaw_deg(blocks, fid), "foundation_blocks": fid}
    cfg = rows[0]["cfg"]
    frames = [r["frames"] for r in rows]
    out = {
        "variant": variant, "family": fam, "task_name": rows[0]["task_name"],
        "instruction_dataset": rows[0]["instruction"], "num_episodes": len(rows),
        "pose_format": "[x, y, z, qw, qx, qy, qz] in the station frame (m); tabletop z = 0",
        "pads": pads, "blocks": blocks, "transfers": [], "pushes": [],
    }
    if fam == "pushdown":
        lm = rows[0]["layout_metadata"]
        assert all([p["block_id"] for p in r["layout_metadata"]["push_targets"]] == [p["block_id"] for p in lm["push_targets"]]
                   for r in rows), f"{variant}: push orders differ between episodes"
        for p in lm["push_targets"]:
            geo_ev = [e for r in rows for e in r["events"] if e.get("event") == "push_geometry" and e.get("block_id") == p["block_id"]]
            dep = [e for r in rows for e in r["events"] if e.get("event") == "push_departed" and e.get("block_id") == p["block_id"]]
            d = np.mean([e["direction"] for e in geo_ev], axis=0)
            d[2] = 0.0
            d = d / np.linalg.norm(d)
            out["pushes"].append({
                "block": p["block_id"], "arm": ARM_NAMES[p["arm"]], "source_layer": int(p["source_layer"]),
                "direction": _r(d, 4),
                "demo": {"push_distance_m": round(float(np.mean([e["push_distance_m"] for e in geo_ev])), 4),
                         "pad_height_m": round(float(np.mean([e["pad_height_m"] for e in geo_ev])), 4),
                         "wrist_pitch_deg": round(float(np.mean([e["wrist_pitch_degrees"] for e in geo_ev])), 2),
                         "support_clearance_at_departure_mm": _r(np.percentile([e["minimum_measured_support_clearance_m"] * 1000 for e in dep], [0, 50, 100]), 2)},
            })
    else:  # insertion: same transfer format as Tasks 1-3 (goal = slot centre), destination = the tower's pad
        plans = [[(m["block_id"], m["arm"], m["source_layer"], m["destination_layer"]) for m in r["target_mapping"]] for r in rows]
        assert all(p == plans[0] for p in plans), f"{variant}: insert plans differ between episodes"
        goals = np.array([[m["goal"] for m in r["target_mapping"]] for r in rows])
        dest = pads["destination"]
        cy, sy = math.cos(math.radians(dest["foundation_yaw_deg"])), math.sin(math.radians(dest["foundation_yaw_deg"]))
        for k, (bid, arm, sl, dl) in enumerate(plans[0]):
            gq = mean_quat_wxyz(goals[:, k][:, [6, 3, 4, 5]])
            gp = goals[:, k, :3].mean(0)
            rel = gp - np.array(dest["center"])
            yaw_rel = math.degrees((yaw_of_wxyz(gq) - math.radians(dest["foundation_yaw_deg"]) + math.pi / 2) % math.pi - math.pi / 2)
            bi = ids.index(bid)
            gpos_ep, gyaw_ep = goals[:, k, :3], np.array([yaw_of_wxyz(q) for q in goals[:, k][:, [6, 3, 4, 5]]])

            def along(ev_name, key="block_target"):
                """Signed block-centre distance from the slot centre along the slot axis at an event (demo mean)."""
                vals = []
                for r, g, gy in zip(rows, gpos_ep, gyaw_ep):
                    e = next((e for e in r["events"] if e.get("event") == ev_name and e.get("block_id") == bid and key in e), None)
                    if e is not None:
                        vals.append(math.cos(gy) * (e[key][0] - g[0]) + math.sin(gy) * (e[key][1] - g[1]))
                return round(float(np.mean(vals)), 4) if vals else None

            fin = p1[:, bi]
            out["transfers"].append({
                "block": bid, "arm": ARM_NAMES[arm], "source_layer": int(sl), "dest_layer": int(dl),
                "goal": _r(np.concatenate([gp, gq])),
                "goal_in_pad": {"pos": _r([cy * rel[0] + sy * rel[1], -sy * rel[0] + cy * rel[1], rel[2]]), "yaw_deg": round(yaw_rel, 3)},
                "goal_std_mm": _r(goals[:, k, :3].std(0) * 1000, 2),
                "side_blocks": sorted(b for b in ids if b.startswith(f"destination_l{dl}_side")),
                "rest_z_final": round(float(fin[:, 2].mean()), 5),
                "final_error_mm": _r(np.percentile(np.linalg.norm(fin[:, :2] - gpos_ep[:, :2], axis=-1) * 1000, [50, 90, 100]), 2),
                "demo": {"outside_alignment_m": along("outside_alignment"), "horizontal_entry_m": along("horizontal_entry")},
            })
    out["jitter"] = {"tower_jitter_m": cfg.get("tower_jitter_m"), "center_block_jitter_m": cfg.get("center_block_jitter_m"),
                     "max_block_std_mm": round(max(max(b["std_mm"][:2]) for b in blocks.values()), 2)}
    out["generator"] = {k: cfg.get(k) for k in ("placement_m", "placement_degrees", "collapse_degrees", "collapse_drop_m",
                                                "collapse_seconds", "gripper_finger_effort_limit_n", "max_seconds",
                                                "settle_seconds", "final_settle_seconds")}
    stages = {r["stage"] for r in rows}
    assert len(stages) == 1, f"{variant}: several generator stages {stages}"
    out["physics"] = {"collider": "cuboid" if "cuboid" in next(iter(stages)) else "hull",
                      "finger_effort_n": float(rows[0]["gripper_effort"]["applied_limit_n"]) if rows[0].get("gripper_effort") else cfg.get("gripper_finger_effort_limit_n")}
    out["demo_duration_s"] = {"min": round(min(frames) / 30, 1), "mean": round(float(np.mean(frames)) / 30, 1), "max": round(max(frames) / 30, 1)}
    out["all_demos_successful"] = all(r["success_final"] for r in rows)
    return out


def write_layout(layout: dict, path: str | Path | None = None) -> Path:
    path = Path(path) if path else LAYOUT_DIR / f"{layout['variant']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)

    class _Dumper(yaml.SafeDumper):
        pass

    def _list(dumper, data):  # short numeric lists inline, everything else block style
        flow = len(data) <= 8 and all(isinstance(x, (int, float)) for x in data)
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)

    _Dumper.add_representer(list, _list)
    meta = METADATA_JSONL if layout.get("family", "transfer") == "transfer" else METADATA_T45_JSONL
    header = f"# {layout['variant']}: generated by scripts/build_layouts.py from data/{meta.name}; do not edit\n"
    path.write_text(header + yaml.dump(layout, Dumper=_Dumper, sort_keys=False, width=120))
    return path


# ---------------------------------------------------------------------- runtime access
@dataclass
class Transfer:
    block: str
    arm: str
    source_layer: int
    dest_layer: int
    goal: np.ndarray  # [7] xyz + wxyz


@dataclass
class Push:
    block: str
    arm: str
    source_layer: int
    direction: np.ndarray  # [3] world unit vector (horizontal): along the block's long axis, away from the arm


@dataclass
class Layout:
    variant: str
    task_name: str
    blocks: dict[str, np.ndarray]  # id -> [7] pose (xyz + wxyz)
    fixed: dict[str, bool]
    layer: dict[str, int]
    side: dict[str, str]
    pads: dict[str, dict]
    transfers: list[Transfer]
    generator: dict
    raw: dict = field(repr=False, default_factory=dict)
    family: str = "transfer"                                  # transfer | pushdown | insertion
    pushes: list[Push] = field(default_factory=list)          # Task 4: ordered push operations
    dims: dict[str, tuple] = field(default_factory=dict)      # blocks whose size differs from 0.15 x 0.05 x 0.03 m
    mass: dict[str, float] = field(default_factory=dict)      # blocks with an explicit mass (Task 5 side blocks)
    physics: dict = field(default_factory=dict)               # {"collider": hull | cuboid, "finger_effort_n": ...}

    def role(self, bid: str) -> str:
        """Task 5 block role (foundation / side / roof / insert); ``block`` elsewhere."""
        return self.raw["blocks"][bid].get("role", "foundation" if self.fixed[bid] else "block")

    @property
    def block_ids(self) -> list[str]:
        return list(self.blocks)

    @property
    def movable_ids(self) -> list[str]:
        return [b for b in self.blocks if not self.fixed[b]]

    @property
    def num_layers(self) -> int:
        return max(self.layer.values())

    @classmethod
    def load(cls, variant: str, path: str | Path | None = None) -> Layout:
        path = Path(path) if path else LAYOUT_DIR / f"{variant}.yaml"
        d = yaml.safe_load(path.read_text())
        blocks = {k: np.array(v["pose"], dtype=np.float64) for k, v in d["blocks"].items()}
        return cls(
            variant=d["variant"], task_name=d["task_name"], blocks=blocks,
            fixed={k: v["fixed"] for k, v in d["blocks"].items()}, layer={k: v["layer"] for k, v in d["blocks"].items()},
            side={k: v["side"] for k, v in d["blocks"].items()}, pads=d["pads"],
            transfers=[Transfer(t["block"], t["arm"], t["source_layer"], t["dest_layer"], np.array(t["goal"], dtype=np.float64))
                       for t in d["transfers"]],
            generator=d["generator"], raw=d,
            family=d.get("family", "transfer"),
            pushes=[Push(p["block"], p["arm"], p["source_layer"], np.array(p["direction"], dtype=np.float64)) for p in d.get("pushes", [])],
            dims={k: tuple(v["dims_m"]) for k, v in d["blocks"].items() if "dims_m" in v},
            mass={k: float(v["mass_kg"]) for k, v in d["blocks"].items() if "mass_kg" in v},
            physics=d.get("physics", {}),
        )
