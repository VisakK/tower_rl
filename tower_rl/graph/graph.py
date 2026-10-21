"""Task graphs per variant (Step 7.3): typed skill-call nodes, compiled from the layout config to editable JSON.

Per transfer k (plan order, top layer first; arms as in the demonstrations):
  transit(arm -> pre-grasp of block k) -> grasp_lift(arm, block k) -> transit(arm -> pre-place of goal k, holding)
  -> place_release(arm, goal k)
with a ``park`` node whenever the next transfer uses the other arm (the idle arm must be out of the way) and at the
end. Edges are implicit and uniform: ``ok`` -> next node; ``failed``/``timeout`` -> retry the transfer from its first
node up to ``retries`` times (re-transit, re-grasp; a failed place re-grasps the dropped/misplaced block), then skip
to the next transfer (LHPR still counts later successes). Every tunable the edit loop may touch is a node field.
Files: ``graph/graphs/<variant>.json`` (pure data; the executor interprets them deterministically).
"""

from __future__ import annotations

import json
from pathlib import Path

from tower_rl.config import load_yaml
from tower_rl.data.layouts import VARIANTS, Layout
from tower_rl.paths import REPO_ROOT

GRAPH_DIR = REPO_ROOT / "tower_rl" / "graph" / "graphs"
SKILLS = ("transit", "grasp_lift", "place_release", "park", "push", "insert", "push_in")


def build_graph_pushdown(variant: str, retries: int = 2) -> dict:
    """Task 4: per push k (prescribed order) transit(arm -> pre-push pose behind block k, fingers closed as a tool) ->
    push(arm, block k: until it left the tower, then retreat); a park whenever the next push uses the other arm and at
    the end. ``transfer`` indexes the push (one "transfer" per push for the executor's retry / skip edges)."""
    lay = Layout.load(variant)
    pg = load_yaml("skills/push")["geometry"]
    nodes = []
    prev_arm = None
    for k, p in enumerate(lay.pushes):
        if prev_arm is not None and prev_arm != p.arm:
            nodes.append({"id": f"park_{prev_arm}_{k}", "skill": "park", "arm": prev_arm, "transfer": k, "timeout_s": 12.0,
                          "first_of_transfer": False})
        tag = f"p{k}"
        nodes += [
            {"id": f"{tag}_to_prepush", "skill": "transit", "arm": p.arm, "transfer": k, "block": p.block, "target": "prepush",
             "direction": [round(float(x), 4) for x in p.direction], "layer": p.source_layer,
             "tol_pos_m": 0.02, "tol_rot_deg": 5.0, "timeout_s": 15.0, "first_of_transfer": True},
            {"id": f"{tag}_push", "skill": "push", "arm": p.arm, "transfer": k, "block": p.block,
             "direction": [round(float(x), 4) for x in p.direction], "layer": p.source_layer,
             "depart_m": pg["depart_m"], "retreat_m": pg["retreat_m"], "structure_mm": 10.0, "timeout_s": 12.0},
        ]
        prev_arm = p.arm
    nodes.append({"id": f"park_{prev_arm}_end", "skill": "park", "arm": prev_arm, "transfer": len(lay.pushes), "timeout_s": 12.0,
                  "first_of_transfer": False})
    return {"variant": variant, "family": "pushdown", "version": 1, "retries": retries, "rate_hz": 10,
            "gripper_closed_width": load_yaml("skills/insert")["geometry"]["held_close_width_m"],
            "chunk": {"transit": 16, "contact": 1}, "nodes": nodes,
            "foundation": {"destination": lay.pads["destination"]["foundation_blocks"], "source": lay.pads["source"]["foundation_blocks"],
                           "destination_yaw_deg": lay.pads["destination"]["foundation_yaw_deg"]},
            "tower": {"pad": "source", "foundation": lay.pads["source"]["foundation_blocks"]},
            "movable": lay.movable_ids}


def build_graph(variant: str, retries: int = 2) -> dict:
    lay = Layout.load(variant)
    if lay.family == "pushdown":
        return build_graph_pushdown(variant, retries)
    if lay.family == "insertion":
        return build_graph_insertion(variant, retries)
    g = load_yaml("skills/grasp_geometry")
    nodes = []
    prev_arm = None
    for k, t in enumerate(lay.transfers):
        if prev_arm is not None and prev_arm != t.arm:
            nodes.append({"id": f"park_{prev_arm}_{k}", "skill": "park", "arm": prev_arm, "transfer": k, "timeout_s": 12.0,
                          "first_of_transfer": False})
        tag = f"t{k}"
        nodes += [
            {"id": f"{tag}_to_pregrasp", "skill": "transit", "arm": t.arm, "transfer": k, "block": t.block, "target": "pregrasp",
             "height_m": g["pregrasp_height_m"], "tol_pos_m": 0.03, "tol_rot_deg": 5.0, "timeout_s": 15.0, "first_of_transfer": True},
            {"id": f"{tag}_grasp", "skill": "grasp_lift", "arm": t.arm, "transfer": k, "block": t.block, "lift_m": 0.06,
             "timeout_s": 15.0},
            {"id": f"{tag}_to_preplace", "skill": "transit", "arm": t.arm, "transfer": k, "block": t.block, "target": "preplace",
             "dest_layer": t.dest_layer, "goal_in_pad": lay.raw["transfers"][k]["goal_in_pad"],
             "height_m": g["preplace_height_m"], "tol_pos_m": 0.03, "tol_rot_deg": 5.0, "timeout_s": 20.0, "held": True},
            {"id": f"{tag}_place", "skill": "place_release", "arm": t.arm, "transfer": k, "block": t.block,
             "dest_layer": t.dest_layer, "goal_in_pad": lay.raw["transfers"][k]["goal_in_pad"],
             "release_clearance_m": g["release_clearance_m"], "tol_pos_m": 0.005, "tol_yaw_deg": 5.0, "settle_s": 1.0,
             "timeout_s": 20.0},
        ]
        prev_arm = t.arm
    nodes.append({"id": f"park_{prev_arm}_end", "skill": "park", "arm": prev_arm, "transfer": len(lay.transfers), "timeout_s": 12.0,
                  "first_of_transfer": False})
    return {"variant": variant, "version": 1, "retries": retries, "rate_hz": 10,
            "chunk": {"transit": 16, "contact": 1}, "nodes": nodes,
            "foundation": {"destination": lay.pads["destination"]["foundation_blocks"], "source": lay.pads["source"]["foundation_blocks"],
                           "destination_yaw_deg": lay.pads["destination"]["foundation_yaw_deg"]},
            "movable": lay.movable_ids}


def validate(graph: dict) -> None:
    ids = [n["id"] for n in graph["nodes"]]
    assert len(ids) == len(set(ids)), "duplicate node ids"
    for n in graph["nodes"]:
        assert n["skill"] in SKILLS, n
        assert n.get("on_fail", "retry") in ("retry", "skip", "skip_if_close"), n
        assert n["arm"] in ("left", "back"), n
        assert n["timeout_s"] > 0, n
    # one arm active between parks: every node of a transfer uses the transfer's arm
    for n in graph["nodes"]:
        if n["skill"] != "park":
            assert any(m["skill"] == "transit" and m["transfer"] == n["transfer"] and m.get("first_of_transfer") for m in graph["nodes"])


def save_graph(graph: dict, path: str | Path | None = None) -> Path:
    validate(graph)
    path = Path(path) if path else GRAPH_DIR / f"{graph['variant']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, indent=1) + "\n")
    return path


def load_graph(variant: str, path: str | Path | None = None) -> dict:
    path = Path(path) if path else GRAPH_DIR / f"{variant}.json"
    g = json.loads(Path(path).read_text())
    validate(g)
    return g


def insertion_direction(goal_pos, goal_yaw: float, arm: str) -> list[float]:
    """Unit world direction of an insertion: along the slot (the goal's long axis), away from the arm (demos: the left
    arm inserts toward +x, the back arm toward +y)."""
    import math

    import numpy as np

    from tower_rl.kinematics import load_model
    d = np.array([math.cos(goal_yaw), math.sin(goal_yaw), 0.0])
    base = np.array(load_model()["base_world"][arm]["pos"][:2])
    if float(np.dot(d[:2], np.asarray(goal_pos[:2]) - base)) < 0:
        d = -d
    return [round(float(x), 6) for x in d]


def build_graph_insertion(variant: str, retries: int = 2) -> dict:
    """Task 5: per insert k (highest gap first) transit(arm -> pre-grasp of the insert block on the source stack) ->
    grasp_lift -> transit(held -> pre-insert: block on the slot axis, ``outside_m`` out, at slot height) -> insert (held
    slide to ``entry_m`` from the slot centre, release, retreat) -> transit(-> pre-push-in behind the block's outer end
    face, fingers at the pusher width) -> push_in (to the slot centre, retreat); a park whenever the next insert uses
    the other arm and at the end. Geometry from the demos (layout ``transfers[].demo``, ``configs/skills/insert.yaml``)."""
    import math

    from tower_rl.data.layouts import yaw_of_wxyz
    lay = Layout.load(variant)
    g = load_yaml("skills/grasp_geometry")
    ig = load_yaml("skills/insert")["geometry"]
    nodes = []
    prev_arm = None
    for k, t in enumerate(lay.transfers):
        if prev_arm is not None and prev_arm != t.arm:
            nodes.append({"id": f"park_{prev_arm}_{k}", "skill": "park", "arm": prev_arm, "transfer": k, "timeout_s": 12.0,
                          "first_of_transfer": False})
        raw = lay.raw["transfers"][k]
        d = insertion_direction(t.goal[:3], yaw_of_wxyz(t.goal[3:7]), t.arm)
        # the layer below this slot is still open (a later insert fills it): the block crosses that gap (5-3 layer 2)
        gap_below = any(u.dest_layer == t.dest_layer - 1 for u in lay.transfers[k + 1:])
        slot = {"goal_in_pad": raw["goal_in_pad"], "dest_layer": t.dest_layer, "goal_z": float(t.goal[2]),
                "direction": d, "side_blocks": raw["side_blocks"]}
        tag = f"i{k}"
        nodes += [
            # pre-grasp 4 cm (not 6): the left arm at pad B cannot reach 6 cm above a stack rotated +5-10 deg (33-41 mm
            # off; the shifted / combined cases failed "pregrasp unreachable"); 4 cm stays within 23 mm
            {"id": f"{tag}_to_pregrasp", "skill": "transit", "arm": t.arm, "transfer": k, "block": t.block, "target": "pregrasp",
             "height_m": 0.04, "tol_pos_m": 0.03, "tol_rot_deg": 5.0, "timeout_s": 15.0, "first_of_transfer": True},
            {"id": f"{tag}_grasp", "skill": "grasp_lift", "arm": t.arm, "transfer": k, "block": t.block, "lift_m": 0.06, "timeout_s": 15.0,
             "lift_v_m_s": 0.04, "lift_a_m_s2": 0.1},  # planned lift as gentle as the demos' (3 N fingers, heavy blocks)
            {"id": f"{tag}_to_preinsert", "skill": "transit", "arm": t.arm, "transfer": k, "block": t.block, "target": "preinsert",
             **slot, "outside_m": ig["outside_m"], "tol_pos_m": 0.01, "tol_rot_deg": 3.0, "timeout_s": 25.0, "held": True,
             "v_max_held": 0.06},
            {"id": f"{tag}_insert", "skill": "insert", "arm": t.arm, "transfer": k, "block": t.block, **slot,
             "entry_m": ig["entry_m"], "entry_lift_m": ig["entry_lift_m"][lay.physics.get("collider", "hull")], "timeout_s": 30.0,
             "on_fail": "skip"},  # a re-grasp from inside the tower wrecks it (5-3 trace): give the transfer up
            {"id": f"{tag}_to_prepushin", "skill": "transit", "arm": t.arm, "transfer": k, "block": t.block, "target": "prepushin",
             **slot, "tol_pos_m": 0.01, "tol_rot_deg": 3.0, "timeout_s": 15.0},
            {"id": f"{tag}_push_in", "skill": "push_in", "arm": t.arm, "transfer": k, "block": t.block, **slot,
             "tol_pos_m": 0.004, "timeout_s": 20.0, "on_fail": "skip", "gap_below": gap_below},
        ]
        prev_arm = t.arm
    nodes.append({"id": f"park_{prev_arm}_end", "skill": "park", "arm": prev_arm, "transfer": len(lay.transfers), "timeout_s": 12.0,
                  "first_of_transfer": False})
    return {"variant": variant, "family": "insertion", "version": 1, "retries": retries, "rate_hz": 10,
            "gripper_closed_width": ig["held_close_width_m"], "grasp_planned": True,
            "chunk": {"transit": 16, "contact": 1}, "nodes": nodes,
            "foundation": {"destination": lay.pads["destination"]["foundation_blocks"], "source": lay.pads["source"]["foundation_blocks"],
                           "destination_yaw_deg": lay.pads["destination"]["foundation_yaw_deg"]},
            "tower": {"pad": "destination", "foundation": lay.pads["destination"]["foundation_blocks"],
                      "structure": [b for b in lay.movable_ids if lay.role(b) in ("side", "roof")]},
            "movable": lay.movable_ids}


if __name__ == "__main__":
    import sys
    for v in (sys.argv[1:] or VARIANTS):
        print(save_graph(build_graph(v)))
