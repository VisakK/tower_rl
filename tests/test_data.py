"""Step 1 acceptance: subset files parse, operation counts match, layouts are consistent with the published descriptor."""

import math

import numpy as np
import pytest

from tower_rl.data.episode import Episode
from tower_rl.data.events import expected_operation_count, operations
from tower_rl.data.layouts import LAYOUT_DIR, VARIANTS, VARIANTS_T4, VARIANTS_T5, Layout, family_of, yaw_of_wxyz
from tower_rl.paths import DATA_DIR

SUBSET = sorted((DATA_DIR / "subset").glob("*/*.npz"))
REQUIRED = ["observations__qpos", "leader__qpos", "observations__eef", "leader__eef", "observations__gripper_status",
            "sim__block_poses", "sim__time", "sim__success", "sim__terminal"]


@pytest.mark.skipif(not SUBSET, reason="run scripts/fetch_subset.py first")
@pytest.mark.parametrize("npz", SUBSET, ids=lambda p: f"{p.parent.name}/{p.stem[:14]}")
def test_subset_episode(npz):
    ep = Episode.load(npz)
    for k in REQUIRED:
        assert k in ep.arrays, k
    T = ep.num_frames
    assert ep.qpos_cmd.shape == (T, 16) and ep.eef_obs.shape == (T, 14)
    assert ep.block_poses.shape == (T, len(ep.block_ids), 7)
    assert np.isfinite(ep.qpos_obs).all() and np.isfinite(ep.block_poses).all()
    assert np.all(np.diff(ep.time) > 0)
    assert npz.parent.name.startswith(ep.task + "_")  # public_task, e.g. task3-3 (task_name says aligned_stack_3)
    assert bool(ep.success[-1])
    if family_of(npz.parent.name) == "pushdown":  # Task 4: no grasps; every push target ends off the tower footprint
        lm = ep.layout_metadata
        c = np.array(lm["tower_centers_m"][0][:2])
        for p in lm["push_targets"]:  # (layer-1 blocks often land on earlier pushed blocks, 3-4 cm above the table)
            assert np.linalg.norm(ep.block_poses[-1, p["block_index"], :2] - c) > 0.14, p
        return
    if family_of(npz.parent.name) == "insertion":  # Task 5: each insert ends at its slot centre
        for m in ep.target_mapping:
            assert np.linalg.norm(ep.block_poses[-1, m["block_index"], :2] - np.array(m["goal"][:2])) < 0.004, m
        return
    ops = operations(ep)
    assert len(ops) == expected_operation_count(npz.parent.name), [o.to_dict() for o in ops]
    for op in ops:
        assert op.open_frame > op.close_frame
        assert op.block_id == op.active_block_id
        assert 0.045 < op.held_width_m < 0.055  # holding a 0.0497 m block, not an empty close
        assert op.lift_m > 0.04
    planned = [(m["block_id"], ("left", "back")[m["arm"]]) for m in ep.target_mapping]
    assert [(o.block_id, o.arm) for o in ops] == planned  # demos follow the plan order and arm assignment


@pytest.mark.parametrize("variant", VARIANTS_T4 + VARIANTS_T5)
def test_layout_file_t45(variant):
    lay = Layout.load(variant)
    n_layers = int(variant[-1])
    if lay.family == "pushdown":  # full layers of three, pushed top layer first, row slot 2 -> 0
        assert len(lay.movable_ids) == 3 * n_layers and len(lay.pushes) == 3 * n_layers and not lay.transfers
        assert [p.source_layer for p in lay.pushes] == sorted([p.source_layer for p in lay.pushes], reverse=True)
        for p in lay.pushes:  # along the block's long axis, away from the arm (left +x, back +y)
            yaw = yaw_of_wxyz(lay.blocks[p.block][3:7])
            assert abs(abs(math.cos(yaw) * p.direction[0] + math.sin(yaw) * p.direction[1]) - 1) < 0.01
            assert p.direction[0 if p.arm == "left" else 1] > 0.99
    else:  # one insert per interior layer, highest gap first; side blocks 32 mm tall
        assert len(lay.transfers) == n_layers - 1
        assert [t.dest_layer for t in lay.transfers] == sorted([t.dest_layer for t in lay.transfers], reverse=True)
        sides = [b for b in lay.movable_ids if lay.role(b) == "side"]
        assert len(sides) == 2 * (n_layers - 1) and all(abs(lay.dims[b][2] - 0.032) < 1e-6 for b in sides)
        dest = np.array(lay.pads["destination"]["center"][:2])
        for t in lay.transfers:
            assert np.linalg.norm(t.goal[:2] - dest) < 0.003
        assert lay.physics["collider"] == ("cuboid" if variant.endswith("_3") else "hull")
    assert lay.physics["finger_effort_n"] == 3.0


@pytest.mark.parametrize("variant", VARIANTS)
def test_layout_file(variant):
    lay = Layout.load(variant)
    assert lay.variant == variant
    n_layers = int(variant[-1])
    per_layer = 2 if variant.startswith("task1") else 1
    assert len(lay.transfers) == n_layers * per_layer
    assert len(lay.movable_ids) == n_layers * per_layer
    # top layer first, destination layers ascending
    src_layers = [t.source_layer for t in lay.transfers]
    assert src_layers == sorted(src_layers, reverse=True)
    dst_layers = [t.dest_layer for t in lay.transfers]
    assert dst_layers == sorted(dst_layers)
    # goals sit on the destination pad at the 0.03 m layer pitch
    dest = np.array(lay.pads["destination"]["center"][:2])
    for t in lay.transfers:
        assert abs(t.goal[2] - (0.015 + 0.03 * t.dest_layer)) < 1e-3
        assert np.linalg.norm(t.goal[:2] - dest) < 0.06
    # the two pads are ~0.36 m apart
    src = np.array(lay.pads["source"]["center"][:2])
    assert 0.33 < np.linalg.norm(src - dest) < 0.38


def test_task23_matches_published_descriptor():
    """tasks/cross_tower_transfer_3-cfg08.json values (from the build instructions): <= 5 mm / 3 deg."""
    lay = Layout.load("task2-3_cross_tower_transfer_3")
    src, dst = lay.pads["source"], lay.pads["destination"]
    assert np.linalg.norm(np.array(src["center"][:2]) - [0.42854, -0.27528]) < 0.005
    assert np.linalg.norm(np.array(dst["center"][:2]) - [0.17174, -0.52788]) < 0.005
    assert src["foundation_yaw_deg"] == -90 and dst["foundation_yaw_deg"] == 0
    desc_yaw = {"source_l1_center": -90.0, "source_l2_center": 0.0, "source_l3_center": -90.0}
    desc_z = {"source_l1_center": 0.045, "source_l2_center": 0.075, "source_l3_center": 0.105}
    for b, yaw in desc_yaw.items():
        p = lay.blocks[b]
        assert np.linalg.norm(p[:2] - np.array(src["center"][:2])) < 0.005
        assert abs(p[2] - desc_z[b]) < 0.005
        d = (math.degrees(yaw_of_wxyz(p[3:7])) - yaw + 90) % 180 - 90
        assert abs(d) < 3.0
    plan = [(t.block, t.arm, t.dest_layer, round(t.goal[2], 3)) for t in lay.transfers]
    assert plan == [("source_l3_center", "back", 1, 0.045), ("source_l2_center", "left", 2, 0.075), ("source_l1_center", "back", 3, 0.105)]


def test_layout_files_exist():
    for v in VARIANTS:
        assert (LAYOUT_DIR / f"{v}.yaml").is_file()
