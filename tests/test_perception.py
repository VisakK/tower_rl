"""Pose-estimator geometry: camera models, box ray casting, labels, and the fit on a synthetic scene (oracle maps)."""

import numpy as np
import pytest
import torch

from tower_rl import geometry as geo
from tower_rl.perception.boxes import corner_visibility, corners_world, raycast, visual_half_extents
from tower_rl.perception.cameras import CAMERA_IDS, HEIGHT, WIDTH, CameraRig, _turned, intrinsics
from tower_rl.perception.data import keypoints, splat
from tower_rl.perception.estimator import EstCfg, PoseEstimator, block_order
from tower_rl.perception.model import SIGMA

HOME = np.array([[-0.0375, -0.8007, -0.0041, 1.7538, 0.0117, -0.0255, 1.2400],
                 [-0.1573, -0.8276, 0.0513, 1.7956, 0.0005, -0.0428, 1.2803]])


def test_turned_camera_projects_to_turned_pixels():
    rig = CameraRig("evaluator")
    cam = rig.static["right_front_camera"]
    k_native = intrinsics("right_front_camera", "evaluator")
    r_native = cam.R_wc @ np.diag([-1.0, -1.0, 1.0])  # undo the turn
    x = np.array([[0.43, -0.28, 0.06], [0.17, -0.52, 0.02]])
    xc = (x - cam.t_wc) @ r_native
    uv_native = xc[:, :2] / xc[:, 2:] * np.diag(k_native)[:2] + k_native[:2, 2]
    uv, _ = cam.project(x)
    np.testing.assert_allclose(uv, np.stack([WIDTH - 1 - uv_native[:, 0], HEIGHT - 1 - uv_native[:, 1]], 1), atol=1e-9)
    k2, r2 = _turned(*_turned(k_native, r_native))
    np.testing.assert_allclose(k2, k_native)
    np.testing.assert_allclose(r2, r_native)


def test_render_and_evaluator_models_differ_by_the_principal_point():
    ev, rd = CameraRig("evaluator").cameras(HOME), CameraRig("render").cameras(HOME)
    for n in CAMERA_IDS:
        np.testing.assert_allclose(ev[n].R_wc, rd[n].R_wc)
        k = intrinsics(n, "evaluator")
        assert abs(rd[n].K[0, 0] - 0.5 * (k[0, 0] + k[1, 1])) < 1e-9


def test_raycast_matches_projected_box():
    cam = CameraRig("evaluator").static["right_front_camera"]
    pos = np.array([[0.43, -0.28, 0.015]])
    quat = np.array([[1.0, 0.0, 0.0, 0.0]])
    half = visual_half_extents([[0.15, 0.05, 0.03]])
    depth, index = raycast(cam, pos, quat, half)
    uv, z = cam.project(corners_world(pos, quat, half)[0])
    lo, hi = uv.min(0), uv.max(0)
    ys, xs = torch.nonzero(index == 0, as_tuple=True)
    assert len(xs) > 500
    assert xs.min() >= np.floor(lo[0]) and xs.max() <= np.ceil(hi[0])
    assert ys.min() >= np.floor(lo[1]) and ys.max() <= np.ceil(hi[1])
    assert float(depth[index == 0].min()) >= z.min() - 1e-4 and float(depth[index == 0].max()) <= z.max() + 1e-4
    # every corner on the near side is visible against the box's own depth; the hidden one is not
    _, _, vis = corner_visibility(cam, corners_world(pos, quat, half), np.nan_to_num(depth.numpy(), posinf=10.0))
    assert 4 <= int(vis.sum()) <= 7


VARIANT = "task2-3_cross_tower_transfer_3"


def _oracle_maps(ids, pos, quat):
    """Corner / edge heatmaps splatted from the blocks' visible-corner and visible-edge labels (no network)."""
    from tower_rl.data.layouts import Layout

    lay = Layout.load(VARIANT)
    half = visual_half_extents([lay.dims.get(b, (0.15, 0.05, 0.03)) for b in ids])
    rec = {"pos": np.asarray(pos).tolist(), "quat": np.asarray(quat).tolist(), "half": half.tolist(),
           "cams": {k: c.to_dict() for k, c in CameraRig("render").cameras(HOME).items()}}
    cm, em = [], []
    for name in CAMERA_IDS:
        c, e = keypoints(rec, name, None)
        c, e = torch.as_tensor(c, dtype=torch.float32)[None], torch.as_tensor(e, dtype=torch.float32)[None]
        cm.append(splat(c, torch.tensor([c.shape[1]]), HEIGHT, WIDTH, SIGMA["corner"], 4)[0])
        em.append(splat(e, torch.tensor([e.shape[1]]), HEIGHT, WIDTH, SIGMA["edge"], 3)[0])
    return torch.stack(cm), torch.stack(em)


@pytest.mark.parametrize("mode", ["prior", "init"])
def test_fit_recovers_poses_from_oracle_maps(mode):
    est = PoseEstimator(None, device="cpu", source="render", cfg=EstCfg())
    est.reset(VARIANT)
    from tower_rl.data.layouts import Layout

    lay = Layout.load(VARIANT)
    assert est.block_ids == block_order(lay)
    p7 = np.array([lay.blocks[b] for b in est.block_ids])
    gt_p, gt_q = p7[:, :3].copy(), p7[:, 3:7]
    rng = np.random.default_rng(0)
    if mode == "prior":  # tracking: a few mm / deg off
        est.pos = torch.as_tensor(gt_p + rng.normal(0, 0.002, gt_p.shape) * [1, 1, 0.5], dtype=torch.float32)
        yaw = torch.as_tensor(rng.normal(0, np.radians(1.5), len(gt_p)), dtype=torch.float32)
        est.R = geo.quat_to_mat(geo.quat_from_yaw(yaw)) @ geo.quat_to_mat(torch.as_tensor(gt_q, dtype=torch.float32))
        est.calls = 1
    else:  # first call from the nominal layout: a "shifted assembly" is found by the search
        gt_p[~est.fixed.numpy()] += [0.012, -0.009, 0.0]
    pos, quat, conf = est(None, HOME, np.array([0.1, 0.1]), maps=_oracle_maps(est.block_ids, gt_p, gt_q))
    err = 1000 * np.linalg.norm(pos - gt_p, axis=-1)
    assert np.median(err) < 0.3 and err.max() < 2.0, err.round(2)
    assert (conf >= 0.5).all()


def test_lost_block_found_on_the_table():
    """A movable block that fell off its tower (lying on the table 11.5 cm away, turned 10 deg, clear of the foundation)
    is found by the table search."""
    est = PoseEstimator(None, device="cpu", source="render", cfg=EstCfg())
    est.reset(VARIANT)
    from tower_rl.data.layouts import Layout

    lay = Layout.load(VARIANT)
    p7 = np.array([lay.blocks[b] for b in est.block_ids])
    gt_p, gt_q = p7[:, :3].copy(), p7[:, 3:7].copy()
    b = est.block_ids.index(sorted(lay.movable_ids, key=lambda x: -lay.layer[x])[0])  # the top block
    gt_p[b] = gt_p[b] + np.array([0.0, 0.115, 0.0])
    gt_p[b, 2] = est.half[b, 2].item()
    gt_q[b] = geo.quat_from_yaw(torch.tensor(np.radians(10.0))).numpy()
    est.calls = 1  # tracking: its prior is still on the tower, and it was moving (falling) last call
    est.vel[b] = torch.tensor([0.0, 0.004, -0.004])
    pos, quat, conf = est(None, HOME, np.array([0.1, 0.1]), maps=_oracle_maps(est.block_ids, gt_p, gt_q))
    assert np.linalg.norm(pos[b] - gt_p[b]) < 0.002, pos[b] - gt_p[b]
    assert conf[b] >= 0.5


def test_knocked_block_found_on_a_layer_not_on_another_block():
    """A block knocked off its place comes to rest flat on another tower's foundation layer, turned 55 deg, 6 cm from
    its prior: the lost-block search finds it on that layer (not a duplicate of a foundation block, whose image
    features a refit can slide onto)."""
    est = PoseEstimator(None, device="cpu", source="render", cfg=EstCfg())
    est.reset(VARIANT)
    from tower_rl.data.layouts import Layout

    lay = Layout.load(VARIANT)
    p7 = np.array([lay.blocks[b] for b in est.block_ids])
    gt_p, gt_q = p7[:, :3].copy(), p7[:, 3:7].copy()
    dest = [est.block_ids.index(x) for x in lay.pads["destination"]["foundation_blocks"]]
    b = est.block_ids.index(sorted(lay.movable_ids, key=lambda x: -lay.layer[x])[0])  # the top block
    top = gt_p[dest, 2].max() + est.half[dest[0], 2].item()
    gt_p[b] = gt_p[dest].mean(0) + np.array([0.01, -0.01, 0.0])
    gt_p[b, 2] = top + est.half[b, 2].item()
    gt_q[b] = geo.quat_from_yaw(torch.tensor(np.radians(55.0))).numpy()
    est.calls = 1
    est.pos[b] = torch.as_tensor(gt_p[b] + np.array([0.04, -0.045, 0.0]), dtype=torch.float32)
    est.R[b] = geo.quat_to_mat(geo.quat_from_yaw(torch.tensor([0.0])))[0]
    est.vel[b] = torch.tensor([-0.003, 0.003, 0.0])  # knocked: moving last call
    pos, quat, conf = est(None, HOME, np.array([0.1, 0.1]), maps=_oracle_maps(est.block_ids, gt_p, gt_q))
    assert np.linalg.norm(pos[b] - gt_p[b]) < 0.002, pos[b] - gt_p[b]
    assert conf[b] >= 0.5
    others = [j for j in range(len(pos)) if j != b]
    assert np.linalg.norm(pos[others] - pos[b], axis=-1).min() > 0.02  # not on top of another block's pose


def test_slid_prior_and_sunken_foundation_recovered():
    """A movable block whose prior slid 9 mm along its length and a foundation whose prior sank 6 mm and tilted 3 deg
    are recovered (regression only: in this uncluttered scene the plain fit recovers them too; the corner-ratio
    re-search and the flat foundations act in cluttered towers, measured by the render stress test, BUILDLOG)."""
    est = PoseEstimator(None, device="cpu", source="render", cfg=EstCfg())
    est.reset(VARIANT)
    from tower_rl.data.layouts import Layout

    lay = Layout.load(VARIANT)
    p7 = np.array([lay.blocks[b] for b in est.block_ids])
    gt_p, gt_q = p7[:, :3].copy(), p7[:, 3:7].copy()
    b = est.block_ids.index(sorted(lay.movable_ids, key=lambda x: -lay.layer[x])[0])  # the top block
    f = int(torch.nonzero(est.fixed)[0, 0])
    est.calls = 1
    R_b = geo.quat_to_mat(torch.as_tensor(gt_q[b], dtype=torch.float32))
    est.pos[b] = torch.as_tensor(gt_p[b], dtype=torch.float32) + 0.009 * R_b[:, 0]  # along its long axis
    est.pos[f, 2] -= 0.006
    est.R[f] = geo.quat_to_mat(geo.quat_from_rotvec(torch.tensor([np.radians(3.0), 0.0, 0.0], dtype=torch.float32))) @ est.R[f]
    pos, quat, conf = est(None, HOME, np.array([0.1, 0.1]), maps=_oracle_maps(est.block_ids, gt_p, gt_q))
    assert np.linalg.norm(pos[b] - gt_p[b]) < 0.0015, pos[b] - gt_p[b]
    assert np.linalg.norm(pos[f] - gt_p[f]) < 0.0015, pos[f] - gt_p[f]
    assert abs(pos[f, 2] - gt_p[f, 2]) < 2e-4


def test_pushed_block_followed_along_its_length():
    """A Task 4 push moved a block 31 mm along its length within one 0.1 s call: the estimate follows (corner test ->
    re-search along the long axis, or the layer search), and the next call's prior carries the motion on."""
    est = PoseEstimator(None, device="cpu", source="render", cfg=EstCfg())
    est.reset(VARIANT)
    from tower_rl.data.layouts import Layout

    lay = Layout.load(VARIANT)
    p7 = np.array([lay.blocks[b] for b in est.block_ids])
    gt_p, gt_q = p7[:, :3].copy(), p7[:, 3:7].copy()
    b = est.block_ids.index(sorted(lay.movable_ids, key=lambda x: -lay.layer[x])[0])  # the top block
    axis = geo.quat_to_mat(torch.as_tensor(gt_q[b], dtype=torch.float32))[:, 0].numpy()
    est.calls = 1
    for k, d in enumerate((0.035, 0.065)):  # 35 mm, then 30 mm more
        p = gt_p.copy()
        p[b] = gt_p[b] + d * axis
        pos, quat, conf = est(None, HOME, np.array([0.1, 0.1]), maps=_oracle_maps(est.block_ids, p, gt_q))
        assert np.linalg.norm(pos[b] - p[b]) < 0.0015, (k, pos[b] - p[b])
        assert conf[b] >= 0.5
    assert np.linalg.norm(est.vel[b].numpy() - 0.03 * axis) < 0.002  # the motion is carried to the next call


def test_block_leaning_against_the_tower_found():
    """A block pushed off the tower top comes to rest leaning (60 deg, low end on the table) next to the tower: the
    lost-block search finds it (Task 4 trace: its estimate otherwise stayed on the tower edge and the push never ended)."""
    est = PoseEstimator(None, device="cpu", source="render", cfg=EstCfg())
    est.reset(VARIANT)
    from tower_rl.data.layouts import Layout

    lay = Layout.load(VARIANT)
    p7 = np.array([lay.blocks[b] for b in est.block_ids])
    gt_p, gt_q = p7[:, :3].copy(), p7[:, 3:7].copy()
    b = est.block_ids.index(sorted(lay.movable_ids, key=lambda x: -lay.layer[x])[0])  # the top block
    th = np.radians(60.0)
    Rb = geo.quat_to_mat(torch.as_tensor(gt_q[b], dtype=torch.float32)).numpy()
    Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    R_new = Rb @ Ry  # tilted about its own short axis
    half = est.half[b].numpy()
    # centre out along the long axis, beyond the foundation's end, low end on the table
    gt_p[b] = gt_p[b] + 0.13 * Rb[:, 0]
    gt_p[b, 2] = half[0] * np.sin(th) + half[2] * np.cos(th)
    gt_q[b] = geo.mat_to_quat(torch.as_tensor(R_new, dtype=torch.float32)).numpy()
    est.calls = 1
    est.pos[b] = torch.as_tensor(p7[b, :3] + 0.08 * Rb[:, 0], dtype=torch.float32)  # last seen at the tower edge,
    est.vel[b] = torch.as_tensor(0.01 * Rb[:, 0], dtype=torch.float32)                # pushed along its length
    for _ in range(3):  # the search runs on the first call it is lost (later ones back off)
        pos, quat, conf = est(None, HOME, np.array([0.1, 0.1]), maps=_oracle_maps(est.block_ids, gt_p, gt_q))
        if conf[b] >= 0.5:
            break
    assert np.linalg.norm(pos[b] - gt_p[b]) < 0.003, pos[b] - gt_p[b]


def test_still_block_out_of_view_keeps_its_pose():
    """A block at rest whose image features vanish (an arm in front of it) is taken as occluded and keeps its pose: it
    is not re-found in the place of an identical block elsewhere (a Task 4 identity swap)."""
    est = PoseEstimator(None, device="cpu", source="render", cfg=EstCfg())
    est.reset(VARIANT)
    from tower_rl.data.layouts import Layout

    lay = Layout.load(VARIANT)
    p7 = np.array([lay.blocks[b] for b in est.block_ids])
    gt_p, gt_q = p7[:, :3].copy(), p7[:, 3:7].copy()
    b = est.block_ids.index(sorted(lay.movable_ids, key=lambda x: -lay.layer[x])[0])  # the top block, at rest
    other = gt_p.copy()
    other[b] = gt_p[b] + np.array([0.0, 0.11, 0.0])  # features of an identical block lying elsewhere ...
    other[b, 2] = est.half[b, 2].item()
    est.calls = 1
    for _ in range(3):  # ... while this block's own features are gone (the maps show the "other" block instead)
        pos, quat, conf = est(None, HOME, np.array([0.1, 0.1]), maps=_oracle_maps(est.block_ids, other, gt_q))
        assert np.linalg.norm(pos[b] - gt_p[b]) < 0.002, pos[b] - gt_p[b]


def test_quaternions_stay_continuous_through_the_foundation_freeze():
    """The box looks the same half-turned, but skills observe raw quaternions: after the foundations freeze (their
    yaws averaged modulo a half turn), every reported quaternion is still the layout's representative (no 180 deg jump;
    a twin from the freeze or a search otherwise turned a pushed block's observed yaw change by 180 deg mid-push)."""
    est = PoseEstimator(None, device="cpu", source="render", cfg=EstCfg(freeze_after=3))
    est.reset(VARIANT)
    from tower_rl.data.layouts import Layout

    lay = Layout.load(VARIANT)
    p7 = np.array([lay.blocks[b] for b in est.block_ids])
    maps = _oracle_maps(est.block_ids, p7[:, :3], p7[:, 3:7])
    for _ in range(5):
        pos, quat, conf = est(None, HOME, np.array([0.1, 0.1]), maps=maps)
    assert bool(est.frozen.any())
    raw = 2 * np.degrees(np.arccos(np.clip(np.abs((quat * p7[:, 3:7]).sum(-1)), 0, 1)))
    assert raw.max() < 5.0, raw.round(1)


def test_video_frame_draws_the_estimates_and_the_truth_where_off():
    """The closed-loop video overlay (tower_rl.perception.video): each block's box is drawn where it projects, and the
    true box (white) only for a block whose estimate is > 2 mm off."""
    from tower_rl.data.layouts import Layout
    from tower_rl.perception.video import BAR_H, PANELS, frame

    lay = Layout.load(VARIANT)
    ids = block_order(lay)
    p7 = np.array([lay.blocks[b] for b in ids], dtype=np.float64)
    pos, quat = p7[:, :3], p7[:, 3:7]
    half = visual_half_extents([lay.dims.get(b, (0.15, 0.05, 0.03)) for b in ids])
    fixed = np.array([bool(lay.fixed[b]) for b in ids])
    cams = CameraRig("render").cameras(HOME)
    black = {n: np.zeros((HEIGHT, WIDTH, 3), np.uint8) for n in CAMERA_IDS}

    def draw(est_pos):
        return frame(black, cams, half, est_pos, quat, np.ones(len(ids)), fixed, [], ids, truth=(pos, quat),
                     focus=len(ids) - 1, lines=["status"])

    ok = draw(pos)
    assert ok.shape == (960 + BAR_H, 1280, 3) and ok.dtype == np.uint8
    p = PANELS.index("right_front_camera")
    panel = ok[BAR_H + (p // 2) * 480: BAR_H + (p // 2 + 1) * 480, (p % 2) * 640: (p % 2 + 1) * 640]
    uv, _ = cams["right_front_camera"].project(corners_world(pos, quat, half)[-1])
    lo, hi = np.floor(uv.min(0)).astype(int), np.ceil(uv.max(0)).astype(int)
    assert panel[lo[1]: hi[1] + 1, lo[0]: hi[0] + 1].any()  # the top block's box, where it projects
    off = pos.copy()
    off[-1, 0] += 0.01

    def white(img):
        return int((img[BAR_H:] > 200).all(-1).sum())

    assert white(draw(off)) > white(ok) + 50  # its true box appears once the estimate is 10 mm off
