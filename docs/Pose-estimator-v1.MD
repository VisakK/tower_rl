# TOWER challenge: pose estimator v1 and the closed loop with the teacher (build session 4)

**Covers** build session 4, 2026-10-20 evening to 2026-10-21 midday. One Claude Code session (Opus 5.5) worked
autonomously overnight on the user's request: "build the pose estimator that gives the best estimates". The video
tooling in section 11 was added on 2026-10-21 afternoon, together with this file.

**Code** is in `~/Documents/tower_rl`: the package `tower_rl/perception/`, its scripts, and small changes to the
rehearsal, runner and executor. In this repository (`corl_challenge`, the organizers' repository), only this file was
added. **Status**: none of this work is committed (section 15).

Read this after `Teacher_build_session.MD` (the teacher for Tasks 1-3) and `Teacher_build_session_2.MD` (Tasks 4 and
5). The step-by-step record, with every number and trace, is `~/Documents/tower_rl/BUILDLOG.md`, section
"2026-10-20/21 (session 4)". The commands are in the README section "Pose estimator". This file covers:
- what the estimator is and how it was built;
- how it was put in the loop with the teacher bundle;
- what it achieves, and how to watch it run;
- what is still open.

---

## 1. Summary

- **The problem.** The challenge observes only four RGB images, the joint angles and the gripper widths. The teacher's
  executor needs every block's 6-DoF pose. Until this session, every result used simulator poses.
  `Teacher_build_session.MD` section 10 set the bar: roughly **1 mm relative accuracy** between a held,
  finger-occluded block and the destination foundation.
- **The decision.** No off-the-shelf pose model fits (section 3). Instead we built two parts:
  - **The network.** A ResNet-34 U-Net, initialized from ImageNet, detects block corners, block edges, a
    visible-block mask, and the block surfaces an arm or finger hides. It was trained on our renders plus organizer
    frames.
  - **The estimator.** A model-based, multi-view robust fit with tracking (`tower_rl/perception/estimator.py`) turns
    those detections into poses.
- **Model to deploy:** `results/perception/models/kp_v2/model_last.pt`, used with
  `PoseEstimator(..., source="evaluator")`.
- **Accuracy on the evaluator's renderer** (held-out organizer frames, foundation truth corrected):

  | Measure | Result |
  |---|---|
  | All blocks while tracking (p50 / p90 / p99) | 0.20 / 0.70 / 1.72 mm |
  | First call of an episode, p99 | 0.91 mm |
  | Held blocks, p90 | 0.24 mm |
  | Whole episodes at 10 Hz: placed blocks relative to the destination foundation | 0.09-0.55 mm |
  | Time per call (A5000) | ~0.1 s |

- **Closed loop.** The network ran on the four rendered cameras and drove the teacher bundle `teacher_t45_final`
  (CPU physics, 4 episodes per tier, D0 and D1):

  | Variant | Episodes succeeded |
  |---|---|
  | Tasks 1-3 (2 and 3 layers) | 48 / 48 |
  | Task 5-2 | 8 / 8 |
  | Task 5-3 | 7 / 8 |
  | Task 4-2 | D0 4 / 4, D1 1 / 4 |
  | Task 4-3 | D0 2 / 4, D1 0 / 4 (the teacher on simulator state: D0 4 / 4 on the same seeds) |

- **Findings that matter beyond the estimator:**
  1. Isaac Lab cameras ignore the principal point; the evaluator's renderer applies it.
  2. The organizers' recorded foundation poses are up to ~7 mm from where their renderer drew the foundations.
  3. The box's half-turn symmetry: the executor's skills need quaternions that stay continuous.
  4. The estimator must know which block the executor is acting on (`focus`).
- **Open:**
  - The learned Task 4 push reacts to brief target losses while a pushed block tips over the tower edge.
  - The Task 5 insert is poorly tracked deep under the roof, although those episodes still succeed.
  - The policy server (`tower-infer-v1`) that wires the estimator, the executor and the bundle is not built yet.
- **New this afternoon:** `rehearse.py --video` writes one mp4 per episode: the four camera images with every
  estimated block drawn in, plus a status bar. `--live` renders the stage after every row for the Isaac Sim window.
  Section 11 has commands for every task.

---

## 2. What the estimator has to deliver

**Input per call.** This is the challenge observation, at 10 Hz:

| Input | Details |
|---|---|
| Static cameras | `left_back_camera`, `right_front_camera`: 640 x 480 RGB uint8, delivered turned by 180 deg (as the dataset JPEGs) |
| Wrist cameras | `left_wrist_camera`, `back_wrist_camera`: 640 x 480 RGB uint8, on the arms |
| Joint angles | `q [2, 7]`, rad: the left arm, then the back arm |
| Gripper widths | `[2]`, m |

There is no depth and no segmentation.

**Output per call.** The executor's `State` block fields, in the scene's block order (`block_order`: source
foundations, destination foundations, then movable blocks by layer and id):
- `pos [B, 3]`, in m;
- `quat [B, 4]`, wxyz;
- `conf [B]`.

**What makes it hard:**
- **Identical blocks.** The blocks are identical wood-textured boxes with a half-turn symmetry, so a block's identity
  can only come from tracking.
- **Fine tolerances.** Blocks sit 1-2 mm apart in a layer, and the Task 5 slot has 1 mm clearance per side.
- **Occlusion and motion.**
  - The fingers hide most of a held block.
  - Blocks stacked on a foundation pad hide the pad.
  - A Task 4 push moves a block up to ~40 mm in one 0.1 s call and tips it over the tower edge.
- **Lighting.** The D1 case "lighting" changes the light's intensity, colour temperature and direction.

**The rendered block.** The station's block hull is 150 x 49.76 x 28.3 mm, not the nominal 150 x 50 x 30 mm. All labels
and the fit use the hull extents. The Task 5 side blocks are the same mesh scaled.

---

## 3. Off-the-shelf models: why none was used

| Model family | Why it does not fit |
|---|---|
| FoundationPose (the strongest zero-shot 6D pose model) | Needs depth; the challenge gives none |
| RGB-only novel-object models (MegaPose, GigaPose, FoundPose, Co-op) | Single-view; centimetre-level on symmetric, weakly textured objects; slow; no notion of identical stacked instances |

What we have, and they lack:
- the exact calibration;
- the exact block model;
- an unlimited, perfectly labelled renderer of the evaluator's scene.

So the design is in the DOPE / PVNet keypoint style:
- **Network.** An off-the-shelf ImageNet backbone (torchvision ResNet-34) in a U-Net that detects corners and edges at
  full resolution. It is trained only on this scene.
- **6-DoF step.** Done geometrically: a robust Gauss-Newton fit of every block's box model to the detections in all
  four cameras at once, with tracking priors.

The network has no notion of block identity or pose; the fit supplies both.

---

## 4. Image formation (measured)

Getting the camera model exactly right was the first job: with f ≈ 607 px, a 1 px error is ~1.5 mm at the static
cameras' working distance of about 1 m.

- **Isaac Lab cameras ignore the principal point.** Our renders were 12-17 px off the dataset JPEGs, and the offset
  is exactly (cx - 320, cy - 240) on all four cameras:

  | Camera | Offset (px) |
  |---|---|
  | `left_back_camera` | 11.9 / 3.0 |
  | `right_front_camera` | 12.1 / 17.0 |
  | `left_wrist_camera` | 4.5 / 6.6 |
  | `back_wrist_camera` | 4.5 / 6.3 |

  These come from sub-pixel registration (correlation 0.74-0.95).
  - Isaac does not support aperture offsets, so its optical axis goes through the image centre, with fx and fy
    averaged.
  - The organizers' generator applies the principal point (`opencv_pinhole_applied_explicitly`).
  - The camera poses of the published station equal the dataset's for both the Sep-9 (Tasks 1-3) and the Sep-14
    (Tasks 4 and 5) data. There was no recalibration; session 1's "~1 deg recalibration" was this offset.
- **Two camera models.** `tower_rl/perception/cameras.py` provides `source="render"` for our Isaac renders and
  `source="evaluator"` for the challenge images.
  - Both renderers count the principal point from the pixel corner. In pixel-centre coordinates the evaluator's
    principal point is (cx - 0.5, cy - 0.5) and Isaac's is (319.5, 239.5).
  - With this model, projected edges of the recorded block poses sit within 0.25 px of the dataset JPEG edges on every
    camera (`scripts/check_camera_model.py`; one 0.5 px case). On our renders they also sit within 0.25 px.
  - Wrist cameras come from FK of the measured joints plus the station's wrist mounts.
  - The static images are delivered turned by 180 deg (`deliver()`).
- **Consequence.** The organizers' 531 episodes are perfectly labelled images from the evaluator's own renderer
  (recorded poses plus FK). This made both validation and fine-tuning on the evaluator domain possible.
- **Lighting.**
  - The station has one RectLight (900 x 600 mm, intensity 3000, 5600 K). RTX ambient light gives most of the base
    illumination: scaling the RectLight x0.25-x2 moves mean brightness only from 108 to 145, and with no RectLight it
    is still 105.
  - Colour temperature shows clearly.
  - A render after a light change needs more than one 4-subframe capture to converge.
  - `tower_rl/sim/lights.py` (`StationLights`) applies the D1 case "lighting" (intensity, colour temperature,
    azimuth) to the rendered light. Before this, rehearsals rendered the nominal light in every case.

---

## 5. Training data

### 5.1 Our renders (`data/perception/r1`, `r1_val`)

`scripts/render_perception_data.py` has the teacher (`teacher_t45_final`, privileged, CPU physics as in rehearsal) run
episodes while the four evaluator cameras are on. The cameras record RGB and depth, with the generator's RTX settings
and 4 subframes, and are captured at random rows (p 0.1 per 0.1 s row).

**Labels per frame:**
- exact block poses;
- the camera models (`source="render"`);
- per-corner visibility from Isaac's depth;
- a per-pixel block-id map: the analytic boxes where Isaac's depth agrees with them, 254 where an arm or finger hides
  a block surface, 255 where there is no block.

**Lighting, randomized per env and batch:**
- 40 % nominal;
- otherwise intensity x0.25-2.0 (log-uniform), 3000-8500 K, azimuth ±60 deg about the table centre, and in half of
  those a tilt of up to 30 deg.

**Run r1:**
- **Training:** all ten variants, D0 and D1, 16 episodes per tier (seed 20000). That gives 28,227 frames, four views
  each, 6.5 GB. Task 2-3, for example, made 2,766 frames in 13 min.
- **Validation:** 4 episodes per tier (seed 90000): 7,372 frames, 1.7 GB.

### 5.2 Organizer frames (the evaluator's renderer)

- **Fetching.** `scripts/fetch_dataset_frames.py` fetches JPEGs of TOWER-SimData episodes by HTTP range requests,
  labelled with the recorded block poses and FK.
- **Validation set** `data/perception/dataset_val`: 3 episodes x 24 frames per variant, 720 frames.
- **Training set** `data/perception/dataset_train`: 70 episodes x 40 frames, 2,800 frames. Subset episodes 8-9 of
  every variant were added late in the session.
- **Labelling.** `scripts/label_dataset_frames.py` writes every recorded state into the station scene (GPU, kinematic)
  and renders its depth. `tower_rl/perception/labels.py` turns that into per-corner visibility and the block-id map in
  the organizers' image coordinates. A self-check aborts if the written state does not reach the renderer (box pixels
  rendered behind the box). It reported ~0 on every variant.

### 5.3 The organizer foundation quirk (important)

The first reading of kp_v1 on organizer frames was odd:
- movable blocks: p90 0.79 mm;
- foundations: 1-4 mm off, the three blocks of a pad off together, z exact.

A network-free check (edge registration of the projected recorded poses on the static JPEGs) agreed:
- **Movable blocks** register within 0.1 px, never more than 1 px.
- **Recorded foundations** are off by more than 1 px in 41 % of views, up to 2.7 px.

Per pad and episode, the drawn foundations sit at a rigid xy offset from the recorded ones:
- median 1.0-1.2 mm, p90 3.9-4.8 mm, max 6.9 mm, no yaw;
- the drawn position is neither the recorded one nor the nominal layout position.

The episodes' goals (`target_mapping`) are built on the same jittered tower centres as the recorded foundations
(`tower_jitter_m` 0.003). So in the organizers' data, the foundation labels (used for training) and the foundation
truth (used for evaluation) were wrong by up to ~7 mm.

**Fix.** `scripts/correct_dataset_foundations.py` registers each pad (its three blocks) per episode:
- method: a coarse-to-fine translation search (radius 8 mm, 1 mm then 0.2 mm steps) on the image gradient along the
  pad's projected visible edges, in both static cameras over 4 frames, with arm pixels masked;
- it rewrites the records and keeps the originals in `pos_recorded` / `quat_recorded`, and is idempotent;
- it writes `foundation_corrections.json` next to the data;
- it was applied to `dataset_val` and `dataset_train`, followed by relabelling.

`eval_perception_sequence.py --foundation_corrections` applies the same corrections to the HDF5 truth.

What this means for the challenge:
- **Our executor.** It takes goals from the perceived (drawn) foundations, so it places relative to what it sees.
- **The evaluator's goals.** The docs say foundations and destination anchors are fixed in the evaluation. If the
  evaluator still jittered them as the generator did, its goals could be up to ~7 mm from where the foundations are
  drawn.
- **Worth asking the organizers** whether evaluation foundations are drawn exactly where the goals assume.

---

## 6. The keypoint network (`tower_rl/perception/model.py`, `scripts/train_perception.py`)

### Architecture

- **Encoder:** torchvision ResNet-34, ImageNet weights, at strides /2 to /32.
- **Decoder:** a U-Net back to full resolution, plus a full-resolution 24-channel branch from the RGB input for
  sub-pixel detail.
- **Output:** four logit maps per image at 480 x 640:

| Head | Target |
|---|---|
| `corner` | Gaussian (σ 1.5 px) at every visible block corner |
| `edge` | Gaussian (σ 1.0 px) at points every 0.7 px along every visible block edge |
| `mask` | Pixels showing a block |
| `hidden` | Block surfaces covered by an arm or finger (ids 254) |

"Visible" is exact per point: the ray through the point meets no block before it (ray casting), and no arm covers it (a
3 x 3 window in the id map).

**Why the `hidden` head.** The estimator does not model the arms geometrically. Without this head, block points behind
a finger would be predicted visible, find no match, and drag the fit. Using the mask head for this instead was tried
and dropped: "no block" also fires on background, which excuses the points of a misplaced pose. That variant had 5.2 % /
2.8 % of blocks more than 5 mm off in the stress test.

### Decoding

Corner peaks are found by 5 x 5 non-maximum suppression with a 0.3 threshold, then refined by a log-parabola through
the three samples on each axis. This is exact for a Gaussian, so the targets were chosen Gaussian.

### Training

| Setting | Value |
|---|---|
| Losses | quality focal loss (soft targets) for corners and edges; BCE for the mask; BCE with positive weight 5 for `hidden`; total = corner + edge + 0.5 mask + 0.5 hidden |
| Optimizer | AdamW, encoder lr x0.3, weight decay 1e-4; 500-iteration warm-up then cosine; bf16 autocast; batch 12 crops of 512 x 384 |
| Geometric augmentation | similarity warp: scale 0.9-1.1, ±4 deg, ±24 px; labels warped alike |
| Photometric augmentation | gamma 0.75-1.35; exposure 0.55-1.45; per-channel white balance ±15 %; contrast 0.75-1.25; saturation 0.6-1.4; blur (p 0.3, σ 0.3-0.9); Gaussian noise σ ≤ 0.012; JPEG re-encoding (p 0.3, quality 70-95) |
| Data mix | r1 renders plus organizer frames sampled 3x as often (`--train data/perception/r1 data/perception/dataset_train:3`) |

### Models

| Model | Start | Iterations | lr | Organizer labels | Wall time (A5000) |
|---|---|---|---|---|---|
| kp_v1 | ImageNet | 25,000 | 1e-3 | uncorrected foundations | ~100 min (4.3 it/s) |
| **kp_v2** | kp_v1 | 12,000 | 5e-4 | corrected foundations (70 training episodes) | 45 min |

The kp_v1 training process hung after `TRAIN_DONE` because of persistent loader workers. `train_perception.py` now ends
with `os._exit(0)`.

### Detection metrics on held-out renders (400 images)

Recall and precision are within 3 px for corners and 1.5 px for edges.

| Model | Corner recall / precision | Matched corner error, median / p90 | Edge recall / precision | Mask IoU | Hidden IoU |
|---|---|---|---|---|---|
| kp_v1 (25k) | 98.4 % / 97.4 % | 0.195 / 0.669 px | 99.4 % / 98.9 % | 0.995 | 0.929 |
| kp_v2 (12k) | 98.6 % / 97.8 % | 0.184 / 0.627 px | 99.45 % / 98.9 % | 0.996 | 0.937 |

---

## 7. The estimator (`tower_rl/perception/estimator.py`)

One `PoseEstimator` per episode stream. `reset(variant)` loads the layout:
- the block ids and order;
- the hull extents;
- the nominal poses;
- which blocks are foundations.

### 7.1 One call: `est(images, q, widths, focus=None)`

1. **Heatmaps.**
   - All four images go through the network in one batch (bf16 on the GPU).
   - Corner peaks are extracted per camera.
   - The `hidden` map is max-pooled 3 x 3.
2. **Cameras.** Static cameras from the station calibration; wrist cameras from FK of `q`.
3. **Priors.**
   - **Every block:** its previous estimate.
   - **Moving blocks:** a block that moved more than 3 mm last call is predicted to move as much again, capped at
     40 mm.
   - **Held blocks:** the gripper's pinch pose (FK) times the block's measured offset.
   - **Anchored blocks:** their rest pose (7.3).
   - **First call:** the variant's nominal layout plus a coarse planar search per block (x, y ±24 mm in 3 mm steps;
     yaw ±15 deg in 1.5 deg steps). Foundations are held toward nominal with σ 5 mm (the dataset's foundations spread
     ~2 mm).
4. **Visibility.** Every block's 8 corners and 152 edge samples are ray cast against all blocks at their priors (a
   "ghost" hides nothing). Points where the `hidden` map fires are dropped.
5. **Fit.**
   - **Method:** robust Gauss-Newton, 6-DoF, all blocks batched, all four cameras jointly, 6 iterations, with
     visibility re-predicted every 2.
   - **Corner residuals:** to the nearest corner peak, gated at 6 px for the first two iterations and 3 px after.
   - **Edge residuals:** to the nearest edge-map maximum along the projected edge's normal, searched ±6 px then ±3 px,
     sub-pixel; each edge sample weighs 0.3 of a corner.
   - **Weights:** Huber (1.5 px) for two iterations, then Tukey (2.5 px), so wrong matches drop out.
   - **Priors:** a weak prior toward the start pose (σ 10 mm / 0.05 rad). For foundations, z, roll and pitch are
     weighted x1e4 and z is pinned to the table (`_flat`).
6. **Score.**
   - **Weighted inliers:** a corner within 2 px counts 2; an edge sample within 1.5 px counts 0.25.
   - **Inlier ratio:** inliers divided by the predicted-visible weight.
   - **Corner ratio:** the share of predicted-visible corners matched.

   On oracle maps every fit with a ratio below 0.5 was wrong, as were half of those at 0.5-0.6.
7. **Re-search.** A fit is searched again if any of these holds:
   - fewer than 8 inliers;
   - ratio below 0.6;
   - fewer than 75 % of its predicted-visible corners matched, when at least 3 are predicted. A block slid along its
     length keeps its long edges but loses its corners.

   The re-search is a planar search around the prior, plus up to ±100 mm along the block's own long axis (a push), then
   a refit with a weak prior. It is accepted if ratio + 0.5 x corner ratio improves and the ratio is at least 0.6.
8. **No interpenetration (`_overlap`).** These results are rejected:
   - any search result inside another block (3 mm tolerance);
   - any fit that moved more than 5 mm into another block, unless its ratio beats that block's by at least 0.15 (then
     the other estimate is the wrong one).

   A lost block's refit could otherwise take a neighbour's image features.
9. **Lost blocks (wide search).**
   - **Which blocks:** only blocks that are moving, touched (7.3), ghosts or the focus. The search covers ±16 cm on
     5 x 5 max-pooled maps in 8 mm steps, any yaw.
   - **Hypotheses:**
     - lying flat or on its side on the table;
     - flat on top of a nearby block (knocked onto a layer);
     - leaning: low end on the table, tilted 25-82 deg about the short axis, yaw in 12 deg steps. This covers a pushed
       block resting against the tower.
   - **Acceptance:** the best hypotheses are refitted and accepted at ratio ≥ 0.6 with at least 24 inliers.
   - **Back-off:** a block not found waits 2, 4, 8, … up to 64 calls before the next search.
   - **Ghosts:** after 3 failed searches a block becomes a "ghost": confidence 0, and it hides and blocks nothing until
     found.
   - **Coasting:** a moving block that is lost keeps its motion for up to 5 calls, decaying x0.8 per call.
10. **Resting blocks.** A fit within 1.5 mm / 0.75 deg of the previous estimate moves it by 0.3 of the difference
    (noise). Beyond that, the fit is taken whole (the block moved).
11. **Held blocks.**
    - **Grasp detected:** the fingers are between 46 and 55 mm with a block centre within 3 cm of the pinch.
    - **Offset filter:**
      - the first 3 fits after a grasp are taken whole (the fingers are still closing);
      - after that, a gain of 0.15;
      - 0.5 after a change over 1 mm / 0.5 deg (a slip).
    - **Release:** the hold ends when the fingers open past 65 mm or close past 46 mm (dropped; the block is 49.7 mm
      wide).
    - **Reporting:** a held block is reported as the gripper's FK pose times the filtered offset.
12. **Foundation freeze.** After 10 good fits, a foundation's mean x, y and yaw (yaw averaged as a doubled angle) are
    frozen. It is no longer fitted, but still hides what lies behind it.
13. **Quaternion continuity.** Each block's quaternion is reported as the box symmetry (one of four half turns) nearest
    the previous output (7.3).
14. **Confidence:**

    | Value | Meaning |
    |---|---|
    | 1.0 | Good: ≥ 24 inliers and ratio ≥ 0.6 |
    | 0.6 | Accepted with fewer inliers (≥ 8 and ratio ≥ 0.5) |
    | 0.3 | Not found this call: the pose is the prior |
    | 0 | Ghost: pose unknown |

    The executor ignores samples below `CONF_MIN` 0.5.

### 7.2 Speed

A tracking call takes ~0.1 s on the A5000; the network is ~40 % of that. A first call searches every block and takes
~0.35 s. Whole organizer episodes ran at 0.085-0.13 s per call on average. On CPU only, with oracle maps, a call took
0.19 s tracking and 0.47 s on the first call.

### 7.3 Rules the closed loop forced

The loop ran with oracle maps, the network, and a truth reference. Each rule below comes from a closed-loop trace in
section 8.

| Rule | Settings | What it fixes |
|---|---|---|
| **Foundation freeze** | `freeze_after` 10 | Blocks stacked on a pad hide it and pull its fit. The destination centre drifted ~2 mm once blocks stood on it. |
| **Anchoring** | `rest_calls` 10, `rest_prior_m` 0.5 mm | A free block unmoved for 10 calls (< 0.5 mm, 0.3 deg per call) is held toward its rest pose with σ 0.5 mm. In Task 5 a side block's estimate crept 1.5 → 12.8 mm while the insert slid in next to it, because its fit took the insert's parallel edges. A real move fails the fit or the corner test, and the weak-prior re-search moves the block, ending the anchor. |
| **Touch rule** | `touch_m` 6 cm, `touch_block_m` 10 cm, `touch_calls` 20 | A still block can only have moved if a gripper's pinch came within 6 cm of it, or a moving block's centre within 10 cm, in the last 20 calls. Untouched lost blocks are treated as occluded and keep their pose. A still block searched wide was otherwise re-found on a pushed-off neighbour's features (identities swapped). |
| **Focus** | `focus=[target index]` | The block the executor's current node acts on is never anchored and is always searchable wide. An anchored push target lagged the push's first 1-3 mm, and the push actor, seeing a block that did not move, pushed harder. |
| **Quaternion continuity** | — | 57 % of reported quaternions were half-turned twins of the simulator's. The skills observe raw quaternions and were trained on the simulator's continuous ones. A re-found pushed block came back at yaw 177 deg instead of -3 deg, and a frozen foundation at +90 instead of -90 deg. |

---

## 8. Closing the loop with the teacher bundle

### 8.1 How it is wired

```
rehearse.py --perception <model.pt>   (station scene, the four evaluator cameras, CPU physics, teacher_t45_final)
  run_batch(world, executor, sampler, observe=CameraPerception(...))       tower_rl/eval/runner.py
    every time the executor needs a new chunk of rows for some envs:
      states = [world.state(e) for e in need]          # simulator state: joints, widths, block poses
      states = observe(world, need, states)           # tower_rl/perception/closed_loop.py
        render the stage (4 RTX subframes), read the 4 cameras of each env, deliver() (static ones turned)
        one network pass for all requesting envs' images
        per env: est(None, q, width, maps=..., focus=[current node's block])
        replace block_pos / block_quat / block_conf; keep the truth in State.privileged
      chunks = executor.act(states, need)             # graph + learned skills see only the estimates
    the local evaluator scores with the simulator's truth
```

- **`tower_rl/eval/runner.py`** gained the `observe` hook. It calls `observe.reset(variant, seeds)` with the policy's
  reset, and `observe.set_domain(world, dom)` so that the D1 case "lighting" reaches the rendered station light.
- **`CameraPerception`** has three modes:
  - **A checkpoint:** the real thing, rendered cameras and the network on the GPU.
  - **`oracle`:** heatmaps splatted from the true visible corners and edges. There is no rendering, and arms hide
    nothing. It tests the estimator logic and its integration on CPU in minutes.
  - **`truth`:** a pass-through reference with the same logging.
- **Camera model:** `source="render"` in rehearsal (Isaac images), `source="evaluator"` on the challenge.
- **What every run writes:**
  - `perception.json`: errors by block class and by skill target;
  - `perception_log.pkl`: per call, the estimates, truth, confidences, widths, held flags, inlier ratios, and the node
    and target;
  - the usual `feedback.md`, `metrics.json` and `episodes.json`.
- **Executor (`tower_rl/graph/executor.py`, `PushNode.step`):**
  - **Hold:** while the push target's confidence is below `CONF_MIN`, the pusher holds still for up to
    `lost_hold_rows` (15) rows.
  - **Departed:** if the target is still unseen after that, and was last seen at least `lost_departed_along_m` (60 mm)
    along the push, it has left the tower ("lost_departed", retreat).
  - **On simulator state** (confidence always 1) nothing changes; verified 4 / 4.

### 8.2 What the loop revealed, in order (failure → fix)

| # | Where | Failure | Fix |
|---|---|---|---|
| 1 | Oracle, Task 2-3 D0 | (integration check) TSR 1.00, errors p99 0.45 mm, ~180 calls per episode | — |
| 2 | Oracle, Task 1-3 D1 trace | Knocked block l3_a: the table search's refit slid onto a destination foundation block (a duplicate pose with ratio 0.69 and confidence 1) | `_overlap` rejection; search "flat on top of other blocks"; hypotheses must not interpenetrate |
| 3 | Same trace | Grasp detected while the fingers were still closing; the block shifted ~8 mm in the pinch frame | First 3 held fits taken whole (`settle_calls`) |
| 4 | Oracle, Task 4-3 D1 | Pushed block moved 31 mm in one call, then 125 mm in 0.4 s. The estimate stayed on the tower (long edges still matched), so the push node never saw the departure and pushed on into the tower | Long-axis re-search ±100 mm; velocity prior (≤ 40 mm); ±16 cm table search on max-pooled maps; leaning poses |
| 5 | Oracle, Task 4-3 | Debris searched every call: one batch took > 80 min | Back-off 2^n up to 64 calls; ghosts after 3 failures |
| 6 | Network, organizer episodes | Destination foundation drifted ~2 mm once blocks stood on it (Task 3-3 middle pad block 5.2 mm median) | Foundation freeze after 10 fits: static p99 5.58 → 0.84 mm |
| 7 | Network, Task 5-3 | Every push-in "failed" ("structure 11.6 / 12.5 mm"): the side block's estimate crept while the insert slid in beside it | Anchoring at rest: push-in target p90 20.4 → 1.3 mm (oracle), Task 5-3 3/4 → 4/4 D0 |
| 8 | Oracle, Task 4-3 | Identity swap: a still top block, its fit failing for a call (pusher in front), was searched wide and taken onto the leaning pushed block | 10-call occlusion delay, then replaced by the touch rule |
| 9 | Network, Task 4-3 D0 | 0 / 4 (truth 4 / 4): a hidden layer-1 block teleported; quaternion twins flipped the push actor's observed yaw | Touch rule; quaternion continuity |
| 10 | Network, Task 4-3 | The anchored push target lagged the push start | `focus` from the executor's node |
| 11 | Oracle vs truth, aligned call by call | Trajectories part in the first push while the pushed block tips over the tower edge, lost for 1-3 calls. The actor acted on a stale or coasted pose; one block flew off the table | Coasting 5 calls; executor push hold (8.1) |
| 12 | Oracle, Task 4-3 D0 | Follow-through variant: while the target is unseen, continue the actor's last motion | Worse (1 / 4, 30 / 36 pushes vs hold 3 / 4, 36 / 36): reverted to the hold |

### 8.3 Results in the loop

All runs below use 4 episodes per tier, seeds 7000, `--num_envs 2`, and CPU physics, with renders, network and
estimator on the GPU. Four episodes per cell is a small sample: read single failures as signals, not rates. "All
blocks mm" is the estimation error over every block at every call, against the simulator.

The estimator changed while the kp_v1 queue ran:
- **"early code":** the 3-layer variants ran before the freeze, anchoring, touch, continuity, focus and push-hold
  changes.
- **"most fixes":** the 2-layer variants ran later, with most of them.

| Variant | D0 | D1 | Estimator / model | All blocks mm, p50 / p90 / p99 | Target block p90, by skill |
|---|---|---|---|---|---|
| task1-2_tower_transfer_2 | 4 / 4 | 4 / 4 | kp_v1, most fixes | 0.16 / 0.34 / 0.54 (D0) | 0.21-0.32 mm |
| task1-3_tower_transfer_3 | 4 / 4 | 4 / 4 | kp_v1, early code | 0.19 / 0.37 / 0.73 (D0), p99 1.16 (D1) | 0.16-0.23 mm |
| task2-2_cross_tower_transfer_2 | 4 / 4 | 4 / 4 | kp_v1, most fixes | 0.08 / 0.36 / 0.70 | 0.21-0.44 mm |
| task2-3_cross_tower_transfer_3 | 4 / 4 | 4 / 4 | kp_v1, early code | 0.07 / 0.30 / 1.50 | 0.17-0.32 mm |
| task3-2_vertical_stacking_2 | 4 / 4 | 4 / 4 | kp_v1, most fixes | 0.07 / 0.21 / 0.97 | 0.25-0.39 mm |
| task3-3_vertical_stacking_3 | 4 / 4 | 4 / 4 | kp_v1, early code | 0.07 / 0.37 / 1.00 | 0.19-0.28 mm |
| task4-2_full_tower_pushdown_2 | 4 / 4 | 1 / 4 | **kp_v2, final code** | 0.11 / 0.33 / 0.76 (D0) | push 0.29 mm (D0) |
| task4-3_full_tower_pushdown_3 | 2 / 4 | 0 / 4 | kp_v1 + all fixes (`kp_v1b`) | 0.13 / 0.40 / 1.08 (D0) | push 12.3 mm (D0) |
| task5-2_tower_insertion_2 | 4 / 4 | 4 / 4 | kp_v1, most fixes | 0.11 / 0.23 / 6.43 | insert 0.09 mm, push-in 31.6 mm |
| task5-3_tower_insertion_3 | 4 / 4 | 3 / 4 | kp_v1 + anchoring + focus (`kp_v1b`) | 0.18 / 0.70 / 10.2 (D0) | insert 0.08 mm, push-in 12.4 mm |

Held blocks p90 was 0.10-0.19 mm in every variant with grasps.

**Comparison: the same teacher on simulator state** (`Teacher_build_session_2.MD`):
- 4-2 D0 / D1: 0.97 / 0.88 (40 episodes each);
- 4-3: 0.90 / 0.72;
- 5-2: 1.00 / 0.75 (24 each);
- 5-3: 1.00 / 0.58.

**On the same seeds:**
- Task 4-3 truth: D0 4 / 4, D1 0 / 2.
- Task 5-3 D1 truth: 0.50 on two of the seeds.

**Reading:**
- **Tasks 1-3 and 5** run as well on the estimated poses as on the simulator's at this sample size.
- **Task 4** does not. Its remaining failures start where the pushed block tips over the tower edge (8.2, rows 11-12).
  After that come either a "lost_departed" end declared while the block is still on the edge, or a "structure"
  failure.
- **The large Task 4 / 5 p99 values** are pushed-off blocks that left the table (truth z far below 0), and the Task 5
  insert during the push-in, deep under the roof (9.3).

Results are in `results/perception/closed_loop/<run>/<variant>/<tier>_perception/`:
- `kp_v1`: all variants;
- `kp_v1b`: Task 4-3 and 5-3 with the later fixes;
- `kp_v2`: Task 4-2;
- `oracle_*`, `truth_*`: the reference runs.

To get the tables: `python -I scripts/summarize_perception_runs.py results/perception/closed_loop/<run>`.

---

## 9. Results reference

### 9.1 Single frames (`scripts/eval_perception.py`)

**Modes:**
- `prior` starts from the truth perturbed by tracking-like noise (1.5 mm / 1 deg yaw / 0.5 deg tilt), or by a stress
  level (4 mm / 3 deg / 1 deg).
- `init` is the first call of an episode: nominal layout plus search.

**Evaluator renderer** (organizer validation frames, corrected truth). Values are confident estimates (99.8-100 % of
blocks), in mm.

| Set | kp_v1 all, p50 / p90 / p99 | kp_v2 all, p50 / p90 / p99 | kp_v2 > 5 mm | kp_v2 movable p90 | kp_v2 held p90 | kp_v2 foundations p50 / p90 / p99 |
|---|---|---|---|---|---|---|
| Tracking prior (400 frames, 4,352 blocks) | 0.31 / 1.15 / 4.05 | **0.20 / 0.70 / 1.72** | 0.07 % | 0.67 | 0.24 | 0.21 / 0.74 / 1.75 |
| First call (frames before 2 s) | 0.29 / 0.78 / 4.07 | **0.18 / 0.41 / 0.91** | 0 | 0.36 | — | — |
| Stress 4 mm / 3 deg (300 frames) | 0.29 / 1.32 / 4.39 | 0.19 / 0.63 / 2.89 | 0.30 % | 0.71 | 0.14 | — |

Foundations were most of kp_v1's error: p90 1.78 mm, against 0.74 mm for kp_v2 (the corrected labels).

**Our renders** (r1_val), in mm:

| Set | kp_v1 all, p50 / p90 / p99 | kp_v2 all, p50 / p90 / p99 |
|---|---|---|
| Tracking prior | 0.13 / 0.63 / 1.29 | 0.12 / 0.62 / 1.39 |
| First call | 0.14 / 0.39 / 0.84 | — |
| Stress | 0.12 / 0.42 / 2.09 | — |

### 9.2 The fit alone (oracle heatmaps, all ten variants, 400 validation frames)

**Stress test** (4 mm / 3 deg / 1 deg priors), share of blocks more than 5 mm off:

| Stage | Foundations | Movables |
|---|---|---|
| Baseline | 1.80 % | 1.59 % |
| + foundations fitted flat | 0.29 % | — |
| + corner-ratio re-search | — | 0.65 % |
| + evidence-aware overlap rule (final) | 0.17 % | 0.57 % |

Final p99: 1.2 mm (foundations), 3.4 mm (movables).

**Realistic tracking priors:** 0 of 4,860 blocks more than 5 mm off, p99 1.06 mm. **First call:** 0 of 880, p99
0.66 mm.

The network adds little to the fit's own error: kp_v2 tracking p99 on organizer frames is 1.72 mm, against 1.06 mm
with oracle maps on renders.

### 9.3 Whole organizer episodes at 10 Hz (`scripts/eval_perception_sequence.py`)

One held-out episode per variant, kp_v2 with the final estimator. All four JPEG streams (every third 30 Hz frame) and
the recorded joints and widths go through the estimator call by call, as on a policy server. There is no executor, so
no `focus`. "Placed vs destination" is a placed block's error relative to the destination foundation 1 s after release.
That is the quantity a placement depends on, since shared errors cancel.

| Variant | Calls | s per call | Static, p50 / p90 / p99 mm | Static > 5 mm | Held p90 mm | Placed vs destination mm |
|---|---|---|---|---|---|---|
| task1-2_tower_transfer_2 | 1162 | 0.097 | 0.18 / 0.47 / 0.47 | 0 | 0.16 | 0.20-0.27 (4) |
| task1-3_tower_transfer_3 | 1785 | 0.097 | 0.20 / 0.44 / 0.81 | 0 | 0.19 | 0.10-0.20 (6) |
| task2-2_cross_tower_transfer_2 | 604 | 0.085 | 0.14 / 0.72 / 0.72 | 0 | 0.23 | 0.12-0.14 (2) |
| task2-3_cross_tower_transfer_3 | 943 | 0.086 | 0.14 / 0.20 / 0.39 | 0 | 0.18 | 0.16-0.18 (3) |
| task3-2_vertical_stacking_2 | 588 | 0.086 | 0.23 / 0.45 / 0.45 | 0 | 0.16 | 0.21-0.56 (2) |
| task3-3_vertical_stacking_3 | 873 | 0.086 | 0.23 / 0.40 / 0.57 | 0.01 % | 0.18 | 0.21-0.48 (3) |
| task4-2_full_tower_pushdown_2 | 1182 | 0.129 | 0.14 / 0.39 / 1.02 | 0 | — | moving p90 0.16 mm |
| task4-3_full_tower_pushdown_3 | 1725 | 0.114 | 0.18 / 0.61 / 0.79 | 0.01 % | — | moving p90 0.30 mm |
| task5-2_tower_insertion_2 | 595 | 0.089 | 0.19 / 0.34 / 11.4 | 1.23 % | 0.10 | 0.26 (1) |
| task5-3_tower_insertion_3 | 1601 | 0.113 | 0.24 / 0.90 / 3.01 | 0.78 % | 0.10 | 0.54-0.58 (2) |

The Task 5 tail is the insert during the demonstration's push-in (5-2: t 51-59 s), under the roof and mostly hidden. It
is off by up to 17 mm along its length.

**With kp_v1 and the estimator before the freeze:**
- placed blocks relative to the destination: 0.1-3.6 mm;
- static p99 on Tasks 1-4: 0.86-6.3 mm.

### 9.4 Results files

| Path | Contents |
|---|---|
| `results/perception/eval/kp_v2/` | `dataset_track.json`, `dataset_init.json`, `dataset_stress.json`, `render_track.json` (each with `.rows.jsonl` per block), `sequences.json` |
| `results/perception/eval/kp_v1/` | The same for kp_v1 |
| `results/perception/eval/oracle/` | The fit alone |

---

## 10. Using it (the policy server contract)

```python
from tower_rl.perception.estimator import PoseEstimator

est = PoseEstimator("results/perception/models/kp_v2/model_last.pt", device="cuda", source="evaluator")
est.reset(variant)                         # every episode start: block set, layout, nominal poses
pos, quat, conf = est(images, q, widths, focus=[target_index])
# images: {camera id: uint8 [480, 640, 3] RGB, exactly as delivered}; q [2, 7] rad (left, back); widths [2] m
# focus: index (in est.block_ids) of the block the executor's current node acts on, or None
# -> pos [B, 3] m, quat [B, 4] wxyz, conf [B] in est.block_ids order (1 good, 0.6 weak, 0.3 not seen, 0 lost)
```

- **One estimator per episode stream.** It is stateful: tracking, held offsets, anchors, frozen foundations.
- **The first call** of an episode searches from the nominal layout. Arms at home give the best start.
- **`focus` comes from the executor:** `runs[e].g["nodes"][runs[e].i]["block"]`, as in `CameraPerception`.
- **Blocks that leave the table** end as ghosts (confidence 0). The executor already skips transfers of blocks that
  fell off.

---

## 11. Watching it run

### 11.1 What the video shows

`rehearse.py --perception <model.pt> --video <dir>` writes one mp4 per episode:
`<variant>_<tier>_seed<episode seed>_<success|fail>.mp4`. The code is `tower_rl/perception/video.py`, hooked into
`CameraPerception`.

**Frames.**
- Each frame is one estimator call: the four delivered camera images exactly as the network saw them.
  - Top: `right_front`, `left_back` (static).
  - Bottom: `left_wrist`, `back_wrist`.
- Size: 1280 x 1040, at 10 fps.
- Learned skills are called every 0.1 s row, so they play in real time. Planned transits are one call per chunk of up
  to 16 rows, so they play faster.

**Overlay.** Every block's estimated pose is drawn as its visible box edges, hidden ones removed by ray casting
against the estimated blocks. Arms are not modelled, so edges show through fingers.

| Colour | Meaning |
|---|---|
| Green | Tracked movable block (confidence 1) |
| Blue | Foundation (frozen after 10 fits) |
| Cyan | Held block |
| Magenta, thick, labelled | Target of the executor's current node |
| Yellow | Weak fit (0.6) |
| Orange | Not seen this call, previous pose kept (0.3) |
| Red | Lost, a ghost (0) |
| White, labelled with the error | The simulator's true pose, only for blocks estimated > 2 mm off |

A target that is not tracked well keeps its state colour; thickness and label still mark it.

**Status bar.**
1. Variant, tier, episode seed, D1 case (for example `lighting`), sim time, estimator call number.
2. The executor node and skill; the target block's error (mm / deg) and confidence; the last node result.
3. Counts of good / weak / not-seen / lost blocks, the worst confident error, and which block each gripper holds.

**`--video_maps`** also blends the network's edge map in blue and marks its corner peaks in yellow: what the network
detects, before the fit.

**`--live`** renders the stage after every row. Without `--headless`, the Isaac Sim window shows env 0 from
`play_episode.py`'s viewpoint inside the curtain enclosure, smoothly, slower than real time. It can be combined with
`--video`.

**Tested.**
- Task 2-2 D0, one episode: TSR 1.00; 121 calls; 51 s wall time; errors p99 0.28 mm; a 1.5 MB mp4.
- Task 3-2 D1 with `--live --video_maps`, headless: TSR 1.00; 117 calls; 72 s; errors p99 0.43 mm.
- The on-screen GUI is not tested here, but `play_episode.py` opens the same Isaac Sim window.

### 11.2 Commands: one episode per task

```bash
cd ~/Documents/tower_rl
source ~/Documents/moves/env_isaaclab/bin/activate

# one episode of one variant: the network (kp_v2) on the four rendered cameras drives the teacher bundle;
# mp4s go to results/perception/videos/, reports to results/perception/viz_runs/<variant>/<tier>_perception/
loop_video() {   # usage: loop_video <variant> [D0|D1] [seed] [more rehearse.py options]
  local v=$1 tier=${2:-D0} seed=${3:-7000}
  shift $(( $# < 3 ? $# : 3 ))
  python scripts/rehearse.py --headless --device cpu --variant "$v" --tiers "$tier" --episodes 1 --num_envs 1 \
      --seed "$seed" --bundle results/bundles/teacher_t45_final \
      --perception results/perception/models/kp_v2/model_last.pt \
      --video results/perception/videos --out_root results/perception/viz_runs "$@"
}

# Task 1, tower transfer
loop_video task1-2_tower_transfer_2
loop_video task1-3_tower_transfer_3
# Task 2, cross tower transfer
loop_video task2-2_cross_tower_transfer_2
loop_video task2-3_cross_tower_transfer_3
# Task 3, vertical stacking: the destination pad disappears under the stack (blue, frozen foundations)
loop_video task3-2_vertical_stacking_2
loop_video task3-3_vertical_stacking_3
# Task 4, push-down: pushed blocks tracked through the slide, lost while tipping (orange), found on the table
loop_video task4-2_full_tower_pushdown_2
loop_video task4-3_full_tower_pushdown_3
# Task 5, insertion: the insert slides under the roof; the side blocks stay anchored
loop_video task5-2_tower_insertion_2
loop_video task5-3_tower_insertion_3

# under domain randomization (D1: lighting, block mass / friction cases), another seed, or the network's maps
loop_video task4-3_full_tower_pushdown_3 D1 7000
loop_video task2-3_cross_tower_transfer_3 D1 7100 --video_maps

# all ten variants, one episode each, one after the other (roughly 30-60 min)
for v in task1-2_tower_transfer_2 task1-3_tower_transfer_3 task2-2_cross_tower_transfer_2 \
         task2-3_cross_tower_transfer_3 task3-2_vertical_stacking_2 task3-3_vertical_stacking_3 \
         task4-2_full_tower_pushdown_2 task4-3_full_tower_pushdown_3 task5-2_tower_insertion_2 \
         task5-3_tower_insertion_3; do
  loop_video $v D0
done

ls -t results/perception/videos | head            # newest first; names end in _success / _fail
xdg-open results/perception/videos/<file>.mp4      # any player; mpv --speed=0.5 <file> to slow it down
```

**Duration** (wall time per episode, from the closed-loop runs):
- Task 2-2 and 3-2: ~1 min.
- Tasks 1-2, 2-3, 3-3 and 5-2: 2-3 min.
- Task 1-3: 5-6 min.
- Task 5-3: 4-16 min.
- Task 4: 3-20 min. A struggling episode runs until its 210 s sim-time limit.

**The terminal prints** each batch's TSR / GSR / PSR / ODR, the estimation summary, and the video's path.

**Episode seeds.** `--seed 7000` with one env is episode seed 7000000, the first episode of the 4-per-tier runs in
8.3. It is not necessarily the same physical episode: one env instead of two, and code changes since, alter the
physics history.

### 11.3 Watching live in the Isaac Sim window

```bash
# drop --headless and add --live: the window opens after about a minute; keep --num_envs 1
python scripts/rehearse.py --live --device cpu --variant task2-3_cross_tower_transfer_3 --tiers D0 --episodes 1 \
    --num_envs 1 --seed 7000 --bundle results/bundles/teacher_t45_final \
    --perception results/perception/models/kp_v2/model_last.pt \
    --video results/perception/videos --out_root results/perception/viz_runs
```

The viewport shows the scene, not the estimates; the mp4 shows the estimates. Rendering the window and the four
cameras at every row is the heaviest GPU load of these commands. Run nothing else on the GPU meanwhile.

### 11.4 Comparisons and numbers

```bash
# the same teacher on simulator state, for comparison: mp4 of the four cameras, one frame per 0.1 s row (real time)
python scripts/trace_episode.py --headless --device cpu --variant task2-3_cross_tower_transfer_3 --tier D0 \
    --num_envs 1 --seed 7000 --bundle results/bundles/teacher_t45_final --cam_scale 1.0 \
    --video results/perception/videos/task2-3_cross_tower_transfer_3_D0_seed7000000_truth.mp4

# the closed-loop numbers (no video): 4 episodes per tier, then a table
python scripts/rehearse.py --headless --device cpu --variant task4-2_full_tower_pushdown_2 --tiers D0 D1 \
    --episodes 4 --num_envs 2 --seed 7000 --bundle results/bundles/teacher_t45_final \
    --perception results/perception/models/kp_v2/model_last.pt --out_root results/perception/closed_loop/<name>
python -I scripts/summarize_perception_runs.py results/perception/closed_loop/<name>

# the estimator logic alone, on CPU (heatmaps splatted from the truth; no rendering, no video): ~1-2 min here,
# but Task 4-3 takes 13-21 min per batch of 2 (its debris searches)
python scripts/rehearse.py --headless --device cpu --variant task2-3_cross_tower_transfer_3 --tiers D0 --episodes 2 \
    --num_envs 2 --seed 7000 --bundle results/bundles/teacher_t45_final --perception oracle \
    --out_root results/perception/closed_loop/oracle_check
```

**Reading a run's per-call log:**

```python
import pickle
d = pickle.load(open("results/perception/viz_runs/<variant>/D0_perception/perception_log.pkl", "rb"))
d["block_ids"]; r = d["log"][100]
# r: seed, t, err_mm [B], err_deg [B], conf [B], held [B], fixed [B], skill, block (target index), pos, quat,
#    true_pos, true_quat, width, est_held, ratio, inliers
```

**House rules:**
- Run one GPU job at a time; check first with `nvidia-smi`.
- For long unattended runs, wrap the command in `scripts/run_logged.sh <log> ...`, which records GPU telemetry.
- Stop a run with `kill -9 <pid>`; Isaac ignores Ctrl-C at times.

---

## 12. Code and artifacts (all in `~/Documents/tower_rl`)

### Code

| Path | Role |
|---|---|
| `tower_rl/perception/cameras.py` | Camera models (`evaluator` / `render`), `CameraRig` (static + FK wrists), `deliver()` |
| `tower_rl/perception/boxes.py` | Hull extents, box corners, analytic ray casting (depth and instance) |
| `tower_rl/perception/calib.py` | Edge-registration checks of a camera model |
| `tower_rl/perception/labels.py` | Per-corner visibility and id maps from rendered depth |
| `tower_rl/perception/data.py` | Keypoint labels, augmentation, the training dataset, Gaussian splatting |
| `tower_rl/perception/model.py` | `KeypointNet`, quality focal loss, sub-pixel `peaks` |
| `tower_rl/perception/estimator.py` | `PoseEstimator`, `EstCfg` (every threshold, commented) |
| `tower_rl/perception/closed_loop.py` | `CameraPerception`: perception in the rehearsal loop; logs |
| `tower_rl/perception/video.py` | **new**: the video overlay and per-episode writers |
| `tower_rl/sim/lights.py` | `StationLights`: the D1 case "lighting" on the rendered light |

### Scripts

| Script | Role |
|---|---|
| `render_perception_data.py` | Our training renders |
| `fetch_dataset_frames.py` | Organizer frames |
| `label_dataset_frames.py` | Occlusion-aware labels for organizer frames |
| `correct_dataset_foundations.py` | The foundation fix |
| `train_perception.py` | Network training |
| `eval_perception.py` | Single-frame evaluation |
| `eval_perception_sequence.py` | Whole organizer episodes; `--trace` writes a per-call `.trace.npz` |
| `check_camera_model.py` | Camera-model check on organizer JPEGs |
| `summarize_perception_runs.py` | Closed-loop tables |
| `rehearse.py` | `--perception`, `--video`, `--video_maps`, `--live` |

### Data

| Path | Contents |
|---|---|
| `data/perception/r1` | 6.5 GB |
| `data/perception/r1_val` | 1.7 GB |
| `data/perception/dataset_train` | 768 MB |
| `data/perception/dataset_val` | 197 MB |

### Models and results

| Path | Contents |
|---|---|
| `results/perception/models/kp_v1/` | Model, checkpoints, `config.json`, `train_log.jsonl` |
| `results/perception/models/kp_v2/` | The same; **deploy this model** |
| `results/perception/eval/`, `results/perception/closed_loop/` | Results (sections 8.3 and 9) |
| `results/perception/calib/` | Camera-model checks |
| `results/perception/logs/` | Every GPU job's log and telemetry (`*.gpu.csv`) |
| `results/perception/gpu_queue_*.sh` | The queues that produced them |

### Tests

`tests/test_perception.py` holds 13 tests. They cover:
- a knocked block found on a layer;
- a slid block's prior and a sunken foundation;
- a pushed block followed along its length;
- a leaning block found;
- a still block out of view keeping its pose;
- quaternion continuity through the foundation freeze;
- the video frame.

Run them with `python -m pytest tests/test_perception.py`. The whole suite passes.

---

## 13. Shortcomings and open issues

1. **Task 4 in the loop**: 4-2 D1 1 / 4, 4-3 D0 2 / 4 and D1 0 / 4, against 0.88-0.97 for the teacher on truth.
   - **Cause.** The learned push actor was trained on simulator poses. It reacts to what simulator state never shows:
     the pushed block unseen for 1-3 calls while it tips over the tower edge (it rotates about the edge and leaves the
     fit's basin), then found on its side.
   - **Current mitigation.** The executor holds the pusher while the target is unseen, which costs follow-through
     (the actor on truth keeps pushing as the block tips). "lost_departed" can then fire while the block is still on
     the edge, and a "structure" failure follows.
   - **D1** adds the lighting case.
2. **Task 5 push-in.** The insert deep under the roof is tracked to 1-17 mm along its length: mostly hidden, with
   parallel neighbours. The closed loop still succeeds (5-2 8 / 8, 5-3 7 / 8), but the margin is unknown.
3. **Small samples.** 4 episodes per tier. Tasks 1-3 at 48 / 48 rules out large losses, not a few percent.
4. **Mixed code versions in 8.3.** The 3-layer variants of Tasks 1-3 ran with kp_v1 and the estimator before the
   freeze, anchoring, touch, continuity and focus changes; the 2-layer variants ran with most of them. Nothing
   suggests a regression, but the final code and kp_v2 have only been run in the loop on Task 4-2 and, today, two
   single episodes (Task 2-2 D0, Task 3-2 D1: both succeeded, errors p99 0.28 / 0.43 mm).
5. **Rehearsal is not the challenge.** Our renders come from Isaac Lab with `source="render"`; the evaluator's images
   differ in principal point and renderer details. kp_v2 was validated on the evaluator's renderer frame by frame and
   episode by episode, but never in a closed loop (only the organizers can run that).
6. **The foundation question** (5.3) is open with the organizers.
7. **Arms are not modelled** geometrically. The learned `hidden` map stands in, and is good (IoU 0.94), but a block
   hidden behind an arm in all four views is coasted or kept, not seen.
8. **No uncertainty is propagated** beyond the four-level confidence. The executor's `CONF_MIN` gate is binary.

---

## 14. Next steps

1. **The policy server (`tower-infer-v1`).** Wire `PoseEstimator(source="evaluator")` to the executor and
   `teacher_t45_final`:
   - one estimator per episode;
   - `focus` from the current node;
   - the images exactly as delivered.

   Measure the end-to-end latency against the 10 Hz budget.
2. **Task 4: fine-tune the push skill on perception-like observations.** Use dropouts of 1-5 calls, stale and coasted
   poses, and re-found poses, as Step 8.1 did for grasp and place. Re-evaluate the push hold and the "lost_departed"
   rule afterwards.
3. **Task 4: track through the tip-over.** A rotation-about-the-edge hypothesis in the re-search would keep the pushed
   block seen while it tips.
4. **Re-run the closed loop for all variants with kp_v2 and the final code**, 8+ episodes per tier, to replace the
   mixed table in 8.3. GPU time is about 4-6 h for all ten.
5. **Task 5 push-in:** a stronger prior along the slot axis for the insert while the push-in node runs. The
   executor knows the slot; the estimator does not.
6. **Ask the organizers** about foundation jitter in the evaluation (5.3).

---

## 15. Housekeeping

None of this work is committed. In `~/Documents/tower_rl`:

**Modified:**
- `BUILDLOG.md`
- `README.md`
- `scripts/rehearse.py`
- `tower_rl/eval/runner.py`
- `tower_rl/graph/executor.py`
- `tower_rl/sim/station_scene.py`

**New:**
- `tower_rl/perception/` (all files, including today's `video.py`)
- `tower_rl/sim/lights.py`
- `scripts/check_camera_model.py`
- `scripts/correct_dataset_foundations.py`
- `scripts/eval_perception.py`
- `scripts/eval_perception_sequence.py`
- `scripts/fetch_dataset_frames.py`
- `scripts/label_dataset_frames.py`
- `scripts/render_perception_data.py`
- `scripts/summarize_perception_runs.py`
- `scripts/train_perception.py`
- `tests/test_perception.py`

`data/` and `results/` are not tracked (as before). In `corl_challenge`, this file is untracked, like the other
session notes.
